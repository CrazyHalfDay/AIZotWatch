// Profile-based ranking, ported from the Python ProfileRanker / ClusterScorer /
// JournalScorer. Similarity = a * S_micro + (1 - a) * S_macro, and
// score = w_sim * similarity + w_if * IF_score.

import { chooseClusterCount, sphericalKMeans } from "./kmeans.js";
import { MAX_CLUSTER_TITLES, topKeywords } from "./topics.js";
import { dot, normalizeInPlace, yieldToEventLoop } from "./vector.js";

export const PREPRINT_SOURCES = new Set(["arxiv", "eartharxiv", "biorxiv", "medrxiv"]);

/** Exponential decay weights by date added: w = max(minWeight, 2^(-age/halflife)). */
export function computeTemporalWeights(dates, { halflifeDays = 0, minWeight = 0.05, now = Date.now() } = {}) {
  if (!halflifeDays || halflifeDays <= 0) return new Float32Array(dates.length).fill(1);
  const decay = Math.log(2) / halflifeDays;
  return Float32Array.from(dates, (d) => {
    const t = d ? Date.parse(d) : NaN;
    if (!Number.isFinite(t)) return 1;
    const ageDays = (now - t) / 86400000;
    if (ageDays <= 0) return 1;
    return Math.max(minWeight, Math.exp(-decay * ageDays));
  });
}

/**
 * Members of one cluster described for topic labels (Python ClusterInfo):
 * profile item keys, the titles closest to the unweighted member mean (the
 * k-means centroid, as in Python) and the most frequent keyword tags.
 */
function describeMembers(matrix, members, items, outputTag) {
  const dim = matrix.dim;
  const mean = new Float32Array(dim);
  for (const i of members) {
    const off = i * dim;
    for (let d = 0; d < dim; d++) mean[d] += matrix.data[off + d];
  }
  normalizeInPlace(mean);
  const titles = members
    .map((i) => ({ i, sim: dot(matrix.data, mean, i * dim, 0, dim) }))
    .sort((a, b) => b.sim - a.sim)
    .map(({ i }) => items[i]?.title)
    .filter((t) => typeof t === "string" && t.trim())
    .slice(0, MAX_CLUSTER_TITLES);
  const memberItems = members.map((i) => items[i]);
  return {
    memberKeys: memberItems.map((item) => item?.key).filter((k) => typeof k === "string"),
    titles,
    keywords: topKeywords(memberItems, { outputTag }),
  };
}

/**
 * Cluster the profile and summarise each cluster by its weighted centroid and
 * effective size (sum of member temporal weights). With `items` (the profile
 * items, aligned with the matrix rows: {key, title, tags}) each cluster also
 * carries `memberKeys`, representative `titles` and `keywords` (tags other
 * than `outputTag` and markers). Clusters are sorted by member count, largest
 * first (Python order, stable for ties).
 */
export async function buildClusters(
  matrix,
  weights,
  { minClusters = 5, maxClusters = 35, tolerance = 0.2, iterations = 20, seed = 42, items = null, outputTag = "" } = {},
) {
  const n = matrix.rows;
  if (n === 0) return { clusters: [], k: 0, silhouette: null };

  let k = 1;
  let silhouette = null;
  if (n >= 5) {
    ({ k, silhouette } = await chooseClusterCount(matrix, { minClusters, maxClusters, tolerance, seed }));
  }

  let assignments;
  if (k <= 1) {
    assignments = new Int32Array(n);
    k = 1;
  } else {
    ({ assignments } = await sphericalKMeans(matrix, k, { iterations, seed, cooperative: true }));
  }

  const dim = matrix.dim;
  const sums = new Float32Array(k * dim);
  const effective = new Float64Array(k);
  const counts = new Int32Array(k);
  const members = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) {
    const c = assignments[i];
    const w = weights[i];
    counts[c]++;
    effective[c] += w;
    members[c].push(i);
    const off = c * dim;
    const rowOff = i * dim;
    for (let d = 0; d < dim; d++) sums[off + d] += matrix.data[rowOff + d] * w;
  }

  const clusters = [];
  for (let c = 0; c < k; c++) {
    if (counts[c] === 0) continue;
    const centroid = sums.slice(c * dim, (c + 1) * dim);
    normalizeInPlace(centroid);
    const cluster = { centroid, memberCount: counts[c], effectiveSize: effective[c] };
    if (items) {
      Object.assign(cluster, describeMembers(matrix, members[c], items, outputTag));
      await yieldToEventLoop();
    }
    clusters.push(cluster);
  }
  clusters.sort((a, b) => b.memberCount - a.memberCount);
  return { clusters, k: clusters.length, silhouette };
}

/** Percentile with numpy's default linear interpolation. */
export function percentile(values, p) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

export function computeThresholds(scores, cfg) {
  if (cfg.mode === "fixed" || scores.length < 2) {
    return { mode: "fixed", mustRead: cfg.mustRead, consider: cfg.consider };
  }
  let mustRead = Math.max(percentile(scores, cfg.mustReadPercentile), cfg.minMustRead);
  const consider = Math.max(percentile(scores, cfg.considerPercentile), cfg.minConsider);
  if (mustRead <= consider) mustRead = consider + 0.01;
  return { mode: "dynamic", mustRead, consider };
}

export function assignLabel(score, thresholds) {
  if (score >= thresholds.mustRead) return "must_read";
  if (score >= thresholds.consider) return "consider";
  return "ignore";
}

/** Journal quality score: log-normalised impact factor, with fixed scores for preprints/CN/unknown. */
export function journalScore(work, journals, cfg) {
  if (PREPRINT_SOURCES.has(work.source)) {
    return { ifScore: cfg.preprintScore, impactFactor: null, isChineseCore: false };
  }
  for (const issn of work.issns || []) {
    const entry = journals.get(issn);
    if (!entry) continue;
    if (entry.isChineseCore) return { ifScore: cfg.chineseCoreScore, impactFactor: null, isChineseCore: true };
    if (entry.impactFactor != null) {
      const normalized = Math.log(entry.impactFactor + 1) / Math.log(cfg.logBase);
      return { ifScore: Math.min(normalized, 1), impactFactor: entry.impactFactor, isChineseCore: false };
    }
  }
  return { ifScore: cfg.unknownScore, impactFactor: null, isChineseCore: false };
}

export class ProfileRanker {
  /**
   * @param {object} profile - { matrix: Matrix, weights: Float32Array, clusters: [] }
   * @param {Map} journals - ISSN -> { impactFactor, isChineseCore }
   * @param {object} cfg - scoring settings
   */
  constructor(profile, journals, cfg) {
    this.matrix = profile.matrix;
    this.weights = profile.weights;
    this.clusters = profile.clusters || [];
    this.journals = journals;
    this.cfg = cfg;
    this.logSizes = this.clusters.map((c) => Math.log(1 + c.effectiveSize));
    this.maxLogSize = this.logSizes.length ? Math.max(...this.logSizes) : 0;
  }

  get isEmpty() {
    return !this.matrix || this.matrix.rows === 0;
  }

  /** k-NN similarity weighted by temporal weights. */
  microScore(vec, neighbors = this.matrix.topK(vec, this.cfg.knn)) {
    if (!neighbors.length) return 0;
    let num = 0;
    let den = 1e-8;
    for (const { index, score } of neighbors) {
      const w = this.weights[index] ?? 1;
      num += score * w;
      den += w;
    }
    return num / den;
  }

  /**
   * max_k(sim_k * ln(1 + E_k)) / max_k ln(1 + E_k). Also returns `cluster`,
   * the macro-best cluster (favours big clusters), and `nearest`, the cluster
   * with the most similar centroid (the work's research direction).
   */
  macroScore(vec) {
    if (!this.clusters.length) return { macro: 0, cluster: null, nearest: null };
    let best = -Infinity;
    let bestIdx = 0;
    let nearestSim = -Infinity;
    let nearest = 0;
    this.clusters.forEach((c, i) => {
      const sim = dot(c.centroid, vec);
      if (sim > nearestSim) {
        nearestSim = sim;
        nearest = i;
      }
      const raw = sim * this.logSizes[i];
      if (raw > best) {
        best = raw;
        bestIdx = i;
      }
    });
    if (this.maxLogSize <= 0) return { macro: 0, cluster: null, nearest };
    return { macro: best / this.maxLogSize, cluster: bestIdx, nearest };
  }

  scoreVector(vec, neighbors) {
    if (this.clusters.length) {
      const micro = this.microScore(vec, neighbors);
      const { macro, cluster, nearest } = this.macroScore(vec);
      const a = this.cfg.microWeight;
      return { similarity: a * micro + (1 - a) * macro, micro, macro, cluster, nearest };
    }
    const [top] = neighbors || this.matrix.topK(vec, 1);
    return { similarity: top ? top.score : 0, micro: null, macro: null, cluster: null, nearest: null };
  }

  /** Similarity and journal scores of works (each with a `vector`, non-empty profile), in input order. */
  async scoreAll(works, { onProgress } = {}) {
    // Neighbours are searched in small blocks; yielding between blocks keeps Zotero responsive
    const scored = [];
    const block = 8;
    const k = this.clusters.length ? this.cfg.knn : 1;
    for (let start = 0; start < works.length; start += block) {
      const slice = works.slice(start, start + block);
      const neighbors = this.matrix.topKBlock(slice.map((w) => w.vector), k);
      slice.forEach((w, j) => {
        const sim = this.scoreVector(w.vector, neighbors[j]);
        const js = journalScore(w, this.journals, this.cfg.journal);
        const score = this.cfg.similarityWeight * sim.similarity + this.cfg.ifWeight * js.ifScore;
        scored.push({ ...w, ...sim, ...js, score });
      });
      onProgress?.(scored.length, works.length);
      await yieldToEventLoop();
    }
    return scored;
  }

  /**
   * Display scores without the similarity gate, thresholds or labels, in input
   * order (Python ProfileRanker.score_works, used by the flagship track). With
   * an empty profile every work scores 0 and gets no journal score.
   */
  async scoreWorks(works, { onProgress } = {}) {
    if (this.isEmpty) {
      return works.map((w) => ({ ...w, similarity: 0, micro: null, macro: null, cluster: null, nearest: null, ifScore: 0, impactFactor: null, score: 0 }));
    }
    return this.scoreAll(works, { onProgress });
  }

  /**
   * Score works (each with a `vector`), apply the similarity gate, compute
   * thresholds and labels, and return them sorted by score.
   */
  async rank(works, { onProgress } = {}) {
    if (!works.length) return { ranked: [], thresholds: null, gated: 0 };
    if (this.isEmpty) {
      // Cold start: random order, all "consider" (mirrors the Python fallback)
      const shuffled = [...works].sort(() => Math.random() - 0.5);
      return {
        ranked: shuffled.map((w) => ({ ...w, ...journalScore(w, this.journals, this.cfg.journal), score: 0, similarity: 0, label: "consider" })),
        thresholds: { mode: "random", mustRead: 0, consider: 0 },
        gated: 0,
      };
    }

    const scored = await this.scoreAll(works, { onProgress });
    let kept = scored;
    if (this.cfg.gateEnabled) kept = scored.filter((s) => s.similarity >= this.cfg.minSimilarity);
    const thresholds = computeThresholds(
      kept.map((s) => s.score),
      this.cfg.thresholds,
    );
    for (const s of kept) s.label = assignLabel(s.score, thresholds);
    kept.sort((a, b) => b.score - a.score);
    return { ranked: kept, thresholds, gated: scored.length - kept.length };
  }
}
