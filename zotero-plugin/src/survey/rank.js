// Ranking, read-set selection and theme clustering of the topic survey (pure
// functions over papers and their vectors).

import { tokenize } from "../embedding/local.js";
import { silhouetteFromDistances, sphericalKMeans } from "../core/kmeans.js";
import { Matrix, dot, normalizeInPlace } from "../core/vector.js";
import { MAX_THEMES, MIN_THEMES } from "./presets.js";
import { VIA_REC, linkedSeeds } from "./pool.js";

/** Weights of the survey score (design C1 stage 5). */
export const SCORE_WEIGHTS = Object.freeze({ sim: 0.55, link: 0.2, cites: 0.15, recency: 0.1 });

const round = (x) => Math.round(x * 1000) / 1000;

/** Normalised mean of vectors (the survey's centre: the topic and the seeds). */
export function centroidOf(vectors) {
  const list = vectors.filter((v) => v && v.length);
  if (!list.length) return null;
  const dim = list[0].length;
  const out = new Float32Array(dim);
  for (const v of list) {
    if (v.length !== dim) continue;
    for (let i = 0; i < dim; i++) out[i] += v[i];
  }
  return normalizeInPlace(out);
}

/**
 * Score every paper: `0.55·sim + 0.20·link + 0.15·cites + 0.10·recency`.
 * - sim: cosine to the centre, rescaled to 0..1 over the pool (like cites and
 *   recency), so that the weights mean the same with every embedder (local
 *   TF-IDF cosines are far lower than neural ones); the raw cosine is kept
 * - link: seeds the paper cites / is cited by (plus one for a recommendation)
 *   over the number of seeds, capped at 1; a seed itself scores 1
 * - cites: log1p(citations) / log1p(max citations in the pool)
 * - recency: linear over the pool's year span (0 without a year)
 * Sets `p.scores = {total, sim, cosine, link, cites, recency}` and returns the
 * papers sorted best first (with `p.rank`, 1-based).
 * @param {object[]} papers
 * @param {Map<object, Float32Array>|((p) => Float32Array)} vectorOf
 * @param {Float32Array|null} centre
 * @param {number} seedCount
 */
export function scorePapers(papers, vectorOf, centre, seedCount) {
  const get = typeof vectorOf === "function" ? vectorOf : (p) => vectorOf.get(p);
  const cosines = papers.map((p) => {
    const v = get(p);
    return centre && v && v.length === centre.length ? dot(v, centre) : 0;
  });
  const lo = Math.min(...cosines);
  const hi = Math.max(...cosines);
  const maxCites = Math.max(0, ...papers.map((p) => Number(p.citations) || 0));
  const years = papers.map((p) => p.year).filter(Number.isInteger);
  const minYear = years.length ? Math.min(...years) : 0;
  const maxYear = years.length ? Math.max(...years) : 0;
  papers.forEach((p, i) => {
    const sim = hi > lo ? (cosines[i] - lo) / (hi - lo) : cosines.length ? 1 : 0;
    const links = linkedSeeds(p.via).size + ((p.via || []).includes(VIA_REC) ? 1 : 0);
    const link = p.seed ? 1 : seedCount > 0 ? Math.min(1, links / seedCount) : 0;
    const cites = maxCites > 0 ? Math.log1p(Math.max(0, Number(p.citations) || 0)) / Math.log1p(maxCites) : 0;
    const recency = Number.isInteger(p.year) ? (maxYear > minYear ? (p.year - minYear) / (maxYear - minYear) : 1) : 0;
    const total = SCORE_WEIGHTS.sim * sim + SCORE_WEIGHTS.link * link + SCORE_WEIGHTS.cites * cites + SCORE_WEIGHTS.recency * recency;
    p.scores = { total: round(total), sim: round(sim), cosine: round(cosines[i]), link: round(link), cites: round(cites), recency: round(recency) };
  });
  const sorted = [...papers].sort((a, b) => b.scores.total - a.scores.total || (b.citations || 0) - (a.citations || 0));
  sorted.forEach((p, i) => (p.rank = i + 1));
  return sorted;
}

const hasText = (v) => typeof v === "string" && v.trim() !== "";

/**
 * The papers read by the LLM: seeds first, then the best ranked, `count` in
 * all; papers without any abstract or full text are skipped.
 * @param {object[]} ranked - papers best first
 * @param {(p) => boolean} [hasFullText]
 */
export function selectReadSet(ranked, count, hasFullText = () => false) {
  const readable = (p) => hasText(p.abstract) || hasFullText(p);
  const seeds = ranked.filter((p) => p.seed && readable(p));
  const rest = ranked.filter((p) => !p.seed && readable(p));
  return [...seeds, ...rest].slice(0, Math.max(0, count));
}

/**
 * Cluster papers into themes: spherical k-means with k from MIN_THEMES to
 * MAX_THEMES (at most n / 3, so themes keep a few papers) chosen by the
 * silhouette; fewer than 6 papers make one theme.
 * @param {Float32Array[]} vectors - L2-normalised, aligned with the papers
 * @returns {Promise<{k: number, assignments: number[], silhouette: number|null}>}
 */
export async function clusterThemes(vectors, { minK = MIN_THEMES, maxK = MAX_THEMES, seed = 42 } = {}) {
  const n = vectors.length;
  const upper = Math.min(maxK, Math.floor(n / 3));
  if (n < 6 || upper < 2) return { k: n ? 1 : 0, assignments: vectors.map(() => 0), silhouette: null };
  const matrix = Matrix.fromRows(vectors);
  const distances = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = Math.max(0, 1 - dot(matrix.data, matrix.data, i * matrix.dim, j * matrix.dim, matrix.dim));
      distances[i * n + j] = d;
      distances[j * n + i] = d;
    }
  }
  let best = null;
  for (let k = Math.max(2, Math.min(minK, upper)); k <= upper; k++) {
    const { assignments } = await sphericalKMeans(matrix, k, { seed });
    const labels = Array.from(assignments);
    // Empty clusters (k-means re-seeding) make the k smaller than it looks
    const used = new Set(labels).size;
    const s = silhouetteFromDistances(distances, n, labels, k);
    if (s === null || !Number.isFinite(s) || used < 2) continue;
    if (!best || s > best.silhouette) best = { k, assignments: labels, silhouette: s };
  }
  if (!best) return { k: 1, assignments: vectors.map(() => 0), silhouette: null };
  // Renumber 0..k-1 in order of first appearance (drops empty clusters)
  const map = new Map();
  const assignments = best.assignments.map((c) => {
    if (!map.has(c)) map.set(c, map.size);
    return map.get(c);
  });
  return { k: map.size, assignments, silhouette: round(best.silhouette) };
}

/**
 * Distinctive terms of a group of texts against all texts (labels of themes
 * without an LLM): term frequency in the group weighted by inverse document
 * frequency over all; English stems and CJK bigrams (embedding/local.js tokenize).
 */
export function topTerms(groupTexts, allTexts, limit = 3) {
  const docFreq = new Map();
  for (const text of allTexts) for (const t of new Set(tokenize(text))) docFreq.set(t, (docFreq.get(t) || 0) + 1);
  const counts = new Map();
  for (const text of groupTexts) for (const t of new Set(tokenize(text))) counts.set(t, (counts.get(t) || 0) + 1);
  const n = Math.max(1, allTexts.length);
  return [...counts.entries()]
    .filter(([t, c]) => c >= Math.min(2, groupTexts.length) && t.length >= 2)
    .map(([t, c]) => [t, (c / groupTexts.length) * Math.log((n + 1) / ((docFreq.get(t) || 0) + 1) + 1)])
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([t]) => t);
}
