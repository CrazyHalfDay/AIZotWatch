// ZotWatch engine: the watch pipeline, independent of Zotero.
// Dependencies are injected so the same code runs inside Zotero and in Node tests:
//   http     - transport, see core/http.js
//   storage  - persistence, see store.js
//   library  - { getItems(): Promise<LibraryItem[]> }
//   settings - see settings.js

import { dedupeAgainstLibrary } from "./core/dedupe.js";
import { excludeByKeywords, filterRecent, filterWithoutAbstract, includeByKeywords, limitPreprints } from "./core/filters.js";
import { newestFirst, partitionByAnchors, splitFlagship } from "./core/flagship.js";
import { libraryVenues, parseWhitelist, venueKey } from "./core/journals.js";
import { ProfileRanker, buildClusters, computeTemporalWeights } from "./core/scoring.js";
import { contentForEmbedding, hashString, normalizeDoi, normalizeIssn } from "./core/text.js";
import { labelStoreOf, matchStoredLabels, sameLabelStore, storedLabelEntries, topKeywords, topicsOf } from "./core/topics.js";
import { attachFollowed, countByTrack, selectSummaryTargets, trackOf, unionNames } from "./core/tracks.js";
import { Matrix, toFloat32 } from "./core/vector.js";
import { createEmbedder } from "./embedding/providers.js";
import { createLlmClient, judgeRelevance, labelClusters, summarizePaper, translateTitles } from "./llm/llm.js";
import { enrichMissingAbstracts } from "./sources/abstracts.js";
import { fetchArxiv } from "./sources/arxiv.js";
import { fetchCrossref, resolveJournalIssns } from "./sources/crossref.js";
import { fetchEartharxiv } from "./sources/eartharxiv.js";
import { fetchFollowedWorks } from "./sources/followed.js";
import { loadVectorStore, saveVectorStore } from "./store.js";

const DAY = 86400000;
const CANDIDATE_CACHE_TTL = 30 * DAY;
const JOURNAL_CACHE_TTL = 90 * DAY;
const MIN_ABSTRACT_ITEMS = 10;
const INBOX_FILE = "inbox.json";
// Ids of followed-author works already fetched once: {version, seen: {id: lastSeenMs}}
const FOLLOWED_FILE = "followed.json";
// Seen ids are kept this long after they leave the search window
const FOLLOWED_SEEN_GRACE_DAYS = 60;
// Research-direction labels: {version, clusters: [{keys: [profile item keys], label}]}.
// Independent of clusters.json (never written in local mode): labels follow
// clusters by member overlap, whatever the embedder
const TOPIC_LABELS_FILE = "cluster-labels.json";
// One write queue per storage backend, shared by all engine instances
const inboxLocks = new WeakMap();

export class CancelledError extends Error {
  constructor() {
    super("Cancelled");
    this.name = "CancelledError";
  }
}

/**
 * Fields kept for each recommendation in the inbox (vectors are dropped).
 * `label` is must_read | consider | ignore for personal recommendations, or the
 * track label "flagship" / "followed" (see core/tracks.js); `followedAuthors`
 * names the followed authors of a work of any track, `topic` its research
 * direction and `gate` the flagship gate details ({pos, neg, judged}).
 */
const WORK_FIELDS = [
  "id", "source", "title", "abstract", "authors", "doi", "url", "published", "venue", "issns", "type",
  "citations", "arxivId", "score", "similarity", "micro", "macro", "ifScore", "impactFactor", "label",
  "followedAuthors", "topic", "gate",
];

function pick(work) {
  const out = {};
  for (const f of WORK_FIELDS) if (work[f] !== undefined) out[f] = work[f];
  return out;
}

function profileItemsOf(items) {
  const titled = items.filter((i) => i.title && i.title.trim());
  const withAbstract = titled.filter((i) => i.abstract && i.abstract.trim());
  // Like the Python profile, prefer items with abstracts; fall back to titles for tiny libraries
  return withAbstract.length >= MIN_ABSTRACT_ITEMS ? withAbstract : titled;
}

export class ZotWatchEngine {
  constructor({ http, storage, library, settings, log = () => {}, now = () => Date.now(), retry = {} }) {
    this.http = http;
    this.storage = storage;
    this.library = library;
    this.settings = settings;
    this.log = log;
    this.now = now;
    this.retry = retry;
  }

  // ---------------------------------------------------------------- profile

  /**
   * Embed the library incrementally (only new/changed items) and (re)cluster it
   * when its content changed.
   */
  async buildProfile(items, { embedder, onProgress, force = false } = {}) {
    embedder ||= createEmbedder(this.settings.embedding, this.http, { retry: this.retry });
    const profileItems = profileItemsOf(items);
    const texts = profileItems.map((i) => contentForEmbedding(i.title, i.abstract));
    const hashes = texts.map((t) => hashString(t));
    const stats = { items: profileItems.length, embedded: 0, reused: 0, clustersRebuilt: false };

    if (!profileItems.length) {
      return { profile: { matrix: new Matrix(0, 1), weights: new Float32Array(0), clusters: [] }, stats, embedder };
    }

    let matrix;
    if (!embedder.cacheable) {
      if (embedder.requiresFit) embedder.fit(texts);
      const vectors = await embedder.embed(texts);
      matrix = Matrix.fromRows(vectors);
      stats.embedded = vectors.length;
    } else {
      const stored = force ? null : await loadVectorStore(this.storage, "profile");
      const usable = stored && stored.signature === embedder.signature ? stored : null;
      const previous = new Map();
      if (usable) usable.ids.forEach((key, row) => previous.set(key, { row, hash: usable.meta[row]?.hash }));

      const missing = [];
      profileItems.forEach((item, i) => {
        const prev = previous.get(item.key);
        if (!prev || prev.hash !== hashes[i]) missing.push(i);
      });
      onProgress?.({ stage: "profile", message: `需要向量化 ${missing.length} / ${profileItems.length} 篇文库条目` });

      const fresh = missing.length
        ? await embedder.embed(missing.map((i) => texts[i]), {
            onProgress: (done, total) => onProgress?.({ stage: "profile", message: `文库向量化 ${done}/${total}`, current: done, total }),
          })
        : [];
      const dim = fresh[0]?.length || usable?.dim;
      if (usable && fresh.length && usable.dim !== dim) {
        // Dimension changed under the same signature: rebuild everything
        return this.buildProfile(items, { embedder, onProgress, force: true });
      }

      matrix = new Matrix(profileItems.length, dim);
      const freshByIndex = new Map(missing.map((idx, j) => [idx, fresh[j]]));
      profileItems.forEach((item, i) => {
        const vec = freshByIndex.get(i);
        if (vec) matrix.data.set(vec, i * dim);
        else matrix.data.set(usable.vectors.subarray(previous.get(item.key).row * dim, (previous.get(item.key).row + 1) * dim), i * dim);
      });
      stats.embedded = missing.length;
      stats.reused = profileItems.length - missing.length;

      const changed = missing.length > 0 || !usable || usable.ids.length !== profileItems.length;
      if (changed) {
        await saveVectorStore(this.storage, "profile", {
          signature: embedder.signature,
          dim,
          ids: profileItems.map((i) => i.key),
          meta: profileItems.map((item, i) => ({ hash: hashes[i] })),
          vectors: matrix.data,
        });
      }
    }

    const temporal = this.settings.scoring.temporal;
    const weights = computeTemporalWeights(
      profileItems.map((i) => i.dateAdded),
      { halflifeDays: temporal.halflifeDays, minWeight: temporal.minWeight, now: this.now() },
    );

    // Clusters are cached by a digest of the profile content
    const digest = hashString(
      `${embedder.signature}|${temporal.halflifeDays}|${profileItems.map((item, i) => `${item.key}:${hashes[i]}`).sort().join(",")}`,
    );
    const outputTag = this.settings.output.tag;
    let clusters = null;
    if (embedder.cacheable && !force) {
      const cached = await this.storage.readJSON("clusters.json").catch(() => null);
      // Caches of 0.1.0 lack the member keys topic labels need: recluster once
      if (cached?.digest === digest && cached.dim === matrix.dim && Array.isArray(cached.clusters) && cached.clusters.every((c) => Array.isArray(c.memberKeys))) {
        // Tags are not part of the digest, so keywords are recounted from the current tags
        const byKey = new Map(profileItems.map((item) => [item.key, item]));
        clusters = cached.clusters.map((c) => ({
          ...c,
          centroid: toFloat32(c.centroid),
          keywords: topKeywords(c.memberKeys.map((key) => byKey.get(key)).filter(Boolean), { outputTag }),
        }));
      }
    }
    if (!clusters) {
      onProgress?.({ stage: "profile", message: "正在对文库进行语义聚类…" });
      const result = await buildClusters(matrix, weights, { ...this.settings.scoring.clustering, items: profileItems, outputTag });
      clusters = result.clusters;
      stats.clustersRebuilt = true;
      stats.silhouette = result.silhouette;
      if (embedder.cacheable) {
        await this.storage.writeJSON("clusters.json", {
          digest,
          dim: matrix.dim,
          silhouette: result.silhouette,
          clusters: clusters.map((c) => ({ ...c, centroid: Array.from(c.centroid) })),
        });
      }
    }
    stats.clusters = clusters.length;
    return { profile: { matrix, weights, clusters }, stats, embedder };
  }

  // --------------------------------------------------------------- journals

  /** Whitelist journals plus (optionally) the top venues of the library. */
  async resolveJournals(items) {
    const cfg = this.settings.journals;
    const journals = parseWhitelist(cfg.whitelist);
    const issns = new Set(journals.keys());
    let venues = [];
    if (cfg.autoFromLibrary) {
      venues = libraryVenues(items)
        .filter((v) => v.count >= cfg.autoMinItems)
        .slice(0, cfg.autoTopN);
      const cache = (await this.storage.readJSON("journals-cache.json").catch(() => null)) || { titles: {} };
      let dirty = false;
      for (const v of venues) {
        let found = v.issns;
        if (!found.length) {
          const key = venueKey(v.title);
          const hit = cache.titles[key];
          if (hit && this.now() - hit.at < JOURNAL_CACHE_TTL) {
            found = hit.issns;
          } else {
            try {
              found = await resolveJournalIssns(this.http, v.title, { mailto: this.settings.sources.crossref.mailto, retry: this.retry });
              cache.titles[key] = { issns: found, at: this.now() };
              dirty = true;
            } catch (e) {
              if (e?.cancelled) throw e;
              this.log(`Journal lookup failed for "${v.title}": ${e.message}`);
            }
          }
        }
        v.issns = found;
        found.forEach((issn) => issns.add(issn));
      }
      if (dirty) await this.storage.writeJSON("journals-cache.json", cache);
    }
    return { journals, issns: [...issns], venues };
  }

  // ------------------------------------------------------------- candidates

  /**
   * Fetch this run's candidates from every enabled source. With flagship ISSNs
   * the flagship journals get their own Crossref query and result budget
   * (reported as "顶刊"), so high-volume journals (Nat Commun, Sci Adv, PNAS)
   * cannot crowd out the personal ones; they are fetched even when the
   * personal Crossref source is off. Works are de-duplicated by id.
   * @param {string[]} [options.flagshipIssns] - normalized flagship ISSNs (empty: track off)
   */
  async fetchCandidates(issns, { flagshipIssns = [] } = {}) {
    const s = this.settings.sources;
    const common = { now: this.now(), retry: this.retry, log: this.log };
    const flagshipSet = new Set(flagshipIssns);
    const tasks = [];
    if (s.crossref.enabled) {
      const personal = flagshipSet.size ? issns.filter((issn) => !flagshipSet.has(normalizeIssn(issn))) : issns;
      tasks.push(["Crossref", "Crossref", () => fetchCrossref(this.http, { ...common, ...s.crossref, issns: personal })]);
    }
    if (flagshipSet.size) {
      tasks.push(["顶刊", "Crossref 顶刊", () => fetchCrossref(this.http, { ...common, ...s.crossref, issns: [...flagshipSet] })]);
    }
    if (s.arxiv.enabled) tasks.push(["arXiv", "arXiv", () => fetchArxiv(this.http, { ...common, ...s.arxiv })]);
    if (s.eartharxiv.enabled) tasks.push(["EarthArXiv", "EarthArXiv", () => fetchEartharxiv(this.http, { ...common, ...s.eartharxiv })]);

    const settled = await Promise.allSettled(tasks.map(([, , fn]) => fn()));
    const cancelled = settled.find((r) => r.status === "rejected" && r.reason?.cancelled);
    if (cancelled) throw cancelled.reason;
    const works = [];
    const seen = new Set();
    const errors = [];
    const perSource = {};
    settled.forEach((r, i) => {
      const [name, source] = tasks[i];
      if (r.status === "fulfilled") {
        // A journal listed in both lists (e.g. its print ISSN here, its online one there) answers twice
        for (const w of r.value) {
          if (seen.has(w.id)) continue;
          seen.add(w.id);
          works.push(w);
        }
        perSource[name] = r.value.length;
      } else {
        errors.push({ stage: "fetch", source, message: r.reason?.message || String(r.reason) });
        this.log(`${source} fetch failed: ${r.reason?.message || r.reason}`);
      }
    });
    return { works, errors, perSource };
  }

  /** Recent works of the followed authors; null (with a non-fatal error) when the fetch failed. */
  async fetchFollowed(errors) {
    const cfg = this.settings.followed;
    try {
      return await fetchFollowedWorks(this.http, {
        authors: cfg.authors,
        daysBack: cfg.daysBack,
        mailto: this.settings.sources.crossref.mailto,
        now: this.now(),
        retry: this.retry,
        log: this.log,
      });
    } catch (e) {
      if (e?.cancelled) throw e;
      errors.push({ stage: "followed", source: "OpenAlex", message: e?.message || String(e) });
      this.log(`Followed authors fetch failed: ${e?.message || e}`);
      return null;
    }
  }

  /**
   * Fill in missing abstracts of this run's works (all tracks) in one pass:
   * Crossref -> Elsevier -> Springer Nature -> OpenAlex, with a persistent
   * cache (see sources/abstracts.js). Best effort: provider problems become
   * non-fatal errors, only cancellation propagates.
   * @returns {Promise<object|null>} counts, or null when skipped (every tier off, or nothing to look up)
   */
  async enrichAbstracts(works, { progress, errors, checkpoint }) {
    const cfg = this.settings.abstracts || { crossref: false, openalex: this.settings.sources.openalexAbstracts, elsevierKey: "", springerKey: "" };
    if (!cfg.crossref && !cfg.openalex && !cfg.elsevierKey && !cfg.springerKey) return null;
    const targets = works.filter((w) => !(typeof w.abstract === "string" ? w.abstract.trim() : w.abstract) && normalizeDoi(w.doi));
    if (!targets.length) return null;

    const message = `补全缺失摘要（${targets.length} 篇）…`;
    progress("enrich", message);
    try {
      const result = await enrichMissingAbstracts(this.http, works, {
        ...cfg,
        mailto: this.settings.sources.crossref.mailto,
        cache: this.storage,
        now: this.now(),
        retry: this.retry,
        log: this.log,
        checkpoint,
        onProgress: (done, total) => progress("enrich", message, { current: done, total }),
      });
      const { errors: providerErrors, ...counts } = result;
      for (const e of providerErrors) errors.push({ stage: "enrich", source: e.source, message: e.message });
      return counts;
    } catch (e) {
      if (e?.cancelled || e instanceof CancelledError) throw e;
      errors.push({ stage: "enrich", message: e?.message || String(e) });
      this.log(`Abstract enrichment failed: ${e?.message || e}`);
      return null;
    }
  }

  async loadFollowedSeen() {
    const data = await this.storage.readJSON(FOLLOWED_FILE).catch(() => null);
    const seen = data?.seen;
    return seen && typeof seen === "object" && !Array.isArray(seen) ? { ...seen } : {};
  }

  /** Persist the seen-set, dropping ids not fetched for longer than the window plus a grace period. */
  async saveFollowedSeen(seen, runAt) {
    const maxAge = (this.settings.followed.daysBack + FOLLOWED_SEEN_GRACE_DAYS) * DAY;
    const kept = {};
    for (const [id, at] of Object.entries(seen)) {
      if (Number.isFinite(at) && runAt - at <= maxAge) kept[id] = at;
    }
    await this.storage.writeJSON(FOLLOWED_FILE, { version: 1, seen: kept });
  }

  /**
   * Flagship track (顶刊地学速览, Python _select_flagship + GeoscienceGate):
   * flagship-journal articles with an abstract, published within the recency
   * window, are embedded with the candidates' embedder and cache, compared
   * with the field anchors and partitioned (core/flagship.js); the gray zone is
   * judged by the LLM when configured, otherwise accepted. Accepted articles
   * get a display score (no similarity gate), label "flagship" and their gate
   * details, newest first, capped. Best effort: any failure but a cancel
   * becomes a non-fatal error and an empty track.
   * @returns {Promise<{works: object[], stats: object}>}
   */
  async selectFlagship(candidates, { profile, journals, embedder, runAt, progress, errors, checkpoint }) {
    const cfg = this.settings.flagship;
    const stats = { candidates: candidates.length, withAbstract: 0, recent: 0, accepted: 0, gray: 0, judged: 0, rejected: 0, recommended: 0 };
    // Always require an abstract (the gate reads it), whatever scoring.requireAbstract says
    let works = filterWithoutAbstract(candidates);
    stats.withAbstract = works.length;
    // Recency before the gate: the same result per article as Python (which
    // filters afterwards) with fewer embeddings and LLM calls
    works = filterRecent(works, this.settings.scoring.recentDays, runAt);
    stats.recent = works.length;
    if (!works.length) return { works: [], stats };

    try {
      progress("flagship", `顶刊地学：判断 ${works.length} 篇文章是否属于地学…`);
      await this.embedCandidates(works, embedder, (p) =>
        progress("flagship", `顶刊文章向量化 ${p.current}/${p.total}`, { current: p.current, total: p.total }),
      );
      checkpoint();
      // Re-embedded every run (one small request); never stored in the candidate cache
      const anchors = await embedder.embed([cfg.positiveAnchor, ...cfg.negativeAnchors], { inputType: "query" });
      checkpoint();
      const negatives = anchors.slice(1);
      const part = partitionByAnchors(
        works.map((w) => w.vector),
        anchors[0],
        negatives,
        { minScore: cfg.minScore, grayLow: cfg.grayLow },
      );
      stats.accepted = part.accepted.length;
      stats.gray = part.gray.length;

      const gray = part.gray.map((i) => works[i]);
      let keptGray = gray; // inclusive by design when no LLM judges the gray zone
      let confirmed = new Set();
      if (gray.length && cfg.llmFallback && this.settings.llm.enabled && this.settings.llm.apiKey) {
        progress("flagship", `大模型复核 ${gray.length} 篇灰区文章…`);
        const llm = createLlmClient(this.settings.llm, this.http, { retry: this.retry });
        const verdict = await judgeRelevance(llm, gray, cfg.llmBoundary, { batchSize: cfg.llmBatchSize });
        keptGray = verdict.kept;
        confirmed = verdict.confirmed;
        stats.judged = verdict.judged;
        if (verdict.error) {
          errors.push({ stage: "flagship", message: `大模型复核失败：${verdict.error}` });
          this.log(`Flagship LLM check failed (${verdict.failedBatches} batches kept): ${verdict.error}`);
        }
        checkpoint();
      }

      const keep = new Set([...part.accepted.map((i) => works[i]), ...keptGray]);
      const passed = works.filter((w) => keep.has(w));
      stats.rejected = works.length - passed.length;
      this.log(`Flagship geoscience gate: ${passed.length}/${works.length} passed (${gray.length} gray-zone, ${stats.judged} judged by the LLM)`);

      const gateOf = new Map(works.map((w, i) => [w, part.scores[i]]));
      const round = (x) => Math.round(x * 1000) / 1000;
      const ranker = new ProfileRanker(profile, journals, this.settings.scoring);
      const scored = await ranker.scoreWorks(passed);
      const labelled = scored.map((w, i) => {
        const { pos, neg } = gateOf.get(passed[i]);
        return { ...w, label: "flagship", gate: { pos: round(pos), neg: negatives.length ? round(neg) : null, judged: confirmed.has(passed[i]) } };
      });
      const selected = newestFirst(labelled, cfg.maxResults);
      stats.recommended = selected.length;
      return { works: selected, stats };
    } catch (e) {
      if (e?.cancelled || e instanceof CancelledError) throw e;
      errors.push({ stage: "flagship", message: e?.message || String(e) });
      this.log(`Flagship track failed: ${e?.message || e}`);
      return { works: [], stats };
    }
  }

  /** Embed candidates, reusing cached vectors for previously seen works. */
  async embedCandidates(works, embedder, onProgress) {
    const texts = works.map((w) => contentForEmbedding(w.title, w.abstract));
    if (!embedder.cacheable) {
      const vectors = await embedder.embed(texts);
      works.forEach((w, i) => (w.vector = vectors[i]));
      return { embedded: works.length, reused: 0 };
    }

    const now = this.now();
    const hashes = texts.map((t) => hashString(t));
    const stored = await loadVectorStore(this.storage, "candidates");
    const cache = new Map();
    if (stored && stored.signature === embedder.signature) {
      stored.ids.forEach((id, row) => {
        const meta = stored.meta[row] || {};
        if (now - (meta.at || 0) < CANDIDATE_CACHE_TTL) {
          cache.set(id, { hash: meta.hash, at: meta.at, vector: stored.vectors.subarray(row * stored.dim, (row + 1) * stored.dim) });
        }
      });
    }

    const missing = [];
    works.forEach((w, i) => {
      const hit = cache.get(w.id);
      if (hit && hit.hash === hashes[i]) w.vector = hit.vector;
      else missing.push(i);
    });
    if (missing.length) {
      const vectors = await embedder.embed(missing.map((i) => texts[i]), {
        onProgress: (done, total) => onProgress?.({ stage: "embed", message: `候选文献向量化 ${done}/${total}`, current: done, total }),
      });
      missing.forEach((idx, j) => {
        works[idx].vector = vectors[j];
        cache.set(works[idx].id, { hash: hashes[idx], at: now, vector: vectors[j] });
      });
    }

    const dim = works.find((w) => w.vector)?.vector.length || stored?.dim;
    if (missing.length && dim) {
      const entries = [...cache.entries()].filter(([, v]) => v.vector.length === dim);
      const vectors = new Float32Array(entries.length * dim);
      entries.forEach(([, v], i) => vectors.set(v.vector, i * dim));
      await saveVectorStore(this.storage, "candidates", {
        signature: embedder.signature,
        dim,
        ids: entries.map(([id]) => id),
        meta: entries.map(([, v]) => ({ hash: v.hash, at: v.at })),
        vectors,
      });
    }
    return { embedded: missing.length, reused: works.length - missing.length };
  }

  // ----------------------------------------------------------------- topics

  /**
   * Research-direction labels of the current clusters, carried over from
   * cluster-labels.json by member-key Jaccard (see core/topics.js). When every
   * cluster keeps its label, the store follows the new memberships so that
   * gradual library growth never drops a label. Null when topic labels are off
   * or there are no clusters.
   * @returns {Promise<{clusters: object[], labels: (string|null)[], nearest: Map<string, number>, generated: number}|null>}
   */
  async loadTopics(clusters) {
    if (!this.settings.llm.topicLabels || !clusters.length || !clusters.every((c) => Array.isArray(c.memberKeys))) return null;
    const stored = await this.storage.readJSON(TOPIC_LABELS_FILE).catch(() => null);
    const labels = matchStoredLabels(clusters, storedLabelEntries(stored));
    if (labels.every(Boolean)) {
      const next = labelStoreOf(clusters, labels);
      if (!sameLabelStore(stored, next)) await this.storage.writeJSON(TOPIC_LABELS_FILE, next);
    }
    return { clusters, labels, nearest: new Map(), generated: 0 };
  }

  /**
   * Label every unlabeled cluster with one LLM call (Python batch prompt),
   * store the labels and patch the topic onto this run's works whose nearest
   * cluster just got one. A failed request or a malformed reply stores
   * nothing (retried next run) and becomes a non-fatal error; a cancel propagates.
   */
  async generateTopicLabels(llm, topics, { progress, errors }) {
    const todo = [];
    topics.labels.forEach((label, i) => {
      if (!label) todo.push(i);
    });
    if (!todo.length) return;
    progress("llm", `生成 ${todo.length} 个研究方向标签…`);
    let generated;
    try {
      generated = await labelClusters(llm, todo.map((i) => topics.clusters[i]));
    } catch (e) {
      if (e?.cancelled) throw e;
      errors.push({ stage: "llm", message: `研究方向标签生成失败：${e?.message || e}` });
      this.log(`Topic labels failed: ${e?.message || e}`);
      return;
    }
    todo.forEach((idx, j) => (topics.labels[idx] = generated[j]));
    topics.generated = todo.length;
    await this.storage.writeJSON(TOPIC_LABELS_FILE, labelStoreOf(topics.clusters, topics.labels));
    this.log(`Topic labels: generated ${todo.length} of ${topics.labels.length}`);
    const fresh = new Set(todo);
    const patches = [...topics.nearest].filter(([, idx]) => fresh.has(idx)).map(([id, idx]) => [id, { topic: topics.labels[idx] }]);
    if (patches.length) await this.patchWorks(patches);
  }

  /** Labeled research directions of the library, largest first: [{label, size}]. */
  async getTopics() {
    return topicsOf(await this.storage.readJSON(TOPIC_LABELS_FILE).catch(() => null));
  }

  // ------------------------------------------------------------------ inbox

  async loadInbox() {
    const inbox = await this.storage.readJSON(INBOX_FILE).catch(() => null);
    return inbox && inbox.works ? inbox : { version: 1, works: {}, lastRun: null };
  }

  /**
   * Locked read-modify-write of the inbox. Runs and UI actions (add, dismiss,
   * summarize) may overlap, so every write goes through here; the mutator must
   * not do slow work such as network calls.
   */
  async withInbox(mutator) {
    const previous = inboxLocks.get(this.storage) || Promise.resolve();
    let release;
    const current = new Promise((resolve) => (release = resolve));
    inboxLocks.set(this.storage, previous.then(() => current));
    await previous;
    try {
      const inbox = await this.loadInbox();
      const result = await mutator(inbox);
      await this.storage.writeJSON(INBOX_FILE, inbox);
      return result;
    } finally {
      release();
    }
  }

  async updateWork(id, patch) {
    return this.withInbox((inbox) => {
      const work = inbox.works[id];
      if (!work) throw new Error(`Unknown recommendation: ${id}`);
      Object.assign(work, patch, patch.status ? { statusAt: this.now() } : {});
      return work;
    });
  }

  /** Generate (and store) an AI summary for one recommendation. */
  async summarize(id) {
    const work = (await this.loadInbox()).works[id];
    if (!work) throw new Error(`Unknown recommendation: ${id}`);
    const llm = createLlmClient(this.settings.llm, this.http, { retry: this.retry });
    const summary = await summarizePaper(llm, work);
    await this.updateWork(id, { summary });
    return summary;
  }

  /** Set a field on several works at once (skipping ones removed meanwhile). */
  async patchWorks(patches) {
    await this.withInbox((inbox) => {
      for (const [id, patch] of patches) if (inbox.works[id]) Object.assign(inbox.works[id], patch);
    });
  }

  /**
   * Merge this run's works (all tracks) into the inbox and prune old entries.
   * A personal or flagship work overwrites its stored copy (label and score
   * included) but keeps the status, the first-seen data and the union of
   * followed authors. A followed-only work that is already stored (by id or
   * DOI, any status) only adds its followed authors to that entry.
   * @param {string[]} [options.followedOrder] - configured followed-author names (ordering of unions)
   * @returns {string[]} ids of the entries this run inserted
   */
  mergeIntoInbox(inbox, works, { runId, runAt, libraryDois, followedOrder = [] }) {
    const inserted = [];
    let storedDois = null;
    const storedByDoi = (doi) => {
      if (!doi) return null;
      if (!storedDois) {
        storedDois = new Map();
        for (const stored of Object.values(inbox.works)) {
          const d = normalizeDoi(stored.doi);
          if (d && !storedDois.has(d)) storedDois.set(d, stored);
        }
      }
      return storedDois.get(doi) || null;
    };
    for (const w of works) {
      const followedOnly = trackOf(w) === "followed";
      const existing = inbox.works[w.id] || (followedOnly ? storedByDoi(normalizeDoi(w.doi)) : null);
      if (!existing) {
        inbox.works[w.id] = { ...pick(w), status: "new", firstSeen: runAt, lastSeen: runAt, firstRunId: runId, lastRunId: runId };
        inserted.push(w.id);
        continue;
      }
      const followedAuthors = unionNames(existing.followedAuthors, w.followedAuthors, followedOrder);
      if (!followedOnly) {
        Object.assign(existing, pick(w), {
          status: existing.status,
          firstSeen: existing.firstSeen,
          firstRunId: existing.firstRunId,
          lastSeen: runAt,
          lastRunId: runId,
        });
      }
      if (followedAuthors.length) existing.followedAuthors = followedAuthors;
    }

    const retention = this.settings.inbox.retentionDays * DAY;
    for (const [id, w] of Object.entries(inbox.works)) {
      const doi = normalizeDoi(w.doi);
      if (w.status === "new" && doi && libraryDois.has(doi)) {
        // Added to the library by other means since it was recommended
        w.status = "added";
        w.statusAt = runAt;
      }
      if (w.status === "new" && runAt - w.lastSeen > retention) delete inbox.works[id];
      else if (w.status !== "new" && runAt - (w.statusAt || w.lastSeen) > Math.max(retention, 60 * DAY)) delete inbox.works[id];
    }
    return inserted;
  }

  // -------------------------------------------------------------------- run

  /**
   * Run the full watch pipeline and merge the results into the inbox.
   * @returns {Promise<object>} run summary (also stored as inbox.lastRun)
   */
  async run({ onProgress = () => {}, isCancelled = () => false } = {}) {
    const runAt = this.now();
    const runId = `run-${runAt}`;
    const progress = (stage, message, extra = {}) => onProgress({ stage, message, ...extra });
    const checkpoint = () => {
      if (isCancelled()) throw new CancelledError();
    };
    const stats = {};
    const errors = [];

    progress("library", "读取 Zotero 文库…");
    const items = await this.library.getItems();
    stats.libraryItems = items.length;
    checkpoint();

    progress("profile", "更新研究画像…");
    const { profile, stats: profileStats, embedder } = await this.buildProfile(items, { onProgress });
    stats.profile = profileStats;
    checkpoint();
    // Research-direction labels kept from earlier runs (new ones come after the merge)
    const topics = await this.loadTopics(profile.clusters);

    progress("journals", "确定关注期刊…");
    const { journals, issns, venues } = await this.resolveJournals(items);
    stats.journals = issns.length;
    stats.libraryVenues = venues.length;
    checkpoint();

    // Flagship journals (顶刊地学速览) are fetched separately and leave the personal pipeline after dedupe
    const flagshipCfg = this.settings.flagship;
    const flagshipIssns = flagshipCfg?.enabled ? flagshipCfg.issns : [];

    progress("fetch", "抓取最新文献…");
    const fetched = await this.fetchCandidates(issns, { flagshipIssns });
    errors.push(...fetched.errors);
    stats.fetched = fetched.works.length;
    stats.perSource = fetched.perSource;
    checkpoint();

    const followedCfg = this.settings.followed || { enabled: false, authors: [], invalid: [] };
    let followedFetched = null;
    if (followedCfg.enabled) {
      for (const line of followedCfg.invalid) errors.push({ stage: "followed", message: `无法识别的作者行：${line}` });
      if (followedCfg.authors.length) {
        progress("followed", "获取关注作者的新论文…");
        followedFetched = await this.fetchFollowed(errors);
        checkpoint();
      }
    }

    const snapshot = await this.loadInbox();
    // Works the user already added or dismissed are never recommended again
    const handledIds = new Set();
    const handledDois = [];
    for (const w of Object.values(snapshot.works)) {
      if (w.status === "new") continue;
      handledIds.add(w.id);
      if (w.doi) handledDois.push(w.doi);
    }

    progress("dedupe", "与文库去重…");
    let candidates = dedupeAgainstLibrary(
      fetched.works.filter((w) => !handledIds.has(w.id)),
      items,
      { extraDois: handledDois },
    );
    stats.afterDedupe = candidates.length;
    // Flagship-journal articles leave the personal pipeline here (Python
    // order): no keyword filters, similarity gate, preprint limit or top K
    let flagshipCandidates = [];
    if (flagshipIssns.length) {
      ({ flagship: flagshipCandidates, personal: candidates } = splitFlagship(candidates, new Set(flagshipIssns)));
    }

    // Followed works: no scoring and no filters (Python parity), only a
    // seen-set (each work is shown once) and the same dedupe as candidates
    let followedNew = [];
    let followedSeen = null;
    if (followedFetched) {
      followedSeen = await this.loadFollowedSeen();
      const isSeen = (w) => Object.hasOwn(followedSeen, w.id) || (w.openalexId && Object.hasOwn(followedSeen, `openalex:${w.openalexId}`));
      const unseen = followedFetched.filter((w) => !isSeen(w) && !handledIds.has(w.id));
      for (const w of followedFetched) {
        followedSeen[w.id] = runAt;
        if (w.openalexId) followedSeen[`openalex:${w.openalexId}`] = runAt;
      }
      followedNew = dedupeAgainstLibrary(unseen, items, { extraDois: handledDois });
      stats.followed = { authors: followedCfg.authors.length, fetched: followedFetched.length, new: followedNew.length };
    }

    // One abstract-enrichment pass over every track that may lack abstracts
    // (flagship candidates join this list), before the keyword filters so they
    // also match filled-in abstracts (Python order)
    const abstracts = await this.enrichAbstracts([...candidates, ...flagshipCandidates, ...followedNew], { progress, errors, checkpoint });
    if (abstracts) {
      stats.abstracts = abstracts;
      stats.abstractsEnriched = abstracts.filled;
    }
    checkpoint();

    const sc = this.settings.scoring;
    candidates = includeByKeywords(candidates, sc.includeKeywords);
    candidates = excludeByKeywords(candidates, sc.excludeKeywords);
    stats.afterKeywords = candidates.length;
    if (sc.requireAbstract) candidates = filterWithoutAbstract(candidates);
    stats.afterAbstract = candidates.length;
    checkpoint();

    let ranked = [];
    let thresholds = null;
    if (candidates.length) {
      progress("embed", `向量化 ${candidates.length} 篇候选文献…`);
      stats.candidateEmbedding = await this.embedCandidates(candidates, embedder, onProgress);
      checkpoint();

      progress("rank", "计算相关性评分…");
      const ranker = new ProfileRanker(profile, journals, sc);
      const result = await ranker.rank(candidates, {
        onProgress: (done, total) => progress("rank", `评分 ${done}/${total}`, { current: done, total }),
      });
      thresholds = result.thresholds;
      stats.afterGate = result.ranked.length;
      ranked = filterRecent(result.ranked, sc.recentDays, runAt);
      stats.afterRecent = ranked.length;
      ranked = limitPreprints(ranked, sc.maxPreprintRatio).slice(0, sc.topK);
    }
    stats.recommended = ranked.length;
    checkpoint();

    let flagshipWorks = [];
    if (flagshipIssns.length) {
      const result = await this.selectFlagship(flagshipCandidates, { profile, journals, embedder, runAt, progress, errors, checkpoint });
      flagshipWorks = result.works;
      stats.flagship = result.stats;
      checkpoint();
    }

    // Personal and flagship results of this run; every fetched followed work
    // annotates a matching one, and only new unmatched ones become entries
    const tracked = [...ranked, ...flagshipWorks];
    // Topic of each work: the label of its nearest cluster (followed-only works are not embedded)
    if (topics) {
      for (const w of tracked) {
        if (!Number.isInteger(w.nearest)) continue;
        topics.nearest.set(w.id, w.nearest);
        if (topics.labels[w.nearest]) w.topic = topics.labels[w.nearest];
      }
    }
    const followedOrder = followedCfg.authors.map((a) => a.name);
    const newFollowedIds = new Set(followedNew.map((w) => w.id));
    const followedOnly = attachFollowed(tracked, followedFetched || [], followedOrder).filter((w) => newFollowedIds.has(w.id));

    const libraryDois = new Set(items.map((i) => normalizeDoi(i.doi)).filter(Boolean));
    stats.mustRead = ranked.filter((w) => w.label === "must_read").length;
    const topicStats = () => topics && { clusters: topics.labels.length, labeled: topics.labels.filter(Boolean).length, generated: topics.generated };
    if (topics) stats.topics = topicStats();
    const lastRun = { runId, runAt, stats, thresholds, errors, signature: embedder.signature, finishedAt: null };
    let inserted = [];
    // Persist before the (slow, optional) LLM step so results are never lost
    await this.withInbox((inbox) => {
      inserted = this.mergeIntoInbox(inbox, [...tracked, ...followedOnly], { runId, runAt, libraryDois, followedOrder });
      stats.newRecommendations = inserted.length;
      stats.newByLabel = countByTrack(inserted.map((id) => inbox.works[id]).filter(Boolean));
      inbox.lastRun = lastRun;
    });
    // Only after the merge: a run cancelled before it shows these works again next time
    if (followedSeen) await this.saveFollowedSeen(followedSeen, runAt);

    const llmIds = [...new Set([...tracked.map((w) => w.id), ...inserted])];
    // Unlabeled research directions are labeled even on a run without new works
    const labelsPending = !!topics && !!this.settings.llm.apiKey && topics.labels.some((label) => !label);
    if (this.settings.llm.enabled && (llmIds.length || labelsPending)) {
      await this.runLlmTasks(llmIds, { progress, errors, checkpoint, topics });
      if (topics) stats.topics = topicStats();
    }

    lastRun.finishedAt = this.now();
    await this.withInbox((inbox) => {
      inbox.lastRun = lastRun;
    });
    const { personal, flagship, followed } = stats.newByLabel;
    const others = `${flagship ? `，顶刊地学 ${flagship} 篇` : ""}${followed ? `，关注作者 ${followed} 篇` : ""}`;
    progress("done", `完成：${stats.recommended} 篇推荐（新增 ${personal}）${others}`);
    return lastRun;
  }

  /**
   * LLM post-processing of this run: labels for unlabeled research directions
   * (see generateTopicLabels), title translation of every new work (all
   * tracks), then summaries within the per-track budget (see
   * selectSummaryTargets).
   */
  async runLlmTasks(ids, { progress, errors, checkpoint, topics = null }) {
    let llm;
    try {
      llm = createLlmClient(this.settings.llm, this.http, { retry: this.retry });
    } catch (e) {
      errors.push({ stage: "llm", message: e.message });
      return;
    }
    if (topics) {
      await this.generateTopicLabels(llm, topics, { progress, errors });
      checkpoint();
    }
    const inbox = await this.loadInbox();
    const works = ids.map((id) => inbox.works[id]).filter((w) => w && w.status === "new");

    if (this.settings.llm.translateTitles) {
      const todo = works.filter((w) => !w.translatedTitle && !/[\u4e00-\u9fff]/.test(w.title));
      if (todo.length) {
        progress("llm", `翻译 ${todo.length} 个标题…`);
        const translated = await translateTitles(llm, todo.map((w) => w.title));
        await this.patchWorks(todo.map((w, i) => [w.id, translated[i] ? { translatedTitle: translated[i] } : {}]));
      }
    }

    const toSummarize = selectSummaryTargets(works, this.settings.llm.summaryTopN);
    for (let i = 0; i < toSummarize.length; i++) {
      checkpoint();
      progress("llm", `生成 AI 摘要 ${i + 1}/${toSummarize.length}…`, { current: i + 1, total: toSummarize.length });
      try {
        const summary = await summarizePaper(llm, toSummarize[i]);
        await this.patchWorks([[toSummarize[i].id, { summary }]]);
      } catch (e) {
        if (e?.cancelled) throw e;
        errors.push({ stage: "llm", message: `摘要失败：${e.message}` });
        if (i === 0) break; // likely a configuration problem; don't repeat it N times
      }
    }
  }
}
