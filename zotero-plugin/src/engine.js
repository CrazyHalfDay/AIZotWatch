// ZotWatch engine: the watch pipeline, independent of Zotero.
// Dependencies are injected so the same code runs inside Zotero and in Node tests:
//   http     - transport, see core/http.js
//   storage  - persistence, see store.js
//   library  - { getItems(): Promise<LibraryItem[]> }
//   settings - see settings.js

import { dedupeAgainstLibrary } from "./core/dedupe.js";
import { excludeByKeywords, filterRecent, filterWithoutAbstract, includeByKeywords, limitPreprints } from "./core/filters.js";
import { libraryVenues, parseWhitelist, venueKey } from "./core/journals.js";
import { ProfileRanker, buildClusters, computeTemporalWeights } from "./core/scoring.js";
import { contentForEmbedding, hashString, normalizeDoi } from "./core/text.js";
import { Matrix, toFloat32 } from "./core/vector.js";
import { createEmbedder } from "./embedding/providers.js";
import { createLlmClient, summarizePaper, translateTitles } from "./llm/llm.js";
import { fetchArxiv } from "./sources/arxiv.js";
import { fetchCrossref, resolveJournalIssns } from "./sources/crossref.js";
import { fetchEartharxiv } from "./sources/eartharxiv.js";
import { enrichAbstracts } from "./sources/openalex.js";
import { loadVectorStore, saveVectorStore } from "./store.js";

const DAY = 86400000;
const CANDIDATE_CACHE_TTL = 30 * DAY;
const JOURNAL_CACHE_TTL = 90 * DAY;
const MIN_ABSTRACT_ITEMS = 10;
const INBOX_FILE = "inbox.json";
// One write queue per storage backend, shared by all engine instances
const inboxLocks = new WeakMap();

export class CancelledError extends Error {
  constructor() {
    super("Cancelled");
    this.name = "CancelledError";
  }
}

/** Fields kept for each recommendation in the inbox (vectors are dropped). */
const WORK_FIELDS = [
  "id", "source", "title", "abstract", "authors", "doi", "url", "published", "venue", "issns", "type",
  "citations", "arxivId", "score", "similarity", "micro", "macro", "ifScore", "impactFactor", "label",
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
    let clusters = null;
    if (embedder.cacheable && !force) {
      const cached = await this.storage.readJSON("clusters.json").catch(() => null);
      if (cached?.digest === digest && cached.dim === matrix.dim) {
        clusters = cached.clusters.map((c) => ({ ...c, centroid: toFloat32(c.centroid) }));
      }
    }
    if (!clusters) {
      onProgress?.({ stage: "profile", message: "正在对文库进行语义聚类…" });
      const result = await buildClusters(matrix, weights, this.settings.scoring.clustering);
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

  async fetchCandidates(issns) {
    const s = this.settings.sources;
    const common = { now: this.now(), retry: this.retry, log: this.log };
    const tasks = [];
    if (s.crossref.enabled) tasks.push(["Crossref", () => fetchCrossref(this.http, { ...common, ...s.crossref, issns })]);
    if (s.arxiv.enabled) tasks.push(["arXiv", () => fetchArxiv(this.http, { ...common, ...s.arxiv })]);
    if (s.eartharxiv.enabled) tasks.push(["EarthArXiv", () => fetchEartharxiv(this.http, { ...common, ...s.eartharxiv })]);

    const settled = await Promise.allSettled(tasks.map(([, fn]) => fn()));
    const works = [];
    const errors = [];
    const perSource = {};
    settled.forEach((r, i) => {
      const name = tasks[i][0];
      if (r.status === "fulfilled") {
        works.push(...r.value);
        perSource[name] = r.value.length;
      } else {
        errors.push({ stage: "fetch", source: name, message: r.reason?.message || String(r.reason) });
        this.log(`${name} fetch failed: ${r.reason?.message || r.reason}`);
      }
    });
    return { works, errors, perSource };
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

  mergeIntoInbox(inbox, ranked, { runId, runAt, libraryDois }) {
    for (const w of ranked) {
      const existing = inbox.works[w.id];
      if (existing) {
        Object.assign(existing, pick(w), {
          status: existing.status,
          firstSeen: existing.firstSeen,
          firstRunId: existing.firstRunId,
          lastSeen: runAt,
          lastRunId: runId,
        });
      } else {
        inbox.works[w.id] = { ...pick(w), status: "new", firstSeen: runAt, lastSeen: runAt, firstRunId: runId, lastRunId: runId };
      }
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

    progress("journals", "确定关注期刊…");
    const { journals, issns, venues } = await this.resolveJournals(items);
    stats.journals = issns.length;
    stats.libraryVenues = venues.length;
    checkpoint();

    progress("fetch", "抓取最新文献…");
    const fetched = await this.fetchCandidates(issns);
    errors.push(...fetched.errors);
    stats.fetched = fetched.works.length;
    stats.perSource = fetched.perSource;
    checkpoint();

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

    const sc = this.settings.scoring;
    candidates = includeByKeywords(candidates, sc.includeKeywords);
    candidates = excludeByKeywords(candidates, sc.excludeKeywords);
    stats.afterKeywords = candidates.length;

    if (this.settings.sources.openalexAbstracts && candidates.some((w) => !w.abstract && w.doi)) {
      progress("enrich", "从 OpenAlex 补全缺失摘要…");
      stats.abstractsEnriched = await enrichAbstracts(this.http, candidates, {
        mailto: this.settings.sources.crossref.mailto,
        retry: this.retry,
        log: this.log,
      });
    }
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

    const libraryDois = new Set(items.map((i) => normalizeDoi(i.doi)).filter(Boolean));
    stats.mustRead = ranked.filter((w) => w.label === "must_read").length;
    const lastRun = { runId, runAt, stats, thresholds, errors, signature: embedder.signature, finishedAt: null };
    // Persist before the (slow, optional) LLM step so results are never lost
    await this.withInbox((inbox) => {
      const known = new Set(Object.keys(inbox.works));
      this.mergeIntoInbox(inbox, ranked, { runId, runAt, libraryDois });
      stats.newRecommendations = ranked.filter((w) => !known.has(w.id)).length;
      inbox.lastRun = lastRun;
    });

    if (this.settings.llm.enabled && ranked.length) {
      await this.runLlmTasks(ranked.map((w) => w.id), { progress, errors, checkpoint });
    }

    lastRun.finishedAt = this.now();
    await this.withInbox((inbox) => {
      inbox.lastRun = lastRun;
    });
    progress("done", `完成：${stats.recommended} 篇推荐（新增 ${stats.newRecommendations}）`);
    return lastRun;
  }

  async runLlmTasks(ids, { progress, errors, checkpoint }) {
    let llm;
    try {
      llm = createLlmClient(this.settings.llm, this.http, { retry: this.retry });
    } catch (e) {
      errors.push({ stage: "llm", message: e.message });
      return;
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

    const toSummarize = works
      .filter((w) => !w.summary)
      .sort((a, b) => b.score - a.score)
      .slice(0, this.settings.llm.summaryTopN);
    for (let i = 0; i < toSummarize.length; i++) {
      checkpoint();
      progress("llm", `生成 AI 摘要 ${i + 1}/${toSummarize.length}…`, { current: i + 1, total: toSummarize.length });
      try {
        const summary = await summarizePaper(llm, toSummarize[i]);
        await this.patchWorks([[toSummarize[i].id, { summary }]]);
      } catch (e) {
        errors.push({ stage: "llm", message: `摘要失败：${e.message}` });
        if (i === 0) break; // likely a configuration problem; don't repeat it N times
      }
    }
  }
}
