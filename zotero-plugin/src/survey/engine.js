// Topic survey engine (主题文献调研), independent of Zotero like engine.js.
// Dependencies are injected:
//   http     - transport (core/http.js), carrying the cancel signal and the shared rate limiter
//   storage  - persistence (store.js): surveys.json, survey-<id>.json, survey-cache.json
//   library  - { getItems(), getItemsByKeys?(keys), getItemFullText?(key, {maxChars}) }
//   settings - see settings.js (llm, embedding, apis, abstracts)
//   llm      - optional chat client (default: from settings.llm when enabled; null: off)
//   embedder - optional embedder (default: from settings.embedding)
//
// A survey runs ten stages (SURVEY_STAGES plus "done"): seeds -> queries ->
// expand -> dedupe -> rank -> enrich -> read -> themes -> write. Each finished
// stage is saved with the survey, so that a cancelled or failed survey can go
// on from the last finished stage (run() picks the first unfinished one).
// Without an LLM, stages 1–6 run as usual, nothing is read, and the report is
// a candidate list grouped by embedding cluster (labels from top terms).
// Problems of one source, paper or section are non-fatal (survey.errors);
// cancellation always propagates.

import { mapWithConcurrency } from "../core/http.js";
import { collapseWhitespace, contentForEmbedding, normalizeDoi, normalizeTitle } from "../core/text.js";
import { CancelledError } from "../engine.js";
import { createEmbedder } from "../embedding/providers.js";
import { createLlmClient } from "../llm/llm.js";
import { enrichMissingAbstracts } from "../sources/abstracts.js";
import { PaperCollector, VIA_REC, VIA_SEARCH, VIA_SEED, dedupePool, sameTitle, selectPool, viaCites, viaRef, yearsAgree } from "./pool.js";
import { SURVEY_DEPTHS, SURVEY_STAGES, STAGE_KEYS } from "./presets.js";
import {
  MAX_EXCERPT_CHARS,
  UNVERIFIED_MARK,
  cleanQuery,
  extractionPrompt,
  fullTextExcerpt,
  overviewPrompt,
  parseExtraction,
  parseOverviewReply,
  parseQueryReply,
  parseSectionReply,
  parseThemeReply,
  queryPrompt,
  sectionPrompt,
  themePrompt,
  validateCitations,
} from "./prompts.js";
import { centroidOf, clusterThemes, scorePapers, selectReadSet, topTerms } from "./rank.js";
import { NO_LLM_NOTICE, referenceOf } from "./report.js";
import { SURVEY_VERSION, loadNoteCache, newSurveyId, noteCacheKey, readCachedNote, saveNoteCache, saveSurvey, writeCachedNote } from "./store.js";
import {
  crossrefSearch,
  openAlexCitingWorks,
  openAlexSearch,
  openAlexSeed,
  openAlexWorksByIds,
  s2CitingPapers,
  s2RecommendedPapers,
  s2ReferencePapers,
  s2SearchPapers,
  s2SeedIds,
  s2Tldrs,
} from "./sources.js";

/** `code` of the warning recorded when OpenAlex is used without a key. */
export const SURVEY_OPENALEX_KEYLESS = "openalex-keyless";
// Requests of one source failing this many times in a row skip the rest of that source
const MAX_SOURCE_FAILURES = 3;
// Reading notes: LLM calls at once, and failures before giving up when none succeeded
const READ_CONCURRENCY = 2;
const READ_GIVE_UP = 3;
// Full text read from the library for an excerpt (the excerpt itself is MAX_EXCERPT_CHARS):
// all of it (Zotero indexes 500,000 characters by default), so that the excerpt finds the
// conclusions and the reference list of a long paper
const FULL_TEXT_CHARS = 500000;
// Papers of a theme shown to the section prompt
const MAX_SECTION_PAPERS = 25;
const MAX_READING = 12;

const STAGE_LABELS = Object.fromEntries(SURVEY_STAGES.map((s) => [s.key, s.label]));
const hasText = (v) => typeof v === "string" && v.trim() !== "";
const message = (e) => String(e?.message || e || "未知错误");
const isCancel = (e) => !!e?.cancelled || e instanceof CancelledError;
// Mostly Latin text: a query the English databases can match
const LATIN = /^[\p{Script=Latin}\p{N}\s\p{P}\p{S}]+$/u;

/**
 * The search result that is the seed: the same normalised title first, else
 * the same title up to spelling variants (pool.js sameTitle: not "…: a
 * review", "Corrigendum to …" or another part), from about the same year.
 */
export function titleMatch(seed, results) {
  const title = normalizeTitle(seed.title);
  const list = (results || []).filter((p) => p && yearsAgree(seed.year, p.year));
  return list.find((p) => normalizeTitle(p.title) === title) || list.find((p) => sameTitle(seed.title, p.title)) || null;
}

/**
 * A new survey record for a validated input (see normalizeSurveyInput).
 * @returns {object} the survey: {version, id, input, status, stage, createdAt, updatedAt,
 *   completed, stages, seeds, queries, pool, themes, references, report, errors, error, counts, llm, model}
 */
export function createSurvey(input, { now = Date.now(), id = newSurveyId(now), llm = false, model = null } = {}) {
  return {
    version: SURVEY_VERSION,
    id,
    input,
    status: "running",
    stage: null,
    createdAt: now,
    updatedAt: now,
    finishedAt: null,
    completed: [],
    stages: {},
    seeds: [],
    queries: [],
    terms: [],
    pool: [],
    readIds: [],
    themes: [],
    references: [],
    report: null,
    errors: [],
    error: null,
    counts: { seeds: input.seedKeys.length, candidates: 0, read: 0, references: 0 },
    llm,
    model,
  };
}

/** Survey counts for the index and the running view. */
export function surveyCounts(survey) {
  return {
    seeds: survey.seeds?.length || survey.input?.seedKeys?.length || 0,
    candidates: survey.pool?.length || 0,
    read: (survey.pool || []).filter((p) => p.read).length,
    references: survey.references?.length || 0,
  };
}

export class SurveyEngine {
  constructor({ http, storage, library, settings, llm, embedder = null, log = () => {}, now = () => Date.now(), retry = {} }) {
    this.http = http;
    this.storage = storage;
    this.library = library;
    this.settings = settings;
    this.log = log;
    this.now = now;
    this.retry = retry;
    this.embedderOverride = embedder;
    const llmCfg = settings.llm || {};
    if (llm === undefined) this.llm = llmCfg.enabled && llmCfg.apiKey ? createLlmClient(llmCfg, http, { retry }) : null;
    else this.llm = llm;
    this.model = this.llm ? llmCfg.model || null : null;
  }

  /** Whether the AI stages (read, theme names, writing) run. */
  get llmOn() {
    return !!this.llm;
  }

  /** The survey's data sources with the keys they need: Semantic Scholar only with a key. */
  sourcesOf(survey) {
    const s = survey.input.sources || {};
    const apis = this.settings.apis || {};
    return { openalex: s.openalex !== false, s2: s.s2 !== false && !!apis.semanticScholarKey, crossref: s.crossref !== false };
  }

  get openAlexOptions() {
    const apis = this.settings.apis || {};
    return { apiKey: apis.openalexKey || "", mailto: apis.mailto || "", retry: this.retry };
  }

  get s2Options() {
    return { key: this.settings.apis?.semanticScholarKey || "", retry: this.retry };
  }

  embedder() {
    this._embedder ||= this.embedderOverride || createEmbedder(this.settings.embedding, this.http, { retry: this.retry });
    return this._embedder;
  }

  async save(survey) {
    survey.updatedAt = this.now();
    survey.counts = surveyCounts(survey);
    await saveSurvey(this.storage, survey);
  }

  /**
   * Run the survey from its first unfinished stage (or from `from`, which
   * reruns that stage and the ones after it). Resolves with the survey (status
   * "done"); a cancellation or failure is saved (status "cancelled" /
   * "failed", resumable) and rethrown.
   * @param {object} survey - see createSurvey (it is updated in place and saved after every stage)
   * @param {object} [options] - {onProgress(event), isCancelled(), from}
   */
  async run(survey, { onProgress = () => {}, isCancelled = () => false, from = null } = {}) {
    const checkpoint = () => {
      if (isCancelled() || this.http?.signal?.cancelled) throw new CancelledError();
    };
    let current = null;
    const progress = (stage, text, extra = {}) =>
      onProgress({
        stage,
        index: STAGE_KEYS.indexOf(stage),
        label: STAGE_LABELS[stage] || "",
        message: text,
        current: null,
        total: null,
        counts: surveyCounts(survey),
        ...extra,
      });
    const ctx = { survey, progress, checkpoint, vectors: new Map(), fullTexts: new Map(), libraryItems: null, cache: null };

    if (from) {
      const at = STAGE_KEYS.indexOf(from);
      if (at < 0) throw new Error(`Unknown survey stage: ${from}`);
      survey.completed = survey.completed.filter((key) => STAGE_KEYS.indexOf(key) < at);
    }
    let start = STAGE_KEYS.findIndex((key) => !survey.completed.includes(key));
    if (!from && start >= 0 && start <= STAGE_KEYS.indexOf("expand")) {
      // Seeds and queries that finished with failed lookups (an outage) are redone
      // before the expansion that needs them; they are cheap
      const redo = STAGE_KEYS.findIndex((key, i) => i < start && survey.stages[key]?.retry);
      if (redo >= 0) {
        start = redo;
        survey.completed = survey.completed.filter((key) => STAGE_KEYS.indexOf(key) < redo);
      }
    }
    survey.status = "running";
    survey.error = null;
    survey.finishedAt = null;
    survey.llm = this.llmOn;
    if (this.llmOn) survey.model = this.model;
    await this.save(survey);
    try {
      for (let i = start < 0 ? STAGE_KEYS.length : start; i < STAGE_KEYS.length; i++) {
        current = STAGE_KEYS[i];
        survey.stage = current;
        // A rerun stage reports its problems afresh
        survey.errors = survey.errors.filter((e) => e.stage !== current);
        checkpoint();
        progress(current, `${STAGE_LABELS[current]}…`);
        const summary = await this[`stage_${current}`](ctx);
        checkpoint();
        survey.stages[current] = { ...(summary || {}), at: this.now() };
        if (!survey.completed.includes(current)) survey.completed.push(current);
        await this.save(survey);
      }
      survey.status = "done";
      survey.stage = "done";
      survey.finishedAt = this.now();
      await this.save(survey);
      onProgress({ stage: "done", index: STAGE_KEYS.length, label: "完成", message: "调研完成", current: null, total: null, counts: surveyCounts(survey) });
      return survey;
    } catch (e) {
      const cancelled = isCancel(e) || isCancelled() || !!this.http?.signal?.cancelled;
      survey.status = cancelled ? "cancelled" : "failed";
      survey.error = cancelled ? null : { stage: current, message: message(e) };
      if (!cancelled) this.log(`Survey ${survey.id} failed in ${current}: ${message(e)}`);
      await this.save(survey).catch((err) => this.log(`Survey save failed: ${message(err)}`));
      throw cancelled && !isCancel(e) ? new CancelledError() : e;
    } finally {
      if (ctx.cache) await saveNoteCache(this.storage, ctx.cache, this.log);
    }
  }

  /** A non-fatal problem of the current stage, shown with the survey. */
  warn(survey, entry) {
    const e = { stage: survey.stage, ...entry };
    if (!survey.errors.some((x) => x.stage === e.stage && x.source === e.source && x.message === e.message)) survey.errors.push(e);
    this.log(`Survey ${survey.stage} ${e.source || ""}: ${e.message}`);
  }

  async libraryItems(ctx) {
    ctx.libraryItems ||= await this.library.getItems();
    return ctx.libraryItems;
  }

  /**
   * Full text of a library item as {length, excerpt} (fullTextExcerpt; only
   * the excerpt is kept for the run), {length: 0, excerpt: ""} when it has none.
   */
  async fullTextOf(ctx, key) {
    if (!key || typeof this.library.getItemFullText !== "function") return { length: 0, excerpt: "" };
    if (!ctx.fullTexts.has(key)) {
      let text = "";
      try {
        text = (await this.library.getItemFullText(key, { maxChars: FULL_TEXT_CHARS })) || "";
      } catch (e) {
        this.log(`Full text of ${key} unavailable: ${message(e)}`);
      }
      text = typeof text === "string" ? text : "";
      ctx.fullTexts.set(key, { length: text.length, excerpt: text ? fullTextExcerpt(text, MAX_EXCERPT_CHARS) : "" });
    }
    return ctx.fullTexts.get(key);
  }

  // --------------------------------------------------------- 1. seeds

  async stage_seeds(ctx) {
    const { survey, checkpoint } = ctx;
    const keys = survey.input.seedKeys;
    const all = await this.libraryItems(ctx);
    checkpoint();
    let items;
    if (keys.length && typeof this.library.getItemsByKeys === "function") items = await this.library.getItemsByKeys(keys);
    else {
      const byKey = new Map(all.map((i) => [i.key, i]));
      items = keys.map((k) => byKey.get(k)).filter(Boolean);
    }
    const found = new Map((items || []).filter((i) => i && i.key).map((i) => [i.key, i]));
    const missing = keys.filter((k) => !found.has(k));
    if (missing.length) this.warn(survey, { source: "Zotero", message: `找不到 ${missing.length} 篇种子文献（可能已删除）` });
    survey.seeds = keys
      .filter((k) => found.has(k))
      .map((key, i) => {
        const item = found.get(key);
        return {
          n: i + 1,
          key,
          title: collapseWhitespace(item.title || ""),
          abstract: item.abstract || "",
          doi: normalizeDoi(item.doi) || null,
          year: Number.isInteger(item.year) ? item.year : null,
          creators: Array.isArray(item.creators) ? item.creators.slice(0, 12) : [],
          tags: Array.isArray(item.tags) ? item.tags.slice(0, 20) : [],
          venue: item.venue || null,
          itemType: item.itemType || null,
          openalexId: null,
          s2Id: null,
          references: [],
          fullText: 0,
        };
      })
      .filter((s) => s.title);
    const seeds = survey.seeds;
    const sources = this.sourcesOf(survey);

    // Lookups that failed (not "not found"): the stage is redone when the survey goes on
    let lookupFailed = false;
    if (sources.openalex && seeds.length) {
      let failures = 0;
      for (const [i, seed] of seeds.entries()) {
        checkpoint();
        ctx.progress("seeds", `在 OpenAlex 中查找种子文献 ${i + 1}/${seeds.length}…`, { current: i, total: seeds.length });
        try {
          let hit = seed.doi ? await openAlexSeed(this.http, { doi: seed.doi }, this.openAlexOptions) : null;
          if (!hit) {
            const match = await this.openAlexByTitle(seed);
            if (match) hit = await openAlexSeed(this.http, { openalexId: match }, this.openAlexOptions);
          }
          if (hit) {
            seed.openalexId = hit.openalexId;
            seed.references = hit.references;
            if (!seed.year && hit.paper?.year) seed.year = hit.paper.year;
            if (!seed.doi && hit.paper?.doi) seed.doi = hit.paper.doi;
            seed.citations = hit.paper?.citations ?? null;
          }
        } catch (e) {
          if (isCancel(e)) throw e;
          lookupFailed = true;
          this.warn(survey, { source: "OpenAlex", message: `种子文献查找失败：${message(e)}` });
          if (++failures >= MAX_SOURCE_FAILURES) break;
        }
      }
    }
    if (sources.s2 && seeds.length) {
      checkpoint();
      try {
        const ids = await s2SeedIds(this.http, seeds.map((s) => s.doi), this.s2Options);
        seeds.forEach((s, i) => (s.s2Id = ids[i]));
        for (const seed of seeds.filter((s) => !s.s2Id)) {
          checkpoint();
          seed.s2Id = await this.s2ByTitle(seed);
        }
      } catch (e) {
        if (isCancel(e)) throw e;
        lookupFailed = true;
        this.warn(survey, { source: "Semantic Scholar", message: `种子文献查找失败：${message(e)}` });
      }
    }
    for (const seed of seeds) {
      checkpoint();
      seed.fullText = (await this.fullTextOf(ctx, seed.key)).length;
    }
    return {
      seeds: seeds.length,
      missing: missing.length,
      openalex: seeds.filter((s) => s.openalexId).length,
      s2: seeds.filter((s) => s.s2Id).length,
      fullText: seeds.filter((s) => s.fullText > 0).length,
      retry: lookupFailed,
    };
  }

  /** OpenAlex id of a seed not found by DOI, by its title (see titleMatch); null when none. */
  async openAlexByTitle(seed) {
    const results = await openAlexSearch(this.http, seed.title, { limit: 5, ...this.openAlexOptions });
    return titleMatch(seed, results)?.openalexId || null;
  }

  async s2ByTitle(seed) {
    const results = await s2SearchPapers(this.http, seed.title, { limit: 5, ...this.s2Options });
    return titleMatch(seed, results)?.s2Id || null;
  }

  // ------------------------------------------------------- 2. queries

  async stage_queries(ctx) {
    const { survey } = ctx;
    const input = survey.input;
    const preset = SURVEY_DEPTHS[input.depth] || SURVEY_DEPTHS.standard;
    const distinct = (list) => {
      const seen = new Set();
      return list.filter((q) => q && !seen.has(q.toLowerCase()) && seen.add(q.toLowerCase()));
    };
    // From the input alone: the topic, the keywords together, and multi-word keywords on their own
    const words = (k) => (/[㐀-鿿]/.test(k) ? (Array.from(k).length >= 4 ? 2 : 1) : k.split(/\s+/).length);
    const inputQueries = distinct(
      [input.topic, input.keywords.length > 1 && input.keywords.length <= 6 ? input.keywords.join(" ") : "", ...input.keywords.filter((k) => words(k) >= 2)].map(cleanQuery),
    );
    let queries = inputQueries.slice(0, preset.maxQueries);
    let terms = [];
    let source = "input";
    let retry = false;
    if (this.llmOn) {
      try {
        const reply = await this.llm.chat(queryPrompt(input, survey.seeds), { maxTokens: 800, temperature: 0.2 });
        const parsed = parseQueryReply(reply, { maxQueries: 8 });
        queries = distinct([...parsed.queries, ...inputQueries.filter((q) => LATIN.test(q))]).slice(0, preset.maxQueries);
        terms = parsed.terms;
        source = "llm";
      } catch (e) {
        if (isCancel(e)) throw e;
        retry = true;
        this.warn(survey, { source: "大模型", message: `检索词生成失败，改用主题和关键词：${message(e)}` });
      }
    }
    survey.queries = queries;
    survey.terms = terms;
    return { queries: queries.length, source, retry };
  }

  // -------------------------------------------------------- 3. expand

  async stage_expand(ctx) {
    const { survey, checkpoint } = ctx;
    const input = survey.input;
    const preset = SURVEY_DEPTHS[input.depth] || SURVEY_DEPTHS.standard;
    const sources = this.sourcesOf(survey);
    const seeds = survey.seeds;
    const collector = new PaperCollector();
    const yearFrom = input.yearFrom || null;
    const apis = this.settings.apis || {};

    if (sources.openalex && !apis.openalexKey) {
      this.warn(survey, { source: "OpenAlex", code: SURVEY_OPENALEX_KEYLESS, message: "未配置 OpenAlex API Key：每天的免费额度很小，结果可能不完整，请在设置 → 学术数据接口中填写" });
    }
    if (input.sources?.s2 !== false && !apis.semanticScholarKey) {
      this.warn(survey, { source: "Semantic Scholar", message: "未配置 Semantic Scholar API Key，已跳过 Semantic Scholar" });
    }

    for (const seed of seeds) {
      collector.add(
        {
          id: seed.doi ? `doi:${seed.doi}` : `zotero:${seed.key}`,
          title: seed.title,
          abstract: seed.abstract || null,
          tldr: null,
          authors: seed.creators.map((name) => ({ given: "", family: name, name })),
          authorCount: seed.creators.length,
          year: seed.year,
          published: null,
          venue: seed.venue,
          issns: [],
          doi: seed.doi,
          url: seed.doi ? `https://doi.org/${seed.doi}` : null,
          type: null,
          citations: Number.isFinite(seed.citations) ? seed.citations : 0,
          openalexId: seed.openalexId,
          s2Id: seed.s2Id,
          sources: ["zotero"],
          seed: seed.n,
          inLibrary: true,
          itemKey: seed.key,
        },
        VIA_SEED,
      );
    }

    // Requests per service, run one after another (the limiter paces each service); the services in parallel
    const chains = { OpenAlex: [], "Semantic Scholar": [], Crossref: [] };
    const linked = (label, run, via) => ({ label, run, via, kind: "linked" });
    const search = (label, run) => ({ label, run, via: VIA_SEARCH, kind: "search" });
    const oa = this.openAlexOptions;
    const s2 = this.s2Options;
    if (sources.openalex) {
      for (const seed of seeds.filter((s) => s.openalexId)) {
        if (seed.references.length) {
          chains.OpenAlex.push(linked(`种子 ${seed.n} 的参考文献`, () => openAlexWorksByIds(this.http, seed.references.slice(0, preset.perSeed * 2), oa), viaRef(seed.n)));
        }
        chains.OpenAlex.push(linked(`引用种子 ${seed.n} 的文献`, () => openAlexCitingWorks(this.http, seed.openalexId, { limit: preset.perSeed, yearFrom, ...oa }), viaCites(seed.n)));
      }
      survey.queries.forEach((q, i) => chains.OpenAlex.push(search(`检索 ${i + 1}/${survey.queries.length}`, () => openAlexSearch(this.http, q, { limit: preset.perQuery, yearFrom, ...oa }))));
    }
    if (sources.s2) {
      const withS2 = seeds.filter((s) => s.s2Id);
      if (withS2.length) chains["Semantic Scholar"].push(linked("推荐文献", () => s2RecommendedPapers(this.http, withS2.map((s) => s.s2Id), { limit: preset.recommendations, ...s2 }), VIA_REC));
      for (const seed of withS2) {
        chains["Semantic Scholar"].push(linked(`种子 ${seed.n} 的参考文献`, () => s2ReferencePapers(this.http, seed.s2Id, { limit: preset.perSeed, ...s2 }), viaRef(seed.n)));
        chains["Semantic Scholar"].push(linked(`引用种子 ${seed.n} 的文献`, () => s2CitingPapers(this.http, seed.s2Id, { limit: preset.perSeed, ...s2 }), viaCites(seed.n)));
      }
      survey.queries.forEach((q, i) => chains["Semantic Scholar"].push(search(`检索 ${i + 1}/${survey.queries.length}`, () => s2SearchPapers(this.http, q, { limit: Math.min(100, preset.perQuery), yearFrom, ...s2 }))));
    }
    if (sources.crossref) {
      survey.queries.forEach((q, i) =>
        chains.Crossref.push(search(`检索 ${i + 1}/${survey.queries.length}`, () => crossrefSearch(this.http, q, { limit: preset.crossrefPerQuery, yearFrom, mailto: apis.mailto || "", retry: this.retry }))),
      );
    }

    const total = Object.values(chains).reduce((n, list) => n + list.length, 0);
    let done = 0;
    const stats = { requests: total, failed: 0, skipped: 0, bySource: {}, byVia: { search: 0, ref: 0, cites: 0, rec: 0 } };
    const runChain = async (name, tasks) => {
      let failures = 0;
      for (const task of tasks) {
        checkpoint();
        if (failures >= MAX_SOURCE_FAILURES) {
          stats.skipped++;
          done++;
          continue;
        }
        let papers = [];
        try {
          papers = await task.run();
          failures = 0;
        } catch (e) {
          if (isCancel(e)) throw e;
          failures++;
          stats.failed++;
          // "检索 6/7 失败" but "推荐文献失败": a space only after a number
          const label = /\d$/.test(task.label) ? `${task.label} ` : task.label;
          this.warn(survey, { source: name, message: `${label}失败：${message(e)}${failures >= MAX_SOURCE_FAILURES ? `（连续失败 ${MAX_SOURCE_FAILURES} 次，跳过其余请求）` : ""}` });
        }
        checkpoint();
        papers.forEach((p, rank) => collector.add(p, task.via, task.kind === "search" ? { searchRank: rank } : {}));
        stats.bySource[name] = (stats.bySource[name] || 0) + papers.length;
        const viaKey = task.via === VIA_SEARCH ? "search" : task.via === VIA_REC ? "rec" : task.via.startsWith("ref:") ? "ref" : "cites";
        stats.byVia[viaKey] += papers.length;
        done++;
        ctx.progress("expand", `扩展候选：${name} ${task.label}`, { current: done, total, counts: { ...ctx.survey.counts, candidates: collector.papers.length } });
      }
    };
    const settled = await Promise.allSettled(Object.entries(chains).map(([name, tasks]) => runChain(name, tasks)));
    const failed = settled.find((r) => r.status === "rejected");
    if (failed) throw failed.reason;

    const found = collector.papers.length - seeds.length;
    if (found <= 0) {
      if (total && stats.failed + stats.skipped === total) throw new Error(`所有数据源都请求失败：${survey.errors.at(-1)?.message || ""}`);
      throw new Error(total ? "没有找到候选文献，请换用其他关键词或放宽起始年份" : "没有可用的检索词或种子文献");
    }
    // The start year holds for every candidate (references, citations and recommendations
    // are not filtered by the services); seeds and papers of unknown year stay
    const inRange = collector.papers.filter((p) => p.seed || !yearFrom || !Number.isInteger(p.year) || p.year >= yearFrom);
    stats.beforeYear = collector.papers.length - inRange.length;
    if (inRange.length <= seeds.length) throw new Error(`没有 ${yearFrom} 年以来的候选文献，请放宽起始年份`);
    survey.pool = selectPool(inRange, preset.pool);
    return { found: collector.papers.length, pool: survey.pool.length, ...stats };
  }

  // -------------------------------------------------------- 4. dedupe

  async stage_dedupe(ctx) {
    const { survey } = ctx;
    const before = survey.pool.length;
    const { papers, merged, inLibrary } = dedupePool(survey.pool, await this.libraryItems(ctx));
    survey.pool = papers;
    return { before, after: papers.length, merged, inLibrary };
  }

  // ---------------------------------------------------------- 5. rank

  /** Query text of the survey (embedded as the survey's topic). */
  queryText(survey) {
    const input = survey.input;
    return [input.topic, input.description, input.keywords.join("; ")].filter(Boolean).join("\n");
  }

  textOf(p) {
    return contentForEmbedding(p.title, p.abstract || p.tldr || "");
  }

  /** Vectors of `papers` (cached for the run); a local embedder is fitted on the whole pool. */
  async vectorsOf(ctx, papers, onProgress) {
    const embedder = this.embedder();
    const missing = papers.filter((p) => !ctx.vectors.has(p.id));
    if (missing.length) {
      if (embedder.requiresFit && !ctx.fitted) {
        embedder.fit([...ctx.survey.pool.map((p) => this.textOf(p)), this.queryText(ctx.survey)]);
        ctx.fitted = true;
      }
      const vectors = await embedder.embed(missing.map((p) => this.textOf(p)), { onProgress });
      missing.forEach((p, i) => ctx.vectors.set(p.id, vectors[i]));
    }
    return papers.map((p) => ctx.vectors.get(p.id));
  }

  async stage_rank(ctx) {
    const { survey, checkpoint } = ctx;
    const pool = survey.pool;
    await this.vectorsOf(ctx, pool, (done, total) => ctx.progress("rank", `候选文献向量化 ${done}/${total}`, { current: done, total }));
    checkpoint();
    const [topicVector] = await this.embedder().embed([this.queryText(survey)], { inputType: "query" });
    const seedVectors = pool.filter((p) => p.seed).map((p) => ctx.vectors.get(p.id));
    const centre = centroidOf([topicVector, ...seedVectors]);
    survey.pool = scorePapers(pool, (p) => ctx.vectors.get(p.id), centre, survey.seeds.length);
    return { ranked: survey.pool.length, embedder: this.embedder().signature || null };
  }

  // -------------------------------------------------------- 6. enrich

  async stage_enrich(ctx) {
    const { survey, checkpoint } = ctx;
    const preset = SURVEY_DEPTHS[survey.input.depth] || SURVEY_DEPTHS.standard;
    const top = survey.pool.slice(0, Math.ceil(preset.deepRead * 1.2));
    const sources = this.sourcesOf(survey);
    const apis = this.settings.apis || {};
    const base = this.settings.abstracts || { crossref: true, openalex: true, elsevierKey: "", springerKey: "" };
    const cfg = { ...base, openalex: base.openalex && sources.openalex, openalexKey: apis.openalexKey || "", s2Key: sources.s2 ? apis.semanticScholarKey || "" : "" };
    const missingBefore = top.filter((p) => !hasText(p.abstract)).length;
    let filled = 0;
    if (missingBefore && (cfg.crossref || cfg.openalex || cfg.elsevierKey || cfg.springerKey || cfg.s2Key)) {
      const text = `补全缺失摘要（${missingBefore} 篇）…`;
      ctx.progress("enrich", text);
      try {
        const result = await enrichMissingAbstracts(this.http, top, {
          ...cfg,
          mailto: apis.mailto || "",
          cache: this.storage,
          now: this.now(),
          retry: this.retry,
          log: this.log,
          checkpoint,
          onProgress: (done, total) => ctx.progress("enrich", text, { current: done, total }),
        });
        filled = result.filled;
        for (const e of result.errors) this.warn(survey, { source: e.source, message: e.message });
      } catch (e) {
        if (isCancel(e)) throw e;
        this.warn(survey, { source: "摘要补全", message: message(e) });
      }
    }
    let tldrs = 0;
    const needTldr = top.filter((p) => !p.tldr && (p.doi || p.s2Id));
    if (sources.s2 && needTldr.length) {
      checkpoint();
      ctx.progress("enrich", `获取 Semantic Scholar TLDR（${needTldr.length} 篇）…`);
      try {
        const found = await s2Tldrs(this.http, needTldr, this.s2Options);
        needTldr.forEach((p, i) => {
          const hit = found[i];
          if (!hit) return;
          if (hit.tldr) {
            p.tldr = hit.tldr;
            tldrs++;
          }
          if (!p.s2Id && hit.s2Id) p.s2Id = hit.s2Id;
          if (!hasText(p.abstract) && hasText(hit.abstract) && hit.abstract.length >= 80) {
            p.abstract = hit.abstract;
            filled++;
          }
        });
      } catch (e) {
        if (isCancel(e)) throw e;
        this.warn(survey, { source: "Semantic Scholar", message: `TLDR 获取失败：${message(e)}` });
      }
    }
    return { targets: top.length, missing: missingBefore, filled, tldr: tldrs, stillMissing: top.filter((p) => !hasText(p.abstract)).length };
  }

  // ---------------------------------------------------------- 7. read

  async stage_read(ctx) {
    const { survey, checkpoint } = ctx;
    if (!this.llmOn) {
      survey.readIds = [];
      return { skipped: true, reason: "llm-off" };
    }
    const preset = SURVEY_DEPTHS[survey.input.depth] || SURVEY_DEPTHS.standard;
    // Library papers without an abstract may still have a full text
    const head = survey.pool.slice(0, Math.ceil(preset.deepRead * 1.5));
    for (const p of head) {
      if (p.itemKey && !hasText(p.abstract)) await this.fullTextOf(ctx, p.itemKey);
    }
    const readSet = selectReadSet(survey.pool, preset.deepRead, (p) => !!(p.itemKey && ctx.fullTexts.get(p.itemKey)?.excerpt));
    survey.readIds = readSet.map((p) => p.id);
    ctx.cache ||= await loadNoteCache(this.storage);
    const cache = ctx.cache;
    const topic = survey.input.topic;
    const stats = { selected: readSet.length, read: 0, cached: 0, failed: 0 };
    let firstError = null;
    let done = 0;
    await mapWithConcurrency(readSet, READ_CONCURRENCY, async (p) => {
      checkpoint();
      if (stats.read === 0 && stats.cached === 0 && stats.failed >= READ_GIVE_UP) return;
      const key = noteCacheKey(p, this.model);
      try {
        const cached = readCachedNote(cache, key, topic);
        if (cached) {
          p.read = cached.note;
          if (!cached.relevanceKnown) {
            // Notes from another survey: the relevance to this topic comes from the similarity
            p.read.relevance = 1 + Math.round(4 * (p.scores?.sim ?? 0));
            p.read.relevanceEstimated = true;
          }
          stats.cached++;
        } else {
          const excerpt = p.itemKey ? (await this.fullTextOf(ctx, p.itemKey)).excerpt : "";
          const reply = await this.llm.chat(extractionPrompt(survey.input, p, excerpt), { maxTokens: 700, temperature: 0.2 });
          // Kept (and cached) even when the survey was cancelled meanwhile: the call is paid for
          p.read = parseExtraction(reply);
          if (excerpt) p.read.fullText = true;
          writeCachedNote(cache, key, topic, p.read, this.now());
          stats.read++;
          checkpoint();
          if (stats.read % 10 === 0) await saveNoteCache(this.storage, cache, this.log);
        }
        delete p.readError;
      } catch (e) {
        if (isCancel(e)) throw e;
        p.readError = message(e);
        stats.failed++;
        firstError ||= message(e);
      }
      done++;
      ctx.progress("read", `精读 ${done}/${readSet.length}：${p.title}`, { current: done, total: readSet.length });
    });
    if (stats.failed) this.warn(survey, { source: "大模型", message: `${stats.failed} 篇文献精读失败：${firstError}` });
    if (readSet.length && !stats.read && !stats.cached) throw new Error(`精读失败：${firstError || "没有可精读的文献"}`);
    if (!readSet.length) throw new Error("候选文献都没有摘要，无法精读");
    return stats;
  }

  // -------------------------------------------------------- 8. themes

  async stage_themes(ctx) {
    const { survey, checkpoint } = ctx;
    const preset = SURVEY_DEPTHS[survey.input.depth] || SURVEY_DEPTHS.standard;
    // With the LLM: the papers read; without: the best ranked papers
    const readIds = new Set(survey.readIds || []);
    const papers = this.llmOn ? survey.pool.filter((p) => readIds.has(p.id) && p.read) : survey.pool.slice(0, preset.deepRead);
    for (const p of survey.pool) {
      delete p.theme;
      delete p.n;
    }
    const vectors = await this.vectorsOf(ctx, papers, (done, total) => ctx.progress("themes", `向量化 ${done}/${total}`, { current: done, total }));
    checkpoint();
    const { k, assignments, silhouette } = await clusterThemes(vectors);
    const groups = Array.from({ length: k }, () => []);
    papers.forEach((p, i) => groups[assignments[i]].push(p));
    const score = (p) => p.scores?.total ?? 0;
    for (const g of groups) g.sort((a, b) => score(b) - score(a));
    groups.sort((a, b) => b.length - a.length || score(b[0]) - score(a[0]));

    const allTexts = papers.map((p) => `${p.title} ${p.abstract || ""}`);
    const termsOf = (g) => topTerms(g.map((p) => `${p.title} ${p.abstract || ""}`), allTexts, 3);
    let names = null;
    let labelled = "terms";
    if (this.llmOn && groups.length) {
      try {
        const reply = await this.llm.chat(themePrompt(survey.input, groups.map((g) => ({ papers: g }))), { maxTokens: 1000, temperature: 0.2 });
        names = parseThemeReply(reply, groups.length);
        labelled = "llm";
      } catch (e) {
        if (isCancel(e)) throw e;
        this.warn(survey, { source: "大模型", message: `主题命名失败，改用关键词：${message(e)}` });
      }
    }
    let n = 0;
    survey.themes = groups.map((g, i) => {
      const terms = termsOf(g);
      g.forEach((p) => {
        p.theme = i;
        p.n = ++n;
      });
      return {
        label: names?.[i]?.label || (terms.length ? terms.join(" · ") : `主题 ${i + 1}`),
        summary: names?.[i]?.summary || "",
        terms,
        paperIds: g.map((p) => p.id),
      };
    });
    survey.references = groups.flatMap((g, i) => g.map((p) => referenceOf(p, p.n, i)));
    return { themes: survey.themes.length, papers: papers.length, labelled, silhouette };
  }

  // --------------------------------------------------------- 9. write

  async stage_write(ctx) {
    const { survey, checkpoint } = ctx;
    const byId = new Map(survey.pool.map((p) => [p.id, p]));
    const themePapers = survey.themes.map((t) => t.paperIds.map((id) => byId.get(id)).filter(Boolean));
    if (!this.llmOn) {
      survey.report = {
        kind: "list",
        title: survey.input.topic,
        notice: NO_LLM_NOTICE,
        groups: survey.themes.map((t, i) => ({ label: t.label, terms: t.terms, refs: themePapers[i].map((p) => p.n) })),
        generatedAt: this.now(),
        model: null,
      };
      return { kind: "list", groups: survey.themes.length };
    }

    const total = survey.themes.length + 1;
    let ok = 0;
    let failed = 0;
    let firstError = null;
    const sections = [];
    for (const [i, theme] of survey.themes.entries()) {
      checkpoint();
      ctx.progress("write", `撰写「${theme.label}」（${i + 1}/${total}）…`, { current: i, total });
      const papers = themePapers[i].slice(0, MAX_SECTION_PAPERS);
      const allowed = new Set(papers.map((p) => p.n));
      let section;
      try {
        const reply = await this.llm.chat(sectionPrompt(survey.input, theme, papers), { maxTokens: 3000, temperature: 0.3 });
        section = parseSectionReply(reply, theme.label);
        ok++;
      } catch (e) {
        if (isCancel(e)) throw e;
        failed++;
        firstError ||= message(e);
        this.warn(survey, { source: "大模型", message: `「${theme.label}」一节生成失败，改为列出各文献的主要发现：${message(e)}` });
        section = {
          heading: theme.label,
          paragraphs: [papers.map((p) => `${p.title}：${p.read?.findings || p.read?.contribution || "（无精读笔记）"} [${p.n}]`).join("；")],
          fallback: true,
        };
      }
      const paragraphs = section.paragraphs.map((t) => validateCitations(t, allowed).text).filter(Boolean);
      const cited = paragraphs.some((t) => /\[\d+(?:, \d+)*\]/.test(t));
      sections.push({
        heading: cited ? section.heading : `${section.heading}${UNVERIFIED_MARK}`,
        theme: i,
        paragraphs,
        unverified: !cited,
        ...(section.fallback ? { fallback: true } : {}),
      });
    }

    checkpoint();
    ctx.progress("write", `撰写总述（${total}/${total}）…`, { current: total - 1, total });
    const references = survey.references.map((r) => ({ ...r, read: byId.get(r.id)?.read || null }));
    const allowed = new Set(survey.references.map((r) => r.n));
    let overview = { title: "", summary: "", debates: [], gaps: [], future: [], reading: [] };
    try {
      const reply = await this.llm.chat(overviewPrompt(survey.input, sections, references), { maxTokens: 3000, temperature: 0.3 });
      overview = parseOverviewReply(reply);
      ok++;
    } catch (e) {
      if (isCancel(e)) throw e;
      failed++;
      firstError ||= message(e);
      this.warn(survey, { source: "大模型", message: `总述生成失败：${message(e)}` });
    }
    if (!ok) throw new Error(`综述生成失败：${firstError}`);
    const clean = (t) => validateCitations(t, allowed).text;
    const list = (items) => items.map(clean).filter(Boolean);
    const reading = [...new Set(overview.reading.filter((n) => allowed.has(n)))].slice(0, MAX_READING);
    survey.report = {
      kind: "review",
      title: overview.title || survey.input.topic,
      summary: clean(overview.summary),
      sections,
      debates: list(overview.debates),
      gaps: list(overview.gaps),
      future: list(overview.future),
      reading,
      generatedAt: this.now(),
      model: this.model,
      partial: failed > 0,
    };
    return { kind: "review", sections: sections.length, failed, unverified: sections.filter((s) => s.unverified).length };
  }
}

/** Whether a survey can go on (resume) from where it stopped. */
export function canResume(survey) {
  return !!survey && survey.status !== "done" && survey.status !== "running";
}

/** The stage `regenerateReport` reruns: "write", or "read" when nothing was read yet (no LLM at the time). */
export function regenerateFrom(survey) {
  if (survey.stages?.read?.skipped || !(survey.pool || []).some((p) => p.read)) return "read";
  return "write";
}
