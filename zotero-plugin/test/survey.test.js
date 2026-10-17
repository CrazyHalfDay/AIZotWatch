// Topic survey (src/survey/): stages with mocked OpenAlex, Semantic Scholar,
// Crossref, embeddings and LLM; the no-LLM path; cancellation and resume;
// failure isolation; citation validation; pool merging and library matching;
// ranking; persistence; Markdown and note rendering.

import assert from "node:assert/strict";
import test from "node:test";

import { CancelledError } from "../src/engine.js";
import { createMemoryStorage } from "../src/store.js";
import { SurveyEngine, canResume, createSurvey, regenerateFrom, surveyCounts } from "../src/survey/engine.js";
import { LINKED_SHARE, PaperCollector, dedupePool, linkedSeeds, selectPool } from "../src/survey/pool.js";
import { SURVEY_DEPTHS, SURVEY_STAGES, normalizeSurveyInput, parseKeywords, surveyEstimate } from "../src/survey/presets.js";
import {
  UNVERIFIED_MARK,
  cleanQuery,
  clip,
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
  validateCitations,
} from "../src/survey/prompts.js";
import { centroidOf, clusterThemes, scorePapers, selectReadSet, topTerms } from "../src/survey/rank.js";
import { escapeHtml, markdownFileName, referenceText, reportCitations, surveyMarkdown, surveyNoteHtml } from "../src/survey/report.js";
import { mapCrossrefPaper, mapOpenAlexPaper, openAlexWorksByIds, resetOpenAlexIdFilter } from "../src/survey/sources.js";
import {
  SURVEY_CACHE_FILE,
  SURVEY_INDEX_FILE,
  SURVEY_LIMIT,
  deleteSurvey,
  isSurveyId,
  loadNoteCache,
  loadSurvey,
  newSurveyId,
  noteCacheKey,
  readCachedNote,
  readIndex,
  saveSurvey,
  writeCachedNote,
} from "../src/survey/store.js";
import { mockHttp, noSleep } from "./helpers.js";
import { CROSSREF_ITEM, FULL_TEXT, LLM_URL, SURVEY_INPUT, WORKS, createLibrary, libraryItems, llmHandler, surveyHttp, surveySettings } from "./survey-fixtures.js";

const NOW = Date.UTC(2026, 9, 17, 8);

function setup({ prefs = {}, overrides = {}, hooks = {}, llm, input = SURVEY_INPUT, library = createLibrary(), retry = noSleep } = {}) {
  const stats = {};
  const counter = {};
  const http = surveyHttp(counter, { llm: llmHandler(stats, hooks), ...overrides });
  const storage = createMemoryStorage();
  const settings = surveySettings(prefs);
  const logs = [];
  let clock = NOW;
  const engine = (opts = {}) =>
    new SurveyEngine({ http, storage, library, settings, llm, retry, log: (m) => logs.push(m), now: () => (clock += 1000), ...opts });
  const survey = createSurvey(normalizeSurveyInput(input, { now: NOW }), { now: NOW, llm: llm !== null });
  return { stats, counter, http, storage, settings, engine, survey, library, logs };
}

const callsTo = (http, prefix) => http.calls.filter((c) => c.url.startsWith(prefix));
const byTitle = (survey, title) => survey.pool.find((p) => p.title === title);

// ------------------------------------------------------------------ full run

test("a survey runs every stage: seeds, expansion, dedupe, ranking, reading, themes and a cited review", async () => {
  const { stats, http, storage, engine, survey, library } = setup();
  const events = [];
  const result = await engine().run(survey, { onProgress: (p) => events.push(p) });
  assert.equal(result, survey);
  assert.equal(survey.status, "done");
  assert.equal(survey.stage, "done");
  assert.deepEqual(survey.completed, SURVEY_STAGES.map((s) => s.key));
  assert.deepEqual(survey.errors, []);
  assert.equal(survey.error, null);
  assert.ok(survey.finishedAt > NOW);

  // Progress: stages in order (with their index and label), determinate counters, then done
  const order = [...new Set(events.map((e) => e.stage))];
  assert.deepEqual(order, [...SURVEY_STAGES.map((s) => s.key), "done"]);
  assert.deepEqual(Object.keys(events[0]).sort(), ["counts", "current", "index", "label", "message", "stage", "total"]);
  const reads = events.filter((e) => e.stage === "read" && e.total);
  assert.equal(reads.at(-1).current, reads.at(-1).total);
  assert.equal(events.find((e) => e.stage === "expand").label, "扩展");
  assert.equal(events.find((e) => e.stage === "themes").index, 7);
  assert.ok(events.filter((e) => e.stage === "expand" && e.total).every((e) => Number.isFinite(e.counts.candidates)));

  // 1. Seeds: OpenAlex by DOI (with references), Semantic Scholar by DOI or by title; full text read
  const [seed1, seed2] = survey.seeds;
  assert.equal(seed1.openalexId, "W100");
  assert.deepEqual(seed1.references, ["W201", "W202", "W203"]);
  assert.equal(seed1.s2Id, "s2seed1");
  assert.equal(seed2.openalexId, null, "OpenAlex does not know seed 2 (404, no title match)");
  assert.equal(seed2.s2Id, "s2seed2", "found on Semantic Scholar by its title");
  assert.ok(seed1.fullText > 1000);
  assert.equal(seed2.fullText, 0);
  assert.deepEqual(seed1.tags, ["Copper isotopes", "Porphyry"]);
  assert.ok(callsTo(http, "https://api.openalex.org/works/doi:10.1000/seed1")[0].url.includes("api_key=oa-key"), "OpenAlex requests carry the key");
  assert.equal(callsTo(http, "https://api.semanticscholar.org")[0].headers["x-api-key"], "s2-key");

  // 2. Queries: the LLM's (cleaned, distinct) plus the Latin input queries
  assert.equal(stats.queries, 1);
  assert.deepEqual(survey.queries, ["copper isotope porphyry", "copper isotope fluid boiling", "Copper isotopes in porphyry systems", "copper isotope"]);
  assert.deepEqual(survey.terms, ["copper isotopes", "porphyry deposits"]);
  assert.match(stats.prompts.find((p) => p.kind === "queries").prompt, /种子文献[\s\S]*Copper isotope fractionation in porphyry copper deposits（2019）\n   标签：Copper isotopes; Porphyry/);

  // 3. Expansion: provenance of every source; duplicates merged by DOI across services
  assert.deepEqual(byTitle(survey, "Copper isotope fractionation in porphyry copper deposits").via.sort(), ["search", "seed"]);
  assert.deepEqual(byTitle(survey, "Copper isotope fractionation during sulfide precipitation").via, ["ref:seed1"]);
  const cite2 = byTitle(survey, "Copper isotopes in arc magmas revisited");
  assert.deepEqual(cite2.via.sort(), ["cites:seed1", "search"]);
  assert.deepEqual(cite2.sources.sort(), ["openalex", "s2"]);
  assert.match(cite2.abstract, /longer abstract from Semantic Scholar/, "the longer abstract wins");
  assert.equal(cite2.openalexId, "W302");
  assert.equal(cite2.s2Id, "s2cit2");
  assert.deepEqual(byTitle(survey, "Sulfur and copper isotope coupling in porphyry systems").via, ["rec"]);
  assert.ok(!survey.pool.some((p) => p.title === "Issue information"), "paratext is skipped");
  const expand = survey.stages.expand;
  assert.equal(expand.failed, 0);
  assert.deepEqual(expand.byVia, { search: 28, ref: 4, cites: 3, rec: 1 });
  assert.equal(callsTo(http, "https://api.openalex.org/works?").filter((c) => c.url.includes("sort=cited_by_count%3Adesc")).length, 1);

  // 4. Dedupe: the preprint without DOI joins its article; library papers stay, flagged
  assert.equal(survey.stages.dedupe.merged, 1);
  assert.deepEqual(byTitle(survey, "Copper isotope tracers of porphyry fluids").sources.sort(), ["openalex", "s2"]);
  const lib3 = byTitle(survey, "Mantle melting under ridges");
  assert.deepEqual([lib3.inLibrary, lib3.itemKey], [true, "LIB3"]);
  const degas = byTitle(survey, "Copper isotope behaviour during magmatic degassing");
  assert.deepEqual([degas.inLibrary, degas.itemKey], [true, "LIB4"], "matched by title (no DOI)");
  assert.match(degas.abstract, /from the library item/, "the library abstract fills a missing one");
  assert.equal(survey.stages.dedupe.inLibrary, 4);

  // 5. Rank: on-topic papers first, all scores kept
  const ranks = Object.fromEntries(survey.pool.map((p) => [p.title, p.rank]));
  assert.ok(ranks["Copper isotope tracers of porphyry fluids"] < ranks["Battery cathode degradation mechanisms"]);
  assert.ok(ranks["Copper isotope systematics of porphyry deposits: a review"] < ranks["Galaxy formation in the early universe"]);
  for (const p of survey.pool) {
    assert.deepEqual(Object.keys(p.scores), ["total", "sim", "cosine", "link", "cites", "recency"]);
    assert.ok(p.scores.total >= 0 && p.scores.total <= 1);
  }
  assert.equal(byTitle(survey, "Copper isotope fractionation in porphyry copper deposits").scores.link, 1, "a seed links fully");
  assert.equal(byTitle(survey, "Copper isotope fractionation during sulfide precipitation").scores.link, 0.5, "one of two seeds");

  // 6. Enrich: the abstract chain (Semantic Scholar tier) and TLDRs
  const cr1 = byTitle(survey, "Isotope fractionation of copper in hydrothermal fluids");
  assert.match(cr1.abstract, /from Semantic Scholar/);
  assert.equal(byTitle(survey, "Copper isotope systematics of porphyry deposits: a review").tldr, "A review of copper isotopes.");
  assert.equal(survey.stages.enrich.stillMissing, 0);

  // 7. Read: one Chinese note per paper, the seed's full-text excerpt in its prompt (no reference list)
  assert.equal(stats.read, survey.pool.length);
  const seedPrompt = stats.prompts.find((p) => p.kind === "read" && p.prompt.includes("论文标题：Copper isotope fractionation in porphyry copper deposits")).prompt;
  assert.match(seedPrompt, /全文节选：\nCopper isotope fractionation[\s\S]*Copper isotopes trace fluid boiling\./);
  assert.doesNotMatch(seedPrompt, /Another reference line/);
  assert.match(seedPrompt, /只根据下面提供的论文信息/);
  const note = byTitle(survey, "Copper isotope tracers of porphyry fluids").read;
  assert.equal(note.relevance, 5);
  assert.equal(Array.from(note.findings).length, 60, "fields are cut to 60 characters");
  assert.ok(note.findings.endsWith("…"));
  assert.equal(byTitle(survey, "Copper isotope fractionation in porphyry copper deposits").read.fullText, true);
  assert.ok(library.calls.fullText.includes("SEED1"));

  // 8. Themes: clusters named by the LLM; references numbered by theme, then score
  assert.ok(survey.themes.length >= 2);
  assert.match(survey.themes[0].label, /^主题1/);
  assert.deepEqual(survey.references.map((r) => r.n), survey.references.map((_, i) => i + 1));
  for (const r of survey.references) assert.equal(byTitle(survey, r.title).n, r.n);
  const themeOrder = survey.references.map((r) => r.theme);
  assert.deepEqual(themeOrder, [...themeOrder].sort((a, b) => a - b));
  assert.equal(survey.references[0].title, "Copper isotope fractionation in porphyry copper deposits");
  assert.deepEqual(survey.references[0].authors, ["Li Wang"]);

  // 9. Write: citations checked against the numbers each prompt gave; uncited sections flagged
  const report = survey.report;
  assert.equal(report.kind, "review");
  assert.equal(report.title, "斑岩铜矿中的铜同位素研究综述");
  assert.equal(report.model, "test-model");
  assert.equal(report.partial, false);
  assert.equal(stats.section, survey.themes.length);
  assert.equal(stats.overview, 1);
  const first = report.sections[0];
  assert.equal(first.paragraphs[0], "研究表明铜同位素可以示踪成矿流体 [1]，并与硫同位素耦合 [1]。");
  assert.equal(first.unverified, false);
  const uncited = report.sections.find((s) => s.unverified);
  assert.equal(uncited.heading, `未引用的一节${UNVERIFIED_MARK}`);
  for (const s of report.sections) {
    const allowed = new Set(survey.themes[s.theme].paperIds.map((id) => survey.pool.find((p) => p.id === id).n));
    for (const p of s.paragraphs) for (const m of p.matchAll(/\[(\d+(?:, \d+)*)\]/g)) for (const n of m[1].split(", ")) assert.ok(allowed.has(Number(n)), `[${n}] cites a paper of its own theme`);
  }
  assert.equal(report.summary, "总述第一段 [1]，以及不存在的文献。\n\n总述第二段 [2]。");
  assert.deepEqual(report.debates, ["争议一 [1]"], "an entry citing nothing valid disappears");
  assert.deepEqual(report.reading, [2, 1]);
  const sectionPromptText = stats.prompts.find((p) => p.kind === "section").prompt;
  assert.match(sectionPromptText, /不得引入列表以外的文献/);
  assert.match(sectionPromptText, /^\[1\] Copper isotope fractionation in porphyry copper deposits（2019，Economic Geology）\n    研究问题：/m);

  // Persistence: the survey file, the index and the shared reading cache
  const stored = await loadSurvey(storage, survey.id);
  assert.equal(stored.status, "done");
  assert.equal(stored.report.title, report.title);
  const [entry] = await readIndex(storage);
  assert.deepEqual(entry.counts, { seeds: 2, candidates: survey.pool.length, read: survey.pool.length, references: survey.references.length });
  assert.equal(entry.status, "done");
  assert.equal(entry.hasReport, true);
  assert.equal((await loadNoteCache(storage)).entries.size, survey.pool.length);
  assert.deepEqual(surveyCounts(survey), entry.counts);
  assert.equal(canResume(survey), false);
  assert.equal(regenerateFrom(survey), "write");
});

test("rerunning the writing stage reuses everything before it", async () => {
  const { stats, engine, survey } = setup();
  await engine().run(survey);
  const reads = stats.read;
  const overviews = stats.overview;
  await engine().run(survey, { from: "write" });
  assert.equal(survey.status, "done");
  assert.equal(stats.read, reads, "nothing is read again");
  assert.equal(stats.overview, overviews + 1);
  assert.equal(stats.queries, 1);
});

// ------------------------------------------------------------------- no LLM

test("without an LLM, stages 1–6 run and the report is a clustered candidate list", async () => {
  const { stats, http, engine, survey } = setup({ prefs: { "llm.enabled": false } });
  await engine().run(survey);
  assert.equal(survey.status, "done");
  assert.equal(survey.llm, false);
  assert.deepEqual(stats, {}, "no LLM request at all");
  assert.equal(callsTo(http, LLM_URL).length, 0);
  assert.deepEqual(survey.stages.read.skipped, true);
  assert.equal(survey.stages.queries.source, "input");
  assert.deepEqual(survey.queries, ["Copper isotopes in porphyry systems", "copper isotope porphyry", "copper isotope"], "the topic, the keywords together, multi-word keywords");
  assert.equal(survey.report.kind, "list");
  assert.match(survey.report.notice, /AI 综述需要开启大模型|未启用大模型/);
  assert.ok(survey.report.groups.length >= 1);
  assert.ok(survey.report.groups.every((g) => g.label && !/^主题\d+：/.test(g.label)), "labels come from top terms");
  const listed = survey.report.groups.flatMap((g) => g.refs).sort((a, b) => a - b);
  assert.deepEqual(listed, survey.references.map((r) => r.n));
  assert.ok(survey.pool.every((p) => !p.read));
  assert.equal(regenerateFrom(survey), "read", "regenerating with an LLM reads first");
  const md = surveyMarkdown(survey);
  assert.match(md, /未启用大模型/);
  assert.match(md, /## 参考文献/);
});

test("a survey from keywords alone searches only", async () => {
  const { http, engine, survey } = setup({ input: { ...SURVEY_INPUT, seedKeys: [] } });
  await engine().run(survey);
  assert.equal(survey.status, "done");
  assert.deepEqual(survey.seeds, []);
  assert.equal(callsTo(http, "https://api.semanticscholar.org/recommendations").length, 0);
  assert.equal(callsTo(http, "https://api.openalex.org/works/").length, 0, "no seed lookups");
  assert.ok(survey.pool.every((p) => p.via.every((v) => v === "search")));
  assert.ok(survey.pool.every((p) => p.scores.link === 0));
  assert.ok(survey.report.sections.length >= 1);
  assert.doesNotMatch(survey.stages.queries ? JSON.stringify(survey.errors) : "", /种子/);
});

// --------------------------------------------------------- cancel and resume

test("a cancelled survey keeps its finished stages and resumes without repeating work", async () => {
  let cancelled = false;
  const { stats, storage, engine, survey } = setup({
    hooks: {
      read: (prompt, s) => {
        if (s.read >= 3) cancelled = true;
        return JSON.parse(JSON.stringify({ question: "问题", findings: "发现", method: "", data: "", contribution: "贡献", limitations: "", relevance: 4 }));
      },
    },
  });
  await assert.rejects(engine().run(survey, { isCancelled: () => cancelled }), (e) => e instanceof CancelledError);
  assert.equal(survey.status, "cancelled");
  assert.equal(survey.error, null);
  assert.ok(survey.completed.includes("enrich"));
  assert.ok(!survey.completed.includes("read"));
  assert.equal(canResume(survey), true);
  const stored = await loadSurvey(storage, survey.id);
  assert.equal(stored.status, "cancelled");
  assert.equal((await readIndex(storage))[0].status, "cancelled");
  // Notes read before the cancel are cached
  const cached = (await loadNoteCache(storage)).entries.size;
  assert.ok(cached >= 3, "the note answered while cancelling is kept too");
  const readBefore = stats.read;
  const expandRequests = stats.queries;

  // Resume from the stored survey with a fresh engine (as after a restart)
  const resumed = await loadSurvey(storage, survey.id);
  cancelled = false;
  await engine().run(resumed);
  assert.equal(resumed.status, "done");
  assert.equal(stats.queries, expandRequests, "earlier stages are not rerun");
  assert.equal(stats.read, resumed.pool.length + (readBefore - cached), "cached notes are not requested again");
  assert.ok(resumed.report.sections.length >= 1);
  assert.equal(resumed.stages.read.cached, cached);
});

test("cancelling while the services answer stops the expansion at once", async () => {
  let cancel = false;
  const { engine, survey } = setup({
    overrides: {
      oaWorks: () => {
        cancel = true;
        return { results: [WORKS.review] };
      },
    },
  });
  await assert.rejects(engine().run(survey, { isCancelled: () => cancel }), (e) => e instanceof CancelledError);
  assert.equal(survey.status, "cancelled");
  assert.ok(!survey.completed.includes("expand"));
});

// ---------------------------------------------------------- failure isolation

test("a failing service, paper or section is reported but does not stop the survey", async () => {
  const { stats, engine, survey } = setup({
    retry: { retries: 0, ...noSleep },
    overrides: {
      s2Search: () => ({ status: 500, data: { message: "Internal error" } }),
      s2Recs: () => ({ status: 503, data: "busy" }),
      crossrefSearch: () => ({ status: 200, data: "<html>maintenance</html>" }),
    },
    hooks: {
      read: (prompt) => (prompt.includes("论文标题：Galaxy") ? "这不是 JSON" : undefined) ?? llmDefault("read", prompt),
      section: (prompt) => (prompt.includes("主题为「主题1") ? { status: 500, data: { error: { message: "LLM down" } } } : llmDefault("section", prompt)),
    },
  });
  await engine().run(survey);
  assert.equal(survey.status, "done");
  const sources = survey.errors.map((e) => `${e.stage}/${e.source}`);
  assert.ok(sources.includes("expand/Semantic Scholar"));
  assert.ok(survey.errors.some((e) => e.stage === "expand" && /连续失败 3 次/.test(e.message)), "a failing source is skipped after 3 failures");
  assert.ok(survey.errors.some((e) => e.stage === "read" && /1 篇文献精读失败/.test(e.message)));
  assert.match(byTitle(survey, "Galaxy formation in the early universe").readError, /解析|JSON/);
  assert.ok(survey.errors.some((e) => e.stage === "write" && /一节生成失败/.test(e.message)));
  const fallback = survey.report.sections.find((s) => s.fallback);
  assert.ok(fallback, "the failed section lists the papers' findings instead");
  assert.match(fallback.paragraphs[0], /\[\d+\]/);
  assert.equal(survey.report.partial, true);
  assert.ok(stats.overview >= 1);
});

const llmDefault = (kind, prompt) => {
  const stats = {};
  const res = llmHandler(stats)({ body: JSON.stringify({ messages: [{ content: prompt }] }) });
  return res.choices[0].message.content;
};

test("a survey fails (resumably) when no source answers, and goes on later", async () => {
  let down = true;
  const fail = (handler) => (req) => (down ? { status: 500, data: "down" } : handler(req));
  const base = surveyHttp();
  const passthrough = (name) => (req) => base.request(req).then((res) => (res.status === 200 ? res.data : res));
  const { engine, survey, storage } = setup({
    retry: { retries: 0, ...noSleep },
    prefs: { "apis.semanticScholarKey": "" },
    overrides: {
      oaSeed: fail(passthrough("oaSeed")),
      oaWorks: fail(passthrough("oaWorks")),
      crossrefSearch: fail(passthrough("crossrefSearch")),
    },
  });
  await assert.rejects(engine().run(survey), /所有数据源都请求失败/);
  assert.equal(survey.status, "failed");
  assert.equal(survey.error.stage, "expand");
  assert.ok(survey.errors.some((e) => e.source === "Semantic Scholar" && /未配置/.test(e.message)));
  assert.equal((await readIndex(storage))[0].error, survey.error.message);
  down = false;
  await engine().run(survey);
  assert.equal(survey.status, "done");
  assert.equal(survey.error, null);
});

test("odd JSON from every service never crashes a survey", async () => {
  const { engine, survey } = setup({
    retry: { retries: 0, ...noSleep },
    overrides: {
      oaSeed: () => "not an object",
      oaWorks: () => ({ results: [null, 5, "x", { id: null }, { id: "https://openalex.org/W9", display_name: 7 }], meta: "odd" }),
      s2Batch: () => ({ error: "not an array" }),
      s2Search: () => ({ data: "nope", total: "many" }),
      s2Edges: () => null,
      s2Recs: () => ({ recommendedPapers: { not: "a list" } }),
      crossrefSearch: () => ({ message: { items: [null, {}, { title: "No DOI" }, { DOI: "10.1000/x", title: [null] }] } }),
    },
  });
  await assert.rejects(engine().run(survey), (e) => !(e instanceof TypeError) && /没有找到候选文献/.test(e.message));
  assert.equal(survey.status, "failed");
  assert.equal(survey.error.stage, "expand");
  assert.ok(survey.errors.some((e) => e.source === "Semantic Scholar" && /unexpected response/.test(e.message)));
});

test("garbage LLM replies fall back where they can; a review nobody could write fails resumably", async () => {
  const garbage = { queries: () => "当然可以！", themes: () => '{"themes": "x"}', section: () => '{"paragraphs": []}', overview: () => "[]" };
  const { engine, survey, stats } = setup({ hooks: garbage });
  await assert.rejects(engine().run(survey), /综述生成失败/);
  assert.equal(survey.status, "failed");
  assert.equal(survey.error.stage, "write");
  assert.equal(survey.stages.queries.source, "input", "the queries come from the input");
  assert.ok(survey.errors.some((e) => e.stage === "queries"));
  assert.ok(survey.themes.every((t) => t.label && !t.summary), "themes are named by their terms");
  assert.ok(survey.errors.some((e) => e.stage === "themes" && /主题命名失败/.test(e.message)));
  assert.ok(survey.completed.includes("themes") && !survey.completed.includes("write"));
  assert.equal(regenerateFrom(survey), "write");
  // The model works again: only the review is written
  delete garbage.section;
  delete garbage.overview;
  const reads = stats.read;
  await engine().run(survey, { from: "write" });
  assert.equal(survey.status, "done");
  assert.equal(stats.read, reads);
});

test("a section without any citation and an overview that fails still give a partial review", async () => {
  const { engine, survey } = setup({ hooks: { overview: () => ({ status: 400, data: { error: { message: "context too long" } } }) } });
  await engine().run(survey);
  assert.equal(survey.status, "done");
  assert.equal(survey.report.partial, true);
  assert.equal(survey.report.title, SURVEY_INPUT.topic, "the topic stands in for the missing title");
  assert.deepEqual([survey.report.summary, survey.report.debates, survey.report.reading], ["", [], []]);
  assert.ok(survey.errors.some((e) => e.stage === "write" && /总述生成失败/.test(e.message)));
});

test("reading gives up when every LLM call fails", async () => {
  const { engine, survey } = setup({ hooks: { read: () => ({ status: 401, data: { error: { message: "Invalid key" } } }) } });
  await assert.rejects(engine().run(survey), /精读失败：HTTP 401/);
  assert.equal(survey.status, "failed");
  assert.equal(survey.error.stage, "read");
});

test("without an OpenAlex key the survey warns and still runs; keyless S2 is skipped", async () => {
  const { http, engine, survey } = setup({ prefs: { "apis.openalexKey": "", "apis.semanticScholarKey": "" } });
  await engine().run(survey);
  assert.equal(survey.status, "done");
  assert.ok(survey.errors.some((e) => e.code === "openalex-keyless"));
  assert.equal(callsTo(http, "https://api.semanticscholar.org").length, 0);
  assert.ok(callsTo(http, "https://api.openalex.org").every((c) => !c.url.includes("api_key")));
});

test("the start year holds for every candidate but the seeds", async () => {
  const { http, engine, survey } = setup({ input: { ...SURVEY_INPUT, yearFrom: 2021 } });
  await engine().run(survey);
  assert.equal(survey.status, "done");
  assert.ok(survey.pool.filter((p) => !p.seed).every((p) => !Number.isInteger(p.year) || p.year >= 2021));
  assert.ok(survey.pool.some((p) => p.seed && p.year < 2021), "seeds stay whatever their year");
  assert.ok(survey.stages.expand.beforeYear > 0);
  // Searches use the year filter (the title lookup of a seed does not)
  const searches = callsTo(http, "https://api.openalex.org/works?").filter((c) => c.url.includes("search=") && !c.url.includes("Sulfur"));
  assert.ok(searches.length && searches.every((c) => decodeURIComponent(c.url).includes("from_publication_date:2021-01-01")));
  assert.ok(callsTo(http, "https://api.semanticscholar.org/graph/v1/paper/search").filter((c) => !c.url.includes("Sulfur")).every((c) => c.url.includes("year=2021-")));
  assert.ok(callsTo(http, "https://api.crossref.org/works?").every((c) => decodeURIComponent(c.url).includes("from-pub-date:2021-01-01")));

  const none = setup({ input: { ...SURVEY_INPUT, yearFrom: 2026 } });
  await assert.rejects(none.engine().run(none.survey), /没有 2026 年以来的候选文献/);
  assert.equal(none.survey.status, "failed");
});

// ------------------------------------------------------------ input, presets

test("survey input is validated and normalised", () => {
  assert.throws(() => normalizeSurveyInput({ topic: " ", keywords: "x" }), /请填写调研主题/);
  assert.throws(() => normalizeSurveyInput({ topic: "T" }), /至少填写一个关键词或选择一篇种子文献/);
  assert.throws(() => normalizeSurveyInput({ topic: "T", keywords: "x", sources: { openalex: false, s2: false, crossref: false } }), /至少选择一个数据源/);
  const input = normalizeSurveyInput(
    { topic: "  铜同位素   示踪 ", keywords: "copper isotope；porphyry\nCopper isotope, ", seedKeys: ["ABCD1234", "ABCD1234", "../x", 5], depth: "huge", yearFrom: "2016", sources: { s2: false } },
    { now: NOW },
  );
  assert.deepEqual(input, {
    topic: "铜同位素 示踪",
    description: "",
    keywords: ["copper isotope", "porphyry"],
    seedKeys: ["ABCD1234"],
    depth: "standard",
    yearFrom: 2016,
    sources: { openalex: true, s2: false, crossref: true },
  });
  assert.equal(normalizeSurveyInput({ topic: "T", keywords: "k", yearFrom: 2999 }, { now: NOW }).yearFrom, null);
  assert.equal(parseKeywords(Array.from({ length: 20 }, (_, i) => `k${i}`)).length, 12);
  assert.deepEqual(Object.keys(SURVEY_DEPTHS), ["quick", "standard", "deep"]);
  assert.deepEqual([SURVEY_DEPTHS.quick.pool, SURVEY_DEPTHS.standard.deepRead, SURVEY_DEPTHS.deep.pool], [150, 40, 1000]);
});

test("cost estimates grow with the depth and drop without an LLM", () => {
  const quick = surveyEstimate("quick");
  const deep = surveyEstimate("deep");
  assert.ok(deep.aiCalls > quick.aiCalls && deep.apiRequests > quick.apiRequests && deep.minutes >= quick.minutes);
  assert.equal(quick.aiCalls, 1 + 15 + 1 + 5 + 1);
  assert.match(quick.text, /^约 \d+ 次 AI 调用、\d+ 次接口请求、预计 \d+ 分钟$/);
  const offline = surveyEstimate("standard", { llm: false });
  assert.equal(offline.aiCalls, 0);
  assert.match(offline.text, /未启用大模型/);
  assert.ok(surveyEstimate("standard", { sources: { s2: false } }).apiRequests < surveyEstimate("standard").apiRequests);
  assert.equal(surveyEstimate("bogus").depth, "standard");
});

// ---------------------------------------------------------- prompts, parsers

test("citations are checked against the numbers a prompt allowed", () => {
  const allowed = new Set([1, 2, 3, 4, 5, 11]);
  assert.deepEqual(validateCitations("铜同位素 [1, 99]，硫同位素 [2-4]。", allowed), { text: "铜同位素 [1]，硫同位素 [2, 3, 4]。", cited: [1, 2, 3, 4] });
  assert.equal(validateCitations("见【3】［5，1］与 [11][11]。", allowed).text, "见[3, 5, 1]与 [11]。");
  assert.equal(validateCitations("只有 [99]。", allowed).text, "只有。");
  assert.equal(validateCitations("年份 [2020] 和 [x] 保留，[1] 有效", allowed).text, "年份 [2020] 和 [x] 保留，[1] 有效");
  assert.equal(validateCitations("巨大范围 [1-500]", allowed).text, "巨大范围");
  assert.deepEqual(validateCitations(null, allowed), { text: "", cited: [] });
});

test("reply parsers accept good JSON and reject or trim bad replies", () => {
  assert.deepEqual(parseQueryReply('```json\n{"queries": ["a \\"b\\" AND c", "A B AND C", "", 5], "terms": ["t1", " t1 "]}\n```'), { queries: ["a b c"], terms: ["t1"] });
  assert.deepEqual(parseQueryReply('["one query", "two query"]').queries, ["one query", "two query"]);
  assert.throws(() => parseQueryReply('{"queries": []}'), /没有给出可用的检索式/);
  assert.equal(cleanQuery("title:(copper OR zinc)*"), "title copper zinc");

  const note = parseExtraction('好的：{"question": "Q", "findings": "F", "method": 3, "relevance": "9"}');
  assert.deepEqual(note, { question: "Q", method: "", data: "", findings: "F", contribution: "", limitations: "", relevance: null });
  assert.equal(parseExtraction('{"question":"Q","relevance":4.4}').relevance, 4);
  assert.throws(() => parseExtraction('{"method": "only"}'), /没有提取出研究问题或主要发现/);
  assert.throws(() => parseExtraction("no json"), /无法解析/);

  assert.deepEqual(parseThemeReply('[{"label": "“斑岩成矿”", "summary": "S"}, "流体演化"]', 2), [
    { label: "斑岩成矿", summary: "S" },
    { label: "流体演化", summary: "" },
  ]);
  assert.throws(() => parseThemeReply('[{"label": "a"}]', 2), /返回了 1 个主题，应为 2 个/);
  assert.throws(() => parseThemeReply('[{"label": ""}]', 1), /主题名为空/);

  assert.deepEqual(parseSectionReply('{"paragraphs": "一段"}', "默认标题"), { heading: "默认标题", paragraphs: ["一段"] });
  assert.throws(() => parseSectionReply('{"heading": "h", "paragraphs": []}'), /没有写出本节内容/);
  const overview = parseOverviewReply('{"title": "T", "summary": ["p1", "p2"], "debates": "一条", "reading": ["[3]", 2, "x", -1]}');
  assert.deepEqual(overview, { title: "T", summary: "p1\n\np2", debates: ["一条"], gaps: [], future: [], reading: [3, 2] });
  assert.throws(() => parseOverviewReply('{"title": "T"}'), /没有写出总述/);
  assert.equal(clip("一二三四五", 3), "一二…");
});

test("prompts carry the papers and the rules, and nothing else", () => {
  const input = normalizeSurveyInput(SURVEY_INPUT, { now: NOW });
  const q = queryPrompt(input, []);
  assert.match(q, /调研主题：Copper isotopes in porphyry systems/);
  assert.match(q, /种子文献（用户认为与主题最相关的论文）：\n（无）/);
  assert.match(q, /只返回 JSON 对象/);
  const paper = { title: "T", year: null, venue: null, abstract: "A", tldr: "TL" };
  const e = extractionPrompt(input, paper, "");
  assert.match(e, /发表年份：年份不详/);
  assert.match(e, /TLDR：TL/);
  assert.doesNotMatch(e, /全文节选/);
  assert.match(e, /每个字段不超过 60 个字/);
  const s = sectionPrompt(input, { label: "L", summary: "" }, [{ n: 7, title: "P", year: 2020, venue: "V", read: { question: "Q", findings: "F" } }]);
  assert.match(s, /^\[7\] P（2020，V）\n    研究问题：Q；主要发现：F$/m);
  assert.match(s, /只能使用列表中出现的编号/);
  const o = overviewPrompt(input, [{ heading: "H", paragraphs: ["x"] }], [{ n: 1, title: "P", year: 2020, read: { contribution: "C" } }]);
  assert.match(o, /\[1\] P（2020）——C/);
  assert.match(o, /reading/);
});

test("full-text excerpts drop the reference list and keep the conclusions", () => {
  const excerpt = fullTextExcerpt(FULL_TEXT, 1000);
  assert.ok(excerpt.length <= 1000);
  assert.match(excerpt, /^Copper isotope fractionation in porphyry copper deposits Abstract/);
  assert.match(excerpt, / … 5\. Conclusions Copper isotopes trace fluid boiling\./);
  assert.doesNotMatch(excerpt, /Another reference line/);
  assert.equal(fullTextExcerpt("short text\nwith lines"), "short text with lines");
  assert.equal(fullTextExcerpt(null), "");
  const noConclusion = fullTextExcerpt("word ".repeat(500), 100);
  assert.equal(noConclusion.length, 100);
  assert.ok(noConclusion.endsWith("…"));
});

// ------------------------------------------------------------ pool and rank

test("the collector merges papers by DOI, OpenAlex id and Semantic Scholar id", () => {
  const c = new PaperCollector();
  const a = c.add({ id: "doi:10.1000/a", title: "A", doi: "10.1000/A", openalexId: "W1", citations: 5, sources: ["openalex"] }, "search", { searchRank: 4 });
  c.add({ id: "s2:x", title: "A", doi: "https://doi.org/10.1000/a", s2Id: "x", abstract: "longer abstract", citations: 9, sources: ["s2"] }, "ref:seed1");
  c.add({ id: "s2:x", title: "A again", s2Id: "x", sources: ["s2"] }, "rec", { searchRank: 1 });
  c.add({ id: "openalex:W2", title: "B", openalexId: "W2", sources: ["openalex"] }, "cites:seed2");
  assert.equal(c.add({ title: "" }, "search"), null);
  assert.equal(c.papers.length, 2);
  assert.deepEqual(a.via, ["search", "ref:seed1", "rec"]);
  assert.equal(a.citations, 9);
  assert.equal(a.abstract, "longer abstract");
  assert.equal(a.s2Id, "x");
  assert.equal(a.searchRank, 1);
  assert.equal(a.searchHits, 2);
  assert.deepEqual([...linkedSeeds(a.via)], [1]);
});

test("the pool cap keeps seeds, shares the rest between linked and search papers", () => {
  const papers = [
    { id: "s", title: "seed", seed: 1, via: ["seed"], citations: 0 },
    ...Array.from({ length: 10 }, (_, i) => ({ id: `l${i}`, title: `l${i}`, via: i < 2 ? ["ref:seed1", "cites:seed2"] : ["rec"], citations: i })),
    ...Array.from({ length: 10 }, (_, i) => ({ id: `q${i}`, title: `q${i}`, via: ["search"], searchRank: i, citations: 0 })),
  ];
  const pool = selectPool(papers, 11);
  assert.equal(pool.length, 11);
  assert.equal(pool[0].id, "s");
  assert.deepEqual(pool.slice(1, 3).map((p) => p.id), ["l1", "l0"], "most seed links first");
  assert.equal(pool.filter((p) => p.id.startsWith("l")).length, Math.floor(10 * LINKED_SHARE));
  assert.deepEqual(pool.filter((p) => p.id.startsWith("q")).map((p) => p.id), ["q0", "q1", "q2", "q3"]);
  // Room the searches leave goes to linked papers
  assert.equal(selectPool(papers.filter((p) => !p.id.startsWith("q")), 8).length, 8);
  assert.equal(selectPool(papers, 100).length, papers.length);
});

test("stage 4 merges near-duplicate titles and flags library papers by DOI or title", () => {
  const pool = [
    { id: "doi:10.1000/a", title: "Copper isotope tracers of porphyry fluids", doi: "10.1000/a", via: ["search"], sources: ["openalex"], citations: 3 },
    { id: "s2:p", title: "Copper Isotope Tracers of Porphyry Fluids.", via: ["rec"], sources: ["s2"], citations: 1, tldr: "TL" },
    { id: "doi:10.1000/b", title: "Copper isotope tracers of porphyry fluid", doi: "10.1000/b", via: ["search"], sources: ["crossref"], citations: 0 },
    { id: "doi:10.1000/c", title: "Mantle melting", doi: "10.1000/C", via: ["search"], sources: [], citations: 0 },
    { id: "openalex:W9", title: "Sulfur in arc magmas", via: ["search"], sources: [], citations: 0 },
  ];
  const library = [
    { key: "K1", doi: "10.1000/c", title: "Other title", abstract: "library abstract" },
    { key: "K2", doi: "", title: "Sulfur in arc magmas", abstract: "" },
    { key: "K3", doi: "10.9999/zzz", title: "Copper isotope tracers of porphyry fluids" },
  ];
  const { papers, merged, inLibrary } = dedupePool(pool, library);
  assert.equal(merged, 1);
  assert.deepEqual(papers.map((p) => p.id), ["doi:10.1000/a", "doi:10.1000/b", "doi:10.1000/c", "openalex:W9"], "two DOIs stay two papers");
  assert.deepEqual(papers[0].via, ["search", "rec"]);
  assert.equal(papers[0].tldr, "TL");
  assert.deepEqual(papers.map((p) => [p.inLibrary, p.itemKey]), [[false, null], [false, null], [true, "K1"], [true, "K2"]]);
  assert.equal(papers[2].abstract, "library abstract");
  assert.equal(inLibrary, 2);
});

test("scores combine similarity, seed links, citations and recency", () => {
  const v = (x, y) => Float32Array.from([x, y]);
  const papers = [
    { id: "a", seed: 1, via: ["seed"], citations: 100, year: 2020 },
    { id: "b", via: ["ref:seed1", "rec"], citations: 0, year: 2024 },
    { id: "c", via: ["search"], citations: 10, year: null },
  ];
  const vectors = new Map([[papers[0], v(1, 0)], [papers[1], v(0.6, 0.8)], [papers[2], v(0, 1)]]);
  const centre = centroidOf([v(1, 0), v(1, 0)]);
  const ranked = scorePapers(papers, vectors, centre, 2);
  assert.deepEqual(ranked.map((p) => p.id), ["a", "b", "c"]);
  assert.deepEqual(papers[0].scores, { total: 0.55 + 0.2 + 0.15, sim: 1, cosine: 1, link: 1, cites: 1, recency: 0 });
  assert.deepEqual(papers[1].scores, { total: 0.55 * 0.6 + 0.2 * 1 + 0.1, sim: 0.6, cosine: 0.6, link: 1, cites: 0, recency: 1 });
  assert.equal(papers[2].scores.total, Math.round(0.15 * (Math.log1p(10) / Math.log1p(100)) * 1000) / 1000);
  assert.deepEqual(ranked.map((p) => p.rank), [1, 2, 3]);
  assert.equal(centroidOf([]), null);
  // Read set: seeds first, papers without text skipped
  const rs = selectReadSet([{ id: 1, abstract: "x" }, { id: 2, seed: 1, abstract: "y" }, { id: 3 }, { id: 4, itemKey: "K" }], 3, (p) => p.itemKey === "K");
  assert.deepEqual(rs.map((p) => p.id), [2, 1, 4]);
});

test("themes are chosen by silhouette and labelled by distinctive terms", async () => {
  const unit = (angle) => Float32Array.from([Math.cos(angle), Math.sin(angle), 0]);
  const vectors = [...[0, 0.05, 0.1, 0.02].map(unit), ...[1.5, 1.55, 1.6, 1.52].map(unit), ...[3, 3.05, 3.1].map(unit)];
  const { k, assignments, silhouette } = await clusterThemes(vectors);
  assert.equal(k, 3);
  assert.deepEqual(assignments.slice(0, 4), [0, 0, 0, 0]);
  assert.equal(new Set(assignments.slice(4, 8)).size, 1);
  assert.ok(silhouette > 0.5);
  assert.deepEqual((await clusterThemes(vectors.slice(0, 4))).assignments, [0, 0, 0, 0], "few papers make one theme");
  assert.deepEqual(await clusterThemes([]), { k: 0, assignments: [], silhouette: null });
  const all = ["copper isotope porphyry", "copper isotope fluid", "mantle melting ridge", "mantle plume melting"];
  assert.deepEqual(topTerms(all.slice(0, 2), all, 2), ["copper", "isotope"]);
});

// ------------------------------------------------------------- persistence

test("surveys are stored in an index capped at 30, newest first; damaged files are ignored", async () => {
  const storage = createMemoryStorage();
  const input = normalizeSurveyInput({ topic: "T", keywords: "k" });
  const surveys = [];
  for (let i = 0; i < SURVEY_LIMIT + 2; i++) {
    const s = createSurvey(input, { now: NOW + i * 1000 });
    surveys.push(s);
    await saveSurvey(storage, s);
  }
  const index = await readIndex(storage);
  assert.equal(index.length, SURVEY_LIMIT);
  assert.equal(index[0].id, surveys.at(-1).id);
  assert.ok(!storage.files.has(`survey-${surveys[0].id}.json`), "files of dropped surveys are removed");
  assert.ok(storage.files.has(`survey-${surveys[2].id}.json`));
  // An old survey saved again stays (the oldest other one makes room)
  await saveSurvey(storage, surveys[2]);
  assert.ok((await readIndex(storage)).some((e) => e.id === surveys[2].id));

  assert.equal(await loadSurvey(storage, "../evil"), null);
  storage.files.set(`survey-${surveys[5].id}.json`, JSON.stringify({ id: "other" }));
  assert.equal(await loadSurvey(storage, surveys[5].id), null);
  storage.files.set(SURVEY_INDEX_FILE, "{not json");
  await assert.rejects(storage.readJSON(SURVEY_INDEX_FILE));
  assert.deepEqual(await readIndex(storage), []);
  storage.files.set(SURVEY_INDEX_FILE, JSON.stringify([{ id: "abc123def", topic: "x", createdAt: 1 }, { id: "../x" }, null]));
  assert.equal((await readIndex(storage)).length, 1);

  assert.equal(await deleteSurvey(storage, "abc123def"), true);
  assert.equal(await deleteSurvey(storage, "abc123def"), false);
  assert.deepEqual(await readIndex(storage), []);
  assert.ok(isSurveyId(newSurveyId(NOW)));
  assert.notEqual(newSurveyId(NOW), newSurveyId(NOW));
});

test("reading notes are cached by paper and model, with the relevance per topic", async () => {
  const storage = createMemoryStorage();
  const cache = await loadNoteCache(storage);
  const paper = { doi: "10.1000/X", title: "T" };
  const key = noteCacheKey(paper, "m1");
  assert.equal(key, noteCacheKey({ doi: "https://doi.org/10.1000/x", title: "other" }, "m1"));
  assert.notEqual(key, noteCacheKey(paper, "m2"));
  assert.notEqual(noteCacheKey({ title: "T" }, "m1"), noteCacheKey({ title: "U" }, "m1"));
  writeCachedNote(cache, key, "topic A", { question: "Q", findings: "F", relevance: 5 }, NOW);
  assert.deepEqual(readCachedNote(cache, key, "Topic A ").note.relevance, 5);
  const other = readCachedNote(cache, key, "topic B");
  assert.equal(other.relevanceKnown, false);
  assert.equal(other.note.question, "Q");
  const { saveNoteCache } = await import("../src/survey/store.js");
  await saveNoteCache(storage, cache);
  assert.equal((await loadNoteCache(storage)).entries.size, 1);
  storage.files.set(SURVEY_CACHE_FILE, JSON.stringify({ version: 1, entries: { a: null, b: { e: {}, at: 1 } } }));
  assert.equal((await loadNoteCache(storage)).entries.size, 1);
});

test("a cached note from another topic estimates the relevance from the similarity", async () => {
  const first = setup();
  await first.engine().run(first.survey);
  const reads = first.stats.read;
  // Same papers, another topic, same storage: notes come from the cache
  const second = createSurvey(normalizeSurveyInput({ ...SURVEY_INPUT, topic: "Porphyry copper fluids" }, { now: NOW }), { now: NOW + 1 });
  await first.engine().run(second);
  assert.equal(first.stats.read, reads, "no extraction requested again");
  const note = second.pool.find((p) => p.read)?.read;
  assert.equal(note.relevanceEstimated, true);
  assert.ok(note.relevance >= 1 && note.relevance <= 5);
});

// ---------------------------------------------------------------- rendering

test("Markdown and note HTML render the report as escaped text with DOI links", () => {
  const survey = {
    id: "abc123def",
    input: { topic: "铜 <同位素> *示踪*" },
    counts: { read: 2, candidates: 10 },
    report: {
      kind: "review",
      title: "综述 <script>alert(1)</script>",
      summary: "第一段 [1]\n\n第二段 [2]",
      sections: [{ heading: "主题 # 一", paragraphs: ["段落 <b>x</b> [1, 2]"] }],
      debates: ["争议 [2]"],
      gaps: [],
      future: ["方向"],
      reading: [2, 1, 9],
      generatedAt: NOW,
      model: "m",
    },
    references: [
      { n: 1, id: "doi:10.1000/a", title: "Paper <A>", authors: ["Ann Lee", "Bo Zhang", "C D", "E F"], authorCount: 4, year: 2020, venue: "GCA", doi: "10.1000/a", url: null, inLibrary: true },
      { n: 2, id: "openalex:W1", title: "Paper B", authors: [], authorCount: 0, year: null, venue: null, doi: null, url: "javascript:alert(1)", inLibrary: false },
    ],
  };
  const html = surveyNoteHtml(survey);
  assert.doesNotMatch(html, /<script>|<b>|javascript:/);
  assert.match(html, /<h1>综述 &lt;script&gt;alert\(1\)&lt;\/script&gt;<\/h1>/);
  assert.match(html, /<p><em>ZotWatch 主题调研：铜 &lt;同位素&gt; \*示踪\* · 生成于 2026-10-17 · 模型 m · 精读 2 篇 · 候选 10 篇<\/em><\/p>/);
  assert.match(html, /本综述由 AI 根据所列文献/);
  assert.match(html, /<p>\[1\] Ann Lee, Bo Zhang, C D et al\. Paper &lt;A&gt;\. GCA, 2020\. <a href="https:\/\/doi\.org\/10\.1000\/a">doi:10\.1000\/a<\/a> （已在文库）<\/p>/);
  assert.match(html, /<p>\[2\] Paper B\.<\/p>/);
  assert.match(html, /<ol><li>\[2\] Paper B<\/li><li>\[1\] Paper &lt;A&gt;<\/li><\/ol>/);
  assert.match(html, /<h2>主要争议<\/h2>/);
  assert.doesNotMatch(html, /研究空白/, "empty lists are left out");

  const md = surveyMarkdown(survey);
  assert.match(md, /^# 综述 \\<script\\>alert\(1\)\\<\/script\\>\n/);
  assert.match(md, /\*\*调研主题\*\*：铜 \\<同位素\\> \\\*示踪\\\*/);
  assert.match(md, /## 主题 \\# 一\n\n段落 \\<b\\>x\\<\/b\\> \[1, 2\]/);
  assert.match(md, /1\. Ann Lee, Bo Zhang, C D et al\. Paper \\<A\\>\. GCA, 2020\. <https:\/\/doi\.org\/10\.1000\/a>（已在文库）/);
  assert.match(md, /## 推荐阅读顺序\n\n1\. \[2\] Paper B\n2\. \[1\] Paper \\<A\\>\n/);
  assert.equal(markdownFileName({ input: { topic: 'a/b:"c"' } }, NOW), "ZotWatch 调研 - a b c - 2026-10-17.md");
  assert.deepEqual(reportCitations(survey.report), [1, 2]);
  assert.equal(referenceText({ authors: ["X"], authorCount: 5, title: "T.", venue: null, year: 2020 }), "X et al. T. 2020.");
  assert.equal(referenceText({ authors: [], authorCount: 0, title: "", venue: null, year: null }), "");
  assert.equal(escapeHtml(`<"&>`), "&lt;&quot;&amp;&gt;");
});

// --------------------------------------------------------------- sources

test("source records are mapped defensively", async () => {
  assert.equal(mapOpenAlexPaper(null), null);
  assert.equal(mapOpenAlexPaper({ ...WORKS.review, type: "dataset" }), null);
  assert.equal(mapOpenAlexPaper({ ...WORKS.review, display_name: "", title: null }), null);
  assert.equal(mapOpenAlexPaper({ ...WORKS.review, id: null, doi: null }), null);
  const p = mapOpenAlexPaper({ ...WORKS.review, authorships: "odd", primary_location: null, cited_by_count: "x", publication_year: "abc" });
  assert.deepEqual([p.authors, p.venue, p.citations, p.year], [[], null, 0, 2022]);
  const cr = mapCrossrefPaper(CROSSREF_ITEM);
  assert.deepEqual([cr.id, cr.year, cr.published, cr.venue, cr.citations, cr.authors[0].name], ["doi:10.6000/cr1", 2020, "2020-06-02T00:00:00Z", "Chemical Geology", 12, "Mei Liu"]);
  assert.equal(mapCrossrefPaper({ ...CROSSREF_ITEM, DOI: null }), null);
  assert.equal(mapCrossrefPaper({ ...CROSSREF_ITEM, type: "book" }), null);

  // An id filter name OpenAlex rejects is replaced by the next alias, and remembered
  resetOpenAlexIdFilter();
  const seen = [];
  const http = mockHttp([
    ["https://api.openalex.org/works", (req) => {
      const filter = new URL(req.url).searchParams.get("filter");
      seen.push(filter.split(":")[0]);
      if (filter.startsWith("openalex:")) return { status: 400, data: { error: "Invalid filter" } };
      return { results: [WORKS.ref1, null, { id: "x" }] };
    }],
  ]);
  const works = await openAlexWorksByIds(http, ["https://openalex.org/W201", "W999", "W201"], { retry: noSleep });
  assert.deepEqual(works.map((w) => w.openalexId), ["W201"]);
  await openAlexWorksByIds(http, ["W201"], { retry: noSleep });
  assert.deepEqual(seen, ["openalex", "ids.openalex", "ids.openalex"]);
  resetOpenAlexIdFilter();
});

test("survey records start clean", () => {
  const input = normalizeSurveyInput(SURVEY_INPUT, { now: NOW });
  const s = createSurvey(input, { now: NOW, llm: true, model: "m" });
  assert.equal(s.status, "running");
  assert.deepEqual(s.counts, { seeds: 2, candidates: 0, read: 0, references: 0 });
  assert.ok(isSurveyId(s.id));
  assert.equal(canResume({ ...s, status: "failed" }), true);
  assert.equal(canResume({ ...s, status: "done" }), false);
  assert.deepEqual(libraryItems().map((i) => i.key), ["SEED1", "SEED2", "LIB3", "LIB4"]);
});
