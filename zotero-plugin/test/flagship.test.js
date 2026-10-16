// Flagship geoscience track (顶刊地学速览): settings and calibration, the
// gate partition, the LLM judge, display scoring, and the engine integration
// (split Crossref fetch, ISSN classification, merge, failures, cancellation).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import { FLAGSHIP_DEFAULTS, cosine, isFlagshipWork, newestFirst, parseAnchorLines, partitionByAnchors, splitFlagship } from "../src/core/flagship.js";
import { CancelledRequestError } from "../src/core/http.js";
import { ProfileRanker } from "../src/core/scoring.js";
import { Matrix } from "../src/core/vector.js";
import { CancelledError, ZotWatchEngine } from "../src/engine.js";
import { createEmbedder } from "../src/embedding/providers.js";
import { judgeRelevance, parseRelevantIds, relevancePrompt } from "../src/llm/llm.js";
import { loadSettings } from "../src/settings.js";
import { createMemoryStorage } from "../src/store.js";
import { flagshipStatusLine } from "../src/zotero/addon.js";
import { embeddingHandler, mockHttp, noSleep } from "./helpers.js";

const here = dirname(fileURLToPath(import.meta.url));
const NOW = Date.UTC(2024, 9, 10, 12);

// ---------------------------------------------------------------- settings

/** Default prefs exactly as Zotero loads them from addon/prefs.js. */
function prefDefaults() {
  const defaults = {};
  vm.runInNewContext(readFileSync(join(here, "../addon/prefs.js"), "utf8"), {
    pref: (key, value) => (defaults[key.replace("extensions.zotwatch.", "")] = value),
  });
  return defaults;
}

test("flagship settings default to the Python track, calibrated per embedding provider", () => {
  const prefs = prefDefaults();
  const local = loadSettings((k) => prefs[k]).flagship;
  assert.equal(local.enabled, false);
  assert.deepEqual(local.issns, FLAGSHIP_DEFAULTS.issns);
  assert.equal(local.issns.length, 12);
  assert.equal(local.positiveAnchor, FLAGSHIP_DEFAULTS.positiveAnchor);
  assert.deepEqual(local.negativeAnchors, FLAGSHIP_DEFAULTS.negativeAnchors);
  assert.equal(local.negativeAnchors.length, 6);
  assert.equal(local.llmBoundary, FLAGSHIP_DEFAULTS.llmBoundary);
  assert.match(local.llmBoundary, /computer science — even when published in a general-science journal\.$/);
  assert.equal(local.llmFallback, true);
  assert.equal(local.maxResults, 30);
  // Local TF-IDF cosines are far lower than neural ones
  assert.equal(local.minScore, 0.12);
  assert.equal(local.grayLow, 0.04);

  for (const provider of ["dashscope", "voyage", "openai"]) {
    const api = loadSettings((k) => ({ ...prefs, "embedding.provider": provider })[k]).flagship;
    assert.equal(api.minScore, 0.3, provider);
    assert.equal(api.grayLow, 0.22, provider);
  }
  // Unset prefs (no prefs.js) give the same defaults
  assert.deepEqual(loadSettings(() => undefined).flagship, local);
});

test("flagship settings normalize ISSNs, split anchors on lines and clamp the thresholds", () => {
  const settings = (prefs) => loadSettings((k) => prefs[k]).flagship;
  const s = settings({
    "flagship.issns": "2662138x, 0028-0836；0028 0836\nnot-an-issn\n1476-4687 Nature (online)",
    "flagship.negativeAnchors": "Atmospheric science, meteorology.\n\n  Chemistry: synthesis, catalysis.  \nAtmospheric science, meteorology.",
    "flagship.positiveAnchor": "  Mantle petrology.  ",
    "flagship.minScore": "0.2",
    "flagship.grayLow": "0.25",
    "flagship.maxResults": "500",
    "flagship.llmFallback": false,
  });
  assert.deepEqual(s.issns, ["2662-138X", "0028-0836", "1476-4687"]);
  assert.deepEqual(s.negativeAnchors, ["Atmospheric science, meteorology.", "Chemistry: synthesis, catalysis."], "commas stay inside an anchor");
  assert.equal(s.positiveAnchor, "Mantle petrology.");
  assert.equal(s.minScore, 0.2);
  assert.equal(s.grayLow, 0.2, "the gray zone never starts above the accept threshold");
  assert.equal(s.maxResults, 200);
  assert.equal(s.llmFallback, false);

  // Blank fields restore the defaults; 0 means no cap
  const blank = settings({
    "flagship.issns": " ",
    "flagship.negativeAnchors": "\n",
    "flagship.positiveAnchor": "",
    "flagship.llmBoundary": "",
    "flagship.maxResults": "0",
    "flagship.minScore": "abc",
  });
  assert.deepEqual(blank.issns, FLAGSHIP_DEFAULTS.issns);
  assert.deepEqual(blank.negativeAnchors, FLAGSHIP_DEFAULTS.negativeAnchors);
  assert.equal(blank.positiveAnchor, FLAGSHIP_DEFAULTS.positiveAnchor);
  assert.equal(blank.llmBoundary, FLAGSHIP_DEFAULTS.llmBoundary);
  assert.equal(blank.maxResults, 0);
  assert.equal(blank.minScore, 0.12, "unparseable threshold falls back to the calibration");
  assert.deepEqual(parseAnchorLines("a, b\r\nc\n"), ["a, b", "c"]);
});

// -------------------------------------------------------------------- gate

test("the gate rejects articles closer to any negative anchor, then applies the score bands", () => {
  const unit = (x, y) => {
    const n = Math.hypot(x, y);
    return Float32Array.of(x / n, y / n, 0);
  };
  const positive = Float32Array.of(1, 0, 0);
  const negatives = [Float32Array.of(0, 1, 0), Float32Array.of(0, 0, 1)];
  const vectors = [
    unit(0.9, 0.1), // pos 0.994 -> accepted
    unit(1, 1), // pos = neg -> a tie rejects
    Float32Array.of(0.6, 0.62, 0.5), // high pos but a negative is higher -> rejected
    unit(0.3, 0.05), // pos 0.986... accepted
    Float32Array.of(0.3, 0.1, 0.1), // not normalized: cosine 0.905 -> accepted
    Float32Array.of(0, 0, 0), // all zeros: pos = neg = 0 -> tie rejects
  ];
  const part = partitionByAnchors(vectors, positive, negatives, { minScore: 0.9, grayLow: 0.5 });
  assert.deepEqual(part.accepted, [0, 3, 4]);
  assert.deepEqual(part.rejected, [1, 2, 5]);
  assert.deepEqual(part.gray, []);
  assert.ok(Math.abs(part.scores[1].pos - part.scores[1].neg) < 1e-6);

  // Bands on the positive cosine: >= minScore accepts, < grayLow rejects
  const angled = (cos) => [cos, 0, Math.sqrt(1 - cos * cos)];
  const bands = partitionByAnchors([angled(0.3), angled(0.29), angled(0.22), angled(0.21)], positive, [], { minScore: 0.3, grayLow: 0.22 });
  assert.deepEqual(bands, {
    accepted: [0],
    gray: [1, 2],
    rejected: [3],
    scores: bands.scores,
  });
  assert.equal(bands.scores[3].neg, 0, "no negatives: neg is 0 and never rejects");
  // Without negatives even a zero vector only fails the bands
  assert.deepEqual(partitionByAnchors([Float32Array.of(0, 0, 0)], positive, [], { minScore: 0, grayLow: 0 }).accepted, [0]);
  assert.equal(cosine(Float32Array.of(0, 0), Float32Array.of(1, 0)), 0);
});

test("flagship articles are recognised by any ISSN and listed newest first", () => {
  const set = new Set(["0028-0836", "2662-138X"]);
  assert.equal(isFlagshipWork({ issns: ["2662138x"] }, set), true);
  assert.equal(isFlagshipWork({ issns: ["0016-7037"] }, set), false);
  assert.equal(isFlagshipWork({ issns: null }, set), false);
  assert.equal(isFlagshipWork({ issns: ["0028-0836"] }, new Set()), false);
  const works = [{ id: "a", issns: ["0016-7037"] }, { id: "b", issns: ["0016-7037", "0028-0836"] }, { id: "c", source: "arxiv" }];
  const { flagship, personal } = splitFlagship(works, set);
  assert.deepEqual(flagship.map((w) => w.id), ["b"]);
  assert.deepEqual(personal.map((w) => w.id), ["a", "c"]);

  const dated = [
    { id: "old", published: "2024-10-01T00:00:00Z" },
    { id: "none", published: null },
    { id: "new", published: "2024-10-09T00:00:00Z" },
    { id: "bad", published: "soon" },
    { id: "new2", published: "2024-10-09T00:00:00Z" },
  ];
  assert.deepEqual(newestFirst(dated).map((w) => w.id), ["new", "new2", "old", "none", "bad"]);
  assert.deepEqual(newestFirst(dated, 2).map((w) => w.id), ["new", "new2"]);
  assert.deepEqual(newestFirst(dated, 0).length, 5);
});

// --------------------------------------------------------------------- LLM

test("the relevance prompt is the Python prompt with short paper ids", () => {
  const prompt = relevancePrompt("ONLY geology.", [
    { title: "Mantle plumes", abstract: "Hot spots." },
    { title: "Batteries", abstract: "" },
  ]);
  assert.equal(
    prompt,
    `You are an academic research assistant. Determine whether each paper is relevant to the user's research interests.

User research interests:
ONLY geology.

Papers:
- P1
  Title: Mantle plumes
  Abstract: Hot spots.
- P2
  Title: Batteries
  Abstract: No abstract available

Return a JSON object with:
{
  "relevant_ids": ["paper_id_1", "paper_id_2", ...],
  "irrelevant_ids": ["paper_id_3", "paper_id_4", ...]
}

Rules:
- Use the title and abstract meaning (semantic relevance), not just keyword overlap.
- If a paper is borderline or uncertain, classify it as irrelevant.
- Return only JSON (no markdown, no extra text).
`,
  );
});

test("relevance replies accept P ids and plain numbers and reject unusable shapes", () => {
  const ids = (reply, n = 4) => [...parseRelevantIds(reply, n)].sort();
  assert.deepEqual(ids('{"relevant_ids": ["P1", "p3", " P4 "], "irrelevant_ids": ["P2"]}'), [0, 2, 3]);
  assert.deepEqual(ids('```json\n{"relevant_ids": [2, "3", "P9", "10.1038/x", null]}\n```'), [1, 2]);
  assert.deepEqual(ids('<think>…</think>{"relevant_ids": "P1, P2"}'), [0, 1]);
  assert.deepEqual(ids('{"irrelevant_ids": ["P1"]}'), [], "no relevant_ids: nothing is relevant (Python)");
  assert.deepEqual(ids('{"relevant_ids": null}'), []);
  assert.throws(() => parseRelevantIds('["P1"]', 2), /not a JSON object/);
  assert.throws(() => parseRelevantIds('{"relevant_ids": {"P1": true}}', 2), /no relevant_ids list/);
  assert.throws(() => parseRelevantIds("I think P1 is relevant.", 2));
});

function scriptedLlm(replies) {
  const prompts = [];
  return {
    prompts,
    async chat(prompt, options) {
      prompts.push({ prompt, options });
      const reply = replies[prompts.length - 1];
      if (reply instanceof Error) throw reply;
      return typeof reply === "function" ? reply(prompt) : reply;
    },
  };
}

const paper = (n) => ({ id: `w${n}`, title: `Paper ${n}`, abstract: `Abstract ${n}` });

test("the LLM judge keeps relevant papers per batch and stays inclusive when a batch fails", async () => {
  const works = [1, 2, 3, 4, 5, 6, 7].map(paper);
  const llm = scriptedLlm([
    '{"relevant_ids": ["P2"], "irrelevant_ids": ["P1"]}',
    "sorry, I cannot help",
    new Error("HTTP 502 for https://llm.example"),
    '{"relevant_ids": [1]}',
  ]);
  const result = await judgeRelevance(llm, works, " ONLY geology. ", { batchSize: 2 });
  assert.equal(llm.prompts.length, 4);
  assert.match(llm.prompts[0].prompt, /User research interests:\nONLY geology\.\n/);
  assert.match(llm.prompts[1].prompt, /- P1\n {2}Title: Paper 3\n[\s\S]*- P2\n {2}Title: Paper 4/, "ids restart in every batch");
  assert.deepEqual(llm.prompts[0].options, { maxTokens: 1024, temperature: 0.3 });
  // Batch 1 judged, batch 2 unparseable (kept), batch 3 failed (kept), batch 4 judged
  assert.deepEqual(result.kept.map((w) => w.id), ["w2", "w3", "w4", "w5", "w6", "w7"]);
  assert.equal(result.judged, 3);
  assert.equal(result.failedBatches, 2);
  assert.match(result.error, /无法解析大模型的回复/);
  assert.deepEqual([...result.confirmed].map((w) => w.id), ["w2", "w7"]);
});

test("the LLM judge stops asking after its first request fails and propagates a cancel", async () => {
  const works = [1, 2, 3, 4, 5].map(paper);
  const failing = scriptedLlm([new Error("HTTP 401 for https://llm.example: invalid key")]);
  const result = await judgeRelevance(failing, works, "geology", { batchSize: 2 });
  assert.equal(failing.prompts.length, 1, "a configuration problem is not repeated for every batch");
  assert.deepEqual(result.kept.map((w) => w.id), ["w1", "w2", "w3", "w4", "w5"]);
  assert.equal(result.failedBatches, 3);
  assert.equal(result.judged, 0);
  assert.equal(result.error, "HTTP 401 for https://llm.example: invalid key");

  // An empty boundary keeps everything without asking (Python parity)
  const silent = scriptedLlm([]);
  assert.equal((await judgeRelevance(silent, works, "  ")).kept.length, 5);
  assert.equal(silent.prompts.length, 0);

  const cancelled = scriptedLlm(['{"relevant_ids": []}', new CancelledRequestError()]);
  await assert.rejects(judgeRelevance(cancelled, works, "geology", { batchSize: 2 }), (e) => e.cancelled === true);
});

// ----------------------------------------------------------- embedding, score

test("voyage embeds anchors as queries; other providers ignore the input type", async () => {
  const bodies = [];
  const http = mockHttp([
    [() => true, (req) => {
      const body = JSON.parse(req.body);
      bodies.push(body);
      return { data: body.input.map((_, index) => ({ index, embedding: [1, 0] })) };
    }],
  ]);
  const voyage = createEmbedder({ provider: "voyage", apiKey: "k" }, http, { retry: noSleep });
  await voyage.embed(["anchor"], { inputType: "query" });
  await voyage.embed(["article"]);
  assert.deepEqual(bodies.map((b) => b.input_type), ["query", "document"]);
  const dashscope = createEmbedder({ provider: "dashscope", apiKey: "k", dimensions: 1024 }, http, { retry: noSleep });
  await dashscope.embed(["anchor"], { inputType: "query" });
  assert.equal(bodies[2].input_type, undefined);
});

test("display scores use the ranking math without the gate, thresholds or labels", async () => {
  const rows = [
    [1, 0],
    [0.9, Math.sqrt(1 - 0.81)],
  ].map((r) => Float32Array.from(r));
  const profile = { matrix: Matrix.fromRows(rows), weights: Float32Array.of(1, 1), clusters: [] };
  const journals = new Map([["0028-0836", { impactFactor: 50, isChineseCore: false }]]);
  const cfg = loadSettings((k) => ({ "embedding.provider": "dashscope", "scoring.minSimilarity": "0.5" })[k]).scoring;
  const ranker = new ProfileRanker(profile, journals, cfg);
  const works = [
    { id: "near", issns: ["0028-0836"], vector: Float32Array.of(1, 0) },
    { id: "far", issns: ["0016-7037"], vector: Float32Array.of(0, 1) },
  ];
  const scored = await ranker.scoreWorks(works);
  assert.deepEqual(scored.map((w) => w.id), ["near", "far"], "input order, nothing gated");
  const { ranked } = await ranker.rank(works);
  assert.deepEqual(ranked.map((w) => w.id), ["near"], "rank() gates the far one");
  assert.equal(scored[0].score, ranked[0].score);
  assert.equal(scored[0].impactFactor, 50);
  assert.equal(scored[0].label, undefined);
  assert.ok(scored[1].similarity < 0.5 && scored[1].score > 0);

  const empty = new ProfileRanker({ matrix: new Matrix(0, 1), weights: new Float32Array(0), clusters: [] }, journals, cfg);
  const cold = await empty.scoreWorks(works);
  assert.deepEqual(cold.map((w) => [w.score, w.similarity, w.ifScore, w.impactFactor]), [
    [0, 0, 0, null],
    [0, 0, 0, null],
  ]);
});

test("the status line shows the flagship settings and the last run's funnel", () => {
  const cfg = { issns: ["0028-0836", "1476-4687"], minScore: 0.3, grayLow: 0.22 };
  assert.equal(flagshipStatusLine(cfg), "顶刊地学速览：2 个 ISSN，接受阈值 0.30，灰区下限 0.22");
  assert.equal(
    flagshipStatusLine(cfg, { candidates: 40, withAbstract: 30, recent: 25, accepted: 6, gray: 5, judged: 5, rejected: 16, recommended: 9 }),
    "顶刊地学速览：2 个 ISSN，接受阈值 0.30，灰区下限 0.22；上次运行 40 篇候选，25 篇有摘要且在时间范围内，入选 9 篇（灰区 5 篇，大模型复核 5 篇）",
  );
  assert.equal(flagshipStatusLine({ issns: [], minScore: 0.3, grayLow: 0.22 }), "顶刊地学速览：未填写有效的 ISSN");
});

// ------------------------------------------------------------------ engine

function libraryItems() {
  const items = [];
  const topics = [
    ["mantle", "Mantle plume dynamics and mantle convection", "Mantle mantle upwelling"],
    ["subduction", "Subduction zone fluids", "Subduction subduction slab dehydration"],
    ["copper", "Copper isotope systematics", "Copper isotope copper fractionation"],
  ];
  for (const [key, title, abstract] of topics) {
    for (let i = 0; i < 6; i++) {
      items.push({
        key: `${key.toUpperCase()}${i}`,
        itemType: "journalArticle",
        title: `${title} ${i}`,
        abstract: `${abstract} part ${i}`,
        doi: `10.9999/lib.${key}.${i}`,
        issn: "0016-7037",
        venue: "Geochimica et Cosmochimica Acta",
        dateAdded: "2024-01-01 00:00:00",
      });
    }
  }
  return items;
}

function crossrefWork(doi, title, abstract, { created = "2024-10-08T10:00:00Z", issns = ["0016-7037"], venue = "GCA" } = {}) {
  return {
    DOI: doi,
    title: [title],
    author: [{ given: "Ann", family: "Lee" }],
    abstract: abstract ? `<jats:p>${abstract}</jats:p>` : undefined,
    "container-title": [venue],
    created: { "date-time": created },
    URL: `https://doi.org/${doi}`,
    type: "journal-article",
    ISSN: issns,
  };
}

const PERSONAL = [
  crossrefWork("10.7777/p.1", "Copper isotope fractionation in arc magmas", "Copper copper isotope isotope"),
  crossrefWork("10.7777/p.2", "Mantle convection dynamics revisited", "Mantle mantle"),
  crossrefWork("10.7777/p.3", "Battery cathode chemistry", "Battery battery cathode"),
];

const nature = (doi, title, abstract, created, issns = ["0028-0836", "1476-4687"]) =>
  crossrefWork(doi, title, abstract, { created, issns, venue: "Nature" });

// With the anchors below (mock embeddings count topic keywords): positive
// cosines A 0.93, gray ones 0.17, B/D close to a negative anchor
const FLAGSHIP = [
  nature("10.1038/a", "Mantle subduction dynamics", "Mantle subduction mantle isotope", "2024-10-08T10:00:00Z"),
  nature("10.1038/b", "Battery cathode", "Battery battery cathode mantle", "2024-10-09T08:00:00Z"),
  nature("10.1038/c", "Copper sulfur deposits", "Copper copper sulfur mantle", "2024-10-09T10:00:00Z"),
  nature("10.1038/c2", "Copper sulfur ores", "Copper sulfur sulfur subduction", "2024-10-07T10:00:00Z"),
  nature("10.1038/d", "Galaxy formation", "Galaxy galaxy mantle", "2024-10-09T09:00:00Z"),
  nature("10.1038/e", "Mantle without abstract", null, "2024-10-09T09:00:00Z"),
  nature("10.1038/f", "Old mantle subduction study", "Mantle subduction isotope", "2024-08-01T00:00:00Z"),
  // Already in the library
  nature("10.9999/lib.mantle.1", "Mantle plume dynamics and mantle convection 1", "Mantle mantle", "2024-10-08T00:00:00Z"),
  // Online ISSN only, written without hyphen
  nature("10.1038/g", "Mantle plume isotope geochemistry", "Mantle isotope subduction", "2024-10-09T12:00:00Z", ["14764687"]),
];

const WHITELIST = "issn,title,category,impact_factor\n0028-0836,Nature,flagship,50.5\n";

function setup({ prefs = {}, llm, embeddings } = {}) {
  const counter = {};
  const state = { personal: PERSONAL, flagship: FLAGSHIP, flagshipHandler: null, openalex: [] };
  // Each journal answers the queries that name one of its ISSNs
  const isFlagshipQuery = (r) => r.url.startsWith("https://api.crossref.org/works") && decodeURIComponent(r.url).includes("issn:1476-4687");
  const crossref = (req) => {
    const url = decodeURIComponent(req.url);
    const items = [...(url.includes("issn:0016-7037") ? state.personal : []), ...(/issn:(0028-0836|1476-4687)/.test(url) ? state.flagship : [])];
    return { message: { "total-results": items.length, items } };
  };
  const routes = [
    [isFlagshipQuery, (req) => (state.flagshipHandler ? state.flagshipHandler(req) : crossref(req))],
    ["https://api.crossref.org/works", crossref],
    ["https://api.openalex.org/works", () => ({ results: state.openalex })],
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddings || embeddingHandler(counter)],
  ];
  if (llm) routes.push(["https://llm.example/v1/chat/completions", llm]);
  const http = mockHttp(routes);
  const allPrefs = {
    "embedding.provider": "dashscope",
    "embedding.apiKey": "sk-test",
    "journals.autoFromLibrary": true,
    "journals.whitelist": WHITELIST,
    "scoring.minSimilarity": "0.3",
    "scoring.excludeKeywords": "dynamics",
    "abstracts.crossref": false,
    "flagship.enabled": true,
    "flagship.positiveAnchor": "mantle subduction isotope",
    "flagship.negativeAnchors": "battery\ngalaxy",
    "flagship.minScore": "0.5",
    "flagship.grayLow": "0.1",
    ...prefs,
  };
  const storage = createMemoryStorage();
  const items = libraryItems();
  const engine = (overrides = {}) =>
    new ZotWatchEngine({
      http,
      storage,
      library: { getItems: async () => items },
      settings: loadSettings((k) => ({ ...allPrefs, ...overrides })[k]),
      now: () => NOW,
      retry: noSleep,
    });
  return { engine, http, storage, state, counter };
}

const llmPrefs = {
  "llm.enabled": true,
  "llm.apiKey": "k",
  "llm.baseURL": "https://llm.example/v1",
  "llm.model": "m",
  "llm.translateTitles": false,
  "llm.summaryTopN": "0",
  // These mocks answer the flagship and summary prompts only (topic labels: test/topics.test.js)
  "llm.topicLabels": false,
};
const promptOf = (req) => JSON.parse(req.body).messages[0].content;
const reply = (content) => ({ choices: [{ message: { content } }] });

test("flagship articles get their own Crossref query, skip the personal filters and pass the gate newest first", async () => {
  const { engine, http } = setup({ prefs: { "sources.crossref.maxResults": "120" } });
  const stages = [];
  const run = await engine().run({ onProgress: (p) => stages.push(p) });
  assert.deepEqual(run.errors, []);

  const crossref = http.calls.filter((c) => c.url.startsWith("https://api.crossref.org/works")).map((c) => decodeURIComponent(c.url));
  assert.equal(crossref.length, 2);
  const [personalQuery, flagshipQuery] = [crossref.find((u) => !u.includes("issn:1476-4687")), crossref.find((u) => u.includes("issn:1476-4687"))];
  assert.match(personalQuery, /issn:0016-7037/);
  assert.doesNotMatch(personalQuery, /issn:0028-0836/, "whitelisted flagship ISSNs leave the personal query");
  assert.match(flagshipQuery, /issn:0028-0836,issn:1476-4687,issn:0036-8075/);
  assert.doesNotMatch(flagshipQuery, /0016-7037/);
  assert.match(personalQuery, /rows=120/);
  assert.match(flagshipQuery, /rows=120/, "a separate result budget");
  assert.deepEqual(run.stats.perSource, { Crossref: 3, 顶刊: 9 });
  assert.equal(run.stats.fetched, 12);
  assert.equal(run.stats.afterDedupe, 11, "the library duplicate is gone; flagship articles still counted here");
  assert.equal(run.stats.afterKeywords, 2, "the personal keyword filter drops p.2 only");

  // Anchors: one query embedding request per run
  const anchorCalls = http.calls.filter((c) => c.url.includes("embeddings") && JSON.parse(c.body).input[0] === "mantle subduction isotope");
  assert.equal(anchorCalls.length, 1);
  assert.deepEqual(JSON.parse(anchorCalls[0].body).input, ["mantle subduction isotope", "battery", "galaxy"]);

  assert.deepEqual(run.stats.flagship, { candidates: 8, withAbstract: 7, recent: 6, accepted: 2, gray: 2, judged: 0, rejected: 2, recommended: 4 });
  assert.equal(run.stats.newByLabel.flagship, 4);
  assert.ok(stages.some((p) => p.stage === "flagship" && p.message === "顶刊地学：判断 6 篇文章是否属于地学…"));
  assert.match(stages.at(-1).message, /，顶刊地学 4 篇$/);

  const works = (await engine().loadInbox()).works;
  const flagship = Object.values(works).filter((w) => w.label === "flagship");
  assert.deepEqual(flagship.map((w) => w.id).sort(), ["doi:10.1038/a", "doi:10.1038/c", "doi:10.1038/c2", "doi:10.1038/g"]);
  const a = works["doi:10.1038/a"];
  assert.equal(a.venue, "Nature");
  assert.equal(a.impactFactor, 50.5, "scored with the journal whitelist");
  assert.ok(a.score > 0 && a.similarity > 0);
  assert.deepEqual(Object.keys(a.gate), ["pos", "neg", "judged"]);
  assert.ok(a.gate.pos > 0.9 && a.gate.neg < 0.05 && a.gate.judged === false);
  assert.equal(a.gate.pos, Math.round(a.gate.pos * 1000) / 1000, "rounded to 3 decimals");
  assert.equal(a.vector, undefined);
  assert.ok(works["doi:10.1038/c"].gate.pos < 0.5, "gray articles are kept when no LLM judges them");
  // Excluded keyword ("dynamics") and similarity gate do not apply to the flagship track
  assert.ok(!works["doi:10.7777/p.2"]);
  // The battery paper fails the personal gate; the flagship battery/galaxy ones fail the anchors
  for (const id of ["doi:10.7777/p.3", "doi:10.1038/b", "doi:10.1038/d", "doi:10.1038/e", "doi:10.1038/f", "doi:10.9999/lib.mantle.1"]) {
    assert.ok(!works[id], id);
  }
  assert.ok(Object.values(works).every((w) => w.label === "flagship" || !w.issns?.includes("0028-0836")), "flagship articles never become personal");
  assert.ok(works["doi:10.7777/p.1"].label !== "flagship");
});

test("flagship articles without an abstract go through the shared abstract pass first", async () => {
  const { engine, http, state } = setup();
  const text = "Mantle subduction and isotope evidence from arc lavas shows how the deep mantle is recycled over time";
  const index = {};
  text.split(" ").forEach((word, i) => (index[word] ||= []).push(i));
  state.openalex = [{ doi: "https://doi.org/10.1038/e", abstract_inverted_index: index }];
  const run = await engine().run();
  assert.deepEqual(run.errors, []);
  const lookup = http.calls.find((c) => c.url.startsWith("https://api.openalex.org/works"));
  assert.match(decodeURIComponent(lookup.url), /doi:https:\/\/doi\.org\/10\.1038\/e/);
  assert.equal(run.stats.abstracts.openalex, 1);
  assert.equal(run.stats.flagship.withAbstract, 8);
  const work = (await engine().loadInbox()).works["doi:10.1038/e"];
  assert.equal(work.label, "flagship");
  assert.equal(work.abstract, text);
});

test("the newest flagship articles are kept up to the cap", async () => {
  const { engine } = setup({ prefs: { "flagship.maxResults": "3" } });
  const run = await engine().run();
  assert.equal(run.stats.flagship.recommended, 3);
  const ids = Object.values((await engine().loadInbox()).works)
    .filter((w) => w.label === "flagship")
    .map((w) => w.id)
    .sort();
  assert.deepEqual(ids, ["doi:10.1038/a", "doi:10.1038/c", "doi:10.1038/g"], "the oldest (c2) is dropped");
});

test("the LLM judges the gray zone with numbered papers and the flagship works join the summary budget", async () => {
  const judged = [];
  const llm = (req) => {
    const prompt = promptOf(req);
    if (prompt.startsWith("You are an academic research assistant.")) {
      judged.push(prompt);
      // P1 is "Copper sulfur deposits" (candidate order), P2 "Copper sulfur ores"
      return reply('{"relevant_ids": ["P1"], "irrelevant_ids": ["P2"]}');
    }
    return reply('{"research_question":"问题","methodology":"方法","key_findings":"发现","innovation":"创新","relevance_note":"相关"}');
  };
  const { engine } = setup({ prefs: { ...llmPrefs, "llm.summaryTopN": "2", "flagship.llmBoundary": "ONLY copper deposits." }, llm });
  const run = await engine().run();
  assert.deepEqual(run.errors, []);
  assert.equal(judged.length, 1);
  assert.ok(
    judged[0].includes(
      "User research interests:\nONLY copper deposits.\n\nPapers:\n- P1\n  Title: Copper sulfur deposits\n  Abstract: Copper copper sulfur mantle\n- P2\n  Title: Copper sulfur ores",
    ),
  );
  assert.deepEqual(run.stats.flagship, { candidates: 8, withAbstract: 7, recent: 6, accepted: 2, gray: 2, judged: 2, rejected: 3, recommended: 3 });

  const works = (await engine().loadInbox()).works;
  assert.equal(works["doi:10.1038/c"].gate.judged, true);
  assert.equal(works["doi:10.1038/a"].gate.judged, false, "accepted by the anchors alone");
  assert.ok(!works["doi:10.1038/c2"], "rejected by the LLM");
  // Summaries: the 2 best personal works plus the 2 newest flagship works
  const summarized = Object.values(works).filter((w) => w.summary);
  assert.deepEqual(summarized.filter((w) => w.label === "flagship").map((w) => w.id).sort(), ["doi:10.1038/c", "doi:10.1038/g"]);
});

test("the gray zone is accepted when the LLM check is off or the LLM is not configured", async () => {
  let calls = 0;
  const llm = () => {
    calls++;
    return reply('{"relevant_ids": []}');
  };
  const { engine } = setup({ prefs: { ...llmPrefs, "flagship.llmFallback": false }, llm });
  const run = await engine().run();
  assert.equal(run.stats.flagship.recommended, 4);
  assert.equal(calls, 0);
  const noKey = await engine({ "llm.apiKey": "", "flagship.llmFallback": true }).run();
  assert.equal(noKey.stats.flagship.judged, 0);
  assert.equal(noKey.stats.flagship.recommended, 4, "the gray zone is accepted");
  // (the summary step reports the missing key at stage "llm"; the flagship track itself does not fail)
  assert.ok(!noKey.errors.some((e) => e.stage === "flagship"));
  assert.equal(calls, 0);
});

test("flagship failures are reported without failing the run", async () => {
  // LLM rejects the key: gray articles are kept, one error
  const { engine, state } = setup({ prefs: llmPrefs, llm: () => ({ status: 401, data: { error: { message: "invalid key" } } }) });
  const run = await engine().run();
  assert.equal(run.stats.flagship.recommended, 4);
  assert.deepEqual(run.errors, [{ stage: "flagship", message: "大模型复核失败：HTTP 401 for https://llm.example/v1/chat/completions: invalid key" }]);
  assert.ok(run.stats.recommended > 0);

  // The flagship Crossref query fails: personal results still arrive
  state.flagshipHandler = () => ({ status: 400, data: { message: "bad filter" } });
  const run2 = await engine().run();
  assert.deepEqual(run2.errors.map((e) => [e.stage, e.source]), [["fetch", "Crossref 顶刊"]]);
  assert.deepEqual(run2.stats.perSource, { Crossref: 3 });
  assert.equal(run2.stats.flagship.candidates, 0);
  assert.ok(run2.stats.recommended > 0);

  // The anchor embedding fails: the track is skipped, the run succeeds
  const { engine: broken } = setup({
    embeddings: (req) => {
      const body = JSON.parse(req.body);
      if (body.input[0] === "mantle subduction isotope") return { status: 400, data: { message: "anchor rejected" } };
      return embeddingHandler({})(req);
    },
  });
  const run3 = await broken().run();
  assert.equal(run3.errors.length, 1);
  assert.equal(run3.errors[0].stage, "flagship");
  assert.match(run3.errors[0].message, /anchor rejected/);
  assert.equal(run3.stats.flagship.recommended, 0);
  assert.equal(run3.stats.newByLabel.flagship, 0);
  assert.ok(run3.stats.recommended > 0);
});

test("cancelling during the flagship stage stops the run before anything is stored", async () => {
  const { engine } = setup();
  let cancelled = false;
  await assert.rejects(
    engine().run({
      onProgress: (p) => {
        if (p.stage === "flagship") cancelled = true;
      },
      isCancelled: () => cancelled,
    }),
    CancelledError,
  );
  assert.deepEqual((await engine().loadInbox()).works, {});

  // A cancelled LLM request is a cancel, not an LLM failure
  const { engine: llmEngine } = setup({
    prefs: llmPrefs,
    llm: () => {
      throw new CancelledRequestError();
    },
  });
  await assert.rejects(llmEngine().run(), (e) => e.cancelled === true);
  assert.deepEqual((await llmEngine().loadInbox()).works, {});
});

test("flagship works keep their status across runs and dismissed ones are not shown again", async () => {
  const { engine, http } = setup();
  await engine().run();
  await engine().updateWork("doi:10.1038/a", { status: "dismissed" });
  // A work first stored as a personal recommendation becomes a flagship work
  // once its journal is on the flagship list
  await engine().withInbox((inbox) => {
    inbox.works["doi:10.1038/g"].label = "consider";
  });
  const embeddedBefore = http.calls.filter((c) => c.url.includes("embeddings")).length;
  const run = await engine().run();
  assert.equal(run.stats.newByLabel.flagship, 0);
  assert.equal(run.stats.flagship.candidates, 7, "the dismissed article is dropped before dedupe");
  const works = (await engine().loadInbox()).works;
  assert.equal(works["doi:10.1038/a"].status, "dismissed");
  assert.equal(works["doi:10.1038/g"].label, "flagship");
  // Cached candidate vectors are reused: only the anchors are embedded again
  const embeddings = http.calls.filter((c) => c.url.includes("embeddings")).slice(embeddedBefore);
  assert.deepEqual(embeddings.map((c) => JSON.parse(c.body).input[0]), ["mantle subduction isotope"]);
});

test("in local mode the LLM check judges every article the lexical bands cannot place", async () => {
  // Typical flagship solid-earth articles without the anchor's words, and a materials one
  const ARTICLES = [
    nature("10.1038/q", "Supershear rupture of a continental earthquake doublet", "Seismic waveforms show that the rupture front outran shear waves along the strike-slip fault for two hundred kilometres beneath Anatolia.", "2024-10-08T10:00:00Z"),
    nature("10.1038/z", "Hadean zircons record early continental crust", "Oxygen and hafnium compositions of detrital zircon grains from the Jack Hills show that evolved crust and liquid water existed four billion years ago.", "2024-10-08T11:00:00Z"),
    nature("10.1038/m", "Ferroelectric switching in twisted thin films", "Condensed matter and materials physics: ferroelectrics, semiconductors, nanomaterials and thin films with topological and magnetic materials for crystal device structures.", "2024-10-08T12:00:00Z"),
  ];
  const judged = [];
  const llm = (req) => {
    const papers = [...promptOf(req).matchAll(/^- P(\d+)\n {2}Title: (.*)$/gm)];
    judged.push(...papers.map(([, , title]) => title));
    return reply(JSON.stringify({ relevant_ids: papers.filter(([, , title]) => /rupture|zircons/.test(title)).map(([, n]) => `P${n}`) }));
  };
  const local = {
    ...llmPrefs,
    "embedding.provider": "local",
    "embedding.apiKey": "",
    "scoring.minSimilarity": "",
    // The shipped defaults: anchors, boundary and the local bands
    "flagship.positiveAnchor": "",
    "flagship.negativeAnchors": "",
    "flagship.llmBoundary": "",
    "flagship.minScore": "",
    "flagship.grayLow": "",
  };
  const { engine, state } = setup({ prefs: local, llm });
  state.flagship = ARTICLES;
  const run = await engine().run();
  assert.deepEqual(run.errors, []);
  assert.deepEqual(judged, [ARTICLES[0].title[0], ARTICLES[1].title[0]], "everything but the clear materials article goes to the LLM");
  assert.deepEqual(run.stats.flagship, { candidates: 3, withAbstract: 3, recent: 3, accepted: 0, gray: 2, judged: 2, rejected: 1, recommended: 2 });
  const works = (await engine().loadInbox()).works;
  for (const id of ["doi:10.1038/q", "doi:10.1038/z"]) {
    assert.equal(works[id].label, "flagship", id);
    assert.equal(works[id].gate.judged, true, id);
  }
  assert.ok(!works["doi:10.1038/m"]);

  // Without the LLM check the lexical bands decide alone (hence the advice to use an API embedder)
  const bandsOnly = setup({ prefs: { ...local, "flagship.llmFallback": false } });
  bandsOnly.state.flagship = ARTICLES;
  assert.equal((await bandsOnly.engine().run()).stats.flagship.recommended, 0);

  // An LLM that cannot judge never lets the widened articles in unjudged
  const broken = setup({ prefs: local, llm: () => ({ status: 401, data: { error: { message: "invalid key" } } }) });
  broken.state.flagship = ARTICLES;
  const failed = await broken.engine().run();
  assert.equal(failed.stats.flagship.recommended, 0);
  assert.deepEqual(failed.errors.map((e) => e.stage), ["flagship"]);
});

test("the flagship track is off by default and works in local mode", async () => {
  const { engine, http } = setup({ prefs: { "flagship.enabled": false } });
  const run = await engine().run();
  assert.equal(run.stats.flagship, undefined);
  assert.equal(http.calls.filter((c) => c.url.startsWith("https://api.crossref.org/works")).length, 1);
  assert.deepEqual(run.stats.perSource, { Crossref: 12 }, "the whitelisted Nature ISSN is an ordinary journal");
  // Without the track, flagship-journal articles are ordinary candidates
  assert.equal(run.stats.afterDedupe, 11);

  const { engine: local, http: localHttp } = setup({
    prefs: {
      "embedding.provider": "local",
      "embedding.apiKey": "",
      "scoring.minSimilarity": "",
      "flagship.minScore": "",
      "flagship.grayLow": "",
      "flagship.positiveAnchor": "Mantle subduction and isotope geochemistry of arc magmas.",
    },
  });
  const localRun = await local().run();
  assert.deepEqual(localRun.errors, []);
  assert.ok(!localHttp.calls.some((c) => c.url.includes("embeddings")));
  const flagship = Object.values((await local().loadInbox()).works).filter((w) => w.label === "flagship");
  assert.ok(flagship.length >= 1);
  assert.ok(flagship.every((w) => typeof w.gate.pos === "number" && Number.isFinite(w.score)));
  assert.ok(flagship.some((w) => w.id === "doi:10.1038/a"));
});
