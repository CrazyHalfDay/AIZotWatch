import assert from "node:assert/strict";
import test from "node:test";

import { CancelledError, HISTORY_LIMIT, ZotWatchEngine, appendHistory, historyEntry } from "../src/engine.js";
import { createEmbedder } from "../src/embedding/providers.js";
import { createLlmClient, labelClusters, parseJsonReply, summarizePaper, translateTitles } from "../src/llm/llm.js";
import { loadSettings } from "../src/settings.js";
import { createMemoryStorage } from "../src/store.js";
import { embeddingHandler, mockHttp, noSleep } from "./helpers.js";

const NOW = Date.UTC(2024, 9, 10, 12);

function libraryItems() {
  const items = [];
  const topics = [
    ["mantle", "Mantle plume dynamics and mantle convection", "Mantle mantle upwelling"],
    ["subduction", "Subduction zone fluids", "Subduction subduction slab dehydration"],
    ["copper", "Copper isotope systematics", "Copper isotope copper fractionation"],
  ];
  topics.forEach(([key, title, abstract]) => {
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
  });
  return items;
}

function crossrefWork(n, title, abstract, created = "2024-10-08T10:00:00Z") {
  return {
    DOI: `10.7777/cand.${n}`,
    title: [title],
    author: [{ given: "Ann", family: "Lee" }],
    abstract: abstract ? `<jats:p>${abstract}</jats:p>` : undefined,
    "container-title": ["Geochimica et Cosmochimica Acta"],
    created: { "date-time": created },
    URL: `https://doi.org/10.7777/cand.${n}`,
    type: "journal-article",
    ISSN: ["0016-7037"],
  };
}

const CANDIDATES = [
  crossrefWork(1, "Copper isotope fractionation in arc magmas", "Copper copper isotope isotope"),
  crossrefWork(2, "Mantle convection revisited", "Mantle mantle"),
  crossrefWork(3, "Battery cathode chemistry", "Battery battery cathode"),
  crossrefWork(4, "Neuron firing patterns", "Neuron neuron"),
  crossrefWork(5, "Already in library", "Copper"),
  crossrefWork(6, "Old copper paper", "Copper isotope", "2024-08-01T00:00:00Z"),
  crossrefWork(7, "Copper without abstract", null),
];
CANDIDATES[4].DOI = "10.9999/lib.copper.0";

function setup({ prefs = {}, candidates = CANDIDATES, llm } = {}) {
  const counter = {};
  const routes = [
    ["https://api.crossref.org/works", () => ({ message: { "total-results": candidates.length, items: candidates } })],
    ["https://api.openalex.org/works", () => ({ results: [] })],
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)],
  ];
  if (llm) routes.push(["https://llm.example/v1/chat/completions", llm]);
  const http = mockHttp(routes);
  const allPrefs = {
    "embedding.provider": "dashscope",
    "embedding.apiKey": "sk-test",
    "journals.autoFromLibrary": true,
    "scoring.minSimilarity": "0.3",
    ...prefs,
  };
  const settings = loadSettings((k) => allPrefs[k]);
  const storage = createMemoryStorage();
  let items = libraryItems();
  const library = { getItems: async () => items };
  const engine = new ZotWatchEngine({ http, storage, library, settings, now: () => NOW, retry: noSleep });
  return { engine, http, storage, counter, setItems: (v) => (items = v), getItems: () => items };
}

test("full run ranks relevant papers and fills the inbox", async () => {
  const { engine, http, counter } = setup();
  const stages = [];
  const run = await engine.run({ onProgress: (p) => stages.push(p.stage) });

  assert.equal(run.stats.libraryItems, 18);
  assert.equal(run.stats.profile.embedded, 18);
  assert.ok(run.stats.profile.clusters >= 1);
  assert.equal(run.stats.journals, 1);
  assert.equal(run.stats.fetched, 7);
  assert.equal(run.stats.afterDedupe, 6);
  assert.equal(run.stats.afterAbstract, 5);
  assert.deepEqual(run.errors, []);
  assert.ok(stages.includes("fetch") && stages.at(-1) === "done");

  const crossrefUrl = decodeURIComponent(http.calls.find((c) => c.url.startsWith("https://api.crossref.org/works")).url);
  assert.match(crossrefUrl, /issn:0016-7037/);

  const inbox = await engine.loadInbox();
  const ids = Object.keys(inbox.works);
  // Off-topic papers are gated out, old papers are dropped by the recency filter
  assert.ok(ids.includes("doi:10.7777/cand.1"));
  assert.ok(ids.includes("doi:10.7777/cand.2"));
  assert.ok(!ids.includes("doi:10.7777/cand.3"));
  assert.ok(!ids.includes("doi:10.7777/cand.4"));
  assert.ok(!ids.includes("doi:10.7777/cand.6"));
  const top = inbox.works["doi:10.7777/cand.1"];
  assert.equal(top.status, "new");
  assert.equal(top.vector, undefined);
  assert.ok(top.score > 0 && top.label);
  assert.equal(inbox.lastRun.runId, run.runId);
  assert.ok(counter.texts >= 18 + 5);
});

test("second run reuses cached embeddings and skips dismissed works", async () => {
  const { engine, counter } = setup();
  await engine.run();
  const firstTexts = counter.texts;

  await engine.updateWork("doi:10.7777/cand.2", { status: "dismissed" });
  const run2 = await engine.run();
  assert.equal(counter.texts, firstTexts, "nothing new should be embedded");
  assert.equal(run2.stats.profile.reused, 18);
  assert.equal(run2.stats.profile.clustersRebuilt, false);
  const inbox = await engine.loadInbox();
  assert.equal(inbox.works["doi:10.7777/cand.2"].status, "dismissed");
  assert.equal(run2.stats.newRecommendations, 0);
});

test("profile updates incrementally when the library changes", async () => {
  const { engine, counter, setItems, getItems } = setup();
  await engine.run();
  const before = counter.texts;
  const items = getItems().slice(1); // one removed
  items[0] = { ...items[0], abstract: "Mantle changed abstract" }; // one changed
  items.push({ ...items[1], key: "NEW1", title: "Sulfur degassing", abstract: "sulfur sulfur" }); // one added
  setItems(items);
  const run = await engine.run();
  assert.equal(run.stats.profile.embedded, 2);
  assert.equal(run.stats.profile.reused, 16);
  assert.equal(run.stats.profile.clustersRebuilt, true);
  assert.equal(counter.texts, before + 2);
});

test("inbox marks works as added once they appear in the library", async () => {
  const { engine, setItems, getItems } = setup();
  await engine.run();
  setItems([...getItems(), { key: "X", itemType: "journalArticle", title: "Copper isotope fractionation in arc magmas", abstract: "x", doi: "10.7777/cand.1" }]);
  await engine.run();
  const inbox = await engine.loadInbox();
  assert.equal(inbox.works["doi:10.7777/cand.1"].status, "added");
});

test("source failures are reported without failing the run", async () => {
  const { engine, http } = setup({ prefs: { "sources.arxiv.enabled": true, "sources.arxiv.categories": "physics.geo-ph" } });
  const run = await engine.run();
  assert.equal(run.errors.length, 1);
  assert.equal(run.errors[0].source, "arXiv");
  assert.ok(run.stats.recommended > 0);
  assert.ok(http.calls.some((c) => c.url.startsWith("https://export.arxiv.org")));
});

test("finished runs are recorded in the inbox history", async () => {
  const { engine, storage } = setup();
  let clock = NOW;
  engine.now = () => (clock += 1000);
  // A missing or damaged history is tolerated
  assert.deepEqual((await engine.loadInbox()).history, []);
  await storage.writeJSON("inbox.json", { version: 1, works: {}, lastRun: null, history: [{ runAt: "x" }, null, 7] });
  assert.deepEqual((await engine.loadInbox()).history, []);

  const first = await engine.run();
  const second = await engine.run();
  const { history } = await engine.loadInbox();
  assert.equal(history.length, 2);
  assert.deepEqual(history[0], {
    runId: first.runId,
    runAt: first.runAt,
    finishedAt: first.finishedAt,
    fetched: 7,
    recommended: first.stats.recommended,
    newByLabel: first.stats.newByLabel,
    newMustRead: first.stats.newMustRead,
    errors: 0,
  });
  assert.equal(history[1].runId, second.runId);
  assert.equal(history[1].newByLabel.personal, 0, "nothing new the second time");

  // Cancelled runs record nothing; the history keeps the newest HISTORY_LIMIT runs
  await assert.rejects(engine.run({ isCancelled: () => true }), CancelledError);
  assert.equal((await engine.loadInbox()).history.length, 2);
  let list = [];
  for (let i = 0; i < HISTORY_LIMIT + 5; i++) list = appendHistory(list, historyEntry({ runId: `r${i}`, runAt: i, stats: {}, errors: [{}] }));
  assert.equal(list.length, HISTORY_LIMIT);
  assert.equal(list[0].runId, "r5");
  assert.deepEqual(list.at(-1), { runId: `r${HISTORY_LIMIT + 4}`, runAt: HISTORY_LIMIT + 4, finishedAt: null, fetched: 0, recommended: 0, newByLabel: { personal: 0, flagship: 0, followed: 0 }, newMustRead: 0, errors: 1 });
  assert.equal(appendHistory(list, { ...list.at(-1), fetched: 9 }).filter((h) => h.runId === list.at(-1).runId).length, 1, "one entry per run");
});

test("cancellation stops the run", async () => {
  const { engine } = setup();
  await assert.rejects(engine.run({ isCancelled: () => true }), CancelledError);
});

test("local provider works without any API key", async () => {
  const { engine, http } = setup({ prefs: { "embedding.provider": "local", "embedding.apiKey": "", "scoring.minSimilarity": "" } });
  const run = await engine.run();
  assert.ok(!http.calls.some((c) => c.url.includes("embeddings")));
  assert.ok(run.stats.recommended >= 1);
  const inbox = await engine.loadInbox();
  const ranked = Object.values(inbox.works).sort((a, b) => b.score - a.score);
  assert.ok(["doi:10.7777/cand.1", "doi:10.7777/cand.2"].includes(ranked[0].id));
});

test("LLM translation and summaries are stored in the inbox", async () => {
  const llm = (req) => {
    const prompt = JSON.parse(req.body).messages[0].content;
    if (prompt.includes("研究方向标签")) {
      const n = (prompt.match(/^Cluster \d+ \(/gm) || []).length;
      return { choices: [{ message: { content: JSON.stringify(Array.from({ length: n }, (_, i) => `方向${i}`)) } }] };
    }
    if (prompt.includes("翻译")) {
      const n = (prompt.match(/^\d+\. /gm) || []).length;
      return { choices: [{ message: { content: "```json\n" + JSON.stringify(Array.from({ length: n }, (_, i) => `标题${i}`)) + "\n```" } }] };
    }
    return { choices: [{ message: { content: '好的：{"research_question":"问题","methodology":"方法","key_findings":"发现","innovation":"创新","relevance_note":"相关"}' } }] };
  };
  const { engine } = setup({
    prefs: { "llm.enabled": true, "llm.apiKey": "k", "llm.baseURL": "https://llm.example/v1", "llm.model": "m", "llm.summaryTopN": "1" },
    llm,
  });
  const run = await engine.run();
  assert.deepEqual(run.errors, []);
  const works = Object.values((await engine.loadInbox()).works);
  assert.ok(works.every((w) => w.translatedTitle?.startsWith("标题")));
  assert.ok(works.every((w) => w.topic?.startsWith("方向")), "every work gets its research direction");
  assert.equal(works.filter((w) => w.summary).length, 1);
  const best = works.sort((a, b) => b.score - a.score)[0];
  assert.equal(best.summary.methodology, "方法");
});

test("user actions during a run are not overwritten by the run", async () => {
  let releaseLlm;
  const llmGate = new Promise((resolve) => (releaseLlm = resolve));
  let llmCalls = 0;
  const llm = async (req) => {
    llmCalls++;
    await llmGate;
    const prompt = JSON.parse(req.body).messages[0].content;
    if (prompt.includes("翻译")) return { choices: [{ message: { content: '["标题"]' } }] };
    return { choices: [{ message: { content: '{"research_question":"q"}' } }] };
  };
  const { engine } = setup({
    prefs: { "llm.enabled": true, "llm.apiKey": "k", "llm.baseURL": "https://llm.example/v1", "llm.model": "m", "llm.summaryTopN": "5" },
    llm,
  });
  const running = engine.run();
  // Wait until the run has merged its results and is blocked in the LLM step
  while (llmCalls === 0) await new Promise((r) => setTimeout(r, 5));
  const inbox = await engine.loadInbox();
  const target = Object.keys(inbox.works)[0];
  await engine.updateWork(target, { status: "dismissed" });
  releaseLlm();
  await running;
  const after = await engine.loadInbox();
  assert.equal(after.works[target].status, "dismissed");
  assert.ok(after.lastRun.finishedAt);
});

test("a cancel during best-effort steps is reported as a cancel", async () => {
  const { CancelledRequestError } = await import("../src/core/http.js");
  // Journal lookup: venues without ISSN trigger Crossref /journals requests
  const items = libraryItems().map((i) => ({ ...i, issn: "" }));
  let lookups = 0;
  const http = mockHttp([
    ["https://api.crossref.org/journals", () => {
      lookups++;
      throw new CancelledRequestError();
    }],
  ]);
  const settings = loadSettings((k) => ({ "embedding.provider": "local" })[k]);
  const engine = new ZotWatchEngine({ http, storage: createMemoryStorage(), library: { getItems: async () => items }, settings, now: () => NOW, retry: noSleep });
  await assert.rejects(engine.resolveJournals(items), (e) => e.cancelled === true);
  assert.equal(lookups, 1, "no further lookups after the cancel");

  // First AI summary cancelled: the run must not finish as a success
  const llm = () => {
    throw new CancelledRequestError();
  };
  // (topic labels off: their request would come first and take the cancel)
  const { engine: llmEngine } = setup({
    prefs: { "llm.enabled": true, "llm.apiKey": "k", "llm.baseURL": "https://llm.example/v1", "llm.model": "m", "llm.translateTitles": false, "llm.topicLabels": false },
    llm,
  });
  await assert.rejects(llmEngine.run(), (e) => e.cancelled === true);

  // Title translation cancelled
  const { engine: trEngine } = setup({
    prefs: { "llm.enabled": true, "llm.apiKey": "k", "llm.baseURL": "https://llm.example/v1", "llm.model": "m", "llm.summaryTopN": "0", "llm.topicLabels": false },
    llm,
  });
  await assert.rejects(trEngine.run(), (e) => e.cancelled === true);
});

test("LLM helpers", async () => {
  assert.deepEqual(parseJsonReply('<think>hmm {"a":1}</think>Result: {"a": "x}y", "b": [1,2]} trailing'), { a: "x}y", b: [1, 2] });
  assert.deepEqual(parseJsonReply("```json\n[\"a\"]\n```"), ["a"]);
  assert.throws(() => parseJsonReply("no json here"));

  const failing = { chat: async () => "not json" };
  assert.deepEqual(await translateTitles(failing, ["a", "b"]), [null, null]);
  await assert.rejects(summarizePaper(failing, { title: "t", authors: [] }));
});

test("reasoning models get room for their reasoning tokens", async () => {
  const http = mockHttp([["https://llm.example/v1/chat/completions", () => ({ choices: [{ message: { content: '["方向"]' } }] })]]);
  const bodyFor = async (model) => {
    const llm = createLlmClient({ apiKey: "k", baseURL: "https://llm.example/v1", model, temperature: 0.3 }, http);
    await labelClusters(llm, [{ memberCount: 1 }]); // asks for 1024 tokens
    return JSON.parse(http.calls.at(-1).body);
  };
  const plain = await bodyFor("deepseek-chat");
  assert.equal(plain.max_tokens, 1024);
  assert.equal(plain.temperature, 0.3);
  const reasoner = await bodyFor("deepseek-reasoner");
  assert.equal(reasoner.max_tokens, 8192);
  assert.equal(reasoner.temperature, undefined);
  assert.equal((await bodyFor("deepseek/deepseek-r1")).max_tokens, 8192);
  assert.equal((await bodyFor("kimi-k2-thinking-turbo")).max_tokens, 16000);
});

test("embedding providers send the right requests", async () => {
  const counter = {};
  const http = mockHttp([
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)],
    ["https://api.voyageai.com/v1/embeddings", embeddingHandler(counter)],
  ]);
  const ds = createEmbedder({ provider: "dashscope", apiKey: "k", dimensions: 1024 }, http);
  const vectors = await ds.embed(Array.from({ length: 23 }, (_, i) => (i % 2 ? "mantle" : "copper")));
  assert.equal(vectors.length, 23);
  assert.equal(counter.batches, 3);
  const body = JSON.parse(http.calls[0].body);
  assert.equal(body.model, "text-embedding-v4");
  assert.equal(body.dimensions, 1024);
  assert.equal(http.calls[0].headers.Authorization, "Bearer k");
  // Responses arrive in reverse order; vectors must follow input order
  assert.ok(vectors[0][2] > 0.9 && vectors[1][0] > 0.9);
  assert.equal(ds.signature, "dashscope:text-embedding-v4:1024");

  const voyage = createEmbedder({ provider: "voyage", apiKey: "v" }, http);
  await voyage.embed(["copper"]);
  const vBody = JSON.parse(http.calls.at(-1).body);
  assert.equal(vBody.input_type, "document");
  assert.equal(vBody.dimensions, undefined);

  assert.throws(() => createEmbedder({ provider: "voyage", apiKey: "" }, http), /API Key/);
  const bad = mockHttp([["https://dashscope", () => ({ data: [] })]]);
  await assert.rejects(createEmbedder({ provider: "dashscope", apiKey: "k" }, bad).embed(["x"]), /expected 1/);
});
