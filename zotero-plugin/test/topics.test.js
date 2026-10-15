// Research-direction topic labels (聚类主题标签): keyword tags, cluster
// metadata, label carry-over, the batch label prompt and its strict parser,
// the nearest-cluster topic per work, and the engine integration.

import assert from "node:assert/strict";
import test from "node:test";

import { CancelledRequestError } from "../src/core/http.js";
import { ProfileRanker, buildClusters } from "../src/core/scoring.js";
import {
  isKeywordTag,
  jaccard,
  labelStoreOf,
  matchStoredLabels,
  sameLabelStore,
  storedLabelEntries,
  topKeywords,
  topicsOf,
} from "../src/core/topics.js";
import { Matrix } from "../src/core/vector.js";
import { CancelledError, ZotWatchEngine } from "../src/engine.js";
import { clusterLabelPrompt, labelClusters, parseClusterLabels } from "../src/llm/llm.js";
import { loadSettings } from "../src/settings.js";
import { createMemoryStorage } from "../src/store.js";
import { topicsStatusLine } from "../src/zotero/addon.js";
import { summaryNoteHtml } from "../src/zotero/items.js";
import { embeddingHandler, mockHttp, noSleep, topicVector } from "./helpers.js";

const NOW = Date.UTC(2024, 9, 10, 12);
const LLM_URL = "https://llm.example/v1/chat/completions";

// ------------------------------------------------------------------ units

test("keyword tags exclude the output tag, workflow tags and marker symbols", () => {
  for (const tag of ["Copper isotopes", "俯冲带", "LIP", "3D", " Geodynamics "]) assert.equal(isKeywordTag(tag, "ZotWatch"), true, tag);
  for (const tag of ["ZotWatch", "zotwatch ", "/unread", "/done", "_todo", "⭐", "⭐⭐⭐⭐⭐", "✔️", "⏳", "💎", "", "  ", null]) {
    assert.equal(isKeywordTag(tag, "ZotWatch"), false, String(tag));
  }
  assert.equal(isKeywordTag("ZotWatch", ""), true, "without an output tag nothing is special");

  const items = [
    { tags: ["b", "a", "⭐", "a"] }, // counted once per item
    { tags: ["a", "c", "ZotWatch"] },
    { tags: [" c ", "b", "/reading"] },
    { tags: undefined },
    { tags: Array.from({ length: 12 }, (_, i) => `t${i}`) },
  ];
  // a, b and c twice each (first-seen order breaks ties), then single tags
  assert.deepEqual(topKeywords(items, { outputTag: "ZotWatch" }), ["b", "a", "c", "t0", "t1", "t2", "t3", "t4", "t5", "t6"]);
  assert.deepEqual(topKeywords(items, { outputTag: "ZotWatch", limit: 2 }), ["b", "a"]);
});

test("stored labels follow clusters by member-key Jaccard, one-to-one", () => {
  assert.equal(jaccard(["a", "b"], ["b", "c"]), 1 / 3);
  assert.equal(jaccard([], []), 0);

  const stored = [
    { keys: ["a1", "a2", "a3", "a4"], label: "铜同位素" },
    { keys: ["b1", "b2"], label: "地幔" },
    { keys: ["c1", "c2", "c3"], label: "俯冲带" },
  ];
  const clusters = [
    { memberKeys: ["a1", "a2", "a3", "a4", "a5"] }, // 4/5: carried over
    { memberKeys: ["b1", "x1", "x2"] }, // 1/4: below 0.5
    { memberKeys: ["c1", "c2"] }, // 2/3
    { memberKeys: ["c3", "c2", "c1", "y"] }, // 3/4 wins c over the cluster above
    { titles: ["no member keys"] },
  ];
  assert.deepEqual(matchStoredLabels(clusters, stored), ["铜同位素", null, null, "俯冲带", null]);
  assert.deepEqual(matchStoredLabels(clusters, []), [null, null, null, null, null]);

  // Validation of the stored file
  const data = {
    version: 1,
    clusters: [...stored, { keys: [], label: "空" }, { keys: ["z"], label: " " }, { keys: [1], label: "数字" }, null, { label: "无成员" }],
  };
  assert.deepEqual(storedLabelEntries(data), stored);
  assert.deepEqual(storedLabelEntries({ clusters: "garbage" }), []);
  assert.deepEqual(storedLabelEntries(null), []);

  const store = labelStoreOf(clusters, ["铜同位素", null, "x", "俯冲带", "无键"]);
  assert.deepEqual(store, {
    version: 1,
    clusters: [
      { keys: ["a1", "a2", "a3", "a4", "a5"], label: "铜同位素" },
      { keys: ["c1", "c2"], label: "x" },
      { keys: ["c3", "c2", "c1", "y"], label: "俯冲带" },
    ],
  });
  assert.ok(sameLabelStore(store, JSON.parse(JSON.stringify(store))));
  assert.ok(!sameLabelStore(store, { clusters: store.clusters.slice(1) }));
  assert.ok(!sameLabelStore(null, store));
  assert.deepEqual(topicsOf(store), [
    { label: "铜同位素", size: 5 },
    { label: "俯冲带", size: 4 },
    { label: "x", size: 2 },
  ]);
});

test("the batch label prompt is the Python prompt verbatim", () => {
  const prompt = clusterLabelPrompt([
    { memberCount: 12, titles: ["Copper A", "Copper B", "Copper C", "Copper D"], keywords: ["k1", "k2", "k3", "k4", "k5", "k6"] },
    { memberCount: 3, titles: ["Mantle A"], keywords: [] },
  ]);
  assert.equal(
    prompt,
    `基于以下多个论文簇，为每个簇生成一个简洁的研究方向标签（仅中文，不超过 10 个词）。

## Clusters:
Cluster 1 (12 papers):
  Titles: "Copper A", "Copper B", "Copper C"
  Keywords: k1, k2, k3, k4, k5

Cluster 2 (3 papers):
  Titles: "Mantle A"
  Keywords: N/A

## Instructions:
1. 为每个簇识别共同的研究主题或方向
2. 用中文生成能概括每个簇核心的简洁标签
3. 适当使用领域术语，保持自然流畅；常用英文缩写可直接使用（如 LLM, SAR 等）
4. 按输入顺序返回 JSON 数组，元素为中文标签字符串

## Output（JSON array of strings, 中文标签）:
`,
  );
});

test("label replies must be a JSON array of exactly N non-empty strings", async () => {
  assert.deepEqual(parseClusterLabels('<think>先想想</think>\n```json\n["铜同位素", "\\"地幔柱\\"", " “俯冲带” "]\n```', 3), ["铜同位素", "地幔柱", "俯冲带"]);
  assert.deepEqual(parseClusterLabels('标签如下：["SAR 干涉\\n测量"]', 1), ["SAR 干涉 测量"]);
  const long = "很".repeat(40);
  assert.equal(parseClusterLabels(JSON.stringify([long]), 1)[0], "很".repeat(30), "capped at 30 characters");

  assert.throws(() => parseClusterLabels('["a", "b"]', 3), /大模型返回了 2 个标签，应为 3 个/);
  assert.throws(() => parseClusterLabels('{"labels": ["a"]}', 1), /不是 JSON 数组/);
  assert.throws(() => parseClusterLabels('["a", ""]', 2), /第 2 个标签为空/);
  assert.throws(() => parseClusterLabels('["a", null]', 2), /第 2 个标签为空/);
  assert.throws(() => parseClusterLabels('["a", 3]', 2), /第 2 个标签为空/);
  assert.throws(() => parseClusterLabels("1. 铜同位素\n2. 地幔柱", 2), /无法解析大模型的回复/);
  assert.throws(() => parseClusterLabels('["a", "b"', 2), /无法解析大模型的回复/);

  const prompts = [];
  const llm = {
    chat: async (prompt, options) => {
      prompts.push([prompt, options]);
      return '["一", "二"]';
    },
  };
  assert.deepEqual(await labelClusters(llm, [{ memberCount: 1 }, { memberCount: 2 }]), ["一", "二"]);
  assert.deepEqual(prompts[0][1], { maxTokens: 1024 });
  assert.deepEqual(await labelClusters(llm, []), []);
  assert.equal(prompts.length, 1, "nothing to label: no call");
  const cancelled = { chat: async () => Promise.reject(new CancelledRequestError()) };
  await assert.rejects(labelClusters(cancelled, [{ memberCount: 1 }]), (e) => e.cancelled === true);
});

test("clusters carry member keys, representative titles and keywords, largest first", async () => {
  const items = [];
  const rows = [];
  const add = (key, title, text, tags) => {
    items.push({ key, title, tags });
    rows.push(topicVector(text));
  };
  // Two copper items lean towards isotopes: they are less central than the plain copper ones
  for (let i = 0; i < 7; i++) add(`C${i}`, `Copper ${i}`, i < 5 ? "copper copper copper" : "copper copper isotope", ["Copper", "⭐", ...(i < 2 ? ["Porphyry"] : [])]);
  for (let i = 0; i < 3; i++) add(`M${i}`, `Mantle ${i}`, "mantle mantle", ["Mantle", "ZotWatch"]);
  for (let i = 0; i < 5; i++) add(`G${i}`, `Galaxy ${i}`, "galaxy galaxy", []);
  const matrix = Matrix.fromRows(rows);
  const weights = new Float32Array(rows.length).fill(1);

  const { clusters } = await buildClusters(matrix, weights, { minClusters: 2, maxClusters: 4, items, outputTag: "ZotWatch" });
  assert.deepEqual(clusters.map((c) => c.memberCount), [7, 5, 3], "sorted by size");
  const [copper, galaxy, mantle] = clusters;
  assert.deepEqual(copper.memberKeys, ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]);
  assert.equal(copper.titles.length, 5, "at most 5 titles");
  assert.ok(copper.titles.every((t) => /^Copper [0-4]$/.test(t)), "closest to the member mean first");
  assert.deepEqual(copper.keywords, ["Copper", "Porphyry"]);
  assert.deepEqual(mantle.keywords, ["Mantle"], "output tag excluded");
  assert.deepEqual(galaxy.keywords, []);
  assert.deepEqual(galaxy.titles, ["Galaxy 0", "Galaxy 1", "Galaxy 2", "Galaxy 3", "Galaxy 4"]);

  // Without items: only the scoring summary
  const plain = await buildClusters(matrix, weights, { minClusters: 2, maxClusters: 4 });
  assert.deepEqual(Object.keys(plain.clusters[0]).sort(), ["centroid", "effectiveSize", "memberCount"]);

  // A single cluster for tiny libraries still describes its members
  const tiny = await buildClusters(Matrix.fromRows(rows.slice(0, 3)), weights.slice(0, 3), { items: items.slice(0, 3) });
  assert.deepEqual(tiny.clusters[0].memberKeys, ["C0", "C1", "C2"]);
});

test("the ranker reports the nearest cluster next to the macro-best one", async () => {
  const settings = loadSettings(() => undefined);
  const big = { centroid: Float32Array.from([1, 0]), memberCount: 100, effectiveSize: 100 };
  const small = { centroid: Float32Array.from([0.6, 0.8]), memberCount: 1, effectiveSize: 1 };
  const matrix = Matrix.fromRows([
    [1, 0],
    [0.6, 0.8],
  ]);
  const ranker = new ProfileRanker({ matrix, weights: Float32Array.from([1, 1]), clusters: [big, small] }, new Map(), settings.scoring);
  const vec = Float32Array.from([0.8, 0.6]);
  const { cluster, nearest } = ranker.macroScore(vec);
  assert.equal(cluster, 0, "the big cluster wins the size-weighted macro score");
  assert.equal(nearest, 1, "but the small one is closer");
  const [scored] = await ranker.scoreWorks([{ id: "w", vector: vec }]);
  assert.equal(scored.nearest, 1);

  const empty = new ProfileRanker({ matrix: new Matrix(0, 2), weights: new Float32Array(0), clusters: [] }, new Map(), settings.scoring);
  assert.equal((await empty.scoreWorks([{ id: "w", vector: vec }]))[0].nearest, null);
});

test("topic display helpers", () => {
  const topics = Array.from({ length: 10 }, (_, i) => ({ label: `方向${i}`, size: 20 - i }));
  assert.equal(topicsStatusLine(topics.slice(0, 2)), "研究方向：方向0（20 篇）、方向1（19 篇）");
  assert.equal(
    topicsStatusLine(topics),
    "研究方向：方向0（20 篇）、方向1（19 篇）、方向2（18 篇）、方向3（17 篇）、方向4（16 篇）、方向5（15 篇）、方向6（14 篇）、方向7（13 篇） 等 10 个",
  );
  const note = summaryNoteHtml({ label: "consider", score: 0.5, similarity: 0.4, topic: "铜 <同位素>", summary: {} });
  assert.match(note, /<p>研究方向：铜 &lt;同位素&gt;<\/p>/);
});

// ----------------------------------------------------------------- engine

function libraryItems({ copper = 8, mantle = 5, subduction = 3, sulfur = 0 } = {}) {
  const items = [];
  const add = (prefix, n, title, abstract, tagsOf) => {
    for (let i = 0; i < n; i++) {
      items.push({
        key: `${prefix}${i}`,
        itemType: "journalArticle",
        title: `${title} ${i}`,
        abstract: `${abstract} part ${i}`,
        doi: `10.9999/lib.${prefix.toLowerCase()}.${i}`,
        issn: "0016-7037",
        venue: "Geochimica et Cosmochimica Acta",
        dateAdded: "2024-01-01 00:00:00",
        tags: tagsOf(i),
      });
    }
  };
  add("COPPER", copper, "Copper isotope systematics", "Copper isotope copper fractionation", (i) => [
    "Copper isotopes",
    "Porphyry",
    "⭐⭐⭐",
    "/unread",
    "ZotWatch",
    ...(i % 2 ? ["_todo", "✔️"] : []),
  ]);
  add("MANTLE", mantle, "Mantle plume dynamics", "Mantle mantle upwelling", () => ["地幔柱", "Geodynamics", "💎"]);
  add("SUBDUCTION", subduction, "Subduction zone fluids", "Subduction subduction slab dehydration", () => []);
  add("SULFUR", sulfur, "Sulfur degassing", "Sulfur sulfur volatiles", () => ["Sulfur"]);
  return items;
}

function crossrefWork(n, title, abstract, { issn = "0016-7037", venue = "Geochimica et Cosmochimica Acta", doi = `10.7777/cand.${n}` } = {}) {
  return {
    DOI: doi,
    title: [title],
    author: [{ given: "Ann", family: "Lee" }],
    abstract: `<jats:p>${abstract}</jats:p>`,
    "container-title": [venue],
    created: { "date-time": "2024-10-08T10:00:00Z" },
    URL: `https://doi.org/${doi}`,
    type: "journal-article",
    ISSN: [issn],
  };
}

const CANDIDATES = [
  crossrefWork(1, "Copper isotope fractionation in arc magmas", "Copper copper isotope isotope"),
  crossrefWork(2, "Mantle convection revisited", "Mantle mantle"),
  crossrefWork(3, "Battery cathode chemistry", "Battery battery cathode"),
];

const promptOf = (req) => JSON.parse(req.body).messages[0].content;
const reply = (content) => ({ choices: [{ message: { content } }] });
const LABELS = [
  ["Copper", "铜同位素"],
  ["Mantle", "地幔动力学"],
  ["Subduction", "俯冲带流体"],
  ["Sulfur", "硫循环"],
];

/** Label reply derived from each cluster's titles (one label per "Cluster n" block). */
function labelReply(prompt) {
  const labels = [...prompt.matchAll(/^Cluster \d+ \(\d+ papers\):\n {2}Titles: (.*)$/gm)].map(
    ([, titles]) => LABELS.find(([word]) => titles.includes(word))?.[1] || "其他",
  );
  return reply(JSON.stringify(labels));
}

const labelLlm = (req) => {
  const prompt = promptOf(req);
  if (prompt.includes("研究方向标签")) return labelReply(prompt);
  if (prompt.includes("翻译")) {
    const n = (prompt.match(/^\d+\. /gm) || []).length;
    return reply(JSON.stringify(Array.from({ length: n }, (_, i) => `标题${i}`)));
  }
  return reply('{"research_question":"问题"}');
};

function setup({ prefs = {}, items = libraryItems(), candidates = CANDIDATES, llm = labelLlm, followed = [] } = {}) {
  const counter = {};
  const state = { items, candidates, llm, followed };
  const http = mockHttp([
    ["https://api.crossref.org/works", () => ({ message: { "total-results": state.candidates.length, items: state.candidates } })],
    [(r) => r.url.startsWith("https://api.openalex.org/works") && decodeURIComponent(r.url).includes("author.orcid:"), () => ({ meta: { next_cursor: null }, results: state.followed })],
    ["https://api.openalex.org/works", () => ({ results: [] })],
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)],
    [LLM_URL, (req) => state.llm(req)],
  ]);
  const allPrefs = {
    "embedding.provider": "dashscope",
    "embedding.apiKey": "sk-test",
    "journals.autoFromLibrary": true,
    "scoring.minSimilarity": "0.3",
    "abstracts.crossref": false,
    "llm.enabled": true,
    "llm.apiKey": "k",
    "llm.baseURL": "https://llm.example/v1",
    "llm.model": "m",
    "llm.translateTitles": false,
    "llm.summaryTopN": "0",
    ...prefs,
  };
  const storage = createMemoryStorage();
  const engine = (overrides = {}) =>
    new ZotWatchEngine({
      http,
      storage,
      library: { getItems: async () => state.items },
      settings: loadSettings((k) => ({ ...allPrefs, ...overrides })[k]),
      now: () => NOW,
      retry: noSleep,
    });
  const llmCalls = () => http.calls.filter((c) => c.url === LLM_URL).map(promptOf);
  const labelCalls = () => llmCalls().filter((p) => p.includes("研究方向标签"));
  return { engine, http, storage, state, counter, llmCalls, labelCalls };
}

const topicsById = async (engine) =>
  Object.fromEntries(Object.values((await engine.loadInbox()).works).map((w) => [w.id, w.topic]));

test("research directions are labeled in one call, first, and every work shows its direction", async () => {
  const { engine, storage, llmCalls, labelCalls } = setup({ prefs: { "llm.translateTitles": true } });
  const stages = [];
  const run = await engine().run({ onProgress: (p) => stages.push(p) });
  assert.deepEqual(run.errors, []);
  assert.equal(run.stats.profile.clusters, 3);
  assert.deepEqual(run.stats.topics, { clusters: 3, labeled: 3, generated: 3 });

  // Labels come before the title translation, in one batch for every cluster
  assert.equal(labelCalls().length, 1);
  assert.equal(llmCalls()[0], labelCalls()[0]);
  assert.ok(llmCalls()[1].includes("翻译"));
  const labelStage = stages.findIndex((p) => p.stage === "llm" && p.message === "生成 3 个研究方向标签…");
  assert.ok(labelStage > 0 && labelStage < stages.findIndex((p) => p.message?.startsWith("翻译")));

  // Largest cluster first; 3 titles and 5 keywords at most; marker and output tags dropped
  const prompt = labelCalls()[0];
  assert.match(prompt, /Cluster 1 \(8 papers\):\n {2}Titles: "Copper isotope systematics \d", "Copper isotope systematics \d", "Copper isotope systematics \d"\n {2}Keywords: Copper isotopes, Porphyry\n/);
  assert.match(prompt, /Cluster 2 \(5 papers\):\n {2}Titles: "Mantle plume dynamics \d", .*\n {2}Keywords: 地幔柱, Geodynamics\n/);
  assert.match(prompt, /Cluster 3 \(3 papers\):\n {2}Titles: .*\n {2}Keywords: N\/A\n/);
  assert.doesNotMatch(prompt, /⭐|✔️|💎|\/unread|_todo|ZotWatch/);

  const store = await storage.readJSON("cluster-labels.json");
  assert.deepEqual(store, {
    version: 1,
    clusters: [
      { keys: ["COPPER0", "COPPER1", "COPPER2", "COPPER3", "COPPER4", "COPPER5", "COPPER6", "COPPER7"], label: "铜同位素" },
      { keys: ["MANTLE0", "MANTLE1", "MANTLE2", "MANTLE3", "MANTLE4"], label: "地幔动力学" },
      { keys: ["SUBDUCTION0", "SUBDUCTION1", "SUBDUCTION2"], label: "俯冲带流体" },
    ],
  });
  assert.deepEqual(await topicsById(engine()), { "doi:10.7777/cand.1": "铜同位素", "doi:10.7777/cand.2": "地幔动力学" });
  assert.deepEqual(await engine().getTopics(), [
    { label: "铜同位素", size: 8 },
    { label: "地幔动力学", size: 5 },
    { label: "俯冲带流体", size: 3 },
  ]);

  // Next run: labels come from the store, no LLM call for them
  const run2 = await engine().run();
  assert.equal(labelCalls().length, 1);
  assert.equal(run2.stats.profile.clustersRebuilt, false);
  assert.deepEqual(run2.stats.topics, { clusters: 3, labeled: 3, generated: 0 });
});

test("labels carry over when the library is reclustered and only new directions are labeled", async () => {
  const { engine, storage, state, labelCalls } = setup();
  await engine().run();
  assert.equal(labelCalls().length, 1);

  // Two more copper papers: reclustered, every direction keeps its label
  state.items = libraryItems({ copper: 10 });
  const run = await engine().run();
  assert.equal(run.stats.profile.clustersRebuilt, true);
  assert.deepEqual(run.errors, []);
  assert.equal(labelCalls().length, 1, "no new label call");
  assert.deepEqual(run.stats.topics, { clusters: 3, labeled: 3, generated: 0 });
  const store = await storage.readJSON("cluster-labels.json");
  assert.equal(store.clusters[0].keys.length, 10, "the store follows the new memberships");
  assert.equal(store.clusters[0].label, "铜同位素");

  // A new research direction and a new paper in it: only that cluster is sent;
  // the paper is merged before the label exists and gets its topic afterwards
  state.items = libraryItems({ copper: 10, sulfur: 4 });
  state.candidates = [...CANDIDATES, crossrefWork(4, "Sulfur degassing from arc volcanoes", "Sulfur sulfur volatiles")];
  const run2 = await engine().run();
  assert.deepEqual(run2.errors, []);
  assert.equal(labelCalls().length, 2);
  assert.match(labelCalls()[1], /^Cluster 1 \(4 papers\):\n {2}Titles: "Sulfur degassing/m);
  assert.doesNotMatch(labelCalls()[1], /Cluster 2/);
  assert.deepEqual(run2.stats.topics, { clusters: 4, labeled: 4, generated: 1 });
  const topics = await topicsById(engine());
  assert.equal(topics["doi:10.7777/cand.4"], "硫循环");
  assert.equal(topics["doi:10.7777/cand.1"], "铜同位素");
  assert.deepEqual((await engine().getTopics()).map((t) => t.label), ["铜同位素", "地幔动力学", "硫循环", "俯冲带流体"]);
});

test("local mode reuses labels although it reclusters on every run", async () => {
  const { engine, storage, labelCalls, http } = setup({
    prefs: { "embedding.provider": "local", "embedding.apiKey": "", "scoring.minSimilarity": "" },
  });
  const run = await engine().run();
  assert.deepEqual(run.errors, []);
  assert.ok(!http.calls.some((c) => c.url.includes("embeddings")));
  assert.equal(labelCalls().length, 1);
  assert.equal(run.stats.topics.generated, run.stats.profile.clusters);
  const run2 = await engine().run();
  assert.equal(run2.stats.profile.clustersRebuilt, true);
  assert.equal(await storage.readJSON("clusters.json"), null, "local mode never caches clusters");
  assert.equal(labelCalls().length, 1, "labels are matched by members instead");
  assert.equal(run2.stats.topics.generated, 0);
  assert.equal((await topicsById(engine()))["doi:10.7777/cand.1"], "铜同位素");
});

test("a bad label reply or a failed request stores nothing, is reported and is retried next run", async () => {
  const { engine, storage, state, labelCalls } = setup();
  const failures = [
    [() => reply('{"labels": ["铜同位素", "地幔动力学", "俯冲带流体"]}'), "研究方向标签生成失败：大模型的回复不是 JSON 数组"],
    [() => reply('["铜同位素"]'), "研究方向标签生成失败：大模型返回了 1 个标签，应为 3 个"],
    [() => reply("1. 铜同位素\n2. 地幔动力学\n3. 俯冲带流体"), /^研究方向标签生成失败：无法解析大模型的回复/],
    [() => ({ status: 401, data: { error: { message: "invalid key" } } }), "研究方向标签生成失败：HTTP 401 for https://llm.example/v1/chat/completions: invalid key"],
  ];
  for (const [handler, message] of failures) {
    state.llm = handler;
    const run = await engine().run();
    assert.equal(run.errors.length, 1);
    assert.equal(run.errors[0].stage, "llm");
    if (typeof message === "string") assert.equal(run.errors[0].message, message);
    else assert.match(run.errors[0].message, message);
    assert.ok(run.stats.recommended > 0, "the run itself succeeds");
    assert.deepEqual(run.stats.topics, { clusters: 3, labeled: 0, generated: 0 });
    assert.equal(await storage.readJSON("cluster-labels.json"), null);
    assert.deepEqual(Object.values(await topicsById(engine())), [undefined, undefined]);
  }
  assert.equal(labelCalls().length, 4, "retried on every run");

  // Server errors are retried by the transport first, then reported the same way
  let attempts = 0;
  state.llm = () => {
    attempts++;
    return { status: 503, data: { message: "busy" } };
  };
  const busy = await engine().run();
  assert.equal(attempts, 3);
  assert.match(busy.errors[0].message, /^研究方向标签生成失败：HTTP 503/);

  // Recovered: this run's works are patched with their topics
  state.llm = labelLlm;
  const run = await engine().run();
  assert.deepEqual(run.errors, []);
  assert.deepEqual(run.stats.topics, { clusters: 3, labeled: 3, generated: 3 });
  assert.deepEqual(await topicsById(engine()), { "doi:10.7777/cand.1": "铜同位素", "doi:10.7777/cand.2": "地幔动力学" });
});

test("cancelling the label request cancels the run after its results are stored", async () => {
  const { engine, storage, state } = setup();
  state.llm = () => {
    throw new CancelledRequestError();
  };
  await assert.rejects(engine().run(), (e) => e.cancelled === true);
  assert.equal(Object.keys((await engine().loadInbox()).works).length, 2, "merged before the LLM step");
  assert.equal(await storage.readJSON("cluster-labels.json"), null);

  // A cancel right after the labels arrive: they are kept, the run stops
  state.llm = labelLlm;
  let cancelled = false;
  await assert.rejects(
    engine().run({
      onProgress: (p) => {
        if (p.stage === "llm") cancelled = true;
      },
      isCancelled: () => cancelled,
    }),
    CancelledError,
  );
  assert.equal((await storage.readJSON("cluster-labels.json")).clusters.length, 3);
});

test("labels can be turned off, apply without the LLM once stored and are generated on empty runs", async () => {
  const off = setup({ prefs: { "llm.topicLabels": false } });
  const offRun = await off.engine().run();
  assert.deepEqual(offRun.errors, []);
  assert.equal(off.labelCalls().length, 0);
  assert.equal(offRun.stats.topics, undefined);
  assert.equal(await off.storage.readJSON("cluster-labels.json"), null);
  assert.deepEqual(Object.values(await topicsById(off.engine())), [undefined, undefined]);

  // An empty run (nothing fetched) still labels the library's directions
  const { engine, state, llmCalls, labelCalls } = setup({ candidates: [] });
  const empty = await engine().run();
  assert.equal(empty.stats.recommended, 0);
  assert.deepEqual(empty.errors, []);
  assert.equal(labelCalls().length, 1);
  assert.deepEqual(empty.stats.topics, { clusters: 3, labeled: 3, generated: 3 });

  // With the LLM switched off, stored labels still name new works (set when merged)
  state.candidates = CANDIDATES;
  const noLlm = await engine({ "llm.enabled": false }).run();
  assert.deepEqual(noLlm.errors, []);
  assert.equal(llmCalls().length, 1, "no LLM request at all");
  assert.deepEqual(await topicsById(engine()), { "doi:10.7777/cand.1": "铜同位素", "doi:10.7777/cand.2": "地幔动力学" });

  // LLM on without a key and nothing new: no repeated "missing key" error
  const noKey = setup({ candidates: [], prefs: { "llm.apiKey": "" } });
  const noKeyRun = await noKey.engine().run();
  assert.deepEqual(noKeyRun.errors, []);
  assert.equal(noKey.labelCalls().length, 0);
});

test("flagship articles get a topic, followed-only works do not", async () => {
  const ann = "0000-0002-1825-0097";
  const followed = {
    id: "https://openalex.org/W2001",
    doi: "https://doi.org/10.8888/f.1",
    title: "Copper isotope tracer",
    display_name: "Copper isotope tracer",
    publication_date: "2024-10-05",
    type: "article",
    is_paratext: false,
    authorships: [{ author: { id: "https://openalex.org/A1", display_name: "Ann Lee", orcid: `https://orcid.org/${ann}` } }],
    primary_location: { source: { display_name: "EPSL", issn: ["0012-821X"] } },
    abstract_inverted_index: null,
    cited_by_count: 0,
  };
  const { engine } = setup({
    candidates: [...CANDIDATES, crossrefWork(9, "Mantle plumes beneath Hawaii", "Mantle mantle subduction", { issn: "0028-0836", venue: "Nature", doi: "10.1038/n.9" })],
    followed: [followed],
    prefs: {
      "flagship.enabled": true,
      "flagship.issns": "0028-0836",
      "flagship.positiveAnchor": "mantle subduction isotope",
      "flagship.negativeAnchors": "battery",
      "flagship.minScore": "0.5",
      "flagship.grayLow": "0.1",
      "followed.enabled": true,
      "followed.authors": `Ann Lee, ${ann}`,
    },
  });
  const run = await engine().run();
  assert.deepEqual(run.errors, []);
  assert.equal(run.stats.newByLabel.flagship, 1);
  assert.equal(run.stats.newByLabel.followed, 1);
  const works = (await engine().loadInbox()).works;
  assert.equal(works["doi:10.1038/n.9"].label, "flagship");
  assert.equal(works["doi:10.1038/n.9"].topic, "地幔动力学");
  assert.equal(works["doi:10.8888/f.1"].label, "followed");
  assert.equal(works["doi:10.8888/f.1"].topic, undefined);
});
