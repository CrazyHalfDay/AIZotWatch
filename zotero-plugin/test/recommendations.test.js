// Recommendations window (addon/content/recommendations.js) rendered into a
// minimal fake DOM: track sections, badges, scores, chips, filters and header.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const code = readFileSync(join(root, "addon/content/recommendations.js"), "utf8");

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.childNodes = [];
    this.attributes = {};
    this.listeners = {};
    this.className = "";
    this.ownText = "";
    this.hidden = false;
    this.disabled = false;
    this.style = {};
    this.dataset = {};
    this.scrollTop = 0;
    const node = this;
    this.classList = {
      add: (c) => node.classList.toggle(c, true),
      remove: (c) => node.classList.toggle(c, false),
      contains: (c) => node.className.split(" ").includes(c),
      toggle(c, on = !node.classList.contains(c)) {
        const set = new Set(node.className.split(" ").filter(Boolean));
        on ? set.add(c) : set.delete(c);
        node.className = [...set].join(" ");
      },
    };
  }
  get textContent() {
    return this.ownText + this.childNodes.map((n) => n.textContent).join("");
  }
  set textContent(value) {
    this.ownText = String(value);
    this.childNodes = [];
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
  }
  addEventListener(type, fn) {
    (this.listeners[type] ||= []).push(fn);
  }
  append(...nodes) {
    this.childNodes.push(...nodes);
  }
  replaceChildren(...nodes) {
    this.ownText = "";
    this.childNodes = nodes;
  }
  querySelectorAll() {
    return [];
  }
}

function walk(node, out = []) {
  out.push(node);
  for (const child of node.childNodes || []) walk(child, out);
  return out;
}
const withClass = (node, cls) => walk(node).filter((n) => (n.className || "").split(" ").includes(cls));

function loadWindow(view) {
  const elements = new Map();
  const statusCalls = [];
  const api = {
    getView: async () => view,
    setStatus: async (ids, status) => statusCalls.push([ids, status]),
  };
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new FakeNode("div"));
      return elements.get(id);
    },
    createElement: (tag) => new FakeNode(tag),
    createTextNode: (text) => ({ textContent: text }),
    querySelectorAll: () => [],
  };
  const context = vm.createContext({
    ChromeUtils: { importESModule: () => ({ Zotero: { ZotWatch: api, logError() {}, isMac: false } }) },
    window: { addEventListener() {}, confirm: () => true },
    document,
  });
  vm.runInContext(code, context, { filename: "recommendations.js" });
  context.__view = view;
  vm.runInContext("state.view = __view; render();", context);
  const run = (src) => vm.runInContext(src, context);
  return { run, $: (id) => document.getElementById(id), statusCalls };
}

const RUN_ID = "run-1";
const base = { status: "new", firstRunId: RUN_ID, authors: [{ name: "Ann Lee" }], url: "https://doi.org/x" };
const WORKS = [
  { ...base, id: "p1", source: "crossref", title: "Personal top", venue: "GCA", label: "must_read", score: 0.85, similarity: 0.7, published: "2024-10-01T00:00:00Z", followedAuthors: ["汪在聪", "Ann Lee"], topic: "俯冲带流体" },
  { ...base, id: "p2", source: "arxiv", title: "Personal second", label: "consider", score: 0.5, similarity: 0.4, published: "2024-10-05T00:00:00Z" },
  { ...base, id: "g1", source: "crossref", title: "Flagship article", venue: "Nature", label: "flagship", score: 0.31, similarity: 0.2, published: "2024-10-02T00:00:00Z", gate: { pos: 0.412, neg: 0.2, judged: true } },
  { ...base, id: "f1", source: "openalex", title: "Followed older", venue: null, label: "followed", score: 0, similarity: 0, published: "2024-09-01T00:00:00Z", followedAuthors: ["Ann Lee"] },
  { ...base, id: "f2", source: "openalex", title: "Followed newer", venue: "EPSL", label: "followed", score: 0, similarity: 0, published: "2024-10-03T00:00:00Z", followedAuthors: ["汪在聪"] },
  { ...base, id: "d1", source: "crossref", title: "Dismissed", label: "consider", score: 0.9, status: "dismissed" },
];

function view(overrides = {}) {
  return {
    works: WORKS.map((w) => ({ ...w })),
    lastRun: {
      runId: RUN_ID,
      runAt: Date.now(),
      stats: { fetched: 40, recommended: 2, newRecommendations: 6, newByLabel: { personal: 3, flagship: 1, followed: 2 }, followed: { fetched: 5 } },
      errors: [],
    },
    running: false,
    progress: null,
    llmEnabled: false,
    provider: "dashscope",
    ...overrides,
  };
}

const cardTitles = (nodes) => nodes.filter((n) => n.tagName === "article").map((n) => withClass(n, "title")[0].textContent);

test("the all view groups works into track sections with counts", () => {
  const { $ } = loadWindow(view());
  const nodes = $("list").childNodes;
  const headers = nodes.filter((n) => withClass(n, "section-header").includes(n));
  assert.deepEqual(headers.map((h) => h.textContent), ["个性推荐2 篇", "顶刊地学速览1 篇", "关注作者2 篇"]);
  assert.deepEqual(headers.map((h) => h.className), ["section-header personal", "section-header flagship", "section-header followed"]);
  // Sorted by score within each section; followed works (score 0) newest first
  assert.deepEqual(cardTitles(nodes), ["Personal top", "Personal second", "Flagship article", "Followed newer", "Followed older"]);

  const subtitle = $("subtitle").textContent;
  assert.match(subtitle, /抓取 40 篇，推荐 2 篇（新增 3） · 顶刊 1 篇 · 关注作者 2 篇/);
});

test("cards show track badges, scores and chips", () => {
  const { $ } = loadWindow(view());
  const cards = Object.fromEntries($("list").childNodes.filter((n) => n.tagName === "article").map((n) => [withClass(n, "title")[0].textContent, n]));

  const top = cards["Personal top"];
  assert.equal(top.className, "card must");
  assert.equal(withClass(top, "badge")[0].textContent, "必读");
  assert.equal(withClass(top, "score")[0].textContent, "85 分");
  assert.deepEqual(withClass(top, "chip").map((c) => [c.className, c.textContent]), [
    ["chip followed", "关注：汪在聪、Ann Lee"],
    ["chip topic", "方向：俯冲带流体"],
  ]);
  assert.match(withClass(top, "card-meta")[0].textContent, /GCA · 2024-10-01/);

  const flagship = cards["Flagship article"];
  assert.equal(flagship.className, "card flagship");
  const badge = withClass(flagship, "badge")[0];
  assert.equal(badge.textContent, "顶刊地学");
  assert.equal(badge.attributes.title, "地学相关度 0.412，排除领域 0.2，经大模型复核");
  const score = withClass(flagship, "score")[0];
  assert.equal(score.className, "score muted");
  assert.equal(score.attributes.title, "与文库相似度（仅供参考）");
  assert.equal(score.textContent, "31 分");

  const followed = cards["Followed older"];
  assert.equal(followed.className, "card followed");
  assert.equal(withClass(followed, "badge")[0].textContent, "关注");
  assert.equal(withClass(followed, "score").length, 0, "followed-only works have no score");
  assert.match(withClass(followed, "card-meta")[0].textContent, /OpenAlex · 2024-09-01/, "no venue: the source name");
  assert.match(withClass(cards["Followed newer"], "card-meta")[0].textContent, /EPSL · 2024-10-03/);
  assert.equal(withClass(cards["Personal second"], "card-meta")[0].textContent.includes("arXiv"), true);
});

test("the track filter shows a flat list and keeps the other filters", async () => {
  const { run, $, statusCalls } = loadWindow(view());
  run("state.track = 'followed'; renderList();");
  assert.deepEqual(cardTitles($("list").childNodes), ["Followed newer", "Followed older"]);
  assert.equal(withClass($("list"), "section-header").length, 0);

  run("state.track = 'personal'; renderList();");
  assert.deepEqual(cardTitles($("list").childNodes), ["Personal top", "Personal second"]);
  run("state.filter = 'must_read'; renderList();");
  assert.deepEqual(cardTitles($("list").childNodes), ["Personal top"]);

  run("state.track = 'flagship'; renderList();");
  assert.equal($("list").textContent, "没有符合当前筛选条件的文献。");

  // Search also covers the configured followed-author names and the topic
  run("state.track = 'all'; state.filter = 'all'; state.query = '汪在聪'; renderList();");
  assert.deepEqual(cardTitles($("list").childNodes), ["Personal top", "Followed newer"]);
  run("state.query = '俯冲带'; renderList();");
  assert.deepEqual(cardTitles($("list").childNodes), ["Personal top"]);
  run("state.query = '';");

  // Dismissing acts on every visible work, across sections
  run("state.track = 'all'; state.filter = 'all'; state.sort = 'date'; renderList();");
  assert.deepEqual(cardTitles($("list").childNodes), ["Personal second", "Personal top", "Flagship article", "Followed newer", "Followed older"]);
  await run("dismissVisible()");
  assert.equal(statusCalls.length, 1);
  assert.deepEqual([...statusCalls[0][0]].sort(), ["f1", "f2", "g1", "p1", "p2"]);
  assert.equal(statusCalls[0][1], "dismissed");
});

test("run problems name the track they come from", () => {
  const v = view();
  v.lastRun.errors = [
    { stage: "followed", message: "无法识别的作者行：x" },
    { stage: "followed", source: "OpenAlex", message: "HTTP 503" },
    { stage: "fetch", source: "Crossref", message: "timeout" },
    { stage: "llm", message: "摘要失败：y" },
  ];
  const { $ } = loadWindow(v);
  assert.equal(
    $("notice").textContent,
    "上次运行有 4 个问题：关注作者：无法识别的作者行：x；关注作者（OpenAlex）：HTTP 503；Crossref：timeout；大模型：摘要失败：y",
  );

  // Runs from before 0.2.0 have no newByLabel; followed-only runs are not "empty"
  const old = view();
  old.lastRun.stats = { fetched: 0, recommended: 0, newRecommendations: 0, followed: { fetched: 3 } };
  const { $: $old } = loadWindow(old);
  assert.match($old("subtitle").textContent, /推荐 0 篇（新增 0）/);
  assert.doesNotMatch($old("subtitle").textContent, /顶刊|关注作者/);
  assert.equal($old("notice").hidden, true);
});
