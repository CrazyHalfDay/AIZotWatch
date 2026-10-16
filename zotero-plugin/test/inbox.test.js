// Main window scripts (addon/content/main.js + inbox.js) rendered into a
// minimal fake DOM: the shell helpers, routing, and the 推荐 page (track
// sections, badges, scores, chips, filters, notices and the header line).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const mainCode = readFileSync(join(root, "addon/content/main.js"), "utf8");
const inboxCode = readFileSync(join(root, "addon/content/inbox.js"), "utf8");
// Values created inside the VM have foreign prototypes; compare them structurally
const plain = (x) => JSON.parse(JSON.stringify(x));

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
    this.value = "";
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
  getAttribute(k) {
    return this.attributes[k] ?? null;
  }
  removeAttribute(k) {
    delete this.attributes[k];
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
  querySelector() {
    return null;
  }
}

function walk(node, out = []) {
  out.push(node);
  for (const child of node.childNodes || []) walk(child, out);
  return out;
}
const withClass = (node, cls) => walk(node).filter((n) => (n.className || "").split(" ").includes(cls));

function loadWindow(view, { now } = {}) {
  const elements = new Map();
  const statusCalls = [];
  const shown = [];
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
    createElementNS: (ns, tag) => new FakeNode(tag),
    createTextNode: (text) => ({ textContent: text }),
    querySelectorAll: () => [],
    documentElement: new FakeNode("html"),
  };
  const location = { hash: "" };
  const context = vm.createContext({
    ChromeUtils: { importESModule: () => ({ Zotero: { ZotWatch: api, logError() {}, isMac: false } }) },
    window: { addEventListener() {}, confirm: () => true },
    document,
    location,
    history: { replaceState: (state, title, url) => (location.hash = url) },
    Intl,
    Event: class {
      constructor(type, init = {}) {
        this.type = type;
        this.bubbles = !!init.bubbles;
      }
    },
    setTimeout,
    clearTimeout,
    ...(now ? { Date: class extends Date {
      static now() {
        return now;
      }
    } } : {}),
  });
  vm.runInContext(mainCode, context, { filename: "main.js" });
  vm.runInContext(inboxCode, context, { filename: "inbox.js" });
  context.__view = view;
  vm.runInContext("ZotWatchInbox.refresh(__view);", context);
  const run = (src) => vm.runInContext(src, context);
  // A probe page records show() calls
  context.__shown = shown;
  run("registerPage('settings', { show: (p) => __shown.push(p) });");
  return { run, $: (id) => document.getElementById(id), statusCalls, api, location };
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
  const { $, run } = loadWindow(view());
  const nodes = $("inbox-list").childNodes;
  const headers = nodes.filter((n) => withClass(n, "section-header").includes(n));
  assert.deepEqual(headers.map((h) => h.textContent), ["个性推荐2 篇", "顶刊地学速览1 篇", "关注作者2 篇"]);
  assert.deepEqual(headers.map((h) => h.className), ["section-header personal", "section-header flagship", "section-header followed"]);
  // Each section header carries its track dot
  assert.deepEqual(headers.map((h) => withClass(h, "dot")[0].className), ["dot personal", "dot flagship", "dot followed"]);
  // Sorted by score within each section; followed works (score 0) newest first
  assert.deepEqual(cardTitles(nodes), ["Personal top", "Personal second", "Flagship article", "Followed newer", "Followed older"]);

  assert.match(run("ZotWatchInbox.subtitle(__view)"), /抓取 40 篇，推荐 2 篇（新增 3） · 顶刊 1 篇 · 关注作者 2 篇 · 向量化：阿里云 DashScope/);
  assert.equal($("count-new").textContent, "5");
  assert.equal($("count-dismissed").textContent, "1");
  assert.equal($("inbox-filters").hidden, false);
});

test("cards show track badges, scores and chips", () => {
  const { $ } = loadWindow(view());
  const cards = Object.fromEntries($("inbox-list").childNodes.filter((n) => n.tagName === "article").map((n) => [withClass(n, "title")[0].textContent, n]));

  const top = cards["Personal top"];
  assert.equal(top.className, "card work must");
  assert.equal(top.attributes["data-track"], "personal");
  assert.equal(withClass(top, "badge")[0].textContent, "必读");
  assert.equal(withClass(top, "badge")[1].textContent, "新", "added by the last run");
  assert.equal(withClass(top, "score")[0].textContent, "85 分");
  assert.deepEqual(withClass(top, "chip").map((c) => [c.className, c.textContent]), [
    ["chip followed", "关注：汪在聪、Ann Lee"],
    ["chip topic", "方向：俯冲带流体"],
  ]);
  assert.match(withClass(top, "card-meta")[0].textContent, /GCA · 2024-10-01/);
  assert.deepEqual(withClass(top, "card-actions")[0].childNodes.map((b) => b.textContent), ["加入 Zotero", "忽略", "打开原文"]);

  const flagship = cards["Flagship article"];
  assert.equal(flagship.className, "card work flagship");
  const badge = withClass(flagship, "badge")[0];
  assert.equal(badge.textContent, "顶刊地学");
  assert.equal(badge.attributes.title, "地学相关度 0.412，排除领域 0.2，经大模型复核");
  assert.equal(withClass(badge, "dot")[0].className, "dot flagship", "track badges carry the track dot");
  const score = withClass(flagship, "score")[0];
  assert.equal(score.className, "score muted");
  assert.equal(score.attributes.title, "与文库相似度（仅供参考）");
  assert.equal(score.textContent, "相似度 20", "the library similarity, as in the saved note (not the 31 score)");

  const followed = cards["Followed older"];
  assert.equal(followed.className, "card work followed");
  assert.equal(withClass(followed, "badge")[0].textContent, "关注");
  assert.equal(withClass(followed, "score").length, 0, "followed-only works have no score");
  assert.match(withClass(followed, "card-meta")[0].textContent, /OpenAlex · 2024-09-01/, "no venue: the source name");
  assert.match(withClass(cards["Followed newer"], "card-meta")[0].textContent, /EPSL · 2024-10-03/);
  assert.equal(withClass(cards["Personal second"], "card-meta")[0].textContent.includes("arXiv"), true);
});

test("the track filter shows a flat list and keeps the other filters", async () => {
  const { run, $, statusCalls } = loadWindow(view());
  const state = (src) => run(`(() => { const state = ZotWatchInbox.state; ${src}; ZotWatchInbox.renderList(); })()`);
  state("state.track = 'followed'");
  assert.deepEqual(cardTitles($("inbox-list").childNodes), ["Followed newer", "Followed older"]);
  assert.equal(withClass($("inbox-list"), "section-header").length, 0);

  state("state.track = 'personal'");
  assert.deepEqual(cardTitles($("inbox-list").childNodes), ["Personal top", "Personal second"]);
  state("state.filter = 'must_read'");
  assert.deepEqual(cardTitles($("inbox-list").childNodes), ["Personal top"]);

  state("state.track = 'flagship'");
  assert.equal(withClass($("inbox-list"), "empty-title")[0].textContent, "没有符合当前筛选条件的文献。");
  assert.equal(withClass($("inbox-list"), "btn")[0].textContent, "清除筛选");

  // Search also covers the configured followed-author names and the topic
  state("state.track = 'all'; state.filter = 'all'; state.query = '汪在聪'");
  assert.deepEqual(cardTitles($("inbox-list").childNodes), ["Personal top", "Followed newer"]);
  state("state.query = '俯冲带'");
  assert.deepEqual(cardTitles($("inbox-list").childNodes), ["Personal top"]);

  // Dismissing acts on every visible work, across sections
  state("state.query = ''; state.sort = 'date'");
  assert.deepEqual(cardTitles($("inbox-list").childNodes), ["Personal second", "Personal top", "Flagship article", "Followed newer", "Followed older"]);
  await run("ZotWatchInbox.dismissVisible()");
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
  const { $, run } = loadWindow(v);
  const notice = $("inbox-notice");
  assert.equal(notice.hidden, false);
  assert.equal(notice.className, "callout warning");
  assert.equal(withClass(notice, "callout-title")[0].textContent, "上次运行有 4 个问题");
  assert.deepEqual(
    withClass(notice, "callout-list")[0].childNodes.map((li) => li.textContent),
    ["关注作者：无法识别的作者行：x", "关注作者（OpenAlex）：HTTP 503", "Crossref：timeout", "大模型：摘要失败：y"],
  );

  // Runs from before 0.2.0 have no newByLabel; followed-only runs are not "empty"
  const old = view();
  old.lastRun.stats = { fetched: 0, recommended: 0, newRecommendations: 0, followed: { fetched: 3 } };
  const { $: $old, run: runOld } = loadWindow(old);
  const line = runOld("ZotWatchInbox.subtitle(__view)");
  assert.match(line, /推荐 0 篇（新增 0）/);
  assert.doesNotMatch(line, /顶刊|关注作者/);
  assert.equal($old("inbox-notice").hidden, true);
  assert.ok(run);

  // Nothing fetched at all: a hint with a shortcut to the journal settings
  const empty = view();
  empty.lastRun.stats = { fetched: 0, recommended: 0, newRecommendations: 0 };
  const { $: $empty } = loadWindow(empty);
  assert.equal(withClass($empty("inbox-notice"), "callout-title")[0].textContent, "上次运行没有抓取到任何文献");
  assert.equal(withClass($empty("inbox-notice"), "btn")[0].textContent, "关注期刊设置");
});

test("empty inbox states", () => {
  // Never run: an action to start the first run; filters are hidden
  const { $ } = loadWindow(view({ works: [], lastRun: null }));
  assert.equal(withClass($("inbox-list"), "empty-title")[0].textContent, "还没有推荐");
  assert.equal(withClass($("inbox-list"), "btn")[0].textContent, "立即检查");
  assert.equal($("inbox-filters").hidden, true);

  // Everything reviewed
  const { $: $done } = loadWindow(view({ works: WORKS.filter((w) => w.status !== "new").map((w) => ({ ...w })) }));
  assert.equal(withClass($done("inbox-list"), "empty-title")[0].textContent, "全部看完了");
});

test("shell helpers: formatting, run status and routes", () => {
  const now = Date.UTC(2024, 9, 10, 12);
  const { run } = loadWindow(view(), { now });
  assert.equal(run("formatCompact(1284)"), "1,284");
  assert.equal(run("formatCompact(12900)"), "12.9K");
  assert.equal(run("formatCompact(20000)"), "20K");
  assert.equal(run("formatCompact(3450000)"), "3.5M");
  assert.equal(run("formatCompact(undefined)"), "–");
  assert.equal(run(`formatRelative(${now - 30000})`), "刚刚");
  assert.equal(run(`formatRelative(${now - 5 * 60000})`), "5 分钟前");
  assert.equal(run(`formatRelative(${now - 2.5 * 3600000})`), "2 小时前");
  assert.equal(run(`formatRelative(${now - 3 * 86400000})`), "3 天前");
  assert.equal(run(`formatUntil(${now + 22 * 3600000})`), "约 22 小时后");
  assert.equal(run(`formatUntil(${now + 10 * 60000})`), "约 10 分钟后");
  assert.equal(run(`formatUntil(${now - 1})`), "即将");
  assert.equal(run("formatDuration(192000)"), "3 分 12 秒");
  assert.equal(run("formatDuration(9000)"), "9 秒");

  const status = (v) => plain(run(`runStatus(${JSON.stringify(v)})`));
  const schedule = { autoRun: true, intervalHours: 24, nextRunAt: now + 22 * 3600000 };
  assert.deepEqual(status({ running: false, lastRun: null, schedule }), { tone: "idle", label: "尚未运行", next: "下次：约 22 小时后" });
  assert.deepEqual(status({ running: true, lastRun: null, schedule: { ...schedule, autoRun: false } }), { tone: "running", label: "正在检查…", next: "自动检查已关闭" });
  const lastRun = { runAt: now - 2 * 3600000, finishedAt: now - 2 * 3600000 + 60000, errors: [] };
  assert.equal(status({ lastRun, schedule }).tone, "good");
  assert.equal(status({ lastRun, schedule }).label, "上次检查 1 小时前");
  assert.equal(status({ lastRun: { ...lastRun, errors: [{}] }, schedule }).tone, "warning");
  assert.equal(status({ lastRun, lastError: { message: "x", at: now - 60000 }, schedule }).tone, "critical");
  assert.equal(status({ lastRun, lastError: { message: "x", at: now - 5 * 3600000 }, schedule }).tone, "good", "a later success wins");
  assert.equal(status({ lastRun, schedule: { ...schedule, nextRunAt: now - 1000 } }).next, "下次：即将检查");

  assert.deepEqual(plain(run("parseRoute('#settings/ai')")), { page: "settings", section: "ai", route: "settings/ai" });
  assert.deepEqual(plain(run("parseRoute('settings/nope')")), { page: "settings", section: null, route: "settings" });
  assert.deepEqual(plain(run("parseRoute('#inbox/ai')")), { page: "inbox", section: null, route: "inbox" });
  assert.deepEqual(plain(run("parseRoute('')")), { page: "overview", section: null, route: "overview" });
});

test("navigate switches pages and remembers the page, without touching the URL", () => {
  const { run, api, location } = loadWindow(view());
  run("navigate('settings/flagship')");
  assert.equal(api.lastPage, "settings/flagship");
  assert.deepEqual(plain(run("__shown")), [{ section: "flagship" }]);
  assert.equal(run("shell.page"), "settings");
  assert.equal(run("$('settings-nav').hidden"), false);
  assert.equal(run("$('page-title').textContent"), "设置");
  run("navigate('inbox')");
  assert.equal(api.lastPage, "inbox");
  assert.equal(run("$('settings-nav').hidden"), true);
  assert.equal(run("$('page-title').textContent"), "推荐");
  run("navigate('#bogus')");
  assert.equal(api.lastPage, "overview");
  // Zotero keys the window's persisted size and position by its document URI
  assert.equal(location.hash, "", "the route never goes into the URL");
});
