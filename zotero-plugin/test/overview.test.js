// The 概览 page (addon/content/overview.js, with main.js) in a fake DOM: the
// pure chart geometry and formatting helpers (scales, stacked columns with
// their surface gap and rounded data end, bar lists, funnel stages, run
// health, KPI tiles), and the rendered dashboard (cards, tooltips on hover
// and keyboard focus, the 表格 toggle, empty states, 测试连接 and reloads).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const mainCode = readFileSync(join(root, "addon/content/main.js"), "utf8");
const overviewCode = readFileSync(join(root, "addon/content/overview.js"), "utf8");
// Values created inside the VM have foreign prototypes; compare them structurally
const plain = (x) => JSON.parse(JSON.stringify(x));

const HOUR = 3600000;
const DAY = 86400000;
const NOW = Date.UTC(2026, 9, 16, 8, 0);

// ------------------------------------------------------------- fake DOM

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
      add: (...cs) => cs.forEach((c) => node.classList.toggle(c, true)),
      remove: (...cs) => cs.forEach((c) => node.classList.toggle(c, false)),
      contains: (c) => node.className.split(" ").includes(c),
      toggle(c, on = !node.classList.contains(c)) {
        const set = new Set(node.className.split(" ").filter(Boolean));
        if (on) set.add(c);
        else set.delete(c);
        node.className = [...set].join(" ");
        return on;
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
    if (k === "class") this.className = String(v);
    else this.attributes[k] = String(v);
  }
  getAttribute(k) {
    return k === "class" ? this.className : (this.attributes[k] ?? null);
  }
  removeAttribute(k) {
    delete this.attributes[k];
  }
  addEventListener(type, fn) {
    (this.listeners[type] ||= []).push(fn);
  }
  fire(type, event = {}) {
    for (const fn of this.listeners[type] || []) fn({ type, target: this, preventDefault() {}, ...event });
  }
  focus() {
    this.fire("focus");
  }
  click() {
    this.fire("click");
  }
  // Strings become text nodes, as in the DOM
  append(...nodes) {
    this.childNodes.push(...nodes.map((n) => (typeof n === "string" ? { textContent: n } : n)));
  }
  replaceChildren(...nodes) {
    this.ownText = "";
    this.childNodes = [];
    this.append(...nodes);
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
const all = (node, cls) => walk(node).filter((n) => (n.className || "").split(" ").includes(cls));
const one = (node, cls) => all(node, cls)[0];
const byTag = (node, tag) => walk(node).filter((n) => n.tagName === tag);

function loadWindow(dashboard, { view = { works: [], running: false }, connections = [] } = {}) {
  const elements = new Map();
  const listeners = new Set();
  const calls = { dashboard: 0, connections: 0, runNow: 0 };
  let current = dashboard;
  let failWith = null;
  // The page's clock (Date.now() inside the window), moved by tests
  const clock = { now: NOW };
  const api = {
    getView: async () => view,
    getDashboard: async () => {
      calls.dashboard += 1;
      if (failWith) throw failWith;
      return structuredClone(current);
    },
    checkConnections: async () => {
      calls.connections += 1;
      return connections;
    },
    runNow: async () => {
      calls.runNow += 1;
    },
    addListener: (fn) => listeners.add(fn),
    removeListener: (fn) => listeners.delete(fn),
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
    setTimeout,
    clearTimeout,
    structuredClone,
    Date: class extends Date {
      constructor(...args) {
        super(...(args.length ? args : [clock.now]));
      }
      static now() {
        return clock.now;
      }
    },
  });
  vm.runInContext(mainCode, context, { filename: "main.js" });
  vm.runInContext(overviewCode, context, { filename: "overview.js" });
  const run = (src) => vm.runInContext(src, context);
  run("ZotWatchOverview.init()");
  return {
    run,
    O: run("ZotWatchOverview"),
    $: (id) => document.getElementById(id),
    calls,
    listeners,
    view,
    setDashboard: (d) => (current = d),
    clock,
    failNext: (e) => (failWith = e),
    async refresh(v = view) {
      context.__view = v;
      await run("ZotWatchOverview.refresh(__view)");
    },
  };
}

// --------------------------------------------------------------- fixtures

// The last run finished a little over two hours ago (it took 3 分 12 秒)
const RUN_AT = NOW - 2 * HOUR - 10 * 60000;

function history(n = 14) {
  return Array.from({ length: n }, (_, i) => ({
    runId: `run-${i}`,
    runAt: RUN_AT - (n - 1 - i) * DAY,
    finishedAt: RUN_AT - (n - 1 - i) * DAY + 150000,
    fetched: 900 + i,
    recommended: 30,
    newByLabel: { personal: i + 1, flagship: i % 3 === 0 ? 2 : 0, followed: i === n - 1 ? 1 : 0 },
    newMustRead: i % 4,
    errors: i === 2 ? 1 : 0,
  }));
}

function dashboard(overrides = {}) {
  const lastRun = {
    runId: "run-13",
    runAt: RUN_AT,
    finishedAt: RUN_AT + 192000,
    stats: {
      fetched: 1284,
      afterDedupe: 1201,
      // The 164 flagship candidates leave the personal funnel here
      afterSplit: 1037,
      afterKeywords: 1009,
      afterAbstract: 871,
      afterGate: 86,
      afterRecent: 41,
      recommended: 30,
      recentDays: 7,
      requireAbstract: true,
      newRecommendations: 23,
      newByLabel: { personal: 18, flagship: 3, followed: 2 },
      newMustRead: 4,
      flagship: { candidates: 164, withAbstract: 151, recent: 120, accepted: 9, gray: 14, judged: 14, passed: 12, rejected: 108, recommended: 12 },
    },
    errors: [{ stage: "fetch", source: "arXiv", message: "HTTP 503" }],
  };
  return {
    version: "0.3.0",
    library: { name: "我的文库", items: 2318, withAbstract: 2120 },
    embedding: { provider: "dashscope", model: "text-embedding-v4", label: "阿里云 DashScope · text-embedding-v4", local: false },
    llm: { enabled: true, configured: true, model: "deepseek-chat", topicLabels: true },
    schedule: { autoRun: true, intervalHours: 24, lastRun: lastRun.finishedAt, lastAttempt: RUN_AT, nextRunAt: lastRun.finishedAt + 24 * HOUR },
    running: false,
    pending: { total: 16, personal: 11, flagship: 3, followed: 2, mustRead: 3 },
    added: 2,
    addedDays: 60,
    scoring: { recentDays: 7, requireAbstract: true },
    lastRun,
    lastError: null,
    history: history(),
    topics: [
      { label: "斑岩铜矿成矿", size: 312 },
      { label: "俯冲带流体与氧化还原", size: 268 },
      { label: "岩浆挥发分", size: 221 },
    ],
    topicCount: 14,
    clusters: 14,
    followed: {
      enabled: true,
      daysBack: 60,
      authors: [
        { name: "汪在聪", orcid: "0000-0002-3584-1673", openalexId: null, pending: 2 },
        { name: "Cin-Ty Lee", orcid: null, openalexId: "A5023888391", pending: 0 },
      ],
      invalid: ["张伟（没有 ORCID）"],
    },
    flagship: { enabled: true, issnCount: 12, stats: lastRun.stats.flagship },
    journals: { count: 52, withImpactFactor: 47, autoFromLibrary: true, error: null },
    sources: [
      { key: "crossref", name: "Crossref", enabled: true, count: 1102, detail: "48 个 ISSN", error: null, ran: true },
      { key: "arxiv", name: "arXiv", enabled: true, count: 18, detail: "physics.geo-ph", error: "HTTP 503" },
      { key: "eartharxiv", name: "EarthArXiv", enabled: false, count: null, detail: null, error: null },
      { key: "llm", name: "大模型", enabled: false, count: null, detail: "未启用", error: null },
    ],
    abstracts: null,
    dataDir: "/tmp/zotwatch",
    ...overrides,
  };
}

function emptyDashboard() {
  return dashboard({
    embedding: { provider: "local", model: "", label: "本地模式", local: true },
    llm: { enabled: false, configured: false, model: "deepseek-chat", topicLabels: true },
    schedule: { autoRun: true, intervalHours: 24, lastRun: null, lastAttempt: null, nextRunAt: NOW - 1 },
    pending: { total: 0, personal: 0, flagship: 0, followed: 0, mustRead: 0 },
    added: 0,
    lastRun: null,
    history: [],
    topics: [],
    topicCount: 0,
    clusters: null,
    followed: { enabled: false, daysBack: 60, authors: [], invalid: [] },
    flagship: { enabled: false, issnCount: 12, stats: null },
  });
}

const cards = (w) => w.$("overview-root").childNodes;
const cardOf = (w, cls) => cards(w).find((n) => n.className.split(" ").includes(cls));

// ------------------------------------------------------------ pure helpers

test("count axes use clean 1 / 2 / 5 steps with at most three gridlines", () => {
  const { O } = loadWindow(dashboard());
  assert.deepEqual(plain(O.niceScale(23)), { max: 30, step: 10, ticks: [0, 10, 20, 30] });
  assert.deepEqual(plain(O.niceScale(18)), { max: 20, step: 10, ticks: [0, 10, 20] });
  assert.deepEqual(plain(O.niceScale(3)), { max: 3, step: 1, ticks: [0, 1, 2, 3] });
  assert.deepEqual(plain(O.niceScale(0)), { max: 1, step: 1, ticks: [0, 1] }, "an all-zero history still has an axis");
  assert.deepEqual(plain(O.niceScale(1284).ticks), [0, 500, 1000, 1500]);
  assert.deepEqual(["0", "30", "1K", "1.5K"], [0, 30, 1000, 1500].map((n) => O.formatTick(n)));
  assert.equal(O.formatPercent(1150 / 1284), "89.6%");
  assert.equal(O.formatPercent(1), "100%");
  assert.equal(O.formatShortDateTime(Date.UTC(2026, 9, 3, 1, 5), NOW).length, 11, "MM-DD HH:mm within the year");
  assert.match(O.formatShortDateTime(Date.UTC(2025, 9, 3, 1, 5), NOW), /^2025-/);
});

test("text width estimates and truncation", () => {
  const { O } = loadWindow(dashboard());
  assert.equal(O.textWidth("斑岩", 12), 24, "CJK characters are one em");
  assert.ok(O.textWidth("1,284", 12) < O.textWidth("斑岩铜矿成", 12));
  assert.equal(O.truncateText("斑岩铜矿成矿", 200, 12), "斑岩铜矿成矿");
  const short = O.truncateText("俯冲带流体与氧化还原过程", 60, 12);
  assert.ok(short.endsWith("…"));
  assert.ok(O.textWidth(short, 12) <= 60);
});

test("history rows keep the last 14 valid runs with per-track totals", () => {
  const { O } = loadWindow(dashboard());
  const rows = O.historySeries([...history(20), null, { runAt: "x" }]);
  assert.equal(rows.length, 14);
  assert.equal(rows.at(-1).runId, "run-19");
  assert.equal(rows[0].runId, "run-6");
  const last = rows.at(-1);
  assert.equal(last.total, last.personal + last.flagship + last.followed);
  assert.deepEqual(plain(O.historySeries([{ runAt: 1, newByLabel: { personal: -3 } }])[0]), {
    runAt: 1,
    finishedAt: null,
    personal: 0,
    flagship: 0,
    followed: 0,
    mustRead: 0,
    errors: 0,
    total: 0,
  });
  // Series: personal always; another track when it is enabled or has values
  assert.deepEqual(plain(O.chartTracks(rows, {})), ["personal", "flagship", "followed"]);
  const personalOnly = rows.map((r) => ({ ...r, flagship: 0, followed: 0 }));
  assert.deepEqual(plain(O.chartTracks(personalOnly, {})), ["personal"]);
  assert.deepEqual(plain(O.chartTracks(personalOnly, { followed: true })), ["personal", "followed"]);
});

test("stacked columns: thin, 2px surface gaps, rounded top segment, exact totals", () => {
  const { O } = loadWindow(dashboard());
  const rows = O.historySeries(history());
  const layout = O.columnLayout(rows, { width: 560, height: 200, fontPx: 11 });
  assert.equal(layout.columns.length, 14);
  assert.deepEqual(plain(layout.ticks.map((t) => t.label)), ["0", "5", "10", "15"], "the largest total is 15");
  assert.equal(layout.ticks[0].y, layout.baseY, "the zero tick is the baseline");
  for (const col of layout.columns) {
    assert.ok(col.w <= 24 && col.w < col.band, "columns never fill their band");
    const segs = col.segments;
    assert.equal(segs.filter((s) => s.top).length, segs.length ? 1 : 0);
    if (segs.length) assert.ok(segs.at(-1).top, "only the topmost segment has the rounded end");
    for (let k = 1; k < segs.length; k++) {
      const below = segs[k - 1];
      assert.ok(Math.abs(below.y - (segs[k].y + segs[k].h) - 2) < 0.11, "2px gap between stacked segments");
    }
    // The column's top is at its total (segments are big enough here)
    const expected = layout.baseY - (col.row.total / layout.scale.max) * layout.plot.h;
    assert.ok(Math.abs(col.top - expected) < 0.11, `column ${col.index} top`);
    assert.equal(segs[0].y + segs[0].h <= layout.baseY + 0.01, true, "segments sit on the baseline");
  }
  // A tiny value still shows as a 2px segment
  const tiny = O.columnLayout([{ runAt: NOW, personal: 400, flagship: 1, followed: 0, total: 401 }], { width: 300 });
  assert.equal(tiny.columns[0].segments[1].h, 2);
  // Labels thin out from the newest run, which is always labelled
  const narrow = O.columnLayout(rows, { width: 260, fontPx: 11 });
  assert.equal(narrow.columns.at(-1).showLabel, true);
  assert.ok(narrow.columns.filter((c) => c.showLabel).length < 14);
  assert.equal(O.columnLayout(rows, { width: 1200, fontPx: 11 }).columns.filter((c) => c.showLabel).length, 14);
  assert.equal(O.labelStep(14, 40, 45), 2);
  assert.equal(O.labelStep(1, 10, 45), 1);
});

test("bar and column paths round only the data end", () => {
  const { O } = loadWindow(dashboard());
  assert.equal(O.columnPath(10, 20, 20, 50), "M10,70V24A4,4 0 0 1 14,20H26A4,4 0 0 1 30,24V70Z");
  // The radius never exceeds the height of a short segment
  assert.equal(O.columnPath(0, 0, 20, 2), "M0,2V2A2,2 0 0 1 2,0H18A2,2 0 0 1 20,2V2Z");
  assert.equal(O.barPath(100, 6, 50, 16), "M100,6H146A4,4 0 0 1 150,10V18A4,4 0 0 1 146,22H100Z");
});

test("bar lists: labels left, thin bars from one baseline, values at the tip or end", () => {
  const { O } = loadWindow(dashboard());
  const rows = [
    { label: "抓取", value: 1284, valueText: "1,284" },
    { label: "通过相似度门槛", value: 86, valueText: "86", note: "−926" },
    { label: "推荐", value: 0, valueText: "0" },
  ];
  const layout = O.barListLayout(rows, { width: 400, fontPx: 12 });
  assert.equal(layout.rows.length, 3);
  assert.ok(layout.barH <= 16);
  assert.ok(layout.rows.every((r) => r.barX === layout.x0 + 1), "one shared baseline");
  assert.equal(layout.rows[2].barW, 0, "zero has no bar");
  const top = layout.rows[0];
  assert.ok(top.valueX + O.textWidth("1,284", 12) <= 400, "the longest bar leaves room for its value");
  assert.ok(Math.abs(layout.rows[1].barW / top.barW - 86 / 1284) < 0.01, "bars are proportional");
  // End-aligned values and truncated long labels
  const end = O.barListLayout([{ label: "非常非常非常非常长的研究方向名称", value: 5, valueText: "5" }], { width: 300, fontPx: 12, valueAt: "end" });
  assert.equal(end.rows[0].valueX, 300);
  assert.ok(end.rows[0].shortLabel.endsWith("…"));
  assert.ok(end.x0 <= 300 * 0.42 + 1);
});

test("funnel stages come from the last run's stats", () => {
  const { O } = loadWindow(dashboard());
  const stats = dashboard().lastRun.stats;
  const stages = O.funnelStages(stats, { recentDays: 7, requireAbstract: true });
  assert.deepEqual(plain(stages.map((s) => s.label)), ["抓取", "去重后", "分出顶刊后", "关键词过滤后", "有摘要", "通过相似度门槛", "最近 7 天", "推荐"]);
  // The flagship articles are a drop of their own, not one of the keyword filter
  assert.deepEqual(plain(stages.map((s) => s.drop)), [0, 83, 164, 28, 138, 785, 45, 11]);
  // Runs recorded before afterSplit existed: derived from the flagship stats
  const { afterSplit, ...older } = stats;
  assert.equal(afterSplit, 1037);
  assert.equal(O.funnelStages(older, { recentDays: 7 }).find((s) => s.key === "afterSplit").value, 1037);
  assert.ok(!O.funnelStages({ ...older, flagship: undefined }, {}).some((s) => s.key === "afterSplit"), "no flagship track, no split");
  assert.equal(stages[0].share, 1);
  assert.ok(Math.abs(stages.at(-1).share - 30 / 1284) < 1e-9);
  // Stages missing from stats (no candidates) are left out; so are filters that are off
  const partial = O.funnelStages({ fetched: 10, afterDedupe: 0, afterKeywords: 0, afterAbstract: 0, recommended: 0 }, { recentDays: 7 });
  assert.deepEqual(plain(partial.map((s) => s.key)), ["fetched", "afterDedupe", "afterKeywords", "afterAbstract", "recommended"]);
  assert.ok(!O.funnelStages(stats, { requireAbstract: false }).some((s) => s.key === "afterAbstract"));
  assert.ok(!O.funnelStages(stats, { recentDays: 0 }).some((s) => s.key === "afterRecent"));
  assert.equal(O.funnelStages(stats, {}).find((s) => s.key === "afterRecent").label, "时间范围内");
  assert.deepEqual(plain(O.funnelStages(null)), []);

  // The filters come from the run, not from today's settings (older runs: the settings)
  assert.deepEqual(plain(O.funnelOptions({ recentDays: 0, requireAbstract: false }, { recentDays: 7, requireAbstract: true })), { recentDays: 0, requireAbstract: false });
  assert.deepEqual(plain(O.funnelOptions({}, { recentDays: 14, requireAbstract: true })), { recentDays: 14, requireAbstract: true });
  const unlimited = O.funnelStages({ ...stats, recentDays: 0, afterRecent: 86 }, O.funnelOptions({ ...stats, recentDays: 0 }, { recentDays: 7 }));
  assert.ok(!unlimited.some((s) => s.key === "afterRecent"), "no 最近 7 天 bar for a run without the window");

  assert.equal(O.flagshipText(stats.flagship, 7), "顶刊：候选 164 → 有摘要 151 → 最近 7 天 120 → 通过 12（灰区 14，大模型复核 14）→ 显示 12");
  assert.equal(O.flagshipText({ candidates: 3, withAbstract: 1, recent: 0, gray: 0, judged: 0, rejected: 0, recommended: 0 }), "顶刊：候选 3 → 有摘要 1 → 最近 0 → 通过 0（灰区 0）→ 显示 0");
  assert.equal(O.flagshipText(null), "");
  // A gate that failed before judging (embedding error) passed nothing
  assert.equal(
    O.flagshipText({ candidates: 8, withAbstract: 7, recent: 6, accepted: 0, gray: 0, judged: 0, rejected: 0, recommended: 0, failed: true }, 7),
    "顶刊：候选 8 → 有摘要 7 → 最近 7 天 6 → 通过 –（未完成）→ 显示 0",
  );
});

test("run health, schedule lines and source statuses", () => {
  const { O } = loadWindow(dashboard());
  const d = dashboard();
  const health = O.runHealth(d, NOW);
  assert.equal(health.tone, "warning");
  assert.equal(health.label, "有 1 个问题");
  assert.equal(health.title, "上次运行的问题");
  assert.deepEqual(plain(health.problems), [{ tone: "warning", origin: "arXiv", message: "HTTP 503" }]);
  assert.equal(O.runHealth({ ...d, lastRun: { ...d.lastRun, errors: [] } }).label, "正常");
  assert.equal(O.runHealth({ ...d, running: true }).label, "正在检查…");
  assert.equal(O.runHealth({ ...d, lastRun: null }).label, "尚未运行");
  // A failure newer than the last finished run comes first
  const failed = O.runHealth({ ...d, lastError: { message: "网络连接失败", at: NOW - 10 * 60000 } }, NOW);
  assert.equal(failed.label, "运行失败");
  assert.equal(failed.title, "最近的问题");
  assert.deepEqual(plain(failed.problems[0]), { tone: "critical", origin: "10 分钟前", message: "网络连接失败" });
  assert.equal(O.runHealth({ ...d, lastError: { message: "old", at: RUN_AT - DAY } }).label, "有 1 个问题", "an older failure is superseded");

  const facts = O.scheduleFacts(d, NOW);
  assert.deepEqual(plain(facts.map((f) => f.label)), ["上次", "耗时", "下次", "间隔"]);
  assert.match(facts[0].text, /^2026-10-1[56] \d\d:\d\d（2 小时前）$/);
  assert.equal(facts[1].text, "3 分 12 秒");
  assert.equal(facts[2].text, "约 22 小时后");
  assert.equal(facts[3].text, "每 24 小时自动检查");
  const off = O.scheduleFacts({ ...d, schedule: { ...d.schedule, autoRun: false } }, NOW);
  assert.deepEqual(plain(off.find((f) => f.key === "next")), { key: "next", label: "下次", text: "自动检查已关闭", link: { href: "#settings/general", text: "去开启" } });
  assert.equal(O.scheduleFacts({ ...d, schedule: { ...d.schedule, nextRunAt: NOW - 1 } }, NOW)[2].text, "即将检查");
  // Automatic checks off: no "每 24 小时自动检查" under "自动检查已关闭"
  assert.deepEqual(plain(off.map((f) => f.key)), ["last", "took", "next"]);

  // A run cancelled after its results were stored (no end time): said so, not 正常
  const cancelledRun = { ...d.lastRun, finishedAt: null, errors: [] };
  const cancelled = O.runHealth({ ...d, lastRun: cancelledRun }, NOW);
  assert.deepEqual(plain([cancelled.tone, cancelled.label]), ["warning", "上次运行已取消"]);
  const cancelledFacts = O.scheduleFacts({ ...d, lastRun: cancelledRun }, NOW);
  assert.match(cancelledFacts[0].text, /（2 小时前，已取消）$/);
  assert.ok(!cancelledFacts.some((f) => f.key === "took"));
  assert.equal(O.kpiTiles({ ...d, lastRun: cancelledRun }, NOW)[0].caption, "已取消 · 个性 18 · 顶刊 3 · 关注 2");
  // Runs from before finishedAt existed are not "cancelled"
  const { finishedAt, ...legacy } = d.lastRun;
  assert.ok(finishedAt);
  assert.equal(O.runHealth({ ...d, lastRun: { ...legacy, errors: [] } }, NOW).label, "正常");

  assert.deepEqual(plain(O.sourceStatus({ enabled: false }, true)), { tone: "off", label: "未启用" });
  assert.deepEqual(plain(O.sourceStatus({ enabled: true, error: "x" }, true)), { tone: "critical", label: "出错" });
  assert.deepEqual(plain(O.sourceStatus({ enabled: true }, false)), { tone: "idle", label: "未运行" });
  assert.deepEqual(plain(O.sourceStatus({ enabled: true }, true)), { tone: "good", label: "正常" });
  // Enabled after the last run: it has not run yet
  assert.deepEqual(plain(O.sourceStatus({ enabled: true, ran: false }, true)), { tone: "idle", label: "未运行" });
  assert.deepEqual(plain(O.sourceStatus({ enabled: true, ran: true }, true)), { tone: "good", label: "正常" });
  assert.equal(O.connectionStatus({ ok: null }).label, "未启用");
});

test("KPI tiles, the pending breakdown and the header line", () => {
  const { O } = loadWindow(dashboard());
  const tiles = O.kpiTiles(dashboard(), NOW);
  assert.deepEqual(plain(tiles.map((t) => [t.label, t.value ?? t.text, t.caption])), [
    ["本次新增", 23, "个性 18 · 顶刊 3 · 关注 2"],
    ["必读", 4, "本次新增中的必读"],
    ["已加入 Zotero", 2, "近 60 天"],
    ["上次检查", "2 小时前", tiles[3].caption],
  ]);
  assert.match(tiles[3].caption, /^10-1[56] \d\d:\d\d · 耗时 3 分 12 秒$/);
  // Disabled tracks without new works are left out of the breakdown
  const personalOnly = O.kpiTiles({ ...dashboard(), flagship: { enabled: false }, followed: { enabled: false }, lastRun: { ...dashboard().lastRun, stats: { newRecommendations: 5, newByLabel: { personal: 5 } } } });
  assert.equal(personalOnly[0].caption, "个性 5");
  const first = O.kpiTiles(emptyDashboard(), NOW);
  assert.deepEqual(plain(first.map((t) => t.value ?? t.text)), [null, null, 0, "尚未运行"]);
  assert.equal(first[0].caption, "尚未运行");

  assert.deepEqual(plain(O.pendingParts({ total: 5, personal: 5, flagship: 0, followed: 0 })), [{ track: "personal", label: "个性", count: 5 }]);
  assert.equal(O.pendingParts(dashboard().pending).length, 3);
  assert.equal(O.headerLine(dashboard()), "文库：我的文库 · 2,318 条（2,120 篇有摘要） · 向量化：阿里云 DashScope · text-embedding-v4");
});

// ------------------------------------------------------------------ render

test("the dashboard renders every card in order", async () => {
  const w = loadWindow(dashboard());
  await w.refresh();
  assert.equal(w.calls.dashboard, 1);
  assert.equal(w.$("overview-root").getAttribute("aria-busy"), "false");
  assert.deepEqual(
    cards(w).map((n) => n.className.split(" ").find((c) => c.startsWith("ov-") && c !== "ov-card")),
    ["ov-hero", "ov-kpis", "ov-runs", "ov-status", "ov-funnel", "ov-sources", "ov-topics", "ov-followed"],
  );

  const hero = cardOf(w, "ov-hero");
  assert.equal(one(hero, "hero-value").textContent, "16");
  assert.equal(one(hero, "badge").textContent, "必读 3");
  assert.deepEqual(all(hero, "hero-part").map((p) => p.textContent), ["个性 11", "顶刊 3", "关注 2"]);
  assert.deepEqual(all(hero, "hero-seg").map((s) => s.attributes.style), ["flex-grow: 11", "flex-grow: 3", "flex-grow: 2"]);
  assert.equal(one(hero, "tile-link").attributes.href, "#inbox");

  const kpis = all(cardOf(w, "ov-kpis"), "tile-value").map((n) => n.textContent);
  assert.deepEqual(kpis, ["23", "4", "2", "2 小时前"]);

  const status = cardOf(w, "ov-status");
  assert.equal(one(status, "ov-health-label").textContent, "有 1 个问题");
  assert.deepEqual(all(status, "ov-problem").map((p) => p.textContent), ["arXivHTTP 503"]);

  const sources = all(cardOf(w, "ov-sources"), "ov-source");
  assert.deepEqual(
    sources.map((s) => [one(s, "ov-source-name").textContent, one(s, "ov-source-status").textContent, one(s, "ov-source-count").textContent]),
    [
      ["Crossref", "正常", "1,102 篇"],
      ["arXiv", "出错", "18 篇"],
      ["EarthArXiv", "未启用", ""],
      ["大模型", "未启用", ""],
    ],
  );
  assert.equal(one(sources[1], "ov-source-error").textContent, "HTTP 503");
  assert.equal(all(sources[3], "ov-source-detail").length, 0, "a detail equal to the status is not repeated");

  const followed = cardOf(w, "ov-followed");
  const rows = all(followed, "ov-row");
  assert.deepEqual(rows.map((r) => one(r, "ov-row-title").textContent), ["汪在聪", "Cin-Ty Lee", "无法识别"]);
  assert.equal(one(rows[0], "ov-id").textContent, "ORCID 0000-0002-3584-1673");
  assert.equal(one(rows[1], "ov-id").textContent, "OpenAlex A5023888391");
  assert.equal(one(rows[0], "ov-row-count").textContent, "2 篇待查看");
  assert.equal(one(rows[1], "ov-row-count").textContent, "暂无新论文");
  assert.equal(one(rows[2], "ov-row-detail").textContent, "张伟（没有 ORCID）");

  assert.equal(w.run("shell.subtitles.overview"), "文库：我的文库 · 2,318 条（2,120 篇有摘要） · 向量化：阿里云 DashScope · text-embedding-v4");
});

test("the run chart: legend, columns, tooltips on hover and keyboard focus, table view", async () => {
  const w = loadWindow(dashboard());
  await w.refresh();
  const card = cardOf(w, "ov-runs");
  assert.deepEqual(all(card, "legend-item").map((n) => n.textContent), ["个性推荐", "顶刊地学", "关注作者"]);
  const hits = all(card, "hit");
  assert.equal(hits.length, 14);
  assert.deepEqual(hits.map((h) => h.attributes.tabindex).filter((t) => t === "0").length, 1, "one tab stop per chart");
  assert.equal(hits[13].attributes.tabindex, "0", "the newest run is the tab stop");
  // The mark's name carries what the tooltip shows (必读, problems when there are any)
  assert.match(hits[13].attributes["aria-label"], /个性推荐 14，顶刊地学 0，关注作者 1，合计 15，必读 1$/);
  assert.match(hits[2].attributes["aria-label"], /合计 3，必读 2，1 个问题$/);
  // Only the newest column carries a direct label (its total on the cap)
  assert.deepEqual(all(card, "cap").map((n) => n.textContent), ["15"]);
  // Segments wear the track classes; each column's top segment is a path with the rounded end
  const firstCol = all(card, "col")[0];
  assert.deepEqual(all(firstCol, "seg").map((s) => [s.tagName, s.className]), [["rect", "seg personal"], ["path", "seg flagship"]]);

  // Hover shows the run's values (value first, then the track name) and the total
  const tip = one(card, "chart-tip");
  hits[13].fire("pointerenter");
  assert.equal(tip.hidden, false);
  assert.ok(all(card, "col")[13].classList.contains("active"));
  assert.deepEqual(all(tip, "tip-row").map((r) => [one(r, "tip-key").className, one(r, "tip-value").textContent, one(r, "tip-name").textContent]), [
    ["tip-key personal", "14", "个性推荐"],
    ["tip-key flagship", "0", "顶刊地学"],
    ["tip-key followed", "1", "关注作者"],
  ]);
  assert.equal(one(tip, "tip-foot").textContent, "合计 15 · 必读 1");
  hits[13].fire("pointerleave");
  assert.equal(tip.hidden, true);

  // Keyboard: focus shows the tooltip with the focus ring; arrows move the tab stop
  hits[13].focus();
  assert.ok(all(card, "col")[13].classList.contains("focused"));
  hits[13].fire("keydown", { key: "ArrowLeft" });
  assert.equal(hits[12].attributes.tabindex, "0");
  assert.equal(hits[13].attributes.tabindex, "-1");
  assert.equal(one(tip, "tip-head").textContent.length, 16, "date and time of run 12");
  hits[12].fire("keydown", { key: "Home" });
  assert.equal(hits[0].attributes.tabindex, "0");

  // 表格 swaps the chart (and its legend) for a table with the same numbers, newest first
  const toggle = one(card, "chart-toggle");
  assert.equal(toggle.textContent, "表格");
  toggle.click();
  assert.equal(toggle.attributes["aria-pressed"], "true");
  assert.equal(toggle.textContent, "图表");
  const table = byTag(card, "table")[0];
  assert.ok(table);
  assert.equal(all(card, "legend").length, 0);
  assert.deepEqual(byTag(table, "th").map((n) => n.textContent), ["时间", "个性推荐", "顶刊地学", "关注作者", "合计", "必读", "问题"]);
  const bodyRows = byTag(byTag(table, "tbody")[0], "tr");
  assert.equal(bodyRows.length, 14);
  assert.deepEqual(byTag(bodyRows[0], "td").slice(1).map((n) => n.textContent), ["14", "0", "1", "15", "1", "0"]);
  // The run with a problem (the tooltip's "· 1 个问题") says so in the table too
  assert.deepEqual(byTag(bodyRows[11], "td").slice(1).map((n) => n.textContent), ["3", "0", "0", "3", "2", "1"]);
  // The choice survives a reload; toggling back restores the chart
  await w.refresh({ ...w.view });
  assert.ok(byTag(cardOf(w, "ov-runs"), "table").length);
  one(cardOf(w, "ov-runs"), "chart-toggle").click();
  assert.equal(all(cardOf(w, "ov-runs"), "hit").length, 14);
});

test("the funnel, flagship mini funnel and research directions", async () => {
  const w = loadWindow(dashboard());
  await w.refresh();
  const funnel = cardOf(w, "ov-funnel");
  assert.deepEqual(all(funnel, "label").map((n) => n.textContent), ["抓取", "去重后", "分出顶刊后", "关键词过滤后", "有摘要", "通过相似度门槛", "最近 7 天", "推荐"]);
  assert.deepEqual(all(funnel, "value").map((n) => n.textContent), ["1,284", "1,201−83", "1,037−164", "1,009−28", "871−138", "86−785", "41−45", "30−11"]);
  assert.equal(all(funnel, "bar").length, 8);
  const tip = one(funnel, "chart-tip");
  all(funnel, "hit")[5].focus();
  assert.equal(one(tip, "tip-head").textContent, "通过相似度门槛");
  assert.equal(one(tip, "tip-foot").textContent, "较上一步 −785 · 占抓取 6.7%");
  const mini = one(funnel, "ov-mini");
  assert.equal(mini.attributes["aria-label"], "顶刊：候选 164 → 有摘要 151 → 最近 7 天 120 → 通过 12（灰区 14，大模型复核 14）→ 显示 12");
  assert.equal(all(mini, "ov-step").length, 5);
  assert.equal(all(mini, "ov-step-arrow").length, 4, "arrows between steps only");

  const topics = cardOf(w, "ov-topics");
  assert.equal(one(topics, "card-caption").textContent, "基于文库的语义聚类（共 14 个方向）");
  assert.deepEqual(all(topics, "value").map((n) => n.textContent), ["312", "268", "221"]);
  all(topics, "hit")[0].fire("pointerenter");
  assert.equal(one(one(topics, "chart-tip"), "tip-foot").textContent, "占文库 13.5%");
  one(topics, "chart-toggle").click();
  // The table has the tooltip's share of the library too
  assert.deepEqual(byTag(one(topics, "chart-table"), "th").map((n) => n.textContent), ["研究方向", "文献数", "占文库"]);
  assert.deepEqual(byTag(one(topics, "chart-table"), "td").map((n) => n.textContent), ["斑岩铜矿成矿", "312", "13.5%", "俯冲带流体与氧化还原", "268", "11.6%", "岩浆挥发分", "221", "9.5%"]);

  // The run had no recency window although the settings now have one: no 最近 7 天 bar
  const d = dashboard();
  d.lastRun.stats = { ...d.lastRun.stats, recentDays: 0, afterRecent: 86 };
  const unlimited = loadWindow(d);
  await unlimited.refresh();
  const labels = all(cardOf(unlimited, "ov-funnel"), "label").map((n) => n.textContent);
  assert.ok(!labels.includes("最近 7 天"), labels.join(" | "));
  assert.equal(all(cardOf(unlimited, "ov-funnel"), "ov-step")[2].textContent.startsWith("最近 "), true);
});

test("first run: callout and empty states point to the next step", async () => {
  const w = loadWindow(emptyDashboard());
  await w.refresh();
  const first = cards(w)[0];
  assert.ok(first.className.includes("ov-first"));
  assert.match(first.textContent, /还没有运行过/);
  assert.match(first.textContent, /本地模式/);
  const runButton = byTag(first, "button").find((b) => b.textContent === "立即检查");
  runButton.click();
  assert.equal(w.calls.runNow, 1);

  assert.equal(one(cardOf(w, "ov-hero"), "hero-note").textContent, "运行后这里显示待查看的推荐");
  assert.equal(all(cardOf(w, "ov-hero"), "hero-share").length, 0);
  assert.equal(one(cardOf(w, "ov-runs"), "empty-title").textContent, "运行几次后这里会显示每次新增的推荐");
  assert.equal(one(cardOf(w, "ov-funnel"), "empty-title").textContent, "运行一次后显示每一步筛选后剩下的篇数");
  assert.equal(one(cardOf(w, "ov-status"), "ov-health-label").textContent, "尚未运行");
  assert.equal(one(cardOf(w, "ov-topics"), "empty-title").textContent, "开启大模型后会为每个研究方向生成名称");
  assert.equal(one(cardOf(w, "ov-followed"), "empty-title").textContent, "关注特定作者的新论文");
  // 去设置 opens the matching settings section
  byTag(cardOf(w, "ov-followed"), "button")[0].click();
  assert.equal(w.run("currentRoute()"), "settings/followed");

  // LLM on but nothing clustered yet / labels pending
  const llm = { enabled: true, configured: true, model: "m", topicLabels: true };
  const w2 = loadWindow({ ...emptyDashboard(), llm });
  await w2.refresh();
  assert.equal(one(cardOf(w2, "ov-topics"), "empty-title").textContent, "运行一次后显示");
  const w3 = loadWindow({ ...emptyDashboard(), llm, clusters: 9 });
  await w3.refresh();
  assert.equal(one(cardOf(w3, "ov-topics"), "empty-title").textContent, "下次运行时会为研究方向生成名称");
  const w4 = loadWindow({ ...emptyDashboard(), llm: { ...llm, topicLabels: false } });
  await w4.refresh();
  assert.equal(one(cardOf(w4, "ov-topics"), "empty-title").textContent, "开启「研究方向标签」后会为每个研究方向生成名称");
  // All caught up after a run
  const w5 = loadWindow({ ...dashboard(), pending: { total: 0, personal: 0, flagship: 0, followed: 0, mustRead: 0 } });
  await w5.refresh();
  assert.equal(one(cardOf(w5, "ov-hero"), "hero-note").textContent, "全部看完了");
});

test("测试连接 shows a results card with a status per service", async () => {
  const connections = [
    { key: "embedding", name: "向量化", target: "dashscope", ok: true, detail: "正常，维度 1024" },
    { key: "llm", name: "大模型", target: null, ok: null, detail: "未启用" },
    { key: "elsevier", name: "Elsevier", target: null, ok: false, detail: "Key 无效（HTTP 401）" },
  ];
  const w = loadWindow(dashboard(), { connections });
  await w.refresh();
  const button = w.$("overview-test");
  button.click();
  assert.equal(button.disabled, true, "disabled while testing");
  assert.match(cards(w)[0].textContent, /正在测试/);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(button.disabled, false);
  const card = cards(w)[0];
  assert.ok(card.className.includes("ov-conn"));
  assert.equal(one(card, "card-caption").textContent.endsWith("1 项正常，1 项失败"), true);
  assert.deepEqual(all(card, "ov-row").map((r) => [one(r, "ov-row-icon").className, one(r, "ov-status-label").textContent]), [
    ["ov-row-icon good", "正常"],
    ["ov-row-icon off", "未启用"],
    ["ov-row-icon critical", "失败"],
  ]);
  assert.equal(one(all(card, "ov-row")[2], "ov-error").textContent, "Key 无效（HTTP 401）");
  // Closing removes the card
  byTag(card, "button")[0].click();
  assert.ok(!cards(w)[0].className.includes("ov-conn"));
  assert.equal(w.calls.connections, 1);
});

test("reloads: dimmed while loading, coalesced, skipped for a page switch, errors kept visible", async () => {
  const w = loadWindow(dashboard());
  await w.refresh();
  assert.equal(w.calls.dashboard, 1);
  // Switching back to the page with the same view re-renders without reloading
  await w.refresh(w.view);
  assert.equal(w.calls.dashboard, 1);
  // A new view (after a run or an inbox change) reloads; the old render stays, dimmed
  const loading = w.run("ZotWatchOverview.refresh({ works: [] })");
  assert.ok(w.$("overview-root").classList.contains("refreshing"));
  assert.equal(w.$("overview-root").getAttribute("aria-busy"), "true");
  w.run("ZotWatchOverview.refresh({ works: [] })"); // queued behind the first
  await loading;
  assert.equal(w.calls.dashboard, 3);
  assert.equal(w.$("overview-root").classList.contains("refreshing"), false);

  // A run starting shows up in the status card at once
  for (const fn of w.listeners) fn({ type: "start" });
  assert.equal(one(cardOf(w, "ov-status"), "ov-health-label").textContent, "正在检查…");

  // Failure without a previous render: an error state with 重试
  const failing = loadWindow(dashboard());
  failing.failNext(new Error("library locked"));
  await failing.refresh();
  assert.equal(one(failing.$("overview-root"), "empty-title").textContent, "无法加载概览");
  assert.equal(one(failing.$("overview-root"), "empty-text").textContent, "library locked");
});

test("relative times move on while the page stays open (the shell's minute tick)", async () => {
  const w = loadWindow(dashboard());
  await w.refresh();
  const lastTile = () => one(cardOf(w, "ov-kpis"), "last");
  const facts = () => all(cardOf(w, "ov-status"), "ov-facts")[0].childNodes.map((n) => n.textContent);
  assert.equal(one(lastTile(), "tile-value").textContent, "2 小时前");
  assert.ok(facts().includes("约 22 小时后"));

  // Six hours later, without a reload
  w.clock.now = NOW + 6 * HOUR;
  w.run("ZotWatchOverview.tick()");
  assert.equal(w.calls.dashboard, 1, "no reload");
  assert.equal(one(lastTile(), "tile-value").textContent, "8 小时前");
  assert.ok(facts().includes("约 16 小时后"), facts().join(" | "));
  assert.match(facts()[1], /（8 小时前）$/);

  // Past the next run: 即将检查
  w.clock.now = NOW + 23 * HOUR;
  w.run("ZotWatchOverview.tick()");
  assert.ok(facts().includes("即将检查"));
});
