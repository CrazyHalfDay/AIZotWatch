// The 调研 page (addon/content/survey.js, with main.js and search.js) in a
// fake DOM: the pure helpers (citations, keywords, validation, provenance,
// sorting, statuses), the new-survey form (validation, seeds from the library
// search and the Zotero selection, the item-menu draft, cost hints, start),
// the history and the running survey with live progress and cancel, the
// result view (review with clickable [n] citations, references, candidates,
// themes, adding papers with failures isolated per row), the toolbar (保存到
// Zotero with progress and stop, 导出 Markdown, 重新生成综述, 继续, 删除), and
// unknown surveys, load failures and stale answers. Survey data comes from
// the preview fixtures (scripts/preview-survey-fixtures.mjs).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import { createSurveyFixtures } from "../scripts/preview-survey-fixtures.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const code = ["main.js", "search.js", "survey.js"].map((f) => [f, readFileSync(join(root, "addon/content", f), "utf8")]);
// Values created inside the VM have foreign prototypes; compare them structurally
const plain = (x) => JSON.parse(JSON.stringify(x));
const clone = (x) => (x === undefined ? undefined : JSON.parse(JSON.stringify(x)));
const flush = async (n = 8) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r));
};
const DONE = "mvq2k3a01x";
const CANCELLED = "mvn8w1k02b";
const FAILED = "mvk3p9x07c";

// ------------------------------------------------------------- fake DOM

class FakeText {
  constructor(text) {
    this.textContent = String(text);
    this.parentNode = null;
  }
}

class FakeNode {
  constructor(doc, tag) {
    this.ownerDocument = doc;
    this.tagName = tag.toUpperCase();
    this.localName = tag;
    this.childNodes = [];
    this.parentNode = null;
    this.attributes = {};
    this.listeners = {};
    this.ownText = "";
    this.checked = false;
    this.style = {};
    this.scrollTop = 0;
    this.scrolled = 0;
    const node = this;
    this.classList = {
      add: (c) => node.classList.toggle(c, true),
      remove: (c) => node.classList.toggle(c, false),
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
  get className() {
    return this.attributes.class || "";
  }
  set className(v) {
    this.attributes.class = String(v);
  }
  get id() {
    return this.attributes.id || "";
  }
  get value() {
    return this.currentValue ?? this.attributes.value ?? "";
  }
  set value(v) {
    this.currentValue = String(v);
  }
  get type() {
    return this.attributes.type || "";
  }
  get hidden() {
    return "hidden" in this.attributes;
  }
  set hidden(v) {
    if (v) this.attributes.hidden = "";
    else delete this.attributes.hidden;
  }
  get disabled() {
    return "disabled" in this.attributes;
  }
  set disabled(v) {
    if (v) this.attributes.disabled = "";
    else delete this.attributes.disabled;
  }
  get textContent() {
    return this.ownText + this.childNodes.map((n) => n.textContent).join("");
  }
  set textContent(value) {
    this.replaceChildren();
    this.ownText = String(value);
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
  }
  getAttribute(k) {
    return this.attributes[k] ?? null;
  }
  hasAttribute(k) {
    return k in this.attributes;
  }
  removeAttribute(k) {
    delete this.attributes[k];
  }
  addEventListener(type, fn) {
    (this.listeners[type] ||= []).push(fn);
  }
  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn);
  }
  /** Fire an event at this node; it bubbles to the ancestors and the document. */
  dispatch(type, extra = {}) {
    const event = {
      type,
      target: this,
      defaultPrevented: false,
      preventDefault() {
        event.defaultPrevented = true;
      },
      stopPropagation() {
        event.stopped = true;
      },
      ...extra,
    };
    for (let n = this; n && !event.stopped; n = n.parentNode) for (const fn of n.listeners?.[type] || []) fn(event);
    if (!event.stopped) for (const fn of this.ownerDocument.listeners[type] || []) fn(event);
    return event;
  }
  click() {
    if (this.disabled) return null;
    return this.dispatch("click");
  }
  // External text never goes through HTML parsing: any use fails the test
  set innerHTML(value) {
    throw new Error(`innerHTML is not allowed (${String(value).slice(0, 40)})`);
  }
  set outerHTML(value) {
    throw new Error(`outerHTML is not allowed (${String(value).slice(0, 40)})`);
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  blur() {
    if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = this.ownerDocument.body;
  }
  scrollIntoView() {
    this.scrolled++;
  }
  append(...nodes) {
    for (const n of nodes) {
      const child = typeof n === "string" ? new FakeText(n) : n;
      if (child.parentNode) child.parentNode.childNodes = child.parentNode.childNodes.filter((c) => c !== child);
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }
  replaceChildren(...nodes) {
    for (const c of this.childNodes) c.parentNode = null;
    this.ownText = "";
    this.childNodes = [];
    this.append(...nodes);
  }
  remove() {
    if (!this.parentNode) return;
    this.parentNode.childNodes = this.parentNode.childNodes.filter((c) => c !== this);
    this.parentNode = null;
  }
  matches(selector) {
    return selector.split(",").some((s) => matchOne(this, s.trim()));
  }
  closest(selector) {
    for (let n = this; n && n instanceof FakeNode; n = n.parentNode) if (n.matches(selector)) return n;
    return null;
  }
  querySelectorAll(selector) {
    return walk(this).filter((n) => n !== this && n instanceof FakeNode && n.matches(selector));
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

/** Tiny selector matcher: tag, .class, [attr], [attr="v"], :checked and combinations without spaces. */
function matchOne(node, selector) {
  const m = /^([a-z]+)?((?:\.[\w-]+)*)((?:\[[\w-]+(?:="[^"]*")?\])*)(:checked)?$/.exec(selector);
  if (!m) return false;
  const [, tag, classes, attrs, checked] = m;
  if (tag && node.localName !== tag) return false;
  for (const c of classes.split(".").filter(Boolean)) if (!node.classList.contains(c)) return false;
  for (const [, name, value] of attrs.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)) {
    if (!(name in node.attributes)) return false;
    if (value !== undefined && node.attributes[name] !== value) return false;
  }
  if (checked && !node.checked) return false;
  return true;
}

function walk(node, out = []) {
  out.push(node);
  for (const child of node.childNodes || []) walk(child, out);
  return out;
}
const all = (node) => walk(node).filter((n) => n instanceof FakeNode);
const byClass = (node, cls) => all(node).filter((n) => n.classList.contains(cls));
const byKey = (node, key) => all(node).find((n) => n.getAttribute("data-key") === key) || null;
const byText = (node, tag, text) => all(node).find((n) => n.localName === tag && n.textContent.trim() === text) || null;
const allByText = (node, tag, text) => all(node).filter((n) => n.localName === tag && n.textContent.trim() === text);
const byId = (node, id) => all(node).find((n) => n.id === id) || null;
const has = (node, text) => node.textContent.includes(text);

function type(input, value) {
  input.value = value;
  input.dispatch("input");
}

// --------------------------------------------------------------- window

/**
 * main.js + search.js + survey.js with a fake Zotero.ZotWatch over the
 * preview fixtures, opened on `hash`. `control` steers the fake API.
 */
async function openPage({ hash = "#survey", options = {}, wide = false, draft = null, now = Date.now() } = {}) {
  const fixtures = createSurveyFixtures(now);
  const doc = { listeners: {}, activeElement: null };
  const elements = new Map();
  doc.getElementById = (id) => {
    if (!elements.has(id)) {
      const node = new FakeNode(doc, "div");
      node.setAttribute("id", id);
      elements.set(id, node);
    }
    return elements.get(id);
  };
  doc.createElement = (tag) => new FakeNode(doc, tag);
  doc.createElementNS = (ns, tag) => new FakeNode(doc, tag);
  doc.createTextNode = (text) => new FakeText(text);
  doc.querySelectorAll = () => [];
  doc.querySelector = () => null;
  doc.addEventListener = (type, fn) => (doc.listeners[type] ||= []).push(fn);
  doc.body = new FakeNode(doc, "body");
  doc.documentElement = new FakeNode(doc, "html");
  doc.activeElement = doc.body;

  const calls = [];
  const listeners = new Set();
  const control = {
    options: { ...clone(fixtures.surveyOptions), ...options },
    list: clone(fixtures.surveys),
    surveys: clone(fixtures.surveyDetails),
    selected: [],
    library: clone(fixtures.libraryResults),
    draft,
    confirmReply: true,
    pickPath: null,
    gates: {},
    errors: {},
    addErrors: {},
    showResult: true,
    saveResult: null,
    // Files that exist (IOUtils.exists)
    existing: [],
  };
  /** Wait at `name` when a gate is set (a promise the test resolves). */
  const gate = async (name) => {
    if (control.gates[name]) await control.gates[name];
  };
  const fail = (name) => {
    if (control.errors[name]) throw new Error(control.errors[name]);
  };
  const api = {
    version: "0.4.0",
    lastPage: null,
    addListener: (fn) => listeners.add(fn),
    removeListener: (fn) => listeners.delete(fn),
    async getView() {
      return { works: [], lastRun: null, history: [], running: false, progress: null, lastError: null, schedule: null, version: "0.4.0" };
    },
    getSurveyOptions: () => clone(control.options),
    surveyEstimate(depth, opts) {
      calls.push(["surveyEstimate", depth, clone(opts)]);
      return { text: `估算：${depth}` };
    },
    async listSurveys() {
      calls.push(["listSurveys"]);
      await gate("list");
      fail("list");
      return clone(control.list);
    },
    async getSurvey(id) {
      calls.push(["getSurvey", id]);
      await gate(`get:${id}`);
      fail("get");
      return clone(control.surveys[id]) || null;
    },
    // Like the addon, a survey that starts runs at once (getSurveyOptions().running)
    async startSurvey(input) {
      calls.push(["startSurvey", clone(input)]);
      fail("start");
      control.options.running = { id: "newsurvey01", progress: null };
      return { id: "newsurvey01" };
    },
    cancelSurvey(id) {
      calls.push(["cancelSurvey", id]);
      return true;
    },
    async resumeSurvey(id) {
      calls.push(["resumeSurvey", id]);
      fail("resume");
      control.options.running = { id, progress: null };
      return { id };
    },
    async regenerateReport(id) {
      calls.push(["regenerateReport", id]);
      control.options.running = { id, progress: null };
      return { id, from: "write" };
    },
    async deleteSurvey(id) {
      calls.push(["deleteSurvey", id]);
      return true;
    },
    async saveSurveyToZotero(id, opts) {
      calls.push(["saveSurveyToZotero", id, clone(opts)]);
      await gate("save");
      fail("save");
      return control.saveResult;
    },
    cancelSurveySave(id) {
      calls.push(["cancelSurveySave", id]);
      return true;
    },
    async addSurveyPaper(id, paperId) {
      calls.push(["addSurveyPaper", id, paperId]);
      await gate(`add:${paperId}`);
      if (control.addErrors[paperId]) throw new Error(control.addErrors[paperId]);
      return { itemID: 77, itemKey: "NEWKEY", existing: false };
    },
    async showItemByKey(key) {
      calls.push(["showItemByKey", key]);
      return control.showResult;
    },
    async exportSurveyMarkdown(id, path = null) {
      calls.push(["exportSurveyMarkdown", id, path]);
      return { filename: "ZotWatch 调研 - 主题 - 2026-10-17.md", text: "# x\n" };
    },
    async searchLibrary(query, opts) {
      calls.push(["searchLibrary", query, clone(opts)]);
      return clone(control.library.filter((i) => i.title.toLowerCase().includes(query.toLowerCase())));
    },
    getSelectedItems() {
      calls.push(["getSelectedItems"]);
      return clone(control.selected);
    },
    takeSurveyDraft() {
      const d = control.draft;
      control.draft = null;
      return clone(d);
    },
    openUrl: (url) => calls.push(["openUrl", url]),
  };
  const emit = (event) => [...listeners].forEach((fn) => fn(event));

  // Manual clock: debounces, toasts and highlights run only when advanced
  let clockNow = 0;
  let nextTimer = 1;
  const timers = new Map();
  const clock = {
    setTimeout(fn, ms = 0) {
      const id = nextTimer++;
      timers.set(id, { fn, at: clockNow + ms });
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
    async advance(ms) {
      const until = clockNow + ms;
      for (;;) {
        const due = [...timers.entries()].filter(([, t]) => t.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        timers.delete(due[0]);
        clockNow = due[1].at;
        due[1].fn();
        await flush(2);
      }
      clockNow = until;
    },
  };

  const confirms = [];
  const winListeners = {};
  const mql = { matches: wide, listeners: [], addEventListener(type, fn) {
    this.listeners.push(fn);
  } };
  class FilePicker {
    constructor() {
      this.modeSave = 1;
      this.returnOK = 0;
      this.returnCancel = 1;
      this.returnReplace = 2;
    }
    init(win, title, mode) {
      calls.push(["FilePicker", title, mode]);
    }
    appendFilter() {}
    async show() {
      calls.push(["FilePicker.show", this.defaultString, this.defaultExtension]);
      this.file = control.pickPath;
      return control.pickPath ? this.returnOK : this.returnCancel;
    }
  }
  const errors = [];
  const Zotero = { ZotWatch: api, isMac: false, debug() {}, logError: (e) => errors.push(e) };
  const window = {
    addEventListener: (type, fn) => (winListeners[type] ||= []).push(fn),
    confirm: (text) => (confirms.push(text), control.confirmReply),
    matchMedia: (query) => {
      calls.push(["matchMedia", query]);
      return mql;
    },
    close() {},
  };
  const context = vm.createContext({
    ChromeUtils: { importESModule: (url) => (url.endsWith("filePicker.mjs") ? { FilePicker } : { Zotero }) },
    window,
    document: doc,
    IOUtils: { exists: async (path) => control.existing.includes(path) },
    location: { hash },
    Intl,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: () => 0,
    clearInterval() {},
  });
  for (const [file, src] of code) vm.runInContext(src, context, { filename: file });
  const run = (src) => vm.runInContext(src, context);
  for (const fn of winListeners.load || []) fn({ type: "load" });
  await flush();
  const page = run("ZotWatchSurvey");
  const rootNode = doc.getElementById("survey-root");
  return {
    page,
    run,
    api,
    emit,
    calls,
    control,
    clock,
    confirms,
    errors,
    mql,
    doc,
    root: rootNode,
    fixtures,
    navigate: async (route) => {
      run(`window.ZotWatchMain.navigate(${JSON.stringify(route)})`);
      await flush();
    },
    toasts: () => doc.getElementById("toasts").textContent,
    callsOf: (name) => calls.filter((c) => c[0] === name),
  };
}

// --------------------------------------------------------------- helpers

test("survey page helpers: citations, keywords, validation, provenance, sorting, statuses", async () => {
  const { page, fixtures } = await openPage();
  assert.deepEqual(plain(page.citationParts("A [1] b [2, 3]。")), [{ text: "A " }, { refs: [1] }, { text: " b " }, { refs: [2, 3] }, { text: "。" }]);
  assert.deepEqual(plain(page.citationParts("[4，5]")), [{ refs: [4, 5] }]);
  assert.deepEqual(plain(page.citationParts("no [x] [ ] 2021]")), [{ text: "no [x] [ ] 2021]" }]);
  assert.deepEqual(plain(page.citationParts(null)), []);
  const report = fixtures.surveyDetails[DONE].report;
  assert.deepEqual(plain(page.citedNumbers(report)).sort((a, b) => a - b), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  assert.deepEqual(plain(page.citedNumbers({ kind: "list" })), []);
  assert.deepEqual(plain(page.citedNumbers(null)), []);

  assert.deepEqual(plain(page.parseKeywords("a, b；c\nA, ,d、e")), ["a", "b", "c", "d", "e"]);
  assert.equal(page.parseKeywords(Array.from({ length: 15 }, (_, i) => `k${i}`).join(",")).length, 12);

  const now = Date.UTC(2026, 9, 17);
  const base = { topic: "T", keywords: "k", seedKeys: [], yearFrom: null, sources: { openalex: true, s2: false, crossref: false } };
  const problem = (input) => page.validateInput({ ...base, ...input }, now)?.field ?? null;
  assert.equal(problem({}), null);
  assert.equal(problem({ topic: "  " }), "topic");
  assert.equal(problem({ keywords: " , " }), "keywords");
  assert.equal(problem({ keywords: "", seedKeys: ["K1"] }), null, "seeds alone are enough");
  for (const year of [1700, 2027, 2010.5, Number.NaN]) assert.equal(problem({ yearFrom: year }), "year", String(year));
  assert.equal(problem({ yearFrom: 2010 }), null);
  assert.equal(problem({ sources: { openalex: false, s2: false, crossref: false } }), "sources");
  assert.match(page.validateInput({ ...base, yearFrom: 1 }, now).message, /1800–2026/);

  const chips = page.provenance(["rec", "ref:seed2", "search", "cites:seed1", "ref:seed1", "seed", 5, "cites:seed3", "ref:seedX"]);
  assert.deepEqual(plain(chips.map((c) => [c.key, c.label])), [["seed", "种子"], ["search", "检索"], ["cites", "引用"], ["ref", "被引"], ["rec", "推荐"]]);
  assert.equal(chips[2].title, "引用了种子文献 1、3");
  assert.equal(chips[3].title, "被种子文献 1、2 引用（在其参考文献中）");
  assert.deepEqual(plain(page.provenance(null)), []);

  const pool = [
    { id: "a", rank: 1, scores: { total: 0.9 }, year: 2019, citations: 10, title: "Beta", inLibrary: true },
    { id: "b", rank: 2, scores: { total: 0.8 }, year: null, citations: 50, title: "alpha", read: { findings: "x" } },
    { id: "c", rank: 3, scores: {}, year: 2023, title: "Gamma" },
    null,
    "junk",
  ];
  const ids = (opts) => plain(page.candidateRows(pool, opts).map((p) => p.id));
  assert.deepEqual(ids(), ["a", "b", "c"], "by score, a missing score last");
  assert.deepEqual(ids({ sort: "year" }), ["c", "a", "b"]);
  assert.deepEqual(ids({ sort: "year", dir: "asc" }), ["a", "c", "b"], "missing values stay last");
  assert.deepEqual(ids({ sort: "citations" }), ["b", "a", "c"]);
  assert.deepEqual(ids({ sort: "title", dir: "asc" }), ["b", "a", "c"]);
  assert.deepEqual(ids({ filter: "new" }), ["b", "c"]);
  assert.deepEqual(ids({ filter: "read" }), ["b"]);
  assert.deepEqual(plain(page.candidateRows(undefined)), []);

  const label = (entry) => plain(page.statusOf(entry));
  assert.deepEqual(label({ status: "running", progress: { label: "精读", current: 17, total: 40 } }), { tone: "running", label: "精读 17/40", title: null });
  assert.equal(label({ status: "running", progress: null }).label, "进行中");
  assert.equal(label({ status: "done" }).label, "已完成");
  assert.deepEqual(label({ status: "failed", error: "HTTP 429" }), { tone: "critical", label: "失败", title: "HTTP 429" });
  assert.equal(label({ status: "failed", error: { stage: "expand", message: "boom" } }).title, "boom");
  assert.equal(label({ status: "cancelled", interrupted: true }).label, "已中断");
  assert.equal(label({ status: "cancelled" }).tone, "off");

  const stages = fixtures.surveyOptions.stages;
  assert.deepEqual(plain(page.stepStates(stages, { index: 2 })).slice(0, 4), ["done", "done", "current", "todo"]);
  assert.equal(plain(page.stepStates(stages, null, { stage: "read", completed: stages.map((s) => s.key) })).join(","), "done,done,done,done,done,done,current,todo,todo", "later stages are to do while running");
  assert.equal(page.lastStageLabel({ completed: ["seeds", "queries", "expand"] }), "扩展");
  assert.equal(page.lastStageLabel({ completed: [] }), "");

  assert.equal(page.linkOf({ doi: "https://doi.org/10.1016/j.x" }), "https://doi.org/10.1016/j.x");
  assert.equal(page.linkOf({ doi: "doi:10.1016/j.x" }), "https://doi.org/10.1016/j.x");
  assert.equal(page.linkOf({ doi: "not a doi", url: "javascript:alert(1)" }), null, "only http(s) links");
  assert.equal(page.linkOf({ url: "https://arxiv.org/abs/1" }), "https://arxiv.org/abs/1");
  assert.equal(page.authorsText(["A", { name: "B" }, null, "C", "D"]), "A, B, C 等");
  assert.equal(page.authorsText(["A"], 5), "A 等");

  assert.equal(
    page.saveResultText({ collectionName: "ZotWatch 调研 · T", total: 5, linked: 1, imported: 3, failed: 1, skipped: 0, cancelled: true }),
    "已保存到「ZotWatch 调研 · T」：笔记已写入，1 篇已在文库的文献加入分类，导入 3 篇，1 篇导入失败；已停止导入，其余文献未加入。",
  );
  assert.equal(page.saveResultText({ collectionName: "C", linked: 0, imported: 0 }), "已保存到「C」：笔记已写入。");

  const lines = new Map(plain(page.processLines(fixtures.surveyDetails[DONE])));
  assert.match(lines.get("检索词"), /copper isotope fractionation porphyry；/);
  assert.match(lines.get("扩展"), /找到 21 篇不重复的文献（各来源返回 OpenAlex 24 条、Semantic Scholar 9 条、Crossref 5 条），请求 31 次（1 次失败），3 篇早于起始年份被排除，候选池保留 18 篇/);
  assert.equal(lines.get("精读"), "精读 12 篇，其中 2 篇来自缓存");
  assert.equal(lines.get("排序"), "按与主题和种子的相似度、与种子的关联、被引次数和新近程度为 17 篇打分");
  assert.equal(lines.get("综述"), "3 节");
  assert.deepEqual([...lines.keys()], ["种子", "检索词", "关键术语", "扩展", "去重", "排序", "补全", "精读", "主题", "综述"], "in stage order");
  assert.equal(new Map(plain(page.processLines({ stages: { write: { kind: "review", sections: 4, failed: 1, unverified: 2 } } }))).get("综述"), "4 节，1 节生成失败，2 节没有有效引用");
  assert.equal(new Map(plain(page.processLines({ stages: { write: { kind: "list", groups: 3 } } }))).get("综述"), "未启用大模型，按 3 组列出候选文献");
  assert.deepEqual(plain(page.processLines({ stages: "garbage", queries: "x" })), []);
  assert.deepEqual(plain(page.processLines({ stages: { read: { skipped: true } } })), [["精读", "未启用大模型，跳过"]]);
});

// ------------------------------------------------------------------ form

test("new survey: validation, seeds from the library and the selection, cost hints and start", async () => {
  const w = await openPage({ options: { semanticScholarKey: false } });
  const { root, calls, control } = w;
  const form = byClass(root, "sv-form")[0];
  assert.ok(form, "the form");
  assert.equal(byClass(root, "sv-hist").length, 4, "the history");
  assert.equal(w.doc.getElementById("survey-new").hidden, true, "新调研 only on a survey");

  // Semantic Scholar needs its key: switched off and disabled with a link to the settings
  const s2 = byId(root, "sv-source-s2");
  assert.equal(s2.disabled, true);
  assert.equal(s2.checked, false);
  const s2Note = s2.parentNode.querySelector(".sv-source-note");
  assert.equal(s2Note.hidden, false);
  assert.equal(s2Note.querySelector("a").getAttribute("href"), "#settings/apis");
  assert.equal(byId(root, "sv-source-openalex").checked, true);
  assert.equal(byClass(root, "sv-llm-note")[0].hidden, true, "the LLM is on");

  // Validation: a topic, then keywords or a seed
  const start = byId(root, "sv-start");
  start.click();
  await flush();
  const error = byClass(root, "sv-form-error")[0];
  assert.equal(error.textContent, "请填写调研主题");
  assert.equal(byId(root, "sv-topic").getAttribute("aria-invalid"), "true");
  assert.equal(w.doc.activeElement, byId(root, "sv-topic"));
  type(byId(root, "sv-topic"), "  斑岩铜矿成矿流体  ");
  assert.equal(error.hidden, true, "typing clears the error");
  start.click();
  await flush();
  assert.equal(error.textContent, "请至少填写一个关键词或选择一篇种子文献");
  assert.equal(w.callsOf("startSurvey").length, 0);

  // 使用 Zotero 中选中的条目: nothing selected, then two items (one already a seed after the second click)
  const useSelected = byId(root, "sv-use-selected");
  useSelected.click();
  await flush();
  assert.equal(byClass(root, "sv-seed-status")[0].textContent, "Zotero 主窗口中没有选中的文献条目（笔记和附件之外的条目）。");
  control.selected = [control.library[0], control.library[1]];
  useSelected.click();
  await flush();
  assert.deepEqual(byClass(root, "sv-seed-title").map((n) => n.textContent), [control.library[0].title, control.library[1].title]);
  assert.equal(byClass(root, "sv-seed-status")[0].textContent, "已添加 2 篇种子文献");
  assert.equal(byId(root, "sv-topic").value, "  斑岩铜矿成矿流体  ", "a typed topic is kept");
  useSelected.click();
  await flush();
  assert.equal(byClass(root, "sv-seed-status")[0].textContent, "选中的条目都已在种子文献中");
  // Removing a seed
  byClass(root, "sv-seed-remove")[1].click();
  assert.equal(byClass(root, "sv-seed").length, 1);

  // The library search box adds seeds; added ones are marked 已添加
  const search = byId(root, "sv-seed-search");
  type(search, "porphyry");
  await w.clock.advance(400);
  await flush();
  assert.deepEqual(w.callsOf("searchLibrary").at(-1), ["searchLibrary", "porphyry", { limit: 20 }]);
  const results = byClass(root, "sb-result");
  assert.equal(results.length, 2);
  assert.equal(results[0].querySelector(".sb-action").textContent, "已添加", "SEEDKEY1 is a seed already");
  assert.equal(results[0].querySelector(".sb-action").disabled, true);
  const add = results[1].querySelector(".sb-action");
  assert.equal(add.textContent, "添加");
  add.click();
  await flush();
  assert.equal(byClass(root, "sv-seed").length, 2);
  assert.equal(add.textContent, "已添加");
  assert.equal(add.disabled, true);

  // Depth: the cost hint follows the depth and the form
  const hint = byClass(root, "sv-estimate")[0];
  assert.equal(hint.textContent, "候选池最多 400 篇 · 精读 40 篇 · 估算：standard");
  byKey(root, "depth:deep").click();
  assert.equal(hint.textContent, "候选池最多 1,000 篇 · 精读 80 篇 · 估算：deep");
  assert.deepEqual(calls.filter((c) => c[0] === "surveyEstimate").at(-1), ["surveyEstimate", "deep", { seeds: 2, sources: { openalex: true, s2: false, crossref: true }, keywords: 0 }]);

  // A start year out of range, then a valid start
  type(byId(root, "sv-year"), "1700");
  start.click();
  await flush();
  assert.match(error.textContent, /起始年份应在 1800–\d{4} 之间/);
  type(byId(root, "sv-year"), "2015");
  type(byId(root, "sv-keywords"), "copper isotopes, ore fluids");
  assert.equal(byClass(root, "sv-count").find((n) => n.textContent === "2 个")?.textContent, "2 个");
  start.click();
  await flush();
  assert.deepEqual(w.callsOf("startSurvey")[0][1], {
    topic: "斑岩铜矿成矿流体",
    description: "",
    keywords: "copper isotopes, ore fluids",
    seedKeys: [control.library[0].key, control.library[2].key],
    depth: "deep",
    yearFrom: 2015,
    sources: { openalex: true, s2: false, crossref: true },
  });
  assert.equal(w.api.lastPage, "survey/newsurvey01", "the new survey is shown");
  assert.equal(w.doc.getElementById("nav-survey-dot").hidden, false, "the sidebar shows a survey running");

  // Back on the list the form is empty for the next survey; depth and sources stay as chosen
  await w.navigate("survey");
  for (const id of ["sv-topic", "sv-description", "sv-keywords", "sv-year", "sv-seed-search"]) assert.equal(byId(root, id).value, "", id);
  assert.equal(byClass(root, "sv-seed").length, 0);
  assert.equal(byClass(root, "sv-seed-status")[0].hidden, true);
  assert.equal(byKey(root, "depth:deep").getAttribute("aria-pressed"), "true");
  assert.equal(byId(root, "sv-source-crossref").checked, true);

  // A refused start (another survey runs) is shown on the form, which keeps its input
  control.errors.start = "已有一项调研正在进行，请等待它完成或先取消";
  const again = byId(root, "sv-start");
  w.control.options.running = null;
  w.emit({ type: "survey", action: "cancelled", id: "newsurvey01" });
  await flush();
  assert.equal(again.disabled, false);
  type(byId(root, "sv-topic"), "第二项调研");
  type(byId(root, "sv-keywords"), "slab fluids");
  again.click();
  await flush();
  assert.equal(byClass(root, "sv-form-error")[0].textContent, "已有一项调研正在进行，请等待它完成或先取消");
  assert.equal(w.api.lastPage, "survey");
});

test("the item context menu's draft fills the form; the form survives reloads", async () => {
  const seeds = [
    { key: "SEEDKEY1", title: "Copper isotope fractionation", year: 2019, creators: ["Li Wang"] },
    { key: "SEEDKEY2", title: "Sulfur isotopes of arc magmas", year: 2020, creators: [] },
  ];
  const w = await openPage({ draft: { topic: "Copper isotope fractionation", topics: seeds.map((s) => s.title), seeds } });
  const { root } = w;
  assert.equal(byId(root, "sv-topic").value, "Copper isotope fractionation");
  assert.deepEqual(byClass(root, "sv-seed-title").map((n) => n.textContent), seeds.map((s) => s.title));
  assert.equal(byClass(root, "sv-seed-status")[0].textContent, "已带入 Zotero 中选中的 2 篇种子文献");
  assert.equal(w.control.draft, null, "taken once");
  assert.equal(w.doc.activeElement, byId(root, "sv-topic"));

  // Typing survives a reload of the list (the form stays mounted)
  const keywords = byId(root, "sv-keywords");
  keywords.focus();
  type(keywords, "copper");
  w.emit({ type: "survey", action: "deleted", id: "other00001" });
  await flush();
  assert.equal(byId(root, "sv-keywords"), keywords);
  assert.equal(w.doc.activeElement, keywords);
  assert.equal(keywords.value, "copper");

  // A later draft replaces the seeds (not the keywords) when the window is switched to 调研
  w.control.draft = { topic: "", seeds: [seeds[1]] };
  await w.navigate("inbox");
  await w.navigate("survey");
  assert.deepEqual(byClass(root, "sv-seed-title").map((n) => n.textContent), [seeds[1].title]);
  assert.equal(byId(root, "sv-topic").value, "Copper isotope fractionation", "an empty draft topic keeps the typed one");
  assert.equal(keywords.value, "copper");
  // A draft of items outside the profile library brings no seeds
  w.control.draft = { topic: "x", seeds: [] };
  await w.navigate("overview");
  await w.navigate("survey");
  assert.equal(byClass(root, "sv-seed").length, 0);
  assert.equal(byClass(root, "sv-seed-status")[0].textContent, "选中的条目不在 ZotWatch 使用的文库中");
});

// ------------------------------------------------------- history, running

test("history and the running survey: live progress, cancel, done, nav dot", async () => {
  const progress = { stage: "read", index: 6, label: "精读", message: "精读 17/40：Fluid exsolution", current: 17, total: 40, counts: { seeds: 2, candidates: 371, read: 17, references: 0 } };
  const runningEntry = { id: "runid00001", topic: "岩浆氧逸度", createdAt: Date.now() - 60000, updatedAt: Date.now(), status: "running", depth: "quick", stage: "read", counts: progress.counts, error: null, hasReport: false, llm: true, interrupted: false, progress };
  const w = await openPage({ options: { running: { id: "runid00001", progress } } });
  w.control.list.unshift(runningEntry);
  await w.navigate("overview");
  await w.navigate("survey");
  const { root, doc } = w;
  assert.equal(doc.getElementById("nav-survey-dot").hidden, false);
  const card = byClass(root, "sv-running-card")[0];
  assert.ok(card);
  assert.equal(byClass(card, "sv-running-topic")[0].textContent, "岩浆氧逸度");
  assert.equal(byClass(card, "sv-running-topic")[0].getAttribute("href"), "#survey/runid00001");
  assert.equal(byClass(card, "sv-running-stage")[0].textContent, "精读 17/40");
  assert.equal(byClass(card, "progress-fill")[0].style.width, "43%");
  // The history lists the others with their statuses
  const rows = byClass(root, "sv-hist");
  assert.equal(rows.length, 4, "the running survey is not repeated");
  assert.deepEqual(rows.map((r) => byClass(r, "sv-hist-status")[0].textContent), ["已完成", "已完成", "已取消", "失败"]);
  assert.match(byClass(rows[0], "sv-hist-meta")[0].textContent, /^\d+ 小时前 · 深度：标准 · 种子 2 · 候选 17 · 精读 12$/);
  assert.match(byClass(rows[1], "sv-hist-meta")[0].textContent, / · 未使用大模型$/);
  assert.equal(rows[3].getAttribute("title"), "所有数据源都请求失败：OpenAlex 免费额度已用完（HTTP 429）");
  assert.equal(byId(root, "sv-start").disabled, true, "one survey at a time");
  assert.match(byClass(root, "sv-start-note")[0].textContent, /已有一项调研正在进行/);

  // Progress events update the card in place (no reload)
  const loads = w.callsOf("listSurveys").length;
  w.emit({ type: "survey", action: "progress", id: "runid00001", progress: { ...progress, current: 30 } });
  assert.equal(byClass(card, "sv-running-stage")[0].textContent, "精读 30/40");
  assert.equal(byClass(card, "progress-fill")[0].style.width, "75%");
  assert.equal(w.callsOf("listSurveys").length, loads);

  // 取消 on the card
  byKey(card, "cancel-running").click();
  assert.deepEqual(w.callsOf("cancelSurvey").at(-1), ["cancelSurvey", "runid00001"]);

  // Done: the dot goes, a toast names the survey, the list reloads and the form is enabled
  w.control.options.running = null;
  w.control.list[0] = { ...runningEntry, status: "done", progress: null, hasReport: true };
  w.emit({ type: "survey", action: "done", id: "runid00001", status: "done" });
  await flush();
  assert.equal(doc.getElementById("nav-survey-dot").hidden, true);
  assert.match(w.toasts(), /调研完成：岩浆氧逸度/);
  assert.equal(byClass(root, "sv-running-card").length, 0);
  assert.equal(byClass(root, "sv-hist").length, 5);
  assert.equal(byId(root, "sv-start").disabled, false);

  // An error event toasts its message
  w.emit({ type: "survey", action: "error", id: "runid00001", message: "OpenAlex 免费额度已用完" });
  assert.match(w.toasts(), /调研失败：OpenAlex 免费额度已用完/);

  // An empty history
  w.control.list = [];
  await w.navigate("inbox");
  await w.navigate("survey");
  assert.equal(byClass(root, "sv-hist").length, 0);
  assert.ok(has(root, "还没有调研记录"));
});

test("the running view: stepper, counters, cancel; it turns into the result when done", async () => {
  const w = await openPage({ hash: `#survey/${CANCELLED}` });
  const { root, control } = w;
  // A cancelled survey: the callout and 继续 (save / export / regenerate need a result)
  assert.equal(byClass(root, "sv-title")[0].textContent, "俯冲带流体的氧化还原状态");
  assert.ok(has(root, "调研已取消"));
  assert.ok(has(root, "已完成到「补全」，「继续」会从下一步开始。"));
  for (const key of ["save", "export", "regenerate"]) assert.equal(byKey(root, key), null, key);
  assert.ok(byKey(root, "delete"));
  // Without a review yet, the candidates found so far are shown first
  assert.equal(byKey(root, "tab:candidates").getAttribute("aria-pressed"), "true");
  assert.equal(byClass(root, "sv-table")[0].querySelectorAll("tr").length, 18, "header + 17 candidates");
  byKey(root, "tab:report").click();
  assert.ok(has(root, "还没有生成综述"));
  assert.equal(byKey(root, "tab:report").getAttribute("aria-pressed"), "true", "a chosen tab stays");

  // 继续 → the survey runs: the running view with its stepper
  const progress = { stage: "read", index: 6, label: "精读", message: "精读 5/15：Paper", current: 5, total: 15, counts: { seeds: 0, candidates: 142, read: 5, references: 0 } };
  control.surveys[CANCELLED] = { ...control.surveys[CANCELLED], status: "running", stage: "read", running: true, progress };
  byKey(root, "resume").click();
  await flush();
  assert.deepEqual(w.callsOf("resumeSurvey"), [["resumeSurvey", CANCELLED]]);
  assert.equal(w.doc.getElementById("nav-survey-dot").hidden, false);
  const steps = byClass(root, "sv-step");
  assert.equal(steps.length, 9);
  assert.deepEqual(steps.map((s) => s.className.replace("sv-step ", "")), ["done", "done", "done", "done", "done", "done", "current", "todo", "todo"]);
  assert.equal(steps[6].getAttribute("aria-current"), "step");
  assert.equal(byClass(root, "sv-counter-value").map((n) => n.textContent).join(","), "0,142,5");
  assert.equal(byClass(root, "progress-text")[0].textContent, "精读 5/15：Paper");
  assert.equal(byClass(root, "progress-count")[0].textContent, "5 / 15");
  assert.equal(byClass(root, "progress-track")[0].getAttribute("aria-valuenow"), "33");

  // Live progress, in place
  w.emit({ type: "survey", action: "progress", id: CANCELLED, progress: { stage: "themes", index: 7, label: "主题", message: "主题…", current: null, total: null, counts: { seeds: 0, candidates: 142, read: 15, references: 0 } } });
  assert.equal(steps[7].className, "sv-step current");
  assert.equal(steps[6].className, "sv-step done");
  assert.equal(byClass(root, "sv-counter-value")[2].textContent, "15");
  assert.equal(byClass(root, "progress-fill")[0].classList.contains("indeterminate"), true);
  assert.equal(byClass(root, "progress-count")[0].textContent, "");
  // Progress of another survey is not shown here
  w.emit({ type: "survey", action: "progress", id: "someother1", progress: { ...progress, index: 1 } });
  assert.equal(steps[7].className, "sv-step current");

  // 取消调研
  const cancel = byKey(root, "cancel");
  cancel.click();
  assert.deepEqual(w.callsOf("cancelSurvey").at(-1), ["cancelSurvey", CANCELLED]);
  assert.equal(cancel.disabled, true);
  assert.equal(cancel.textContent, "正在取消…");

  // Done: the result replaces the running view
  control.surveys[CANCELLED] = { ...clone(w.fixtures.surveyDetails[DONE]), id: CANCELLED };
  control.options.running = null;
  w.emit({ type: "survey", action: "done", id: CANCELLED, status: "done" });
  await flush();
  assert.equal(byClass(root, "sv-step").length, 0);
  assert.ok(byClass(root, "sv-report")[0]);
  assert.ok(byKey(root, "save"));
});

// ----------------------------------------------------------- result view

test("a finished survey: review with [n] citations, references, candidates and themes", async () => {
  const w = await openPage({ hash: `#survey/${DONE}` });
  const { root, doc, control } = w;
  const survey = control.surveys[DONE];
  assert.equal(doc.getElementById("survey-new").hidden, false);
  assert.equal(byClass(root, "sv-title")[0].textContent, survey.report.title);
  assert.equal(byClass(root, "sv-status")[0].textContent, "已完成");
  assert.match(byClass(root, "sv-meta")[0].textContent, / · 深度：标准 · 种子 2 篇 · 候选 17 篇 · 精读 12 篇 · 模型 deepseek\/deepseek-chat$/);
  assert.ok(has(root, "本综述由 AI 根据所列文献"), "the AI notice");
  assert.ok(has(root, "调研中有 1 个问题（其余步骤照常完成）"));
  assert.equal(byClass(root, "sv-sec-title").map((n) => n.textContent).join("|"), "总述|岩浆–热液过程中的铜同位素分馏|铜同位素分带与勘查应用|流体沸腾与去气的实验约束|主要争议|研究空白|未来方向|推荐阅读顺序");
  // Model text is text: "[1, 4]" became two buttons inside a superscript group
  const summary = byClass(root, "sv-summary")[0];
  const firstGroup = byClass(summary, "sv-cite-group")[0];
  assert.equal(firstGroup.textContent, "[1, 4]");
  assert.deepEqual(byClass(firstGroup, "sv-cite").map((b) => b.textContent), ["1", "4"]);
  assert.match(byClass(firstGroup, "sv-cite")[0].getAttribute("title"), /^Copper isotope fractionation during magmatic sulfide saturation/);
  // References: numbered, with DOI links, 已在文库 or 加入 Zotero, and reading notes
  const refs = byClass(root, "sv-ref");
  assert.equal(refs.length, survey.references.length);
  assert.equal(byClass(refs[0], "sv-ref-n")[0].textContent, "[1]");
  assert.equal(byClass(refs[0], "sv-doi")[0].getAttribute("href"), "https://doi.org/10.5555/survey.1");
  assert.equal(byClass(refs[0], "sv-lib")[0].textContent, "已在文库");
  assert.equal(byKey(refs[1], "add:doi:10.5555/survey.2").textContent, "加入 Zotero");

  // A citation jumps to (and highlights) its reference
  byClass(firstGroup, "sv-cite")[1].click();
  const ref4 = byClass(root, "sv-ref")[3];
  assert.equal(ref4.classList.contains("flash"), true);
  assert.equal(ref4.scrolled, 1);
  assert.equal(doc.activeElement, ref4);
  await w.clock.advance(2000);
  assert.equal(ref4.classList.contains("flash"), false);

  // Reading notes open under a reference
  byKey(root, "note:doi:10.5555/survey.2").click();
  const note = byClass(root, "sv-note")[0];
  assert.match(note.textContent, /^研究问题流体出溶过程是否造成铜同位素分馏？研究方法/);
  assert.match(note.textContent, /相关度5 \/ 5$/);

  // Candidates: the table with provenance chips, sorting and filters; focus stays on the tab
  const candidatesTab = byKey(root, "tab:candidates");
  candidatesTab.focus();
  candidatesTab.click();
  assert.equal(doc.activeElement.getAttribute("data-key"), "tab:candidates", "focus survives the re-render");
  let rows = byClass(root, "sv-paper");
  assert.equal(rows.length, survey.pool.length);
  assert.deepEqual(byClass(rows[1], "sv-via").map((c) => c.textContent), ["检索", "引用", "推荐"]);
  assert.match(byClass(rows[0], "sv-read")[0].textContent, /^已精读 \[1\]$/);
  assert.equal(byClass(root, "sv-score-value")[0].textContent, "93");
  byKey(root, "sort:year").click();
  rows = byClass(root, "sv-paper");
  assert.equal(byClass(rows[0], "sv-paper-title")[0].textContent, "Machine-learning prediction of porphyry fertility from zircon and copper isotopes");
  assert.equal(byKey(root, "sort:year").parentNode.getAttribute("aria-sort"), "descending");
  byKey(root, "sort:year").click();
  assert.equal(byKey(root, "sort:year").parentNode.getAttribute("aria-sort"), "ascending");
  byKey(root, "filter:new").click();
  assert.equal(byClass(root, "sv-paper").length, survey.pool.filter((p) => !p.inLibrary).length);
  byKey(root, "filter:read").click();
  assert.equal(byClass(root, "sv-paper").length, 12);

  // Adding papers: busy while it runs, 已在文库 after; a failure stays on its row
  byKey(root, "filter:all").click();
  byKey(root, "sort:score").click();
  assert.equal(byKey(root, "sort:score").parentNode.getAttribute("aria-sort"), "descending");
  let release;
  control.gates["add:doi:10.5555/survey.2"] = new Promise((r) => (release = r));
  control.addErrors["doi:10.5555/survey.3"] = "DOI 无法解析";
  byKey(root, "add:doi:10.5555/survey.2").click();
  await flush();
  assert.equal(byKey(root, "add:doi:10.5555/survey.2").textContent, "正在加入…");
  assert.equal(byKey(root, "add:doi:10.5555/survey.2").disabled, true);
  byKey(root, "add:doi:10.5555/survey.3").click();
  await flush();
  assert.equal(byClass(root, "sv-row-error").map((n) => n.textContent).join("|"), "加入失败：DOI 无法解析");
  assert.equal(byKey(root, "add:doi:10.5555/survey.2").textContent, "正在加入…", "the other row is still busy");
  release();
  await flush();
  assert.equal(byKey(root, "add:doi:10.5555/survey.2"), null);
  assert.equal(byKey(root, "show:doi:10.5555/survey.2").textContent, "已在文库");
  assert.match(w.toasts(), /已加入 Zotero/);
  assert.deepEqual(w.callsOf("addSurveyPaper").map((c) => c[2]), ["doi:10.5555/survey.2", "doi:10.5555/survey.3"]);
  // 已在文库 shows the item; a deleted item says so
  control.showResult = false;
  byKey(root, `show:${survey.pool[0].id}`).click();
  await flush();
  assert.deepEqual(w.callsOf("showItemByKey").at(-1), ["showItemByKey", "SEEDKEY1"]);
  assert.match(w.toasts(), /这篇文献已不在 Zotero 文库中/);
  // 打开 links the DOI (main.js opens web links through api.openUrl)
  const open = byClass(root, "sv-open")[0];
  assert.equal(open.getAttribute("href"), "https://doi.org/10.5555/survey.1");
  open.click();
  assert.deepEqual(w.callsOf("openUrl").at(-1), ["openUrl", "https://doi.org/10.5555/survey.1"]);

  // Themes, and a citation there jumps back to the review's references
  byKey(root, "tab:themes").click();
  const themes = byClass(root, "sv-theme");
  assert.equal(themes.length, 3);
  assert.equal(byClass(themes[0], "sv-theme-title")[0].textContent, "岩浆–热液过程中的铜同位素分馏");
  assert.deepEqual(byClass(themes[1], "chip").map((c) => c.textContent), ["zoning", "exploration", "supergene"]);
  assert.equal(byClass(themes[1], "sv-cite")[0].textContent, "6");
  byClass(themes[1], "sv-cite")[0].click();
  assert.equal(byKey(root, "tab:report").classList.contains("active"), true, "back on the review");
  assert.equal(byClass(root, "sv-ref")[5].classList.contains("flash"), true);
});

test("toolbar: 保存到 Zotero with progress and stop, Markdown export, regenerate and delete", async () => {
  const w = await openPage({ hash: `#survey/${DONE}` });
  const { root, control } = w;
  const survey = control.surveys[DONE];

  // The save panel: what to add with the note
  byKey(root, "save").click();
  assert.equal(byKey(root, "save").getAttribute("aria-expanded"), "true");
  const radios = root.querySelectorAll('input[name="sv-save-mode"]');
  assert.deepEqual(radios.map((r) => [r.value, r.checked]), [["cited", true], ["read", false], ["none", false]]);
  assert.deepEqual(byClass(root, "sv-radio").map((r) => r.textContent), ["笔记 + 综述引用的文献（12 篇）", "笔记 + 全部精读文献（12 篇）", "只保存笔记"]);
  assert.ok(has(root, "在分类「ZotWatch 调研 · 斑岩铜矿成矿流体中的铜同位素示踪」中保存一篇笔记"));
  radios[1].checked = true;
  radios[1].dispatch("change");

  let release;
  control.gates.save = new Promise((r) => (release = r));
  control.saveResult = { collectionName: "ZotWatch 调研 · 斑岩铜矿成矿流体中的铜同位素示踪", total: 12, linked: 1, imported: 3, failed: 0, skipped: 0, cancelled: true };
  byKey(root, "save-go").click();
  await flush();
  assert.deepEqual(w.callsOf("saveSurveyToZotero")[0], ["saveSurveyToZotero", DONE, { addPapers: "read" }]);
  assert.ok(has(root, "正在保存笔记…"));
  assert.equal(root.querySelectorAll('input[name="sv-save-mode"]').every((r) => r.disabled), true);
  // While it is saved, the survey is neither written again nor deleted
  assert.deepEqual(["save", "regenerate", "delete"].map((k) => byKey(root, k).disabled), [true, true, true]);
  w.emit({ type: "survey", action: "save-progress", id: DONE, progress: { current: 1, total: 12, title: "Fluid exsolution" } });
  assert.ok(has(root, "正在加入文献 2/12：Fluid exsolution"));
  assert.equal(byClass(byClass(root, "sv-save")[0], "progress-fill")[0].style.width, "8%");
  byKey(root, "save-stop").click();
  assert.deepEqual(w.callsOf("cancelSurveySave"), [["cancelSurveySave", DONE]]);
  const reloads = w.callsOf("getSurvey").length;
  release();
  await flush();
  assert.equal(byClass(root, "sv-save-result")[0].textContent, "已保存到「ZotWatch 调研 · 斑岩铜矿成矿流体中的铜同位素示踪」：笔记已写入，1 篇已在文库的文献加入分类，导入 3 篇；已停止导入，其余文献未加入。");
  assert.match(w.toasts(), /已停止导入，笔记已保存/);
  assert.ok(w.callsOf("getSurvey").length > reloads, "reloaded: imported papers are in the library now");
  assert.deepEqual(["save", "regenerate", "delete"].map((k) => byKey(root, k).disabled), [false, false, false]);
  byKey(root, "save-close").click();
  assert.equal(byClass(root, "sv-save").length, 0);

  // A failed save is shown with 重试
  control.errors.save = "Zotero 正在同步";
  byKey(root, "save").click();
  byKey(root, "save-go").click();
  await flush();
  assert.equal(byClass(root, "sv-save")[0].querySelector(".sv-row-error").textContent, "保存失败：Zotero 正在同步");
  assert.ok(byKey(root, "save-again"));
  delete control.errors.save;

  // 导出 Markdown through the FilePicker; cancelling writes nothing
  byKey(root, "export").click();
  await flush();
  assert.deepEqual(w.callsOf("exportSurveyMarkdown"), [["exportSurveyMarkdown", DONE, null]]);
  assert.deepEqual(w.callsOf("FilePicker.show").at(-1), ["FilePicker.show", "ZotWatch 调研 - 主题 - 2026-10-17.md", "md"]);
  control.pickPath = "/home/me/综述";
  byKey(root, "export").click();
  await flush();
  assert.deepEqual(w.callsOf("exportSurveyMarkdown").at(-1), ["exportSurveyMarkdown", DONE, "/home/me/综述.md"]);
  assert.match(w.toasts(), /已导出：综述\.md/);
  // The picker only checked "综述": replacing an existing "综述.md" is confirmed here
  control.existing = ["/home/me/综述.md"];
  control.confirmReply = false;
  const exports = w.callsOf("exportSurveyMarkdown").length;
  byKey(root, "export").click();
  await flush();
  assert.match(w.confirms.at(-1), /「综述\.md」已存在，要替换它吗？/);
  assert.equal(w.callsOf("exportSurveyMarkdown").length, exports + 1, "only the text was asked for, nothing written");
  assert.equal(w.callsOf("exportSurveyMarkdown").at(-1)[2], null);
  control.confirmReply = true;
  byKey(root, "export").click();
  await flush();
  assert.deepEqual(w.callsOf("exportSurveyMarkdown").at(-1), ["exportSurveyMarkdown", DONE, "/home/me/综述.md"]);
  control.existing = [];
  control.pickPath = "C:\\docs\\review.markdown";
  byKey(root, "export").click();
  await flush();
  assert.deepEqual(w.callsOf("exportSurveyMarkdown").at(-1), ["exportSurveyMarkdown", DONE, "C:\\docs\\review.markdown"]);

  // 重新生成综述: confirmed only
  control.confirmReply = false;
  byKey(root, "regenerate").click();
  await flush();
  assert.equal(w.callsOf("regenerateReport").length, 0);
  assert.match(w.confirms.at(-1), /重新撰写综述（不重新检索和精读）/);
  control.confirmReply = true;
  control.surveys[DONE] = { ...survey, status: "running", running: true, progress: null };
  byKey(root, "regenerate").click();
  await flush();
  assert.deepEqual(w.callsOf("regenerateReport"), [["regenerateReport", DONE]]);
  assert.equal(byClass(root, "sv-step").length, 9, "the running view");
  control.surveys[DONE] = survey;
  control.options.running = null;
  w.emit({ type: "survey", action: "done", id: DONE, status: "done" });
  await flush();

  // 删除: confirmed, then the list
  byKey(root, "delete").click();
  await flush();
  assert.deepEqual(w.callsOf("deleteSurvey"), [["deleteSurvey", DONE]]);
  assert.match(w.confirms.at(-1), /删除调研「斑岩铜矿成矿流体中的铜同位素示踪」？/);
  assert.equal(w.api.lastPage, "survey");
  assert.ok(byClass(root, "sv-form")[0]);
  assert.match(w.toasts(), /已删除这项调研/);
});

test("a save belongs to its survey: another survey shown meanwhile, and coming back to it", async () => {
  const w = await openPage({ hash: `#survey/${DONE}` });
  const { root, control } = w;
  const OTHER = "mvq2k3a02y";
  control.surveys[OTHER] = { ...control.surveys[DONE], id: OTHER, input: { ...control.surveys[DONE].input, topic: "另一项调研" }, saved: null };
  let release;
  control.gates.save = new Promise((r) => (release = r));
  control.saveResult = { collectionName: "ZotWatch 调研 · 第一项", total: 3, linked: 1, imported: 2, failed: 0, skipped: 0, cancelled: false };
  byKey(root, "save").click();
  byKey(root, "save-go").click();
  await flush();

  // Another survey: no save panel, its actions free
  await w.navigate(`survey/${OTHER}`);
  assert.equal(byClass(root, "sv-save").length, 0);
  assert.deepEqual(["save", "regenerate", "delete"].map((k) => byKey(root, k).disabled), [false, false, false]);
  w.emit({ type: "survey", action: "save-progress", id: DONE, progress: { current: 1, total: 3, title: "x" } });
  assert.equal(byClass(root, "sv-save").length, 0, "the other survey's progress is not shown here");
  release();
  await flush();
  w.emit({ type: "survey", action: "updated", id: OTHER });
  await flush();
  assert.equal(byClass(root, "sv-save").length, 0, "nor its result");
  assert.ok(!has(root, "ZotWatch 调研 · 第一项"));

  // Coming back to a survey whose save still runs shows its progress and 停止导入
  control.gates.save = new Promise((r) => (release = r));
  byKey(root, "save").click();
  byKey(root, "save-go").click();
  await flush();
  await w.navigate(`survey/${DONE}`);
  w.emit({ type: "survey", action: "save-progress", id: OTHER, progress: { current: 0, total: 3, title: "y" } });
  await w.navigate(`survey/${OTHER}`);
  assert.ok(has(root, "正在加入文献 1/3：y"));
  assert.ok(byKey(root, "save-stop"));
  assert.equal(byKey(root, "delete").disabled, true);
  release();
  await flush();
  assert.equal(byKey(root, "delete").disabled, false);
});

test("a survey opened from the list scrolls back to where it was left once it has loaded", async () => {
  const w = await openPage({ hash: `#survey/${DONE}` });
  const { root, control, doc } = w;
  // Like Gecko, the scroll container clamps scrollTop to its content: only a loaded survey (or the list) is tall
  const pages = doc.getElementById("pages");
  let top = 0;
  Object.defineProperty(pages, "scrollTop", {
    configurable: true,
    get: () => top,
    set: (value) => {
      const tall = byClass(root, "sv-head").length > 0 || byClass(root, "sv-form").length > 0;
      top = Math.max(0, Math.min(Number(value) || 0, tall ? 5000 : 0));
    },
  });
  pages.scrollTop = 1500;
  assert.equal(pages.scrollTop, 1500);
  await w.navigate("survey");
  let release;
  control.gates[`get:${DONE}`] = new Promise((r) => (release = r));
  await w.navigate(`survey/${DONE}`);
  assert.ok(has(root, "正在加载…"));
  assert.equal(pages.scrollTop, 0, "clamped while loading");
  release();
  await flush();
  assert.ok(byClass(root, "sv-head").length);
  assert.equal(pages.scrollTop, 1500, "restored once the survey is shown");
  // A later reload (an event) does not jump back
  pages.scrollTop = 200;
  w.emit({ type: "survey", action: "updated", id: DONE });
  await flush();
  assert.equal(pages.scrollTop, 200);
});

test("keyboard focus stays near a control that turns busy or is replaced", async () => {
  const w = await openPage({ hash: `#survey/${DONE}` });
  const { root, control, doc } = w;
  const inTree = (node) => all(root).includes(node);
  // 加入 Zotero: the busy button is disabled, focus parks on its reference, and stays there
  const key = "add:doi:10.5555/survey.2";
  let release;
  control.gates[key] = new Promise((r) => (release = r));
  byKey(root, key).focus();
  byKey(root, key).click();
  await flush();
  assert.equal(doc.activeElement.getAttribute("data-key"), "item:doi:10.5555/survey.2");
  assert.ok(inTree(doc.activeElement), "a node of the new view");
  release();
  await flush();
  assert.ok(inTree(doc.activeElement));

  // A toolbar action disables the toolbar: focus goes to the toolbar itself
  let releaseExport;
  const exportGate = new Promise((r) => (releaseExport = r));
  const exportMarkdown = w.api.exportSurveyMarkdown;
  w.api.exportSurveyMarkdown = async (...args) => (await exportGate, exportMarkdown(...args));
  byKey(root, "export").focus();
  byKey(root, "export").click();
  await flush();
  assert.equal(doc.activeElement.getAttribute("data-key"), "toolbar");
  assert.ok(inTree(doc.activeElement));
  releaseExport();
  await flush();

  // The history list is redrawn every minute: a focused entry keeps the focus
  await w.navigate("survey");
  const entry = byClass(root, "sv-hist")[0];
  entry.focus();
  w.page.tick();
  assert.equal(doc.activeElement.getAttribute("data-key"), entry.getAttribute("data-key"));
  assert.notEqual(doc.activeElement, entry, "a new node");
  assert.ok(inTree(doc.activeElement));
});

test("model text and paper metadata with markup are shown as text", async () => {
  const w = await openPage({ hash: "#survey" });
  const { root, control } = w;
  const s = control.surveys[DONE];
  s.report.summary = "总述 <b>粗体</b> <img src=x onerror=alert(1)> [1]\n\n第二段 [2]";
  s.report.sections[0].paragraphs[0] = "一节 <script>alert(1)</script> [1]";
  s.report.debates = ["争议 <i>斜体</i> [1]"];
  s.references[0].title = "Isotopes of <sup>65</sup>Cu & <em>fluids</em>";
  await w.navigate(`survey/${DONE}`);
  const pars = byClass(root, "sv-par");
  assert.equal(pars[0].textContent, "总述 <b>粗体</b> <img src=x onerror=alert(1)> [1]");
  assert.ok(pars.some((p) => p.textContent === "一节 <script>alert(1)</script> [1]"));
  assert.ok(has(root, "争议 <i>斜体</i> [1]"));
  assert.ok(byClass(root, "sv-ref-title").some((n) => n.textContent === "Isotopes of <sup>65</sup>Cu & <em>fluids</em>"));
  // (the page uses <sup> itself; compare names, not nodes, so that a failure stays readable)
  assert.deepEqual(all(root).map((n) => n.localName).filter((name) => ["b", "img", "script", "i", "em"].includes(name)), [], "no element was made from the text");
});

test("actions that do not apply are disabled: no LLM, another survey running, a list without a review", async () => {
  const w = await openPage({ hash: `#survey/${DONE}`, options: { llm: false } });
  const { root, control } = w;
  assert.equal(byKey(root, "regenerate").disabled, true);
  assert.equal(byKey(root, "regenerate").getAttribute("title"), "需要开启大模型（设置 → AI 助手）");

  // Another survey runs: no 重新生成 / 继续
  control.options = { ...control.options, llm: true, running: { id: "someother1", progress: null } };
  w.emit({ type: "survey", action: "start", id: "someother1" });
  await flush();
  assert.equal(byKey(root, "regenerate").disabled, true);
  await w.navigate(`survey/${FAILED}`);
  assert.equal(byKey(root, "resume").disabled, true);
  assert.ok(has(root, "调研失败（扩展）"));
  assert.ok(has(root, "所有数据源都请求失败：OpenAlex 免费额度已用完（HTTP 429）。已完成到「检索词」，「继续」会从下一步开始。另一项调研完成或取消后才能继续。"));
  assert.ok(has(root, "还没有生成综述"));
  assert.ok(has(root, "还没有候选文献") === false, "the report tab is shown first");
  // A stopped survey's problems do not claim that the rest went on
  assert.ok(has(root, "调研中有 1 个问题扩展 · Semantic Scholar：检索 1/3 失败"));
  assert.equal(has(root, "其余步骤照常完成"), false);
  byKey(root, "tab:candidates").click();
  assert.ok(has(root, "还没有候选文献"));

  // An interrupted survey (Zotero quit while it ran)
  control.surveys[CANCELLED] = { ...control.surveys[CANCELLED], interrupted: true };
  await w.navigate(`survey/${CANCELLED}`);
  assert.ok(has(root, "调研被中断（Zotero 已关闭）"));

  // A survey made without an LLM: references grouped by cluster, and 生成综述 (the LLM is on now)
  control.options.running = null;
  await w.navigate("survey/mvl5t2r08d");
  assert.ok(has(root, "这项调研没有使用大模型"));
  assert.ok(has(root, "现在已开启大模型，可以「生成综述」"));
  assert.equal(byKey(root, "regenerate").textContent, "生成综述");
  assert.equal(byKey(root, "tab:report").textContent, "文献分组");
  assert.equal(byClass(root, "sv-sec-title").map((n) => n.textContent).join("|"), "fractionation · sulfide · melt|zoning · exploration · supergene|brine · vapor · degassing");
  assert.equal(byClass(root, "sv-reading-title").length, 12);
  assert.match(byClass(root, "sv-meta")[0].textContent, / · 未使用大模型$/);
  assert.equal(byClass(root, "sv-note-toggle").length, 0, "nothing was read");
  byKey(root, "save").click();
  assert.deepEqual(byClass(root, "sv-radio").map((r) => r.textContent), ["笔记 + 列出的文献（12 篇）", "只保存笔记"]);
  // Unknown reference numbers in a group are dropped
  control.surveys.mvl5t2r08d.report.groups[1].refs.push(99);
  await w.navigate(`survey/${DONE}`);
  await w.navigate("survey/mvl5t2r08d");
  assert.equal(byClass(root, "sv-reading-title").length, 12);
  // 生成综述 asks first (it reads the papers before writing)
  w.control.confirmReply = false;
  byKey(root, "regenerate").click();
  assert.match(w.confirms.at(-1), /将先精读排名靠前的文献/);
  assert.equal(w.callsOf("regenerateReport").length, 0);
  // Without the LLM the notice points to the settings
  control.options.llm = false;
  await w.navigate(`survey/${DONE}`);
  await w.navigate("survey/mvl5t2r08d");
  assert.ok(has(root, "AI 综述需要开启大模型"));
  assert.equal(byText(root, "a", "去设置").getAttribute("href"), "#settings/ai");
});

test("unknown surveys, load failures, stale answers and deleted surveys", async () => {
  const w = await openPage({ hash: "#survey/zzzzzz9999" });
  const { root, control } = w;
  assert.ok(has(root, "找不到这项调研"));
  byText(root, "button", "返回调研列表").click();
  await flush();
  assert.equal(w.api.lastPage, "survey");

  // A failed load offers 重试
  control.errors.get = "survey file is locked";
  await w.navigate(`survey/${DONE}`);
  assert.ok(has(root, "无法读取这项调研"));
  assert.ok(has(root, "survey file is locked"));
  delete control.errors.get;
  byText(root, "button", "重试").click();
  await flush();
  assert.equal(byClass(root, "sv-title")[0].textContent, control.surveys[DONE].report.title);

  // Switching surveys while one still loads: the late answer is dropped
  let releaseA;
  control.gates[`get:${CANCELLED}`] = new Promise((r) => (releaseA = r));
  w.run(`window.ZotWatchMain.navigate("survey/${CANCELLED}")`);
  await flush(2);
  assert.ok(has(root, "正在加载…"), "no stale survey while loading");
  await w.navigate(`survey/${FAILED}`);
  releaseA();
  await flush();
  assert.equal(byClass(root, "sv-title")[0].textContent, "Tibetan Plateau uplift and monsoon climate");

  // The shown survey deleted elsewhere: back to the list
  w.emit({ type: "survey", action: "deleted", id: FAILED });
  await flush();
  assert.equal(w.api.lastPage, "survey");
  assert.ok(byClass(root, "sv-form")[0]);

  // A failing list shows its error in the history card
  control.errors.list = "surveys.json is damaged";
  await w.navigate("inbox");
  w.run("ZotWatchSurvey.state.list = null");
  await w.navigate("survey");
  assert.ok(has(root, "无法读取调研记录"));
  assert.deepEqual(w.errors.map((e) => e.message), ["survey file is locked", "surveys.json is damaged"]);
});

test("wide windows show the review beside the candidates and themes", async () => {
  const w = await openPage({ hash: `#survey/${DONE}`, wide: true });
  const { root, mql } = w;
  assert.ok(w.callsOf("matchMedia").some((c) => c[1] === "(min-width: 1240px)"));
  const result = byClass(root, "sv-result")[0];
  assert.equal(result.className, "sv-result wide");
  assert.equal(byKey(root, "tab:report"), null, "the review is always shown");
  assert.ok(byClass(root, "sv-report")[0]);
  assert.ok(byClass(root, "sv-cands")[0], "candidates beside it");
  byKey(root, "side:themes").click();
  assert.equal(byClass(root, "sv-theme").length, 3);
  assert.ok(byClass(root, "sv-report")[0]);
  // A citation in a theme scrolls the review without switching panes
  byClass(byClass(root, "sv-theme")[0], "sv-cite")[0].click();
  assert.equal(byClass(root, "sv-theme").length, 3);
  assert.equal(byClass(root, "sv-ref")[0].classList.contains("flash"), true);
  // Narrowing the window: one column with tabs (the side pane stays chosen)
  mql.matches = false;
  mql.listeners.forEach((fn) => fn({ matches: false }));
  assert.equal(byClass(root, "sv-result")[0].className, "sv-result");
  assert.equal(byKey(root, "tab:themes").classList.contains("active"), true);
});
