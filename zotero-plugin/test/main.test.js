// The main window shell (addon/content/main.js) started the way Zotero starts
// it, by the window's "load" event, with the 推荐 page (inbox.js) in a fake
// DOM: the page it opens on, the bridge addon.js calls, keyboard shortcuts,
// coalesced reloads, the progress strip and the error banner, focus after a
// page switch, saving on close, and the inbox controls wired by init().

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const mainCode = readFileSync(join(root, "addon/content/main.js"), "utf8");
const inboxCode = readFileSync(join(root, "addon/content/inbox.js"), "utf8");
const tick = () => new Promise((r) => setImmediate(r));
// Values created inside the VM have foreign prototypes; compare them structurally
const plain = (x) => JSON.parse(JSON.stringify(x));

// ------------------------------------------------------------- fake DOM

class FakeNode {
  constructor(doc, tag, attrs = {}) {
    this.ownerDocument = doc;
    this.tagName = tag;
    this.childNodes = [];
    this.parentNode = null;
    this.attributes = {};
    this.listeners = {};
    this.className = "";
    this.ownText = "";
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.type = "";
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
        if (on) set.add(c);
        else set.delete(c);
        node.className = [...set].join(" ");
        return on;
      },
    };
    Object.assign(this, attrs);
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
    return this.attributes[k] ?? null;
  }
  removeAttribute(k) {
    delete this.attributes[k];
  }
  addEventListener(type, fn) {
    (this.listeners[type] ||= []).push(fn);
  }
  dispatchEvent(event) {
    event.target ??= this;
    for (const fn of this.listeners[event.type] || []) fn(event);
    return true;
  }
  click() {
    this.dispatchEvent({ type: "click", preventDefault() {} });
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  blur() {
    if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = this.ownerDocument.body;
  }
  select() {}
  append(...nodes) {
    for (const n of nodes) {
      const child = typeof n === "string" ? { textContent: n } : n;
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }
  replaceChildren(...nodes) {
    this.ownText = "";
    this.childNodes = [];
    this.append(...nodes);
  }
  closest(selector) {
    if (selector !== "[hidden]") throw new Error(`closest(${selector}) is not faked`);
    for (let n = this; n; n = n.parentNode) if (n.hidden) return n;
    return null;
  }
  querySelectorAll(selector) {
    if (selector === "button[data-value]") return this.childNodes.filter((n) => n.dataset?.value !== undefined);
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
const buttonsIn = (node, text) => walk(node).filter((n) => n.tagName === "button" && n.textContent === text);

/**
 * The window of main.xhtml as far as the shell and the inbox use it: pages,
 * nav buttons, inbox tabs and segmented controls.
 */
function fakeDocument() {
  const doc = { activeElement: null };
  const elements = new Map();
  const make = (tag, attrs) => new FakeNode(doc, tag, attrs);
  doc.body = make("body");
  doc.documentElement = make("html");
  doc.activeElement = doc.body;
  doc.getElementById = (id) => {
    if (!elements.has(id)) elements.set(id, make("div", { id }));
    return elements.get(id);
  };
  const add = (id, tag, attrs = {}, parent = null) => {
    const node = make(tag, { id, ...attrs });
    elements.set(id, node);
    if (parent) parent.append(node);
    return node;
  };
  const pages = ["overview", "inbox", "settings"].map((page) => add(`page-${page}`, "section", { className: "page", dataset: { page } }, add("pages", "main")));
  const navItems = ["overview", "inbox", "settings"].map((page) => make("button", { className: "nav-item", dataset: { page } }));
  const pageActions = ["overview", "inbox", "settings"].map((page) => make("div", { className: "page-actions", dataset: { page } }));
  const inbox = pages[1];
  const tabs = ["new", "added", "dismissed"].map((tab) => {
    const node = make("button", { className: "tab", dataset: { tab } });
    inbox.append(node);
    return node;
  });
  add("inbox-search", "input", { type: "search" }, inbox);
  add("inbox-dismiss", "button", {}, inbox);
  add("inbox-list", "div", {}, inbox);
  const seg = (id, values) => {
    const group = add(id, "div", {}, inbox);
    for (const value of values) group.append(make("button", { dataset: { value } }));
  };
  seg("inbox-track", ["all", "personal", "flagship", "followed"]);
  seg("inbox-label", ["all", "must_read", "latest"]);
  seg("inbox-sort", ["score", "date"]);
  add("page-title", "h1");
  add("nav-count", "span");
  add("run", "button");
  doc.querySelectorAll = (selector) =>
    ({ ".page": pages, ".nav-item": navItems, ".page-actions": pageActions, "#page-inbox .tab": tabs, ".subnav-item": [] })[selector] || [];
  doc.createElement = (tag) => make(tag);
  doc.createElementNS = (ns, tag) => make(tag);
  doc.createTextNode = (text) => ({ textContent: text });
  doc.addEventListener = (type, fn) => (doc.listeners ||= {})[type] = fn;
  return { doc, pages: Object.fromEntries(pages.map((p) => [p.dataset.page, p])), navItems, tabs };
}

const RUN_ID = "run-1";
const WORKS = [
  { id: "p1", status: "new", firstRunId: RUN_ID, label: "must_read", source: "crossref", title: "Zircon ages", score: 0.8, similarity: 0.7, url: "https://doi.org/p1", authors: [] },
  { id: "p2", status: "new", firstRunId: RUN_ID, label: "consider", source: "crossref", title: "Mantle", score: 0.5, similarity: 0.4, url: "https://doi.org/p2", authors: [] },
  { id: "f1", status: "new", firstRunId: RUN_ID, label: "followed", source: "openalex", title: "Followed", score: 0, similarity: 0, authors: [] },
  { id: "a1", status: "added", itemID: 42, label: "consider", source: "crossref", title: "Added one", score: 0.6, similarity: 0.5, authors: [] },
  { id: "d1", status: "dismissed", label: "consider", source: "crossref", title: "Dismissed one", score: 0.6, similarity: 0.5, authors: [] },
];

function makeView(overrides = {}) {
  return {
    works: WORKS.map((w) => ({ ...w })),
    lastRun: { runId: RUN_ID, runAt: Date.now() - 3600000, finishedAt: Date.now() - 3500000, stats: { fetched: 10, recommended: 2, newRecommendations: 3 }, errors: [] },
    running: false,
    progress: null,
    lastError: null,
    llmEnabled: false,
    provider: "dashscope",
    version: "0.3.0",
    schedule: { autoRun: true, intervalHours: 24, nextRunAt: Date.now() + 3600000 },
    ...overrides,
  };
}

/** Load main.js + inbox.js; `hash` is the URL fragment, `lastPage` what openWindow() set. */
function openWindow({ hash = "", lastPage = null, view = makeView() } = {}) {
  const { doc, pages, navItems, tabs } = fakeDocument();
  const calls = [];
  const errors = [];
  const listeners = new Set();
  const state = { view, getView: null };
  const api = {
    lastPage,
    getView: async () => {
      calls.push(["getView"]);
      if (state.getView) return state.getView();
      return structuredClone(state.view);
    },
    addListener: (fn) => listeners.add(fn),
    removeListener: (fn) => listeners.delete(fn),
    runNow: async () => calls.push(["runNow"]),
    cancelRun: () => calls.push(["cancelRun"]),
    setStatus: async (ids, status) => calls.push(["setStatus", ids, status]),
    addToLibrary: async (id) => calls.push(["addToLibrary", id]),
    showItem: async (id) => calls.push(["showItem", id]),
    summarize: async (id) => calls.push(["summarize", id]),
    openUrl: (url) => calls.push(["openUrl", url]),
  };
  const winListeners = {};
  const intervals = [];
  const window = {
    outerWidth: 1120,
    outerHeight: 820,
    closed: false,
    addEventListener: (type, fn) => (winListeners[type] ||= []).push(fn),
    close() {
      this.closed = true;
    },
    confirm: () => true,
  };
  const context = vm.createContext({
    ChromeUtils: { importESModule: () => ({ Zotero: { ZotWatch: api, logError: (e) => errors.push(e), isMac: false } }) },
    window,
    document: doc,
    location: { hash },
    Intl,
    structuredClone,
    setTimeout,
    clearTimeout,
    setInterval: (fn, ms) => intervals.push({ fn, ms }),
    clearInterval() {},
    Event: class {
      constructor(type, init = {}) {
        this.type = type;
        this.bubbles = !!init.bubbles;
      }
    },
  });
  vm.runInContext(mainCode, context, { filename: "main.js" });
  vm.runInContext(inboxCode, context, { filename: "inbox.js" });
  const run = (src) => vm.runInContext(src, context);
  const fire = (type, event = {}) => (winListeners[type] || []).forEach((fn) => fn({ type, ...event }));
  const key = (key, { ctrl = false, target = doc.body } = {}) => {
    const event = { type: "keydown", key, ctrlKey: ctrl, metaKey: false, altKey: false, shiftKey: false, target, defaultPrevented: false };
    event.preventDefault = () => (event.defaultPrevented = true);
    fire("keydown", event);
    return event;
  };
  const emit = (event) => [...listeners].forEach((fn) => fn(event));
  const last = (name) => plain(calls.filter(([c]) => c === name).at(-1) ?? null);
  return { run, fire, key, emit, doc, pages, navItems, tabs, window, api, calls, last, errors, state, intervals, $: (id) => doc.getElementById(id) };
}

// ------------------------------------------------------------------ tests

test("the window opens on the page openWindow() chose and exposes navigate to the addon", async () => {
  const w = openWindow({ lastPage: "inbox" });
  assert.equal(w.run("shell.page"), null, "nothing is shown before the load event");
  w.fire("load");
  assert.equal(w.run("shell.page"), "inbox");
  assert.equal(w.pages.inbox.hidden, false);
  assert.equal(w.pages.overview.hidden, true);
  // addon.js switches an open window with existing.ZotWatchMain.navigate(route)
  assert.equal(typeof w.window.ZotWatchMain.navigate, "function");
  w.window.ZotWatchMain.navigate("overview");
  assert.equal(w.run("shell.page"), "overview");
  assert.equal(w.api.lastPage, "overview");
  await tick();
  assert.equal(w.doc.documentElement.getAttribute("data-ready"), "true");

  // A route in the URL (the preview harness) wins; no route at all opens the overview
  const hashed = openWindow({ hash: "#settings/ai", lastPage: "inbox" });
  hashed.fire("load");
  assert.equal(hashed.run("currentRoute()"), "settings/ai");
  const plainWindow = openWindow();
  plainWindow.fire("load");
  assert.equal(plainWindow.run("shell.page"), "overview");
});

test("keyboard shortcuts switch pages, focus search, and Escape clears, leaves or closes", async () => {
  const w = openWindow();
  w.fire("load");
  await tick();
  w.key("2", { ctrl: true });
  assert.equal(w.run("shell.page"), "inbox");
  w.key("3", { ctrl: true });
  assert.equal(w.run("shell.page"), "settings");
  w.key("1", { ctrl: true });
  assert.equal(w.run("shell.page"), "overview");
  w.key("f", { ctrl: true });
  assert.equal(w.run("shell.page"), "inbox");
  const search = w.$("inbox-search");
  assert.equal(w.doc.activeElement, search, "Ctrl+F focuses the inbox search");

  // Escape in a search box with text clears it (Firefox does not) and filters again
  search.value = "zircon";
  search.dispatchEvent({ type: "input" });
  assert.equal(w.run("ZotWatchInbox.state.query"), "zircon");
  const clear = w.key("Escape", { target: search });
  assert.equal(search.value, "");
  assert.equal(clear.defaultPrevented, true);
  assert.equal(w.run("ZotWatchInbox.state.query"), "", "the list is filtered again");
  assert.equal(w.doc.activeElement, search, "the first Escape keeps the focus");
  // The next one leaves the field; neither closes the window
  w.key("Escape", { target: search });
  assert.notEqual(w.doc.activeElement, search);
  assert.equal(w.window.closed, false);
  // Outside a text field Escape closes, as does Ctrl+W
  w.key("Escape");
  assert.equal(w.window.closed, true);
  const w2 = openWindow();
  w2.fire("load");
  w2.key("w", { ctrl: true });
  assert.equal(w2.window.closed, true);
});

test("reloads are coalesced: one in flight and one queued", async () => {
  const w = openWindow();
  w.fire("load");
  await tick();
  const before = w.calls.filter(([c]) => c === "getView").length;
  let release;
  w.state.getView = () => new Promise((r) => (release = () => r(structuredClone(w.state.view))));
  const first = w.run("refresh()");
  w.run("refresh()");
  w.run("refresh()");
  release();
  await tick();
  release();
  await first;
  assert.equal(w.calls.filter(([c]) => c === "getView").length - before, 2);
});

test("sidebar count, progress strip, and the minute ticker keep the live run state", async () => {
  const w = openWindow();
  const ticks = [];
  w.run("(fn) => registerPage('overview', { tick: () => fn('overview') })")((name) => ticks.push(name));
  w.fire("load");
  await tick();
  assert.equal(w.$("nav-count").hidden, false);
  assert.equal(w.$("nav-count").textContent, "3", "pending (new) works only");

  // A run starts while the window is open, then reports a determinate stage
  w.emit({ type: "start" });
  assert.equal(w.$("progress").hidden, false);
  assert.equal(w.$("run").disabled, true);
  w.emit({ type: "progress", progress: { message: "生成 AI 摘要 3/10…", current: 3, total: 10 } });
  const strip = () => [w.$("progress-text").textContent, w.$("progress-count").textContent, w.$("progress-bar").style.width];
  assert.deepEqual(strip(), ["生成 AI 摘要 3/10…", "3 / 10", "30%"]);
  // The 60 s ticker repaints the sidebar's relative times: the strip keeps the live stage
  assert.equal(w.intervals[0].ms, 60000);
  w.intervals[0].fn();
  assert.deepEqual(strip(), ["生成 AI 摘要 3/10…", "3 / 10", "30%"]);
  assert.deepEqual(ticks, ["overview"], "the active page's relative times move on too");
  w.emit({ type: "progress", progress: { message: "抓取最新文献…" } });
  w.intervals[0].fn();
  assert.deepEqual(strip(), ["抓取最新文献…", "", ""]);
  assert.equal(w.$("progress-bar").classList.contains("indeterminate"), true);

  // Nothing pending: the count pill is hidden
  w.state.view = makeView({ works: WORKS.filter((x) => x.status !== "new") });
  await w.run("refresh()");
  assert.equal(w.$("nav-count").hidden, true);
  assert.equal(w.$("progress").hidden, true);
});

test("the error banner: load failures clear on the next load, a dismissed run failure stays hidden", async () => {
  const w = openWindow();
  w.fire("load");
  await tick();
  w.state.getView = async () => {
    throw new Error("disk busy");
  };
  await w.run("refresh()");
  assert.equal(w.errors.length, 1, "the failure is logged");
  w.errors.length = 0;
  assert.equal(w.$("banner").hidden, false);
  assert.equal(w.$("banner-text").textContent, "加载失败：disk busy");
  w.state.getView = null;
  await w.run("refresh()");
  assert.equal(w.$("banner").hidden, true, "the next successful load clears it");

  // A failed run shows until dismissed, and stays dismissed on reloads
  w.state.view = makeView({ lastError: { message: "网络连接失败", at: Date.now() } });
  await w.run("refresh()");
  assert.equal(w.$("banner-text").textContent, "运行失败：网络连接失败");
  w.$("banner-close").click();
  assert.equal(w.$("banner").hidden, true);
  await w.run("refresh()");
  assert.equal(w.$("banner").hidden, true);
});

test("a page switch from a control it hides moves focus to the new page's heading", async () => {
  const w = openWindow({ lastPage: "inbox" });
  w.fire("load");
  await tick();
  // A button on the inbox page (e.g. the 关注期刊设置 callout) switches to the overview
  const dismiss = w.$("inbox-dismiss");
  dismiss.focus();
  w.run("navigate('overview')");
  assert.equal(w.doc.activeElement, w.$("page-title"));
  // A sidebar button stays visible and keeps the focus
  const nav = w.navItems[1];
  nav.focus();
  nav.click();
  assert.equal(w.run("shell.page"), "inbox");
  assert.equal(w.doc.activeElement, nav);
});

test("closing the window saves every page's pending edits", async () => {
  const w = openWindow();
  const flushed = [];
  w.run("(fn) => registerPage('settings', { flush: () => fn('settings') })")((name) => flushed.push(name));
  w.fire("load");
  await tick();
  w.fire("unload");
  assert.deepEqual(flushed, ["settings"]);
});

test("inbox controls: tabs, filters, search, 全部忽略 and the card actions", async () => {
  const w = openWindow({ lastPage: "inbox", view: makeView({ llmEnabled: true }) });
  w.fire("load");
  await tick();
  const state = (k) => w.run(`ZotWatchInbox.state.${k}`);
  const list = w.$("inbox-list");

  // Card actions of a new work
  const [add] = buttonsIn(list, "加入 Zotero");
  add.click();
  await tick();
  assert.deepEqual(w.last("addToLibrary"), ["addToLibrary", "p1"]);
  buttonsIn(list, "忽略")[0].click();
  await tick();
  assert.deepEqual(w.last("setStatus"), ["setStatus", "p1", "dismissed"]);
  buttonsIn(list, "AI 摘要")[0].click();
  await tick();
  assert.deepEqual(w.last("summarize"), ["summarize", "p1"]);
  buttonsIn(list, "打开原文")[0].click();
  assert.deepEqual(w.last("openUrl"), ["openUrl", "https://doi.org/p1"]);

  // Segmented filters set their own state only
  const segButton = (id, value) => w.$(id).childNodes.find((b) => b.dataset.value === value);
  segButton("inbox-track", "followed").click();
  assert.equal(state("track"), "followed");
  assert.equal(state("filter"), "all");
  segButton("inbox-label", "must_read").click();
  assert.equal(state("filter"), "must_read");
  segButton("inbox-sort", "date").click();
  assert.equal(state("sort"), "date");
  segButton("inbox-track", "all").click();
  segButton("inbox-label", "all").click();

  // Search filters by its text
  const search = w.$("inbox-search");
  search.value = "Mantle";
  search.dispatchEvent({ type: "input" });
  assert.equal(state("query"), "Mantle");
  assert.deepEqual(walk(list).filter((n) => n.tagName === "article").length, 1);

  // 全部忽略 dismisses the visible works
  w.$("inbox-dismiss").click();
  await tick();
  assert.deepEqual(w.last("setStatus"), ["setStatus", ["p2"], "dismissed"]);
  search.value = "";
  search.dispatchEvent({ type: "input" });

  // Tabs: added works can be shown in Zotero or moved back; dismissed ones restored
  w.tabs[1].click();
  assert.equal(state("tab"), "added");
  assert.equal(w.tabs[1].getAttribute("aria-selected"), "true");
  buttonsIn(list, "在 Zotero 中显示")[0].click();
  assert.deepEqual(w.last("showItem"), ["showItem", 42]);
  buttonsIn(list, "移回待查看")[0].click();
  await tick();
  assert.deepEqual(w.last("setStatus"), ["setStatus", "a1", "new"]);
  w.tabs[2].click();
  assert.equal(state("tab"), "dismissed");
  buttonsIn(list, "恢复")[0].click();
  await tick();
  assert.deepEqual(w.last("setStatus"), ["setStatus", "d1", "new"]);
  assert.deepEqual(w.errors, []);
});
