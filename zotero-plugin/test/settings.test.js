// The 设置 page (addon/content/settings.js, with main.js) in a fake DOM: the
// pure binding helpers (pref types, clamp notes, statuses), the field table
// against src/settings.js, and the rendered page writing prefs with the
// default's type, debouncing and flushing text input, live previews,
// 恢复默认, the journal whitelist and the LLM presets.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import { parseFollowedAuthors } from "../src/core/authors.js";
import { journalRows, mergeWhitelistRows, summarizeWhitelist } from "../src/core/journals.js";
import { EMBEDDING_PRESETS } from "../src/embedding/providers.js";
import { autoValues, loadSettings } from "../src/settings.js";
import { PREFS_0_2_0 } from "./helpers.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const mainCode = readFileSync(join(root, "addon/content/main.js"), "utf8");
const searchCode = readFileSync(join(root, "addon/content/search.js"), "utf8");
const settingsCode = readFileSync(join(root, "addon/content/settings.js"), "utf8");
const PREFIX = "extensions.zotwatch.";
// Values created inside the VM have foreign prototypes; compare them structurally
const plain = (x) => JSON.parse(JSON.stringify(x));

/** Default prefs of addon/prefs.js without the prefix. */
function prefDefaults() {
  const defaults = {};
  vm.runInNewContext(readFileSync(join(root, "addon/prefs.js"), "utf8"), { pref: (k, v) => (defaults[k.slice(PREFIX.length)] = v) });
  return defaults;
}

// ------------------------------------------------------------- fake DOM

class FakeNode {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.localName = tag;
    this.childNodes = [];
    this.attributes = {};
    this.listeners = {};
    this.ownText = "";
    this.checked = false;
    this.scrollTop = 0;
    this.style = {};
    this.dataset = {};
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
  // Reflected attributes (an input's value starts as its value attribute)
  get value() {
    return this.currentValue ?? this.attributes.value ?? "";
  }
  set value(v) {
    this.currentValue = String(v);
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
  get type() {
    return this.attributes.type || "";
  }
  set type(v) {
    this.attributes.type = String(v);
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
    this.ownText = String(value);
    this.childNodes = [];
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
  /** Fire an event; `extra` adds fields such as `key` for keyboard events. */
  dispatch(type, extra = {}) {
    const event = { type, target: this, defaultPrevented: false, preventDefault() {
      event.defaultPrevented = true;
    }, stopPropagation() {}, ...extra };
    for (const fn of this.listeners[type] || []) fn(event);
    return event;
  }
  focus() {}
  // External text never goes through HTML parsing: any use fails the test
  set innerHTML(value) {
    throw new Error(`innerHTML is not allowed (${String(value).slice(0, 40)})`);
  }
  set outerHTML(value) {
    throw new Error(`outerHTML is not allowed (${String(value).slice(0, 40)})`);
  }
  append(...nodes) {
    this.childNodes.push(...nodes);
  }
  replaceChildren(...nodes) {
    this.ownText = "";
    this.childNodes = nodes;
  }
  remove() {}
  querySelectorAll() {
    return [];
  }
  querySelector() {
    return null;
  }
  scrollIntoView() {}
}

function walk(node, out = []) {
  out.push(node);
  for (const child of node.childNodes || []) walk(child, out);
  return out;
}
const byClass = (node, cls) => walk(node).filter((n) => (n.className || "").split(" ").includes(cls));
const byId = (node, id) => walk(node).find((n) => n.id === id);
const byPref = (node, pref) => walk(node).filter((n) => n.getAttribute?.("data-pref") === pref);
/** The .field row of a pref (hidden when the field does not apply). */
const rowOf = (node, pref) => walk(node).find((n) => n.getAttribute?.("data-field") === pref);

// --------------------------------------------------------------- window

/**
 * Load main.js + settings.js with a fake Zotero (prefs typed like Firefox's:
 * writing another type than the default's throws) and a manual clock.
 */
function loadSettingsPage({ prefs = {}, whitelist = "" } = {}) {
  const defaults = prefDefaults();
  const values = new Map(Object.entries({ ...defaults, ...prefs }));
  const writes = [];
  const Prefs = {
    get: (key) => values.get(key.slice(PREFIX.length)),
    set(key, value) {
      const short = key.slice(PREFIX.length);
      if (!(short in defaults)) throw new Error(`No default for ${key}`);
      if (typeof value !== typeof defaults[short]) throw new Error(`Pref ${key} expects a ${typeof defaults[short]}, got ${typeof value}`);
      values.set(short, value);
      writes.push([short, value]);
    },
  };

  let now = 0;
  let nextId = 1;
  const timers = new Map();
  const clock = {
    setTimeout(fn, ms = 0) {
      const id = nextId++;
      timers.set(id, { fn, at: now + ms });
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
    /** Run the timers due within `ms` (in order). */
    async advance(ms) {
      const until = now + ms;
      for (;;) {
        const due = [...timers.entries()].filter(([, t]) => t.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        timers.delete(due[0]);
        now = due[1].at;
        due[1].fn();
        await new Promise((r) => setImmediate(r));
      }
      now = until;
    },
  };

  const saved = [];
  const checks = [];
  const resets = [];
  // failSaves: how many of the next whitelist saves fail (like the addon: nothing changes then)
  // checkRows: rows checkConnections() answers per key (default: every key passes, llm fails);
  // apiStatus: what getApiStatus() returns (the 学术数据接口 budget line)
  const control = {
    failSaves: 0,
    confirmReply: true,
    allowErrors: false,
    errors: [],
    checkRows: {},
    apiStatus: { openalexKey: false, semanticScholarKey: false, mailto: false, openalexUsage: null },
    // Journal tools: proposals of 从文库生成, answers of the search boxes and of the name lookup
    proposals: { proposals: [], warnings: [], missing: [], stats: {} },
    fills: [],
    fillResult: (text) => ({ text, filled: 0, incomplete: 0, notFound: [] }),
    journalCancels: 0,
    searches: [],
    searchJournals: async () => [],
    searchAuthors: async () => [],
    nameLookups: [],
    lookupNames: async () => ({ names: {}, error: null }),
  };
  const api = {
    version: "0.3.0",
    whitelistError: null,
    lastPage: null,
    getAutoValues: () => autoValues(values.get("embedding.provider")),
    parseFollowedAuthors,
    summarizeWhitelist,
    getJournalWhitelist: () => whitelist,
    async setJournalWhitelist(text) {
      if (control.failSaves > 0) {
        control.failSaves -= 1;
        throw new Error("journals.csv is locked");
      }
      whitelist = text;
      saved.push(text);
    },
    async resetData() {
      resets.push(true);
    },
    // OpenAlex-backed journal tools (journal-proposals.js / openalex-entities.js), with the real merge
    async suggestJournals() {
      return control.proposals;
    },
    mergeJournals: (text, journals) => mergeWhitelistRows(text, journalRows(journals)),
    async fillJournalMetadata(text) {
      control.fills.push(text);
      return control.fillResult(text);
    },
    cancelJournalAction() {
      control.journalCancels += 1;
    },
    async searchJournals(query) {
      control.searches.push(["journals", query]);
      return control.searchJournals(query);
    },
    async searchAuthors(query) {
      control.searches.push(["authors", query]);
      return control.searchAuthors(query);
    },
    async lookupJournalNames(issns) {
      control.nameLookups.push(issns);
      return control.lookupNames(issns);
    },
    async loadPresetJournals() {
      return "issn,title\n0016-7037,GCA";
    },
    async checkConnections(keys) {
      checks.push(keys);
      return keys.map((key) => control.checkRows[key] || { key, name: key, ok: key !== "llm", detail: key === "llm" ? "HTTP 401" : "正常，维度 1024" });
    },
    getApiStatus: () => control.apiStatus,
    getDataDir: () => "/data/zotwatch",
    async getView() {
      return { works: [], lastRun: null, history: [], running: false, schedule: null, version: "0.3.0" };
    },
    addListener() {},
    removeListener() {},
  };
  const elements = new Map();
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
    addEventListener() {},
  };
  const confirms = [];
  const winListeners = {};
  // 导入 CSV… picks this file; its text comes from Zotero.File.getContentsAsync
  const picker = { file: "/home/me/journals.csv", text: "issn,title\n1234-5679,Imported" };
  class FilePicker {
    constructor() {
      this.modeOpen = 0;
      this.filterAll = 1;
      this.returnOK = 0;
      this.returnCancel = 1;
    }
    init() {}
    appendFilter() {}
    appendFilters() {}
    async show() {
      this.file = picker.file;
      return picker.file ? this.returnOK : this.returnCancel;
    }
  }
  const Zotero = {
    ZotWatch: api,
    Prefs,
    isMac: false,
    File: { getContentsAsync: async () => picker.text },
    logError(e) {
      // Unexpected errors fail the test; expected ones (a failing save) are recorded
      if (!control.allowErrors) throw e;
      control.errors.push(e);
    },
  };
  const context = vm.createContext({
    ChromeUtils: { importESModule: (url) => (url.endsWith("filePicker.mjs") ? { FilePicker } : { Zotero }) },
    window: {
      addEventListener: (type, fn) => (winListeners[type] ||= []).push(fn),
      confirm: (text) => (confirms.push(text), control.confirmReply),
      close() {},
    },
    document,
    location: { hash: "" },
    history: { replaceState() {} },
    Intl,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    setInterval: () => 0,
    clearInterval() {},
  });
  vm.runInContext(mainCode, context, { filename: "main.js" });
  vm.runInContext(searchCode, context, { filename: "search.js" });
  vm.runInContext(settingsCode, context, { filename: "settings.js" });
  const run = (src) => vm.runInContext(src, context);
  const page = run("ZotWatchSettings");
  /** Fire a window event ("load" starts the shell like Zotero does, "unload" closes the window). */
  const fire = (type) => (winListeners[type] || []).forEach((fn) => fn({ type }));
  return { page, run, fire, root: document.getElementById("settings-root"), values, writes, saved, checks, confirms, resets, control, picker, clock, api, toasts: document.getElementById("toasts") };
}

// ------------------------------------------------------------- helpers

test("settings helpers: values keep the default's type and number notes mirror the clamping", () => {
  const { page } = loadSettingsPage();
  const { coercePrefValue, effectiveNumber, clampNote, formatSettingValue, rangeText } = page.helpers;

  // Booleans for switches, strings for everything else (numbers are stored as strings)
  assert.equal(coercePrefValue(true, false), true);
  assert.equal(coercePrefValue("false", true), false);
  assert.equal(coercePrefValue("24", "7"), "24");
  assert.equal(coercePrefValue(7, ""), "7");
  assert.equal(coercePrefValue(null, "x"), "");
  // Integer prefs (none ship today) are rounded, garbage keeps the old value
  assert.equal(coercePrefValue("12.6", 3), 13);
  assert.equal(coercePrefValue("abc", 3), 3);
  // No default: never written
  assert.equal(coercePrefValue("x", undefined), undefined);

  const days = page.FIELDS.find((f) => f.pref === "followed.daysBack");
  assert.deepEqual(plain(effectiveNumber("60", days, 60)), { value: 60, reason: null });
  assert.deepEqual(plain(effectiveNumber(" ", days, 60)), { value: 60, reason: "blank" });
  assert.equal(clampNote("999", days, 60), "超出范围，将按 365 使用");
  assert.equal(clampNote("3", days, 60), "超出范围，将按 7 使用");
  assert.equal(clampNote("60", days, 60), "");
  assert.equal(clampNote("", days, 60), "", "blank uses the default silently (shown as placeholder)");
  assert.equal(clampNote("abc", days, 60), "不是有效的数字，将按默认值 60 使用");
  assert.equal(clampNote("12.5", days, 60), "只能填整数，将按 12 使用");
  assert.equal(rangeText(days), "7–365 天");

  const minScore = page.FIELDS.find((f) => f.pref === "flagship.minScore");
  assert.equal(clampNote("1.5", minScore, 0.3), "超出范围，将按 1.00 使用");
  assert.equal(clampNote("x", minScore, 0.3), "不是有效的数字，将按自动值 0.30 使用");
  assert.equal(clampNote("0.3x", minScore, 0.3), "将按 0.30 使用");
  assert.equal(clampNote("0.25", minScore, 0.3), "");
  assert.equal(formatSettingValue(0.2, minScore), "0.20");
  assert.equal(formatSettingValue(24), "24");
});

test("settings helpers: flagship defaults, ISSN counts and section statuses", () => {
  const { page } = loadSettingsPage();
  const { issnList, isDefaultText, restoreMeta, sectionStatus } = page.helpers;
  const auto = autoValues("dashscope");
  const issns = page.FIELDS.find((f) => f.pref === "flagship.issns");
  const anchors = page.FIELDS.find((f) => f.pref === "flagship.negativeAnchors");

  assert.deepEqual(plain(issnList("0028-0836, 00280836\n1476-4687；bad")), ["0028-0836", "1476-4687"]);
  assert.equal(isDefaultText(issns, "", auto), true);
  assert.equal(isDefaultText(issns, [...auto.flagship.issns].reverse().join(", "), auto), true, "ISSN order and separators do not matter");
  assert.equal(isDefaultText(issns, "0028-0836", auto), false);
  assert.equal(isDefaultText(anchors, `  ${auto.flagship.negativeAnchors.join("\n\n")}\n`, auto), true);
  assert.equal(restoreMeta(issns, "", auto), "12 个 ISSN · 默认");
  assert.equal(restoreMeta(issns, "0028-0836\n1476-4687", auto), "2 个 ISSN · 已自定义");
  assert.equal(restoreMeta(anchors, "", auto), "6 个排除领域 · 默认");

  const prefs = { ...prefDefaults(), "embedding.provider": "dashscope", "followed.enabled": true };
  const get = (key) => prefs[key];
  const ctx = { auto, version: "0.3.0", whitelist: { count: 3 }, followed: parseFollowedAuthors("A, 0000-0002-1825-0097\nnobody") };
  assert.deepEqual(plain(sectionStatus("general", get, ctx)), { tone: "good", text: "每 24 小时自动检查" });
  assert.deepEqual(plain(sectionStatus("embedding", get, ctx)), { tone: "warning", text: "DashScope · 未填写 API Key" });
  prefs["embedding.apiKey"] = "sk";
  assert.deepEqual(plain(sectionStatus("embedding", get, ctx)), { tone: "good", text: "DashScope · 已配置" });
  assert.deepEqual(plain(sectionStatus("sources", get, ctx)), { tone: "good", text: "Crossref" });
  assert.deepEqual(plain(sectionStatus("journals", get, ctx)), { tone: "good", text: "3 个期刊 · 并自动关注文库常见期刊" });
  assert.deepEqual(plain(sectionStatus("followed", get, ctx)), { tone: "warning", text: "1 位作者，1 行无法识别" });
  assert.deepEqual(plain(sectionStatus("flagship", get, ctx)), { tone: "off", text: "未启用" });
  prefs["flagship.enabled"] = true;
  assert.deepEqual(plain(sectionStatus("flagship", get, ctx)), { tone: "good", text: "12 个 ISSN · 最多 30 篇" });
  assert.deepEqual(plain(sectionStatus("ranking", get, ctx)), { tone: "neutral", text: "每次 30 篇 · 最近 7 天" });
  assert.deepEqual(plain(sectionStatus("ai", get, ctx)), { tone: "off", text: "未启用" });
  prefs["llm.enabled"] = true;
  assert.deepEqual(plain(sectionStatus("ai", get, ctx)), { tone: "warning", text: "未填写 API Key" });
  assert.deepEqual(plain(sectionStatus("zotero", get, ctx)), { tone: "neutral", text: "分类「ZotWatch 推荐」 · 标签 ZotWatch" });
  assert.deepEqual(plain(sectionStatus("data", get, ctx)), { tone: "neutral", text: "ZotWatch v0.3.0" });
  prefs["schedule.autoRun"] = false;
  assert.deepEqual(plain(sectionStatus("general", get, ctx)), { tone: "off", text: "自动检查已关闭" });
});

test("number fields use the ranges and fallbacks of src/settings.js", () => {
  const { page } = loadSettingsPage();
  const defaults = prefDefaults();
  const numbers = page.FIELDS.filter((f) => f.kind === "number");
  assert.ok(numbers.length >= 15);
  for (const spec of numbers) {
    // flagship.grayLow is capped at minScore: lift it so the field's own range shows
    const base = spec.pref === "flagship.grayLow" ? { ...defaults, "flagship.minScore": "1" } : defaults;
    const read = (value) => JSON.stringify(loadSettings((key) => (key === spec.pref ? value : base[key])));
    const step = spec.int ? 1 : 0.01;
    const fallback = spec.auto ? autoValues(base["embedding.provider"])[spec.auto] : spec.fallback;
    assert.ok(Number.isFinite(fallback), `${spec.pref}: fallback`);
    assert.equal(read(""), read(String(fallback)), `${spec.pref}: blank means ${fallback}`);
    assert.equal(read(String(spec.max + 1)), read(String(spec.max)), `${spec.pref}: clamped at ${spec.max}`);
    assert.notEqual(read(String(spec.max)), read(String(spec.max - step)), `${spec.pref}: ${spec.max} is reachable`);
    assert.equal(read(String(spec.min - 1)), read(String(spec.min)), `${spec.pref}: clamped at ${spec.min}`);
    assert.notEqual(read(String(spec.min)), read(String(spec.min + step)), `${spec.pref}: ${spec.min} is reachable`);
    if (spec.int) assert.equal(read(String(spec.min + 1.6)), read(String(spec.min + 1)), `${spec.pref}: integer`);
    assert.equal(typeof defaults[spec.pref], "string", `${spec.pref}: numbers are stored as strings`);
  }
});

test("embedding providers and LLM presets match the services", () => {
  const { page } = loadSettingsPage();
  for (const [key, preset] of Object.entries(EMBEDDING_PRESETS)) {
    assert.equal(page.PROVIDERS[key].model, preset.model, `${key} model`);
    assert.equal(page.PROVIDERS[key].baseURL, preset.baseURL, `${key} endpoint (the 接口地址 placeholder)`);
  }
  assert.deepEqual(plain(page.FIELDS.find((f) => f.pref === "embedding.provider").options.map((o) => o.value)), ["local", "dashscope", "voyage", "openai"]);
  // The default LLM endpoint and model are the OpenRouter preset
  const defaults = prefDefaults();
  const openrouter = page.LLM_PRESETS.find((p) => p.key === "openrouter");
  assert.equal(openrouter.baseURL, defaults["llm.baseURL"]);
  assert.equal(openrouter.model, defaults["llm.model"]);
});

// ------------------------------------------------------------- the page

test("the settings page binds every setting and writes it with the default's type", async () => {
  const { page, root, values, writes, clock } = loadSettingsPage({ prefs: { "embedding.provider": "dashscope" } });
  page.init();
  page.show({});

  // Every field is rendered with data-pref; every 0.2.0 pref is among them
  const bound = new Set(walk(root).map((n) => n.getAttribute?.("data-pref")).filter(Boolean));
  assert.deepEqual([...bound].sort(), plain(page.FIELDS.map((f) => f.pref)).sort());
  assert.deepEqual(PREFS_0_2_0.filter((key) => !bound.has(key)), []);
  const defaults = prefDefaults();
  for (const key of bound) assert.ok(key in defaults, `no default for ${key}`);
  assert.equal(writes.length, 0, "showing the page writes nothing");

  // The index lists every section with a status; sections are hidden until chosen
  assert.equal(byClass(root, "sf-index-item").length, 11);
  assert.equal(byId(root, "settings-index").hidden, false);
  assert.equal(byId(root, "settings-ai").hidden, true);
  const statuses = byClass(root, "sf-index-status").map((n) => n.textContent);
  assert.ok(statuses.includes("DashScope · 未填写 API Key"), statuses.join(" | "));

  page.show({ section: "general" });
  assert.equal(byId(root, "settings-index").hidden, true);
  assert.equal(byId(root, "settings-general").hidden, false);
  assert.equal(byId(root, "settings-ai").hidden, true);

  // Values are filled from the prefs
  const [autoRun] = byPref(root, "schedule.autoRun");
  const [interval] = byPref(root, "schedule.intervalHours");
  assert.equal(autoRun.checked, true);
  assert.equal(interval.value, "24");
  assert.equal(interval.getAttribute("placeholder"), "24");

  // A switch writes a boolean at once
  autoRun.checked = false;
  autoRun.dispatch("change");
  assert.deepEqual(writes.at(-1), ["schedule.autoRun", false]);

  // Text input is written as a string 600 ms after typing stops
  interval.value = " 12 ";
  interval.dispatch("input");
  assert.equal(values.get("schedule.intervalHours"), "24");
  await clock.advance(599);
  assert.equal(values.get("schedule.intervalHours"), "24");
  await clock.advance(1);
  assert.equal(values.get("schedule.intervalHours"), "12");

  // An out-of-range value shows the value that will be used, and is stored as typed
  const [days] = byPref(root, "followed.daysBack");
  const note = byClass(rowOf(root, "followed.daysBack"), "field-note")[0];
  days.value = "999";
  days.dispatch("input");
  assert.equal(note.hidden, false);
  assert.equal(note.textContent, "超出范围，将按 365 使用");
  days.value = "90";
  days.dispatch("input");
  assert.equal(note.hidden, true);

  // flush() (page switch, window close) saves what is still waiting
  page.flush();
  assert.equal(values.get("followed.daysBack"), "90");

  // A change event (blur) writes immediately; unchanged values are not rewritten
  const [mailto] = byPref(root, "sources.crossref.mailto");
  mailto.value = "me@example.org";
  mailto.dispatch("change");
  assert.equal(values.get("sources.crossref.mailto"), "me@example.org");
  const count = writes.length;
  mailto.dispatch("change");
  assert.equal(writes.length, count);

  // Segmented control: the threshold mode
  const [mode] = byPref(root, "scoring.thresholdMode");
  assert.equal(mode.childNodes.find((b) => b.getAttribute("data-value") === "dynamic").classList.contains("active"), true);
  mode.childNodes.find((b) => b.getAttribute("data-value") === "fixed").dispatch("click");
  assert.equal(values.get("scoring.thresholdMode"), "fixed");
});

test("provider radio cards switch the fields, placeholders and the flagship notice", () => {
  const { page, root, values } = loadSettingsPage({ prefs: { "embedding.provider": "dashscope" } });
  page.init();
  page.show({ section: "embedding" });
  const radios = byPref(root, "embedding.provider");
  assert.equal(radios.length, 4);
  assert.equal(radios.find((r) => r.checked).value, "dashscope");
  assert.equal(rowOf(root, "embedding.apiKey").hidden, false);
  // Every API service sends its requests to 接口地址 (a stored one stays visible and editable)
  assert.equal(rowOf(root, "embedding.baseURL").hidden, false);
  assert.equal(byPref(root, "embedding.baseURL")[0].getAttribute("placeholder"), "https://dashscope.aliyuncs.com/compatible-mode/v1");
  assert.equal(rowOf(root, "embedding.dimensions").hidden, false);
  assert.equal(byPref(root, "embedding.model")[0].getAttribute("placeholder"), "text-embedding-v4");
  assert.equal(byPref(root, "scoring.minSimilarity")[0].getAttribute("placeholder"), "自动（0.20）");
  assert.equal(byPref(root, "flagship.minScore")[0].getAttribute("placeholder"), "自动（0.30）");

  for (const r of radios) r.checked = r.value === "local";
  radios.find((r) => r.value === "local").dispatch("change");
  assert.equal(values.get("embedding.provider"), "local");
  assert.equal(rowOf(root, "embedding.apiKey").hidden, true);
  assert.equal(rowOf(root, "embedding.model").hidden, true);
  assert.equal(rowOf(root, "embedding.baseURL").hidden, true, "local mode sends nothing");
  assert.equal(byClass(byId(root, "settings-embedding"), "sf-advanced")[0].hidden, true, "no empty advanced group");
  assert.equal(byPref(root, "scoring.minSimilarity")[0].getAttribute("placeholder"), "自动（0.06）");
  assert.equal(byPref(root, "flagship.minScore")[0].getAttribute("placeholder"), "自动（0.12）");
  const callout = byClass(byId(root, "settings-flagship"), "callout").find((n) => n.classList.contains("warning"));
  assert.equal(callout.hidden, false, "the local-mode explanation shows in the flagship section");

  for (const r of radios) r.checked = r.value === "openai";
  radios.find((r) => r.value === "openai").dispatch("change");
  assert.equal(rowOf(root, "embedding.baseURL").hidden, false);
  assert.equal(byPref(root, "embedding.baseURL")[0].getAttribute("placeholder"), "https://api.openai.com/v1");
  assert.equal(rowOf(root, "embedding.dimensions").hidden, true);
  assert.equal(callout.hidden, true);

  // An endpoint entered for one service is still shown after switching to another, which uses it too
  const [endpoint] = byPref(root, "embedding.baseURL");
  endpoint.value = "https://api.siliconflow.cn/v1";
  endpoint.dispatch("change");
  for (const r of radios) r.checked = r.value === "voyage";
  radios.find((r) => r.value === "voyage").dispatch("change");
  assert.equal(rowOf(root, "embedding.baseURL").hidden, false);
  assert.equal(endpoint.value, "https://api.siliconflow.cn/v1");
  assert.equal(endpoint.getAttribute("placeholder"), "https://api.voyageai.com/v1");
  assert.equal(loadSettings((k) => values.get(k)).embedding.baseURL, "https://api.siliconflow.cn/v1");

  // The password field has a show/hide button
  const [key] = byPref(root, "embedding.apiKey");
  const eye = byClass(rowOf(root, "embedding.apiKey"), "sf-eye")[0];
  assert.equal(key.type, "password");
  eye.dispatch("click");
  assert.equal(key.type, "text");
  assert.equal(eye.getAttribute("aria-pressed"), "true");
  eye.dispatch("click");
  assert.equal(key.type, "password");
});

test("followed authors and the journal list have live previews; whitelist edits are saved", async () => {
  const { page, root, values, saved, clock, confirms, api } = loadSettingsPage({ whitelist: "issn,title,category,impact_factor\n0016-7037,GCA,GEO,5.0" });
  page.init();
  page.show({ section: "followed" });

  const [authors] = byPref(root, "followed.authors");
  authors.value = "汪在聪, 0000-0002-3584-1673\nA. Smith A5023888391\nnobody";
  authors.dispatch("input");
  const preview = byClass(root, "sf-preview")[0];
  assert.equal(byClass(preview, "sf-author").length, 2);
  assert.match(byClass(preview, "sf-author")[0].textContent, /汪在聪ORCID0000-0002-3584-1673/);
  assert.match(byClass(preview, "sf-author")[1].textContent, /OpenAlexA5023888391/);
  assert.equal(byClass(preview, "sf-invalid")[0].textContent, "无法识别：nobody");
  assert.equal(byClass(preview, "sf-preview-head")[0].textContent, "已识别 2 位作者 · 1 行无法识别");
  // Like the 顶刊列表: a list title next to the counts, shown only with rows
  const listTitle = byClass(preview, "sf-list-title")[0];
  assert.equal(listTitle.textContent, "关注列表");
  assert.equal(listTitle.hidden, false);
  await clock.advance(600);
  assert.equal(values.get("followed.authors"), authors.value);
  authors.value = "";
  authors.dispatch("input");
  assert.equal(listTitle.hidden, true);
  assert.match(byClass(preview, "sf-preview-head")[0].textContent, /^还没有关注的作者/);
  authors.value = "汪在聪, 0000-0002-3584-1673\nA. Smith A5023888391\nnobody";
  authors.dispatch("input");
  await clock.advance(600);

  // Showing a section never writes the whitelist back (it is only saved after an edit)
  page.show({ section: "journals" });
  page.show({ section: "general" });
  page.show({ section: "journals" });
  assert.deepEqual(saved, []);
  const list = byId(root, "sp-whitelist");
  assert.equal(list.value, "issn,title,category,impact_factor\n0016-7037,GCA,GEO,5.0");
  assert.equal(byClass(root, "sf-summary")[0].textContent, "已识别 1 个期刊，1 个有影响因子");

  list.value += "\n0012-821X,EPSL,GEO\nnot an issn,X";
  list.dispatch("input");
  assert.equal(byClass(root, "sf-summary")[0].textContent, "已识别 2 个期刊，1 个有影响因子");
  const warning = byClass(root, "sf-inline-warning")[0];
  assert.equal(warning.hidden, false);
  assert.match(warning.textContent, /1 行无法识别（第一列须为 ISSN）：not an issn,X/);
  await clock.advance(800);
  assert.deepEqual(saved, [list.value]);

  // Generating the proposals locks the field and its buttons until they arrive (nothing is written)
  let finishSuggest;
  api.suggestJournals = () => new Promise((r) => (finishSuggest = r));
  const suggest = byId(root, "sp-journals-suggest");
  const buttons = ["sp-journals-suggest", "sp-journals-fill", "sp-journals-import", "sp-journals-preset"].map((id) => byId(root, id));
  suggest.dispatch("click");
  await new Promise((r) => setImmediate(r));
  assert.equal(list.disabled, true, "the list is locked while it is generated");
  assert.deepEqual(buttons.map((b) => b.disabled), [true, true, true, true]);
  assert.match(byClass(root, "sf-action-message")[0].textContent, /正在分析文库期刊/);
  const savedBefore = saved.length;
  finishSuggest({ proposals: [{ title: "Chemical Geology", issns: ["0009-2541"], category: "GEOCHEMISTRY_AND_PETROLOGY", impactFactor: 3.9, libraryCount: 3, origin: "library", checked: true, verified: true }], warnings: [], missing: [] });
  await new Promise((r) => setImmediate(r));
  assert.equal(list.disabled, false);
  assert.deepEqual(buttons.map((b) => b.disabled), [false, false, false, false]);
  assert.equal(byId(root, "sp-journals-review").hidden, false, "the review panel opens");
  assert.equal(saved.length, savedBefore, "nothing is saved before 加入所选");
  byId(root, "sp-journals-review-cancel").dispatch("click");
  assert.equal(byId(root, "sp-journals-review").hidden, true);

  // The example list replaces a non-empty list only after a confirmation
  byId(root, "sp-journals-preset").dispatch("click");
  await new Promise((r) => setImmediate(r));
  assert.match(confirms.at(-1), /将替换当前列表中的 \d+ 个期刊/);
  assert.equal(list.value, "issn,title\n0016-7037,GCA");
});

test("恢复默认, LLM presets and inline connection tests", async () => {
  const { page, root, values, checks } = loadSettingsPage({ prefs: { "embedding.provider": "dashscope" } });
  page.init();
  page.show({ section: "flagship" });
  const auto = autoValues("dashscope");

  const [issns] = byPref(root, "flagship.issns");
  const row = rowOf(root, "flagship.issns");
  const restore = byClass(row, "btn")[0];
  assert.equal(issns.value, auto.flagship.issns.join("\n"), "the default list is shown");
  assert.equal(restore.disabled, true);
  assert.equal(byClass(row, "sf-meta")[0].textContent, "12 个 ISSN · 默认");

  issns.value = "0028-0836\n1476-4687";
  issns.dispatch("input");
  assert.equal(byClass(row, "sf-meta")[0].textContent, "2 个 ISSN · 已自定义");
  assert.equal(restore.disabled, false);
  issns.dispatch("change");
  assert.equal(values.get("flagship.issns"), "0028-0836\n1476-4687");
  restore.dispatch("click");
  assert.equal(values.get("flagship.issns"), "", "恢复默认 stores a blank value (= the default)");
  assert.equal(issns.value, auto.flagship.issns.join("\n"));
  assert.equal(restore.disabled, true);

  // An edit that equals the default is stored blank too
  const [boundary] = byPref(root, "flagship.llmBoundary");
  boundary.value = `${auto.flagship.llmBoundary}\n`;
  boundary.dispatch("change");
  assert.equal(values.get("flagship.llmBoundary"), "");

  // A gray-zone floor above the acceptance threshold is capped at it
  const [grayLow] = byPref(root, "flagship.grayLow");
  grayLow.value = "0.4";
  grayLow.dispatch("input");
  assert.equal(byClass(rowOf(root, "flagship.grayLow"), "field-note")[0].textContent, "高于接受阈值，将按 0.30 使用");

  // LLM quick fill writes the endpoint and the model
  page.show({ section: "ai" });
  const chips = byClass(root, "sf-preset");
  assert.deepEqual(chips.map((c) => c.textContent), ["OpenRouter", "DeepSeek", "Kimi", "DashScope"]);
  assert.equal(chips[0].classList.contains("active"), true);
  chips[1].dispatch("click");
  assert.equal(values.get("llm.baseURL"), "https://api.deepseek.com/v1");
  assert.equal(values.get("llm.model"), "deepseek-chat");
  assert.equal(byPref(root, "llm.model")[0].value, "deepseek-chat");
  assert.equal(chips[1].classList.contains("active"), true);
  assert.equal(chips[0].classList.contains("active"), false);

  // 测试大模型 saves the typed key first, then shows the LLM row inline
  const [key] = byPref(root, "llm.apiKey");
  key.value = "sk-new";
  key.dispatch("input");
  const test = byClass(byId(root, "settings-ai"), "sf-test")[0];
  test.childNodes[0].dispatch("click");
  await new Promise((r) => setImmediate(r));
  assert.equal(values.get("llm.apiKey"), "sk-new");
  assert.deepEqual(plain(checks), [["llm"]]);
  const result = byClass(test, "sf-result")[0];
  assert.equal(result.classList.contains("critical"), true);
  assert.equal(result.textContent, "失败：HTTP 401");
});

const tick = () => new Promise((r) => setImmediate(r));

test("学术数据接口: keys with an eye toggle, the moved Crossref address, status, budget and 测试接口", async () => {
  const { page, root, values, checks, control } = loadSettingsPage();
  page.init();
  page.show({});
  const statusOf = (id) => byClass(root, "sf-index-item").find((n) => n.getAttribute("data-section") === id);
  assert.equal(byClass(statusOf("apis"), "sf-index-status")[0].textContent, "未配置 OpenAlex API Key");
  assert.equal(byClass(statusOf("apis"), "dot")[0].className, "dot warning");
  // The section comes right after 向量化
  assert.deepEqual(byClass(root, "sf-index-item").slice(1, 3).map((n) => n.getAttribute("data-section")), ["embedding", "apis"]);

  page.show({ section: "apis" });
  const section = byId(root, "settings-apis");
  assert.equal(section.hidden, false);
  // The Crossref contact address moved here from 文献来源 (same pref, bound once)
  assert.equal(byPref(section, "sources.crossref.mailto").length, 1);
  assert.equal(byPref(byId(root, "settings-sources"), "sources.crossref.mailto").length, 0);
  assert.equal(byPref(root, "sources.crossref.mailto").length, 1);
  assert.match(rowOf(section, "sources.crossref.mailto").textContent, /联系邮箱（Crossref 礼貌池）/);

  const [openalexKey] = byPref(section, "apis.openalexKey");
  const [s2Key] = byPref(section, "apis.semanticScholarKey");
  assert.equal(openalexKey.type, "password");
  assert.equal(s2Key.type, "password");
  const eye = byClass(rowOf(section, "apis.openalexKey"), "sf-eye")[0];
  eye.dispatch("click");
  assert.equal(openalexKey.type, "text");
  eye.dispatch("click");
  assert.equal(openalexKey.type, "password");

  // Without a key, the budget line warns
  const usage = byClass(section, "sf-usage")[0];
  assert.equal(usage.textContent, "未配置 Key：每日只有很少的免费额度");
  assert.equal(byClass(usage, "dot")[0].className, "dot warning");

  openalexKey.value = " oa-key ";
  openalexKey.dispatch("change");
  assert.equal(values.get("apis.openalexKey"), "oa-key");
  assert.equal(byClass(statusOf("apis"), "sf-index-status")[0].textContent, "OpenAlex 已配置");
  assert.equal(usage.textContent, "运行一次或点击「测试接口」后显示");
  s2Key.value = "s2-key";
  s2Key.dispatch("change");
  assert.equal(byClass(statusOf("apis"), "sf-index-status")[0].textContent, "OpenAlex、Semantic Scholar 已配置");
  assert.equal(byClass(statusOf("apis"), "dot")[0].className, "dot good");

  // 测试接口 checks the three services together; each result names its service
  control.checkRows = {
    openalex: { key: "openalex", name: "OpenAlex", ok: true, detail: "Key 有效，今日剩余 $0.95 / $1.00" },
    semanticscholar: { key: "semanticscholar", name: "Semantic Scholar", ok: false, detail: "Key 无效（HTTP 403）" },
    crossref: { key: "crossref", name: "Crossref", ok: true, detail: "可访问" },
  };
  control.apiStatus = { ...control.apiStatus, openalexKey: true, openalexUsage: { remainingUsd: 0.95, limitUsd: 1, at: Date.now() - 3 * 3600000 - 60000, text: "今日剩余 $0.95 / $1.00" } };
  const test = byClass(section, "sf-test")[0];
  assert.match(test.childNodes[0].textContent, /测试接口/);
  test.childNodes[0].dispatch("click");
  await tick();
  assert.deepEqual(plain(checks.at(-1)), ["openalex", "semanticscholar", "crossref"]);
  assert.deepEqual(byClass(test, "sf-result").map((n) => n.textContent), ["OpenAlex：Key 有效，今日剩余 $0.95 / $1.00", "Semantic Scholar：Key 无效（HTTP 403）", "Crossref：可访问"]);
  assert.deepEqual(byClass(test, "sf-result").map((n) => n.className), ["sf-result good", "sf-result critical", "sf-result good"]);
  // OpenAlex reachable without a key is a warning, Semantic Scholar without one a plain notice
  control.checkRows.openalex = { key: "openalex", name: "OpenAlex", ok: null, detail: "可访问，但未配置 API Key（每日免费额度很小）" };
  control.checkRows.semanticscholar = { key: "semanticscholar", name: "Semantic Scholar", ok: null, detail: "未配置 API Key" };
  test.childNodes[0].dispatch("click");
  await tick();
  assert.deepEqual(byClass(test, "sf-result").map((n) => n.className), ["sf-result warning", "sf-result off", "sf-result good"]);
  // The budget the OpenAlex check read is shown right away
  assert.equal(usage.textContent, "今日剩余 $0.95 / $1.00 · 3 小时前更新");
  assert.equal(byClass(usage, "dot")[0].className, "dot good");
});

test("pending edits are saved on a page switch, a section switch and when the window closes", async () => {
  const { page, run, fire, root, values, saved } = loadSettingsPage({ whitelist: "issn,title\n0016-7037,GCA" });
  // The shell starts the way Zotero starts it (init() of every page, then the first page)
  fire("load");
  await tick();
  run("navigate('settings/general')");

  // Typed, then a page switch before the 600 ms debounce
  const [interval] = byPref(root, "schedule.intervalHours");
  interval.value = "12";
  interval.dispatch("input");
  assert.equal(values.get("schedule.intervalHours"), "24");
  run("navigate('inbox')");
  assert.equal(values.get("schedule.intervalHours"), "12");

  // Typed, then another section
  run("navigate('settings/ranking')");
  const [topK] = byPref(root, "scoring.topK");
  topK.value = "50";
  topK.dispatch("input");
  run("navigate('settings/ai')");
  assert.equal(values.get("scoring.topK"), "50");

  // The journal list (800 ms debounce), then the window closes
  run("navigate('settings/journals')");
  const list = byId(root, "sp-whitelist");
  list.value += "\n0012-821X,EPSL";
  list.dispatch("input");
  assert.deepEqual(saved, []);
  fire("unload");
  await tick();
  assert.deepEqual(saved, [list.value]);
  assert.equal(page.state.disposed, true);
});

test("guards and retries: reset and import confirmations, 测试向量化, failed whitelist saves", async () => {
  const original = "issn,title\n0016-7037,GCA";
  const { page, root, saved, checks, confirms, resets, control, picker, clock, toasts } = loadSettingsPage({ whitelist: original });
  page.init();

  // 清除缓存与推荐记录… asks first; declining deletes nothing
  page.show({ section: "data" });
  const reset = byClass(byId(root, "settings-data"), "danger")[0];
  control.confirmReply = false;
  reset.dispatch("click");
  await tick();
  assert.match(confirms.at(-1), /^将删除向量缓存、聚类结果和全部推荐记录/);
  assert.equal(resets.length, 0);
  control.confirmReply = true;
  reset.dispatch("click");
  await tick();
  assert.equal(resets.length, 1);

  // 导入 CSV… replaces a non-empty list only after a confirmation
  page.show({ section: "journals" });
  const list = byId(root, "sp-whitelist");
  const importButton = byId(root, "sp-journals-import");
  control.confirmReply = false;
  importButton.dispatch("click");
  await tick();
  assert.match(confirms.at(-1), /^将替换当前列表中的 1 个期刊/);
  assert.equal(list.value, original);
  assert.deepEqual(saved, []);
  control.confirmReply = true;
  importButton.dispatch("click");
  await tick();
  assert.equal(list.value, picker.text);
  assert.deepEqual(saved, [picker.text]);

  // 测试向量化 tests the embedding service only
  page.show({ section: "embedding" });
  const testRow = byClass(byId(root, "settings-embedding"), "sf-test")[0];
  testRow.childNodes[0].dispatch("click");
  await tick();
  assert.deepEqual(plain(checks.at(-1)), ["embedding"]);
  assert.equal(byClass(testRow, "sf-result")[0].textContent, "正常，维度 1024");

  // A failed save (journals.csv locked) is retried by the next flush
  page.show({ section: "journals" });
  control.allowErrors = true;
  control.failSaves = 1;
  list.value += "\n0012-821X,EPSL";
  list.dispatch("input");
  await clock.advance(800);
  assert.notEqual(saved.at(-1), list.value);
  assert.match(toasts.textContent, /期刊列表保存失败：journals.csv is locked/);
  await page.flush();
  assert.equal(saved.at(-1), list.value, "retried");

  // ... also when the list came from a button (载入地学期刊示例)
  control.failSaves = 1;
  byId(root, "sp-journals-preset").dispatch("click");
  await tick();
  assert.equal(list.value, "issn,title\n0016-7037,GCA");
  assert.notEqual(saved.at(-1), list.value);
  await page.flush();
  assert.equal(saved.at(-1), list.value, "retried");
  assert.equal(control.errors.length, 2);
});

test("向量维度 warns about sizes DashScope rejects; blank arXiv categories mean the defaults", () => {
  const { page, root } = loadSettingsPage({ prefs: { "embedding.provider": "dashscope" } });
  page.init();
  page.show({ section: "embedding" });
  const [dims] = byPref(root, "embedding.dimensions");
  const note = byClass(rowOf(root, "embedding.dimensions"), "field-note")[0];
  dims.value = "1000";
  dims.dispatch("input");
  assert.equal(note.hidden, false);
  assert.match(note.textContent, /^DashScope 不支持 1000 维/);
  dims.value = "1536";
  dims.dispatch("input");
  assert.equal(note.hidden, true);
  dims.value = "4096";
  dims.dispatch("input");
  assert.equal(note.textContent, "超出范围，将按 2048 使用");
  assert.equal(loadSettings((k) => (k === "embedding.dimensions" ? "4096" : undefined)).embedding.dimensions, 2048);

  // The 分类 placeholder is what a blank field uses (it no longer turns arXiv off)
  const spec = page.FIELDS.find((f) => f.pref === "sources.arxiv.categories");
  const defaults = prefDefaults();
  const blank = loadSettings((k) => (k === "sources.arxiv.categories" ? " " : defaults[k]));
  assert.deepEqual(blank.sources.arxiv.categories, spec.placeholder.split(/,\s*/));
  assert.deepEqual(blank.sources.arxiv.categories, loadSettings((k) => defaults[k]).sources.arxiv.categories);
});

// ------------------------------------------- 0.4.0: search boxes, review panel, lists

const flushPromises = () => new Promise((r) => setImmediate(r));
const journal = (title, issns, extra = {}) => ({ title, issns, impactFactor: 3.9, worksCount: 12000, publisher: "Elsevier BV", country: "NL", category: "GEOCHEMISTRY_AND_PETROLOGY", subfield: "Geochemistry and Petrology", ...extra });

test("search box: debounce, minimum length, stale answers, keyboard and 已在列表", async () => {
  const { page, root, clock, control, saved } = loadSettingsPage({ whitelist: "issn,title,category,impact_factor\n0016-7037,GCA,GEO,5.0" });
  const answers = [];
  control.searchJournals = (query) => new Promise((resolve) => answers.push({ query, resolve }));
  page.init();
  page.show({ section: "journals" });
  const input = byId(root, "sp-journal-search");
  const box = walk(root).find((n) => n.className === "sb" && walk(n).includes(input));
  const results = byId(root, "sp-journal-search-results");
  assert.equal(input.getAttribute("role"), "combobox");
  assert.equal(input.getAttribute("aria-controls"), "sp-journal-search-results");

  input.value = "g";
  input.dispatch("input");
  await clock.advance(1000);
  assert.deepEqual(control.searches, [], "one character does not search");

  input.value = "geo";
  input.dispatch("input");
  await clock.advance(300);
  input.value = "geochim";
  input.dispatch("input");
  await clock.advance(399);
  assert.deepEqual(control.searches, [], "debounced");
  await clock.advance(1);
  assert.deepEqual(control.searches, [["journals", "geochim"]]);
  assert.equal(byClass(box, "sb-spinner")[0].hidden, false, "spinner while searching");
  input.value = "geochimica";
  input.dispatch("input");
  await clock.advance(400);
  // The older answer arrives last and is ignored
  answers[1].resolve([journal("Geochimica et Cosmochimica Acta", ["0016-7037", "1872-9533"], { impactFactor: 5.04 }), journal("Chemical Geology", ["0009-2541"])]);
  await flushPromises();
  answers[0].resolve([journal("Stale result", ["1111-1119"])]);
  await flushPromises();
  const rows = byClass(results, "sb-result");
  assert.equal(results.hidden, false);
  assert.equal(input.getAttribute("aria-expanded"), "true");
  assert.equal(rows.length, 2);
  assert.match(rows[0].textContent, /^Geochimica et Cosmochimica ActaElsevier BV · NL · 0016-7037 \/ 1872-9533 · 近似 IF 5\.0 · 发文 12,000 篇已在列表$/);
  assert.equal(byClass(rows[0], "sb-action")[0].disabled, true, "listed journals cannot be added again");
  assert.equal(byClass(rows[1], "sb-action")[0].textContent, "添加");
  assert.equal(byClass(box, "sb-spinner")[0].hidden, true);

  // ↓ ↓ wraps to the first row, ↑ goes back; Enter adds the highlighted journal
  input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(input.getAttribute("aria-activedescendant"), "sp-journal-search-results-0");
  input.dispatch("keydown", { key: "ArrowDown" });
  input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(input.getAttribute("aria-activedescendant"), "sp-journal-search-results-0");
  input.dispatch("keydown", { key: "ArrowUp" });
  assert.equal(rows[1].getAttribute("aria-selected"), "true");
  const enter = input.dispatch("keydown", { key: "Enter" });
  assert.equal(enter.defaultPrevented, true);
  await flushPromises();
  await flushPromises();
  const list = byId(root, "sp-whitelist");
  assert.equal(list.value, "issn,title,category,impact_factor\n0016-7037,GCA,GEO,5.0\n0009-2541,Chemical Geology,GEOCHEMISTRY_AND_PETROLOGY,3.9");
  assert.equal(saved.at(-1), list.value, "added rows are saved at once");
  assert.equal(byClass(rows[1], "sb-action")[0].textContent, "已在列表");
  assert.match(byClass(box, "sb-status")[0].textContent, /已添加 Chemical Geology（1 行 ISSN，近似 IF 3\.9）/);

  // Esc closes the list, a second Esc clears the input
  input.dispatch("keydown", { key: "Escape" });
  assert.equal(results.hidden, true);
  assert.equal(input.getAttribute("aria-expanded"), "false");
  input.dispatch("keydown", { key: "Escape" });
  assert.equal(input.value, "");

  // Failures are shown; Enter searches at once
  control.searchJournals = async () => {
    throw new Error("OpenAlex 免费额度已用完（HTTP 429），请在「学术数据接口」中填写 API Key");
  };
  input.value = "lithos";
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  assert.match(byClass(box, "sb-status")[0].textContent, /^搜索失败：OpenAlex 免费额度已用完/);
  control.searchJournals = async () => [];
  input.value = "nothing";
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  assert.equal(byClass(box, "sb-status")[0].textContent, "OpenAlex 中没有找到匹配的期刊");
  // A cancelled (superseded) search says nothing
  control.searchJournals = async () => {
    const e = new Error("已取消");
    e.cancelled = true;
    throw e;
  };
  input.value = "cancelled";
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  assert.equal(byClass(box, "sb-status")[0].hidden, true);
});

test("search box: a highlight never outlives its results (Enter searches, never adds an unseen row)", async () => {
  const { page, root, clock, control, saved } = loadSettingsPage({ whitelist: "issn,title,category,impact_factor\n0016-7037,GCA,GEO,5.0" });
  control.searchJournals = async (query) =>
    query.startsWith("nature") ? [journal("Nature", ["0028-0836"]), journal("Nature Geoscience", ["1752-0894"])] : [journal("Chemical Geology", ["0009-2541"])];
  page.init();
  page.show({ section: "journals" });
  const input = byId(root, "sp-journal-search");
  const results = byId(root, "sp-journal-search-results");
  input.value = "nature";
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  input.dispatch("keydown", { key: "ArrowDown" });
  assert.equal(input.getAttribute("aria-activedescendant"), "sp-journal-search-results-0");

  // A new query: typing drops the highlight, and the new rows start unselected
  input.value = "chemical geology";
  input.dispatch("input");
  assert.equal(input.getAttribute("aria-activedescendant"), null);
  await clock.advance(400);
  await flushPromises();
  const rows = byClass(results, "sb-result");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].getAttribute("aria-selected"), "false");
  assert.equal(input.getAttribute("aria-activedescendant"), null);

  // Enter (while the list is open and nothing is highlighted) searches again; nothing is added
  const searches = control.searches.length;
  input.value = "chemical geology letters";
  input.dispatch("input");
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  assert.equal(control.searches.length, searches + 1, "searched at once");
  assert.equal(saved.length, 0, "no journal was added");

  // A hovered row is forgotten as soon as the query changes
  byClass(results, "sb-result")[0].dispatch("mousemove");
  assert.ok(input.getAttribute("aria-activedescendant"));
  input.value = "chemical geol";
  input.dispatch("input");
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  assert.equal(saved.length, 0);
});

test("an unreadable journals.csv is never replaced by one click on 添加 / 加入所选 / 补全", async () => {
  const { page, root, control, saved, api } = loadSettingsPage({ whitelist: "" });
  api.whitelistError = "无法读取 journals.csv（请以 UTF-8 编码保存）：not UTF-8 encoded";
  control.searchJournals = async () => [journal("Chemical Geology", ["0009-2541"])];
  control.proposals = { proposals: [{ title: "Lithos", issns: ["0024-4937"], category: "GEOLOGY", impactFactor: 2.1, libraryCount: 3, origin: "library", checked: true, verified: true }], warnings: [], missing: [] };
  page.init();
  page.show({ section: "journals" });
  const input = byId(root, "sp-journal-search");
  input.value = "chemical geology";
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  const add = byClass(byId(root, "sp-journal-search-results"), "sb-action")[0];
  assert.equal(add.disabled, true, "「添加」 is off while the file cannot be read");
  input.dispatch("keydown", { key: "ArrowDown" });
  input.dispatch("keydown", { key: "Enter" });
  await flushPromises();
  assert.equal(saved.length, 0);

  byId(root, "sp-journals-suggest").dispatch("click");
  await flushPromises();
  await flushPromises();
  byId(root, "sp-journals-add").dispatch("click");
  await flushPromises();
  assert.equal(saved.length, 0, "加入所选 does not write");
  assert.match(byClass(root, "sf-action-message")[0].textContent, /期刊列表文件无法读取/);

  byId(root, "sp-journals-fill").disabled = false;
  byId(root, "sp-journals-fill").dispatch("click");
  await flushPromises();
  assert.deepEqual(control.fills, [], "补全 does not run");
  assert.equal(saved.length, 0);
});

test("从文库生成: a review panel with defaults, 已在列表, 全选 and 加入所选", async () => {
  const { page, root, control, saved } = loadSettingsPage({ whitelist: "issn,title,category,impact_factor\n# mine\n0012-821X,EPSL,library:9," });
  control.proposals = {
    proposals: [
      { title: "Earth and Planetary Science Letters", libraryTitle: "EPSL", issns: ["0012-821X", "1385-013X"], category: "GEOPHYSICS", subfield: "Geophysics", impactFactor: 4.6, libraryCount: 9, origin: "library", checked: true, verified: true },
      { title: "Acta Petrologica Sinica", issns: ["1000-0569"], category: "GEOLOGY (CN)", subfield: "Geology", impactFactor: 1.2, libraryCount: 4, origin: "library", checked: true, verified: true },
      { title: "Lithos", issns: ["0024-4937"], category: "", impactFactor: null, libraryCount: 1, origin: "library", checked: false, verified: false },
      { title: "Journal of Petrology", issns: ["0022-3530"], category: "GEOCHEMISTRY_AND_PETROLOGY", subfield: "Geochemistry and Petrology", impactFactor: 3.2, libraryCount: 0, origin: "ai", checked: true, verified: true },
    ],
    warnings: ["大模型推荐失败：HTTP 401"],
    missing: ["Unknown Newsletter"],
  };
  page.init();
  page.show({ section: "journals" });
  byId(root, "sp-journals-suggest").dispatch("click");
  await flushPromises();
  await flushPromises();
  const panel = byId(root, "sp-journals-review");
  assert.equal(panel.hidden, false);
  assert.match(byClass(panel, "jr-counts")[0].textContent, /^文库期刊 3 种 · AI 推荐 1 种 · 1 种已在列表$/);
  assert.equal(byClass(panel, "jr-warning")[0].textContent, "大模型推荐失败：HTTP 401");
  const rows = byClass(panel, "jr-row");
  assert.equal(rows.length, 4);
  const checks = rows.map((r) => byClass(r, "jr-check")[0]);
  assert.deepEqual(checks.map((c) => [c.checked, c.disabled]), [[false, true], [true, false], [false, false], [true, false]], "listed ones locked; defaults from the proposals");
  assert.match(rows[0].textContent, /文库中写作：EPSL已在列表/);
  assert.match(rows[1].textContent, /Geology中文核心1\.24 篇$/);
  assert.match(rows[2].textContent, /OpenAlex 中未找到.*–.*–1 篇$/);
  assert.match(rows[3].textContent, /AI 推荐$/);
  assert.deepEqual(byClass(panel, "jr-group").map((g) => g.textContent), ["来自文库（3）", "AI 推荐 · 已通过 OpenAlex 核实（1）"]);
  assert.match(byClass(panel, "jr-note").map((n) => n.textContent).join(" "), /另有 1 种期刊没有找到 ISSN，无法加入：Unknown Newsletter/);
  const add = byId(root, "sp-journals-add");
  assert.equal(add.textContent, "加入所选（2）");
  checks[2].checked = true;
  checks[2].dispatch("change");
  assert.equal(add.textContent, "加入所选（3）");
  const [selectAll, selectNone] = byClass(panel, "jr-head-actions")[0].childNodes;
  selectNone.dispatch("click");
  assert.equal(add.textContent, "加入所选（0）");
  assert.equal(add.disabled, true);
  selectAll.dispatch("click");
  assert.equal(add.textContent, "加入所选（3）");
  assert.equal(checks[0].checked, false, "全选 skips listed journals");
  checks[3].checked = false;
  checks[3].dispatch("change");

  add.dispatch("click");
  await flushPromises();
  await flushPromises();
  const list = byId(root, "sp-whitelist");
  assert.deepEqual(list.value.split("\n"), [
    "issn,title,category,impact_factor",
    "# mine",
    "0012-821X,EPSL,library:9,",
    "1000-0569,Acta Petrologica Sinica,GEOLOGY (CN),1.2",
    "0024-4937,Lithos,,",
  ]);
  assert.equal(saved.at(-1), list.value);
  assert.equal(panel.hidden, true, "the panel closes after adding");
  assert.match(byClass(root, "sf-action-message")[0].textContent, /^已加入 2 种期刊（2 行 ISSN）。$/);
});

test("补全学科与影响因子 and 取消 of the OpenAlex actions", async () => {
  const text = "issn,title,category,impact_factor\n1674-7313,Science China Earth Sciences,library:14,\n0016-7037,GCA,GEO,5.0";
  const { page, root, control, saved, api } = loadSettingsPage({ whitelist: text });
  control.fillResult = (t) => ({ text: t.replace("library:14,", "GEOLOGY,4.8"), filled: 1, incomplete: 1, notFound: [] });
  page.init();
  page.show({ section: "journals" });
  const fill = byId(root, "sp-journals-fill");
  assert.equal(fill.disabled, false);
  assert.match(fill.getAttribute("title"), /^1 个期刊缺少/);
  fill.dispatch("click");
  await flushPromises();
  await flushPromises();
  assert.deepEqual(control.fills, [text]);
  const list = byId(root, "sp-whitelist");
  assert.match(list.value, /1674-7313,Science China Earth Sciences,GEOLOGY,4\.8/);
  assert.equal(saved.at(-1), list.value);
  assert.match(byClass(root, "sf-action-message")[0].textContent, /已补全 1 个期刊的学科或影响因子/);
  assert.equal(fill.disabled, true, "nothing left to fill");
  assert.equal(fill.getAttribute("title"), "所有期刊都已有学科和影响因子");

  // 从文库生成 can be cancelled while it runs
  let reject;
  page.show({ section: "journals" });
  api.suggestJournals = () => new Promise((_, r) => (reject = r));
  byId(root, "sp-journals-suggest").dispatch("click");
  await flushPromises();
  const cancel = byId(root, "sp-journals-cancel");
  assert.equal(cancel.textContent, "取消");
  cancel.dispatch("click");
  assert.equal(control.journalCancels, 1);
  reject(Object.assign(new Error("已取消"), { cancelled: true }));
  await flushPromises();
  assert.match(byClass(root, "sf-action-message")[0].textContent, /^已取消$/);
  assert.equal(list.disabled, false);
});

test("关注作者: search and 关注, removable rows, 编辑文本 and the missing-key notice", async () => {
  const { page, root, control, values, clock } = loadSettingsPage({
    prefs: { "followed.enabled": false, "followed.authors": "汪在聪, 0000-0002-3584-1673\nnobody\n# a comment" },
  });
  control.searchAuthors = async () => [
    { id: "A5000000001", name: "Zaicong Wang", orcid: "0000-0002-3584-1673", institution: "China University of Geosciences", country: "CN", worksCount: 120, citedByCount: 3000, topic: "Platinum-group elements" },
    { id: "A5023888391", name: "Cin-Ty Lee", orcid: null, institution: "Rice University", worksCount: 250, citedByCount: 18000, topic: null },
  ];
  page.init();
  page.show({ section: "followed" });
  const section = byId(root, "settings-followed");
  // No OpenAlex key: a notice under the search box links to 学术数据接口
  const hint = byClass(section, "sb-hint")[0];
  assert.equal(hint.hidden, false);
  assert.match(hint.textContent, /未配置 OpenAlex API Key/);
  assert.equal(walk(hint).find((n) => n.tagName === "A").getAttribute("href"), "#settings/apis");
  // The textarea lives under a collapsed 「编辑文本」
  const [area] = byPref(root, "followed.authors");
  const details = walk(section).find((n) => n.tagName === "DETAILS" && walk(n).includes(area));
  assert.match(details.textContent, /^编辑文本/);

  const input = byId(root, "sp-author-search");
  input.value = "wang";
  input.dispatch("input");
  await clock.advance(400);
  await flushPromises();
  const rows = byClass(byId(root, "sp-author-search-results"), "sb-result");
  assert.equal(rows.length, 2);
  assert.match(rows[0].textContent, /^Zaicong WangORCID0000-0002-3584-1673China University of Geosciences · 120 篇论文 · 被引 3,000 · Platinum-group elements已关注$/);
  assert.match(rows[1].textContent, /^Cin-Ty Lee无 ORCIDRice University · 250 篇论文 · 被引 18,000关注$/);
  const follow = byClass(rows[1], "sb-action")[0];
  assert.equal(follow.getAttribute("aria-label"), "关注：Cin-Ty Lee");
  follow.dispatch("click");
  await flushPromises();
  assert.equal(values.get("followed.authors"), "汪在聪, 0000-0002-3584-1673\nnobody\n# a comment\nCin-Ty Lee, A5023888391", "no ORCID: the OpenAlex id");
  assert.equal(follow.textContent, "已关注");
  assert.match(byClass(section, "sb-status")[0].textContent, /已关注 Cin-Ty Lee（「关注指定作者的新论文」尚未开启）/);

  // Rows: the parsed authors and the unparseable line, each removable
  const preview = byClass(root, "sf-preview")[0];
  assert.equal(byClass(preview, "sf-preview-head")[0].textContent, "已识别 2 位作者 · 1 行无法识别");
  const authorRows = byClass(preview, "sf-author");
  assert.equal(authorRows.length, 2);
  byClass(authorRows[0], "sf-remove")[0].dispatch("click");
  assert.equal(values.get("followed.authors"), "nobody\n# a comment\nCin-Ty Lee, A5023888391");
  assert.equal(byClass(rows[0], "sb-action")[0].textContent, "关注", "an unfollowed author can be followed again");
  byClass(byClass(preview, "sf-invalid")[0], "sf-remove")[0].dispatch("click");
  assert.equal(values.get("followed.authors"), "# a comment\nCin-Ty Lee, A5023888391", "comments stay");
  assert.equal(area.value, values.get("followed.authors"));
  assert.equal(byClass(preview, "sf-preview-head")[0].textContent, "已识别 1 位作者");

  // With a key the notice goes away
  const [key] = byPref(root, "apis.openalexKey");
  page.show({ section: "apis" });
  key.value = "oa-key";
  key.dispatch("change");
  assert.equal(hint.hidden, true);
});

test("顶刊: journals grouped by name with removable rows, search to add, the default list kept blank", async () => {
  const { page, root, control, values, clock } = loadSettingsPage({ prefs: { "apis.openalexKey": "oa-key" } });
  const auto = autoValues("local");
  const known = { "0028-0836": "Nature", "1476-4687": "Nature", "0036-8075": "Science", "1095-9203": "Science" };
  control.lookupNames = async (issns) => ({
    names: Object.fromEntries(issns.map((issn) => [issn, known[issn] ? { title: known[issn], issns: [issn] } : issn === "9999-9999" ? null : { title: `Journal ${issn}`, issns: [issn] }])),
    error: null,
  });
  control.searchJournals = async () => [{ title: "Geology", issns: ["0091-7613", "1943-2682"], impactFactor: 4.8, worksCount: 15000, publisher: "GSA", country: "US" }];
  page.init();
  // Names are only looked up while the section is shown
  page.show({ section: "general" });
  assert.deepEqual(control.nameLookups, []);
  page.show({ section: "flagship" });
  await flushPromises();
  await flushPromises();
  assert.equal(control.nameLookups.length, 1);
  const section = byId(root, "settings-flagship");
  assert.equal(byClass(section, "sb-hint")[0].hidden, true, "a key is set: no notice");
  const list = byClass(section, "fj-list")[0];
  let rows = byClass(list, "fj-row");
  assert.equal(rows[0].textContent, "Nature0028-08361476-4687");
  assert.equal(rows[1].textContent, "Science0036-80751095-9203");
  assert.equal(rows.length, auto.flagship.issns.length - 2);
  assert.match(byClass(section, "sf-list-head")[0].textContent, /^顶刊列表10 种期刊 · 12 个 ISSN · 默认$/);

  // 移除 Nature: both ISSNs go
  byClass(rows[0], "sf-remove")[0].dispatch("click");
  const stored = values.get("flagship.issns");
  assert.deepEqual(stored.split("\n"), auto.flagship.issns.filter((i) => !["0028-0836", "1476-4687"].includes(i)));
  rows = byClass(list, "fj-row");
  assert.equal(rows[0].textContent.startsWith("Science"), true);
  assert.match(byClass(section, "sf-list-head")[0].textContent, /9 种期刊 · 10 个 ISSN · 已自定义/);

  // Search and 添加
  const input = byId(root, "sp-flagship-search");
  input.value = "geology";
  input.dispatch("input");
  await clock.advance(400);
  await flushPromises();
  const result = byClass(byId(root, "sp-flagship-search-results"), "sb-result")[0];
  byClass(result, "sb-action")[0].dispatch("click");
  await flushPromises();
  assert.deepEqual(values.get("flagship.issns").split("\n").slice(-2), ["0091-7613", "1943-2682"]);
  assert.equal(byClass(result, "sb-action")[0].textContent, "已添加");
  rows = byClass(list, "fj-row");
  assert.equal(rows.at(-1).textContent, "Geology0091-76131943-2682", "the searched name is used, no lookup");

  // Back to the defaults stores a blank value; an unknown ISSN gets its own row; the last row cannot go
  const [area] = byPref(root, "flagship.issns");
  area.value = auto.flagship.issns.join("\n");
  area.dispatch("input");
  await clock.advance(600);
  assert.equal(values.get("flagship.issns"), "");
  area.value = "9999-9999";
  area.dispatch("input");
  await flushPromises();
  await flushPromises();
  rows = byClass(list, "fj-row");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].textContent, "未识别的期刊9999-9999");
  assert.equal(byClass(rows[0], "sf-remove")[0].disabled, true);
  assert.match(byClass(rows[0], "sf-remove")[0].getAttribute("title"), /至少保留一种期刊/);
});
