// Smoke tests of the built plugin in simulated Zotero environments: load
// build/addon/bootstrap.js into a VM context with stubbed Zotero/Firefox APIs
// and drive the full lifecycle (startup -> run -> add item -> shutdown), once
// as Zotero 10 (MenuManager, sandboxed preference scripts) and once as Zotero 7.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import { parseXml } from "../src/core/xml.js";
import { connectionLine, nextRunAt, normalizePage } from "../src/zotero/addon.js";
import { PREFS_0_2_0, embeddingHandler, mockHttp } from "./helpers.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// Values created inside the VM have foreign prototypes; compare them structurally
const plain = (x) => JSON.parse(JSON.stringify(x));
const addonDir = join(root, "build", "addon");
const PLUGIN_ID = "zotwatch@crazyhalfday.github.io";

test.before(() => {
  execFileSync(process.execPath, [join(root, "scripts", "build.mjs")], { stdio: "pipe" });
});

function fakeElement(doc, tag) {
  const el = {
    tagName: tag,
    id: "",
    attributes: {},
    listeners: {},
    children: [],
    disabled: false,
    setAttribute(k, v) {
      this.attributes[k] = v;
    },
    getAttribute(k) {
      return this.attributes[k];
    },
    addEventListener(type, fn) {
      (this.listeners[type] ||= []).push(fn);
    },
    removeEventListener(type, fn) {
      this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn);
    },
    dispatch(type) {
      for (const fn of this.listeners[type] || []) fn({ type, target: this });
    },
    append(...els) {
      for (const child of els) {
        this.children.push(child);
        child.parent = this;
        if (child.id) doc.elements.set(child.id, child);
      }
    },
    remove() {
      if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
      doc.elements.delete(this.id);
      doc.links.delete(this);
    },
  };
  return el;
}

function fakeWindow() {
  const doc = { elements: new Map(), links: new Set(), focused: true };
  doc.getElementById = (id) => doc.elements.get(id) || null;
  doc.createXULElement = (tag) => fakeElement(doc, tag);
  doc.hasFocus = () => doc.focused;
  doc.querySelector = (selector) => {
    const href = selector.match(/href="([^"]+)"/)?.[1];
    return [...doc.links].find((l) => l.attributes.href === href) || null;
  };
  const popup = fakeElement(doc, "menupopup");
  popup.id = "menu_ToolsPopup";
  doc.elements.set(popup.id, popup);
  const opened = [];
  const listeners = {};
  return {
    document: doc,
    ZoteroPane: { async selectItem(id) {
      this.selected = id;
    } },
    opened,
    MozXULElement: {
      insertFTLIfNeeded(href) {
        if (doc.querySelector(`link[href="${href}"]`)) return;
        const link = fakeElement(doc, "link");
        link.attributes = { rel: "localization", href };
        doc.links.add(link);
      },
    },
    openDialog(url, name, features) {
      const routes = [];
      const dialog = { url, name, features, closed: false, routes, focus() {}, close() { this.closed = true; }, ZotWatchMain: { navigate: (route) => routes.push(route) } };
      opened.push(dialog);
      return dialog;
    },
    addEventListener(type, fn) {
      (listeners[type] ||= []).push(fn);
    },
    removeEventListener(type, fn) {
      listeners[type] = (listeners[type] || []).filter((f) => f !== fn);
    },
    fire(type) {
      const fns = listeners[type] || [];
      listeners[type] = [];
      for (const fn of fns) fn({ type });
    },
    focus() {},
  };
}

function createEnvironment({ dataDir, items, http, zoteroVersion }) {
  const prefs = new Map();
  const prefDefaults = new Map();
  const savedItems = [];
  const collections = [];
  const timers = new Set();
  const mainWindow = fakeWindow();
  const registeredPanes = [];
  const menus = { registered: [], unregistered: [] };
  let nextId = 100;

  class FakeItem {
    constructor(type) {
      this.itemType = type;
      this.fields = {};
      this.creators = [];
      this.collections = [];
      this.tags = [];
      this.libraryID = null;
    }
    setField(field, value) {
      if (field === "ISSN" && this.itemType !== "journalArticle") throw new Error("invalid field");
      this.fields[field] = value;
    }
    getField(field) {
      return this.fields[field] || "";
    }
    setCreators(creators) {
      this.creators = creators;
    }
    addToCollection(id) {
      this.collections.push(id);
    }
    addTag(tag) {
      this.tags.push(tag);
    }
    setNote(html) {
      this.note = html;
    }
    async saveTx() {
      this.id ||= nextId++;
      this.key ||= `KEY${this.id}`;
      if (!savedItems.includes(this)) savedItems.push(this);
      return this.id;
    }
  }

  class FakeCollection {
    constructor({ name, libraryID }) {
      this.name = name;
      this.libraryID = libraryID;
    }
    async saveTx() {
      await new Promise((r) => setTimeout(r, 5));
      this.id = nextId++;
      collections.push(this);
    }
  }

  class CancelledException extends Error {}
  class SecurityException extends Error {}
  const libraries = {
    1: { libraryType: "user", editable: true, name: "我的文库", waited: 0, async waitForDataLoad(type) {
      assert.equal(type, "item");
      this.waited++;
    } },
    2: { libraryType: "feed", editable: false, name: "Feed", async waitForDataLoad() {} },
  };

  const Zotero = {
    version: { 7: "7.0.25", 8: "8.0.4", 10: "10.0.2" }[zoteroVersion],
    initializationPromise: Promise.resolve(),
    isMac: false,
    debug() {},
    logError(e) {
      Zotero.errors.push(e);
    },
    warn() {},
    errors: [],
    DataDirectory: { dir: dataDir },
    Prefs: {
      get: (k) => (prefs.has(k) ? prefs.get(k) : prefDefaults.get(k)),
      set: (k, v) => prefs.set(k, v),
    },
    PreferencePanes: {
      async register(opts) {
        registeredPanes.push(opts);
        return opts.id;
      },
    },
    getMainWindows: () => [mainWindow],
    getMainWindow: () => mainWindow,
    HTTP: {
      CancelledException,
      SecurityException,
      checkSecurity() {},
      requests: [],
      async request(method, url, options) {
        Zotero.HTTP.requests.push({ method, url, options });
        let cancel;
        const cancelled = new Promise((_, reject) => {
          cancel = () => reject(new CancelledException());
        });
        options.cancellerReceiver?.(cancel);
        const res = await Promise.race([
          http.request({ method, url, headers: options.headers, body: options.body, responseType: options.responseType }),
          cancelled,
        ]);
        const data = options.responseType === "json" && typeof res.data === "string" ? null : res.data;
        return { status: res.status, response: data, channel: null, getResponseHeader: () => null };
      },
    },
    Libraries: { userLibraryID: 1, get: (id) => libraries[id] || false },
    Items: {
      async getAll(libraryID, onlyTopLevel) {
        assert.equal(libraryID, 1);
        assert.equal(onlyTopLevel, true);
        return items.map((data) => ({
          ...data,
          // Items without tags lack getTags(): the library adapter must tolerate it
          ...(data.tags ? { getTags: () => data.tags.map((tag) => ({ tag, type: 0 })) } : {}),
          isRegularItem: () => data.itemType !== "note",
          getField(name, unformatted, baseMapped) {
            if (name === "publicationTitle" || name === "title") assert.equal(baseMapped, true);
            return { title: data.title, abstractNote: data.abstract, DOI: data.doi, ISSN: data.issn, publicationTitle: data.venue, extra: data.extra || "" }[name] ?? "";
          },
        }));
      },
      async loadDataTypes() {},
    },
    ItemTypes: { getID: (t) => (t === "preprint" ? 31 : 1) },
    Item: FakeItem,
    Collection: FakeCollection,
    Collections: {
      recent: [],
      getByLibrary: () => collections,
      addToRecent(c) {
        this.recent.push(c.id);
      },
    },
    Translate: {
      Search: class {
        setIdentifier(id) {
          this.identifier = id;
        }
        async getTranslators() {
          return [];
        }
      },
    },
    ProgressWindow: class {
      constructor() {
        Zotero.notifications.push(this);
      }
      changeHeadline(t) {
        this.headline = t;
      }
      addDescription(t) {
        this.text = t;
      }
      show() {}
      startCloseTimer(ms, requireMouseOver) {
        this.closeTimer = [ms, requireMouseOver];
      }
    },
    notifications: [],
    UIProperties: { registerRoot() {} },
    Utilities: { Internal: { openPreferences(id) {
      Zotero.openedPrefs = id;
    } } },
    launchURL(url) {
      Zotero.launched = url;
    },
    File: {
      getResourceAsync: async (url) => readFileSync(join(addonDir, url.replace("chrome://zotwatch/", "")), "utf8"),
      revealed: [],
      async reveal(path) {
        assert.ok(existsSync(path), "only existing paths can be revealed");
        Zotero.File.revealed.push(path);
      },
    },
  };
  if (zoteroVersion >= 8) {
    Zotero.MenuManager = {
      registerMenu(options) {
        menus.registered.push(options);
        return "menu-key-1";
      },
      unregisterMenu(key) {
        menus.unregistered.push(key);
        return true;
      },
    };
  }

  const IOUtils = {
    makeDirectory: async (p) => mkdirSync(p, { recursive: true }),
    exists: async (p) => existsSync(p),
    readJSON: async (p) => JSON.parse(readFileSync(p, "utf8")),
    writeJSON: async (p, obj) => writeFileSync(p, JSON.stringify(obj)),
    read: async (p) => new Uint8Array(readFileSync(p)),
    write: async (p, bytes) => writeFileSync(p, bytes),
    readUTF8: async (p) => readFileSync(p, "utf8"),
    writeUTF8: async (p, text) => writeFileSync(p, text, "utf8"),
    remove: async (p) => rmSync(p, { recursive: true, force: true }),
  };

  const chromeRegistrations = [];
  const context = {
    Zotero,
    IOUtils,
    PathUtils: { join: (...parts) => join(...parts) },
    Components: {
      classes: {
        "@mozilla.org/addons/addon-manager-startup;1": {
          getService: () => ({
            registerChrome(manifestURI, entries) {
              chromeRegistrations.push({ manifestURI, entries, destroyed: false });
              return { destruct: () => (chromeRegistrations.at(-1).destroyed = true) };
            },
          }),
        },
      },
      interfaces: { amIAddonManagerStartup: {} },
    },
    Services: {
      io: { newURI: (spec) => ({ spec }) },
      locale: { appLocaleAsBCP47: "zh-CN" },
      wm: {
        getMostRecentWindow: (type) => {
          assert.equal(type, "zotwatch:main");
          return mainWindow.opened.findLast((d) => !d.closed) || null;
        },
      },
      ww: {
        openWindow: () => {
          throw new Error("main window exists; ww.openWindow should not be used");
        },
      },
      scriptloader: {
        loadSubScriptWithOptions(url, { target }) {
          assert.ok(url.startsWith("chrome://zotwatch/content/"), url);
          const code = readFileSync(join(addonDir, url.replace("chrome://zotwatch/", "")), "utf8");
          assert.ok(target, "scripts are loaded into the plugin scope");
          vm.runInContext(code, context, { filename: url });
        },
      },
    },
    setTimeout: (fn, ms) => {
      const t = setTimeout(() => {
        timers.delete(t);
        fn();
      }, ms);
      timers.add(t);
      return t;
    },
    clearTimeout: (t) => {
      clearTimeout(t);
      timers.delete(t);
    },
    setInterval: (fn, ms) => {
      const t = setInterval(fn, ms);
      timers.add(t);
      return t;
    },
    clearInterval: (t) => {
      clearInterval(t);
      timers.delete(t);
    },
    APP_SHUTDOWN: 2,
    ADDON_DISABLE: 4,
    console,
  };
  vm.createContext(context);

  // Default prefs, loaded the way Zotero does (prefs.js with a `pref` function)
  vm.runInNewContext(readFileSync(join(addonDir, "prefs.js"), "utf8"), { pref: (k, v) => prefDefaults.set(k, v) });

  const env = { context, Zotero, prefs, prefDefaults, mainWindow, registeredPanes, chromeRegistrations, savedItems, collections, timers, menus, libraries };
  env.cleanup = () => {
    for (const timer of timers) {
      clearTimeout(timer);
      clearInterval(timer);
    }
  };
  return env;
}

function libraryItems() {
  return [
    ...Array.from({ length: 12 }, (_, i) => ({
      key: `C${i}`,
      itemType: "journalArticle",
      title: `Copper isotope fractionation in porphyry systems ${i}`,
      abstract: `Copper isotopes and sulfide saturation in arc magmas, study ${i}`,
      doi: i === 0 ? "" : `10.4444/lib.${i}`,
      extra: i === 0 ? "Original DOI: 10.9999/other\nDOI: 10.4444/lib.0" : "",
      issn: "0016-7037",
      venue: "Geochimica et Cosmochimica Acta",
      dateAdded: "2024-05-01 10:00:00",
      tags: ["Copper isotopes", "Porphyry", "⭐⭐⭐", "/unread", "ZotWatch"],
    })),
    ...Array.from({ length: 4 }, (_, i) => ({
      key: `M${i}`,
      itemType: "journalArticle",
      title: `Mantle melting under ridges ${i}`,
      abstract: `Mantle peridotite melting and basalt genesis ${i}`,
      doi: `10.4444/mantle.${i}`,
      issn: "0016-7037",
      venue: "Geochimica et Cosmochimica Acta",
      dateAdded: "2024-05-01 10:00:00",
    })),
    { key: "N1", itemType: "note", title: "a note" },
  ];
}

function recentCrossref() {
  const created = new Date(Date.now() - 2 * 86400000).toISOString();
  const mk = (n, title, abstract) => ({
    DOI: `10.5555/new.${n}`,
    title: [title],
    author: [{ given: "Li", family: "Wang" }],
    abstract: `<jats:p>${abstract}</jats:p>`,
    "container-title": ["Geochimica et Cosmochimica Acta"],
    created: { "date-time": created },
    URL: `https://doi.org/10.5555/new.${n}`,
    type: "journal-article",
    ISSN: ["0016-7037"],
  });
  return [
    mk(1, "Copper isotope fractionation during sulfide saturation in arc magmas", "Copper isotopes sulfide saturation arc magmas porphyry"),
    mk(2, "Mantle peridotite melting revisited", "Mantle peridotite melting basalt"),
    mk(3, "Graphene battery electrodes", "Graphene electrodes for lithium batteries and supercapacitors"),
    // The library holds this one through the "DOI:" line of an item's Extra field
    mk(4, "An already collected copper paper", "Copper isotopes sulfide"),
  ].map((w, i) => (i === 3 ? { ...w, DOI: "10.4444/lib.0" } : w));
}

function standardHttp(counter, { gate } = {}) {
  return mockHttp([
    ["https://api.crossref.org/works", async () => {
      if (gate) await gate;
      return { message: { "total-results": 4, items: recentCrossref() } };
    }],
    ["https://api.crossref.org/journals", () => ({ message: { items: [] } })],
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)],
  ]);
}

async function boot(env) {
  vm.runInContext(readFileSync(join(addonDir, "bootstrap.js"), "utf8"), env.context, { filename: "bootstrap.js" });
  await vm.runInContext(`startup({ id: "${PLUGIN_ID}", version: "0.1.0", rootURI: "jar:file:///zotwatch.xpi!/" })`, env.context);
  return env.Zotero.ZotWatch;
}

function useApiEmbeddings(env) {
  env.prefs.set("extensions.zotwatch.embedding.provider", "dashscope");
  env.prefs.set("extensions.zotwatch.embedding.apiKey", "sk-test");
}

test("Zotero 10: lifecycle with MenuManager, run, add, cancel and shutdown", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const counter = {};
  const items = libraryItems();
  const env = createEnvironment({ dataDir, items, http: standardHttp(counter), zoteroVersion: 10 });
  t.after(env.cleanup);
  const { Zotero, mainWindow, menus } = env;

  // A whitelist stored in the pref by an earlier version is migrated to a file
  env.prefs.set("extensions.zotwatch.journals.whitelist", "issn,title,category,impact_factor\n0016-7037,GCA,GEO,5.0");

  const addon = await boot(env);
  assert.deepEqual(plain(env.chromeRegistrations[0].entries), [["content", "zotwatch", "jar:file:///zotwatch.xpi!/content/"]]);
  assert.ok(addon, "Zotero.ZotWatch is exposed");
  assert.equal(env.registeredPanes.length, 1);
  assert.equal(env.registeredPanes[0].src, "chrome://zotwatch/content/preferences.xhtml");
  assert.equal(env.registeredPanes[0].pluginID, PLUGIN_ID);
  assert.match(env.registeredPanes[0].helpURL, /^https:\/\//);
  assert.equal(env.timers.size, 2, "scheduler armed");
  assert.equal(addon.isRunDue(), true, "first run is due");

  assert.equal(readFileSync(join(dataDir, "zotwatch", "journals.csv"), "utf8").split("\n")[1], "0016-7037,GCA,GEO,5.0");
  assert.equal(env.prefs.get("extensions.zotwatch.journals.whitelist"), "");
  assert.match(addon.getJournalWhitelist(), /0016-7037/);

  // Menus go through MenuManager; nothing is injected into the DOM
  assert.equal(menus.registered.length, 1);
  const menuOptions = menus.registered[0];
  assert.equal(menuOptions.target, "main/menubar/tools");
  assert.equal(menuOptions.pluginID, PLUGIN_ID);
  assert.deepEqual(plain(menuOptions.menus.map((m) => m.l10nID || m.menuType)), ["separator", "zotwatch-menu-open", "zotwatch-menu-run"]);
  assert.equal(mainWindow.document.getElementById("menu_ToolsPopup").children.length, 0);
  assert.ok(mainWindow.document.querySelector('link[href="zotwatch-mainWindow.ftl"]'), "Fluent file inserted");

  // Opening twice focuses the existing window, which switches to the requested page
  menuOptions.menus[1].onCommand();
  menuOptions.menus[1].onCommand();
  assert.equal(mainWindow.opened.length, 1);
  // No route in the URL (the window's persisted size and position are keyed by
  // the document URI): the new window starts on lastPage
  assert.equal(mainWindow.opened[0].url, "chrome://zotwatch/content/main.xhtml");
  assert.equal(addon.lastPage, "inbox");
  assert.equal(mainWindow.opened[0].name, "zotwatch-main");
  assert.doesNotMatch(mainWindow.opened[0].features, /width=/, "size comes from persisted attributes");
  assert.deepEqual(plain(mainWindow.opened[0].routes), ["inbox"]);
  addon.openWindow("settings/ai");
  addon.openWindow("settings/<script>");
  assert.deepEqual(plain(mainWindow.opened[0].routes), ["inbox", "settings/ai", "settings"]);
  assert.equal(mainWindow.opened.length, 1);
  // Without a page: the page last shown in this session (reported by the window)
  addon.lastPage = "settings/journals";
  addon.openWindow();
  assert.equal(mainWindow.opened[0].routes.at(-1), "settings/journals");

  // --- run
  useApiEmbeddings(env);
  const events = [];
  addon.addListener((e) => events.push(e.type));
  const result = await addon.runNow();
  assert.equal(result.stats.libraryItems, 16, "notes are excluded");
  assert.equal(result.stats.fetched, 4);
  assert.equal(result.stats.afterDedupe, 3, "Extra-field DOI counts as in the library");
  assert.ok(result.stats.recommended >= 1);
  assert.ok(events.includes("start") && events.includes("progress") && events.includes("done"));
  assert.equal(env.libraries[1].waited > 0, true, "waits for library data");
  assert.ok(Zotero.HTTP.requests.every((r) => r.options.anon === true), "requests are anonymous");
  assert.ok(existsSync(join(dataDir, "zotwatch", "profile.bin")));

  // The menu label carries the number of pending recommendations
  const context = { args: null, enabled: null, setL10nArgs(a) {
    this.args = a;
  }, setEnabled(e) {
    this.enabled = e;
  } };
  menuOptions.menus[1].onShowing({}, context);
  assert.equal(JSON.parse(context.args).count, addon.pendingCount);
  assert.ok(addon.pendingCount >= 1);
  menuOptions.menus[2].onShowing({}, context);
  assert.equal(context.enabled, true);

  const view = await addon.getView();
  const top = view.works.sort((a, b) => b.score - a.score)[0];
  assert.equal(top.id, "doi:10.5555/new.1");
  assert.ok(!view.works.some((w) => w.id === "doi:10.5555/new.3"), "off-topic paper is gated");
  // The main window's view: run history, version and the schedule
  assert.equal(view.version, "0.1.0");
  assert.equal(view.history.length, 1);
  assert.equal(view.history[0].runId, result.runId);
  assert.equal(view.history[0].fetched, 4);
  assert.deepEqual(plain(view.history[0].newByLabel), plain(result.stats.newByLabel));
  assert.equal(view.lastError, null);
  assert.equal(view.schedule.autoRun, true);
  assert.equal(view.schedule.intervalHours, 24);
  assert.equal(view.schedule.lastRun, Number(env.prefs.get("extensions.zotwatch.schedule.lastRun")));
  assert.equal(view.schedule.nextRunAt, view.schedule.lastRun + 24 * 3600000);
  assert.equal(addon.isRunDue(), false);
  assert.equal(addon.isRunDue(view.schedule.nextRunAt), true, "isRunDue and nextRunAt agree");

  // The dashboard
  const dash = await addon.getDashboard();
  assert.equal(dash.version, "0.1.0");
  assert.deepEqual(plain(dash.library), { name: "我的文库", items: 16, withAbstract: 16 });
  assert.deepEqual(plain(dash.embedding), { provider: "dashscope", model: "text-embedding-v4", label: "阿里云 DashScope · text-embedding-v4", local: false });
  assert.deepEqual(plain(dash.llm), { enabled: false, configured: false, model: "deepseek/deepseek-chat", topicLabels: true });
  assert.equal(dash.pending.total, view.works.filter((w) => w.status === "new").length);
  assert.equal(dash.pending.personal + dash.pending.flagship + dash.pending.followed, dash.pending.total);
  assert.equal(dash.added, 0);
  // Added works stay at least 60 days (the 已加入 tile); the funnel labels follow the filters
  assert.equal(dash.addedDays, 60);
  assert.deepEqual(plain(dash.scoring), { recentDays: 7, requireAbstract: true });
  assert.equal(dash.lastRun.runId, result.runId);
  assert.equal(dash.history.length, 1);
  assert.deepEqual(plain(dash.topics), []);
  assert.equal(dash.topicCount, 0);
  assert.deepEqual(plain(dash.followed), { enabled: false, daysBack: 60, authors: [], invalid: [] });
  assert.deepEqual(plain(dash.flagship), { enabled: false, issnCount: 12, stats: null });
  assert.deepEqual(plain(dash.journals), { count: 1, withImpactFactor: 1, autoFromLibrary: true, error: null });
  assert.deepEqual(plain(dash.sources.map((s) => s.key)), ["crossref", "flagship", "arxiv", "eartharxiv", "followed", "abstracts", "embedding", "llm"]);
  const crossrefSource = dash.sources.find((s) => s.key === "crossref");
  // ISSNs of the personal query (a journal often has two), not journals
  assert.deepEqual(plain(crossrefSource), { key: "crossref", name: "Crossref", enabled: true, count: 4, detail: "1 个 ISSN", error: null, ran: true });
  assert.equal(dash.sources.find((s) => s.key === "llm").detail, "未启用");
  assert.equal(dash.dataDir, join(dataDir, "zotwatch"));
  assert.equal(addon.getDataDir(), dash.dataDir);
  assert.deepEqual(plain(dash.schedule), plain(view.schedule));
  // The tiles' numbers: must-read among the pending works, items with an abstract
  assert.equal(dash.pending.mustRead, view.works.filter((w) => w.status === "new" && w.label === "must_read").length);
  assert.ok(dash.pending.mustRead > 0);
  items.push({ key: "NOABS", itemType: "journalArticle", title: "Untitled note-like item", abstract: "", doi: "10.4444/noabs", issn: "0016-7037", venue: "GCA", dateAdded: "2024-05-01 10:00:00" });
  assert.deepEqual(plain((await addon.getDashboard()).library), { name: "我的文库", items: 17, withAbstract: 16 });
  items.pop();
  // A source enabled after the last run has not run yet (未运行, not 正常)
  env.prefs.set("extensions.zotwatch.sources.arxiv.enabled", true);
  const arxivRow = (await addon.getDashboard()).sources.find((s) => s.key === "arxiv");
  assert.deepEqual(plain([arxivRow.enabled, arxivRow.ran, arxivRow.count]), [true, false, null]);
  env.prefs.set("extensions.zotwatch.sources.arxiv.enabled", false);

  // Helpers for the settings page
  assert.deepEqual(plain(addon.summarizeWhitelist("issn,title,category,impact_factor\n0016-7037,GCA,GEO,5.0\nbad line\n# note\n1234-5679,X")), {
    count: 2,
    withImpactFactor: 1,
    invalidLines: ["bad line"],
  });
  assert.deepEqual(plain(addon.parseFollowedAuthors("Li Wang, 0000-0002-1825-0097\nnobody")), {
    authors: [{ name: "Li Wang", key: "orcid:0000-0002-1825-0097", orcid: "0000-0002-1825-0097" }],
    invalid: ["nobody"],
  });
  const auto = addon.getAutoValues();
  assert.equal(auto.provider, "dashscope");
  assert.equal(auto.minSimilarity, 0.2);
  assert.equal(auto.flagshipMinScore, 0.3);
  assert.equal(auto.flagshipGrayLow, 0.22);
  assert.equal(auto.flagship.issns.length, 12);
  assert.equal(auto.flagship.negativeAnchors.length, 6);
  assert.match(auto.flagship.llmBoundary, /^ONLY solid-earth/);
  assert.equal(await addon.revealDataDir(), true);
  assert.deepEqual(plain(Zotero.File.revealed), [join(dataDir, "zotwatch")]);

  // --- two quick adds create a single collection
  const others = view.works.filter((w) => w.id !== top.id && w.status === "new");
  const [itemID] = await Promise.all([addon.addToLibrary(top.id), ...others.map((w) => addon.addToLibrary(w.id))]);
  assert.equal(env.collections.length, 1);
  assert.equal(env.collections[0].name, "ZotWatch 推荐");
  assert.ok(Zotero.Collections.recent.includes(env.collections[0].id));
  const item = env.savedItems.find((i) => i.id === itemID);
  assert.equal(item.itemType, "journalArticle");
  assert.equal(item.fields.title, top.title);
  assert.equal(item.fields.DOI, "10.5555/new.1");
  assert.equal(item.fields.publicationTitle, "Geochimica et Cosmochimica Acta");
  assert.deepEqual(plain(item.creators), [{ creatorType: "author", firstName: "Li", lastName: "Wang" }]);
  assert.deepEqual(plain(item.tags), ["ZotWatch"]);
  assert.deepEqual(plain(item.collections), [env.collections[0].id]);
  assert.equal(addon.pendingCount, 0);
  // Added works leave the pending counts; 已加入 counts them
  const handled = (await addon.getView()).works;
  const dash2 = await addon.getDashboard();
  assert.ok(dash2.added > 0);
  assert.equal(dash2.added, handled.filter((w) => w.status === "added").length);
  assert.equal(dash2.pending.total, handled.filter((w) => w.status === "new").length);
  assert.equal(dash2.pending.personal + dash2.pending.flagship + dash2.pending.followed, dash2.pending.total);
  assert.equal(dash2.pending.mustRead, 0);

  // --- bulk status change, status text, journals, preferences, URLs
  await addon.setStatus([top.id, others[0]?.id].filter(Boolean), "dismissed");
  assert.equal((await addon.getView()).works.find((w) => w.id === top.id).status, "dismissed");

  // --- a run that fails at the embedding service is shown on that source's row
  env.prefs.set("extensions.zotwatch.embedding.apiKey", "");
  await assert.rejects(addon.runNow(), /API Key/);
  assert.equal(addon.lastError.stage, "profile");
  const failedDash = await addon.getDashboard();
  assert.match(failedDash.sources.find((s) => s.key === "embedding").error, /API Key/);
  assert.equal(failedDash.sources.find((s) => s.key === "crossref").error, null);
  assert.equal(Zotero.errors.length, 1, "the failure is logged");
  Zotero.errors.length = 0;
  env.prefs.set("extensions.zotwatch.embedding.apiKey", "sk-test");
  assert.match(await addon.getStatus(), /我的文库，16 个条目/);
  assert.match(await addon.loadPresetJournals(), /^issn,title,category,impact_factor/);
  assert.match(await addon.suggestJournals(), /0016-7037,GCA,GEO,5/);
  await addon.setJournalWhitelist("issn,title\n1234-5678,X");
  assert.equal(readFileSync(join(dataDir, "zotwatch", "journals.csv"), "utf8"), "issn,title\n1234-5678,X");
  addon.openPreferences();
  assert.equal(Zotero.openedPrefs, "zotwatch-prefpane");
  addon.openUrl("javascript:alert(1)");
  assert.equal(Zotero.launched, undefined, "only http(s) URLs are opened");
  await addon.showItem(itemID);
  assert.equal(mainWindow.ZoteroPane.selected, itemID);

  // --- a feed library is never used for the profile
  env.prefs.set("extensions.zotwatch.profile.libraryID", "2");
  assert.match(await addon.getStatus(), /我的文库/);
  env.prefs.set("extensions.zotwatch.profile.libraryID", "");

  // --- cancelling aborts in-flight requests, but not requests made for the UI
  let release;
  const gate = new Promise((r) => (release = r));
  let slow = standardHttp(counter, { gate });
  const inFlight = new Set();
  let hangCrossref = false;
  Zotero.HTTP.request = ((original) => async (method, url, options) => {
    if (url.startsWith("https://api.crossref.org/works")) {
      let cancel;
      const cancelled = new Promise((_, reject) => (cancel = () => reject(new Zotero.HTTP.CancelledException())));
      // A hung request ignores cancellation, like code that never yields
      if (!hangCrossref) options.cancellerReceiver(cancel);
      inFlight.add(url);
      try {
        const res = await Promise.race([hangCrossref ? new Promise(() => {}) : slow.request({ method, url }), cancelled]);
        return { status: res.status, response: res.data, channel: null, getResponseHeader: () => null };
      } finally {
        inFlight.delete(url);
      }
    }
    return original(method, url, options);
  })(Zotero.HTTP.request);
  const waitForCrossref = async () => {
    while (!inFlight.size) await new Promise((r) => setTimeout(r, 5));
  };

  env.prefs.set("extensions.zotwatch.schedule.lastRun", "0");
  const running = addon.runNow();
  await waitForCrossref();
  addon.cancelRun();
  assert.equal(await running, null);
  assert.ok(events.includes("cancelled"));
  assert.equal(inFlight.size, 0, "the gated request was aborted");
  release();
  const status = await addon.testConnections();
  assert.match(status, /✅ Crossref/, "UI requests still work after a run was cancelled");
  assert.equal(status, ["✅ 向量化（dashscope）：正常，维度 8", "ℹ️ 大模型：未启用", "✅ Crossref：可访问"].join("\n"));
  assert.deepEqual(plain(await addon.checkConnections(["crossref", "llm"])), [
    { key: "llm", name: "大模型", target: null, ok: null, detail: "未启用" },
    { key: "crossref", name: "Crossref", target: null, ok: true, detail: "可访问" },
  ]);
  // A cancelled run records no history entry
  assert.equal((await addon.getView()).history.length, 1);

  // --- shutdown cancels a run and waits for it before tearing anything down
  slow = standardHttp(counter, { gate: new Promise(() => {}) });
  const settledWith = [];
  const pending = addon.runNow();
  pending.then(() => settledWith.push({ addon: !!Zotero.ZotWatch, chrome: !env.chromeRegistrations[0].destroyed }));
  await waitForCrossref();
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
  assert.equal(await pending, null);
  assert.deepEqual(settledWith, [{ addon: true, chrome: true }], "run settled before teardown");
  assert.deepEqual(plain(menus.unregistered), ["menu-key-1"]);
  assert.equal(mainWindow.document.querySelector('link[href="zotwatch-mainWindow.ftl"]'), null);
  assert.equal(Zotero.ZotWatch, undefined);
  assert.equal(env.timers.size, 0);
  assert.equal(env.chromeRegistrations[0].destroyed, true);
  assert.equal(mainWindow.opened[0].closed, true);
  assert.deepEqual(Zotero.errors.map(String), []);

  // --- a run that ignores cancellation does not block shutdown beyond the grace period
  const env2 = createEnvironment({ dataDir: mkdtempSync(join(tmpdir(), "zotwatch-")), items: libraryItems(), http: standardHttp(counter), zoteroVersion: 10 });
  t.after(env2.cleanup);
  const addon2 = await boot(env2);
  addon2.shutdownGraceMs = 50;
  env2.Zotero.HTTP.request = async () => new Promise(() => {});
  addon2.runNow().catch(() => {});
  await new Promise((r) => setTimeout(r, 20));
  const started = Date.now();
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env2.context);
  assert.ok(Date.now() - started < 1000, "teardown after the grace period");
  assert.equal(env2.Zotero.ZotWatch, undefined);
});

test("Zotero 7: DOM menus and a deferred sticky notification", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const env = createEnvironment({ dataDir, items: libraryItems(), http: standardHttp({}), zoteroVersion: 7 });
  t.after(env.cleanup);
  const { Zotero, mainWindow } = env;

  const addon = await boot(env);
  const menu = mainWindow.document.getElementById("menu_ToolsPopup");
  assert.deepEqual(menu.children.map((c) => c.id), ["zotwatch-menu-separator", "zotwatch-menu-open", "zotwatch-menu-run"]);
  assert.equal(menu.children[1].attributes.label, "ZotWatch 文献推荐…");
  assert.equal(menu.children[2].attributes.label, "ZotWatch：立即检查新文献");
  assert.equal(mainWindow.document.querySelector('link[href="zotwatch-mainWindow.ftl"]'), null, "no Fluent file in the main window");

  // A second main window load must not duplicate menu items
  vm.runInContext("onMainWindowLoad({ window: __win })", Object.assign(env.context, { __win: mainWindow }));
  assert.equal(menu.children.length, 3);

  // A failed run is reported to the main window (banner, sidebar) until the next run starts
  env.prefs.set("extensions.zotwatch.embedding.provider", "dashscope");
  const failures = [];
  addon.addListener((e) => e.type === "error" && failures.push(e.message));
  await assert.rejects(addon.runNow(), /API Key/);
  assert.equal(failures.length, 1);
  const failedView = await addon.getView();
  assert.match(failedView.lastError.message, /API Key/);
  assert.ok(failedView.lastError.at > 0);
  assert.equal(failedView.history.length, 0, "a failed run records no history");
  assert.equal(failedView.schedule.nextRunAt, failedView.schedule.lastAttempt + 2 * 3600000, "retried 2 hours after the failure");
  Zotero.errors.length = 0;

  // Background run while Zotero is in the background: the notice waits for focus
  useApiEmbeddings(env);
  mainWindow.document.focused = false;
  const result = await addon.runNow({ background: true });
  assert.ok(result.stats.newRecommendations > 0);
  assert.equal((await addon.getView()).lastError, null, "cleared by the next run");
  assert.equal(Zotero.notifications.length, 0, "deferred until Zotero is focused");
  mainWindow.fire("activate");
  assert.equal(Zotero.notifications.length, 1);
  assert.match(Zotero.notifications[0].text, /新推荐/);
  assert.deepEqual(plain(Zotero.notifications[0].closeTimer), [20000, true], "stays until clicked");

  // The menu label shows the pending count when the menu opens
  menu.dispatch("popupshowing");
  assert.equal(menu.children[1].attributes.label, `ZotWatch 文献推荐（${addon.pendingCount} 篇待查看）…`);

  // Notices are plain text even when an error message contains markup
  addon.notify("ZotWatch", '<a href=x>proxy</a> failed');
  assert.doesNotMatch(Zotero.notifications.at(-1).text, /</);

  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
  assert.equal(menu.children.length, 0);
  assert.deepEqual(Zotero.errors.map(String), []);
});

test("Zotero 8: MenuManager is skipped so the main window keeps its locale", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const env = createEnvironment({ dataDir, items: libraryItems(), http: standardHttp({}), zoteroVersion: 8 });
  t.after(env.cleanup);
  await boot(env);
  assert.equal(env.menus.registered.length, 0);
  const menu = env.mainWindow.document.getElementById("menu_ToolsPopup");
  assert.equal(menu.children.length, 3);
  assert.equal(env.mainWindow.document.querySelector('link[href="zotwatch-mainWindow.ftl"]'), null);
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
  assert.equal(menu.children.length, 0);
});

test("Zotero 10: followed authors reach the inbox, the notice, the status and the library", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const published = new Date(Date.now() - 3 * 86400000).toISOString().slice(0, 10);
  const followedWork = {
    id: "https://openalex.org/W77",
    doi: "https://doi.org/10.31223/X5ZZZ",
    display_name: "Melt inclusions record slab-derived <i>sulfur</i>",
    publication_date: published,
    type: "preprint",
    is_paratext: false,
    authorships: [
      { author: { id: "https://openalex.org/A1", display_name: "Li Wang", orcid: "https://orcid.org/0000-0002-1825-0097" } },
      { author: null },
    ],
    primary_location: { landing_page_url: "https://eartharxiv.org/x", source: { display_name: "EarthArXiv", issn: null } },
    abstract_inverted_index: { Sulfur: [0], isotopes: [1] },
    cited_by_count: null,
  };
  const openalexQueries = [];
  const standard = standardHttp({});
  let openalexDown = false;
  const http = {
    async request(req) {
      if (req.url.startsWith("https://api.openalex.org/works")) {
        openalexQueries.push(decodeURIComponent(req.url));
        if (openalexDown) return { status: 403, data: { error: "Forbidden" } };
        return { status: 200, data: { meta: { next_cursor: null }, results: [followedWork] } };
      }
      // arXiv answers with an empty feed (counted as 0 fetched)
      if (req.url.startsWith("https://export.arxiv.org/api/query")) {
        return { status: 200, data: '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>' };
      }
      return standard.request(req);
    },
  };
  const env = createEnvironment({ dataDir, items: libraryItems(), http, zoteroVersion: 10 });
  t.after(env.cleanup);
  const addon = await boot(env);
  useApiEmbeddings(env);
  env.prefs.set("extensions.zotwatch.followed.enabled", true);
  env.prefs.set("extensions.zotwatch.followed.authors", "Li Wang, https://orcid.org/0000-0002-1825-0097\nnot an author line");
  env.prefs.set("extensions.zotwatch.sources.arxiv.enabled", true);

  const result = await addon.runNow({ background: true });
  assert.equal(result.stats.newByLabel.followed, 1);
  assert.ok(result.stats.newByLabel.personal >= 1);
  assert.deepEqual(plain(result.errors), [{ stage: "followed", message: "无法识别的作者行：not an author line" }]);
  assert.match(openalexQueries[0], /filter=author\.orcid:https:\/\/orcid\.org\/0000-0002-1825-0097,from_publication_date:\d{4}-\d{2}-\d{2}&/);
  assert.ok(env.Zotero.HTTP.requests.every((r) => r.options.anon === true));
  assert.equal(env.Zotero.notifications.length, 1);
  assert.match(env.Zotero.notifications[0].text, /^发现 \d+ 篇新推荐(?:（\d+ 篇必读）)?，关注作者 1 篇。/);
  assert.ok(existsSync(join(dataDir, "zotwatch", "followed.json")));

  const view = await addon.getView();
  const followed = view.works.find((w) => w.label === "followed");
  assert.equal(followed.id, "doi:10.31223/x5zzz");
  assert.equal(followed.title, "Melt inclusions record slab-derived sulfur");
  assert.deepEqual(plain(followed.followedAuthors), ["Li Wang"]);
  assert.equal(followed.score, 0);
  assert.match(await addon.getStatus(), /关注作者：1 位（Li Wang），检查最近 60 天；1 行无法识别：not an author line/);

  // The dashboard: the author's pending works; an unparseable line is no source error
  const dash = await addon.getDashboard();
  const byLiWang = view.works.filter((w) => w.status === "new" && (w.followedAuthors || []).includes("Li Wang")).length;
  assert.ok(byLiWang > 0);
  assert.deepEqual(plain(dash.followed.authors), [{ name: "Li Wang", orcid: "0000-0002-1825-0097", openalexId: null, pending: byLiWang }]);
  assert.deepEqual(plain(dash.followed.invalid), ["not an author line"]);
  const followedRow = dash.sources.find((s) => s.key === "followed");
  assert.deepEqual(plain([followedRow.count, followedRow.error, followedRow.ran]), [1, null, true]);
  const arxivRow = dash.sources.find((s) => s.key === "arxiv");
  assert.deepEqual(plain([arxivRow.count, arxivRow.error, arxivRow.ran, arxivRow.detail]), [0, null, true, "physics.geo-ph, astro-ph.EP"]);

  // OpenAlex preprints become preprint items
  const itemID = await addon.addToLibrary(followed.id);
  const item = env.savedItems.find((i) => i.id === itemID);
  assert.equal(item.itemType, "preprint");
  assert.equal(item.fields.repository, "EarthArXiv");
  assert.equal(item.fields.DOI, "10.31223/x5zzz");
  assert.deepEqual(plain(item.creators), [{ creatorType: "author", firstName: "Li", lastName: "Wang" }]);

  // Shown once: the next run finds nothing new from the followed authors
  const again = await addon.runNow();
  assert.equal(again.stats.newByLabel.followed, 0);

  // A failed OpenAlex fetch is the row's error (not the unparseable line listed before it)
  openalexDown = true;
  const down = await addon.runNow();
  assert.deepEqual(plain(down.errors.map((e) => [e.stage, e.source ?? null])), [["followed", null], ["followed", "OpenAlex"]]);
  assert.match((await addon.getDashboard()).sources.find((s) => s.key === "followed").error, /403/);
  openalexDown = false;

  // Resetting forgets the seen-set with everything else
  await addon.resetData();
  assert.ok(!existsSync(join(dataDir, "zotwatch", "followed.json")));
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
  assert.deepEqual(env.Zotero.errors.map(String), []);
});

test("Zotero 10: missing abstracts are filled from publisher APIs, cached and their keys checked", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const abstract = "Copper isotopes trace sulfide saturation and porphyry ore formation beneath continental arcs in this study.";
  const standard = standardHttp({});
  const publisherCalls = [];
  const http = {
    async request(req) {
      if (req.url.startsWith("https://api.crossref.org/works?")) {
        // The copper paper arrives from Crossref without an abstract, under an Elsevier DOI
        const res = await standard.request(req);
        const items = res.data.message.items.map((w, i) => (i === 0 ? { ...w, DOI: "10.1016/j.gca.2099.1", abstract: undefined } : w));
        return { status: 200, data: { message: { ...res.data.message, items } } };
      }
      if (req.url.startsWith("https://api.elsevier.com/")) {
        publisherCalls.push(req);
        if (req.headers["X-ELS-APIKey"] !== "els-key") return { status: 401, data: {} };
        return req.url.includes("j.gca.2099.1")
          ? { status: 200, data: { "full-text-retrieval-response": { coredata: { "dc:description": abstract } } } }
          : { status: 404, data: {} };
      }
      if (req.url.startsWith("https://api.springernature.com/")) return { status: 401, data: { message: "Authentication failed" } };
      if (req.url.startsWith("https://api.openalex.org/works")) return { status: 200, data: { results: [] } };
      return standard.request(req);
    },
  };
  const env = createEnvironment({ dataDir, items: libraryItems(), http, zoteroVersion: 10 });
  t.after(env.cleanup);
  const addon = await boot(env);
  useApiEmbeddings(env);
  env.prefs.set("extensions.zotwatch.abstracts.elsevierKey", "els-key");

  const result = await addon.runNow();
  assert.deepEqual(plain(result.errors), []);
  assert.equal(result.stats.abstracts.elsevier, 1);
  assert.equal(result.stats.abstractsEnriched, 1);
  assert.deepEqual(plain(publisherCalls[0].headers), { "X-ELS-APIKey": "els-key", Accept: "application/json" });
  assert.ok(env.Zotero.HTTP.requests.every((r) => r.options.anon === true));
  const view = await addon.getView();
  assert.equal(view.works.find((w) => w.id === "doi:10.1016/j.gca.2099.1")?.abstract, abstract);
  assert.ok(existsSync(join(dataDir, "zotwatch", "abstracts-cache.json")));
  assert.match(await addon.getStatus(), /摘要补全：上次运行补全 1 \/ 1 篇（Elsevier 1）/);

  // The next run answers from the cache
  publisherCalls.length = 0;
  const again = await addon.runNow();
  assert.equal(again.stats.abstracts.cacheHits, 1);
  assert.equal(publisherCalls.length, 0);

  env.prefs.set("extensions.zotwatch.abstracts.springerKey", "spr-key");
  const status = await addon.testConnections();
  assert.match(status, /✅ Elsevier：Key 有效/);
  assert.match(status, /❌ Springer Nature：Key 无效（HTTP 401）/);

  await addon.resetData();
  assert.ok(!existsSync(join(dataDir, "zotwatch", "abstracts-cache.json")));
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
  assert.deepEqual(env.Zotero.errors.map(String), []);
});

test("Zotero 10: flagship-journal articles are gated into their own track", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const created = new Date(Date.now() - 2 * 86400000).toISOString();
  const nature = (n, title, abstract) => ({
    DOI: `10.1038/s41586-0${n}`,
    title: [title],
    author: [{ given: "Li", family: "Wang" }],
    abstract: `<jats:p>${abstract}</jats:p>`,
    "container-title": ["Nature"],
    created: { "date-time": created },
    URL: `https://doi.org/10.1038/s41586-0${n}`,
    type: "journal-article",
    ISSN: ["0028-0836", "1476-4687"],
  });
  const flagshipQueries = [];
  const standard = standardHttp({});
  let flagshipDown = false;
  const http = {
    async request(req) {
      const url = decodeURIComponent(req.url);
      if (req.url.startsWith("https://api.crossref.org/works?") && url.includes("issn:0028-0836")) {
        flagshipQueries.push(url);
        if (flagshipDown) return { status: 400, data: { message: "bad filter" } };
        const items = [
          nature(1, "Copper isotope records of mantle sulfur", "Copper isotope and sulfur budgets of the mantle beneath arcs"),
          nature(2, "Fast-charging battery anodes", "Battery battery anodes for electric vehicles"),
        ];
        return { status: 200, data: { message: { "total-results": items.length, items } } };
      }
      return standard.request(req);
    },
  };
  const env = createEnvironment({ dataDir, items: libraryItems(), http, zoteroVersion: 10 });
  t.after(env.cleanup);
  const addon = await boot(env);
  useApiEmbeddings(env);
  env.prefs.set("extensions.zotwatch.flagship.enabled", true);
  env.prefs.set("extensions.zotwatch.flagship.positiveAnchor", "Copper isotope geochemistry of mantle sulfur.");
  env.prefs.set("extensions.zotwatch.flagship.negativeAnchors", "Battery chemistry.\nGalaxy surveys.");

  const result = await addon.runNow({ background: true });
  assert.deepEqual(plain(result.errors), []);
  assert.equal(flagshipQueries.length, 1);
  assert.match(flagshipQueries[0], /issn:0028-0836,issn:1476-4687,issn:0036-8075/);
  assert.equal(result.stats.perSource["顶刊"], 2);
  assert.equal(result.stats.newByLabel.flagship, 1);
  assert.match(env.Zotero.notifications[0].text, /，顶刊地学 1 篇。/);

  const view = await addon.getView();
  const flagship = view.works.filter((w) => w.label === "flagship");
  assert.deepEqual(plain(flagship.map((w) => w.id)), ["doi:10.1038/s41586-01"]);
  assert.equal(flagship[0].venue, "Nature");
  assert.equal(flagship[0].gate.judged, false);
  assert.ok(flagship[0].gate.pos >= 0.3 && flagship[0].gate.neg < flagship[0].gate.pos);
  assert.ok(!view.works.some((w) => w.id === "doi:10.1038/s41586-02"), "rejected by the battery anchor");
  assert.match(
    await addon.getStatus(),
    /顶刊地学速览：12 个 ISSN，接受阈值 0\.30，灰区下限 0\.22；上次运行 2 篇候选，2 篇有摘要且在时间范围内，入选 1 篇（灰区 0 篇）/,
  );

  // The dashboard: the flagship funnel and the 顶刊 / Crossref rows
  const dash = await addon.getDashboard();
  assert.deepEqual(plain(dash.flagship.stats), plain(result.stats.flagship));
  assert.equal(dash.flagship.stats.passed, 1);
  const flagshipRow = dash.sources.find((s) => s.key === "flagship");
  assert.deepEqual(plain([flagshipRow.count, flagshipRow.error, flagshipRow.ran]), [2, null, true]);
  assert.equal(dash.sources.find((s) => s.key === "crossref").detail, `${result.stats.personalIssns} 个 ISSN`);

  const itemID = await addon.addToLibrary(flagship[0].id);
  assert.equal(env.savedItems.find((i) => i.id === itemID).itemType, "journalArticle");

  // A failed flagship query is the 顶刊 row's error
  flagshipDown = true;
  await addon.runNow();
  assert.match((await addon.getDashboard()).sources.find((s) => s.key === "flagship").error, /bad filter|400/);
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
  assert.deepEqual(env.Zotero.errors.map(String), []);
});

test("Zotero 10: research directions are labeled from library titles and tags", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const prompts = [];
  const standard = standardHttp({});
  let llmDown = false;
  const http = {
    async request(req) {
      if (req.url === "https://llm.example/v1/chat/completions") {
        if (llmDown) return { status: 401, data: { error: { message: "invalid key" } } };
        const prompt = JSON.parse(req.body).messages[0].content;
        prompts.push(prompt);
        const content = prompt.includes("研究方向标签")
          ? JSON.stringify([...prompt.matchAll(/^ {2}Titles: (.*)$/gm)].map(([, titles]) => (titles.includes("Copper") ? "铜同位素分馏" : "地幔熔融")))
          : '{"research_question":"问题","methodology":"方法"}';
        return { status: 200, data: { choices: [{ message: { content } }] } };
      }
      return standard.request(req);
    },
  };
  const env = createEnvironment({ dataDir, items: libraryItems(), http, zoteroVersion: 10 });
  t.after(env.cleanup);
  const addon = await boot(env);
  useApiEmbeddings(env);
  const llmPrefs = { "llm.enabled": true, "llm.apiKey": "k", "llm.baseURL": "https://llm.example/v1", "llm.model": "m", "llm.translateTitles": false, "llm.summaryTopN": "1" };
  for (const [key, value] of Object.entries(llmPrefs)) env.prefs.set(`extensions.zotwatch.${key}`, value);
  assert.equal(addon.getSettings().llm.topicLabels, true, "on by default");

  const result = await addon.runNow();
  assert.deepEqual(plain(result.errors), []);
  assert.deepEqual(plain(result.stats.topics), { clusters: 2, labeled: 2, generated: 2 });
  const labelPrompts = prompts.filter((p) => p.includes("研究方向标签"));
  assert.equal(labelPrompts.length, 1);
  assert.match(labelPrompts[0], /Cluster 1 \(12 papers\):\n {2}Titles: "Copper isotope fractionation in porphyry systems \d+", .*\n {2}Keywords: Copper isotopes, Porphyry\n/);
  assert.match(labelPrompts[0], /Cluster 2 \(4 papers\):\n {2}Titles: .*\n {2}Keywords: N\/A\n/, "items without tags");
  assert.doesNotMatch(labelPrompts[0], /⭐|\/unread|ZotWatch/);
  assert.ok(existsSync(join(dataDir, "zotwatch", "cluster-labels.json")));

  const view = await addon.getView();
  const top = view.works.find((w) => w.id === "doi:10.5555/new.1");
  assert.equal(top.topic, "铜同位素分馏");
  assert.equal(view.works.find((w) => w.id === "doi:10.5555/new.2").topic, "地幔熔融");
  assert.match(await addon.getStatus(), /\n研究方向：铜同位素分馏（12 篇）、地幔熔融（4 篇）\n/);
  const dash = await addon.getDashboard();
  assert.deepEqual(plain(dash.topics), [{ label: "铜同位素分馏", size: 12 }, { label: "地幔熔融", size: 4 }]);
  assert.equal(dash.topicCount, 2);
  assert.equal(dash.clusters, 2);
  assert.equal(dash.sources.find((s) => s.key === "llm").error, null);

  // The summary note names the research direction
  await addon.addToLibrary(top.id);
  const note = env.savedItems.find((i) => typeof i.note === "string");
  assert.match(note.note, /<p>研究方向：铜同位素分馏<\/p>/);

  // Turned off: no line in the status, no new label request
  env.prefs.set("extensions.zotwatch.llm.topicLabels", false);
  assert.doesNotMatch(await addon.getStatus(), /研究方向/);
  await addon.resetData();
  await addon.runNow();
  assert.equal(prompts.filter((p) => p.includes("研究方向标签")).length, 1);
  assert.ok(!existsSync(join(dataDir, "zotwatch", "cluster-labels.json")), "labels are cleared with the cache");

  // A failing LLM (summaries) is the 大模型 row's error
  llmDown = true;
  await addon.resetData();
  const failing = await addon.runNow();
  assert.ok(failing.errors.some((e) => e.stage === "llm"));
  assert.match((await addon.getDashboard()).sources.find((s) => s.key === "llm").error, /invalid key/);
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
  assert.deepEqual(env.Zotero.errors.map(String), []);
});

test("an unreadable journals.csv does not stop the plugin", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  mkdirSync(join(dataDir, "zotwatch"), { recursive: true });
  writeFileSync(join(dataDir, "zotwatch", "journals.csv"), Buffer.from([0xc6, 0xda, 0xbf, 0xaf])); // GBK bytes
  const env = createEnvironment({ dataDir, items: libraryItems(), http: standardHttp({}), zoteroVersion: 10 });
  t.after(env.cleanup);
  const readUTF8 = env.context.IOUtils.readUTF8;
  env.context.IOUtils.readUTF8 = async (p) => {
    if (p.endsWith("journals.csv")) throw new Error("Could not read file because it is not UTF-8 encoded");
    return readUTF8(p);
  };
  const addon = await boot(env);
  assert.ok(env.Zotero.ZotWatch, "startup completed");
  assert.equal(env.registeredPanes.length, 1);
  assert.match(await addon.getStatus(), /UTF-8/);
  assert.equal(env.Zotero.errors.length, 1);
  env.Zotero.errors.length = 0;
  await addon.setJournalWhitelist("issn,title\n1234-5678,X");
  assert.doesNotMatch(await addon.getStatus(), /UTF-8/);

  // A failed write (the CSV locked by Excel) keeps the saved list, so the
  // settings page sees the edit as unsaved and the next attempt writes it
  const writeUTF8 = env.context.IOUtils.writeUTF8;
  env.context.IOUtils.writeUTF8 = async (p, text, opts) => {
    if (p.endsWith("journals.csv")) throw new Error("The file is locked");
    return writeUTF8(p, text, opts);
  };
  await assert.rejects(addon.setJournalWhitelist("issn,title\n0012-821X,EPSL"), /locked/);
  assert.equal(addon.getJournalWhitelist(), "issn,title\n1234-5678,X");
  env.context.IOUtils.writeUTF8 = writeUTF8;
  await addon.setJournalWhitelist("issn,title\n0012-821X,EPSL");
  assert.equal(readFileSync(join(dataDir, "zotwatch", "journals.csv"), "utf8"), "issn,title\n0012-821X,EPSL");
  await vm.runInContext("shutdown({}, ADDON_DISABLE)", env.context);
});

test("preference pane initialises from Zotero's events inside a sandbox", async () => {
  const code = readFileSync(join(addonDir, "content", "preferences.js"), "utf8");
  const docListeners = [];
  const elements = new Map();
  const el = (id) => {
    const node = { id, listeners: {}, textContent: "", addEventListener(type, fn) {
      (this.listeners[type] ||= []).push(fn);
    } };
    elements.set(id, node);
    return node;
  };
  const rootEl = el("zw-prefs-root");
  for (const id of ["zw-pending", "zw-last", "zw-embedding", "zw-open", "zw-open-settings", "zw-run"]) el(id);
  const calls = [];
  const addonListeners = new Set();
  const now = Date.now();
  const view = {
    works: [{ status: "new" }, { status: "new" }, { status: "added" }],
    lastRun: { runAt: now - 3 * 3600000, finishedAt: now - 2 * 3600000 - 60000 },
    running: false,
    lastError: null,
  };
  const api = {
    getView: async () => {
      calls.push(["getView"]);
      return view;
    },
    embeddingInfo: () => ({ label: "阿里云 DashScope · text-embedding-v4" }),
    openWindow: (page) => calls.push(["openWindow", page]),
    runFromMenu: () => calls.push(["runFromMenu"]),
    addListener: (fn) => addonListeners.add(fn),
    removeListener: (fn) => addonListeners.delete(fn),
  };
  const document = {
    addEventListener: (type, fn, capture) => docListeners.push({ type, fn, capture }),
    getElementById: (id) => elements.get(id) || null,
  };
  // Zotero 8+: scripts run in their own global, so nothing is visible to inline handlers
  const sandbox = vm.createContext({ document, Zotero: { ZotWatch: api, logError() {} } });
  vm.runInContext(code, sandbox);
  const fire = (type, target) => docListeners.filter((l) => l.type === type).forEach((l) => l.fn({ type, target }));
  const tick = () => new Promise((r) => setTimeout(r, 0));
  assert.ok(docListeners.every((l) => l.capture === true), "capture listeners see non-bubbling pane events");

  fire("load", { id: "something-else" });
  assert.equal(addonListeners.size, 0);
  fire("load", rootEl);
  assert.equal(addonListeners.size, 1, "the pane follows runs while it is open");
  assert.deepEqual(calls, [], "nothing is read until the pane is shown");
  fire("showing", rootEl);
  await tick();
  assert.equal(elements.get("zw-pending").textContent, "2 篇");
  assert.match(elements.get("zw-last").textContent, /^2 小时前（\d{4}-\d{2}-\d{2} \d{2}:\d{2}）$/);
  assert.equal(elements.get("zw-embedding").textContent, "阿里云 DashScope · text-embedding-v4");

  // The buttons open the main window (overview / settings) or start a run like the Tools menu
  for (const id of ["zw-open", "zw-open-settings", "zw-run"]) elements.get(id).listeners.command[0]();
  assert.deepEqual(calls.slice(1), [["openWindow", "overview"], ["openWindow", "settings"], ["runFromMenu"]]);

  // Run events refresh the status
  view.running = true;
  [...addonListeners][0]({ type: "start" });
  await tick();
  assert.equal(elements.get("zw-last").textContent, "正在检查…");
  view.running = false;
  view.lastError = { message: "网络连接失败", at: now };
  [...addonListeners][0]({ type: "error", message: "网络连接失败" });
  await tick();
  assert.equal(elements.get("zw-last").textContent, "失败：网络连接失败");

  // A finished run updates 上次检查 and the pending count
  view.lastError = null;
  view.works.push({ status: "new" });
  view.lastRun = { runAt: now - 120000, finishedAt: now - 60000 };
  [...addonListeners][0]({ type: "done" });
  await tick();
  assert.equal(elements.get("zw-pending").textContent, "3 篇");
  assert.match(elements.get("zw-last").textContent, /^1 分钟前（/);
  // A recommendation handled in the main window (inbox event)
  view.works[0].status = "dismissed";
  [...addonListeners][0]({ type: "inbox" });
  await tick();
  assert.equal(elements.get("zw-pending").textContent, "2 篇");
  // A cancelled run no longer reads 正在检查…
  view.running = true;
  [...addonListeners][0]({ type: "start" });
  await tick();
  assert.equal(elements.get("zw-last").textContent, "正在检查…");
  view.running = false;
  [...addonListeners][0]({ type: "cancelled" });
  await tick();
  assert.match(elements.get("zw-last").textContent, /^1 分钟前（/);

  fire("unload", rootEl);
  assert.equal(addonListeners.size, 0, "the listener is removed when the pane closes");
});

const MAIN_SCRIPTS = ["main.js", "overview.js", "inbox.js", "settings.js"];

/** Every element below `node` (document order). */
const allElements = (node, out = []) => {
  for (const el of node.elements) {
    out.push(el);
    allElements(el, out);
  }
  return out;
};
const hasClass = (el, cls) => (el.attrs.class || "").split(/\s+/).includes(cls);

const collectIds = (node, out) => {
  for (const el of node.elements) {
    if (el.attrs.id) out.add(el.attrs.id);
    collectIds(el, out);
  }
  return out;
};

/**
 * data-pref keys bound by the main window: in its markup, set from its
 * scripts, or listed in the settings page's FIELDS table (`pref: "<key>"`,
 * from which settings.js builds the controls; test/settings.test.js checks
 * the rendered data-pref attributes).
 */
function mainWindowPrefKeys() {
  const sources = ["addon/content/main.xhtml", ...MAIN_SCRIPTS.map((f) => `addon/content/${f}`)].map((f) => readFileSync(join(root, f), "utf8"));
  const keys = new Set();
  for (const text of sources) {
    for (const [, key] of text.matchAll(/(?:data-pref(?:=|"?\s*:\s*)|\bpref:\s*)"([\w.]+)"/g)) keys.add(key);
  }
  return keys;
}

test("XHTML files are well-formed and scripts reference existing element ids", () => {
  const mainXhtml = readFileSync(join(root, "addon/content/main.xhtml"), "utf8");
  const mainDoc = parseXml(mainXhtml, { strict: true });
  const mainIds = collectIds(mainDoc, new Set());
  assert.equal(mainDoc.attr("id"), "zotwatch-main");
  assert.equal(mainDoc.attr("windowtype"), "zotwatch:main");
  assert.equal(mainDoc.attr("persist"), "screenX screenY width height sizemode");
  assert.doesNotMatch(mainXhtml, /\son[a-z]+=/, "no inline handlers in the main window");
  assert.doesNotMatch(mainXhtml, /<select[\s>]/, "no HTML <select> in the chrome window");
  for (const file of MAIN_SCRIPTS) {
    assert.match(mainXhtml, new RegExp(`<script src="chrome://zotwatch/content/${file}">`), `main.xhtml loads ${file}`);
    const css = file.replace(/\.js$/, ".css");
    assert.match(mainXhtml, new RegExp(`href="chrome://zotwatch/content/${css}"`), `main.xhtml loads ${css}`);
    const js = readFileSync(join(root, "addon/content", file), "utf8");
    assert.doesNotMatch(js, /\.innerHTML\s*=|insertAdjacentHTML/, `${file} inserts text with textContent only`);
    // Ids reach $() / getElementById directly or through segmented("…") / setSegmented("…")
    for (const [, id] of js.matchAll(/(?:\$|getElementById|segmented|setSegmented)\("([\w-]+)"/g)) {
      assert.ok(mainIds.has(id), `main.xhtml lacks #${id} (used by ${file})`);
    }
    // Ids in selectors: querySelector("#page-inbox .tab") and friends
    for (const [, id] of js.matchAll(/querySelector(?:All)?\(\s*["'`]#([\w-]+)/g)) {
      assert.ok(mainIds.has(id), `main.xhtml lacks #${id} (used in a selector by ${file})`);
    }
    // Ids built from a prefix: `count-${status}` and friends
    for (const [, prefix] of js.matchAll(/\$\(`([\w-]+)-\$\{/g)) {
      assert.ok([...mainIds].some((id) => id.startsWith(`${prefix}-`)), `main.xhtml has no #${prefix}-* element (used by ${file})`);
    }
  }
  for (const s of ["new", "added", "dismissed"]) assert.ok(mainIds.has(`count-${s}`));
  for (const id of ["inbox-track", "inbox-label", "inbox-sort"]) assert.ok(mainIds.has(id));

  // The values the scripts switch on: pages (main.js PAGE_KEYS), inbox tabs and filters
  const mainJs = readFileSync(join(root, "addon/content/main.js"), "utf8");
  const inboxJs = readFileSync(join(root, "addon/content/inbox.js"), "utf8");
  const pageKeys = JSON.parse(mainJs.match(/const PAGE_KEYS = (\[[^\]]*\]);/)[1]);
  assert.deepEqual(pageKeys, ["overview", "inbox", "settings"]);
  const elements = allElements(mainDoc);
  const dataOf = (cls, attr) => elements.filter((el) => hasClass(el, cls)).map((el) => el.attrs[attr]);
  assert.deepEqual(dataOf("nav-item", "data-page"), pageKeys, "one sidebar button per page");
  assert.deepEqual(dataOf("page", "data-page"), pageKeys, "one <section class=page> per page");
  assert.deepEqual(dataOf("page-actions", "data-page"), pageKeys, "one header action group per page");
  assert.equal(elements.find((el) => el.attrs.id === "page-inbox")?.attrs["data-page"], "inbox");
  const tabs = elements.filter((el) => hasClass(el, "tab")).map((el) => el.attrs["data-tab"]);
  assert.deepEqual(tabs, JSON.parse(inboxJs.match(/const TABS = (\[[^\]]*\]);/)[1]));
  const segValues = (id) => allElements(elements.find((el) => el.attrs.id === id)).map((el) => el.attrs["data-value"]).filter(Boolean);
  assert.deepEqual(segValues("inbox-track"), ["all", "personal", "flagship", "followed"]);
  assert.deepEqual(segValues("inbox-label"), ["all", "must_read", "latest"]);
  assert.deepEqual(segValues("inbox-sort"), ["score", "date"]);
  for (const value of ["must_read", "latest"]) assert.match(inboxJs, new RegExp(`state\\.filter === "${value}"`), `inbox.js filters on ${value}`);
  assert.match(inboxJs, /state\.sort === "date"/);

  // Zotero's edit overlay gives the text fields their Cut / Copy / Paste context menu (before main.js)
  const scripts = elements.filter((el) => el.name === "script").map((el) => el.attrs.src);
  assert.deepEqual(scripts.slice(0, 3), ["chrome://global/content/globalOverlay.js", "chrome://zotero/content/editMenuOverlay.js", "chrome://zotwatch/content/main.js"]);
  const l10n = elements.filter((el) => el.name === "link" && el.attrs.rel === "localization").map((el) => el.attrs.href);
  assert.ok(l10n.includes("toolkit/global/textActions.ftl"), "the context menu's labels");
  // addon.js switches an open window through window.ZotWatchMain.navigate (test/main.test.js drives it)
  assert.match(mainJs, /window\.ZotWatchMain = \{[^}]*\bnavigate\b/);
  // Inline SVG must carry the SVG namespace (strict XHTML)
  for (const [tag] of mainXhtml.matchAll(/<svg\b[^>]*>/g)) assert.match(tag, /xmlns="http:\/\/www\.w3\.org\/2000\/svg"/);
  // Every icon used from the scripts exists in the sprite
  const symbols = new Set([...mainXhtml.matchAll(/<symbol id="i-([\w-]+)"/g)].map(([, name]) => name));
  for (const [, name] of mainXhtml.matchAll(/href="#i-([\w-]+)"/g)) assert.ok(symbols.has(name), `missing icon ${name}`);
  for (const file of MAIN_SCRIPTS) {
    const js = readFileSync(join(root, "addon/content", file), "utf8");
    for (const [, name] of js.matchAll(/(?:svgIcon\(|icon: )"([\w-]+)"/g)) assert.ok(symbols.has(name), `missing icon ${name} (used by ${file})`);
  }

  // Every preference the main window binds has a default in prefs.js
  const prefsJs = readFileSync(join(root, "addon/prefs.js"), "utf8");
  for (const key of mainWindowPrefKeys()) {
    assert.ok(prefsJs.includes(`pref("extensions.zotwatch.${key}",`), `no default for ${key}`);
  }

  const prefXhtml = readFileSync(join(root, "addon/content/preferences.xhtml"), "utf8");
  assert.doesNotMatch(prefXhtml, /\son[a-z]+=/, "no inline handlers: pane scripts are sandboxed on Zotero 8+");
  // Zotero parses the pane as a XUL fragment with the html: prefix bound
  const prefDoc = parseXml(
    `<box xmlns="http://www.mozilla.org/keymaster/gatekeeper/there.is.only.xul" xmlns:html="http://www.w3.org/1999/xhtml">${prefXhtml}</box>`,
    { strict: true },
  );
  const prefIds = collectIds(prefDoc, new Set());
  const prefJs = readFileSync(join(root, "addon/content/preferences.js"), "utf8");
  const paneIds = [...prefJs.matchAll(/(?:on|getElementById|setText)\("(zw-[\w-]+)"/g)].map(([, id]) => id);
  assert.ok(paneIds.length >= 6);
  for (const id of paneIds) assert.ok(prefIds.has(id), `preferences.xhtml lacks #${id}`);
  // Since 0.3.0 the pane only shows a status summary: every setting lives in the main window
  assert.doesNotMatch(prefXhtml, /\spreference=/, "no preference bindings remain in the Zotero settings pane");

  // Both locales define every Fluent message used by the menus
  for (const locale of ["zh-CN", "en-US"]) {
    const ftl = readFileSync(join(root, `addon/locale/${locale}/zotwatch-mainWindow.ftl`), "utf8");
    for (const id of ["zotwatch-menu-open", "zotwatch-menu-run"]) assert.match(ftl, new RegExp(`^${id} =`, "m"), `${locale} lacks ${id}`);
  }
});

test("every setting of the 0.2.0 settings pane is editable in the main window", () => {
  const bound = mainWindowPrefKeys();
  const missing = PREFS_0_2_0.filter((key) => !bound.has(key));
  assert.deepEqual(missing, []);
  // The settings page also binds a few advanced prefs that had no control before
  for (const key of ["inbox.retentionDays", "embedding.dimensions", "scoring.thresholdMode", "scoring.temporalHalflifeDays"]) {
    assert.ok(bound.has(key), `${key} is editable`);
  }
});

test("window routes and the run schedule", () => {
  assert.equal(normalizePage("inbox"), "inbox");
  assert.equal(normalizePage("#settings/ai"), "settings/ai");
  assert.equal(normalizePage("settings/../x"), "settings");
  assert.equal(normalizePage("chrome://evil"), "overview");
  assert.equal(normalizePage(undefined), "overview");

  const H = 3600000;
  assert.equal(nextRunAt({ autoRun: false, intervalHours: 24, lastRun: 0, lastAttempt: 0 }), null);
  assert.equal(nextRunAt({ autoRun: true, intervalHours: 24, lastRun: 100 * H, lastAttempt: 100 * H }), 124 * H);
  // A failed attempt after the last success: retried 2 hours later, not before the interval is over
  assert.equal(nextRunAt({ autoRun: true, intervalHours: 1, lastRun: 100 * H, lastAttempt: 110 * H }), 112 * H);
  assert.equal(nextRunAt({ autoRun: true, intervalHours: 24, lastRun: 0, lastAttempt: 0 }), 24 * H, "never run: due at once");

  assert.equal(connectionLine({ name: "向量化", target: "local", ok: true, detail: "正常，维度 1024" }), "✅ 向量化（local）：正常，维度 1024");
  assert.equal(connectionLine({ name: "Elsevier", target: null, ok: false, detail: "Key 无效（HTTP 401）" }), "❌ Elsevier：Key 无效（HTTP 401）");
  assert.equal(connectionLine({ name: "大模型", ok: null, detail: "未启用" }), "ℹ️ 大模型：未启用");
});
