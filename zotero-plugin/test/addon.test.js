// Smoke test of the built plugin in a simulated Zotero environment: loads
// build/addon/bootstrap.js into a VM context with stubbed Zotero/Firefox APIs
// and drives the full lifecycle (startup -> run -> add item -> shutdown).

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import { parseXml } from "../src/core/xml.js";
import { embeddingHandler, mockHttp } from "./helpers.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// Values created inside the VM have foreign prototypes; compare them structurally
const plain = (x) => JSON.parse(JSON.stringify(x));
const addonDir = join(root, "build", "addon");

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
    setAttribute(k, v) {
      this.attributes[k] = v;
    },
    getAttribute(k) {
      return this.attributes[k];
    },
    addEventListener(type, fn) {
      (this.listeners[type] ||= []).push(fn);
    },
    dispatch(type) {
      for (const fn of this.listeners[type] || []) fn({ type });
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
    },
  };
  return el;
}

function fakeWindow() {
  const doc = { elements: new Map() };
  doc.getElementById = (id) => doc.elements.get(id) || null;
  doc.createXULElement = (tag) => fakeElement(doc, tag);
  const popup = fakeElement(doc, "menupopup");
  popup.id = "menu_ToolsPopup";
  doc.elements.set(popup.id, popup);
  const opened = [];
  return {
    document: doc,
    ZoteroPane: { selectItem() {} },
    opened,
    openDialog(url, name, features) {
      const dialog = { url, name, features, closed: false, focus() {}, close() { this.closed = true; } };
      opened.push(dialog);
      return dialog;
    },
    focus() {},
  };
}

function createEnvironment({ dataDir, items, http }) {
  const prefs = new Map();
  const prefDefaults = new Map();
  const savedItems = [];
  const collections = [];
  const timers = new Set();
  const mainWindow = fakeWindow();
  const registeredPanes = [];
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
      this.id = nextId++;
      collections.push(this);
    }
  }

  const Zotero = {
    initializationPromise: Promise.resolve(),
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
      async request(method, url, options) {
        const res = await http.request({ method, url, headers: options.headers, body: options.body, responseType: options.responseType });
        const data = options.responseType === "json" && typeof res.data === "string" ? null : res.data;
        return { status: res.status, response: data };
      },
    },
    Libraries: { userLibraryID: 1, get: (id) => id === 1 },
    Items: {
      async getAll(libraryID, onlyTopLevel) {
        assert.equal(libraryID, 1);
        assert.equal(onlyTopLevel, true);
        return items.map((data) => ({
          ...data,
          isRegularItem: () => data.itemType !== "note",
          getField(name, unformatted, baseMapped) {
            if (name === "publicationTitle") assert.equal(baseMapped, true);
            return { title: data.title, abstractNote: data.abstract, DOI: data.doi, ISSN: data.issn, publicationTitle: data.venue, extra: "" }[name] ?? "";
          },
        }));
      },
      async loadDataTypes() {},
    },
    ItemTypes: { getID: (t) => (t === "preprint" ? 31 : 1) },
    Item: FakeItem,
    Collection: FakeCollection,
    Collections: { getByLibrary: () => collections },
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
      startCloseTimer() {}
    },
    notifications: [],
    Utilities: { Internal: { openPreferences(id) {
      Zotero.openedPrefs = id;
    } } },
    launchURL(url) {
      Zotero.launched = url;
    },
    File: {
      getResourceAsync: async (url) => readFileSync(join(addonDir, url.replace("chrome://zotwatch/", "")), "utf8"),
    },
  };

  const IOUtils = {
    makeDirectory: async (p) => mkdirSync(p, { recursive: true }),
    exists: async (p) => existsSync(p),
    readJSON: async (p) => JSON.parse(readFileSync(p, "utf8")),
    writeJSON: async (p, obj) => writeFileSync(p, JSON.stringify(obj)),
    read: async (p) => new Uint8Array(readFileSync(p)),
    write: async (p, bytes) => writeFileSync(p, bytes),
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

  return { context, Zotero, prefs, prefDefaults, mainWindow, registeredPanes, chromeRegistrations, savedItems, collections, timers };
}

function libraryItems() {
  return [
    ...Array.from({ length: 12 }, (_, i) => ({
      key: `C${i}`,
      itemType: "journalArticle",
      title: `Copper isotope fractionation in porphyry systems ${i}`,
      abstract: `Copper isotopes and sulfide saturation in arc magmas, study ${i}`,
      doi: `10.4444/lib.${i}`,
      issn: "0016-7037",
      venue: "Geochimica et Cosmochimica Acta",
      dateAdded: "2024-05-01 10:00:00",
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
  ];
}

test("plugin lifecycle in a simulated Zotero", async (t) => {
  const dataDir = mkdtempSync(join(tmpdir(), "zotwatch-"));
  t.after(() => rmSync(dataDir, { recursive: true, force: true }));
  const counter = {};
  const http = mockHttp([
    ["https://api.crossref.org/works", () => ({ message: { "total-results": 3, items: recentCrossref() } })],
    ["https://api.crossref.org/journals", () => ({ message: { items: [] } })],
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)],
  ]);
  const env = createEnvironment({ dataDir, items: libraryItems(), http });
  const { context, Zotero, mainWindow } = env;
  t.after(() => {
    for (const timer of env.timers) {
      clearTimeout(timer);
      clearInterval(timer);
    }
  });

  // --- startup
  vm.runInContext(readFileSync(join(addonDir, "bootstrap.js"), "utf8"), context, { filename: "bootstrap.js" });
  await vm.runInContext('startup({ id: "zotwatch@crazyhalfday.github.io", version: "0.1.0", rootURI: "jar:file:///zotwatch.xpi!/" })', context);

  assert.deepEqual(plain(env.chromeRegistrations[0].entries), [["content", "zotwatch", "jar:file:///zotwatch.xpi!/content/"]]);
  const addon = Zotero.ZotWatch;
  assert.ok(addon, "Zotero.ZotWatch is exposed");
  assert.equal(env.registeredPanes.length, 1);
  assert.equal(env.registeredPanes[0].src, "chrome://zotwatch/content/preferences.xhtml");
  assert.equal(env.registeredPanes[0].pluginID, "zotwatch@crazyhalfday.github.io");
  const menu = mainWindow.document.getElementById("menu_ToolsPopup");
  assert.deepEqual(menu.children.map((c) => c.id), ["zotwatch-menu-separator", "zotwatch-menu-open", "zotwatch-menu-run"]);
  assert.equal(env.timers.size, 2, "scheduler armed");
  assert.equal(addon.isRunDue(), true, "first run is due");

  // A second main window load must not duplicate menu items
  vm.runInContext("onMainWindowLoad({ window: __win })", Object.assign(context, { __win: mainWindow }));
  assert.equal(menu.children.length, 3);

  // --- menu opens the recommendations window
  mainWindow.document.getElementById("zotwatch-menu-open").dispatch("command");
  assert.equal(mainWindow.opened[0].url, "chrome://zotwatch/content/recommendations.xhtml");

  // --- run with a configured API provider
  env.prefs.set("extensions.zotwatch.embedding.provider", "dashscope");
  env.prefs.set("extensions.zotwatch.embedding.apiKey", "sk-test");
  const events = [];
  addon.addListener((e) => events.push(e.type));
  const result = await addon.runNow();
  assert.equal(result.stats.libraryItems, 16, "notes are excluded");
  assert.equal(result.stats.fetched, 3);
  assert.ok(result.stats.recommended >= 1);
  assert.ok(events.includes("start") && events.includes("progress") && events.includes("done"));
  assert.ok(Number(env.prefs.get("extensions.zotwatch.schedule.lastRun")) > 0);
  assert.equal(addon.isRunDue(), false);
  assert.ok(existsSync(join(dataDir, "zotwatch", "inbox.json")));
  assert.ok(existsSync(join(dataDir, "zotwatch", "profile.bin")));

  const view = await addon.getView();
  const top = view.works.sort((a, b) => b.score - a.score)[0];
  assert.equal(top.id, "doi:10.5555/new.1");
  assert.ok(!view.works.some((w) => w.id === "doi:10.5555/new.3"), "off-topic paper is gated");

  // --- add to library (translator finds nothing -> item built from metadata)
  const itemID = await addon.addToLibrary(top.id);
  const item = env.savedItems.find((i) => i.id === itemID);
  assert.equal(item.itemType, "journalArticle");
  assert.equal(item.fields.title, top.title);
  assert.equal(item.fields.DOI, "10.5555/new.1");
  assert.equal(item.fields.publicationTitle, "Geochimica et Cosmochimica Acta");
  assert.deepEqual(plain(item.creators), [{ creatorType: "author", firstName: "Li", lastName: "Wang" }]);
  assert.deepEqual(plain(item.tags), ["ZotWatch"]);
  assert.equal(env.collections[0].name, "ZotWatch 推荐");
  assert.deepEqual(plain(item.collections), [env.collections[0].id]);
  const afterAdd = await addon.getView();
  assert.equal(afterAdd.works.find((w) => w.id === top.id).status, "added");

  // --- dismiss, status text, preset journals, preferences
  const other = afterAdd.works.find((w) => w.status === "new");
  if (other) {
    await addon.setStatus(other.id, "dismissed");
    assert.equal((await addon.getView()).works.find((w) => w.id === other.id).status, "dismissed");
  }
  assert.match(await addon.getStatus(), /文库条目：16/);
  assert.match(await addon.loadPresetJournals(), /^issn,title,category,impact_factor/);
  assert.match(await addon.suggestJournals(), /0016-7037,Geochimica et Cosmochimica Acta/);
  addon.openPreferences();
  assert.equal(Zotero.openedPrefs, "zotwatch-prefpane");
  addon.openUrl("javascript:alert(1)");
  assert.equal(Zotero.launched, undefined, "only http(s) URLs are opened");

  // --- background run notifies only when there is something new
  env.prefs.set("extensions.zotwatch.schedule.lastRun", "0");
  env.prefs.set("extensions.zotwatch.schedule.lastAttempt", "0");
  await addon.runNow({ background: true });
  assert.equal(Zotero.notifications.length, 0);

  // --- shutdown cleans up
  vm.runInContext("shutdown({}, ADDON_DISABLE)", context);
  assert.equal(menu.children.length, 0);
  assert.equal(Zotero.ZotWatch, undefined);
  assert.equal(env.timers.size, 0);
  assert.equal(env.chromeRegistrations[0].destroyed, true);
  assert.equal(mainWindow.opened[0].closed, true);
  assert.deepEqual(Zotero.errors.map(String), []);
});

test("XHTML files are well-formed and scripts reference existing element ids", () => {
  const recXhtml = readFileSync(join(root, "addon/content/recommendations.xhtml"), "utf8");
  const recDoc = parseXml(recXhtml, { strict: true });
  const recIds = new Set();
  const collectIds = (node, out) => {
    for (const el of node.elements) {
      if (el.attrs.id) out.add(el.attrs.id);
      collectIds(el, out);
    }
    return out;
  };
  collectIds(recDoc, recIds);
  const recJs = readFileSync(join(root, "addon/content/recommendations.js"), "utf8");
  for (const [, id] of recJs.matchAll(/\$\("([\w-]+)"\)/g)) assert.ok(recIds.has(id), `recommendations.xhtml lacks #${id}`);
  for (const s of ["new", "added", "dismissed"]) assert.ok(recIds.has(`count-${s}`));

  const prefXhtml = readFileSync(join(root, "addon/content/preferences.xhtml"), "utf8");
  // Zotero parses the pane as a XUL fragment with the html: prefix bound
  const prefDoc = parseXml(
    `<box xmlns="http://www.mozilla.org/keymaster/gatekeeper/there.is.only.xul" xmlns:html="http://www.w3.org/1999/xhtml">${prefXhtml}</box>`,
    { strict: true },
  );
  const prefIds = collectIds(prefDoc, new Set());
  const prefJs = readFileSync(join(root, "addon/content/preferences.js"), "utf8");
  for (const [, id] of prefJs.matchAll(/(?:on|getElementById)\("(zw-[\w-]+)"/g)) assert.ok(prefIds.has(id), `preferences.xhtml lacks #${id}`);

  // Every preference bound in the pane has a default in prefs.js
  const prefsJs = readFileSync(join(root, "addon/prefs.js"), "utf8");
  for (const [, key] of prefXhtml.matchAll(/preference="([\w.]+)"/g)) {
    assert.ok(prefsJs.includes(`pref("${key}",`), `no default for ${key}`);
  }
});
