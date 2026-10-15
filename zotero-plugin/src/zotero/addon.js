// Plugin object: lifecycle, Tools menu, scheduler, and the API used by the
// recommendations window and the preference pane (exposed as Zotero.ZotWatch).

/* global Zotero, Services */

import { libraryVenues, parseWhitelist, serializeWhitelist } from "../core/journals.js";
import { CancelledError, ZotWatchEngine } from "../engine.js";
import { createEmbedder } from "../embedding/providers.js";
import { createLlmClient } from "../llm/llm.js";
import { addWorkToZotero } from "./items.js";
import {
  createZoteroLibrary,
  createZoteroStorage,
  createZoteroTransport,
  getPref,
  log,
  profileLibraryID,
  readSettings,
  setPref,
  targetLibraryID,
} from "./env.js";

const PANE_ID = "zotwatch-prefpane";
const HELP_URL = "https://github.com/CrazyHalfDay/AIZotWatch/blob/main/zotero-plugin/README.md";
const DIALOG_URL = "chrome://zotwatch/content/recommendations.xhtml";
const DIALOG_TYPE = "zotwatch:recommendations";
const MENU_ICON = "chrome://zotwatch/content/icons/menu.svg";
const MAIN_WINDOW_FTL = "zotwatch-mainWindow.ftl";
const WHITELIST_FILE = "journals.csv";
const FIRST_CHECK_DELAY = 3 * 60 * 1000;
const CHECK_INTERVAL = 30 * 60 * 1000;
const RETRY_AFTER_FAILURE = 2 * 3600 * 1000;
const SHUTDOWN_GRACE = 5000;
const QUIT_GRACE = 1500;
const DOM_MENU_IDS = ["zotwatch-menu-separator", "zotwatch-menu-open", "zotwatch-menu-run"];

/** Wait for `promise` to settle, but at most `ms` milliseconds (leaves no timer behind). */
async function settleWithin(promise, ms) {
  let timer;
  await Promise.race([promise.catch(() => {}), new Promise((resolve) => (timer = setTimeout(resolve, ms)))]);
  clearTimeout(timer);
}

/** ProgressWindow parses its text as markup; keep arbitrary messages inert. */
export function plainNoticeText(text) {
  return String(text ?? "").replace(/</g, "‹").replace(/>/g, "›");
}

export class ZotWatchAddon {
  constructor() {
    this.id = null;
    this.version = null;
    this.rootURI = null;
    this.running = null;
    this.progress = null;
    this.cancelled = false;
    this.listeners = new Set();
    this.timers = [];
    this.storage = null;
    this.transport = null;
    this.cancellers = new Set();
    this.whitelist = "";
    this.pendingCount = 0;
    this.menuKey = null;
    this.windowHooks = new Map();
    this.pendingNotice = null;
  }

  // ------------------------------------------------------------- lifecycle

  async startup({ id, version, rootURI }) {
    this.id = id;
    this.version = version;
    this.rootURI = rootURI;
    await Zotero.initializationPromise;
    this.storage = createZoteroStorage();
    this.transport = createZoteroTransport({ cancellers: this.cancellers });
    await this.loadWhitelist();
    Zotero.ZotWatch = this;

    await Zotero.PreferencePanes.register({
      pluginID: id,
      id: PANE_ID,
      label: "ZotWatch",
      src: "chrome://zotwatch/content/preferences.xhtml",
      scripts: ["chrome://zotwatch/content/preferences.js"],
      stylesheets: ["chrome://zotwatch/content/preferences.css"],
      image: MENU_ICON,
      helpURL: HELP_URL,
    });

    this.registerMenus();
    for (const win of Zotero.getMainWindows()) {
      if (win.ZoteroPane) this.addToWindow(win);
    }
    await this.refreshPendingCount().catch((e) => Zotero.logError(e));
    this.startScheduler();
    log(`started ${version}${this.menuKey ? " (MenuManager)" : ""}`);
  }

  /**
   * Stop timers, abort a running pipeline and wait briefly for it to settle,
   * then remove all UI. On app shutdown only the run is stopped.
   */
  async shutdown({ appShutdown = false } = {}) {
    this.stopScheduler();
    this.cancelRun();
    if (this.running) await settleWithin(this.running, appShutdown ? QUIT_GRACE : SHUTDOWN_GRACE);
    if (appShutdown) return;

    if (this.menuKey) {
      Zotero.MenuManager?.unregisterMenu(this.menuKey);
      this.menuKey = null;
    }
    for (const win of Zotero.getMainWindows()) {
      if (win.ZoteroPane) this.removeFromWindow(win);
    }
    Services.wm.getMostRecentWindow(DIALOG_TYPE)?.close();
    this.listeners.clear();
    if (Zotero.ZotWatch === this) delete Zotero.ZotWatch;
  }

  // ------------------------------------------------------------------ menus

  /** Zotero 8+: register the Tools menu items through Zotero.MenuManager. */
  registerMenus() {
    if (typeof Zotero.MenuManager?.registerMenu !== "function") return;
    const key = Zotero.MenuManager.registerMenu({
      menuID: "zotwatch-tools",
      pluginID: this.id,
      target: "main/menubar/tools",
      menus: [
        { menuType: "separator" },
        {
          menuType: "menuitem",
          l10nID: "zotwatch-menu-open",
          l10nArgs: this.menuArgs(),
          icon: MENU_ICON,
          onShowing: (event, context) => context.setL10nArgs(this.menuArgs()),
          onCommand: () => this.openRecommendations(),
        },
        {
          menuType: "menuitem",
          l10nID: "zotwatch-menu-run",
          onShowing: (event, context) => context.setEnabled(!this.running),
          onCommand: () => this.runFromMenu(),
        },
      ],
    });
    this.menuKey = key || null;
    if (!key) log("MenuManager registration failed; using DOM menu items");
  }

  menuArgs() {
    return JSON.stringify({ count: this.pendingCount });
  }

  runFromMenu() {
    this.openRecommendations();
    this.runNow().catch(() => {});
  }

  addToWindow(win) {
    const doc = win.document;
    win.MozXULElement?.insertFTLIfNeeded(MAIN_WINDOW_FTL);
    if (this.menuKey || this.windowHooks.has(win)) return;

    // Zotero 7: inject the menu items into the Tools menu ourselves
    const popup = doc.getElementById("menu_ToolsPopup");
    if (!popup || doc.getElementById(DOM_MENU_IDS[1])) return;
    const separator = doc.createXULElement("menuseparator");
    separator.id = DOM_MENU_IDS[0];
    const open = doc.createXULElement("menuitem");
    open.id = DOM_MENU_IDS[1];
    open.setAttribute("data-l10n-id", "zotwatch-menu-open");
    open.setAttribute("data-l10n-args", this.menuArgs());
    open.addEventListener("command", () => this.openRecommendations());
    const run = doc.createXULElement("menuitem");
    run.id = DOM_MENU_IDS[2];
    run.setAttribute("data-l10n-id", "zotwatch-menu-run");
    run.addEventListener("command", () => this.runFromMenu());
    popup.append(separator, open, run);

    const onShowing = (event) => {
      if (event.target !== popup) return;
      open.setAttribute("data-l10n-args", this.menuArgs());
      run.disabled = !!this.running;
    };
    popup.addEventListener("popupshowing", onShowing);
    this.windowHooks.set(win, () => popup.removeEventListener("popupshowing", onShowing));
  }

  removeFromWindow(win) {
    const doc = win.document;
    this.windowHooks.get(win)?.();
    this.windowHooks.delete(win);
    for (const id of DOM_MENU_IDS) doc.getElementById(id)?.remove();
    doc.querySelector(`link[rel="localization"][href="${MAIN_WINDOW_FTL}"]`)?.remove();
    if (this.pendingNotice?.win === win) {
      win.removeEventListener("activate", this.pendingNotice.show);
      this.pendingNotice = null;
    }
  }

  // ------------------------------------------------------------- scheduler

  startScheduler() {
    this.stopScheduler();
    this.timers.push(setTimeout(() => this.maybeAutoRun(), FIRST_CHECK_DELAY));
    this.timers.push(setInterval(() => this.maybeAutoRun(), CHECK_INTERVAL));
  }

  stopScheduler() {
    for (const t of this.timers) {
      clearTimeout(t);
      clearInterval(t);
    }
    this.timers = [];
  }

  isRunDue(now = Date.now()) {
    const settings = readSettings();
    if (!settings.schedule.autoRun) return false;
    const lastRun = Number(getPref("schedule.lastRun")) || 0;
    const lastAttempt = Number(getPref("schedule.lastAttempt")) || 0;
    return now - lastRun >= settings.schedule.intervalHours * 3600000 && now - lastAttempt >= RETRY_AFTER_FAILURE;
  }

  maybeAutoRun() {
    if (this.running || !this.isRunDue()) return;
    this.runNow({ background: true }).catch(() => {});
  }

  // ------------------------------------------------------------------- run

  /** Current settings, with the journal whitelist that lives in a data file. */
  getSettings() {
    const settings = readSettings();
    settings.journals.whitelist = this.whitelist;
    return settings;
  }

  createEngine(settings = this.getSettings()) {
    return new ZotWatchEngine({
      http: this.transport,
      storage: this.storage,
      library: createZoteroLibrary(),
      settings,
      log,
    });
  }

  emit(event) {
    for (const listener of [...this.listeners]) {
      try {
        listener(event);
      } catch (e) {
        // Listeners from closed windows become dead objects; drop them
        this.listeners.delete(listener);
        if (!/dead object/i.test(String(e))) Zotero.logError(e);
      }
    }
  }

  addListener(fn) {
    this.listeners.add(fn);
  }

  removeListener(fn) {
    this.listeners.delete(fn);
  }

  /** Run the watch pipeline once (concurrent calls share the same run). */
  runNow({ background = false } = {}) {
    if (this.running) return this.running;
    this.cancelled = false;
    setPref("schedule.lastAttempt", String(Date.now()));
    const engine = this.createEngine();

    this.running = (async () => {
      this.emit({ type: "start" });
      try {
        const result = await engine.run({
          onProgress: (p) => {
            this.progress = p;
            this.emit({ type: "progress", progress: p });
          },
          isCancelled: () => this.cancelled,
        });
        setPref("schedule.lastRun", String(Date.now()));
        await this.refreshPendingCount().catch(() => {});
        this.emit({ type: "done", result });
        if (background && result.stats.newRecommendations > 0) {
          this.notify(
            "ZotWatch",
            `发现 ${result.stats.newRecommendations} 篇新推荐（${result.stats.mustRead} 篇必读）。在「工具 → ZotWatch 文献推荐」中查看。`,
            { sticky: true },
          );
        }
        return result;
      } catch (e) {
        if (e instanceof CancelledError || e?.cancelled || this.cancelled) {
          this.emit({ type: "cancelled" });
          return null;
        }
        Zotero.logError(e);
        this.emit({ type: "error", message: e.message });
        if (background) this.notify("ZotWatch 运行失败", e.message, { sticky: true });
        throw e;
      } finally {
        this.running = null;
        this.progress = null;
      }
    })();
    return this.running;
  }

  /** Stop the current run: set the flag checked between stages and abort open requests. */
  cancelRun() {
    if (!this.running) return;
    this.cancelled = true;
    for (const cancel of [...this.cancellers]) {
      try {
        cancel();
      } catch (e) {
        Zotero.logError(e);
      }
    }
    this.cancellers.clear();
  }

  /**
   * Show a toast. Zotero 10 attaches it to the main window instead of keeping
   * it on top, so sticky notices wait until Zotero is focused and stay until clicked.
   */
  notify(headline, text, { sticky = false } = {}) {
    const show = () => {
      this.pendingNotice = null;
      const pw = new Zotero.ProgressWindow({ closeOnClick: true });
      pw.changeHeadline(plainNoticeText(headline));
      pw.addDescription(plainNoticeText(text));
      pw.show();
      pw.startCloseTimer(sticky ? 20000 : 8000, sticky);
    };
    const win = Zotero.getMainWindow();
    if (sticky && win && !win.document.hasFocus()) {
      if (this.pendingNotice) this.pendingNotice.win.removeEventListener("activate", this.pendingNotice.show);
      this.pendingNotice = { win, show };
      win.addEventListener("activate", show, { once: true });
    } else {
      show();
    }
  }

  async refreshPendingCount() {
    const inbox = await this.createEngine().loadInbox();
    this.pendingCount = Object.values(inbox.works).filter((w) => w.status === "new").length;
    return this.pendingCount;
  }

  // ------------------------------------------------------ dialog API (inbox)

  openRecommendations() {
    const existing = Services.wm.getMostRecentWindow(DIALOG_TYPE);
    if (existing) {
      existing.focus();
      return existing;
    }
    const features = "chrome,dialog=no,resizable,centerscreen";
    const win = Zotero.getMainWindow();
    // On macOS Zotero can run without a main window
    return win
      ? win.openDialog(DIALOG_URL, "zotwatch-recommendations", features)
      : Services.ww.openWindow(null, DIALOG_URL, "zotwatch-recommendations", features, null);
  }

  openPreferences() {
    Zotero.Utilities.Internal.openPreferences(PANE_ID);
  }

  async getView() {
    const settings = this.getSettings();
    const inbox = await this.createEngine(settings).loadInbox();
    return {
      works: Object.values(inbox.works),
      lastRun: inbox.lastRun,
      running: !!this.running,
      progress: this.progress,
      llmEnabled: settings.llm.enabled && !!settings.llm.apiKey,
      provider: settings.embedding.provider,
      collection: settings.output.collection,
    };
  }

  /** Change the status of one or more recommendations with a single update event. */
  async setStatus(ids, status) {
    const list = Array.isArray(ids) ? ids : [ids];
    const engine = this.createEngine();
    await engine.patchWorks(list.map((id) => [id, { status, statusAt: Date.now() }]));
    await this.refreshPendingCount().catch(() => {});
    this.emit({ type: "inbox" });
  }

  async addToLibrary(id) {
    const settings = this.getSettings();
    const engine = this.createEngine(settings);
    const inbox = await engine.loadInbox();
    const work = inbox.works[id];
    if (!work) throw new Error("推荐条目不存在");
    const item = await addWorkToZotero(work, {
      libraryID: targetLibraryID(),
      collectionName: settings.output.collection,
      tag: settings.output.tag,
      withNote: settings.output.summaryNote,
      useTranslator: settings.output.useTranslator,
      log,
    });
    await engine.updateWork(id, { status: "added", itemID: item.id, itemKey: item.key });
    await this.refreshPendingCount().catch(() => {});
    this.emit({ type: "inbox" });
    return item.id;
  }

  async summarize(id) {
    const summary = await this.createEngine().summarize(id);
    this.emit({ type: "inbox" });
    return summary;
  }

  async showItem(itemID) {
    const win = Zotero.getMainWindow();
    if (!win) return;
    try {
      await win.ZoteroPane.selectItem(itemID);
      win.focus();
    } catch (e) {
      Zotero.logError(e);
    }
  }

  openUrl(url) {
    if (/^https?:\/\//i.test(url || "")) Zotero.launchURL(url);
  }

  // ------------------------------------------------------ journal whitelist

  /**
   * The whitelist can exceed the 4 KB advisory size of a pref, so it lives in
   * <data dir>/zotwatch/journals.csv. Older versions stored it in a pref; migrate once.
   */
  async loadWhitelist() {
    const legacy = getPref("journals.whitelist");
    const stored = await this.storage.readText(WHITELIST_FILE);
    if (stored === null && typeof legacy === "string" && legacy.trim()) {
      await this.storage.writeText(WHITELIST_FILE, legacy);
      this.whitelist = legacy;
    } else {
      this.whitelist = stored || "";
    }
    if (legacy) setPref("journals.whitelist", "");
    return this.whitelist;
  }

  getJournalWhitelist() {
    return this.whitelist;
  }

  async setJournalWhitelist(text) {
    const value = String(text ?? "");
    if (value === this.whitelist) return;
    this.whitelist = value;
    await this.storage.writeText(WHITELIST_FILE, value);
  }

  // ------------------------------------------------- preference pane helpers

  async testConnections() {
    const settings = this.getSettings();
    const lines = [];
    try {
      const embedder = createEmbedder(settings.embedding, this.transport, { retry: { retries: 0 } });
      if (embedder.requiresFit) embedder.fit(["test"]);
      const [vec] = await embedder.embed(["ZotWatch connection test"]);
      lines.push(`✅ 向量化（${settings.embedding.provider}）：正常，维度 ${vec.length}`);
    } catch (e) {
      lines.push(`❌ 向量化（${settings.embedding.provider}）：${e.message}`);
    }
    if (settings.llm.enabled) {
      try {
        const reply = await createLlmClient(settings.llm, this.transport, { retry: { retries: 0 } }).chat("请只回复：OK", { maxTokens: 20 });
        lines.push(`✅ 大模型（${settings.llm.model}）：${reply.trim().slice(0, 30)}`);
      } catch (e) {
        lines.push(`❌ 大模型（${settings.llm.model}）：${e.message}`);
      }
    } else {
      lines.push("ℹ️ 大模型：未启用");
    }
    try {
      const res = await this.transport.request({ url: "https://api.crossref.org/works?rows=0", timeout: 20000 });
      lines.push(res.status === 200 ? "✅ Crossref：可访问" : `❌ Crossref：HTTP ${res.status}`);
    } catch (e) {
      lines.push(`❌ Crossref：${e.message}`);
    }
    return lines.join("\n");
  }

  async getStatus() {
    const items = await createZoteroLibrary().getItems();
    const withAbstract = items.filter((i) => i.abstract).length;
    const inbox = await this.createEngine().loadInbox();
    const pending = Object.values(inbox.works).filter((w) => w.status === "new").length;
    const last = inbox.lastRun;
    const library = Zotero.Libraries.get(profileLibraryID());
    return [
      `文库：${library?.name || "我的文库"}，${items.length} 个条目（含摘要 ${withAbstract}）`,
      `待查看推荐：${pending}`,
      last ? `上次运行：${new Date(last.runAt).toLocaleString()}，推荐 ${last.stats.recommended} 篇，抓取 ${last.stats.fetched} 篇` : "尚未运行",
      `数据目录：${this.storage.dir}`,
    ].join("\n");
  }

  /** Whitelist CSV with the top journals of the library merged into the current list. */
  async suggestJournals() {
    const settings = this.getSettings();
    const engine = this.createEngine({ ...settings, journals: { ...settings.journals, autoFromLibrary: true } });
    const items = await createZoteroLibrary().getItems();
    const { venues } = await engine.resolveJournals(items);
    const merged = parseWhitelist(settings.journals.whitelist);
    for (const v of venues.length ? venues : libraryVenues(items).slice(0, settings.journals.autoTopN)) {
      for (const issn of v.issns) {
        if (!merged.has(issn)) merged.set(issn, { issn, title: v.title, category: `library:${v.count}`, impactFactor: null });
      }
    }
    return serializeWhitelist([...merged.values()]);
  }

  async loadPresetJournals() {
    return Zotero.File.getResourceAsync("chrome://zotwatch/content/presets/geoscience-journals.csv");
  }

  async resetData() {
    if (this.running) throw new Error("正在运行，请稍后再试");
    await this.storage.clear();
    // The whitelist is user data, not cache: keep it
    if (this.whitelist) await this.storage.writeText(WHITELIST_FILE, this.whitelist);
    setPref("schedule.lastRun", "");
    this.pendingCount = 0;
    this.emit({ type: "inbox" });
  }
}
