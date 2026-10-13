// Plugin object: lifecycle, Tools menu, scheduler, and the API used by the
// recommendations window and the preference pane (exposed as Zotero.ZotWatch).

/* global Zotero */

import { libraryVenues, parseWhitelist, serializeWhitelist } from "../core/journals.js";
import { CancelledError, ZotWatchEngine } from "../engine.js";
import { createEmbedder } from "../embedding/providers.js";
import { createLlmClient } from "../llm/llm.js";
import { addWorkToZotero } from "./items.js";
import { createZoteroLibrary, createZoteroStorage, getPref, log, profileLibraryID, readSettings, setPref, zoteroTransport } from "./env.js";

const PANE_ID = "zotwatch-prefpane";
const DIALOG_URL = "chrome://zotwatch/content/recommendations.xhtml";
const FIRST_CHECK_DELAY = 3 * 60 * 1000;
const CHECK_INTERVAL = 30 * 60 * 1000;
const RETRY_AFTER_FAILURE = 2 * 3600 * 1000;
const MENU_IDS = ["zotwatch-menu-separator", "zotwatch-menu-open", "zotwatch-menu-run"];

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
    this.dialog = null;
    this.storage = null;
  }

  // ------------------------------------------------------------- lifecycle

  async startup({ id, version, rootURI }) {
    this.id = id;
    this.version = version;
    this.rootURI = rootURI;
    await Zotero.initializationPromise;
    this.storage = createZoteroStorage();
    Zotero.ZotWatch = this;

    await Zotero.PreferencePanes.register({
      pluginID: id,
      id: PANE_ID,
      label: "ZotWatch",
      src: "chrome://zotwatch/content/preferences.xhtml",
      scripts: ["chrome://zotwatch/content/preferences.js"],
      stylesheets: ["chrome://zotwatch/content/preferences.css"],
      image: "chrome://zotwatch/content/icons/icon.svg",
    });

    for (const win of Zotero.getMainWindows()) {
      if (win.ZoteroPane) this.addToWindow(win);
    }
    this.startScheduler();
    log(`started ${version}`);
  }

  shutdown() {
    this.stopScheduler();
    this.cancelled = true;
    for (const win of Zotero.getMainWindows()) {
      if (win.ZoteroPane) this.removeFromWindow(win);
    }
    if (this.dialog && !this.dialog.closed) this.dialog.close();
    this.listeners.clear();
    if (Zotero.ZotWatch === this) delete Zotero.ZotWatch;
  }

  addToWindow(win) {
    const doc = win.document;
    const popup = doc.getElementById("menu_ToolsPopup");
    if (!popup || doc.getElementById(MENU_IDS[1])) return;

    const separator = doc.createXULElement("menuseparator");
    separator.id = MENU_IDS[0];
    const open = doc.createXULElement("menuitem");
    open.id = MENU_IDS[1];
    open.setAttribute("label", "ZotWatch 文献推荐…");
    open.addEventListener("command", () => this.openRecommendations());
    const run = doc.createXULElement("menuitem");
    run.id = MENU_IDS[2];
    run.setAttribute("label", "ZotWatch：立即检查新文献");
    run.addEventListener("command", () => {
      this.openRecommendations();
      this.runNow().catch(() => {});
    });
    popup.append(separator, open, run);
  }

  removeFromWindow(win) {
    for (const id of MENU_IDS) win.document.getElementById(id)?.remove();
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

  createEngine(settings = readSettings()) {
    return new ZotWatchEngine({
      http: zoteroTransport,
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
        Zotero.logError(e);
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
        this.emit({ type: "done", result });
        if (background && result.stats.newRecommendations > 0) {
          this.notify(
            "ZotWatch",
            `发现 ${result.stats.newRecommendations} 篇新推荐（${result.stats.mustRead} 篇必读）。在「工具 → ZotWatch 文献推荐」中查看。`,
          );
        }
        return result;
      } catch (e) {
        if (e instanceof CancelledError) {
          this.emit({ type: "cancelled" });
          return null;
        }
        Zotero.logError(e);
        this.emit({ type: "error", message: e.message });
        if (background) this.notify("ZotWatch 运行失败", e.message);
        throw e;
      } finally {
        this.running = null;
        this.progress = null;
      }
    })();
    return this.running;
  }

  cancelRun() {
    if (this.running) this.cancelled = true;
  }

  notify(headline, text) {
    const pw = new Zotero.ProgressWindow({ closeOnClick: true });
    pw.changeHeadline(headline);
    pw.addDescription(text);
    pw.show();
    pw.startCloseTimer(10000);
  }

  // ------------------------------------------------------ dialog API (inbox)

  openRecommendations() {
    if (this.dialog && !this.dialog.closed) {
      this.dialog.focus();
      return this.dialog;
    }
    const win = Zotero.getMainWindow();
    this.dialog = win.openDialog(DIALOG_URL, "zotwatch-recommendations", "chrome,dialog=no,resizable,centerscreen,width=1040,height=800");
    return this.dialog;
  }

  openPreferences() {
    Zotero.Utilities.Internal.openPreferences(PANE_ID);
  }

  async getView() {
    const settings = readSettings();
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

  async setStatus(id, status) {
    const work = await this.createEngine().updateWork(id, { status });
    this.emit({ type: "inbox" });
    return work;
  }

  async addToLibrary(id) {
    const settings = readSettings();
    const engine = this.createEngine(settings);
    const inbox = await engine.loadInbox();
    const work = inbox.works[id];
    if (!work) throw new Error("推荐条目不存在");
    const libraryID = profileLibraryID();
    const item = await addWorkToZotero(work, {
      libraryID,
      collectionName: settings.output.collection,
      tag: settings.output.tag,
      withNote: settings.output.summaryNote,
      useTranslator: settings.output.useTranslator,
      log,
    });
    await engine.updateWork(id, { status: "added", itemID: item.id, itemKey: item.key });
    this.emit({ type: "inbox" });
    return item.id;
  }

  async summarize(id) {
    const summary = await this.createEngine().summarize(id);
    this.emit({ type: "inbox" });
    return summary;
  }

  showItem(itemID) {
    const win = Zotero.getMainWindow();
    if (!win) return;
    win.ZoteroPane.selectItem(itemID);
    win.focus();
  }

  openUrl(url) {
    if (/^https?:\/\//i.test(url || "")) Zotero.launchURL(url);
  }

  // ------------------------------------------------- preference pane helpers

  async testConnections() {
    const settings = readSettings();
    const lines = [];
    try {
      const embedder = createEmbedder(settings.embedding, zoteroTransport, { retry: { retries: 0 } });
      if (embedder.requiresFit) embedder.fit(["test"]);
      const [vec] = await embedder.embed(["ZotWatch connection test"]);
      lines.push(`✅ 向量化（${settings.embedding.provider}）：正常，维度 ${vec.length}`);
    } catch (e) {
      lines.push(`❌ 向量化（${settings.embedding.provider}）：${e.message}`);
    }
    if (settings.llm.enabled) {
      try {
        const reply = await createLlmClient(settings.llm, zoteroTransport, { retry: { retries: 0 } }).chat("请只回复：OK", { maxTokens: 20 });
        lines.push(`✅ 大模型（${settings.llm.model}）：${reply.trim().slice(0, 30)}`);
      } catch (e) {
        lines.push(`❌ 大模型（${settings.llm.model}）：${e.message}`);
      }
    } else {
      lines.push("ℹ️ 大模型：未启用");
    }
    try {
      const xhr = await zoteroTransport.request({ url: "https://api.crossref.org/works?rows=0", timeout: 20000 });
      lines.push(xhr.status === 200 ? "✅ Crossref：可访问" : `❌ Crossref：HTTP ${xhr.status}`);
    } catch (e) {
      lines.push(`❌ Crossref：${e.message}`);
    }
    return lines.join("\n");
  }

  async getStatus() {
    const settings = readSettings();
    const items = await createZoteroLibrary().getItems();
    const withAbstract = items.filter((i) => i.abstract).length;
    const inbox = await this.createEngine(settings).loadInbox();
    const pending = Object.values(inbox.works).filter((w) => w.status === "new").length;
    const last = inbox.lastRun;
    return [
      `文库条目：${items.length}（含摘要 ${withAbstract}）`,
      `待查看推荐：${pending}`,
      last ? `上次运行：${new Date(last.runAt).toLocaleString()}，推荐 ${last.stats.recommended} 篇，抓取 ${last.stats.fetched} 篇` : "尚未运行",
      `数据目录：${this.storage.dir}`,
    ].join("\n");
  }

  /** Whitelist CSV with the top journals of the library merged into the current list. */
  async suggestJournals() {
    const settings = readSettings();
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
    setPref("schedule.lastRun", "");
    this.emit({ type: "inbox" });
  }
}
