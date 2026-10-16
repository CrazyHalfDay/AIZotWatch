// Plugin object: lifecycle, Tools menu, scheduler, and the API used by the
// ZotWatch main window and the preference pane (exposed as Zotero.ZotWatch).

/* global Zotero, Services */

import { createCancelSignal } from "../core/http.js";
import { parseFollowedAuthors } from "../core/authors.js";
import { libraryVenues, parseWhitelist, serializeWhitelist, summarizeWhitelist } from "../core/journals.js";
import { countByTrack } from "../core/tracks.js";
import { CancelledError, HANDLED_MIN_DAYS, ZotWatchEngine } from "../engine.js";
import { EMBEDDING_PRESETS, createEmbedder } from "../embedding/providers.js";
import { createLlmClient } from "../llm/llm.js";
import { autoValues } from "../settings.js";
import { checkPublisherKeyRows } from "../sources/abstracts.js";
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
// README of the installed release (each release is tagged zotero-plugin-v<version>)
const helpURL = (version) => `https://github.com/CrazyHalfDay/AIZotWatch/blob/zotero-plugin-v${version}/zotero-plugin/README.md`;
const DIALOG_URL = "chrome://zotwatch/content/main.xhtml";
const DIALOG_TYPE = "zotwatch:main";
const PAGES = new Set(["overview", "inbox", "settings"]);
const MENU_ICON = "chrome://zotwatch/content/icons/menu.svg";
const MAIN_WINDOW_FTL = "zotwatch-mainWindow.ftl";
const WHITELIST_FILE = "journals.csv";
const FIRST_CHECK_DELAY = 3 * 60 * 1000;
const CHECK_INTERVAL = 30 * 60 * 1000;
const RETRY_AFTER_FAILURE = 2 * 3600 * 1000;
const SHUTDOWN_GRACE = 5000;
const QUIT_GRACE = 1500;
// Zotero 9 registers plugin Fluent files with a per-file fallback. On 7/8 a
// main-window FTL missing for the UI locale would switch the whole window to
// English, so older versions get plain-text menu labels instead.
const MIN_FLUENT_MENU_VERSION = 9;
const DOM_MENU_IDS = ["zotwatch-menu-separator", "zotwatch-menu-open", "zotwatch-menu-run"];

/** Wait for `promise` to settle, but at most `ms` milliseconds (leaves no timer behind). */
async function settleWithin(promise, ms) {
  let timer;
  await Promise.race([promise.catch(() => {}), new Promise((resolve) => (timer = setTimeout(resolve, ms)))]);
  clearTimeout(timer);
}

/** Plain-text menu labels for Zotero 7/8 (no Fluent in the main window there). */
export function plainMenuLabels(count, locale) {
  if (/^zh/i.test(locale || "")) {
    return {
      open: count > 0 ? `ZotWatch 文献推荐（${count} 篇待查看）…` : "ZotWatch 文献推荐…",
      run: "ZotWatch：立即检查新文献",
    };
  }
  return {
    open: count > 0 ? `ZotWatch Recommendations (${count} new)…` : "ZotWatch Recommendations…",
    run: "ZotWatch: Check for New Papers",
  };
}

/** Background notice for a run that found new recommendations (all tracks). */
export function newRecommendationsNotice(stats) {
  const { flagship = 0, followed = 0 } = stats.newByLabel || {};
  const tracks = `${flagship ? `，顶刊地学 ${flagship} 篇` : ""}${followed ? `，关注作者 ${followed} 篇` : ""}`;
  // Must-read papers among the new ones only (stats.mustRead also counts earlier ones)
  const mustRead = stats.newMustRead ?? 0;
  return `发现 ${stats.newRecommendations} 篇新推荐${mustRead ? `（${mustRead} 篇必读）` : ""}${tracks}。在「工具 → ZotWatch 文献推荐」中查看。`;
}

/** Status line for the followed-authors setting (so unparseable lines show up before a run). */
export function followedStatusLine({ authors, invalid, daysBack }) {
  const names = authors.map((a) => a.name);
  const list = names.length > 6 ? `${names.slice(0, 6).join("、")} 等` : names.join("、");
  const head = authors.length ? `关注作者：${authors.length} 位（${list}），检查最近 ${daysBack} 天` : "关注作者：未填写有效的作者";
  return invalid.length ? `${head}；${invalid.length} 行无法识别：${invalid[0]}` : head;
}

/** Status line for the flagship track: its settings and the last run's funnel (stats.flagship). */
export function flagshipStatusLine(cfg, stats) {
  const head = cfg.issns.length
    ? `顶刊地学速览：${cfg.issns.length} 个 ISSN，接受阈值 ${cfg.minScore.toFixed(2)}，灰区下限 ${cfg.grayLow.toFixed(2)}`
    : "顶刊地学速览：未填写有效的 ISSN";
  if (!stats) return head;
  const judged = stats.judged ? `，大模型复核 ${stats.judged} 篇` : "";
  return `${head}；上次运行 ${stats.candidates} 篇候选，${stats.recent} 篇有摘要且在时间范围内，入选 ${stats.recommended} 篇（灰区 ${stats.gray} 篇${judged}）`;
}

/** Status line for the abstract fallback of the last run (stats.abstracts). */
export function abstractsStatusLine(counts) {
  const sources = [
    ["缓存", counts.cacheHits],
    ["Crossref", counts.crossref],
    ["Elsevier", counts.elsevier],
    ["Springer Nature", counts.springer],
    ["OpenAlex", counts.openalex],
  ].filter(([, n]) => n > 0);
  const detail = sources.length ? `（${sources.map(([name, n]) => `${name} ${n}`).join("，")}）` : "";
  return `摘要补全：上次运行补全 ${counts.filled || 0} / ${counts.missing || 0} 篇${detail}`;
}

/** Status line for the research-direction labels: the largest directions and their sizes. */
export function topicsStatusLine(topics, max = 8) {
  const shown = topics.slice(0, max).map((t) => `${t.label}（${t.size} 篇）`).join("、");
  return `研究方向：${shown}${topics.length > max ? ` 等 ${topics.length} 个` : ""}`;
}

/**
 * Route of the main window: "overview", "inbox", "settings" or
 * "settings/<section>"; anything else falls back to "overview".
 */
export function normalizePage(page) {
  const [name, section] = String(page ?? "").replace(/^#/, "").split("/");
  if (!PAGES.has(name)) return "overview";
  return name === "settings" && /^[a-z]+$/.test(section || "") ? `settings/${section}` : name;
}

/**
 * When the scheduler starts the next automatic run (ms since the epoch), or
 * null when automatic runs are off. Same rule as isRunDue: the interval since
 * the last successful run and the retry delay since the last attempt must both
 * have passed (a past time means "at the next check", which happens every 30 minutes).
 */
export function nextRunAt({ autoRun, intervalHours, lastRun, lastAttempt }, retryMs = RETRY_AFTER_FAILURE) {
  if (!autoRun) return null;
  return Math.max((lastRun || 0) + intervalHours * 3600000, (lastAttempt || 0) + retryMs);
}

/** Connection check result as a text line ("✅ Crossref：可访问"). */
export function connectionLine(row) {
  const icon = row.ok === true ? "✅" : row.ok === false ? "❌" : "ℹ️";
  return `${icon} ${row.name}${row.target ? `（${row.target}）` : ""}：${row.detail}`;
}

/** First message of the run errors matching `test`, or null. */
function errorMessage(errors, test) {
  return (errors || []).find(test)?.message || null;
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
    this.runSignal = null;
    this.progress = null;
    this.listeners = new Set();
    this.timers = [];
    this.storage = null;
    // Requests made for UI actions (manual summary, connection test, ...) use
    // their own signal, so cancelling a run never aborts them
    this.uiSignal = null;
    this.transport = null;
    this.shutdownGraceMs = SHUTDOWN_GRACE;
    this.whitelist = "";
    this.whitelistError = null;
    this.pendingCount = 0;
    this.menuKey = null;
    this.windowHooks = new Map();
    this.pendingNotice = null;
    // Message and time of the last failed run (cleared when a run starts)
    this.lastError = null;
    // Last page shown in the main window during this session
    this.lastPage = null;
  }

  // ------------------------------------------------------------- lifecycle

  async startup({ id, version, rootURI }) {
    this.id = id;
    this.version = version;
    this.rootURI = rootURI;
    await Zotero.initializationPromise;
    this.storage = createZoteroStorage();
    this.uiSignal = createCancelSignal();
    this.transport = createZoteroTransport({ signal: this.uiSignal });
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
      helpURL: helpURL(version),
    });

    this.registerMenus();
    for (const win of Zotero.getMainWindows()) {
      if (win.ZoteroPane) this.addToWindow(win);
    }
    await this.refreshPendingCount().catch((e) => Zotero.logError(e));
    this.startScheduler();
    log(`started ${version} on Zotero ${Zotero.version}${this.menuKey ? " (MenuManager)" : ""}`);
  }

  /**
   * Stop timers, abort a running pipeline and wait briefly for it to settle,
   * then remove all UI. On app shutdown only the run is stopped.
   */
  async shutdown({ appShutdown = false } = {}) {
    this.stopScheduler();
    this.cancelRun();
    this.uiSignal?.cancel();
    if (this.running) await settleWithin(this.running, appShutdown ? QUIT_GRACE : this.shutdownGraceMs);
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

  /** Zotero 9+: register the Tools menu items through Zotero.MenuManager. */
  registerMenus() {
    if (typeof Zotero.MenuManager?.registerMenu !== "function") return;
    if (Number.parseInt(Zotero.version, 10) < MIN_FLUENT_MENU_VERSION) return;
    const key = Zotero.MenuManager.registerMenu({
      menuID: "zotwatch-tools",
      pluginID: this.id,
      target: "main/menubar/tools",
      menus: [
        { menuType: "separator" },
        {
          menuType: "menuitem",
          l10nID: "zotwatch-menu-open",
          // l10nArgs is set in onShowing (Zotero 8 only accepted objects here)
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
    if (this.menuKey) {
      // MenuManager items are labelled through this Fluent file
      win.MozXULElement?.insertFTLIfNeeded(MAIN_WINDOW_FTL);
      return;
    }
    if (this.windowHooks.has(win)) return;

    // Zotero 7/8: inject plain-text menu items into the Tools menu ourselves
    const popup = doc.getElementById("menu_ToolsPopup");
    if (!popup || doc.getElementById(DOM_MENU_IDS[1])) return;
    const labels = () => plainMenuLabels(this.pendingCount, Services.locale?.appLocaleAsBCP47);
    const separator = doc.createXULElement("menuseparator");
    separator.id = DOM_MENU_IDS[0];
    const open = doc.createXULElement("menuitem");
    open.id = DOM_MENU_IDS[1];
    open.setAttribute("label", labels().open);
    open.addEventListener("command", () => this.openRecommendations());
    const run = doc.createXULElement("menuitem");
    run.id = DOM_MENU_IDS[2];
    run.setAttribute("label", labels().run);
    run.addEventListener("command", () => this.runFromMenu());
    popup.append(separator, open, run);

    const onShowing = (event) => {
      if (event.target !== popup) return;
      open.setAttribute("label", labels().open);
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

  /** Automatic-run settings and timestamps (ms, null when never) with the next scheduled run. */
  getSchedule(settings = readSettings()) {
    const lastRun = Number(getPref("schedule.lastRun")) || 0;
    const lastAttempt = Number(getPref("schedule.lastAttempt")) || 0;
    const { autoRun, intervalHours } = settings.schedule;
    return {
      autoRun,
      intervalHours,
      lastRun: lastRun || null,
      lastAttempt: lastAttempt || null,
      nextRunAt: nextRunAt({ autoRun, intervalHours, lastRun, lastAttempt }),
    };
  }

  isRunDue(now = Date.now()) {
    const next = this.getSchedule().nextRunAt;
    return next !== null && now >= next;
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

  createEngine(settings = this.getSettings(), http = this.transport) {
    return new ZotWatchEngine({
      http,
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
    setPref("schedule.lastAttempt", String(Date.now()));
    // Each run gets its own signal: cancelling it aborts only this run's requests
    const signal = createCancelSignal();
    this.runSignal = signal;
    const engine = this.createEngine(this.getSettings(), createZoteroTransport({ signal }));

    this.running = (async () => {
      this.lastError = null;
      this.emit({ type: "start" });
      try {
        const result = await engine.run({
          onProgress: (p) => {
            this.progress = p;
            this.emit({ type: "progress", progress: p });
          },
          isCancelled: () => signal.cancelled,
        });
        setPref("schedule.lastRun", String(Date.now()));
        await this.refreshPendingCount().catch(() => {});
        this.emit({ type: "done", result });
        if (background && result.stats.newRecommendations > 0) {
          this.notify("ZotWatch", newRecommendationsNotice(result.stats), { sticky: true });
        }
        return result;
      } catch (e) {
        if (e instanceof CancelledError || e?.cancelled || signal.cancelled) {
          this.emit({ type: "cancelled" });
          return null;
        }
        Zotero.logError(e);
        this.lastError = { message: e.message, at: Date.now() };
        this.emit({ type: "error", message: e.message });
        if (background) this.notify("ZotWatch 运行失败", e.message, { sticky: true });
        throw e;
      } finally {
        this.running = null;
        this.runSignal = null;
        this.progress = null;
        // Results may have been merged into the inbox before a cancel or failure
        await this.refreshPendingCount().catch(() => {});
      }
    })();
    return this.running;
  }

  /** Stop the current run: abort its open requests, end retry waits and stop at the next stage. */
  cancelRun() {
    this.runSignal?.cancel();
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

  // ------------------------------------------------------- main window API

  /**
   * Open the ZotWatch window on `page` ("overview", "inbox", "settings" or
   * "settings/<section>"; default: the page last shown this session), or
   * focus the open one and switch it to that page.
   */
  openWindow(page) {
    const route = normalizePage(page || this.lastPage || "overview");
    const existing = Services.wm.getMostRecentWindow(DIALOG_TYPE);
    if (existing) {
      existing.focus();
      existing.ZotWatchMain?.navigate(route);
      return existing;
    }
    const url = `${DIALOG_URL}#${route}`;
    const features = "chrome,dialog=no,resizable,centerscreen";
    const win = Zotero.getMainWindow();
    // On macOS Zotero can run without a main window
    return win
      ? win.openDialog(url, "zotwatch-main", features)
      : Services.ww.openWindow(null, url, "zotwatch-main", features, null);
  }

  openRecommendations() {
    return this.openWindow("inbox");
  }

  /** Zotero's settings window on the ZotWatch pane (the main window has its own settings page). */
  openPreferences() {
    Zotero.Utilities.Internal.openPreferences(PANE_ID);
  }

  /** Path of the data directory (<Zotero data dir>/zotwatch), null before startup. */
  getDataDir() {
    return this.storage?.dir || null;
  }

  /** Show the data directory (<Zotero data dir>/zotwatch) in the file manager. */
  async revealDataDir() {
    const dir = this.storage?.dir;
    if (!dir) return false;
    // Zotero.File.reveal throws for a missing path (e.g. right after a reset)
    await this.storage.ensureDir?.();
    if (typeof Zotero.File?.reveal === "function") await Zotero.File.reveal(dir);
    else Zotero.launchFile?.(dir);
    return true;
  }

  async getView() {
    const settings = this.getSettings();
    const inbox = await this.createEngine(settings).loadInbox();
    return {
      works: Object.values(inbox.works),
      lastRun: inbox.lastRun,
      history: inbox.history || [],
      running: !!this.running,
      progress: this.progress,
      lastError: this.lastError,
      llmEnabled: settings.llm.enabled && !!settings.llm.apiKey,
      provider: settings.embedding.provider,
      collection: settings.output.collection,
      version: this.version,
      schedule: this.getSchedule(settings),
    };
  }

  /** Display name of the embedding service ("阿里云 DashScope · text-embedding-v4"). */
  embeddingInfo(settings = this.getSettings()) {
    const { provider } = settings.embedding;
    const preset = EMBEDDING_PRESETS[provider];
    if (!preset) return { provider, model: "", label: "本地模式", local: true };
    const model = settings.embedding.model || preset.model;
    return { provider, model, label: `${preset.label} · ${model}`, local: false };
  }

  /**
   * Everything the overview page shows. Reads the library (item counts), so
   * call it when the page loads, not on every progress event.
   */
  async getDashboard() {
    const settings = this.getSettings();
    const engine = this.createEngine(settings);
    const [items, inbox, allTopics] = await Promise.all([
      createZoteroLibrary().getItems(),
      engine.loadInbox(),
      settings.llm.topicLabels ? engine.getTopics() : [],
    ]);
    const works = Object.values(inbox.works);
    const pendingWorks = works.filter((w) => w.status === "new");
    const last = inbox.lastRun;
    const stats = last?.stats || {};
    const errors = last?.errors || [];
    const perSource = stats.perSource || {};
    const library = Zotero.Libraries.get(profileLibraryID());
    const journals = [...parseWhitelist(this.whitelist).values()];
    const { followed, flagship, sources, abstracts, llm } = settings;
    const embedding = this.embeddingInfo(settings);
    const sourceCount = (name) => (Number.isFinite(perSource[name]) ? perSource[name] : null);
    const pendingBy = (name) => pendingWorks.filter((w) => (w.followedAuthors || []).includes(name)).length;
    const abstractTiers = abstracts.crossref || abstracts.openalex || !!abstracts.elsevierKey || !!abstracts.springerKey;

    return {
      version: this.version,
      library: { name: library?.name || "我的文库", items: items.length, withAbstract: items.filter((i) => i.abstract).length },
      embedding,
      llm: { enabled: llm.enabled, configured: !!llm.apiKey, model: llm.model, topicLabels: llm.topicLabels },
      schedule: this.getSchedule(settings),
      running: !!this.running,
      pending: { total: pendingWorks.length, ...countByTrack(pendingWorks), mustRead: pendingWorks.filter((w) => w.label === "must_read").length },
      added: works.filter((w) => w.status === "added").length,
      // Added works stay in the inbox this long (the 已加入 tile's window)
      addedDays: Math.max(settings.inbox.retentionDays, HANDLED_MIN_DAYS),
      scoring: { recentDays: settings.scoring.recentDays, requireAbstract: settings.scoring.requireAbstract },
      lastRun: last,
      lastError: this.lastError,
      history: inbox.history || [],
      topics: allTopics.slice(0, 10),
      topicCount: allTopics.length,
      clusters: stats.topics?.clusters ?? stats.profile?.clusters ?? null,
      followed: {
        enabled: followed.enabled,
        daysBack: followed.daysBack,
        authors: followed.authors.map((a) => ({ name: a.name, orcid: a.orcid || null, openalexId: a.openalexId || null, pending: pendingBy(a.name) })),
        invalid: followed.invalid,
      },
      flagship: { enabled: flagship.enabled, issnCount: flagship.issns.length, stats: stats.flagship || null },
      journals: {
        count: journals.length,
        withImpactFactor: journals.filter((j) => j.impactFactor !== null).length,
        autoFromLibrary: settings.journals.autoFromLibrary,
        error: this.whitelistError,
      },
      sources: [
        {
          key: "crossref",
          name: "Crossref",
          enabled: sources.crossref.enabled,
          count: sourceCount("Crossref"),
          detail: stats.journals ? `${stats.journals} 个期刊` : null,
          error: errorMessage(errors, (e) => e.stage === "fetch" && e.source === "Crossref"),
        },
        {
          key: "flagship",
          name: "顶刊",
          enabled: flagship.enabled,
          count: sourceCount("顶刊"),
          detail: `${flagship.issns.length} 个 ISSN`,
          error: errorMessage(errors, (e) => e.stage === "flagship" || (e.stage === "fetch" && e.source === "Crossref 顶刊")),
        },
        { key: "arxiv", name: "arXiv", enabled: sources.arxiv.enabled, count: sourceCount("arXiv"), detail: sources.arxiv.categories.join(", ") || null, error: errorMessage(errors, (e) => e.stage === "fetch" && e.source === "arXiv") },
        { key: "eartharxiv", name: "EarthArXiv", enabled: sources.eartharxiv.enabled, count: sourceCount("EarthArXiv"), detail: null, error: errorMessage(errors, (e) => e.stage === "fetch" && e.source === "EarthArXiv") },
        {
          key: "followed",
          name: "关注作者",
          enabled: followed.enabled,
          count: stats.followed ? stats.followed.fetched : null,
          detail: `OpenAlex，${followed.authors.length} 位`,
          error: errorMessage(errors, (e) => e.stage === "followed"),
        },
        {
          key: "abstracts",
          name: "摘要补全",
          enabled: abstractTiers,
          count: stats.abstracts ? stats.abstracts.filled ?? 0 : null,
          detail: stats.abstracts ? `上次补全 ${stats.abstracts.filled ?? 0} / ${stats.abstracts.missing ?? 0}` : null,
          error: errorMessage(errors, (e) => e.stage === "enrich"),
        },
        { key: "embedding", name: "向量化", enabled: true, count: null, detail: embedding.label, error: null },
        {
          key: "llm",
          name: "大模型",
          enabled: llm.enabled && !!llm.apiKey,
          count: null,
          detail: llm.enabled ? (llm.apiKey ? llm.model : "未填写 API Key") : "未启用",
          error: errorMessage(errors, (e) => e.stage === "llm"),
        },
      ],
      abstracts: stats.abstracts || null,
      dataDir: this.storage.dir,
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
    this.whitelistError = null;
    let stored = null;
    try {
      stored = await this.storage.readText(WHITELIST_FILE);
    } catch (e) {
      // e.g. the CSV was re-saved as GBK/ANSI in Excel; keep the plugin usable
      this.whitelistError = `无法读取 ${WHITELIST_FILE}（请以 UTF-8 编码保存）：${e.message || e}`;
      Zotero.logError(e);
    }
    try {
      if (stored === null && !this.whitelistError && typeof legacy === "string" && legacy.trim()) {
        await this.storage.writeText(WHITELIST_FILE, legacy);
        stored = legacy;
      }
      if (legacy && !this.whitelistError) setPref("journals.whitelist", "");
    } catch (e) {
      Zotero.logError(e);
    }
    this.whitelist = stored ?? (typeof legacy === "string" ? legacy : "");
    return this.whitelist;
  }

  getJournalWhitelist() {
    return this.whitelist;
  }

  async setJournalWhitelist(text) {
    const value = String(text ?? "");
    if (value === this.whitelist && !this.whitelistError) return;
    this.whitelist = value;
    await this.storage.writeText(WHITELIST_FILE, value);
    this.whitelistError = null;
  }

  // ---------------------------------------------------------- settings helpers

  /**
   * Check the configured services: [{key, name, target, ok, detail}], where ok
   * is true, false, or null for a service that is not enabled. `only` limits
   * the checks to some keys ("embedding", "llm", "crossref", "elsevier", "springer");
   * publisher keys are only checked when filled in.
   */
  async checkConnections(only = null) {
    const settings = this.getSettings();
    const want = (key) => !only || only.includes(key);
    const rows = [];
    if (want("embedding")) {
      const row = { key: "embedding", name: "向量化", target: settings.embedding.provider, ok: false, detail: "" };
      try {
        const embedder = createEmbedder(settings.embedding, this.transport, { retry: { retries: 0 } });
        if (embedder.requiresFit) embedder.fit(["test"]);
        const [vec] = await embedder.embed(["ZotWatch connection test"]);
        Object.assign(row, { ok: true, detail: `正常，维度 ${vec.length}` });
      } catch (e) {
        row.detail = String(e?.message || e);
      }
      rows.push(row);
    }
    if (want("llm")) {
      if (settings.llm.enabled) {
        const row = { key: "llm", name: "大模型", target: settings.llm.model, ok: false, detail: "" };
        try {
          const reply = await createLlmClient(settings.llm, this.transport, { retry: { retries: 0 } }).chat("请只回复：OK", { maxTokens: 20 });
          Object.assign(row, { ok: true, detail: reply.trim().slice(0, 30) });
        } catch (e) {
          row.detail = String(e?.message || e);
        }
        rows.push(row);
      } else {
        rows.push({ key: "llm", name: "大模型", target: null, ok: null, detail: "未启用" });
      }
    }
    if (want("crossref")) {
      const row = { key: "crossref", name: "Crossref", target: null, ok: false, detail: "" };
      try {
        const res = await this.transport.request({ url: "https://api.crossref.org/works?rows=0", timeout: 20000 });
        Object.assign(row, res.status === 200 ? { ok: true, detail: "可访问" } : { detail: `HTTP ${res.status}` });
      } catch (e) {
        row.detail = String(e?.message || e);
      }
      rows.push(row);
    }
    if (want("elsevier") || want("springer")) {
      const keys = await checkPublisherKeyRows(this.transport, settings.abstracts);
      rows.push(...keys.filter((r) => want(r.key)).map((r) => ({ ...r, target: null })));
    }
    return rows;
  }

  /** The connection checks as text, one line per service. */
  async testConnections() {
    return (await this.checkConnections()).map(connectionLine).join("\n");
  }

  /** Parsed followed-authors text ({authors, invalid}) for the settings preview. */
  parseFollowedAuthors(text) {
    return parseFollowedAuthors(text);
  }

  /** Whitelist CSV summary ({count, withImpactFactor, invalidLines}) for the settings preview. */
  summarizeWhitelist(text) {
    return summarizeWhitelist(text);
  }

  /** Values used for blank fields with the current embedding provider (placeholders, restore default). */
  getAutoValues() {
    return autoValues(this.getSettings().embedding.provider);
  }

  // ------------------------------------------ status text and journal / data actions

  async getStatus() {
    const settings = this.getSettings();
    const items = await createZoteroLibrary().getItems();
    const withAbstract = items.filter((i) => i.abstract).length;
    const engine = this.createEngine(settings);
    const inbox = await engine.loadInbox();
    const pending = Object.values(inbox.works).filter((w) => w.status === "new").length;
    const last = inbox.lastRun;
    const library = Zotero.Libraries.get(profileLibraryID());
    const followed = settings.followed;
    const topics = settings.llm.topicLabels ? await engine.getTopics() : [];
    return [
      ...(this.whitelistError ? [`⚠️ ${this.whitelistError}`] : []),
      `文库：${library?.name || "我的文库"}，${items.length} 个条目（含摘要 ${withAbstract}）`,
      `待查看推荐：${pending}`,
      last ? `上次运行：${new Date(last.runAt).toLocaleString()}，推荐 ${last.stats.recommended} 篇，抓取 ${last.stats.fetched} 篇` : "尚未运行",
      ...(topics.length ? [topicsStatusLine(topics)] : []),
      ...(followed.enabled ? [followedStatusLine(followed)] : []),
      ...(settings.flagship.enabled ? [flagshipStatusLine(settings.flagship, last?.stats?.flagship)] : []),
      ...(last?.stats?.abstracts ? [abstractsStatusLine(last.stats.abstracts)] : []),
      `数据目录：${this.storage.dir}`,
    ].join("\n");
  }

  /** Whitelist CSV with the top journals of the library merged into the current list. */
  async suggestJournals() {
    const settings = this.getSettings();
    const engine = this.createEngine({ ...settings, journals: { ...settings.journals, autoFromLibrary: true } });
    const items = await createZoteroLibrary().getItems();
    const { venues } = await engine.resolveJournals(items);
    // Merge into the list as it is now, not as it was before the (slow) lookups
    const merged = parseWhitelist(this.whitelist);
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
