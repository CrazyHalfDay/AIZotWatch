// Plugin object: lifecycle, Tools menu, scheduler, and the API used by the
// ZotWatch main window and the preference pane (exposed as Zotero.ZotWatch).

/* global Zotero, Services */

import { apiUsage, createCancelSignal, limitedRequest } from "../core/http.js";
import { parseFollowedAuthors } from "../core/authors.js";
import { DEFAULT_RATE_LIMITS, mergeRateLimits, sharedRateLimiter } from "../core/ratelimit.js";
import { journalRows, mergeWhitelistRows, parseWhitelist, summarizeWhitelist } from "../core/journals.js";
import { FLAGSHIP_JOURNALS } from "../core/flagship.js";
import { normalizeIssn } from "../core/text.js";
import { countByTrack } from "../core/tracks.js";
import { CancelledError, HANDLED_MIN_DAYS, OPENALEX_KEYLESS, ZotWatchEngine } from "../engine.js";
import { EMBEDDING_PRESETS, createEmbedder } from "../embedding/providers.js";
import { fillJournalMetadata, proposeJournals } from "../journal-proposals.js";
import { createLlmClient } from "../llm/llm.js";
import { autoValues } from "../settings.js";
import { checkPublisherKeyRows } from "../sources/abstracts.js";
import { checkOpenAlexRow, openAlexUsageText } from "../sources/openalex.js";
import { lookupSourcesByIssn, openAlexErrorText, searchAuthors, searchSources } from "../sources/openalex-entities.js";
import { checkSemanticScholarRow } from "../sources/semanticscholar.js";
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
// Search-box results kept for the session (per box and query)
const SEARCH_CACHE_SIZE = 60;
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
    ["Semantic Scholar", counts.s2],
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

/**
 * Rate limits from the hidden pref extensions.zotwatch.http.rateLimits (JSON
 * {host: {rps, concurrency} | null}) merged into the defaults; malformed text
 * is ignored.
 */
export function rateLimitsFromPref(text) {
  let overrides = null;
  if (typeof text === "string" && text.trim()) {
    try {
      overrides = JSON.parse(text);
    } catch {
      overrides = null;
    }
  }
  return mergeRateLimits(DEFAULT_RATE_LIMITS, overrides);
}

/**
 * The 学术数据接口 summary: whether each key is set and OpenAlex's latest known
 * daily budget ({...usage, text} or null), for the dashboard and the settings page.
 */
export function apiStatus(settings, usage = apiUsage.openalex) {
  const apis = settings.apis || {};
  const openalex = usage ? { ...usage, text: openAlexUsageText(usage), short: openAlexUsageText(usage, { compact: true }) } : null;
  return { openalexKey: !!apis.openalexKey, semanticScholarKey: !!apis.semanticScholarKey, mailto: !!apis.mailto, openalexUsage: openalex };
}

/** Status line for the 学术数据接口 keys (and OpenAlex's budget when known). */
export function apiStatusLine(status) {
  const openalex = status.openalexKey ? "OpenAlex 已配置 Key" : "OpenAlex 未配置 Key（免费额度很小）";
  const budget = status.openalexUsage?.text ? `（${status.openalexUsage.text}）` : "";
  return `学术数据接口：${openalex}${budget}，Semantic Scholar ${status.semanticScholarKey ? "已配置 Key" : "未配置"}`;
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

// Data-source row of a run that failed in these stages (the embedding requests)
const FAILED_STAGE_SOURCES = { profile: "embedding", embed: "embedding" };

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
    // 关注期刊 actions and the settings search boxes (own cancel signals, session caches)
    this.journalSignal = null;
    this.searchSignals = null;
    this.searchCache = null;
    this.journalNames = null;
  }

  // ------------------------------------------------------------- lifecycle

  async startup({ id, version, rootURI }) {
    this.id = id;
    this.version = version;
    this.rootURI = rootURI;
    await Zotero.initializationPromise;
    this.storage = createZoteroStorage();
    this.applyRateLimits();
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
    this.journalSignal?.cancel();
    for (const signal of this.searchSignals?.values() || []) signal.cancel();
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

  /** The per-host rate limiter shared by every transport of this addon instance. */
  get limiter() {
    return sharedRateLimiter();
  }

  /** Apply the hidden rate-limit pref to the shared limiter (at startup and before each run). */
  applyRateLimits() {
    try {
      sharedRateLimiter().setLimits(rateLimitsFromPref(getPref("http.rateLimits")));
    } catch (e) {
      Zotero.logError(e);
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
    this.applyRateLimits();
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
        // The stage it failed in names the service on the dashboard (progress is cleared below)
        this.lastError = { message: e.message, at: Date.now(), stage: this.progress?.stage ?? null };
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
    // The new window starts on lastPage. The route is not put in the URL:
    // the persisted size and position are stored under the document URI
    this.lastPage = route;
    const features = "chrome,dialog=no,resizable,centerscreen";
    const win = Zotero.getMainWindow();
    // On macOS Zotero can run without a main window
    return win
      ? win.openDialog(DIALOG_URL, "zotwatch-main", features)
      : Services.ww.openWindow(null, DIALOG_URL, "zotwatch-main", features, null);
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
    // A failure newer than the last run, matched to the service it stopped at
    const finished = last ? last.finishedAt || last.runAt : 0;
    const failure = this.lastError && (!finished || this.lastError.at > finished) ? this.lastError : null;
    const failedSource = failure ? FAILED_STAGE_SOURCES[failure.stage] || null : null;
    // A fetch source took part in the last run when it reported a count or an error
    const fetchRow = (key, name, enabled, perSourceName, detail, error) => ({
      key,
      name,
      enabled,
      count: sourceCount(perSourceName),
      detail,
      error,
      ran: Object.hasOwn(perSource, perSourceName) || !!error,
    });
    // A keyless fetch cut to its first page still worked: the OpenAlex row tells about the key
    const followedError = errorMessage(errors, (e) => e.stage === "followed" && e.source === "OpenAlex" && e.code !== OPENALEX_KEYLESS);
    const apis = apiStatus(settings);
    const pendingBy = (name) => pendingWorks.filter((w) => (w.followedAuthors || []).includes(name)).length;
    const abstractTiers = abstracts.crossref || abstracts.openalex || !!abstracts.elsevierKey || !!abstracts.springerKey || !!settings.apis?.semanticScholarKey;

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
        fetchRow(
          "crossref",
          "Crossref",
          sources.crossref.enabled,
          "Crossref",
          // ISSNs (a journal often has two), without the flagship ones queried by the 顶刊 row
          Number.isFinite(stats.personalIssns ?? stats.journals) ? `${stats.personalIssns ?? stats.journals} 个 ISSN` : null,
          errorMessage(errors, (e) => e.stage === "fetch" && e.source === "Crossref"),
        ),
        fetchRow(
          "flagship",
          "顶刊",
          flagship.enabled,
          "顶刊",
          `${flagship.issns.length} 个 ISSN`,
          errorMessage(errors, (e) => e.stage === "flagship" || (e.stage === "fetch" && e.source === "Crossref 顶刊")),
        ),
        fetchRow("arxiv", "arXiv", sources.arxiv.enabled, "arXiv", sources.arxiv.categories.join(", ") || null, errorMessage(errors, (e) => e.stage === "fetch" && e.source === "arXiv")),
        fetchRow("eartharxiv", "EarthArXiv", sources.eartharxiv.enabled, "EarthArXiv", null, errorMessage(errors, (e) => e.stage === "fetch" && e.source === "EarthArXiv")),
        {
          key: "followed",
          name: "关注作者",
          enabled: followed.enabled,
          count: stats.followed ? stats.followed.fetched : null,
          detail: `OpenAlex，${followed.authors.length} 位`,
          // The OpenAlex fetch only: unparseable author lines are warnings of the 关注作者 card
          error: followedError,
          ran: !!stats.followed || !!followedError,
        },
        {
          key: "abstracts",
          name: "摘要补全",
          enabled: abstractTiers,
          count: stats.abstracts ? stats.abstracts.filled ?? 0 : null,
          detail: stats.abstracts ? `上次补全 ${stats.abstracts.filled ?? 0} / ${stats.abstracts.missing ?? 0}` : null,
          error: errorMessage(errors, (e) => e.stage === "enrich"),
        },
        {
          key: "openalex",
          name: "OpenAlex",
          enabled: true,
          count: null,
          detail: apis.openalexUsage?.short || (apis.openalexKey ? "已配置 API Key" : "免费额度很小"),
          error: errorMessage(errors, (e) => e.source === "OpenAlex" && e.code !== OPENALEX_KEYLESS),
          // Rows of API services carry their own states (keys, not fetch counts)
          status: apis.openalexKey ? { tone: "good", label: "正常" } : { tone: "warning", label: "未配置 Key" },
        },
        {
          key: "semanticscholar",
          name: "Semantic Scholar",
          enabled: apis.semanticScholarKey,
          count: null,
          detail: apis.semanticScholarKey ? "摘要与 TLDR" : "可选",
          error: errorMessage(errors, (e) => e.source === "Semantic Scholar"),
          status: apis.semanticScholarKey ? { tone: "good", label: "已配置" } : { tone: "off", label: "未配置" },
        },
        { key: "embedding", name: "向量化", enabled: true, count: null, detail: embedding.label, error: failedSource === "embedding" ? failure.message : null },
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
      apis,
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
    // Written first: after a failed write the list in memory stays the saved
    // one, so the settings page sees the edit as unsaved and retries it
    await this.storage.writeText(WHITELIST_FILE, value);
    this.whitelist = value;
    this.whitelistError = null;
  }

  // ---------------------------------------------------------- settings helpers

  /**
   * Check the configured services: [{key, name, target, ok, detail}], where ok
   * is true, false, or null for a service that is not enabled (or, for
   * OpenAlex, reachable without a key). `only` limits the checks to some keys
   * ("embedding", "llm", "crossref", "openalex", "semanticscholar", "elsevier",
   * "springer"); publisher keys are only checked when filled in.
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
        const res = await limitedRequest(this.transport, { url: "https://api.crossref.org/works?rows=0", timeout: 20000 });
        Object.assign(row, res.status === 200 ? { ok: true, detail: "可访问" } : { detail: `HTTP ${res.status}` });
      } catch (e) {
        row.detail = String(e?.message || e);
      }
      rows.push(row);
    }
    if (want("openalex")) {
      rows.push({ ...(await checkOpenAlexRow(this.transport, { apiKey: settings.apis.openalexKey, mailto: settings.apis.mailto })), target: null });
    }
    if (want("semanticscholar")) {
      rows.push({ ...(await checkSemanticScholarRow(this.transport, { key: settings.apis.semanticScholarKey })), target: null });
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

  /** Which API keys are set and OpenAlex's latest known daily budget (see apiStatus). */
  getApiStatus() {
    return apiStatus(this.getSettings());
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
      apiStatusLine(apiStatus(settings)),
      `数据目录：${this.storage.dir}`,
    ].join("\n");
  }

  // ------------------------------------------- journal proposals and lookups

  /**
   * Run a journal action (「从文库生成」, 「补全学科与影响因子」) with its own
   * cancel signal: cancelJournalAction() stops it without touching runs or
   * other UI requests. A new action cancels the previous one.
   */
  async withJournalSignal(task) {
    this.journalSignal?.cancel();
    const signal = createCancelSignal();
    this.journalSignal = signal;
    try {
      return await task(createZoteroTransport({ signal }));
    } finally {
      if (this.journalSignal === signal) this.journalSignal = null;
    }
  }

  /** Stop 「从文库生成」 or 「补全学科与影响因子」 (their open requests are aborted). */
  cancelJournalAction() {
    this.journalSignal?.cancel();
  }

  /**
   * Journal proposals for the review panel of 「从文库生成」 (nothing is
   * written): {proposals, warnings, missing, stats}, see journal-proposals.js.
   * With the LLM enabled, the model also suggests related journals (named
   * after the library's research directions when they are labelled).
   */
  async suggestJournals() {
    const settings = this.getSettings();
    return this.withJournalSignal(async (http) => {
      const items = await createZoteroLibrary().getItems();
      const llmOn = settings.llm.enabled && !!settings.llm.apiKey;
      let researchFocus = "";
      if (llmOn && settings.llm.topicLabels) {
        const topics = await this.createEngine(settings).getTopics().catch(() => []);
        researchFocus = topics.slice(0, 10).map((t) => t.label).join("、");
      }
      return proposeJournals({
        http,
        storage: this.storage,
        items,
        // Proposals already listed are marked against the list as it is now
        whitelist: this.whitelist,
        settings,
        llm: llmOn ? createLlmClient(settings.llm, http) : null,
        researchFocus,
        log,
      });
    });
  }

  /**
   * Merge journals (proposals or search results: {title, issns, category,
   * impactFactor}) into whitelist CSV text, one row per ISSN; listed rows
   * only get their blanks filled. Returns {text, added, filled}; nothing is saved.
   */
  mergeJournals(text, journals) {
    return mergeWhitelistRows(text, journalRows(journals));
  }

  /**
   * 「补全学科与影响因子」 for whitelist CSV text: {text, filled, incomplete,
   * notFound}; nothing is saved. Throws a readable message when OpenAlex fails.
   */
  async fillJournalMetadata(text) {
    const settings = this.getSettings();
    return this.withJournalSignal(async (http) => {
      try {
        return await fillJournalMetadata({ http, text, settings });
      } catch (e) {
        if (e?.cancelled) throw e;
        throw new Error(openAlexErrorText(e, { keyed: !!settings.apis.openalexKey }));
      }
    });
  }

  /**
   * One search of a search box: `kind` names the box ("journals" or
   * "authors"); a newer search of the same box cancels the older one. Results
   * are cached for the session. Throws a readable message when OpenAlex fails.
   */
  async searchOpenAlex(kind, query, search) {
    const q = String(query ?? "").trim();
    const settings = this.getSettings();
    const cacheKey = `${kind}\n${q.toLowerCase()}\n${settings.apis.openalexKey ? "k" : ""}`;
    this.searchCache ||= new Map();
    if (this.searchCache.has(cacheKey)) return this.searchCache.get(cacheKey);
    this.searchSignals ||= new Map();
    this.searchSignals.get(kind)?.cancel();
    const signal = createCancelSignal();
    this.searchSignals.set(kind, signal);
    const http = createZoteroTransport({ signal });
    try {
      const results = await search(http, q, { apiKey: settings.apis.openalexKey, mailto: settings.apis.mailto, retry: { retries: 1 } });
      if (this.searchCache.size >= SEARCH_CACHE_SIZE) this.searchCache.delete(this.searchCache.keys().next().value);
      this.searchCache.set(cacheKey, results);
      return results;
    } catch (e) {
      if (e?.cancelled) throw e;
      throw new Error(openAlexErrorText(e, { keyed: !!settings.apis.openalexKey }));
    } finally {
      if (this.searchSignals.get(kind) === signal) this.searchSignals.delete(kind);
    }
  }

  /** Journals matching a name or an ISSN (OpenAlex sources; see openalex-entities.js mapOpenAlexSource). */
  searchJournals(query) {
    return this.searchOpenAlex("journals", query, (http, q, opts) => searchSources(http, q, { limit: 10, ...opts }));
  }

  /** Authors matching a name, an ORCID iD or an OpenAlex author id (see mapOpenAlexAuthor). */
  searchAuthors(query) {
    return this.searchOpenAlex("authors", query, (http, q, opts) => searchAuthors(http, q, { limit: 10, ...opts }));
  }

  /**
   * Journal names of ISSNs (the 顶刊 chips): {issn: {title, issns, openalexId}
   * | null} for every requested ISSN. Known names (the built-in flagship
   * journals, the journal list) need no request; the others are looked up on
   * OpenAlex once per session (null: OpenAlex does not know the ISSN). When
   * OpenAlex fails, the unknown ISSNs are left out and `error` says why.
   */
  async lookupJournalNames(issns) {
    const wanted = [...new Set((issns || []).map(normalizeIssn).filter(Boolean))];
    this.journalNames ||= new Map(FLAGSHIP_JOURNALS.flatMap((j) => j.issns.map((issn) => [issn, { title: j.title, issns: [...j.issns], openalexId: null }])));
    for (const j of parseWhitelist(this.whitelist).values()) {
      if (j.title && !this.journalNames.has(j.issn)) this.journalNames.set(j.issn, { title: j.title, issns: [j.issn], openalexId: null, fromList: true });
    }
    const unknown = wanted.filter((issn) => !this.journalNames.has(issn));
    let error = null;
    if (unknown.length) {
      const settings = this.getSettings();
      try {
        const found = await lookupSourcesByIssn(this.transport, unknown, { apiKey: settings.apis.openalexKey, mailto: settings.apis.mailto, retry: { retries: 1 } });
        for (const issn of unknown) {
          const j = found.get(issn);
          this.journalNames.set(issn, j ? { title: j.title, issns: j.issns, openalexId: j.openalexId } : null);
        }
      } catch (e) {
        if (e?.cancelled) throw e;
        error = openAlexErrorText(e, { keyed: !!settings.apis.openalexKey });
      }
    }
    const names = {};
    for (const issn of wanted) if (this.journalNames.has(issn)) names[issn] = this.journalNames.get(issn);
    return { names, error };
  }

  async loadPresetJournals() {
    return Zotero.File.getResourceAsync("chrome://zotwatch/content/presets/geoscience-journals.csv");
  }

  async resetData() {
    if (this.running) throw new Error("正在运行，请稍后再试");
    await this.storage.clear();
    // In-memory OpenAlex answers of the settings page are caches too
    this.searchCache?.clear();
    this.journalNames = null;
    // The whitelist is user data, not cache: keep it
    if (this.whitelist) await this.storage.writeText(WHITELIST_FILE, this.whitelist);
    setPref("schedule.lastRun", "");
    this.pendingCount = 0;
    this.emit({ type: "inbox" });
  }
}
