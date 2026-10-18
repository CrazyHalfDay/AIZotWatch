/* global ChromeUtils, Services, window, document, location, Intl, Event */

// ZotWatch main window shell: sidebar, router, page header, run progress,
// error banner, toasts, keyboard shortcuts and addon events.
//
// The page scripts (overview.js, inbox.js, survey.js, settings.js) are classic scripts
// sharing this window's global scope: the top-level helpers below ($, el,
// svgIcon, segmented, format*, toast, refresh, ...) are theirs too. Each page
// registers { init(), show(params), refresh(view), hide?(), flush?(), tick?()
// (every minute while shown), heading?() (focus target after a switch) } with
// registerPage(). All data comes from Zotero.ZotWatch (src/zotero/addon.js);
// external text is only ever inserted with textContent.

var { Zotero } = ChromeUtils.importESModule("chrome://zotero/content/zotero.mjs");

const SVG_NS = "http://www.w3.org/2000/svg";
const PAGE_TITLES = { overview: "概览", inbox: "推荐", survey: "调研", settings: "设置" };
const PAGE_KEYS = ["overview", "inbox", "survey", "settings"];
// Keyboard shortcuts of the pages (Ctrl/Cmd + digit); 调研 came after the first three
const PAGE_SHORTCUTS = { 1: "overview", 2: "inbox", 3: "settings", 4: "survey" };
// Survey ids (src/survey/store.js isSurveyId): #survey/<id>
const SURVEY_ID = /^[a-z0-9]{6,40}$/;

/** Sections of the settings page (sidebar sub-navigation and #settings/<id>). */
const SETTINGS_SECTIONS = [
  { id: "general", title: "常规", icon: "clock", description: "自动检查新文献的频率和推荐保留时长。" },
  { id: "embedding", title: "向量化", icon: "vector", description: "计算文献与文库语义相似度的向量化服务。" },
  { id: "apis", title: "学术数据接口", icon: "key", description: "OpenAlex、Semantic Scholar 的 API Key 与 Crossref 联系邮箱。" },
  { id: "sources", title: "文献来源", icon: "rss", description: "从哪些来源抓取新文献，以及缺少摘要时如何补全。" },
  { id: "journals", title: "关注期刊", icon: "journal", description: "Crossref 按这些期刊的 ISSN 抓取新论文，影响因子参与打分。" },
  { id: "followed", title: "关注作者", icon: "user", description: "按 ORCID 追踪指定研究者的新论文。" },
  { id: "flagship", title: "顶刊地学", icon: "mountain", description: "从 Nature、Science 等综合性顶刊中单独挑出地学文章。" },
  { id: "ranking", title: "推荐与过滤", icon: "filter", description: "推荐数量、时间范围、相似度门槛和关键词过滤。" },
  { id: "ai", title: "AI 助手", icon: "sparkle", description: "标题翻译、AI 摘要和研究方向标签（可选）。" },
  { id: "zotero", title: "加入 Zotero", icon: "plus-square", description: "加入文库时使用的分类、标签和子笔记。" },
  { id: "data", title: "数据与维护", icon: "database", description: "数据目录、缓存清理和版本信息。" },
];

/** Display names of the recommendation tracks (see src/core/tracks.js). */
const TRACK_NAMES = {
  personal: { name: "个性推荐", short: "个性" },
  flagship: { name: "顶刊地学", short: "顶刊" },
  followed: { name: "关注作者", short: "关注" },
};
const STAGE_NAMES = { followed: "关注作者", flagship: "顶刊地学", enrich: "摘要补全", llm: "大模型" };

const api = () => Zotero.ZotWatch;
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------- DOM helpers

/** Create an element: attrs may hold class, text, on<event> handlers and plain attributes. */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

/** An icon from the sprite in main.xhtml (#i-<name>). */
function svgIcon(name, cls = "icon") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

/** Empty state: icon, one line, an optional sentence and an optional action button. */
function emptyState({ icon = "info", title, text = null, action = null, compact = false }) {
  return el(
    "div",
    { class: compact ? "empty compact" : "empty" },
    el("span", { class: "empty-icon" }, svgIcon(icon)),
    el("p", { class: "empty-title", text: title }),
    text ? el("p", { class: "empty-text", text }) : null,
    action ? el("button", { type: "button", class: action.primary ? "btn primary" : "btn", onclick: action.onclick }, action.label) : null,
  );
}

/** Wire a .seg group (buttons with data-value) as a single-choice control. */
function segmented(id, onSelect) {
  const group = $(id);
  const buttons = () => [...group.querySelectorAll("button[data-value]")];
  for (const button of buttons()) {
    button.addEventListener("click", () => {
      setSegmented(group, button.dataset.value);
      onSelect(button.dataset.value);
    });
  }
  return group;
}

/** Mark the button with `value` active in a .seg group (element or id). */
function setSegmented(group, value) {
  const node = typeof group === "string" ? $(group) : group;
  for (const b of node.querySelectorAll("button[data-value]")) {
    const on = b.dataset.value === value;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  }
}

function isTextField(target) {
  if (!target || !target.tagName) return false;
  const tag = String(target.tagName).toLowerCase();
  if (tag === "textarea" || target.isContentEditable) return true;
  if (tag !== "input") return false;
  return !["checkbox", "radio", "button", "submit", "reset", "range", "color", "file"].includes(String(target.type || "text").toLowerCase());
}

function confirmAction(message) {
  return typeof Services !== "undefined" && Services.prompt ? Services.prompt.confirm(window, "ZotWatch", message) : window.confirm(message);
}

// -------------------------------------------------------------- formatting

const NUMBER_FORMAT = new Intl.NumberFormat("zh-CN");
const MINUTE = 60000;
const HOUR = 3600000;
const DAY = 86400000;

/** 1,284 (grouped); "–" for missing values. */
function formatNumber(n) {
  return Number.isFinite(n) ? NUMBER_FORMAT.format(n) : "–";
}

/** Auto-compact count for stat tiles: 1,284 below 10,000, then 12.9K / 3.4M. */
function formatCompact(n) {
  if (!Number.isFinite(n)) return "–";
  const abs = Math.abs(n);
  const short = (value, unit) => `${(Math.round(value * 10) / 10).toFixed(1).replace(/\.0$/, "")}${unit}`;
  if (abs >= 1e6) return short(n / 1e6, "M");
  if (abs >= 1e4) return short(n / 1e3, "K");
  return NUMBER_FORMAT.format(Math.round(n));
}

/** 刚刚 / N 分钟前 / N 小时前 / N 天前 ("" without a time). */
function formatRelative(ms, now = Date.now()) {
  if (!ms) return "";
  const diff = Math.max(0, now - ms);
  if (diff < MINUTE) return "刚刚";
  if (diff < HOUR) return `${Math.floor(diff / MINUTE)} 分钟前`;
  if (diff < DAY) return `${Math.floor(diff / HOUR)} 小时前`;
  return `${Math.floor(diff / DAY)} 天前`;
}

/** 约 N 分钟后 / 约 N 小时后 / 约 N 天后; "即将" when the time has come. */
function formatUntil(ms, now = Date.now()) {
  const diff = ms - now;
  if (diff < MINUTE) return "即将";
  if (diff < HOUR) return `约 ${Math.ceil(diff / MINUTE)} 分钟后`;
  if (diff < 2 * DAY) return `约 ${Math.round(diff / HOUR)} 小时后`;
  return `约 ${Math.round(diff / DAY)} 天后`;
}

/** Duration as "12 秒" / "3 分 12 秒" / "1 小时 5 分". */
function formatDuration(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  return `${Math.floor(s / 3600)} 小时 ${Math.floor((s % 3600) / 60)} 分`;
}

const pad2 = (n) => String(n).padStart(2, "0");

/** Local "2024-10-08 09:30". */
function formatDateTime(ms) {
  if (!ms) return "";
  const d = new Date(ms);
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

/** Local "10-08" (chart axis labels). */
function formatMonthDay(ms) {
  const d = new Date(ms);
  return `${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

/** Publication date as YYYY-MM-DD ("" when missing or invalid). */
function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toISOString().slice(0, 10);
}

/** Track of a work: "flagship", "followed" or "personal" (same rule as src/core/tracks.js). */
function trackOf(work) {
  return work.label === "flagship" || work.label === "followed" ? work.label : "personal";
}

/** "关注作者（OpenAlex）" for a stage with a source, the source alone for fetch errors. */
function errorOrigin(error) {
  const stage = STAGE_NAMES[error.stage];
  if (error.source && stage) return `${stage}（${error.source}）`;
  return error.source || stage || error.stage || "运行";
}

/**
 * Sidebar run status: {tone, label, next} where tone is running | good |
 * warning | critical | idle (the status dot), label the status line and next
 * the scheduler line.
 */
function runStatus(view, now = Date.now()) {
  const last = view?.lastRun;
  const finished = last ? last.finishedAt || last.runAt : 0;
  let tone;
  let label;
  if (view?.running) {
    tone = "running";
    label = "正在检查…";
  } else if (view?.lastError && (!finished || view.lastError.at > finished)) {
    tone = "critical";
    label = `检查失败 · ${formatRelative(view.lastError.at, now)}`;
  } else if (last) {
    tone = last.errors?.length ? "warning" : "good";
    label = `上次检查 ${formatRelative(finished, now)}`;
  } else {
    tone = "idle";
    label = "尚未运行";
  }
  const schedule = view?.schedule;
  let next = "";
  if (schedule) {
    if (!schedule.autoRun) next = "自动检查已关闭";
    else if (schedule.nextRunAt) next = schedule.nextRunAt <= now ? "下次：即将检查" : `下次：${formatUntil(schedule.nextRunAt, now)}`;
  }
  return { tone, label, next };
}

/**
 * Parse "#settings/ai" or "#survey/<id>" into {page, section, route}; unknown
 * pages fall back to the overview. The section of 调研 is the survey shown.
 */
function parseRoute(route) {
  const [name, section] = String(route || "").replace(/^#/, "").split("/");
  const page = PAGE_TITLES[name] ? name : "overview";
  let sectionId = null;
  if (page === "settings" && SETTINGS_SECTIONS.some((s) => s.id === section)) sectionId = section;
  if (page === "survey" && SURVEY_ID.test(section || "")) sectionId = section;
  return { page, section: sectionId, route: sectionId ? `${page}/${sectionId}` : page };
}

// ------------------------------------------------------------------ shell

const pages = {};
const shell = {
  page: null,
  section: null,
  view: null,
  subtitles: {},
  scroll: {},
  bannerDismissed: null,
  // The banner shows a load failure (cleared by the next successful load)
  loadError: false,
  ticker: null,
};

function registerPage(key, page) {
  pages[key] = page;
}

/** Subtitle of a page's header (shown while that page is active). */
function setSubtitle(key, text) {
  shell.subtitles[key] = text || "";
  if (shell.page === key) {
    const node = $("page-subtitle");
    node.textContent = shell.subtitles[key];
    node.setAttribute("title", shell.subtitles[key]);
  }
}

/** Highlight a settings section in the sidebar (also used by the settings page's scroll spy). */
function setActiveSection(id) {
  shell.section = id || null;
  highlightSection(shell.section);
}

function highlightSection(id) {
  for (const link of document.querySelectorAll(".subnav-item")) {
    const on = link.dataset.section === id;
    link.classList.toggle("active", on);
    if (on) link.setAttribute("aria-current", "true");
    else link.removeAttribute("aria-current");
  }
}

function renderNav() {
  for (const item of document.querySelectorAll(".nav-item")) {
    if (item.dataset.page === shell.page) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  }
  $("settings-nav").hidden = shell.page !== "settings";
  // The section of another page (the survey shown on 调研) is not a settings section
  if (shell.page === "settings") setActiveSection(shell.section);
  else highlightSection(null);
}

/** Route of the page shown now ("inbox", "settings/ai", ...). */
function currentRoute() {
  return shell.section ? `${shell.page}/${shell.section}` : shell.page;
}

/** Whether keyboard focus is on nothing, the document, or an element a page switch just hid. */
function focusLost() {
  const active = document.activeElement;
  return !active || active === document.body || active === document.documentElement || !!active.closest?.("[hidden]");
}

/**
 * Key of a remembered scroll position: the page, or the survey shown on 调研
 * (its list and each survey keep their own position).
 */
function scrollKey(page, section) {
  return page === "survey" && section ? `survey/${section}` : page;
}

/**
 * Switch to a page ("inbox", "settings/ai", ...). The document URL keeps no
 * route: Zotero stores the window's size and position under the document URI.
 */
function navigate(route) {
  const next = parseRoute(route);
  const previous = shell.page;
  const pagesNode = $("pages");
  // 调研's list and each survey are separate views of one page, with their own scroll positions
  const moved = previous !== next.page || (next.page === "survey" && scrollKey(previous, shell.section) !== scrollKey(next.page, next.section));
  if (previous && moved) shell.scroll[scrollKey(previous, shell.section)] = pagesNode.scrollTop;
  if (previous && previous !== next.page) {
    try {
      // Unsaved input (e.g. a settings field still in its debounce) is saved on leaving
      pages[previous]?.flush?.();
      pages[previous]?.hide?.();
    } catch (e) {
      Zotero.logError(e);
    }
  }
  shell.page = next.page;
  shell.section = next.section;

  for (const section of document.querySelectorAll(".page")) section.hidden = section.dataset.page !== next.page;
  for (const group of document.querySelectorAll(".page-actions")) group.hidden = group.dataset.page !== next.page;
  renderNav();
  $("page-title").textContent = PAGE_TITLES[next.page];
  setSubtitle(next.page, shell.subtitles[next.page]);
  // Remembered for this Zotero session (openWindow() without a page)
  if (api()) api().lastPage = next.route;

  const page = pages[next.page];
  try {
    page?.show?.({ section: next.section });
    if (shell.view) page?.refresh?.(shell.view);
  } catch (e) {
    Zotero.logError(e);
  }
  // A settings section scrolls itself into view; 调研 remembers its list and each survey
  if (moved && (!next.section || next.page === "survey")) pagesNode.scrollTop = shell.scroll[scrollKey(next.page, next.section)] || 0;
  // Switching from a control the switch hides (a link or button on the old
  // page, Ctrl+1…4 in a field) would drop focus to the window: move it to
  // the new page's heading (sidebar buttons stay visible and keep it)
  if (previous && focusLost()) {
    try {
      (page?.heading?.() || $("page-title")).focus?.({ preventScroll: true });
    } catch (e) {
      Zotero.logError(e);
    }
  }
}

function renderRunState(running, progress) {
  $("progress").hidden = !running;
  $("run").disabled = !!running;
  if (!running) return;
  $("progress-text").textContent = progress?.message || "正在运行…";
  const bar = $("progress-bar");
  if (progress?.total) {
    const pct = Math.max(0, Math.min(100, Math.round((100 * progress.current) / progress.total)));
    bar.classList.remove("indeterminate");
    bar.style.width = `${pct}%`;
    $("progress-count").textContent = `${formatNumber(progress.current)} / ${formatNumber(progress.total)}`;
  } else {
    bar.classList.add("indeterminate");
    bar.style.width = "";
    $("progress-count").textContent = "";
  }
}

function showBanner(text) {
  $("banner-text").textContent = text;
  $("banner").hidden = false;
}

function hideBanner() {
  $("banner").hidden = true;
}

/** Sidebar count, version and run status, the progress strip and the error banner. */
function renderShell(view) {
  const pending = view.works.filter((w) => w.status === "new").length;
  const count = $("nav-count");
  count.hidden = pending === 0;
  count.textContent = pending > 999 ? "999+" : String(pending);
  $("brand-version").textContent = view.version ? `v${view.version}` : "";

  const status = runStatus(view);
  $("run-dot").className = `dot ${status.tone}`;
  $("run-status").textContent = status.label;
  $("run-next").textContent = status.next;
  $("run-footer").setAttribute("title", [status.label, status.next].filter(Boolean).join("\n"));

  renderRunState(view.running, view.progress);
  const failure = view.lastError;
  if (view.running) hideBanner();
  else if (failure && shell.bannerDismissed !== failure.at && !(view.lastRun && (view.lastRun.finishedAt || view.lastRun.runAt) > failure.at)) {
    showBanner(`运行失败：${failure.message}`);
  }
}

let refreshing = null;
let refreshQueued = false;

/** Reload the view and re-render. Calls made while a refresh runs are coalesced into one more. */
function refresh() {
  if (refreshing) {
    refreshQueued = true;
    return refreshing;
  }
  refreshing = (async () => {
    try {
      do {
        refreshQueued = false;
        if (!api()) {
          showBanner("ZotWatch 插件未加载。");
          return;
        }
        shell.view = await api().getView();
        if (shell.loadError) {
          // renderShell shows a run failure again if there is one
          shell.loadError = false;
          hideBanner();
        }
        renderShell(shell.view);
        if (shell.page) pages[shell.page]?.refresh?.(shell.view);
      } while (refreshQueued);
    } catch (e) {
      shell.loadError = true;
      showBanner(`加载失败：${e.message || e}`);
      Zotero.logError(e);
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
}

// ------------------------------------------------------------------ toasts

const toastTimers = new Map();

/**
 * Show a short message at the bottom right for 2.5 s. A toast with the same
 * `key` that is still visible is updated and kept longer instead of stacking.
 */
function toast(text, { tone = "info", key = null, duration = 2500 } = {}) {
  const host = $("toasts");
  let node = key ? host.querySelector(`.toast[data-key="${key}"]`) : null;
  if (!node) {
    node = el("div", { class: `toast ${tone}`, role: "status", "data-key": key || null });
    host.append(node);
  }
  node.className = `toast ${tone}`;
  node.replaceChildren(svgIcon(tone === "error" ? "alert" : "check"), el("span", { text }));
  clearTimeout(toastTimers.get(node));
  toastTimers.set(
    node,
    setTimeout(() => {
      node.classList.add("leaving");
      toastTimers.set(
        node,
        setTimeout(() => {
          node.remove();
          toastTimers.delete(node);
        }, 160),
      );
    }, duration),
  );
  return node;
}

// ------------------------------------------------------------------ events

function onAddonEvent(event) {
  // Topic-survey events concern the 调研 page only (they would reload the shell on every progress step)
  if (event.type === "survey") return;
  // shell.view keeps the run state current: the 60 s sidebar ticker repaints from it
  if (event.type === "start") {
    shell.bannerDismissed = null;
    shell.loadError = false;
    hideBanner();
    if (shell.view) {
      shell.view.running = true;
      shell.view.progress = null;
    }
    renderRunState(true, null);
    if (shell.view) renderShell(shell.view);
    return;
  }
  if (event.type === "progress") {
    if (shell.view) {
      shell.view.running = true;
      shell.view.progress = event.progress;
    }
    renderRunState(true, event.progress);
    return;
  }
  if (event.type === "error") {
    refresh().then(() => showBanner(`运行失败：${event.message}`));
    return;
  }
  // done, cancelled, inbox
  refresh();
}

function onKeyDown(event) {
  const accel = Zotero.isMac ? event.metaKey : event.ctrlKey;
  const key = String(event.key || "").toLowerCase();
  if (accel && !event.altKey && !event.shiftKey) {
    const page = PAGE_SHORTCUTS[key];
    if (page) {
      event.preventDefault();
      navigate(page);
      return;
    }
    if (key === "f") {
      event.preventDefault();
      navigate("inbox");
      pages.inbox?.focusSearch?.();
      return;
    }
    if (key === "w") {
      event.preventDefault();
      window.close();
      return;
    }
  }
  if (event.key === "Escape") {
    if (isTextField(event.target)) {
      // First Escape leaves the field; a search box is cleared first (Firefox
      // does not clear search inputs on Escape itself)
      if (event.target.type === "search" && event.target.value) {
        event.preventDefault();
        event.target.value = "";
        event.target.dispatchEvent(new Event("input", { bubbles: true }));
        return;
      }
      event.target.blur();
      return;
    }
    event.preventDefault();
    window.close();
  }
}

function buildSettingsNav() {
  const nav = $("settings-nav");
  nav.replaceChildren(
    ...SETTINGS_SECTIONS.map((s) => el("a", { class: "subnav-item", href: `#settings/${s.id}`, "data-section": s.id, title: s.description, text: s.title })),
  );
}

/** Save pending edits of every page (before closing, or when leaving a page). */
function flushPages() {
  for (const key of PAGE_KEYS) {
    try {
      pages[key]?.flush?.();
    } catch (e) {
      Zotero.logError(e);
    }
  }
}

function initShell() {
  // Follow Zotero's font size and density preferences
  Zotero.UIProperties?.registerRoot(document.body);
  // Size comes from the persisted/default root attributes; guard against a tiny first window
  if (window.outerWidth < 500 || window.outerHeight < 400) window.resizeTo(1120, 820);

  buildSettingsNav();
  for (const item of document.querySelectorAll(".nav-item")) {
    item.addEventListener("click", () => navigate(item.dataset.page));
    // Shortcut hints in the platform's notation (⌘1 on macOS)
    if (Zotero.isMac) item.setAttribute("title", (item.getAttribute("title") || "").replace("Ctrl+", "⌘"));
  }
  // In-page links (#inbox, #settings/ai, ...) switch pages; web links open in the browser
  document.addEventListener("click", (event) => {
    const link = event.target?.closest?.("a[href]");
    if (!link) return;
    const href = link.getAttribute("href");
    event.preventDefault();
    if (href.startsWith("#")) navigate(href);
    else api()?.openUrl(href);
  });
  window.addEventListener("hashchange", () => {
    if (parseRoute(location.hash).route !== currentRoute()) navigate(location.hash);
  });
  $("run").addEventListener("click", () => api()?.runNow().catch(() => {}));
  $("cancel").addEventListener("click", () => api()?.cancelRun());
  $("banner-close").addEventListener("click", () => {
    shell.bannerDismissed = shell.view?.lastError?.at ?? null;
    hideBanner();
  });
  window.addEventListener("keydown", onKeyDown);

  for (const key of PAGE_KEYS) {
    try {
      pages[key]?.init?.();
    } catch (e) {
      Zotero.logError(e);
    }
  }
  // The page to open on comes from openWindow() (api.lastPage); a #route in
  // the URL (the preview harness) wins
  navigate(location.hash || api()?.lastPage || "overview");

  api()?.addListener(onAddonEvent);
  // Relative times ("2 小时前") stay current in the sidebar and on the active page
  shell.ticker = setInterval(() => {
    if (shell.view) renderShell(shell.view);
    try {
      pages[shell.page]?.tick?.();
    } catch (e) {
      Zotero.logError(e);
    }
  }, 60000);
  refresh().then(() => document.documentElement.setAttribute("data-ready", "true"));
}

/** The remembered scroll position of a page (and 调研 survey), for a page that renders after loading. */
function rememberedScroll(page, section = null) {
  return shell.scroll[scrollKey(page, section)] || 0;
}

window.ZotWatchMain = { navigate, refresh, toast, setSubtitle, setActiveSection, registerPage, rememberedScroll };

window.addEventListener("load", initShell);

window.addEventListener("unload", () => {
  flushPages();
  clearInterval(shell.ticker);
  api()?.removeListener(onAddonEvent);
});
