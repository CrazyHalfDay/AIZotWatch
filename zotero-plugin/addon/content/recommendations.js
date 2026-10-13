/* global ChromeUtils, Services, window, document */

// Recommendations window. All data comes from Zotero.ZotWatch (src/zotero/addon.js);
// external text is only ever inserted with textContent.

var { Zotero } = ChromeUtils.importESModule("chrome://zotero/content/zotero.mjs");

const LABELS = {
  must_read: { text: "必读", cls: "must" },
  consider: { text: "推荐", cls: "consider" },
  ignore: { text: "一般", cls: "ignore" },
};
const SOURCES = { crossref: "期刊", arxiv: "arXiv", eartharxiv: "EarthArXiv" };
const SUMMARY_FIELDS = [
  ["research_question", "研究问题"],
  ["methodology", "研究方法"],
  ["key_findings", "主要发现"],
  ["innovation", "创新点"],
  ["relevance_note", "相关性"],
];

const state = {
  view: null,
  tab: "new",
  filter: "all",
  sort: "score",
  query: "",
  expanded: new Set(),
  busy: new Map(),
  errors: new Map(),
};

const api = () => Zotero.ZotWatch;
const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toISOString().slice(0, 10);
}

function formatRelative(ms) {
  if (!ms) return "";
  const diff = Date.now() - ms;
  if (diff < 60000) return "刚刚";
  if (diff < 3600000) return `${Math.round(diff / 60000)} 分钟前`;
  if (diff < 86400000) return `${Math.round(diff / 3600000)} 小时前`;
  return new Date(ms).toLocaleString();
}

function authorLine(authors) {
  const names = (authors || []).map((a) => a.name).filter(Boolean);
  return names.length > 6 ? `${names.slice(0, 6).join(", ")} 等 ${names.length} 人` : names.join(", ");
}

// ------------------------------------------------------------------ data

async function refresh() {
  if (!api()) {
    $("subtitle").textContent = "ZotWatch 插件未加载";
    return;
  }
  state.view = await api().getView();
  render();
}

function visibleWorks() {
  const { view } = state;
  if (!view) return [];
  const latestRun = view.lastRun?.runId;
  const q = state.query.trim().toLowerCase();
  let works = view.works.filter((w) => w.status === state.tab);
  if (state.filter === "must_read") works = works.filter((w) => w.label === "must_read");
  if (state.filter === "latest") works = works.filter((w) => w.firstRunId === latestRun);
  if (q) {
    works = works.filter((w) =>
      [w.title, w.translatedTitle, w.venue, authorLine(w.authors), w.abstract].some((s) => (s || "").toLowerCase().includes(q)),
    );
  }
  const byDate = (w) => Date.parse(w.published || "") || 0;
  works.sort(state.sort === "date" ? (a, b) => byDate(b) - byDate(a) || b.score - a.score : (a, b) => b.score - a.score);
  return works;
}

// ---------------------------------------------------------------- render

function renderHeader() {
  const { view } = state;
  const run = view.lastRun;
  const parts = [];
  if (run) {
    parts.push(`上次检查：${formatRelative(run.runAt)}`);
    parts.push(`抓取 ${run.stats.fetched ?? 0} 篇，推荐 ${run.stats.recommended ?? 0} 篇（新增 ${run.stats.newRecommendations ?? 0}）`);
  } else {
    parts.push("尚未运行");
  }
  parts.push(view.provider === "local" ? "向量化：本地模式" : `向量化：${view.provider}`);
  $("subtitle").textContent = parts.join(" · ");

  for (const status of ["new", "added", "dismissed"]) {
    $(`count-${status}`).textContent = String(view.works.filter((w) => w.status === status).length);
  }
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === state.tab);
  }
  $("dismiss-visible").hidden = state.tab !== "new";

  const notice = $("notice");
  const errors = run?.errors || [];
  if (errors.length) {
    notice.hidden = false;
    notice.textContent = `上次运行有 ${errors.length} 个问题：${errors.map((e) => `${e.source || e.stage}：${e.message}`).join("；")}`;
  } else if (run && !run.stats.fetched) {
    notice.hidden = false;
    notice.textContent = "上次运行没有抓取到任何文献：请在「设置 → 关注期刊」中确认期刊列表（或开启 arXiv / EarthArXiv），并检查网络连接。";
  } else if (view.provider === "local" && view.works.length) {
    notice.hidden = false;
    notice.textContent = "提示：当前使用本地向量化（无需 API Key），推荐质量有限。在「设置」中配置 DashScope 或 Voyage 可显著提升效果。";
  } else {
    notice.hidden = true;
  }
  renderProgress(view.running, view.progress);
}

function renderProgress(running, progress) {
  $("progress").hidden = !running;
  $("run").disabled = !!running;
  if (!running) return;
  $("progress-text").textContent = progress?.message || "正在运行…";
  const bar = $("progress-bar");
  if (progress?.total) {
    bar.classList.remove("indeterminate");
    bar.style.width = `${Math.round((100 * progress.current) / progress.total)}%`;
  } else {
    bar.classList.add("indeterminate");
    bar.style.width = "";
  }
}

function summaryBlock(work) {
  if (!work.summary) return null;
  return el(
    "div",
    { class: "summary" },
    el("div", { class: "summary-title", text: "AI 摘要" }),
    ...SUMMARY_FIELDS.filter(([k]) => work.summary[k]).map(([k, label]) =>
      el("div", { class: "summary-row" }, el("span", { class: "summary-key", text: label }), el("span", { text: work.summary[k] })),
    ),
  );
}

function actionButton(text, onclick, { primary = false, busyKey = null } = {}) {
  const busy = busyKey && state.busy.get(busyKey);
  return el("button", { type: "button", class: primary ? "primary small" : "small", disabled: !!busy, onclick }, busy || text);
}

function card(work) {
  const label = LABELS[work.label] || LABELS.consider;
  const isLatest = work.firstRunId && work.firstRunId === state.view.lastRun?.runId;
  const expanded = state.expanded.has(work.id);
  const meta = [
    SOURCES[work.source] === "期刊" ? work.venue : SOURCES[work.source] || work.source,
    formatDate(work.published),
    work.impactFactor ? `IF ${work.impactFactor}` : null,
  ].filter(Boolean);

  const actions = [];
  if (work.status === "new") {
    actions.push(actionButton("加入 Zotero", () => addToZotero(work), { primary: true, busyKey: `${work.id}:add` }));
    actions.push(actionButton("忽略", () => setStatus(work, "dismissed")));
  } else if (work.status === "added") {
    if (work.itemID) actions.push(actionButton("在 Zotero 中显示", () => api().showItem(work.itemID)));
    actions.push(actionButton("移回待查看", () => setStatus(work, "new")));
  } else {
    actions.push(actionButton("恢复", () => setStatus(work, "new")));
  }
  if (!work.summary && state.view.llmEnabled) {
    actions.push(actionButton("AI 摘要", () => summarize(work), { busyKey: `${work.id}:summary` }));
  }
  if (work.url) actions.push(actionButton("打开原文", () => api().openUrl(work.url)));

  const error = state.errors.get(work.id);
  return el(
    "article",
    { class: `card ${label.cls}` },
    el(
      "div",
      { class: "card-meta" },
      el("span", { class: `badge ${label.cls}`, text: label.text }),
      el("span", { class: "score", title: `相似度 ${((work.similarity || 0) * 100).toFixed(1)}`, text: `${Math.round(work.score * 100)} 分` }),
      isLatest && work.status === "new" ? el("span", { class: "badge fresh", text: "新" }) : null,
      el("span", { class: "muted", text: meta.join(" · ") }),
    ),
    el("h2", { class: "title" }, el("a", { href: "#", onclick: (e) => (e.preventDefault(), api().openUrl(work.url)), text: work.title })),
    work.translatedTitle ? el("div", { class: "translated", text: work.translatedTitle }) : null,
    el("div", { class: "authors muted", text: authorLine(work.authors) }),
    work.abstract
      ? el("p", {
          class: expanded ? "abstract" : "abstract clamped",
          title: expanded ? "点击收起" : "点击展开摘要",
          text: work.abstract,
          onclick: () => {
            expanded ? state.expanded.delete(work.id) : state.expanded.add(work.id);
            renderList();
          },
        })
      : null,
    summaryBlock(work),
    error ? el("div", { class: "error", text: error }) : null,
    el("div", { class: "card-actions" }, actions),
  );
}

function emptyState() {
  const { view } = state;
  if (state.tab !== "new") return el("div", { class: "empty", text: "这里还没有内容。" });
  if (!view.lastRun) {
    return el(
      "div",
      { class: "empty" },
      el("h2", { text: "还没有推荐" }),
      el("p", { text: "点击右上角「立即检查」，ZotWatch 会分析你的文库并抓取最近的新文献。首次运行需要向量化整个文库，可能需要几分钟。" }),
    );
  }
  return el("div", { class: "empty" }, el("h2", { text: "全部看完了 🎉" }), el("p", { text: "没有待查看的推荐。新的文献会在下次检查时出现。" }));
}

function renderList() {
  const list = $("list");
  const scroll = list.scrollTop;
  const works = visibleWorks();
  list.replaceChildren(...(works.length ? works.map(card) : [emptyState()]));
  list.scrollTop = scroll;
}

function render() {
  if (!state.view) return;
  renderHeader();
  renderList();
}

// --------------------------------------------------------------- actions

async function withBusy(key, label, workId, fn) {
  state.busy.set(key, label);
  state.errors.delete(workId);
  renderList();
  try {
    await fn();
  } catch (e) {
    state.errors.set(workId, `出错：${e.message || e}`);
    Zotero.logError(e);
  } finally {
    state.busy.delete(key);
    await refresh();
  }
}

function addToZotero(work) {
  return withBusy(`${work.id}:add`, "正在加入…", work.id, () => api().addToLibrary(work.id));
}

function summarize(work) {
  return withBusy(`${work.id}:summary`, "生成中…", work.id, () => api().summarize(work.id));
}

async function setStatus(work, status) {
  await api().setStatus(work.id, status);
  await refresh();
}

async function dismissVisible() {
  const works = visibleWorks();
  if (!works.length) return;
  const message = `将当前列表中的 ${works.length} 篇推荐全部标记为已忽略？`;
  const ok = typeof Services !== "undefined" ? Services.prompt.confirm(window, "ZotWatch", message) : window.confirm(message);
  if (!ok) return;
  for (const w of works) await api().setStatus(w.id, "dismissed");
  await refresh();
}

/** Wire a group of .seg buttons as a single-choice control. */
function segmented(id, onSelect) {
  const group = $(id);
  for (const button of group.querySelectorAll(".seg")) {
    button.addEventListener("click", () => {
      for (const b of group.querySelectorAll(".seg")) b.classList.toggle("active", b === button);
      onSelect(button.dataset.value);
      renderList();
    });
  }
}

function onAddonEvent(event) {
  if (event.type === "progress") {
    renderProgress(true, event.progress);
    return;
  }
  if (event.type === "start") {
    renderProgress(true, null);
    $("notice").hidden = true;
    return;
  }
  if (event.type === "error") {
    refresh().then(() => {
      const notice = $("notice");
      notice.hidden = false;
      notice.textContent = `运行失败：${event.message}`;
    });
    return;
  }
  refresh();
}

window.addEventListener("load", () => {
  $("run").addEventListener("click", () => api()?.runNow().catch(() => {}));
  $("prefs").addEventListener("click", () => api()?.openPreferences());
  $("cancel").addEventListener("click", () => api()?.cancelRun());
  $("dismiss-visible").addEventListener("click", dismissVisible);
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      state.tab = tab.dataset.tab;
      render();
    });
  }
  segmented("label-filter", (value) => (state.filter = value));
  segmented("sort", (value) => (state.sort = value));
  $("search").addEventListener("input", (e) => {
    state.query = e.target.value;
    renderList();
  });

  api()?.addListener(onAddonEvent);
  refresh().catch((e) => {
    $("subtitle").textContent = `加载失败：${e.message}`;
    Zotero.logError(e);
  });
});

window.addEventListener("unload", () => {
  api()?.removeListener(onAddonEvent);
});
