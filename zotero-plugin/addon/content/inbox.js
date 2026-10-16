/* global Zotero, window, document, $, el, svgIcon, emptyState, segmented, setSegmented, refresh, navigate, setSubtitle,
   registerPage, confirmAction, api, trackOf, errorOrigin, formatDate, formatRelative, formatNumber, TRACK_NAMES */

// 推荐 page: the recommendations inbox (待查看 / 已加入 / 已忽略) with track
// sections, filters, sort and search, and per-card actions (add to Zotero,
// dismiss, AI summary, show item, open the article). Uses the helpers of main.js.

var ZotWatchInbox = (() => {
  // Badge of each label: priority for personal works, the track for the others
  const LABELS = {
    must_read: { text: "必读", cls: "must", title: "必读：与文库最相关" },
    consider: { text: "推荐", cls: "consider", title: null },
    ignore: { text: "一般", cls: "ignore", title: null },
    flagship: { text: "顶刊地学", cls: "flagship", title: null },
    followed: { text: "关注", cls: "followed", title: "关注作者的新论文" },
  };
  // Sections of the "all" view, in display order
  const SECTIONS = [
    ["personal", "个性推荐"],
    ["flagship", "顶刊地学速览"],
    ["followed", "关注作者"],
  ];
  const SOURCES = { crossref: "期刊", arxiv: "arXiv", eartharxiv: "EarthArXiv", openalex: "OpenAlex" };
  // Sources whose works show their journal instead of the source name
  const VENUE_SOURCES = new Set(["crossref", "openalex"]);
  const SUMMARY_FIELDS = [
    ["research_question", "研究问题"],
    ["methodology", "研究方法"],
    ["key_findings", "主要发现"],
    ["innovation", "创新点"],
    ["relevance_note", "相关性"],
  ];
  const TABS = ["new", "added", "dismissed"];

  const state = {
    view: null,
    tab: "new",
    track: "all",
    filter: "all",
    sort: "score",
    query: "",
    expanded: new Set(),
    busy: new Map(),
    errors: new Map(),
  };

  function sourceLabel(work) {
    if (VENUE_SOURCES.has(work.source)) return work.venue || (work.source === "crossref" ? null : SOURCES[work.source]);
    return SOURCES[work.source] || work.source;
  }

  function authorLine(authors) {
    const names = (authors || []).map((a) => a.name).filter(Boolean);
    return names.length > 6 ? `${names.slice(0, 6).join(", ")} 等 ${names.length} 人` : names.join(", ");
  }

  const isFiltered = () => state.track !== "all" || state.filter !== "all" || !!state.query.trim();

  function visibleWorks() {
    const { view } = state;
    if (!view) return [];
    const latestRun = view.lastRun?.runId;
    const q = state.query.trim().toLowerCase();
    let works = view.works.filter((w) => w.status === state.tab);
    if (state.track !== "all") works = works.filter((w) => trackOf(w) === state.track);
    if (state.filter === "must_read") works = works.filter((w) => w.label === "must_read");
    if (state.filter === "latest") works = works.filter((w) => w.firstRunId === latestRun);
    if (q) {
      works = works.filter((w) =>
        [w.title, w.translatedTitle, w.venue, authorLine(w.authors), (w.followedAuthors || []).join(" "), w.topic, w.abstract].some((s) =>
          (s || "").toLowerCase().includes(q),
        ),
      );
    }
    const byDate = (w) => Date.parse(w.published || "") || 0;
    const byScore = (w) => w.score || 0;
    // Followed-only works all score 0: newest first among equal scores
    works.sort(
      state.sort === "date"
        ? (a, b) => byDate(b) - byDate(a) || byScore(b) - byScore(a)
        : (a, b) => byScore(b) - byScore(a) || byDate(b) - byDate(a),
    );
    return works;
  }

  // -------------------------------------------------------------- header

  /** One-line summary of the last run for the page header. */
  function subtitle(view) {
    const run = view.lastRun;
    const parts = [];
    if (run) {
      const stats = run.stats;
      const byLabel = stats.newByLabel;
      parts.push(`上次检查 ${formatRelative(run.finishedAt || run.runAt)}`);
      parts.push(`抓取 ${formatNumber(stats.fetched ?? 0)} 篇，推荐 ${stats.recommended ?? 0} 篇（新增 ${byLabel?.personal ?? stats.newRecommendations ?? 0}）`);
      if (byLabel?.flagship) parts.push(`顶刊 ${byLabel.flagship} 篇`);
      if (byLabel?.followed) parts.push(`关注作者 ${byLabel.followed} 篇`);
    } else {
      parts.push("尚未运行");
    }
    parts.push(view.provider === "local" ? "向量化：本地模式" : `向量化：${view.provider}`);
    return parts.join(" · ");
  }

  function callout(tone, icon, title, body, action = null) {
    const node = $("inbox-notice");
    node.className = `callout ${tone}`;
    node.replaceChildren(svgIcon(icon), el("div", { class: "callout-body" }, el("div", { class: "callout-title", text: title }), body));
    if (action) node.append(el("div", { class: "callout-actions" }, el("button", { type: "button", class: "btn small", onclick: action.onclick }, action.label)));
    node.hidden = false;
  }

  /** Run problems, an empty fetch, or the local-mode hint above the list. */
  function renderNotice() {
    const { view } = state;
    const run = view.lastRun;
    const errors = run?.errors || [];
    const node = $("inbox-notice");
    if (errors.length) {
      callout(
        "warning",
        "alert",
        `上次运行有 ${errors.length} 个问题`,
        el("ul", { class: "callout-list" }, errors.map((e) => el("li", { text: `${errorOrigin(e)}：${e.message}` }))),
      );
    } else if (run && !run.stats.fetched && !run.stats.followed?.fetched) {
      callout(
        "warning",
        "alert",
        "上次运行没有抓取到任何文献",
        el("div", { class: "callout-text", text: "请确认关注期刊列表（或开启 arXiv / EarthArXiv），并检查网络连接。" }),
        { label: "关注期刊设置", onclick: () => navigate("settings/journals") },
      );
    } else if (view.provider === "local" && view.works.length) {
      callout(
        "info",
        "info",
        "当前使用本地向量化（无需 API Key），推荐质量有限",
        el("div", { class: "callout-text", text: "配置 DashScope 或 Voyage 可显著提升效果。" }),
        { label: "去设置", onclick: () => navigate("settings/embedding") },
      );
    } else {
      node.hidden = true;
      node.replaceChildren();
    }
  }

  function renderHeader() {
    const { view } = state;
    setSubtitle("inbox", subtitle(view));
    for (const status of TABS) {
      $(`count-${status}`).textContent = formatNumber(view.works.filter((w) => w.status === status).length);
    }
    for (const tab of document.querySelectorAll("#page-inbox .tab")) {
      const on = tab.dataset.tab === state.tab;
      tab.classList.toggle("active", on);
      tab.setAttribute("aria-selected", on ? "true" : "false");
    }
    $("inbox-dismiss").hidden = state.tab !== "new";
    // Filters are pointless before there is anything to filter
    $("inbox-filters").hidden = !view.works.length;
    renderNotice();
  }

  // --------------------------------------------------------------- cards

  function summaryBlock(work) {
    if (!work.summary) return null;
    return el(
      "div",
      { class: "summary" },
      el("div", { class: "summary-title" }, svgIcon("sparkle"), "AI 摘要"),
      ...SUMMARY_FIELDS.filter(([k]) => work.summary[k]).map(([k, label]) =>
        el("div", { class: "summary-row" }, el("span", { class: "summary-key", text: label }), el("span", { class: "summary-value", text: work.summary[k] })),
      ),
    );
  }

  function actionButton(text, onclick, { variant = "", busyKey = null, icon = null } = {}) {
    const busy = busyKey && state.busy.get(busyKey);
    return el(
      "button",
      { type: "button", class: `btn small ${variant}`.trim(), disabled: !!busy, onclick },
      icon && !busy ? svgIcon(icon) : null,
      busy || text,
    );
  }

  /**
   * Score shown on a card: personal works show their score; flagship works
   * show their (muted) library similarity, as in the saved note; followed-only
   * works have none.
   */
  function scoreBadge(work) {
    const track = trackOf(work);
    if (track === "followed") return null;
    if (track === "flagship") {
      return el("span", { class: "score muted", title: "与文库相似度（仅供参考）", text: `相似度 ${Math.round((work.similarity || 0) * 100)}` });
    }
    return el("span", { class: "score", title: `相似度 ${((work.similarity || 0) * 100).toFixed(1)}`, text: `${Math.round((work.score || 0) * 100)} 分` });
  }

  /** Tooltip with the flagship gate details ({pos, neg, judged}). */
  function gateTitle(gate) {
    if (!gate || typeof gate.pos !== "number") return null;
    const neg = typeof gate.neg === "number" ? `，排除领域 ${gate.neg}` : "";
    return `地学相关度 ${gate.pos}${neg}${gate.judged ? "，经大模型复核" : ""}`;
  }

  function chips(work) {
    const out = [];
    const names = (work.followedAuthors || []).filter(Boolean);
    if (names.length) out.push(el("span", { class: "chip followed", title: "关注作者" }, el("span", { class: "dot followed" }), `关注：${names.join("、")}`));
    if (work.topic) out.push(el("span", { class: "chip topic", title: "最接近的研究方向" }, `方向：${work.topic}`));
    return out;
  }

  function badge(work) {
    const label = LABELS[work.label] || LABELS.consider;
    const track = trackOf(work);
    const title = track === "flagship" ? gateTitle(work.gate) : label.title;
    if (track !== "personal") return el("span", { class: `badge ${label.cls}`, title }, el("span", { class: `dot ${track}` }), label.text);
    return el("span", { class: `badge ${label.cls}`, title, text: label.text });
  }

  function card(work) {
    const label = LABELS[work.label] || LABELS.consider;
    const track = trackOf(work);
    const isLatest = work.firstRunId && work.firstRunId === state.view.lastRun?.runId;
    const expanded = state.expanded.has(work.id);
    const meta = [sourceLabel(work), formatDate(work.published), work.impactFactor ? `IF ${work.impactFactor}` : null].filter(Boolean);

    const actions = [];
    if (work.status === "new") {
      actions.push(actionButton("加入 Zotero", () => addToZotero(work), { variant: "primary", busyKey: `${work.id}:add`, icon: "plus" }));
      actions.push(actionButton("忽略", () => setStatus(work, "dismissed")));
    } else if (work.status === "added") {
      if (work.itemID) actions.push(actionButton("在 Zotero 中显示", () => api().showItem(work.itemID)));
      actions.push(actionButton("移回待查看", () => setStatus(work, "new")));
    } else {
      actions.push(actionButton("恢复", () => setStatus(work, "new")));
    }
    if (!work.summary && state.view.llmEnabled) {
      actions.push(actionButton("AI 摘要", () => summarize(work), { busyKey: `${work.id}:summary`, icon: "sparkle" }));
    }
    if (work.url) actions.push(actionButton("打开原文", () => api().openUrl(work.url), { variant: "ghost", icon: "external" }));

    const error = state.errors.get(work.id);
    const workChips = chips(work);
    return el(
      "article",
      { class: `card work ${label.cls}`, "data-track": track },
      el(
        "div",
        { class: "card-meta" },
        badge(work),
        isLatest && work.status === "new" ? el("span", { class: "badge fresh", title: "本次检查新增", text: "新" }) : null,
        el("span", { class: "meta-text", text: meta.join(" · ") }),
        scoreBadge(work),
      ),
      el(
        "h2",
        { class: "title" },
        el("a", { href: work.url || "#", onclick: (e) => (e.preventDefault(), e.stopPropagation(), api().openUrl(work.url)), text: work.title }),
      ),
      work.translatedTitle ? el("div", { class: "translated", text: work.translatedTitle }) : null,
      el("div", { class: "authors", text: authorLine(work.authors) }),
      workChips.length ? el("div", { class: "chips" }, workChips) : null,
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
      error ? el("div", { class: "card-error", role: "alert" }, svgIcon("alert"), el("span", { text: error })) : null,
      el("div", { class: "card-actions" }, actions),
    );
  }

  function clearFilters() {
    state.track = "all";
    state.filter = "all";
    state.query = "";
    setSegmented("inbox-track", "all");
    setSegmented("inbox-label", "all");
    $("inbox-search").value = "";
    renderList();
  }

  function emptyNode() {
    const { view } = state;
    if (isFiltered() && view.works.some((w) => w.status === state.tab)) {
      return emptyState({ icon: "search", title: "没有符合当前筛选条件的文献。", action: { label: "清除筛选", onclick: clearFilters }, compact: true });
    }
    if (state.tab === "added") return emptyState({ icon: "plus-square", title: "还没有加入 Zotero 的推荐", text: "在「待查看」中点「加入 Zotero」后会出现在这里。" });
    if (state.tab === "dismissed") return emptyState({ icon: "inbox", title: "没有已忽略的推荐" });
    if (!view.lastRun) {
      return emptyState({
        icon: "inbox",
        title: "还没有推荐",
        text: "ZotWatch 会分析你的文库并抓取最近的新文献。首次运行需要向量化整个文库，可能需要几分钟。",
        action: { label: "立即检查", primary: true, onclick: () => api()?.runNow().catch(() => {}) },
      });
    }
    return emptyState({ icon: "check-circle", title: "全部看完了", text: "没有待查看的推荐，新的文献会在下次检查时出现。" });
  }

  /** Cards grouped by track, each group under a header with its count (empty groups omitted). */
  function sectioned(works) {
    const nodes = [];
    for (const [track, title] of SECTIONS) {
      const group = works.filter((w) => trackOf(w) === track);
      if (!group.length) continue;
      nodes.push(
        el(
          "div",
          { class: `section-header ${track}` },
          el("span", { class: `dot ${track}` }),
          el("span", { class: "section-title", text: title }),
          el("span", { class: "section-count", text: `${group.length} 篇` }),
        ),
        ...group.map(card),
      );
    }
    return nodes;
  }

  function renderList() {
    const list = $("inbox-list");
    const scroller = $("pages");
    const scroll = scroller.scrollTop;
    const works = visibleWorks();
    let nodes;
    if (!works.length) nodes = [emptyNode()];
    else if (state.track === "all") nodes = sectioned(works);
    else nodes = works.map(card);
    list.replaceChildren(...nodes);
    scroller.scrollTop = scroll;
  }

  function render() {
    if (!state.view) return;
    renderHeader();
    renderList();
  }

  // ------------------------------------------------------------- actions

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
    if (!confirmAction(`将当前列表中的 ${works.length} 篇推荐全部标记为已忽略？`)) return;
    await api().setStatus(works.map((w) => w.id), "dismissed");
    await refresh();
  }

  function focusSearch() {
    const input = $("inbox-search");
    input.focus();
    input.select?.();
  }

  // ---------------------------------------------------------- page hooks

  function init() {
    for (const tab of document.querySelectorAll("#page-inbox .tab")) {
      tab.addEventListener("click", () => {
        state.tab = tab.dataset.tab;
        render();
      });
    }
    segmented("inbox-track", (value) => {
      state.track = value;
      renderList();
    });
    segmented("inbox-label", (value) => {
      state.filter = value;
      renderList();
    });
    segmented("inbox-sort", (value) => {
      state.sort = value;
      renderList();
    });
    $("inbox-search").addEventListener("input", (e) => {
      state.query = e.target.value;
      renderList();
    });
    $("inbox-dismiss").addEventListener("click", dismissVisible);
  }

  function show() {}

  function refreshView(view) {
    state.view = view;
    render();
  }

  return { init, show, refresh: refreshView, focusSearch, state, render, renderList, visibleWorks, dismissVisible, subtitle };
})();

registerPage("inbox", ZotWatchInbox);
