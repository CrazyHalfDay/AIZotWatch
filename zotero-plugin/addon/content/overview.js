/* global window, document, $, el, svgIcon, emptyState, api, registerPage, setSubtitle, navigate, toast, errorOrigin,
   formatNumber, formatCompact, formatRelative, formatUntil, formatDuration, formatDateTime, formatMonthDay,
   TRACK_NAMES, SVG_NS, Zotero */

// 概览 page: the dashboard. Data comes from api().getDashboard(), loaded when
// the page is shown and after runs or inbox changes; while it reloads the
// previous render stays at reduced opacity (no skeleton flash).
//
// Charts are inline SVG drawn by hand and sized by their container (one
// viewBox unit = one pixel, redrawn when the width changes). Every chart has
// hover and keyboard-focus tooltips on its marks (one tab stop per chart,
// arrow keys move between marks) and a 表格 toggle that swaps it for a table
// with the same numbers. Marks follow the shared chart rules: thin bars with a
// 4px rounded data end, a 2px surface gap between stacked segments, hairline
// grid and baseline, text in ink tokens and colour by track. Geometry and
// formatting live in pure helpers, exported on ZotWatchOverview for the tests.

var ZotWatchOverview = (() => {
  const TRACKS = ["personal", "flagship", "followed"];
  const MAX_RUNS = 14;
  const MAX_PROBLEMS = 5;
  const GAP = 2; // surface gap between stacked segments
  const RADIUS = 4; // rounded data end of bars and columns
  const DEFAULT_WIDTH = 560; // chart width when the container cannot be measured
  // Stages of the personal pipeline as recorded in lastRun.stats. 抓取 and 去重后
  // include the flagship journals' articles, which then leave for their own track
  const FUNNEL = [
    ["fetched", "抓取"],
    ["afterDedupe", "去重后"],
    ["afterSplit", "分出顶刊后"],
    ["afterKeywords", "关键词过滤后"],
    ["afterAbstract", "有摘要"],
    ["afterGate", "通过相似度门槛"],
    ["afterRecent", "最近 N 天"],
    ["recommended", "推荐"],
  ];

  const state = {
    dash: null,
    view: null,
    loadedView: null,
    loading: null,
    queued: false,
    error: null,
    tables: new Set(),
    problemsOpen: false,
    connections: null,
    drawers: [],
    width: 0,
    frame: 0,
    loadingTimer: null,
    // Nodes with relative times, updated by tick() without a reload
    live: null,
  };

  // ------------------------------------------------------------ pure helpers

  const round1 = (n) => Math.round(n * 10) / 10;
  const count = (n) => (Number.isFinite(n) && n > 0 ? n : 0);

  /** Approximate rendered width of `text` at `fontPx` (CJK = 1em, digits ≈ 0.58em); errs wide. */
  function textWidth(text, fontPx = 12) {
    let em = 0;
    for (const ch of String(text ?? "")) {
      const code = ch.codePointAt(0);
      if (code >= 0x2e80) em += 1;
      else if (ch >= "0" && ch <= "9") em += 0.58;
      else if (ch >= "A" && ch <= "Z") em += 0.68;
      else if (ch >= "a" && ch <= "z") em += 0.54;
      else if (ch === " ") em += 0.3;
      else em += 0.38;
    }
    return em * fontPx;
  }

  /** `text` shortened with "…" to fit `maxWidth` (by textWidth). */
  function truncateText(text, maxWidth, fontPx = 12) {
    const value = String(text ?? "");
    if (textWidth(value, fontPx) <= maxWidth) return value;
    const chars = [...value];
    while (chars.length > 1 && textWidth(`${chars.join("")}…`, fontPx) > maxWidth) chars.pop();
    return `${chars.join("").trimEnd()}…`;
  }

  /** Axis tick label: 0 / 10 / 1.5K. */
  function formatTick(n) {
    return Math.abs(n) >= 1000 ? `${round1(n / 1000)}K` : String(n);
  }

  /** "10-16 09:30" within the current year, otherwise "2025-10-16 09:30". */
  function formatShortDateTime(ms, now = Date.now()) {
    if (!ms) return "";
    const full = formatDateTime(ms);
    return new Date(ms).getFullYear() === new Date(now).getFullYear() ? full.slice(5) : full;
  }

  /** 93.5% (one decimal, none for whole numbers). */
  function formatPercent(ratio) {
    return Number.isFinite(ratio) ? `${round1(ratio * 100)}%` : "–";
  }

  /**
   * Clean count axis from 0: a step of 1 / 2 / 5 × 10^k (at least 1) giving at
   * most `steps` gridlines above the baseline. Returns {max, step, ticks}.
   */
  function niceScale(maxValue, steps = 3) {
    const top = Math.max(1, Number.isFinite(maxValue) ? maxValue : 0);
    const raw = top / steps;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const step = Math.max(1, [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw));
    const max = Math.ceil(top / step) * step;
    const ticks = [];
    for (let v = 0; v <= max; v += step) ticks.push(v);
    return { max, step, ticks };
  }

  /** The last `limit` runs of the history as chart rows, oldest first. */
  function historySeries(history, limit = MAX_RUNS) {
    const valid = (Array.isArray(history) ? history : []).filter((h) => h && Number.isFinite(h.runAt));
    return valid.slice(-limit).map((h) => {
      const by = h.newByLabel || {};
      const row = {
        runId: h.runId,
        runAt: h.runAt,
        finishedAt: h.finishedAt ?? null,
        personal: count(by.personal),
        flagship: count(by.flagship),
        followed: count(by.followed),
        mustRead: count(h.newMustRead),
        errors: count(h.errors),
      };
      row.total = row.personal + row.flagship + row.followed;
      return row;
    });
  }

  /** Series of the run chart: personal, plus each track that is enabled or has values. */
  function chartTracks(rows, { flagship = false, followed = false } = {}) {
    const on = { personal: true, flagship, followed };
    return TRACKS.filter((t) => on[t] || rows.some((r) => r[t] > 0));
  }

  /** Every `step`-th x label fits; counted from the newest so the latest run is always labelled. */
  function labelStep(n, band, labelWidth) {
    if (n <= 1 || !(band > 0)) return 1;
    return Math.max(1, Math.ceil(labelWidth / band));
  }

  /** SVG path of a column: rounded top (the data end), square base. */
  function columnPath(x, y, w, h, r = RADIUS) {
    const rr = round1(Math.max(0, Math.min(r, w / 2, h)));
    const [x0, y0, x1, y1] = [x, y, x + w, y + h].map(round1);
    return `M${x0},${y1}V${round1(y0 + rr)}A${rr},${rr} 0 0 1 ${round1(x0 + rr)},${y0}H${round1(x1 - rr)}A${rr},${rr} 0 0 1 ${x1},${round1(y0 + rr)}V${y1}Z`;
  }

  /** SVG path of a horizontal bar: rounded right end (the data end), square base. */
  function barPath(x, y, w, h, r = RADIUS) {
    const rr = round1(Math.max(0, Math.min(r, h / 2, w)));
    const [x0, y0, x1, y1] = [x, y, x + w, y + h].map(round1);
    return `M${x0},${y0}H${round1(x1 - rr)}A${rr},${rr} 0 0 1 ${x1},${round1(y0 + rr)}V${round1(y1 - rr)}A${rr},${rr} 0 0 1 ${round1(x1 - rr)},${y1}H${x0}Z`;
  }

  /**
   * Geometry of the stacked run chart (1 unit = 1 px): the plot box, y ticks
   * and one column per run. Columns are at most 24px wide; segments stack
   * personal → flagship → followed with a 2px gap and only the top segment
   * gets the rounded end. A segment with a value is at least 2px tall.
   */
  function columnLayout(rows, { width = DEFAULT_WIDTH, height = 200, fontPx = 11, tracks = TRACKS } = {}) {
    const scale = niceScale(Math.max(0, ...rows.map((r) => r.total)));
    const tickLabels = scale.ticks.map(formatTick);
    const left = Math.ceil(Math.max(...tickLabels.map((t) => textWidth(t, fontPx)))) + 10;
    const top = Math.ceil(fontPx * 1.8);
    const bottom = Math.ceil(fontPx * 2.1);
    const plot = { x: left, y: top, w: Math.max(40, width - left - 2), h: Math.max(40, height - top - bottom) };
    const baseY = plot.y + plot.h;
    const band = plot.w / Math.max(1, rows.length);
    const colW = Math.max(4, Math.min(24, Math.floor(band * 0.6)));
    const scaled = (v) => (v / scale.max) * plot.h;
    const step = labelStep(rows.length, band, textWidth("00-00", fontPx) + 10);
    const columns = rows.map((row, i) => {
      const cx = plot.x + band * (i + 0.5);
      const x = Math.round(cx - colW / 2);
      const visible = tracks.filter((t) => row[t] > 0);
      const segments = [];
      let y = baseY;
      visible.forEach((track, k) => {
        const gap = k > 0 ? GAP : 0;
        const h = Math.max(2, scaled(row[track]) - gap);
        y -= gap + h;
        segments.push({ track, value: row[track], x, y: round1(y), w: colW, h: round1(h), top: k === visible.length - 1 });
      });
      return {
        row,
        index: i,
        x,
        w: colW,
        cx: round1(cx),
        bandX: round1(plot.x + band * i),
        band: round1(band),
        top: round1(y),
        segments,
        label: formatMonthDay(row.runAt),
        showLabel: (rows.length - 1 - i) % step === 0,
      };
    });
    const ticks = scale.ticks.map((value, i) => ({ value, y: Math.round(baseY - scaled(value)), label: tickLabels[i] }));
    return { plot, baseY, scale, ticks, columns, band: round1(band), height };
  }

  /**
   * Geometry of a horizontal bar list (1 unit = 1 px): stage / topic labels on
   * the left, bars from a shared baseline (at most 16px thick) and the value
   * either at the bar tip ("tip") or right-aligned at the end ("end").
   * Rows are {label, value, valueText, note}.
   */
  function barListLayout(rows, { width = DEFAULT_WIDTH, fontPx = 12, valueAt = "tip", labelShare = 0.42, rowFactor = 2.3 } = {}) {
    const rowH = Math.round(Math.max(24, fontPx * rowFactor));
    const barH = Math.min(16, Math.round(rowH * 0.55));
    const labelSpace = Math.ceil(Math.max(0, ...rows.map((r) => textWidth(r.label, fontPx)))) + 14;
    const x0 = Math.round(Math.min(labelSpace, width * labelShare));
    const valueSpace = Math.ceil(Math.max(0, ...rows.map((r) => textWidth(`${r.valueText}${r.note ? `  ${r.note}` : ""}`, fontPx)))) + 12;
    const barMax = Math.max(24, width - x0 - 1 - valueSpace);
    const max = Math.max(1, ...rows.map((r) => r.value));
    const items = rows.map((r, i) => {
      const w = r.value > 0 ? Math.max(2, (r.value / max) * barMax) : 0;
      const y = i * rowH;
      return {
        ...r,
        index: i,
        y,
        h: rowH,
        barX: x0 + 1,
        barY: y + Math.round((rowH - barH) / 2),
        barW: round1(w),
        barH,
        midY: y + rowH / 2,
        shortLabel: truncateText(r.label, x0 - 14, fontPx),
        valueX: valueAt === "tip" ? round1(x0 + 1 + w + 6) : width,
      };
    });
    return { x0, rowH, barH, barMax: round1(barMax), width, height: rows.length * rowH, rows: items, valueAt };
  }

  /** Personal candidates after the flagship articles left (runs before 0.3.0: derived from the flagship stats). */
  function afterSplitOf(stats) {
    if (Number.isFinite(stats.afterSplit)) return stats.afterSplit;
    const f = stats.flagship;
    return f && Number.isFinite(f.candidates) && Number.isFinite(stats.afterDedupe) ? Math.max(0, stats.afterDedupe - f.candidates) : null;
  }

  /**
   * The filters the last run applied (recorded in its stats since 0.3.0);
   * older runs fall back to the current settings.
   */
  function funnelOptions(stats, scoring = {}) {
    return {
      recentDays: Number.isFinite(stats?.recentDays) ? stats.recentDays : scoring.recentDays ?? null,
      requireAbstract: typeof stats?.requireAbstract === "boolean" ? stats.requireAbstract : scoring.requireAbstract ?? true,
    };
  }

  /**
   * Bars of the selection funnel: the stages of the last run that exist in
   * stats, each with its drop from the previous stage and its share of the
   * first. "分出顶刊后" appears when the flagship track took articles out,
   * "有摘要" is left out when abstracts are not required, "最近 N 天" when the
   * window is unlimited (0).
   */
  function funnelStages(stats, { recentDays = null, requireAbstract = true } = {}) {
    if (!stats) return [];
    const stages = [];
    for (const [key, name] of FUNNEL) {
      const value = key === "afterSplit" ? afterSplitOf(stats) : stats[key];
      if (!Number.isFinite(value)) continue;
      if (key === "afterAbstract" && requireAbstract === false) continue;
      if (key === "afterRecent" && recentDays === 0) continue;
      const label = key === "afterRecent" ? (recentDays ? `最近 ${recentDays} 天` : "时间范围内") : name;
      const prev = stages.at(-1);
      const first = stages[0]?.value ?? value;
      stages.push({ key, label, value, drop: prev ? Math.max(0, prev.value - value) : 0, share: first > 0 ? value / first : 0 });
    }
    return stages;
  }

  /**
   * Steps of the flagship track's own funnel (stats.flagship), [] without
   * stats. 通过 is "–" (未完成) when the gate failed before judging (f.failed).
   */
  function flagshipSteps(f, recentDays = null) {
    if (!f || !Number.isFinite(f.candidates)) return [];
    let passed;
    if (Number.isFinite(f.passed)) passed = f.passed;
    else if (f.failed) passed = null;
    else passed = Math.max(0, count(f.recent) - count(f.rejected)); // runs before 0.3.0
    const review = [`灰区 ${formatNumber(count(f.gray))}`];
    if (f.judged) review.push(`大模型复核 ${formatNumber(f.judged)}`);
    if (f.failed) review.splice(0, review.length, "未完成");
    return [
      { label: "候选", value: f.candidates },
      { label: "有摘要", value: count(f.withAbstract) },
      { label: recentDays ? `最近 ${recentDays} 天` : "最近", value: count(f.recent) },
      { label: "通过", value: passed, note: review.join("，") },
      { label: "显示", value: count(f.recommended) },
    ];
  }

  /** "顶刊：候选 164 → 有摘要 151 → … → 显示 12" ("" without stats). */
  function flagshipText(f, recentDays = null) {
    const steps = flagshipSteps(f, recentDays);
    if (!steps.length) return "";
    // No space between a closing full-width bracket and the arrow
    return `顶刊：${steps.map((s) => `${s.label} ${formatNumber(s.value)}${s.note ? `（${s.note}）` : ""}`).join(" → ").replace(/） →/g, "）→")}`;
  }

  /**
   * Status of a data-source row: {tone, label} (the dot class and its text).
   * `source.ran === false`: enabled, but not part of the last run (turned on since).
   */
  function sourceStatus(source, hasRun) {
    if (!source.enabled) return { tone: "off", label: "未启用" };
    if (source.error) return { tone: "critical", label: "出错" };
    if (!hasRun || source.ran === false) return { tone: "idle", label: "未运行" };
    return { tone: "good", label: "正常" };
  }

  /** A run stopped by 取消 after its results were stored: lastRun without an end time. */
  const wasCancelled = (last) => !!last && last.finishedAt === null;

  /**
   * Run health for the 运行状态 card: {tone, icon, label, problems}, where a
   * failure newer than the last finished run comes first among the problems,
   * followed by the last run's non-fatal errors.
   */
  function runHealth(dash, now = Date.now()) {
    const last = dash.lastRun;
    const finished = last ? last.finishedAt || last.runAt : 0;
    const failure = dash.lastError && (!finished || dash.lastError.at > finished) ? dash.lastError : null;
    const errors = last?.errors || [];
    const problems = [];
    if (failure) problems.push({ tone: "critical", origin: formatRelative(failure.at, now) || "运行失败", message: failure.message || "" });
    for (const e of errors) problems.push({ tone: "warning", origin: errorOrigin(e), message: e.message || "" });
    let status;
    if (dash.running) status = { tone: "running", icon: null, label: "正在检查…" };
    else if (failure) status = { tone: "critical", icon: "alert", label: "运行失败" };
    else if (!last) status = { tone: "idle", icon: "clock", label: "尚未运行" };
    // Its works are stored, but it is missing from the 最近运行 chart (only finished runs are)
    else if (wasCancelled(last)) status = { tone: "warning", icon: "alert", label: "上次运行已取消" };
    else if (errors.length) status = { tone: "warning", icon: "alert", label: `有 ${errors.length} 个问题` };
    else status = { tone: "good", icon: "check-circle", label: "正常" };
    return { ...status, problems, title: failure ? "最近的问题" : "上次运行的问题" };
  }

  /** Lines of the 运行状态 card: [{key, label, text, link}]. */
  function scheduleFacts(dash, now = Date.now()) {
    const last = dash.lastRun;
    const schedule = dash.schedule || {};
    const facts = [];
    if (last) {
      const at = last.finishedAt || last.runAt;
      const cancelled = wasCancelled(last) ? "，已取消" : "";
      facts.push({ key: "last", label: "上次", text: `${formatDateTime(at)}（${formatRelative(at, now)}${cancelled}）` });
      if (last.finishedAt) facts.push({ key: "took", label: "耗时", text: formatDuration(last.finishedAt - last.runAt) });
    } else {
      facts.push({ key: "last", label: "上次", text: "尚未运行" });
    }
    if (!schedule.autoRun) facts.push({ key: "next", label: "下次", text: "自动检查已关闭", link: { href: "#settings/general", text: "去开启" } });
    else if (dash.running) facts.push({ key: "next", label: "下次", text: "正在检查" });
    else if (schedule.nextRunAt) facts.push({ key: "next", label: "下次", text: schedule.nextRunAt <= now ? "即将检查" : formatUntil(schedule.nextRunAt, now) });
    if (schedule.autoRun && schedule.intervalHours) facts.push({ key: "interval", label: "间隔", text: `每 ${formatNumber(schedule.intervalHours)} 小时自动检查` });
    return facts;
  }

  /** Track parts of the pending count (personal always, the others when they hold works). */
  function pendingParts(pending = {}) {
    return TRACKS.filter((t) => t === "personal" || pending[t] > 0).map((t) => ({ track: t, label: TRACK_NAMES[t].short, count: count(pending[t]) }));
  }

  /** The four KPI tiles: [{key, label, value | text, caption}] (value null = no run yet). */
  function kpiTiles(dash, now = Date.now()) {
    const last = dash.lastRun;
    const stats = last?.stats || {};
    const by = stats.newByLabel || {};
    const shown = [["personal", true], ["flagship", dash.flagship?.enabled], ["followed", dash.followed?.enabled]].filter(
      ([t, on]) => on || by[t] > 0,
    );
    const breakdown = shown.map(([t]) => `${TRACK_NAMES[t].short} ${formatNumber(count(by[t]))}`).join(" · ");
    const finished = last ? last.finishedAt || last.runAt : 0;
    const took = last?.finishedAt ? formatDuration(last.finishedAt - last.runAt) : "";
    return [
      { key: "new", label: "本次新增", value: last ? count(stats.newRecommendations) : null, caption: last ? (wasCancelled(last) ? `已取消 · ${breakdown}` : breakdown) : "尚未运行" },
      { key: "must", label: "必读", value: last ? count(stats.newMustRead) : null, caption: "本次新增中的必读" },
      { key: "added", label: "已加入 Zotero", value: count(dash.added), caption: `近 ${dash.addedDays || 60} 天` },
      {
        key: "last",
        label: "上次检查",
        text: last ? formatRelative(finished, now) : "尚未运行",
        caption: last ? `${formatShortDateTime(finished, now)}${took ? ` · 耗时 ${took}` : ""}` : "点击「立即检查」开始",
      },
    ];
  }

  /** Page subtitle: library, item counts and the embedding service. */
  function headerLine(dash) {
    const lib = dash.library || {};
    return [
      `文库：${lib.name || "我的文库"}`,
      `${formatNumber(count(lib.items))} 条（${formatNumber(count(lib.withAbstract))} 篇有摘要）`,
      `向量化：${dash.embedding?.label || "本地模式"}`,
    ].join(" · ");
  }

  /** Status of a connection check row: {tone, icon, label}. */
  function connectionStatus(row) {
    if (row.ok === true) return { tone: "good", icon: "check-circle", label: "正常" };
    if (row.ok === false) return { tone: "critical", icon: "alert", label: "失败" };
    return { tone: "off", icon: "info", label: "未启用" };
  }

  // --------------------------------------------------------- DOM building

  function svg(tag, attrs = {}, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, String(v));
    }
    for (const c of children.flat()) if (c) node.append(typeof c === "string" ? document.createTextNode(c) : c);
    return node;
  }

  const fontSizeOf = (node) => {
    try {
      return parseFloat(window.getComputedStyle(node).fontSize) || 12;
    } catch (e) {
      return 12;
    }
  };

  /** A card with a title, an optional caption and header actions; `body` replaces the default body element. */
  function card(cls, title, { caption = null, actions = [], body: bodyNode = null } = {}, ...body) {
    return el(
      "section",
      { class: `card ov-card ${cls}`, "aria-label": title },
      el(
        "header",
        { class: "card-header ov-card-header" },
        el("div", { class: "ov-heading" }, el("h2", { class: "card-title", text: title }), caption ? el("p", { class: "card-caption", text: caption }) : null),
        actions.length ? el("div", { class: "ov-actions" }, ...actions) : null,
      ),
      bodyNode || el("div", { class: "card-body ov-card-body" }, ...body),
    );
  }

  function linkButton(label, route) {
    return el("button", { type: "button", class: "btn ghost small", onclick: () => navigate(route) }, label);
  }

  // ---------------------------------------------------------------- tooltips

  function showTip(host, tip, nodes, anchor) {
    tip.replaceChildren(...nodes);
    tip.hidden = false;
    const hostW = host.clientWidth || 0;
    const w = tip.offsetWidth || 0;
    const h = tip.offsetHeight || 0;
    const left = Math.max(0, Math.min(anchor.x - w / 2, hostW - w));
    // Above the mark (it may cover the card header); below it when that would leave the visible page
    let roomAbove = anchor.y;
    try {
      roomAbove += host.getBoundingClientRect().top - $("pages").getBoundingClientRect().top;
    } catch (e) {
      // Not laid out (tests): measure from the chart's top
    }
    const top = roomAbove >= h + 14 ? anchor.y - h - 10 : anchor.bottom + 10;
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top = `${Math.round(top)}px`;
  }

  function hideTip(tip) {
    tip.hidden = true;
  }

  /**
   * Hover and keyboard focus for the marks of one chart. Each mark is
   * {hit, group, label, tip(), anchor}; the chart is one tab stop (the first
   * mark) and the arrow keys, Home and End move between marks.
   */
  function bindMarks(host, tip, marks, tabStop = 0) {
    let current = -1;
    const activate = (i, focused) => {
      if (current >= 0 && current !== i) marks[current].group.classList.remove("active", "focused");
      current = i;
      marks[i].group.classList.add("active");
      if (focused) marks[i].group.classList.add("focused");
      showTip(host, tip, marks[i].tip(), marks[i].anchor);
    };
    const deactivate = (i) => {
      marks[i].group.classList.remove("active", "focused");
      if (current === i) {
        current = -1;
        hideTip(tip);
      }
    };
    marks.forEach((mark, i) => {
      const { hit } = mark;
      hit.setAttribute("tabindex", i === tabStop ? "0" : "-1");
      hit.setAttribute("role", "img");
      hit.setAttribute("aria-label", mark.label);
      hit.addEventListener("pointerenter", () => activate(i, false));
      hit.addEventListener("pointerleave", () => deactivate(i));
      hit.addEventListener("focus", () => {
        let visible = true;
        try {
          visible = hit.matches(":focus-visible");
        } catch (e) {
          // Older engines: always show the focus ring
        }
        activate(i, visible);
      });
      hit.addEventListener("blur", () => deactivate(i));
      hit.addEventListener("keydown", (event) => {
        const target = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: marks.length - 1 }[event.key];
        if (target === undefined) return;
        event.preventDefault();
        const j = Math.max(0, Math.min(marks.length - 1, target));
        marks.forEach((m, k) => m.hit.setAttribute("tabindex", k === j ? "0" : "-1"));
        marks[j].hit.focus();
      });
    });
  }

  function tipRow(track, value, name) {
    return el(
      "div",
      { class: track ? "tip-row" : "tip-row plain" },
      track ? el("span", { class: `tip-key ${track}` }) : null,
      el("strong", { class: "tip-value", text: value }),
      el("span", { class: "tip-name", text: name }),
    );
  }

  // ------------------------------------------------------------------ charts

  /**
   * A chart block: the SVG (drawn once the container can be measured) or, when
   * the card's 表格 toggle is on, a table. Returns {host, toggle}.
   */
  function chartBlock(key, { draw, table, legend = null, minHeight = 0 }) {
    const host = el("div", { class: "chart", "data-chart": key });
    const toggle = el("button", { type: "button", class: "btn ghost small chart-toggle" });
    const fill = () => {
      const asTable = state.tables.has(key);
      toggle.replaceChildren(svgIcon(asTable ? "chart" : "table"), asTable ? "图表" : "表格");
      toggle.setAttribute("aria-pressed", asTable ? "true" : "false");
      toggle.setAttribute("title", asTable ? "以图表显示" : "以表格显示同样的数字");
      host.classList.toggle("as-table", asTable);
      if (asTable) {
        host.style.minHeight = "";
        host.replaceChildren(el("div", { class: "chart-table" }, table()));
        return true;
      }
      // Keep the chart's height while it waits to be measured (no layout jump)
      if (minHeight) host.style.minHeight = `${minHeight}px`;
      const width = host.clientWidth;
      if (width === 0) return false; // not laid out (hidden page): drawn on the next resize
      const tip = el("div", { class: "chart-tip", role: "tooltip", hidden: true });
      // The legend belongs to the chart (the table has its own column headers)
      const plot = el("div", { class: "chart-plot" });
      host.replaceChildren(...[legend ? legend() : null, plot].filter(Boolean));
      const plotWidth = plot.clientWidth;
      plot.replaceChildren(draw({ width: Math.floor(plotWidth || width || DEFAULT_WIDTH), fontPx: fontSizeOf(plot), host: plot, tip }), tip);
      host.style.minHeight = "";
      return true;
    };
    toggle.addEventListener("click", () => {
      if (state.tables.has(key)) state.tables.delete(key);
      else state.tables.add(key);
      fill();
    });
    state.drawers.push(fill);
    return { host, toggle };
  }

  function dataTable(head, rows, numeric) {
    return el(
      "table",
      { class: "table" },
      el("thead", {}, el("tr", {}, ...head.map((h, i) => el("th", { class: numeric[i] ? "num" : null, scope: "col", text: h })))),
      el("tbody", {}, ...rows.map((cells) => el("tr", {}, ...cells.map((c, i) => el("td", { class: numeric[i] ? "num" : null, text: c }))))),
    );
  }

  /** The stacked run chart. */
  function drawRuns(rows, tracks) {
    return ({ width, fontPx, host, tip }) => {
      const height = Math.round(Math.max(168, fontPx * 15));
      const layout = columnLayout(rows, { width, height, fontPx, tracks });
      const { plot, baseY } = layout;
      const root = svg("svg", {
        class: "chart-svg",
        viewBox: `0 0 ${width} ${height}`,
        width: "100%",
        height,
        role: "group",
        "aria-label": `最近 ${rows.length} 次运行每次新增的推荐`,
      });
      for (const t of layout.ticks) {
        const y = t.y + 0.5;
        root.append(svg("line", { class: t.value === 0 ? "baseline" : "grid", x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y }));
        root.append(svg("text", { class: "tick", x: plot.x - 8, y, dy: "0.35em", "text-anchor": "end", text: t.label }));
      }
      const marks = [];
      for (const col of layout.columns) {
        const group = svg("g", { class: "col" });
        group.append(svg("rect", { class: "hl", x: col.bandX + 1, y: plot.y - 4, width: Math.max(0, col.band - 2), height: plot.h + 4, rx: 4 }));
        for (const s of col.segments) {
          group.append(s.top ? svg("path", { class: `seg ${s.track}`, d: columnPath(s.x, s.y, s.w, s.h) }) : svg("rect", { class: `seg ${s.track}`, x: s.x, y: s.y, width: s.w, height: s.h }));
        }
        if (col.showLabel) {
          group.append(svg("text", { class: "xlabel", x: col.cx, y: baseY + Math.round(fontPx * 1.5), "text-anchor": "middle", text: col.label }));
        }
        // The newest run carries its total on the cap (the only direct label)
        if (col.index === rows.length - 1 && col.row.total > 0) {
          group.append(svg("text", { class: "cap", x: col.cx, y: col.top - 6, "text-anchor": "middle", text: formatNumber(col.row.total) }));
        }
        const hit = svg("rect", { class: "hit", x: col.bandX, y: 0, width: col.band, height });
        group.append(hit);
        root.append(group);
        const r = col.row;
        marks.push({
          hit,
          group,
          label: `${formatDateTime(r.runAt)}：${tracks.map((t) => `${TRACK_NAMES[t].name} ${r[t]}`).join("，")}，合计 ${r.total}${r.mustRead ? `，必读 ${r.mustRead}` : ""}${r.errors ? `，${r.errors} 个问题` : ""}`,
          anchor: { x: col.cx, y: Math.min(col.top, baseY - 8), bottom: baseY },
          tip: () => [
            el("div", { class: "tip-head", text: formatDateTime(r.runAt) }),
            ...tracks.map((t) => tipRow(t, formatNumber(r[t]), TRACK_NAMES[t].name)),
            el(
              "div",
              { class: "tip-foot" },
              `合计 ${formatNumber(r.total)}`,
              r.mustRead ? ` · 必读 ${formatNumber(r.mustRead)}` : "",
              r.errors ? ` · ${r.errors} 个问题` : "",
            ),
          ],
        });
      }
      bindMarks(host, tip, marks, marks.length - 1);
      return root;
    };
  }

  /** A horizontal bar list (funnel stages, research directions). */
  function drawBars(rows, { valueAt = "tip", ariaLabel, tipFor, labelShare, rowFactor }) {
    return ({ width, fontPx, host, tip }) => {
      const layout = barListLayout(rows, { width, fontPx, valueAt, labelShare, rowFactor });
      const root = svg("svg", {
        class: "chart-svg bars",
        viewBox: `0 0 ${width} ${layout.height}`,
        width: "100%",
        height: layout.height,
        role: "group",
        "aria-label": ariaLabel,
      });
      root.append(svg("line", { class: "baseline", x1: layout.x0 + 0.5, x2: layout.x0 + 0.5, y1: 2, y2: layout.height - 2 }));
      const marks = [];
      for (const r of layout.rows) {
        const group = svg("g", { class: "bar-row" });
        group.append(svg("rect", { class: "hl", x: 0, y: r.y + 1, width, height: r.h - 2, rx: 4 }));
        group.append(svg("text", { class: "label", x: 0, y: r.midY, dy: "0.35em", text: r.shortLabel }));
        if (r.barW > 0) group.append(svg("path", { class: "bar", d: barPath(r.barX, r.barY, r.barW, r.barH) }));
        const value = svg("text", { class: "value", x: r.valueX, y: r.midY, dy: "0.35em", "text-anchor": valueAt === "tip" ? "start" : "end" }, r.valueText);
        if (r.note) value.append(svg("tspan", { class: "note", dx: "6", text: r.note }));
        group.append(value);
        const hit = svg("rect", { class: "hit", x: 0, y: r.y, width, height: r.h });
        group.append(hit);
        root.append(group);
        marks.push({
          hit,
          group,
          label: `${r.label}：${r.valueText}${r.note ? `（${r.note}）` : ""}`,
          anchor: { x: Math.min(width - 40, r.barX + r.barW), y: r.y + 2, bottom: r.y + r.h },
          tip: () => tipFor(r),
        });
      }
      bindMarks(host, tip, marks, 0);
      return root;
    };
  }

  // ------------------------------------------------------------------ cards

  function heroTile(dash) {
    const pending = dash.pending || {};
    const total = count(pending.total);
    const parts = pendingParts(pending);
    let detail;
    if (total > 0) {
      // Part-to-whole by track: a thin segmented bar, direct-labelled by the counts below it
      const share = el(
        "div",
        { class: "hero-share", "aria-hidden": "true" },
        ...parts.filter((p) => p.count > 0).map((p) => el("span", { class: `hero-seg ${p.track}`, style: `flex-grow: ${p.count}`, title: `${TRACK_NAMES[p.track].name} ${p.count} 篇` })),
      );
      detail = [
        share,
        el(
          "div",
          { class: "hero-parts" },
          ...parts.map((p) => el("span", { class: "hero-part" }, el("span", { class: `dot ${p.track}` }), `${p.label} ${formatNumber(p.count)}`)),
        ),
      ];
    } else {
      detail = el("p", { class: "hero-note", text: dash.lastRun ? "全部看完了" : "运行后这里显示待查看的推荐" });
    }
    return el(
      "div",
      { class: "tile ov-hero" },
      el("span", { class: "tile-label", text: "待查看" }),
      el(
        "div",
        { class: "hero-row" },
        el("span", { class: "hero-value", text: formatCompact(total) }),
        pending.mustRead > 0 ? el("span", { class: "badge must", text: `必读 ${formatNumber(pending.mustRead)}` }) : null,
      ),
      detail,
      el("a", { class: "tile-link", href: "#inbox" }, total > 0 ? "去查看" : "打开推荐", svgIcon("arrow-right")),
    );
  }

  function kpiTileNodes(dash) {
    return kpiTiles(dash).map((t) =>
      el(
        "div",
        { class: `tile ov-kpi ${t.key}` },
        el("span", { class: "tile-label", text: t.label }),
        el("span", { class: t.text ? "tile-value text" : "tile-value", text: t.text ?? (t.value === null ? "–" : formatCompact(t.value)) }),
        el("span", { class: "tile-caption", text: t.caption }),
      ),
    );
  }

  function kpiGrid(dash) {
    const node = el("div", { class: "ov-kpis" }, ...kpiTileNodes(dash));
    state.live.kpis = node;
    return node;
  }

  function runsCard(dash) {
    const rows = historySeries(dash.history);
    if (!rows.length) {
      return card("ov-runs", "最近运行", { caption: "每次运行新增的推荐" }, emptyState({ icon: "chart", title: "运行几次后这里会显示每次新增的推荐", compact: true }));
    }
    const tracks = chartTracks(rows, { flagship: dash.flagship?.enabled, followed: dash.followed?.enabled });
    const legend =
      tracks.length > 1
        ? () =>
            el(
              "div",
              { class: "legend", role: "list", "aria-label": "图例" },
              ...tracks.map((t) => el("span", { class: "legend-item", role: "listitem" }, el("span", { class: `legend-swatch ${t}` }), TRACK_NAMES[t].name)),
            )
        : null;
    const { host, toggle } = chartBlock("runs", {
      draw: drawRuns(rows, tracks),
      legend,
      minHeight: 196,
      table: () =>
        dataTable(
          ["时间", ...tracks.map((t) => TRACK_NAMES[t].name), "合计", "必读", "问题"],
          [...rows].reverse().map((r) => [formatDateTime(r.runAt), ...tracks.map((t) => formatNumber(r[t])), formatNumber(r.total), formatNumber(r.mustRead), formatNumber(r.errors)]),
          [false, ...tracks.map(() => true), true, true, true],
        ),
    });
    const caption = `${tracks.length > 1 ? "每次运行新增的推荐" : `每次运行新增的${TRACK_NAMES.personal.name}`} · 最近 ${rows.length} 次`;
    return card("ov-runs", "最近运行", { caption, actions: [toggle] }, host);
  }

  function statusCard(dash) {
    const body = el("div", { class: "card-body ov-card-body" }, ...statusParts(dash));
    state.live.status = body;
    return card("ov-status", "运行状态", { body });
  }

  /** Contents of the 运行状态 card (rebuilt by tick() as times pass). */
  function statusParts(dash) {
    const health = runHealth(dash);
    const head = el(
      "div",
      { class: `ov-health ${health.tone}` },
      health.icon ? el("span", { class: `ov-health-icon ${health.tone}` }, svgIcon(health.icon)) : el("span", { class: "ov-health-icon" }, el("span", { class: "dot running" })),
      el("span", { class: "ov-health-label", text: health.label }),
    );
    const facts = el(
      "dl",
      { class: "ov-facts" },
      ...scheduleFacts(dash).flatMap((f) => [
        el("dt", { text: f.label }),
        el("dd", {}, f.text, f.link ? el("a", { class: "ov-inline-link", href: f.link.href, text: f.link.text }) : null),
      ]),
    );
    const body = [head, facts];
    if (health.problems.length) {
      const shown = state.problemsOpen ? health.problems : health.problems.slice(0, MAX_PROBLEMS);
      body.push(
        el(
          "div",
          { class: "ov-problems" },
          el("p", { class: "ov-subhead", text: `${health.title}（${health.problems.length}）` }),
          el(
            "ul",
            { class: "ov-problem-list" },
            ...shown.map((p) =>
              el(
                "li",
                { class: "ov-problem" },
                el("span", { class: `dot ${p.tone}` }),
                el("span", { class: "ov-problem-text" }, el("span", { class: "ov-problem-origin", text: p.origin }), el("span", { class: "ov-problem-message", text: p.message })),
              ),
            ),
          ),
          health.problems.length > MAX_PROBLEMS
            ? el(
                "button",
                {
                  type: "button",
                  class: "link-btn ov-more",
                  onclick: () => {
                    state.problemsOpen = !state.problemsOpen;
                    render();
                  },
                },
                state.problemsOpen ? "收起" : `更多…（还有 ${health.problems.length - MAX_PROBLEMS} 个）`,
              )
            : null,
        ),
      );
    }
    return body;
  }

  function funnelCard(dash) {
    const last = dash.lastRun;
    // Labelled with the filters that run applied, not today's settings
    const options = funnelOptions(last?.stats, dash.scoring || {});
    const stages = funnelStages(last?.stats, options);
    if (!stages.length) {
      return card("ov-funnel", "筛选漏斗", { caption: "上次运行每一步筛选后剩下的篇数" }, emptyState({ icon: "filter", title: "运行一次后显示每一步筛选后剩下的篇数", compact: true }));
    }
    const rows = stages.map((s, i) => ({ ...s, valueText: formatNumber(s.value), note: i > 0 && s.drop > 0 ? `−${formatNumber(s.drop)}` : null }));
    const firstLabel = stages[0].label;
    const { host, toggle } = chartBlock("funnel", {
      draw: drawBars(rows, {
        valueAt: "tip",
        ariaLabel: "上次运行的筛选漏斗",
        tipFor: (r) => [
          el("div", { class: "tip-head", text: r.label }),
          tipRow(null, formatNumber(r.value), "篇"),
          r.index > 0 ? el("div", { class: "tip-foot", text: `较上一步 −${formatNumber(r.drop)} · 占${firstLabel} ${formatPercent(r.share)}` }) : null,
        ].filter(Boolean),
      }),
      minHeight: rows.length * 30,
      table: () =>
        dataTable(
          ["阶段", "篇数", "较上一步", `占${firstLabel}`],
          stages.map((s, i) => [s.label, formatNumber(s.value), i > 0 ? `−${formatNumber(s.drop)}` : "", formatPercent(s.share)]),
          [false, true, true, true],
        ),
    });
    const body = [host];
    const steps = flagshipSteps(last?.stats?.flagship, options.recentDays);
    if (steps.length) {
      body.push(
        el(
          "div",
          { class: "ov-mini", "aria-label": flagshipText(last.stats.flagship, options.recentDays) },
          el("span", { class: "ov-mini-head" }, el("span", { class: "dot flagship" }), "顶刊地学"),
          el(
            "ol",
            { class: "ov-steps" },
            ...steps.map((s, i) =>
              el(
                "li",
                { class: "ov-step" },
                `${s.label} `,
                el("strong", { text: formatNumber(s.value) }),
                s.note ? el("span", { class: "ov-step-note", text: `（${s.note}）` }) : null,
                // The arrow ends a step, so a wrapped line never starts with one
                i < steps.length - 1 ? el("span", { class: "ov-step-arrow", "aria-hidden": "true", text: "→" }) : null,
              ),
            ),
          ),
        ),
      );
    }
    const finished = last.finishedAt || last.runAt;
    return card("ov-funnel", "筛选漏斗", { caption: `个性推荐 · 上次运行 ${formatDateTime(finished)}`, actions: [toggle] }, ...body);
  }

  function sourcesCard(dash) {
    const hasRun = !!dash.lastRun;
    const rows = (dash.sources || []).map((source) => {
      const status = sourceStatus(source, hasRun);
      const detail = source.detail && source.detail !== status.label ? source.detail : null;
      const showCount = source.enabled && Number.isFinite(source.count);
      return el(
        "li",
        { class: `ov-source ${status.tone}`, title: [source.name, detail, source.error].filter(Boolean).join(" · ") },
        el("span", { class: "ov-source-main" }, el("span", { class: "ov-source-name", text: source.name }), detail ? el("span", { class: "ov-source-detail", text: detail }) : null),
        el("span", { class: "status ov-source-status" }, el("span", { class: `dot ${status.tone}` }), status.label),
        el("span", { class: "ov-source-count", text: showCount ? `${formatNumber(source.count)} 篇` : "" }),
        source.error ? el("span", { class: "ov-source-error", text: source.error }) : null,
      );
    });
    return card("ov-sources", "数据源", { caption: "状态与上次运行的篇数", actions: [linkButton("设置", "settings/sources")] }, el("ul", { class: "ov-list" }, ...rows));
  }

  function topicsCard(dash) {
    const topics = (dash.topics || []).filter((t) => t && t.label);
    const total = dash.clusters || dash.topicCount || topics.length;
    const llm = dash.llm || {};
    if (!topics.length) {
      let empty;
      if (!llm.enabled || !llm.configured) {
        empty = { icon: "sparkle", title: "开启大模型后会为每个研究方向生成名称", text: total ? `文库已聚类为 ${total} 个研究方向。` : null, action: { label: "去设置", onclick: () => navigate("settings/ai") } };
      } else if (!llm.topicLabels) {
        empty = { icon: "sparkle", title: "开启「研究方向标签」后会为每个研究方向生成名称", action: { label: "去设置", onclick: () => navigate("settings/ai") } };
      } else if (!dash.clusters) {
        empty = { icon: "chart", title: "运行一次后显示", text: "首次运行会对文库做语义聚类，并为每个方向起名。" };
      } else {
        empty = { icon: "sparkle", title: "下次运行时会为研究方向生成名称" };
      }
      return card("ov-topics", "研究方向", { caption: "基于文库的语义聚类" }, emptyState({ ...empty, compact: true }));
    }
    const library = count(dash.library?.items);
    const rows = topics.slice(0, 10).map((t) => ({ label: t.label, value: count(t.size), valueText: formatNumber(count(t.size)) }));
    const { host, toggle } = chartBlock("topics", {
      draw: drawBars(rows, {
        valueAt: "end",
        labelShare: 0.46,
        rowFactor: 2.1,
        ariaLabel: "文库中最大的研究方向",
        tipFor: (r) => [
          el("div", { class: "tip-head", text: r.label }),
          tipRow(null, formatNumber(r.value), "篇文献"),
          library ? el("div", { class: "tip-foot", text: `占文库 ${formatPercent(r.value / library)}` }) : null,
        ].filter(Boolean),
      }),
      minHeight: rows.length * 30,
      // The same numbers as the chart and its tooltips (share of the library when known)
      table: () =>
        library
          ? dataTable(["研究方向", "文献数", "占文库"], rows.map((r) => [r.label, r.valueText, formatPercent(r.value / library)]), [false, true, true])
          : dataTable(["研究方向", "文献数"], rows.map((r) => [r.label, r.valueText]), [false, true]),
    });
    return card("ov-topics", "研究方向", { caption: `基于文库的语义聚类（共 ${formatNumber(total)} 个方向）`, actions: [toggle] }, host);
  }

  function followedCard(dash) {
    const followed = dash.followed || { enabled: false, authors: [], invalid: [] };
    const edit = linkButton("编辑", "settings/followed");
    if (!followed.enabled) {
      return card(
        "ov-followed",
        "关注作者",
        {},
        emptyState({
          icon: "user",
          title: "关注特定作者的新论文",
          text: "按 ORCID 或 OpenAlex ID 追踪研究者，每次运行检查他们的新论文。",
          action: { label: "去设置", onclick: () => navigate("settings/followed") },
          compact: true,
        }),
      );
    }
    const authors = followed.authors || [];
    const invalid = followed.invalid || [];
    if (!authors.length && !invalid.length) {
      return card(
        "ov-followed",
        "关注作者",
        {},
        emptyState({ icon: "user", title: "还没有填写作者", text: "每行一位：姓名 + ORCID 或 OpenAlex ID。", action: { label: "去设置", onclick: () => navigate("settings/followed") }, compact: true }),
      );
    }
    const rows = authors.map((a) => {
      const id = a.orcid ? ["ORCID", a.orcid] : a.openalexId ? ["OpenAlex", a.openalexId] : null;
      return el(
        "li",
        { class: "ov-row ov-author" },
        el("span", { class: "dot followed" }),
        el(
          "div",
          { class: "ov-row-main" },
          el("div", { class: "ov-row-title", text: a.name }),
          id ? el("div", { class: "ov-row-detail ov-id" }, el("span", { class: "chip-key", text: `${id[0]} ` }), el("span", { class: "mono", text: id[1] })) : null,
        ),
        el("span", { class: a.pending > 0 ? "ov-row-count" : "ov-row-count zero", text: a.pending > 0 ? `${formatNumber(a.pending)} 篇待查看` : "暂无新论文" }),
      );
    });
    for (const line of invalid) {
      rows.push(
        el(
          "li",
          { class: "ov-row ov-invalid" },
          el("span", { class: "ov-row-icon warning" }, svgIcon("alert")),
          el("div", { class: "ov-row-main" }, el("div", { class: "ov-row-title", text: "无法识别" }), el("div", { class: "ov-row-detail", text: line })),
          el("span", { class: "ov-row-count" }),
        ),
      );
    }
    const caption = `${formatNumber(authors.length)} 位 · 检查最近 ${formatNumber(followed.daysBack)} 天`;
    return card("ov-followed", "关注作者", { caption, actions: [edit] }, el("ul", { class: "ov-list" }, ...rows));
  }

  function connectionsCard() {
    const c = state.connections;
    const close = el(
      "button",
      {
        type: "button",
        class: "btn ghost small icon-only",
        title: "关闭",
        "aria-label": "关闭连接测试结果",
        onclick: () => {
          state.connections = null;
          render();
        },
      },
      svgIcon("x"),
    );
    let body;
    let caption = null;
    if (c.running) {
      body = el("p", { class: "ov-waiting" }, el("span", { class: "dot running" }), "正在测试向量化、大模型、Crossref 和已填写的 API Key…");
    } else if (c.error) {
      body = el("p", { class: "ov-error-line", text: `测试失败：${c.error}` });
    } else {
      const ok = c.rows.filter((r) => r.ok === true).length;
      const failed = c.rows.filter((r) => r.ok === false).length;
      caption = `${formatDateTime(c.at)} · ${ok} 项正常${failed ? `，${failed} 项失败` : ""}`;
      body = el(
        "ul",
        { class: "ov-list" },
        ...c.rows.map((r) => {
          const status = connectionStatus(r);
          return el(
            "li",
            { class: `ov-row ov-conn ${status.tone}` },
            el("span", { class: `ov-row-icon ${status.tone}` }, svgIcon(status.icon)),
            el(
              "div",
              { class: "ov-row-main" },
              el("div", { class: "ov-row-title" }, r.name, r.target ? el("span", { class: "ov-target mono", text: r.target }) : null),
              el("div", { class: status.tone === "critical" ? "ov-row-detail ov-error" : "ov-row-detail", text: r.detail || "" }),
            ),
            el("span", { class: "ov-row-count ov-status-label", text: status.label }),
          );
        }),
      );
    }
    return card("ov-conn", "连接测试", { caption, actions: [close] }, body);
  }

  function firstRunCallout(dash) {
    const local = dash.embedding?.local;
    return el(
      "div",
      { class: "callout info ov-first" },
      svgIcon("sparkle"),
      el(
        "div",
        { class: "callout-body" },
        el("div", { class: "callout-title", text: "还没有运行过" }),
        el(
          "div",
          { class: "callout-text" },
          "点「立即检查」抓取第一批推荐。首次运行要为整个文库计算向量，可能需要几分钟。",
          local ? "当前使用本地模式向量化，配置 DashScope 等服务后推荐更准确。" : "",
        ),
      ),
      el(
        "div",
        { class: "callout-actions" },
        local ? el("button", { type: "button", class: "btn small", onclick: () => navigate("settings/embedding") }, "设置向量化") : null,
        el("button", { type: "button", class: "btn small primary", onclick: () => api()?.runNow().catch(() => {}) }, "立即检查"),
      ),
    );
  }

  // ------------------------------------------------------------------ render

  /** The loaded dashboard with the run state of the latest view. */
  const currentDash = () => ({ ...state.dash, running: !!(state.dash.running || state.view?.running) });

  function render() {
    const root = $("overview-root");
    clearTimeout(state.loadingTimer);
    state.drawers = [];
    state.live = null;
    $("overview-test").disabled = !!state.connections?.running;
    if (!state.dash) {
      if (state.error) {
        root.replaceChildren(
          emptyState({ icon: "alert", title: "无法加载概览", text: String(state.error.message || state.error), action: { label: "重试", onclick: () => load() } }),
        );
      }
      return;
    }
    const dash = currentDash();
    state.live = {};
    setSubtitle("overview", headerLine(dash));
    root.replaceChildren(
      ...[
        !dash.lastRun && !dash.running ? firstRunCallout(dash) : null,
        state.connections ? connectionsCard() : null,
        heroTile(dash),
        kpiGrid(dash),
        runsCard(dash),
        statusCard(dash),
        funnelCard(dash),
        sourcesCard(dash),
        topicsCard(dash),
        followedCard(dash),
      ].filter(Boolean),
    );
    drawCharts();
  }

  /**
   * Called every minute by the shell while the overview is shown: relative
   * times (上次检查, 下次, problem times) move on without a reload. Only the
   * KPI tiles and the status card are rebuilt; the status card is left alone
   * while it holds the focus.
   */
  function tick() {
    if (!state.dash || !state.live?.kpis) return;
    const dash = currentDash();
    state.live.kpis.replaceChildren(...kpiTileNodes(dash));
    const status = state.live.status;
    if (status && !status.contains?.(document.activeElement)) status.replaceChildren(...statusParts(dash));
  }

  /** Draw (or redraw) every chart for the current container width. */
  function drawCharts() {
    const root = $("overview-root");
    state.width = root.clientWidth || 0;
    for (const draw of state.drawers) {
      try {
        draw();
      } catch (e) {
        Zotero.logError(e);
      }
    }
  }

  /** Redraw the charts after a width change (one per frame; avoids ResizeObserver loops). */
  function scheduleRedraw() {
    if (state.frame || typeof window.requestAnimationFrame !== "function") return;
    state.frame = window.requestAnimationFrame(() => {
      state.frame = 0;
      const width = $("overview-root").clientWidth || 0;
      if (width && width !== state.width) drawCharts();
    });
  }

  // ------------------------------------------------------------------- data

  /** Reload the dashboard; calls made while loading are coalesced into one more. */
  function load() {
    if (state.loading) {
      state.queued = true;
      return state.loading;
    }
    const root = $("overview-root");
    root.setAttribute("aria-busy", "true");
    if (state.dash) {
      root.classList.add("refreshing");
    } else {
      // Only a slow first load shows a placeholder (no flash on fast ones)
      clearTimeout(state.loadingTimer);
      state.loadingTimer = setTimeout(() => {
        if (!state.dash && !state.error) root.replaceChildren(el("p", { class: "ov-loading", text: "正在加载概览…" }));
      }, 400);
    }
    state.loading = (async () => {
      try {
        do {
          state.queued = false;
          const view = state.view;
          const dash = await api().getDashboard();
          state.dash = dash;
          state.loadedView = view;
          state.error = null;
        } while (state.queued);
      } catch (e) {
        state.error = e;
        Zotero.logError(e);
        if (state.dash) toast(`概览刷新失败：${e.message || e}`, { tone: "error", key: "overview" });
      } finally {
        state.loading = null;
        root.classList.remove("refreshing");
        root.setAttribute("aria-busy", "false");
      }
      render();
    })();
    return state.loading;
  }

  async function testConnections() {
    if (state.connections?.running) return;
    state.connections = { running: true };
    render();
    $("pages").scrollTop = 0;
    try {
      const rows = await api().checkConnections();
      state.connections = { rows, at: Date.now() };
    } catch (e) {
      state.connections = { error: e.message || String(e), at: Date.now() };
    }
    render();
  }

  function onAddonEvent(event) {
    // A run started: the status card says so right away (results reload after it ends)
    if (event.type === "start" && state.dash) {
      state.dash.running = true;
      render();
    }
  }

  // ------------------------------------------------------------- page hooks

  function init() {
    $("overview-test").addEventListener("click", testConnections);
    if (typeof window.ResizeObserver === "function") {
      state.observer = new window.ResizeObserver(scheduleRedraw);
      state.observer.observe($("overview-root"));
    }
    api()?.addListener(onAddonEvent);
    window.addEventListener("unload", () => {
      api()?.removeListener(onAddonEvent);
      state.observer?.disconnect();
    });
  }

  function show() {
    scheduleRedraw();
  }

  /** Shell refresh: reload the dashboard unless the view is the one already shown (a page switch). */
  function refreshView(view) {
    state.view = view;
    if (state.dash && view && view === state.loadedView && !state.loading) {
      render();
      return null;
    }
    return load();
  }

  return {
    init,
    show,
    refresh: refreshView,
    tick,
    load,
    render,
    state,
    // Pure helpers (tests)
    textWidth,
    truncateText,
    formatTick,
    formatPercent,
    formatShortDateTime,
    niceScale,
    historySeries,
    chartTracks,
    labelStep,
    columnPath,
    barPath,
    columnLayout,
    barListLayout,
    funnelStages,
    funnelOptions,
    flagshipSteps,
    flagshipText,
    sourceStatus,
    runHealth,
    scheduleFacts,
    pendingParts,
    kpiTiles,
    headerLine,
    connectionStatus,
  };
})();

registerPage("overview", ZotWatchOverview);
