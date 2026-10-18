/* global Zotero, ChromeUtils, window, document, $, el, svgIcon, emptyState, navigate, setSubtitle, registerPage, confirmAction,
   api, toast, formatNumber, formatRelative, formatDateTime, createSearchBox, searchResultText */

// 调研 page: topic literature surveys (src/survey/, run by src/zotero/addon.js).
//   #survey        the new-survey form, the running survey (if any) and the history
//   #survey/<id>   one survey: its live progress while it runs, else the result
//                  (the review with clickable [n] citations and the references,
//                  the candidate table and the themes) with the toolbar
//                  保存到 Zotero / 导出 Markdown / 重新生成综述 / 继续 / 删除
// Survey events ({type: "survey"}) reach this page only (main.js ignores them).
// Everything written by the LLM or coming from the library and the APIs is
// inserted as text (el() uses textContent); links are built from DOIs and URLs.

var ZotWatchSurvey = (() => {
  // Fallbacks of api.getSurveyOptions() (src/survey/presets.js)
  const STAGES = [
    { key: "seeds", label: "种子" },
    { key: "queries", label: "检索词" },
    { key: "expand", label: "扩展" },
    { key: "dedupe", label: "去重" },
    { key: "rank", label: "排序" },
    { key: "enrich", label: "补全" },
    { key: "read", label: "精读" },
    { key: "themes", label: "主题" },
    { key: "write", label: "综述" },
  ];
  const DEPTHS = [
    { key: "quick", label: "快速", pool: 150, deepRead: 15 },
    { key: "standard", label: "标准", pool: 400, deepRead: 40 },
    { key: "deep", label: "深度", pool: 1000, deepRead: 80 },
  ];
  const DEPTH_LABELS = { quick: "快速", standard: "标准", deep: "深度" };
  const MAX_SEEDS = 20;
  const MAX_KEYWORDS = 12;
  const MIN_YEAR = 1800;
  // Reading-note fields (src/survey/prompts.js EXTRACTION_FIELDS)
  const NOTE_FIELDS = [
    ["question", "研究问题"],
    ["method", "研究方法"],
    ["data", "数据与样本"],
    ["findings", "主要发现"],
    ["contribution", "贡献"],
    ["limitations", "局限"],
  ];
  const AI_NOTICE = "本综述由 AI 根据所列文献的摘要、TLDR 和全文节选自动生成，可能存在遗漏或错误，引用前请核对原文。";
  const NO_LLM_TEXT = "未启用大模型，因此没有精读和撰写综述；下面按语义聚类列出排名靠前的候选文献，组名为自动提取的关键词。";
  const SOURCES = [
    { key: "openalex", name: "OpenAlex", help: "检索，种子的参考文献与施引文献" },
    { key: "s2", name: "Semantic Scholar", help: "检索、相关推荐和 TLDR" },
    { key: "crossref", name: "Crossref", help: "关键词检索，无需 Key" },
  ];
  const COLLECTION_PREFIX = "ZotWatch 调研 · ";
  const PAGE_SIZE = 100;
  const MAX_PROBLEMS = 4;
  // Two columns (report | candidates and themes) from this window width on
  const WIDE_QUERY = "(min-width: 1240px)";
  const FLASH_MS = 1800;
  // [3] or [1, 4] (the engine writes ", "; full-width separators are tolerated)
  const CITATION = /\[(\d+(?:\s*[,，、]\s*\d+)*)\]/g;
  const KEYWORD_SPLIT = /[\n,;，；、]+/;

  const state = {
    // Survey shown (#survey/<id>), null for the list
    id: null,
    visible: false,
    seq: 0,
    loading: false,
    loadError: null,
    options: null,
    list: null,
    survey: null,
    form: {
      topic: "",
      description: "",
      keywords: "",
      seeds: [],
      depth: "standard",
      yearFrom: "",
      sources: { openalex: true, s2: true, crossref: true },
    },
    starting: false,
    formNodes: null,
    home: null,
    // Result view: tab of the single column (null: not chosen, see activeTab), and the side pane of the two-column layout
    tab: null,
    side: "candidates",
    table: { filter: "all", sort: "score", dir: "desc", limit: PAGE_SIZE },
    expanded: new Set(),
    busy: new Map(),
    rowErrors: new Map(),
    action: null,
    save: { open: false, mode: "cited", running: false, progress: null, result: null, error: null },
    saveHost: null,
    saveButton: null,
    bodyHost: null,
    heading: null,
    refNodes: new Map(),
    live: null,
    wide: null,
    flash: null,
  };

  // ------------------------------------------------------------ helpers

  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const isRecord = (v) => !!v && typeof v === "object" && !Array.isArray(v);

  /** Text split at its [n] citation groups: [{text} | {refs: [n, …]}]. */
  function citationParts(text) {
    const s = String(text ?? "");
    const parts = [];
    let last = 0;
    for (const m of s.matchAll(CITATION)) {
      if (m.index > last) parts.push({ text: s.slice(last, m.index) });
      parts.push({ refs: m[1].split(/\s*[,，、]\s*/).map(Number) });
      last = m.index + m[0].length;
    }
    if (last < s.length) parts.push({ text: s.slice(last) });
    return parts;
  }

  /** Reference numbers a review cites (summary, sections and lists), unique, in order of first use. */
  function citedNumbers(report) {
    if (!report || report.kind !== "review") return [];
    const texts = [report.summary, ...(report.sections || []).flatMap((s) => s.paragraphs || []), ...(report.debates || []), ...(report.gaps || []), ...(report.future || [])];
    const out = [];
    for (const t of texts) for (const part of citationParts(t)) for (const n of part.refs || []) if (!out.includes(n)) out.push(n);
    return out;
  }

  /** Paragraphs of a multi-line text. */
  function paragraphsOf(text) {
    return String(text || "")
      .split(/\n\s*\n|\n/)
      .map((p) => p.trim())
      .filter(Boolean);
  }

  /** Keywords of the form (as src/survey/presets.js parseKeywords counts them). */
  function parseKeywords(text) {
    const seen = new Set();
    const out = [];
    for (const raw of String(text ?? "").split(KEYWORD_SPLIT)) {
      const k = raw.replace(/\s+/g, " ").trim();
      if (!k || seen.has(k.toLowerCase())) continue;
      seen.add(k.toLowerCase());
      out.push(k);
      if (out.length >= MAX_KEYWORDS) break;
    }
    return out;
  }

  /**
   * Problem of a survey input ({field, message}) or null: a topic, keywords
   * or seeds, a plausible start year and at least one source.
   */
  function validateInput(input, now = Date.now()) {
    if (!String(input.topic || "").trim()) return { field: "topic", message: "请填写调研主题" };
    if (!parseKeywords(input.keywords).length && !(input.seedKeys || []).length) {
      return { field: "keywords", message: "请至少填写一个关键词或选择一篇种子文献" };
    }
    const thisYear = new Date(now).getFullYear();
    if (input.yearFrom !== null && !(Number.isInteger(input.yearFrom) && input.yearFrom >= MIN_YEAR && input.yearFrom <= thisYear)) {
      return { field: "year", message: `起始年份应在 ${MIN_YEAR}–${thisYear} 之间，或者留空` };
    }
    const s = input.sources || {};
    if (!s.openalex && !s.s2 && !s.crossref) return { field: "sources", message: "请至少选择一个数据源" };
    return null;
  }

  /**
   * Provenance chips of a candidate (its `via` tags), in the order 种子 ·
   * 检索 · 引用 (the paper cites a seed) · 被引 (a seed cites it) · 推荐.
   */
  function provenance(via) {
    const tags = Array.isArray(via) ? via.filter((t) => typeof t === "string") : [];
    const seedsOf = (prefix) =>
      tags
        .map((t) => new RegExp(`^${prefix}:seed(\\d+)$`).exec(t))
        .filter(Boolean)
        .map((m) => Number(m[1]))
        .sort((a, b) => a - b);
    const out = [];
    if (tags.includes("seed")) out.push({ key: "seed", label: "种子", title: "种子文献本身" });
    if (tags.includes("search")) out.push({ key: "search", label: "检索", title: "检索词的检索结果" });
    const cites = seedsOf("cites");
    if (cites.length) out.push({ key: "cites", label: "引用", title: `引用了种子文献 ${cites.join("、")}` });
    const refs = seedsOf("ref");
    if (refs.length) out.push({ key: "ref", label: "被引", title: `被种子文献 ${refs.join("、")} 引用（在其参考文献中）` });
    if (tags.includes("rec")) out.push({ key: "rec", label: "推荐", title: "Semantic Scholar 根据种子文献推荐" });
    return out;
  }

  const SORT_KEYS = {
    score: (p) => (Number.isFinite(p.scores?.total) ? p.scores.total : null),
    year: (p) => (Number.isInteger(p.year) ? p.year : null),
    citations: (p) => (Number.isFinite(p.citations) ? p.citations : null),
    title: (p) => (p.title ? String(p.title).toLowerCase() : null),
  };

  /**
   * Candidates of the table: filter "all" | "new" (not in the library) |
   * "read"; sort by score | year | citations | title, `dir` asc | desc.
   * Missing values sort last; ties keep the rank order.
   */
  function candidateRows(pool, { filter = "all", sort = "score", dir = "desc" } = {}) {
    let rows = (Array.isArray(pool) ? pool : []).filter(isRecord);
    if (filter === "new") rows = rows.filter((p) => !p.inLibrary);
    else if (filter === "read") rows = rows.filter((p) => p.read);
    const key = SORT_KEYS[sort] || SORT_KEYS.score;
    const sign = dir === "asc" ? 1 : -1;
    const rank = (p, i) => (Number.isFinite(p.rank) ? p.rank : 1e9 + i);
    return rows
      .map((p, i) => ({ p, k: key(p), r: rank(p, i) }))
      .sort((a, b) => {
        if (a.k === null || b.k === null) return a.k === b.k ? a.r - b.r : a.k === null ? 1 : -1;
        if (a.k < b.k) return -sign;
        if (a.k > b.k) return sign;
        return a.r - b.r;
      })
      .map((x) => x.p);
  }

  /** Status of a survey (index entry or survey): {tone, label, title}. */
  function statusOf(entry) {
    if (!entry) return { tone: "off", label: "", title: null };
    if (entry.status === "running") return { tone: "running", label: progressLabel(entry.progress) || "进行中", title: null };
    if (entry.status === "done") return { tone: "good", label: "已完成", title: null };
    if (entry.status === "failed") return { tone: "critical", label: "失败", title: (entry.error?.message ?? entry.error) || null };
    if (entry.interrupted) return { tone: "warning", label: "已中断", title: "Zotero 关闭时中断，可以继续" };
    return { tone: "off", label: "已取消", title: "可以继续" };
  }

  /** "精读 17/40" / "扩展" for a progress event; "" without one. */
  function progressLabel(p) {
    if (!p || !p.label) return "";
    return p.total ? `${p.label} ${formatNumber(p.current ?? 0)}/${formatNumber(p.total)}` : p.label;
  }

  /**
   * State of each stepper stage of a running survey: "done" | "current" |
   * "todo" (the current stage from the progress, else the survey's stage).
   */
  function stepStates(stages, progress, survey = null) {
    let index = Number.isInteger(progress?.index) ? progress.index : -1;
    if (index < 0 && survey?.stage) index = stages.findIndex((s) => s.key === survey.stage);
    if (index < 0) index = 0;
    return stages.map((s, i) => (i === index ? "current" : i < index ? "done" : "todo"));
  }

  /** Label of the last finished stage ("" when none). */
  function lastStageLabel(survey, stages = STAGES) {
    const done = new Set(survey?.completed || []);
    const last = stages.filter((s) => done.has(s.key)).at(-1);
    return last ? last.label : "";
  }

  function stageLabel(key) {
    return (state.options?.stages || STAGES).find((s) => s.key === key)?.label || key || "";
  }

  /** "Wang L, Chen M 等" style author line (names as given). */
  function authorsText(authors, count = 0) {
    const names = (Array.isArray(authors) ? authors : []).map((a) => (typeof a === "string" ? a : a?.name)).filter(Boolean);
    if (!names.length) return "";
    const total = Math.max(count || 0, names.length);
    return total > 3 ? `${names.slice(0, 3).join(", ")} 等` : names.join(", ");
  }

  /** "Li Wang, Mei Chen 等 · Economic Geology · 2020" of a reference. */
  function referenceMeta(ref) {
    return [authorsText(ref.authors, ref.authorCount), ref.venue, ref.year].filter(Boolean).join(" · ");
  }

  /** Link of a paper or reference: its DOI, else an http(s) URL, else null. */
  function linkOf(p) {
    const doi = String(p?.doi || "")
      .trim()
      .replace(/^https?:\/\/(dx\.)?doi\.org\//i, "")
      .replace(/^doi:\s*/i, "");
    if (/^10\.\d{4,9}\/\S+$/.test(doi)) return `https://doi.org/${doi}`;
    return /^https?:\/\//i.test(p?.url || "") ? p.url : null;
  }

  /** Score 0–1 as 0–100 ("–" when missing). */
  function scoreText(p) {
    return Number.isFinite(p.scores?.total) ? String(Math.round(p.scores.total * 100)) : "–";
  }

  /** Tooltip with the parts of a candidate's score. */
  function scoreTitle(p) {
    const s = p.scores || {};
    const part = (label, v) => (Number.isFinite(v) ? `${label} ${v.toFixed(2)}` : null);
    return [
      "综合得分 = 0.55 相似度 + 0.20 关联 + 0.15 被引 + 0.10 新近",
      [part("相似度", s.sim), part("关联", s.link), part("被引", s.cites), part("新近", s.recency)].filter(Boolean).join(" · "),
    ]
      .filter(Boolean)
      .join("\n");
  }

  /** Collection name of a survey (src/zotero/survey.js surveyCollectionName). */
  function collectionName(s) {
    if (s.saved?.collectionName) return s.saved.collectionName;
    const topic = Array.from(String(s.input?.topic || "").replace(/\s+/g, " ").trim()).slice(0, 60).join("");
    return `${COLLECTION_PREFIX}${topic || s.id}`;
  }

  /** "已保存到「…」：…" summary of saveSurveyToZotero's result. */
  function saveResultText(r) {
    if (!r) return "";
    const parts = [];
    if (r.linked) parts.push(`${r.linked} 篇已在文库的文献加入分类`);
    if (r.imported) parts.push(`导入 ${r.imported} 篇`);
    if (r.failed) parts.push(`${r.failed} 篇导入失败`);
    if (r.skipped) parts.push(`${r.skipped} 篇在其他文库中，未加入`);
    const tail = r.cancelled ? "；已停止导入，其余文献未加入" : "";
    return `已保存到「${r.collectionName}」：笔记已写入${parts.length ? `，${parts.join("，")}` : ""}${tail}。`;
  }

  /** Lines of the 调研过程 panel from the stage summaries: [[label, text]]. */
  function processLines(s) {
    const st = isRecord(s?.stages) ? s.stages : {};
    const n = (v) => (Number.isFinite(v) ? formatNumber(v) : null);
    const lines = [];
    const seeds = st.seeds;
    if (seeds) {
      const found = [seeds.openalex ? `OpenAlex 找到 ${n(seeds.openalex)} 篇` : null, seeds.s2 ? `Semantic Scholar 找到 ${n(seeds.s2)} 篇` : null, seeds.fullText ? `${n(seeds.fullText)} 篇有全文` : null];
      lines.push(["种子", [`${n(seeds.seeds) ?? 0} 篇`, ...found].filter(Boolean).join("，")]);
    }
    if (Array.isArray(s?.queries) && s.queries.length) lines.push(["检索词", s.queries.join("；")]);
    const ex = st.expand;
    if (ex) {
      const by = isRecord(ex.bySource) ? Object.entries(ex.bySource).filter(([, v]) => Number.isFinite(v) && v > 0) : [];
      const bits = [
        Number.isFinite(ex.found) ? `找到 ${n(ex.found)} 篇不重复的文献` : null,
        by.length ? `（各来源返回 ${by.map(([k, v]) => `${k} ${n(v)} 条`).join("、")}）` : null,
        Number.isFinite(ex.requests) ? `，请求 ${n(ex.requests)} 次${ex.failed ? `（${n(ex.failed)} 次失败）` : ""}` : null,
        ex.beforeYear ? `，${n(ex.beforeYear)} 篇早于起始年份被排除` : null,
        Number.isFinite(ex.pool) ? `，候选池保留 ${n(ex.pool)} 篇` : null,
      ];
      lines.push(["扩展", bits.filter(Boolean).join("")]);
    }
    const dd = st.dedupe;
    if (dd) lines.push(["去重", [`保留 ${n(dd.after) ?? 0} 篇`, dd.merged ? `合并 ${n(dd.merged)} 篇重复` : null, dd.inLibrary ? `${n(dd.inLibrary)} 篇已在文库` : null].filter(Boolean).join("，")]);
    if (st.rank) lines.push(["排序", `按相似度、与种子的关联、被引次数和新近程度为 ${n(st.rank.ranked) ?? 0} 篇打分`]);
    if (st.enrich) {
      const e = st.enrich;
      lines.push(["补全", [Number.isFinite(e.missing) ? `补全摘要 ${n(e.filled ?? 0)}/${n(e.missing)} 篇` : null, e.tldr ? `TLDR ${n(e.tldr)} 篇` : null].filter(Boolean).join("，") || "无需补全"]);
    }
    const rd = st.read;
    if (rd) {
      lines.push(["精读", rd.skipped ? "未启用大模型，跳过" : [`精读 ${n((rd.read ?? 0) + (rd.cached ?? 0))} 篇`, rd.cached ? `其中 ${n(rd.cached)} 篇来自缓存` : null, rd.failed ? `${n(rd.failed)} 篇失败` : null].filter(Boolean).join("，")]);
    }
    if (st.themes) lines.push(["主题", `${n(st.themes.themes) ?? 0} 个${st.themes.labelled === "llm" ? "，由大模型命名" : "，以关键词命名"}`]);
    const wr = st.write;
    if (wr?.kind === "review") {
      lines.push([
        "综述",
        [`${n(wr.sections) ?? 0} 节`, wr.failed ? `${n(wr.failed)} 节生成失败` : null, wr.unverified ? `${n(wr.unverified)} 节没有有效引用` : null].filter(Boolean).join("，"),
      ]);
    } else if (wr?.kind === "list") {
      lines.push(["综述", `未启用大模型，按 ${n(wr.groups) ?? 0} 组列出候选文献`]);
    }
    return lines;
  }

  // -------------------------------------------------------------- data

  const options = () => state.options || {};
  const stagesOf = () => (Array.isArray(options().stages) && options().stages.length ? options().stages : STAGES);
  const depthsOf = () => (Array.isArray(options().depths) && options().depths.length ? options().depths : DEPTHS);
  /** The running survey ({id, progress}) or null. */
  const runningTask = () => options().running || null;
  /** Another survey than `id` (or any, without one) runs. */
  const otherRunning = (id = null) => !!runningTask() && runningTask().id !== id;

  function loadOptions() {
    try {
      state.options = api()?.getSurveyOptions?.() || state.options;
    } catch (e) {
      Zotero.logError(e);
    }
    setNavDot(!!runningTask());
  }

  function setNavDot(on) {
    const dot = $("nav-survey-dot");
    if (dot) dot.hidden = !on;
  }

  function isWide() {
    try {
      return !!state.wide?.matches;
    } catch {
      return false;
    }
  }

  /** Load what the current route shows (the list or one survey); stale answers are dropped. */
  async function load() {
    const seq = ++state.seq;
    const root = $("survey-root");
    state.loading = true;
    root.setAttribute("aria-busy", "true");
    // A survey being reloaded fades; the list keeps its form fully visible
    if (state.id) root.classList.add("is-loading");
    try {
      if (state.id) {
        const survey = await api().getSurvey(state.id);
        if (seq !== state.seq) return;
        state.survey = survey || null;
      } else {
        const list = await api().listSurveys();
        if (seq !== state.seq) return;
        state.list = Array.isArray(list) ? list : [];
      }
      state.loadError = null;
    } catch (e) {
      if (seq !== state.seq) return;
      state.loadError = e?.message || String(e);
      Zotero.logError(e);
    } finally {
      if (seq === state.seq) {
        state.loading = false;
        root.setAttribute("aria-busy", "false");
        root.classList.remove("is-loading");
      }
    }
    render();
  }

  // -------------------------------------------------------------- shell

  function render() {
    if (!api()) return;
    $("survey-new").hidden = !state.id;
    if (state.id) renderSurvey();
    else renderHome();
    setSubtitle("survey", subtitle());
  }

  function subtitle() {
    if (state.id) {
      const s = state.survey;
      return s ? `主题：${s.input?.topic || ""}` : "";
    }
    const count = state.list?.length || 0;
    return count ? `检索、精读并撰写文献综述 · 共 ${count} 项调研` : "围绕一个主题检索、精读并撰写文献综述";
  }

  /** Remember the focused control (by data-key) across a re-render. */
  function withFocus(fn) {
    const active = document.activeElement;
    const key = active?.getAttribute?.("data-key");
    fn();
    if (!key) return;
    const root = $("survey-root");
    const next = root.querySelector?.(`[data-key="${String(key).replace(/["\\]/g, "\\$&")}"]`);
    if (next && !next.disabled) next.focus?.({ preventScroll: true });
  }

  // --------------------------------------------------------------- form

  /**
   * Local segmented control: {node, set(value), buttons}. `name` keys the
   * buttons (data-key) so that focus stays on them across re-renders.
   */
  function seg(name, items, value, onSelect, attrs = {}) {
    const buttons = items.map(([key, label, extra]) =>
      el("button", { type: "button", "data-value": key, "data-key": `${name}:${key}`, "aria-pressed": "false" }, label, extra ?? null),
    );
    const node = el("div", { class: "seg", role: "group", ...attrs }, buttons);
    const set = (v) => {
      for (const b of buttons) {
        const on = b.getAttribute("data-value") === v;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      }
    };
    for (const b of buttons) {
      b.addEventListener("click", () => {
        const v = b.getAttribute("data-value");
        set(v);
        onSelect(v);
      });
    }
    set(value);
    return { node, set, buttons };
  }

  const countBadge = (n) => el("span", { class: "tab-count", text: formatNumber(n) });

  function formRow(label, control, { id = null, help = null, optional = false, extra = null, className = "" } = {}) {
    return el(
      "div",
      { class: `sv-row ${className}`.trim() },
      el("div", { class: "sv-row-head" }, el(id ? "label" : "span", { class: "sv-label", for: id }, label), optional ? el("span", { class: "sv-optional", text: "可选" }) : null, extra),
      control,
      help ? el("p", { class: "sv-help", text: help }) : null,
    );
  }

  function creatorsText(creators) {
    const names = (Array.isArray(creators) ? creators : []).filter((c) => typeof c === "string" && c);
    if (!names.length) return "";
    return names.length > 2 ? `${names[0]} 等` : names.join(", ");
  }

  function buildForm() {
    const f = state.form;
    const n = {};
    const thisYear = new Date().getFullYear();

    n.title = el("h2", { class: "card-title", id: "sv-form-title", tabindex: "-1", text: "新的调研" });
    n.topic = el("input", { type: "text", class: "input", id: "sv-topic", maxlength: "200", autocomplete: "off", placeholder: "例如：斑岩铜矿成矿流体中的铜同位素示踪" });
    n.topic.addEventListener("input", () => {
      f.topic = n.topic.value;
      clearFormError();
    });
    n.description = el("textarea", {
      class: "textarea",
      id: "sv-description",
      rows: "2",
      maxlength: "2000",
      placeholder: "想弄清的问题、关注的方法或研究区域；大模型据此生成检索词并组织综述",
    });
    n.description.addEventListener("input", () => (f.description = n.description.value));
    n.keywords = el("textarea", { class: "textarea", id: "sv-keywords", rows: "2", placeholder: "copper isotopes, porphyry copper deposits, ore-forming fluids" });
    n.keywordCount = el("span", { class: "sv-count" });
    n.keywords.addEventListener("input", () => {
      f.keywords = n.keywords.value;
      clearFormError();
      updateKeywordCount();
      updateEstimate();
    });

    n.seedSearch = createSearchBox({
      id: "sv-seed-search",
      label: "在文库中搜索种子文献",
      placeholder: "按标题、作者、年份或 DOI 搜索文库…",
      search: async (query) => (await api().searchLibrary(query, { limit: 20 })) || [],
      render: (item) => searchResultText(item.title || "（无标题）", [creatorsText(item.creators), item.year ? String(item.year) : "", item.doi ? `doi:${item.doi}` : ""]),
      action: (item) => {
        if (state.form.seeds.some((s) => s.key === item.key)) return { label: "已添加", disabled: true };
        if (state.form.seeds.length >= MAX_SEEDS) return { label: "添加", disabled: true, title: `最多 ${MAX_SEEDS} 篇种子文献` };
        return { label: "添加" };
      },
      pick: async (item) => (addSeeds([item]) ? `已添加：${item.title}` : ""),
      nameOf: (item) => item.title || "",
      emptyText: "文库中没有匹配的条目",
    });
    n.useSelected = el("button", { type: "button", class: "btn small", id: "sv-use-selected" }, svgIcon("plus"), "使用 Zotero 中选中的条目");
    n.useSelected.addEventListener("click", () => useSelected());
    n.seedStatus = el("p", { class: "sv-help sv-seed-status", role: "status", hidden: true });
    n.seedCount = el("span", { class: "sv-count" });
    n.seeds = el("ul", { class: "sv-seeds", "aria-label": "已选的种子文献" });

    n.depth = seg(
      "depth",
      depthsOf().map((d) => [d.key, d.label]),
      f.depth,
      (value) => {
        f.depth = value;
        updateEstimate();
      },
      { id: "sv-depth", "aria-label": "调研深度" },
    );
    n.estimate = el("p", { class: "sv-help sv-estimate", "aria-live": "polite" });
    n.year = el("input", { type: "number", class: "input sv-year", id: "sv-year", min: String(MIN_YEAR), max: String(thisYear), step: "1", placeholder: "不限" });
    n.year.addEventListener("input", () => {
      f.yearFrom = n.year.value;
      clearFormError();
    });

    n.sources = SOURCES.map((s) => {
      const input = el("input", { type: "checkbox", role: "switch", class: "switch", id: `sv-source-${s.key}` });
      input.addEventListener("change", () => {
        f.sources[s.key] = input.checked;
        clearFormError();
        updateEstimate();
      });
      const note = el("span", { class: "sv-source-note", hidden: true });
      const node = el(
        "div",
        { class: "sv-source" },
        input,
        el("div", { class: "sv-source-body" }, el("label", { class: "sv-source-name", for: `sv-source-${s.key}`, text: s.name }), el("span", { class: "sv-source-help", text: s.help }), note),
      );
      return { key: s.key, input, note, node };
    });

    n.llmNote = el(
      "div",
      { class: "callout info sv-llm-note", hidden: true },
      svgIcon("info"),
      el(
        "div",
        { class: "callout-body" },
        el("div", { class: "callout-title", text: "未启用大模型" }),
        el("div", { class: "callout-text", text: "只检索、去重和排序候选文献，并按语义聚类分组；不会精读，也不会撰写综述。" }),
      ),
      el("div", { class: "callout-actions" }, el("a", { class: "btn small", href: "#settings/ai" }, "去设置")),
    );

    n.error = el("p", { class: "sv-form-error", role: "alert", hidden: true });
    n.start = el("button", { type: "button", class: "btn primary", id: "sv-start" }, svgIcon("search"), "开始调研");
    n.start.addEventListener("click", () => start());
    n.startNote = el("span", { class: "sv-help sv-start-note" });

    const seedBlock = el(
      "div",
      { class: "sv-seed-block" },
      n.seedSearch.node,
      el("div", { class: "sv-seed-tools" }, n.useSelected, n.seedStatus),
      n.seeds,
    );
    const depthBlock = el("div", { class: "sv-depth-block" }, n.depth.node, n.estimate);
    n.card = el(
      "section",
      { class: "card sv-form", "aria-labelledby": "sv-form-title" },
      el(
        "div",
        { class: "card-header" },
        el("div", {}, n.title, el("p", { class: "card-caption", text: "从关键词和种子文献出发，沿检索、参考文献、施引文献和推荐找到相关文献，再由 AI 精读并撰写综述。" })),
      ),
      el(
        "div",
        { class: "card-body sv-form-body" },
        formRow("主题", n.topic, { id: "sv-topic" }),
        formRow("描述", n.description, { id: "sv-description", optional: true }),
        formRow("关键词", n.keywords, {
          id: "sv-keywords",
          help: "逗号、分号或换行分隔，最多 12 个；英文关键词在 OpenAlex 等数据库中检索效果更好。",
          extra: n.keywordCount,
        }),
        formRow("种子文献", seedBlock, {
          id: "sv-seed-search",
          optional: true,
          help: "文库中与主题最相关的几篇论文；插件沿它们的参考文献、施引文献和推荐扩展。至少需要关键词或一篇种子文献。",
          extra: n.seedCount,
        }),
        el(
          "div",
          { class: "sv-row-pair" },
          formRow("深度", depthBlock, { className: "sv-row-depth" }),
          formRow("起始年份", el("span", { class: "input-unit" }, n.year, "年以来"), { id: "sv-year", optional: true, className: "sv-row-year" }),
        ),
        formRow("数据源", el("div", { class: "sv-sources" }, n.sources.map((s) => s.node))),
        n.llmNote,
        n.error,
        el("div", { class: "sv-submit" }, n.start, n.startNote),
      ),
    );
    state.formNodes = n;
    syncFormInputs();
  }

  /** Put state.form into the form's controls (after a draft from the item context menu). */
  function syncFormInputs() {
    const n = state.formNodes;
    if (!n) return;
    const f = state.form;
    n.topic.value = f.topic;
    n.description.value = f.description;
    n.keywords.value = f.keywords;
    n.year.value = f.yearFrom;
    n.depth.set(f.depth);
    renderSeeds();
    updateKeywordCount();
    updateFormOptions();
  }

  function updateKeywordCount() {
    const n = state.formNodes;
    if (!n) return;
    const count = parseKeywords(state.form.keywords).length;
    n.keywordCount.textContent = count ? `${count} 个` : "";
  }

  /** The sources the survey will use: Semantic Scholar only with its key. */
  function effectiveSources() {
    const s = { ...state.form.sources };
    if (!options().semanticScholarKey) s.s2 = false;
    return s;
  }

  function updateEstimate() {
    const n = state.formNodes;
    if (!n) return;
    const f = state.form;
    const depth = depthsOf().find((d) => d.key === f.depth) || DEPTHS[1];
    let estimate = null;
    try {
      estimate = api()?.surveyEstimate?.(f.depth, { seeds: f.seeds.length, sources: effectiveSources(), keywords: parseKeywords(f.keywords).length }) || null;
    } catch (e) {
      Zotero.logError(e);
    }
    const parts = [`候选池最多 ${formatNumber(depth.pool)} 篇`, options().llm ? `精读 ${formatNumber(depth.deepRead)} 篇` : null, estimate?.text || null];
    n.estimate.textContent = parts.filter(Boolean).join(" · ");
  }

  /** Key notices, the LLM callout and the start button from the survey options. */
  function updateFormOptions() {
    const n = state.formNodes;
    if (!n) return;
    const o = options();
    const f = state.form;
    for (const s of n.sources) {
      const keyless = (s.key === "s2" && !o.semanticScholarKey) || (s.key === "openalex" && !o.openalexKey);
      s.input.disabled = s.key === "s2" && !o.semanticScholarKey;
      s.input.checked = !!f.sources[s.key] && !s.input.disabled;
      s.node.classList.toggle("is-disabled", s.input.disabled);
      s.note.hidden = !keyless;
      if (keyless) {
        s.note.replaceChildren(
          s.key === "s2" ? "需要 API Key，" : "未配置 API Key，每天只有约 100 次免费额度，",
          el("a", { href: "#settings/apis" }, "去设置"),
        );
      }
    }
    n.llmNote.hidden = !!o.llm;
    const running = runningTask();
    n.start.disabled = !!running || state.starting;
    n.startNote.textContent = running
      ? "已有一项调研正在进行；它完成或取消后才能开始新的调研。"
      : "调研在后台进行：可以关闭此窗口或继续使用 Zotero，完成后会通知你。";
    updateEstimate();
  }

  function renderSeeds() {
    const n = state.formNodes;
    if (!n) return;
    const seeds = state.form.seeds;
    n.seeds.replaceChildren(
      ...seeds.map((s) =>
        el(
          "li",
          { class: "sv-seed", title: [s.title, creatorsText(s.creators), s.year].filter(Boolean).join(" · ") },
          el("span", { class: "sv-seed-title", text: s.title || "（无标题）" }),
          s.year ? el("span", { class: "sv-seed-year", text: String(s.year) }) : null,
          el(
            "button",
            { type: "button", class: "sv-seed-remove", title: "移除", "aria-label": `移除种子文献：${s.title || s.key}`, onclick: () => removeSeed(s.key) },
            svgIcon("x"),
          ),
        ),
      ),
    );
    n.seeds.hidden = !seeds.length;
    n.seedCount.textContent = seeds.length ? `${seeds.length} / ${MAX_SEEDS}` : "";
    n.useSelected.disabled = seeds.length >= MAX_SEEDS;
    n.seedSearch.refresh();
  }

  function setSeedStatus(text) {
    const n = state.formNodes;
    n.seedStatus.textContent = text || "";
    n.seedStatus.hidden = !text;
  }

  /** Add library items as seeds (no duplicates, at most MAX_SEEDS); returns how many were added. */
  function addSeeds(items, { suggestTopic = false } = {}) {
    const f = state.form;
    let added = 0;
    for (const item of items || []) {
      if (!item || typeof item.key !== "string" || f.seeds.some((s) => s.key === item.key)) continue;
      if (f.seeds.length >= MAX_SEEDS) break;
      f.seeds.push({ key: item.key, title: item.title || "", year: item.year || null, creators: Array.isArray(item.creators) ? item.creators : [], doi: item.doi || null });
      added++;
    }
    // The first seed's title stands in for a topic not typed yet
    if (suggestTopic && added && !f.topic.trim() && f.seeds[0]?.title) f.topic = f.seeds[0].title;
    if (added) clearFormError();
    syncFormInputs();
    return added;
  }

  function removeSeed(key) {
    state.form.seeds = state.form.seeds.filter((s) => s.key !== key);
    renderSeeds();
    updateEstimate();
    state.formNodes.seedSearch.input.focus?.();
  }

  async function useSelected() {
    let items = [];
    try {
      items = (await api().getSelectedItems()) || [];
    } catch (e) {
      setSeedStatus(`出错：${e?.message || e}`);
      return;
    }
    if (!items.length) {
      setSeedStatus("Zotero 主窗口中没有选中的文献条目（笔记和附件之外的条目）。");
      return;
    }
    const before = state.form.seeds.length;
    const added = addSeeds(items, { suggestTopic: true });
    if (added) setSeedStatus(`已添加 ${added} 篇种子文献${before + items.length > MAX_SEEDS && added < items.length ? `（最多 ${MAX_SEEDS} 篇）` : ""}`);
    else setSeedStatus(state.form.seeds.length >= MAX_SEEDS ? `种子文献已达上限 ${MAX_SEEDS} 篇` : "选中的条目都已在种子文献中");
  }

  /** The draft of the item context menu (api.openSurvey): its seeds and topic replace the form's. */
  function takeDraft() {
    let draft = null;
    try {
      draft = api()?.takeSurveyDraft?.() || null;
    } catch (e) {
      Zotero.logError(e);
    }
    if (!draft) return false;
    if (!state.formNodes) buildForm();
    state.form.seeds = [];
    const added = addSeeds(Array.isArray(draft.seeds) ? draft.seeds : []);
    if (typeof draft.topic === "string" && draft.topic.trim()) state.form.topic = draft.topic.trim();
    syncFormInputs();
    setSeedStatus(added ? `已带入 Zotero 中选中的 ${added} 篇种子文献` : "选中的条目不在 ZotWatch 使用的文库中");
    return true;
  }

  function formInput() {
    const f = state.form;
    const year = String(f.yearFrom ?? "").trim();
    return {
      topic: f.topic.trim(),
      description: f.description.trim(),
      keywords: f.keywords,
      seedKeys: f.seeds.map((s) => s.key),
      depth: f.depth,
      yearFrom: year ? Number(year) : null,
      sources: effectiveSources(),
    };
  }

  /**
   * Empty the form for the next survey once one has started (its input is
   * kept with the survey); the depth and the sources stay as chosen.
   */
  function resetForm() {
    Object.assign(state.form, { topic: "", description: "", keywords: "", seeds: [], yearFrom: "" });
    if (!state.formNodes) return;
    setSeedStatus("");
    state.formNodes.seedSearch.clear();
    syncFormInputs();
  }

  function clearFormError() {
    const n = state.formNodes;
    if (!n || n.error.hidden) return;
    n.error.hidden = true;
    n.error.textContent = "";
    for (const node of [n.topic, n.keywords, n.year]) node.removeAttribute("aria-invalid");
  }

  function showFormError(problem) {
    const n = state.formNodes;
    n.error.textContent = problem.message;
    n.error.hidden = false;
    const field = { topic: n.topic, keywords: n.keywords, year: n.year, sources: n.sources.find((s) => !s.input.disabled)?.input }[problem.field];
    if (field && problem.field !== "sources") field.setAttribute("aria-invalid", "true");
    field?.focus?.();
  }

  async function start() {
    const input = formInput();
    const problem = validateInput(input);
    if (problem) {
      showFormError(problem);
      return;
    }
    state.starting = true;
    updateFormOptions();
    try {
      const { id } = await api().startSurvey(input);
      clearFormError();
      resetForm();
      state.options = { ...options(), running: { id, progress: null } };
      setNavDot(true);
      navigate(`survey/${id}`);
    } catch (e) {
      showFormError({ field: null, message: e?.message || String(e) });
    } finally {
      state.starting = false;
      updateFormOptions();
    }
  }

  // --------------------------------------------------------------- home

  function renderHome() {
    const root = $("survey-root");
    if (!state.formNodes) buildForm();
    if (!state.home) {
      state.home = { runningHost: el("div", { class: "sv-running-host" }), historyHost: el("div", { class: "sv-history-host" }) };
      state.home.node = el("div", { class: "sv-home" }, state.home.runningHost, state.formNodes.card, state.home.historyHost);
    }
    // The form stays mounted (typing, focus and search results survive reloads)
    if (state.home.node.parentNode !== root) root.replaceChildren(state.home.node);
    state.heading = state.formNodes.title;
    updateFormOptions();
    renderRunningCard();
    renderHistory();
  }

  function runningEntry() {
    const task = runningTask();
    if (!task) return null;
    const entry = (state.list || []).find((e) => e.id === task.id);
    return { id: task.id, topic: entry?.topic || "", progress: task.progress || entry?.progress || null, createdAt: entry?.createdAt || null };
  }

  function renderRunningCard() {
    const host = state.home.runningHost;
    const entry = runningEntry();
    state.live = null;
    if (!entry) {
      host.replaceChildren();
      return;
    }
    const stage = el("span", { class: "sv-running-stage" });
    const bar = el("div", { class: "progress-fill indeterminate" });
    const cancel = el("button", { type: "button", class: "btn ghost small", "data-key": "cancel-running", onclick: () => cancelSurvey(entry.id, cancel) }, "取消");
    host.replaceChildren(
      el(
        "section",
        { class: "card sv-running-card", "aria-label": "正在进行的调研" },
        el(
          "div",
          { class: "sv-running-row" },
          el("span", { class: "status" }, el("span", { class: "dot running" }), "正在调研"),
          el("a", { class: "sv-running-topic", href: `#survey/${entry.id}`, text: entry.topic || "（未命名）" }),
          stage,
          el("a", { class: "btn small", href: `#survey/${entry.id}` }, "查看进度", svgIcon("arrow-right")),
          cancel,
        ),
        el("div", { class: "progress-track" }, bar),
      ),
    );
    state.live = { kind: "card", id: entry.id, stage, bar };
    updateLive(entry.progress);
  }

  function historyMeta(e) {
    const c = e.counts || {};
    return [
      formatRelative(e.createdAt),
      DEPTH_LABELS[e.depth] ? `深度：${DEPTH_LABELS[e.depth]}` : null,
      c.seeds ? `种子 ${formatNumber(c.seeds)}` : null,
      `候选 ${formatNumber(c.candidates ?? 0)}`,
      c.read ? `精读 ${formatNumber(c.read)}` : null,
      e.llm === false ? "未使用大模型" : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }

  function renderHistory() {
    const host = state.home.historyHost;
    const runningId = runningTask()?.id;
    const entries = (state.list || []).filter((e) => e.id !== runningId);
    let body;
    if (!state.list && state.loadError) {
      body = el(
        "div",
        { class: "callout error" },
        svgIcon("alert"),
        el("div", { class: "callout-body" }, el("div", { class: "callout-title", text: "无法读取调研记录" }), el("div", { class: "callout-text", text: state.loadError })),
        el("div", { class: "callout-actions" }, el("button", { type: "button", class: "btn small", onclick: () => load() }, "重试")),
      );
    } else if (!state.list) {
      body = el("p", { class: "sv-loading", text: "正在加载…" });
    } else if (!entries.length) {
      body = emptyState({
        icon: "survey",
        title: state.list.length ? "没有其他调研记录" : "还没有调研记录",
        text: "每项调研的过程和结果都会保存在这里（最近 30 项），可以随时回来查看、继续或保存到 Zotero。",
        compact: true,
      });
    } else {
      body = el(
        "ul",
        { class: "sv-history" },
        entries.map((e) => {
          const status = statusOf(e);
          return el(
            "li",
            {},
            el(
              "a",
              { class: "sv-hist", href: `#survey/${e.id}`, title: status.title || null },
              el("span", { class: `dot ${status.tone}` }),
              el("span", { class: "sv-hist-main" }, el("span", { class: "sv-hist-topic", text: e.topic || "（未命名）" }), el("span", { class: "sv-hist-meta", text: historyMeta(e) })),
              el("span", { class: `sv-hist-status ${status.tone}`, text: status.label }),
              svgIcon("chevron-right", "icon sv-hist-chevron"),
            ),
          );
        }),
      );
    }
    host.replaceChildren(
      el(
        "section",
        { class: "card sv-history-card", "aria-labelledby": "sv-history-title" },
        el(
          "div",
          { class: "card-header" },
          el("h2", { class: "card-title", id: "sv-history-title", text: "调研记录" }),
          entries.length ? el("span", { class: "card-caption", text: `${formatNumber(entries.length)} 项` }) : null,
        ),
        el("div", { class: "card-body" }, body),
      ),
    );
  }

  // ------------------------------------------------------- running view

  /** Update the live progress (stepper, bar, counters, or the list's card) in place. */
  function updateLive(p) {
    const live = state.live;
    if (!live) return;
    const pct = p?.total ? clamp(Math.round((100 * (p.current || 0)) / p.total), 0, 100) : null;
    if (pct === null) {
      live.bar.classList.add("indeterminate");
      live.bar.style.width = "";
    } else {
      live.bar.classList.remove("indeterminate");
      live.bar.style.width = `${pct}%`;
    }
    if (live.kind === "card") {
      live.stage.textContent = progressLabel(p) || "准备中…";
      return;
    }
    const states = stepStates(stagesOf(), p, state.survey);
    live.steps.forEach((step, i) => {
      const st = states[i];
      step.li.className = `sv-step ${st}`;
      if (st === "current") step.li.setAttribute("aria-current", "step");
      else step.li.removeAttribute("aria-current");
      step.dot.replaceChildren(st === "done" ? svgIcon("check") : document.createTextNode(String(i + 1)));
    });
    live.message.textContent = p?.message || "准备中…";
    live.count.textContent = p?.total ? `${formatNumber(p.current || 0)} / ${formatNumber(p.total)}` : "";
    if (pct === null) live.track.removeAttribute("aria-valuenow");
    else live.track.setAttribute("aria-valuenow", String(pct));
    const c = p?.counts || state.survey?.counts || {};
    live.counters.seeds.textContent = formatNumber(c.seeds ?? 0);
    live.counters.candidates.textContent = formatNumber(c.candidates ?? 0);
    live.counters.read.textContent = formatNumber(c.read ?? 0);
  }

  function runningView(s) {
    const steps = stagesOf().map((stage) => {
      const dot = el("span", { class: "sv-step-dot", "aria-hidden": "true" });
      const li = el("li", { class: "sv-step" }, dot, el("span", { class: "sv-step-label", text: stage.label }));
      return { li, dot };
    });
    const message = el("span", { class: "progress-text" });
    const count = el("span", { class: "progress-count" });
    const bar = el("div", { class: "progress-fill indeterminate" });
    const track = el("div", { class: "progress-track", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-label": "调研进度" }, bar);
    const counter = (label) => {
      const value = el("dd", { class: "sv-counter-value", text: "0" });
      return { value, node: el("div", { class: "sv-counter" }, el("dt", { text: label }), value) };
    };
    const counters = { seeds: counter("种子文献"), candidates: counter("候选文献"), read: counter("已精读") };
    const cancel = el("button", { type: "button", class: "btn", "data-key": "cancel" }, svgIcon("x"), "取消调研");
    cancel.addEventListener("click", () => cancelSurvey(s.id, cancel));
    state.live = {
      kind: "view",
      id: s.id,
      steps,
      message,
      count,
      bar,
      track,
      counters: { seeds: counters.seeds.value, candidates: counters.candidates.value, read: counters.read.value },
    };
    const heading = el("h2", { class: "sv-title", tabindex: "-1", text: s.input?.topic || "调研" });
    state.heading = heading;
    const node = el(
      "section",
      { class: "card sv-head sv-run" },
      el("div", { class: "sv-head-top" }, heading, statusBadge({ status: "running", progress: null })),
      el("p", { class: "sv-meta", text: inputMeta(s) }),
      inputChips(s),
      el("ol", { class: "sv-steps", "aria-label": "调研步骤" }, steps.map((x) => x.li)),
      el("div", { class: "sv-live", role: "status" }, el("div", { class: "progress-row" }, message, count), track),
      el("dl", { class: "sv-counters" }, counters.seeds.node, counters.candidates.node, counters.read.node),
      el("div", { class: "sv-run-actions" }, cancel, el("span", { class: "sv-help", text: "调研在后台进行：可以关闭此窗口或继续使用 Zotero，完成后会通知你。" })),
    );
    updateLive(s.progress || (runningTask()?.id === s.id ? runningTask().progress : null));
    return node;
  }

  async function cancelSurvey(id, button) {
    button.disabled = true;
    button.textContent = "正在取消…";
    try {
      if (!api().cancelSurvey(id)) {
        toast("这项调研已经结束");
        load();
      }
    } catch (e) {
      button.disabled = false;
      toast(`取消失败：${e?.message || e}`, { tone: "error" });
    }
  }

  // -------------------------------------------------------- result view

  function statusBadge(s) {
    const status = statusOf(s);
    return el("span", { class: `status sv-status ${status.tone}`, title: status.title || null }, el("span", { class: `dot ${status.tone}` }), status.label);
  }

  /** "开始于 3 小时前 · 标准深度 · 起始年份 2010 · OpenAlex、Crossref" of the input. */
  function inputMeta(s) {
    const input = s.input || {};
    const sources = SOURCES.filter((src) => input.sources?.[src.key] !== false).map((src) => src.name);
    return [
      s.createdAt ? `开始于 ${formatRelative(s.createdAt)}` : null,
      DEPTH_LABELS[input.depth] ? `深度：${DEPTH_LABELS[input.depth]}` : null,
      input.yearFrom ? `${input.yearFrom} 年以来` : null,
      sources.length ? sources.join("、") : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }

  function resultMeta(s) {
    const c = s.counts || {};
    return [
      s.createdAt ? formatDateTime(s.createdAt) : null,
      DEPTH_LABELS[s.input?.depth] ? `深度：${DEPTH_LABELS[s.input.depth]}` : null,
      c.seeds ? `种子 ${formatNumber(c.seeds)} 篇` : null,
      `候选 ${formatNumber(c.candidates ?? 0)} 篇`,
      c.read ? `精读 ${formatNumber(c.read)} 篇` : null,
      s.report?.model || s.model ? `模型 ${s.report?.model || s.model}` : s.llm === false ? "未使用大模型" : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }

  function inputChips(s) {
    const input = s.input || {};
    const keywords = Array.isArray(input.keywords) ? input.keywords : parseKeywords(input.keywords);
    const seeds = Array.isArray(s.seeds) && s.seeds.length ? s.seeds : [];
    if (!keywords.length && !seeds.length && !input.description) return null;
    return el(
      "dl",
      { class: "sv-inputs" },
      input.description ? [el("dt", { text: "描述" }), el("dd", { class: "sv-desc", text: input.description })] : null,
      keywords.length ? [el("dt", { text: "关键词" }), el("dd", {}, keywords.map((k) => el("span", { class: "chip", text: k })))] : null,
      seeds.length
        ? [el("dt", { text: "种子" }), el("dd", {}, seeds.map((seed) => el("span", { class: "chip sv-seed-chip", title: seed.title, text: seed.title || seed.key })))]
        : null,
    );
  }

  function toolbarButton(key, label, icon, onclick, { cls = "btn", disabled = false, title = null, expanded = null } = {}) {
    const busy = state.action === key;
    return el(
      "button",
      {
        type: "button",
        class: cls,
        "data-key": key,
        disabled: !!state.action || disabled,
        title,
        "aria-busy": busy ? "true" : null,
        "aria-expanded": expanded === null ? null : expanded ? "true" : "false",
        onclick,
      },
      svgIcon(icon),
      busy ? "请稍候…" : label,
    );
  }

  function toolbar(s) {
    const hasReport = !!s.report;
    const resumable = s.status !== "done";
    const other = otherRunning(s.id);
    const llm = !!options().llm;
    const regenerateTitle = !llm
      ? "需要开启大模型（设置 → AI 助手）"
      : other
        ? "已有一项调研正在进行"
        : s.report?.kind === "list"
          ? "精读排名靠前的文献并撰写综述，不重新检索"
          : "用当前的大模型设置重新撰写综述，不重新检索";
    return el(
      "div",
      { class: "sv-toolbar", role: "toolbar", "aria-label": "调研操作" },
      resumable
        ? toolbarButton("resume", "继续", "play", () => resume(), { cls: "btn primary", disabled: other, title: other ? "已有一项调研正在进行" : "从中断的那一步继续" })
        : null,
      (state.saveButton = hasReport
        ? toolbarButton("save", "保存到 Zotero", "note", () => toggleSave(), {
            expanded: isSaveOpen(),
            cls: resumable ? "btn" : "btn primary",
            disabled: state.save.running,
            title: "保存为 Zotero 笔记，可同时加入文献",
          })
        : null),
      hasReport ? toolbarButton("export", "导出 Markdown", "download", () => exportMarkdown(), { title: "保存为 Markdown 文件" }) : null,
      s.status === "done"
        ? toolbarButton("regenerate", s.report?.kind === "list" ? "生成综述" : "重新生成综述", "sparkle", () => regenerate(), { disabled: !llm || other, title: regenerateTitle })
        : null,
      el("span", { class: "sv-toolbar-spacer" }),
      toolbarButton("delete", "删除", "trash", () => remove(), { cls: "btn ghost sv-delete", title: "删除这项调研（不影响已保存到 Zotero 的内容）" }),
    );
  }

  function processDetails(s) {
    const lines = processLines(s);
    if (!lines.length) return null;
    return el(
      "details",
      { class: "sv-process" },
      el("summary", { class: "sv-process-summary" }, svgIcon("chevron-right", "icon sv-process-chevron"), "调研过程与检索词"),
      el("dl", { class: "sv-process-list" }, lines.flatMap(([label, text]) => [el("dt", { text: label }), el("dd", { text })])),
    );
  }

  function headCard(s) {
    const r = s.report;
    const title = r?.title || s.input?.topic || "文献调研";
    const heading = el("h2", { class: "sv-title", tabindex: "-1", text: title });
    state.heading = heading;
    state.saveHost = el("div", { class: "sv-save-host" });
    renderSave();
    return el(
      "section",
      { class: "card sv-head" },
      el("div", { class: "sv-head-top" }, heading, statusBadge(s)),
      el("p", { class: "sv-meta", text: resultMeta(s) }),
      inputChips(s),
      toolbar(s),
      state.saveHost,
      processDetails(s),
    );
  }

  function calloutNode(tone, icon, title, text, actions = []) {
    return el(
      "div",
      { class: `callout ${tone}` },
      svgIcon(icon),
      el("div", { class: "callout-body" }, el("div", { class: "callout-title", text: title }), text ? el("div", { class: "callout-text", text }) : null),
      actions.length ? el("div", { class: "callout-actions" }, actions) : null,
    );
  }

  function statusCallout(s) {
    if (s.status === "done") {
      return s.report?.partial ? calloutNode("warning", "alert", "部分章节生成失败", "这些章节改为列出各文献的主要发现；可以稍后「重新生成综述」。") : null;
    }
    const last = lastStageLabel(s, stagesOf());
    const where = last ? `已完成到「${last}」，「继续」会从下一步开始。` : "还没有完成任何一步，「继续」会从头开始。";
    const busy = otherRunning(s.id) ? "另一项调研完成或取消后才能继续。" : "";
    if (s.status === "failed") {
      const stage = s.error?.stage ? `（${stageLabel(s.error.stage)}）` : "";
      return calloutNode("error", "alert", `调研失败${stage}`, `${String(s.error?.message || "未知错误").replace(/[。.]$/, "")}。${where}${busy}`);
    }
    if (s.interrupted) return calloutNode("warning", "alert", "调研被中断（Zotero 已关闭）", `${where}${busy}`);
    return calloutNode("info", "info", "调研已取消", `${where}${busy}`);
  }

  function problemsCallout(s) {
    const errors = (Array.isArray(s.errors) ? s.errors : []).filter(isRecord);
    if (!errors.length) return null;
    const line = (e) => el("li", { text: `${[stageLabel(e.stage), e.source].filter(Boolean).join(" · ")}：${e.message || ""}` });
    const list = el("ul", { class: "callout-list" }, errors.slice(0, MAX_PROBLEMS).map(line));
    if (errors.length > MAX_PROBLEMS) {
      const more = el("button", { type: "button", class: "link-btn" }, `更多（${errors.length - MAX_PROBLEMS}）…`);
      more.addEventListener("click", () => list.replaceChildren(...errors.map(line)));
      list.append(el("li", {}, more));
    }
    return el(
      "div",
      { class: "callout warning sv-problems" },
      svgIcon("alert"),
      // Only a finished survey went on past its problems; a stopped one says why in its own callout
      el("div", { class: "callout-body" }, el("div", { class: "callout-title", text: `调研中有 ${errors.length} 个问题${s.status === "done" ? "（其余步骤照常完成）" : ""}` }), list),
    );
  }

  function renderSurvey() {
    const root = $("survey-root");
    const s = state.survey;
    state.refNodes = new Map();
    state.live = null;
    state.heading = null;
    const back = el("a", { class: "sv-back", href: "#survey" }, svgIcon("chevron-left"), "调研列表");
    if (!s || s.id !== state.id) {
      let body;
      if (state.loadError) {
        body = calloutNode("error", "alert", "无法读取这项调研", state.loadError, [el("button", { type: "button", class: "btn small", onclick: () => load() }, "重试")]);
      } else if (state.loading || (s && s.id !== state.id)) {
        body = el("p", { class: "sv-loading", text: "正在加载…" });
      } else {
        body = emptyState({
          icon: "survey",
          title: "找不到这项调研",
          text: "它可能已被删除，或数据目录已被清除。",
          action: { label: "返回调研列表", onclick: () => navigate("survey") },
        });
      }
      root.replaceChildren(el("div", { class: "sv-detail" }, back, body));
      return;
    }
    withFocus(() => {
      if (s.running) {
        root.replaceChildren(el("div", { class: "sv-detail" }, back, runningView(s)));
        return;
      }
      state.bodyHost = el("div", { class: "sv-body-host" });
      root.replaceChildren(el("div", { class: "sv-detail" }, back, headCard(s), statusCallout(s), problemsCallout(s), state.bodyHost));
      renderBody();
    });
  }

  /**
   * Tab of the single column: the one chosen, else the review, or the
   * candidates found so far of a survey without one (cancelled or failed).
   */
  function activeTab(s) {
    if (["report", "candidates", "themes"].includes(state.tab)) return state.tab;
    return !s?.report && Array.isArray(s?.pool) && s.pool.length ? "candidates" : "report";
  }

  /** The tabs and panes under the header (re-rendered alone on a tab, filter or sort change). */
  function renderBody() {
    const s = state.survey;
    if (!s || !state.bodyHost) return;
    state.refNodes = new Map();
    const themes = Array.isArray(s.themes) ? s.themes : [];
    const pool = Array.isArray(s.pool) ? s.pool : [];
    const reportLabel = s.report?.kind === "list" ? "文献分组" : "综述";
    let node;
    if (isWide()) {
      const side = state.side === "themes" ? "themes" : "candidates";
      const sideSeg = seg(
        "side",
        [
          ["candidates", "候选文献", countBadge(pool.length)],
          ["themes", "主题", countBadge(themes.length)],
        ],
        side,
        (value) => {
          state.side = value;
          state.tab = value;
          renderBody();
        },
        { "aria-label": "候选文献与主题", class: "seg sv-tabs-seg" },
      );
      node = el(
        "div",
        { class: "sv-result wide" },
        el("div", { class: "sv-col sv-col-main" }, reportPane(s)),
        el("div", { class: "sv-col sv-col-side" }, el("div", { class: "sv-tabs" }, sideSeg.node), side === "themes" ? themesPane(s) : candidatesPane(s)),
      );
    } else {
      const tab = activeTab(s);
      const tabSeg = seg(
        "tab",
        [
          ["report", reportLabel],
          ["candidates", "候选文献", countBadge(pool.length)],
          ["themes", "主题", countBadge(themes.length)],
        ],
        tab,
        (value) => {
          state.tab = value;
          if (value !== "report") state.side = value;
          renderBody();
        },
        { "aria-label": "调研结果", class: "seg sv-tabs-seg" },
      );
      const pane = tab === "candidates" ? candidatesPane(s) : tab === "themes" ? themesPane(s) : reportPane(s);
      node = el("div", { class: "sv-result" }, el("div", { class: "sv-tabs" }, tabSeg.node), pane);
    }
    withFocus(() => state.bodyHost.replaceChildren(node));
  }

  // ---------------------------------------------------------- report

  function referenceByN(n) {
    return (state.survey?.references || []).find((r) => r.n === n) || null;
  }

  function citeButton(n) {
    const ref = referenceByN(n);
    if (!ref) return el("span", { class: "sv-cite missing", text: String(n) });
    return el(
      "button",
      { type: "button", class: "sv-cite", title: [ref.title, referenceMeta(ref)].filter(Boolean).join("\n"), "aria-label": `参考文献 ${n}：${ref.title || ""}`, onclick: () => jumpToRef(n) },
      String(n),
    );
  }

  /** Text with its [n] groups as superscript buttons that jump to the reference. */
  function citedText(tag, text, attrs = {}) {
    const children = citationParts(text).map((part) =>
      part.refs ? el("sup", { class: "sv-cite-group" }, "[", ...part.refs.flatMap((n, i) => [i ? ", " : null, citeButton(n)]), "]") : part.text,
    );
    return el(tag, attrs, ...children);
  }

  /** Scroll to a reference and highlight it (switching to the 综述 tab in one column). */
  function jumpToRef(n) {
    if (!isWide() && activeTab(state.survey) !== "report") {
      state.tab = "report";
      renderBody();
    }
    const node = state.refNodes.get(n);
    if (!node) return;
    node.scrollIntoView?.({ block: "center" });
    node.focus?.({ preventScroll: true });
    if (state.flash) {
      state.flash.node.classList.remove("flash");
      clearTimeout(state.flash.timer);
    }
    node.classList.add("flash");
    state.flash = { node, timer: setTimeout(() => node.classList.remove("flash"), FLASH_MS) };
  }

  function reportSection(heading, children, cls = "") {
    return el("section", { class: `sv-sec ${cls}`.trim() }, el("h3", { class: "sv-sec-title", text: heading }), children);
  }

  function reviewArticle(s) {
    const r = s.report;
    const list = (items) => el("ul", { class: "sv-list" }, items.map((t) => citedText("li", t)));
    const reading = (r.reading || []).filter((n) => referenceByN(n));
    return el(
      "article",
      { class: "card sv-report", "aria-label": "综述" },
      el("p", { class: "sv-ai-notice" }, svgIcon("sparkle"), AI_NOTICE),
      r.summary ? reportSection("总述", paragraphsOf(r.summary).map((p) => citedText("p", p, { class: "sv-par" })), "sv-summary") : null,
      (r.sections || []).map((section) =>
        reportSection(
          section.heading || "",
          (section.paragraphs || []).map((p) => citedText("p", p, { class: "sv-par" })),
          section.unverified ? "unverified" : "",
        ),
      ),
      r.debates?.length ? reportSection("主要争议", list(r.debates)) : null,
      r.gaps?.length ? reportSection("研究空白", list(r.gaps)) : null,
      r.future?.length ? reportSection("未来方向", list(r.future)) : null,
      reading.length
        ? reportSection(
            "推荐阅读顺序",
            el(
              "ol",
              { class: "sv-reading" },
              reading.map((n) => el("li", {}, el("span", { class: "sv-reading-n" }, "[", citeButton(n), "]"), el("span", { class: "sv-reading-title", text: referenceByN(n).title || "" }))),
            ),
          )
        : null,
    );
  }

  function listArticle(s) {
    const r = s.report;
    // With the LLM on now, the toolbar's 生成综述 writes the review; otherwise point to the settings
    const notice = options().llm
      ? calloutNode("info", "info", "这项调研没有使用大模型", `${NO_LLM_TEXT}现在已开启大模型，可以「生成综述」：先精读排名靠前的文献，再按主题撰写。`)
      : calloutNode("info", "info", "AI 综述需要开启大模型", NO_LLM_TEXT, [el("a", { class: "btn small", href: "#settings/ai" }, "去设置")]);
    return el(
      "article",
      { class: "card sv-report", "aria-label": "文献分组" },
      notice,
      (r.groups || []).map((g) =>
        reportSection(
          g.label || "",
          el(
            "ol",
            { class: "sv-reading" },
            (g.refs || [])
              .filter((n) => referenceByN(n))
              .map((n) => el("li", {}, el("span", { class: "sv-reading-n" }, "[", citeButton(n), "]"), el("span", { class: "sv-reading-title", text: referenceByN(n).title || "" }))),
          ),
        ),
      ),
    );
  }

  function libraryButton(paper) {
    const busy = state.busy.get(paper.id);
    if (paper.inLibrary) {
      return el(
        "button",
        { type: "button", class: "sv-lib", "data-key": `show:${paper.id}`, disabled: !paper.itemKey, title: paper.itemKey ? "在 Zotero 中显示" : "已在文库中", onclick: () => showInZotero(paper) },
        svgIcon("check"),
        "已在文库",
      );
    }
    const running = !!state.survey?.running;
    return el(
      "button",
      { type: "button", class: "btn ghost small sv-add", "data-key": `add:${paper.id}`, disabled: !!busy || running, title: "加入 Zotero（放进这项调研的分类）", onclick: () => addPaper(paper) },
      busy ? null : svgIcon("plus"),
      busy || "加入 Zotero",
    );
  }

  function openLink(paper) {
    const link = linkOf(paper);
    return link ? el("a", { class: "btn ghost small sv-open", href: link, title: link }, svgIcon("external"), "打开") : null;
  }

  function rowError(paper) {
    const message = state.rowErrors.get(paper.id);
    return message ? el("span", { class: "sv-row-error", role: "alert", text: message }) : null;
  }

  function referenceItem(ref) {
    const paper = (state.survey.pool || []).find((p) => p.id === ref.id) || null;
    const target = paper || ref;
    const read = paper?.read;
    const expanded = state.expanded.has(ref.id);
    const link = linkOf(ref);
    const noteToggle = read
      ? el(
          "button",
          { type: "button", class: "link-btn sv-note-toggle", "data-key": `note:${ref.id}`, "aria-expanded": expanded ? "true" : "false", onclick: () => toggleNote(ref.id) },
          expanded ? "收起精读笔记" : "精读笔记",
        )
      : null;
    const li = el(
      "li",
      { class: "sv-ref", tabindex: "-1" },
      el("span", { class: "sv-ref-n", text: `[${ref.n}]` }),
      el(
        "div",
        { class: "sv-ref-body" },
        el("div", { class: "sv-ref-title", text: ref.title || "（无标题）" }),
        el("div", { class: "sv-ref-meta", text: referenceMeta(ref) }),
        el(
          "div",
          { class: "sv-ref-actions" },
          link ? el("a", { class: "sv-doi", href: link, title: "在浏览器中打开" }, svgIcon("external"), ref.doi ? `doi:${ref.doi}` : "原文") : null,
          libraryButton(target),
          noteToggle,
          rowError(target),
        ),
        read && expanded
          ? el(
              "dl",
              { class: "sv-note" },
              NOTE_FIELDS.filter(([key]) => read[key]).flatMap(([key, label]) => [el("dt", { text: label }), el("dd", { text: read[key] })]),
              Number.isInteger(read.relevance) ? [el("dt", { text: "相关度" }), el("dd", { text: `${read.relevance} / 5` })] : null,
            )
          : null,
      ),
    );
    state.refNodes.set(ref.n, li);
    return li;
  }

  function toggleNote(id) {
    if (state.expanded.has(id)) state.expanded.delete(id);
    else state.expanded.add(id);
    renderBody();
  }

  function referencesCard(s) {
    const refs = (s.references || []).filter(isRecord);
    if (!refs.length) return null;
    const inLibrary = refs.filter((r) => {
      const paper = (s.pool || []).find((p) => p.id === r.id);
      return (paper || r).inLibrary;
    }).length;
    return el(
      "section",
      { class: "card sv-refs", "aria-labelledby": "sv-refs-title" },
      el(
        "div",
        { class: "card-header" },
        el("h3", { class: "card-title", id: "sv-refs-title", text: "参考文献" }),
        el("span", { class: "card-caption", text: `${formatNumber(refs.length)} 篇${inLibrary ? ` · 已在文库 ${formatNumber(inLibrary)} 篇` : ""}` }),
      ),
      el("ol", { class: "sv-ref-list" }, refs.map(referenceItem)),
    );
  }

  function reportPane(s) {
    const r = s.report;
    let article;
    if (r?.kind === "review") article = reviewArticle(s);
    else if (r?.kind === "list") article = listArticle(s);
    else {
      const resumable = s.status !== "done";
      article = el(
        "div",
        { class: "card" },
        emptyState({
          icon: "note",
          title: "还没有生成综述",
          text: resumable ? "调研还没有进行到撰写综述这一步；「继续」后会接着完成。" : "这项调研没有生成结果。",
          compact: true,
        }),
      );
    }
    return el("div", { class: "sv-pane sv-report-pane" }, article, referencesCard(s));
  }

  // ------------------------------------------------------- candidates

  function sortBy(key) {
    const t = state.table;
    if (t.sort === key) t.dir = t.dir === "asc" ? "desc" : "asc";
    else {
      t.sort = key;
      t.dir = key === "title" ? "asc" : "desc";
    }
    renderBody();
  }

  function candidateRow(p) {
    const ref = (state.survey.references || []).find((r) => r.id === p.id);
    const total = Number.isFinite(p.scores?.total) ? clamp(p.scores.total, 0, 1) : 0;
    const fill = el("span", { class: "sv-score-fill" });
    fill.style.width = `${Math.round(total * 100)}%`;
    return el(
      "tr",
      { class: p.read ? "is-read" : null },
      el("td", { class: "num sv-rank", text: Number.isFinite(p.rank) ? String(p.rank) : "" }),
      el("td", { class: "num sv-score", title: scoreTitle(p) }, el("span", { class: "sv-score-value", text: scoreText(p) }), el("span", { class: "sv-score-bar", "aria-hidden": "true" }, fill)),
      el(
        "td",
        { class: "sv-paper" },
        el("div", { class: "sv-paper-title", text: p.title || "（无标题）" }),
        el("div", { class: "sv-paper-meta", text: [authorsText(p.authors, p.authorCount), p.venue].filter(Boolean).join(" · ") }),
        el(
          "div",
          { class: "sv-paper-foot" },
          provenance(p.via).map((c) => el("span", { class: `sv-via ${c.key}`, title: c.title, text: c.label })),
          p.read ? el("span", { class: "sv-read" }, "已精读", ref ? [" [", citeButton(ref.n), "]"] : null) : null,
          el("span", { class: "sv-row-actions" }, libraryButton(p), openLink(p)),
          rowError(p),
        ),
      ),
      el("td", { class: "num", text: Number.isInteger(p.year) ? String(p.year) : "–" }),
      el("td", { class: "num", text: Number.isFinite(p.citations) ? formatNumber(p.citations) : "–" }),
    );
  }

  function candidatesPane(s) {
    const t = state.table;
    const pool = (Array.isArray(s.pool) ? s.pool : []).filter(isRecord);
    if (!pool.length) {
      return el("div", { class: "sv-pane" }, el("div", { class: "card" }, emptyState({ icon: "search", title: "还没有候选文献", text: "调研进行到「扩展」之后，这里会列出找到的文献。", compact: true })));
    }
    const filterSeg = seg(
      "filter",
      [
        ["all", "全部", countBadge(pool.length)],
        ["new", "未在文库", countBadge(pool.filter((p) => !p.inLibrary).length)],
        ["read", "已精读", countBadge(pool.filter((p) => p.read).length)],
      ],
      t.filter,
      (value) => {
        t.filter = value;
        t.limit = PAGE_SIZE;
        renderBody();
      },
      { "aria-label": "筛选候选文献" },
    );
    const rows = candidateRows(pool, t);
    const header = (key, label, cls = null) => {
      const active = t.sort === key;
      return el(
        "th",
        { class: cls, scope: "col", "aria-sort": active ? (t.dir === "asc" ? "ascending" : "descending") : "none" },
        el(
          "button",
          { type: "button", class: active ? "sv-sort active" : "sv-sort", "data-key": `sort:${key}`, onclick: () => sortBy(key) },
          label,
          active ? svgIcon(t.dir === "asc" ? "chevron-up" : "chevron-down", "icon sv-sort-icon") : null,
        ),
      );
    };
    const table = el(
      "table",
      { class: "table sv-table" },
      el("colgroup", {}, el("col", { class: "c-rank" }), el("col", { class: "c-score" }), el("col", { class: "c-paper" }), el("col", { class: "c-year" }), el("col", { class: "c-cites" })),
      el("thead", {}, el("tr", {}, el("th", { class: "num", scope: "col", title: "综合排名" }, "#"), header("score", "得分", "num"), header("title", "文献"), header("year", "年份", "num"), header("citations", "被引", "num"))),
      el("tbody", {}, rows.slice(0, t.limit).map(candidateRow)),
    );
    const more =
      rows.length > t.limit
        ? el(
            "button",
            {
              type: "button",
              class: "btn small sv-more",
              "data-key": "more",
              onclick: () => {
                t.limit += PAGE_SIZE;
                renderBody();
              },
            },
            `显示更多（还有 ${formatNumber(rows.length - t.limit)} 篇）`,
          )
        : null;
    return el(
      "div",
      { class: "sv-pane" },
      el(
        "section",
        { class: "card sv-cands", "aria-label": "候选文献" },
        el(
          "div",
          { class: "sv-cands-bar" },
          filterSeg.node,
          el("span", { class: "sv-cands-caption", text: "来源：检索 · 引用（引用了种子）· 被引（种子的参考文献）· 推荐" }),
        ),
        rows.length
          ? el("div", { class: "sv-table-wrap" }, table)
          : emptyState({ icon: "filter", title: t.filter === "new" ? "候选文献都已在文库中" : "没有符合条件的文献", compact: true }),
        more,
      ),
    );
  }

  // ----------------------------------------------------------- themes

  function themesPane(s) {
    const themes = (Array.isArray(s.themes) ? s.themes : []).filter(isRecord);
    if (!themes.length) {
      return el("div", { class: "sv-pane" }, el("div", { class: "card" }, emptyState({ icon: "survey", title: "还没有划分主题", text: "调研进行到「主题」这一步后，这里会显示聚类得到的主题和各自的文献。", compact: true })));
    }
    const byId = new Map((s.pool || []).filter(isRecord).map((p) => [p.id, p]));
    const refById = new Map((s.references || []).filter(isRecord).map((r) => [r.id, r]));
    return el(
      "div",
      { class: "sv-pane sv-themes" },
      themes.map((theme, i) => {
        const ids = Array.isArray(theme.paperIds) ? theme.paperIds : [];
        return el(
          "section",
          { class: "card sv-theme" },
          el("div", { class: "sv-theme-head" }, el("span", { class: "sv-theme-index", text: `主题 ${i + 1}` }), el("span", { class: "pill", text: `${formatNumber(ids.length)} 篇` })),
          el("h3", { class: "sv-theme-title", text: theme.label || `主题 ${i + 1}` }),
          theme.summary ? el("p", { class: "sv-theme-summary", text: theme.summary }) : null,
          Array.isArray(theme.terms) && theme.terms.length ? el("div", { class: "sv-terms" }, theme.terms.map((term) => el("span", { class: "chip", text: term }))) : null,
          el(
            "ol",
            { class: "sv-theme-papers" },
            ids.map((id) => {
              const paper = byId.get(id);
              const ref = refById.get(id);
              if (!paper && !ref) return null;
              const p = paper || ref;
              return el(
                "li",
                {},
                ref ? el("span", { class: "sv-reading-n" }, "[", citeButton(ref.n), "]") : null,
                el("span", { class: "sv-theme-paper" }, el("span", { class: "sv-reading-title", text: p.title || "" }), p.year ? el("span", { class: "sv-theme-year", text: ` · ${p.year}` }) : null),
              );
            }),
          ),
        );
      }),
    );
  }

  // ---------------------------------------------------------- actions

  /** Run a toolbar action with the toolbar disabled; errors become toasts. */
  async function act(key, fn) {
    if (state.action) return;
    state.action = key;
    render();
    try {
      await fn();
    } catch (e) {
      if (!e?.cancelled) toast(e?.message || String(e), { tone: "error" });
      Zotero.debug?.(`[ZotWatch] survey ${key}: ${e?.message || e}`);
    } finally {
      state.action = null;
      if (state.visible) render();
    }
  }

  function resume() {
    const id = state.survey?.id;
    if (!id) return Promise.resolve();
    return act("resume", async () => {
      await api().resumeSurvey(id);
      state.options = { ...options(), running: { id, progress: null } };
      setNavDot(true);
      await load();
    });
  }

  function regenerate() {
    const s = state.survey;
    if (!s) return Promise.resolve();
    const text =
      s.report?.kind === "review"
        ? "将使用当前的大模型设置重新撰写综述（不重新检索和精读），现有综述会被替换。确定继续吗？"
        : "将先精读排名靠前的文献，再按主题撰写综述（需要较多的 AI 调用），现有的文献分组会被替换。确定继续吗？";
    if (!confirmAction(text)) return Promise.resolve();
    return act("regenerate", async () => {
      await api().regenerateReport(s.id);
      state.options = { ...options(), running: { id: s.id, progress: null } };
      setNavDot(true);
      await load();
    });
  }

  function remove() {
    const s = state.survey;
    if (!s) return Promise.resolve();
    if (!confirmAction(`删除调研「${s.input?.topic || ""}」？已保存到 Zotero 的笔记和文献不受影响。`)) return Promise.resolve();
    return act("delete", async () => {
      await api().deleteSurvey(s.id);
      toast("已删除这项调研");
      state.survey = null;
      navigate("survey");
    });
  }

  /** A save path from Zotero's FilePicker, or null when cancelled. */
  async function pickSavePath(filename) {
    const { FilePicker } = ChromeUtils.importESModule("chrome://zotero/content/modules/filePicker.mjs");
    const fp = new FilePicker();
    fp.init(window, "导出 Markdown", fp.modeSave);
    fp.appendFilter("Markdown", "*.md");
    fp.defaultString = filename;
    fp.defaultExtension = "md";
    const rv = await fp.show();
    if (rv !== fp.returnOK && rv !== fp.returnReplace) return null;
    let path = typeof fp.file === "string" ? fp.file : fp.file?.path;
    if (!path) return null;
    // A name typed without any extension gets ".md"
    if (!/\.[^\\/.]+$/.test(path)) path += ".md";
    return path;
  }

  function exportMarkdown() {
    const id = state.survey?.id;
    if (!id) return Promise.resolve();
    return act("export", async () => {
      const { filename } = await api().exportSurveyMarkdown(id);
      const path = await pickSavePath(filename);
      if (!path) return;
      await api().exportSurveyMarkdown(id, path);
      toast(`已导出：${String(path).split(/[\\/]/).pop()}`);
    });
  }

  async function addPaper(paper) {
    const id = state.survey?.id;
    if (!id || state.busy.has(paper.id)) return;
    state.busy.set(paper.id, "正在加入…");
    state.rowErrors.delete(paper.id);
    renderBody();
    try {
      const result = await api().addSurveyPaper(id, paper.id);
      // Mark the paper (candidate and reference) until the reload arrives
      for (const p of [...(state.survey?.pool || []), ...(state.survey?.references || [])]) {
        if (p.id !== paper.id) continue;
        p.inLibrary = true;
        p.itemKey = result?.itemKey || p.itemKey || null;
      }
      toast(result?.existing ? "这篇文献已在 Zotero 中" : "已加入 Zotero");
    } catch (e) {
      state.rowErrors.set(paper.id, `加入失败：${e?.message || e}`);
    } finally {
      state.busy.delete(paper.id);
      if (state.survey?.id === id) renderBody();
    }
  }

  async function showInZotero(paper) {
    if (!paper.itemKey) return;
    try {
      if (!(await api().showItemByKey(paper.itemKey))) toast("这篇文献已不在 Zotero 文库中", { tone: "error" });
    } catch (e) {
      toast(e?.message || String(e), { tone: "error" });
    }
  }

  // ------------------------------------------------------------- save

  function toggleSave() {
    const sv = state.save;
    if (sv.running) return;
    sv.open = !sv.open;
    sv.result = null;
    sv.error = null;
    renderSave();
    if (sv.open) state.saveHost?.querySelector?.("input:checked")?.focus?.();
  }

  function saveOptions(s) {
    const refs = (s.references || []).length;
    if (s.report?.kind === "review") {
      const cited = citedNumbers(s.report).filter((n) => referenceByN(n)).length;
      return [
        ["cited", `笔记 + 综述引用的文献（${formatNumber(cited)} 篇）`],
        ["read", `笔记 + 全部精读文献（${formatNumber(refs)} 篇）`],
        ["none", "只保存笔记"],
      ];
    }
    return [
      ["cited", `笔记 + 列出的文献（${formatNumber(refs)} 篇）`],
      ["none", "只保存笔记"],
    ];
  }

  const isSaveOpen = () => !!(state.save.open || state.save.running || state.save.result || state.save.error);

  /** The 保存到 Zotero panel under the toolbar (re-rendered in place on progress). */
  function renderSave() {
    const host = state.saveHost;
    const s = state.survey;
    if (!host || !s) return;
    const sv = state.save;
    state.saveButton?.setAttribute("aria-expanded", isSaveOpen() ? "true" : "false");
    if (state.saveButton) state.saveButton.disabled = !!state.action || sv.running;
    if (!isSaveOpen()) {
      host.replaceChildren();
      return;
    }
    const choices = saveOptions(s);
    if (!choices.some(([key]) => key === sv.mode)) sv.mode = "cited";
    const radios = choices.map(([key, label]) => {
      const input = el("input", { type: "radio", name: "sv-save-mode", value: key });
      input.checked = sv.mode === key;
      input.disabled = sv.running;
      input.addEventListener("change", () => {
        if (input.checked) sv.mode = key;
      });
      return el("label", { class: "sv-radio" }, input, el("span", { text: label }));
    });
    let progress = null;
    if (sv.running) {
      const p = sv.progress;
      const fill = el("div", { class: p?.total ? "progress-fill" : "progress-fill indeterminate" });
      if (p?.total) fill.style.width = `${clamp(Math.round((100 * p.current) / p.total), 0, 100)}%`;
      progress = el(
        "div",
        { class: "sv-save-progress", role: "status" },
        el(
          "div",
          { class: "progress-row" },
          el("span", { class: "progress-text", text: p?.total ? `正在加入文献 ${formatNumber(Math.min(p.current + 1, p.total))}/${formatNumber(p.total)}${p.title ? `：${p.title}` : ""}` : "正在保存笔记…" }),
        ),
        el("div", { class: "progress-track" }, fill),
      );
    }
    const buttons = sv.running
      ? [el("button", { type: "button", class: "btn small", "data-key": "save-stop", onclick: () => api().cancelSurveySave(s.id) }, "停止导入")]
      : sv.result || sv.error
        ? [
            el("button", { type: "button", class: "btn small", "data-key": "save-again", onclick: () => doSave() }, sv.error ? "重试" : "再次保存"),
            el("button", { type: "button", class: "btn ghost small", "data-key": "save-close", onclick: () => closeSave() }, "完成"),
          ]
        : [
            el("button", { type: "button", class: "btn primary small", "data-key": "save-go", onclick: () => doSave() }, svgIcon("note"), "保存"),
            el("button", { type: "button", class: "btn ghost small", "data-key": "save-cancel", onclick: () => closeSave() }, "取消"),
          ];
    withFocus(() =>
      host.replaceChildren(
        el(
          "div",
          { class: "sv-save", role: "group", "aria-label": "保存到 Zotero" },
          el("div", { class: "sv-save-title", text: "保存到 Zotero" }),
          el("p", { class: "sv-help", text: `在分类「${collectionName(s)}」中保存一篇笔记（综述正文与编号参考文献，注明由 AI 生成）。已在文库中的文献直接加入分类，其余通过 DOI 导入。` }),
          s.saved?.at ? el("p", { class: "sv-help", text: `上次保存于 ${formatDateTime(s.saved.at)}；再次保存会更新同一篇笔记。` }) : null,
          el("div", { class: "sv-radios", role: "radiogroup", "aria-label": "一同加入的文献" }, radios),
          progress,
          sv.result ? el("p", { class: "sv-save-result", role: "status" }, svgIcon("check-circle"), saveResultText(sv.result)) : null,
          sv.error ? el("p", { class: "sv-row-error", role: "alert", text: `保存失败：${sv.error}` }) : null,
          el("div", { class: "sv-save-actions" }, buttons),
        ),
      ),
    );
  }

  function closeSave() {
    Object.assign(state.save, { open: false, result: null, error: null });
    renderSave();
    $("survey-root").querySelector?.('[data-key="save"]')?.focus?.();
  }

  async function doSave() {
    const s = state.survey;
    const sv = state.save;
    if (!s || sv.running) return;
    Object.assign(sv, { open: true, running: true, progress: null, result: null, error: null });
    renderSave();
    try {
      sv.result = await api().saveSurveyToZotero(s.id, { addPapers: sv.mode });
      toast(sv.result?.cancelled ? "已停止导入，笔记已保存" : "已保存到 Zotero");
    } catch (e) {
      sv.error = e?.message || String(e);
    } finally {
      sv.running = false;
      sv.progress = null;
    }
    if (state.survey?.id === s.id) await load();
  }

  // ----------------------------------------------------------- events

  function topicOf(id) {
    if (state.survey?.id === id) return state.survey.input?.topic || "";
    return (state.list || []).find((e) => e.id === id)?.topic || "";
  }

  function onEvent(event) {
    if (event?.type !== "survey") return;
    const setRunning = (task) => {
      state.options = { ...options(), running: task };
      setNavDot(!!task);
    };
    switch (event.action) {
      case "start":
        setRunning({ id: event.id, progress: null });
        if (state.visible && (!state.id || state.id === event.id)) load();
        break;
      case "progress":
        setRunning({ id: event.id, progress: event.progress || null });
        if (state.live?.id === event.id) updateLive(event.progress);
        else if (state.visible && !state.id && state.home) load();
        break;
      case "done":
      case "cancelled":
      case "error": {
        setRunning(null);
        const topic = topicOf(event.id);
        if (event.action === "done") toast(`调研完成${topic ? `：${topic}` : ""}`);
        else if (event.action === "error") toast(`调研失败：${event.message || "未知错误"}`, { tone: "error", duration: 5000 });
        if (state.visible) load();
        break;
      }
      case "deleted":
        if (state.id === event.id) {
          state.survey = null;
          if (state.visible) navigate("survey");
        } else if (state.visible && !state.id) load();
        break;
      case "updated":
      case "saved":
        if (state.visible && state.id === event.id && !state.save.running) load();
        break;
      case "save-progress":
        if (state.id === event.id && state.save.running) {
          state.save.progress = event.progress || null;
          renderSave();
        }
        break;
      default:
    }
  }

  // ------------------------------------------------------------ page

  function init() {
    $("survey-new").addEventListener("click", () => {
      navigate("survey");
      state.formNodes?.topic.focus?.();
    });
    loadOptions();
    api()?.addListener(onEvent);
    try {
      state.wide = window.matchMedia?.(WIDE_QUERY) || null;
      state.wide?.addEventListener?.("change", () => {
        if (state.visible && state.id && state.survey && !state.survey.running) renderBody();
      });
    } catch (e) {
      state.wide = null;
    }
    window.addEventListener("unload", () => api()?.removeListener(onEvent));
  }

  function show({ section = null } = {}) {
    const id = section || null;
    state.visible = true;
    loadOptions();
    const fromDraft = takeDraft();
    if (fromDraft && id) {
      // The draft belongs to the form: show the list instead of a survey
      navigate("survey");
      return;
    }
    if (id !== state.id) {
      state.id = id;
      state.survey = state.survey?.id === id ? state.survey : null;
      state.tab = null;
      state.table = { ...state.table, filter: "all", limit: PAGE_SIZE };
      state.expanded = new Set();
      state.rowErrors = new Map();
      Object.assign(state.save, { open: false, running: false, progress: null, result: null, error: null });
      $("pages").scrollTop = 0;
    }
    // load() marks the page as loading before it awaits: render() shows 正在加载… rather than 找不到
    load();
    render();
    if (fromDraft) state.formNodes?.topic.focus?.();
  }

  function hide() {
    state.visible = false;
    state.formNodes?.seedSearch.close();
  }

  /** Relative times of the list stay current. */
  function tick() {
    if (state.visible && !state.id && state.home) renderHistory();
  }

  return {
    init,
    show,
    hide,
    tick,
    refresh() {},
    heading: () => state.heading,
    load,
    render,
    state,
    // Pure helpers (tests)
    citationParts,
    citedNumbers,
    parseKeywords,
    validateInput,
    provenance,
    candidateRows,
    statusOf,
    progressLabel,
    stepStates,
    lastStageLabel,
    authorsText,
    referenceMeta,
    linkOf,
    saveResultText,
    processLines,
  };
})();

registerPage("survey", ZotWatchSurvey);
