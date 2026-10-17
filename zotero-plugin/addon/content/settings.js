/* global Zotero, ChromeUtils, window, $, el, svgIcon, api, registerPage, setSubtitle, setActiveSection, toast,
   confirmAction, refresh, runStatus, formatRelative, SETTINGS_SECTIONS, createSearchBox, searchResultText, searchCount */

// 设置 page. `#settings` shows an index of the sections (SETTINGS_SECTIONS in
// main.js) with a status line each; `#settings/<id>` shows that section's
// cards. Every control is built from FIELDS and bound to its preference
// (extensions.zotwatch.<pref>, marked with data-pref): values are read from
// Zotero.Prefs when a section is shown and written on change (text fields
// also 600 ms after typing stops) with the JS type of the pref's default.
// Pending edits are saved when the section or page changes and when the
// window closes (flush(), called by main.js). The journal whitelist is a data
// file, not a pref, and is saved through api().setJournalWhitelist().
//
// 关注期刊, 关注作者 and 顶刊地学 also have OpenAlex search boxes
// (createSearchBox, search.js): 「添加」 merges a journal into the list (one
// row per ISSN), 「关注」 appends an author line, and the parsed authors and
// the flagship journals (grouped by name) are listed with 移除 buttons; the
// texts stay editable under 「编辑文本」. 「从文库生成」 opens a review panel
// of proposals (api().suggestJournals()) inside the section, and
// 「补全学科与影响因子」 fills blank categories / impact factors from OpenAlex.

var ZotWatchSettings = (() => {
  const PREF_PREFIX = "extensions.zotwatch.";
  const INPUT_DELAY = 600;
  const WHITELIST_DELAY = 800;
  const VIEW_REFRESH_DELAY = 800;
  const README_URL = (version) => `https://github.com/CrazyHalfDay/AIZotWatch/blob/zotero-plugin-v${version}/zotero-plugin/README.md`;
  const RESET_CONFIRM = "将删除向量缓存、聚类结果和全部推荐记录，以及主题调研的记录（不会影响 Zotero 文库、期刊列表和已保存到 Zotero 的调研笔记）。下次运行会重新向量化文库。确定继续吗？";

  /** Embedding services; models and endpoints as in src/embedding/providers.js (EMBEDDING_PRESETS). */
  const PROVIDERS = {
    local: { name: "本地模式", title: "本地模式", text: "无需 API Key，效果一般" },
    dashscope: {
      name: "DashScope",
      title: "阿里云 DashScope",
      text: "text-embedding-v4，中英文效果都好",
      recommended: true,
      model: "text-embedding-v4",
      baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
      keyHelp: "在阿里云百炼控制台创建，与 AIZotWatch 的 DASHSCOPE_API_KEY 相同。",
    },
    voyage: { name: "Voyage AI", title: "Voyage AI", text: "voyage-3.5", model: "voyage-3.5", baseURL: "https://api.voyageai.com/v1", keyHelp: "在 dash.voyageai.com 创建。" },
    openai: {
      name: "OpenAI 兼容",
      title: "OpenAI 兼容接口",
      text: "OpenAI、硅基流动、Ollama 等",
      model: "text-embedding-3-small",
      baseURL: "https://api.openai.com/v1",
      keyHelp: "本地 Ollama 等不需要 Key 的服务可以留空。",
    },
  };

  /** 「快速填写」 for the LLM: endpoint and model of common services (same presets as the 0.2.0 settings pane). */
  const LLM_PRESETS = [
    { key: "openrouter", name: "OpenRouter", title: "OpenRouter", baseURL: "https://openrouter.ai/api/v1", model: "deepseek/deepseek-chat" },
    { key: "deepseek", name: "DeepSeek", title: "DeepSeek", baseURL: "https://api.deepseek.com/v1", model: "deepseek-chat" },
    { key: "kimi", name: "Kimi", title: "Kimi（Moonshot）", baseURL: "https://api.moonshot.cn/v1", model: "kimi-k2-turbo-preview" },
    { key: "dashscope", name: "DashScope", title: "阿里云百炼（通义千问）", baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: "qwen-plus" },
  ];

  const isProvider = (...names) => (get) => names.includes(get("embedding.provider") || "local");
  // Output sizes DashScope's text-embedding-v4 accepts (any other value fails every request)
  const DASHSCOPE_DIMENSIONS = [64, 128, 256, 512, 768, 1024, 1536, 2048];

  /**
   * Every bound preference. kind: switch | number | text | password | textarea
   * | radio | seg. Number fields mirror the parsing of src/settings.js: min,
   * max, int and the fallback used for a blank field (`auto` names a
   * provider-calibrated value of getAutoValues() instead); a test checks them
   * against loadSettings(). `when` hides a field that does not apply,
   * `restore` marks the long flagship texts that have 「恢复默认」.
   */
  const FIELDS = [
    // 常规
    { pref: "schedule.autoRun", kind: "switch", label: "自动检查新文献", help: "Zotero 运行时在后台进行：启动约 3 分钟后检查一次，之后每 30 分钟判断是否到期。" },
    { pref: "schedule.intervalHours", kind: "number", label: "检查间隔", help: "两次自动检查之间的间隔；检查失败后 2 小时再重试。", unit: "小时", min: 1, max: 720, fallback: 24 },
    { pref: "inbox.retentionDays", kind: "number", label: "推荐保留天数", help: "待查看的推荐超过这些天没有再出现就会移除；已加入或已忽略的记录至少保留 60 天。", unit: "天", min: 1, max: 365, int: true, fallback: 30 },

    // 向量化
    {
      pref: "embedding.provider",
      kind: "radio",
      label: "服务商",
      options: ["local", "dashscope", "voyage", "openai"].map((value) => ({ value, ...PROVIDERS[value] })),
    },
    { pref: "embedding.apiKey", kind: "password", label: "API Key", long: true, when: isProvider("dashscope", "voyage", "openai") },
    { pref: "embedding.model", kind: "text", label: "模型", help: "留空使用默认模型。", long: true, when: isProvider("dashscope", "voyage", "openai") },
    // Every API service sends its requests here (blank: the service's official endpoint, the placeholder)
    { pref: "embedding.baseURL", kind: "text", label: "接口地址", help: "留空使用所选服务的官方地址；用代理或国际站时填写。", long: true, when: isProvider("dashscope", "voyage", "openai") },
    {
      pref: "embedding.dimensions",
      kind: "number",
      label: "向量维度",
      help: `仅 DashScope 使用；text-embedding-v4 支持 ${DASHSCOPE_DIMENSIONS.join(" / ")}。更改后会重新向量化文库。`,
      min: 64,
      max: 2048,
      int: true,
      fallback: 1024,
      advanced: true,
      when: isProvider("dashscope"),
    },

    // 学术数据接口
    {
      pref: "apis.openalexKey",
      kind: "password",
      label: "OpenAlex API Key",
      help: "免费注册 OpenAlex 账号后，在账号设置中获取 API Key。",
      long: true,
    },
    {
      pref: "apis.semanticScholarKey",
      kind: "password",
      label: "Semantic Scholar API Key",
      help: "在 Semantic Scholar 的 API 申请页提交申请，审核通过后通过邮件获得。",
      long: true,
    },
    {
      pref: "sources.crossref.mailto",
      kind: "text",
      inputType: "email",
      label: "联系邮箱（Crossref 礼貌池）",
      help: "进入 Crossref 的礼貌池，请求更稳定（推荐填写）；也会随 OpenAlex 请求发送。",
      long: true,
      placeholder: "name@example.com",
    },

    // 文献来源
    { pref: "sources.crossref.enabled", kind: "switch", label: "Crossref", help: "按关注期刊的 ISSN 抓取正式发表的论文。" },
    { pref: "sources.crossref.daysBack", kind: "number", label: "回溯天数", help: "每次抓取最近这些天发表的论文。", unit: "天", min: 1, max: 60, int: true, fallback: 7 },
    { pref: "sources.arxiv.enabled", kind: "switch", label: "arXiv 预印本", help: "按分类抓取最近 7 天的预印本。" },
    { pref: "sources.arxiv.categories", kind: "text", label: "分类", help: "逗号分隔，留空使用默认分类；只保留主分类匹配的论文。", long: true, placeholder: "physics.geo-ph, astro-ph.EP" },
    { pref: "sources.eartharxiv.enabled", kind: "switch", label: "EarthArXiv 地学预印本", help: "抓取最近 14 天的地学预印本（OAI-PMH）。" },
    { pref: "abstracts.crossref", kind: "switch", label: "从 Crossref 补全", help: "用于预印本等不是从 Crossref 抓到的论文。" },
    { pref: "sources.openalexAbstracts", kind: "switch", label: "从 OpenAlex 补全", help: "按 DOI 批量查询；建议在「学术数据接口」中填写 OpenAlex API Key。" },
    { pref: "abstracts.elsevierKey", kind: "password", label: "Elsevier API Key", help: "可选，补全 10.1016 期刊的摘要。", long: true },
    { pref: "abstracts.springerKey", kind: "password", label: "Springer Nature API Key", help: "可选，补全 10.1007 / 10.1038 期刊的摘要。", long: true },

    // 关注期刊
    { pref: "journals.autoFromLibrary", kind: "switch", label: "自动关注文库中最常见的期刊", help: "取文库中出现至少 2 次的前 N 种期刊，与上面的列表合并；条目缺少 ISSN 时通过 Crossref 按刊名查询。" },
    { pref: "journals.autoTopN", kind: "number", label: "自动关注的期刊数量", unit: "种", min: 1, max: 300, int: true, fallback: 40 },

    // 关注作者
    {
      pref: "followed.enabled",
      kind: "switch",
      label: "关注指定作者的新论文",
      help: "通过 OpenAlex 按 ORCID 识别作者。关注作者的论文不参与评分，只与文库去重。未在「学术数据接口」中填写 OpenAlex API Key 时，每次只能获取第一页结果。",
    },
    { pref: "followed.daysBack", kind: "number", label: "检查最近 N 天发表的论文", help: "每次运行都查询这段时间，已出现过的论文不会再推荐。", unit: "天", min: 7, max: 365, int: true, fallback: 60 },
    {
      pref: "followed.authors",
      kind: "textarea",
      label: "作者列表",
      help: "每行一位作者：姓名 + ORCID（或 OpenAlex 作者 ID），以 # 开头的行会被忽略。",
      rows: 5,
      placeholder: "汪在聪, 0000-0002-3584-1673\nJay J. Ague, https://orcid.org/0000-0001-8408-0279",
    },

    // 顶刊地学
    { pref: "flagship.enabled", kind: "switch", label: "单独追踪顶刊中的地学文章", help: "Nature、Science、PNAS 等综合性顶刊中的固体地球科学与古生物文章，单独列为「顶刊地学速览」。" },
    { pref: "flagship.maxResults", kind: "number", label: "每次最多显示", help: "取最新发表的 N 篇；0 为不限。", unit: "篇", min: 0, max: 200, int: true, fallback: 30 },
    { pref: "flagship.llmFallback", kind: "switch", label: "灰区文章交给大模型复核", help: "需要在「AI 助手」中启用大模型；未启用时灰区文章全部保留。" },
    { pref: "flagship.minScore", kind: "number", label: "接受阈值", help: "与地学领域描述的相似度达到它就保留。", min: 0, max: 1, auto: "flagshipMinScore" },
    { pref: "flagship.grayLow", kind: "number", label: "灰区下限", help: "低于它直接排除；介于灰区下限和接受阈值之间的是灰区。", min: 0, max: 1, auto: "flagshipGrayLow" },
    { pref: "flagship.issns", kind: "textarea", label: "顶刊 ISSN", help: "逗号或换行分隔；期刊的任一 ISSN 命中即算顶刊文章。", rows: 3, mono: true, restore: "issns" },
    { pref: "flagship.positiveAnchor", kind: "textarea", label: "地学领域描述", help: "英文；摘要与这段描述越接近，越可能被保留。", rows: 3, restore: "positiveAnchor" },
    { pref: "flagship.negativeAnchors", kind: "textarea", label: "排除领域", help: "每行一个；离任一排除领域比地学领域描述更近的文章会被排除。", rows: 6, restore: "negativeAnchors" },
    { pref: "flagship.llmBoundary", kind: "textarea", label: "大模型复核标准", help: "灰区文章按这段标准交给大模型判断。", rows: 3, restore: "llmBoundary" },

    // 推荐与过滤
    { pref: "scoring.topK", kind: "number", label: "每次推荐数量", help: "个性推荐每次最多保留的篇数。", unit: "篇", min: 1, max: 500, int: true, fallback: 30 },
    { pref: "scoring.recentDays", kind: "number", label: "只保留最近 N 天发表", help: "0 为不限。", unit: "天", min: 0, max: 365, int: true, fallback: 7 },
    { pref: "scoring.minSimilarity", kind: "number", label: "最低相似度", help: "与文库的相似度低于它的论文不推荐；留空时按向量化服务自动取值。", min: 0, max: 1, auto: "minSimilarity" },
    { pref: "scoring.maxPreprintRatio", kind: "number", label: "预印本最大占比", help: "推荐中预印本最多占的比例。", min: 0, max: 1, fallback: 0.7 },
    { pref: "scoring.requireAbstract", kind: "switch", label: "排除没有摘要的文献", help: "没有摘要的论文很难可靠地判断相关性。" },
    { pref: "scoring.includeKeywords", kind: "textarea", label: "必须包含的关键词", help: "任一命中即保留，留空不限；逗号或换行分隔。", rows: 2 },
    { pref: "scoring.excludeKeywords", kind: "textarea", label: "排除关键词", help: "逗号或换行分隔；整词匹配，中文按子串匹配。", rows: 2 },
    {
      pref: "scoring.thresholdMode",
      kind: "seg",
      label: "阈值模式",
      help: "动态：按每批分数的第 95 / 70 百分位划分「必读 / 推荐」（设有下限）；固定：0.75 / 0.55。",
      options: [
        { value: "dynamic", title: "动态" },
        { value: "fixed", title: "固定" },
      ],
      advanced: true,
    },
    { pref: "scoring.temporalHalflifeDays", kind: "number", label: "时间衰减半衰期", help: "较早加入文库的条目在研究画像中的权重按半衰期递减；0 为不衰减。", unit: "天", min: 0, max: 36500, fallback: 0, advanced: true },

    // AI 助手
    { pref: "llm.enabled", kind: "switch", label: "启用大模型", help: "支持任意 OpenAI 兼容接口，如 OpenRouter、DeepSeek、Kimi、通义千问。" },
    { pref: "llm.baseURL", kind: "text", label: "接口地址", help: "OpenAI 兼容的 /chat/completions 接口。", long: true, placeholder: "https://openrouter.ai/api/v1" },
    { pref: "llm.apiKey", kind: "password", label: "API Key", long: true },
    { pref: "llm.model", kind: "text", label: "模型", long: true, placeholder: "deepseek/deepseek-chat" },
    { pref: "llm.translateTitles", kind: "switch", label: "将英文标题翻译为中文", help: "覆盖本次所有新增论文。" },
    { pref: "llm.summaryTopN", kind: "number", label: "每次自动生成摘要的篇数", help: "个性推荐取分数最高的 N 篇，顶刊地学和关注作者各另取最新的 N 篇。", unit: "篇", min: 0, max: 100, int: true, fallback: 10 },
    {
      pref: "llm.topicLabels",
      kind: "switch",
      label: "为文库研究方向生成主题标签",
      help: [
        "会把每个方向的代表性标题和常用 Zotero 标签发送给大模型。",
        "研究方向来自文库的语义聚类，标签只生成一次并在之后的运行中沿用（文库变化较大时才为新的方向补充标签）；推荐卡片和子笔记会显示论文最接近的研究方向。",
      ],
    },

    // 加入 Zotero
    { pref: "output.collection", kind: "text", label: "保存到分类", help: "加入文库的论文放进这个分类（不存在时自动创建）；留空则不放入分类。", long: true },
    { pref: "output.tag", kind: "text", label: "添加标签", help: "留空则不添加标签。", long: true },
    { pref: "output.useTranslator", kind: "switch", label: "通过 DOI / arXiv 号抓取完整元数据", help: "与 Zotero 的「通过标识符添加条目」相同；失败时使用推荐中的信息。" },
    { pref: "output.summaryNote", kind: "switch", label: "把 AI 摘要保存为子笔记", help: "已生成 AI 摘要的论文加入文库时附上一条子笔记。" },
  ];
  const FIELD = Object.fromEntries(FIELDS.map((f) => [f.pref, f]));

  // ------------------------------------------------------------ pure helpers

  /**
   * The value to store for a control value, with the JS type of the pref's
   * current value (which has its default's type: Zotero.Prefs.set throws on
   * another type). Numbers are stored as strings in this plugin; integer
   * prefs are handled anyway. undefined means "do not write" (no such pref).
   */
  function coercePrefValue(value, current) {
    switch (typeof current) {
      case "boolean":
        return value === true || value === "true";
      case "number": {
        const n = typeof value === "number" ? value : Number(String(value ?? "").trim());
        return Number.isFinite(n) ? Math.round(n) : current;
      }
      case "string":
        return value === null || value === undefined ? "" : String(value);
      default:
        return undefined;
    }
  }

  /**
   * How src/settings.js reads a number field (its num()): {value, reason},
   * where reason is null when the text is used as typed, "blank" or
   * "invalid" when `fallback` is used, "range" when clamped to min/max and
   * "adjusted" when only a leading number was read ("12.5" for an integer, "0.3x").
   */
  function effectiveNumber(raw, { min = -Infinity, max = Infinity, int = false } = {}, fallback = null) {
    const text = String(raw ?? "").trim();
    if (!text) return { value: fallback, reason: "blank" };
    const n = int ? parseInt(text, 10) : parseFloat(text);
    if (!Number.isFinite(n)) return { value: fallback, reason: "invalid" };
    const value = Math.min(max, Math.max(min, n));
    if (value !== n) return { value, reason: "range" };
    if (Number(text) !== n) return { value, reason: "adjusted" };
    return { value, reason: null };
  }

  /** A number as the settings show it: thresholds with two decimals, counts as they are. */
  function formatSettingValue(value, spec = {}) {
    if (!Number.isFinite(value)) return "";
    if (spec.max !== undefined && spec.max <= 1) return value.toFixed(2);
    return String(Math.round(value * 1000) / 1000);
  }

  /** "1–720 小时" */
  function rangeText(spec) {
    const unit = spec.unit ? ` ${spec.unit}` : "";
    return `${spec.min}–${spec.max}${unit}`;
  }

  /** Inline note under a number field whose value will not be used as typed ("" when it will). */
  function clampNote(raw, spec, fallback = null) {
    const { value, reason } = effectiveNumber(raw, spec, fallback);
    const shown = formatSettingValue(value, spec);
    if (reason === "invalid") return shown ? `不是有效的数字，将按${spec.auto ? "自动值" : "默认值"} ${shown} 使用` : "不是有效的数字，将按自动值使用";
    if (reason === "range") return `超出范围，将按 ${shown} 使用`;
    if (reason === "adjusted") return spec.int ? `只能填整数，将按 ${shown} 使用` : `将按 ${shown} 使用`;
    return "";
  }

  /** ISSN in "1234-567X" form, "" when the text is not one (same rule as src/core/text.js). */
  function normalizeIssn(value) {
    const compact = String(value || "").toUpperCase().replace(/[^0-9X]/g, "");
    return compact.length === 8 ? `${compact.slice(0, 4)}-${compact.slice(4)}` : "";
  }

  /** Distinct valid ISSNs in a comma/newline separated list. */
  function issnList(text) {
    return [...new Set(String(text || "").split(/[\n,;，；]+/).map(normalizeIssn).filter(Boolean))];
  }

  /** Distinct non-blank lines (anchors contain commas, so one per line). */
  function textLines(text) {
    return [...new Set(String(text || "").split(/\r?\n/).map((s) => s.trim()).filter(Boolean))];
  }

  /** The default text of a 「恢复默认」 field from getAutoValues().flagship. */
  function defaultText(spec, auto) {
    const value = auto?.flagship?.[spec.restore];
    return Array.isArray(value) ? value.join("\n") : value || "";
  }

  /** Whether a flagship text means the same as its default (blank counts as default). */
  function isDefaultText(spec, text, auto) {
    if (!String(text || "").trim()) return true;
    const fallback = defaultText(spec, auto);
    if (spec.restore === "issns") {
      const a = issnList(text);
      const b = issnList(fallback);
      return a.length === b.length && a.every((issn) => b.includes(issn));
    }
    return textLines(text).join("\n") === textLines(fallback).join("\n");
  }

  /** Summary line of a 「恢复默认」 field: count of entries and whether it is the default. */
  function restoreMeta(spec, text, auto) {
    const usesDefault = isDefaultText(spec, text, auto);
    const effective = String(text || "").trim() ? text : defaultText(spec, auto);
    const state = usesDefault ? "默认" : "已自定义";
    if (spec.restore === "issns") return `${issnList(effective).length} 个 ISSN · ${state}`;
    if (spec.restore === "negativeAnchors") return `${textLines(effective).length} 个排除领域 · ${state}`;
    return state;
  }

  /**
   * Status of each settings section for the index: {tone, text}, tone being a
   * status dot class (good, warning, critical, off, neutral). `get` returns a
   * pref value; ctx holds {whitelist, whitelistError, followed, version}.
   */
  function sectionStatus(id, get, ctx = {}) {
    const number = (pref) => effectiveNumber(get(pref), FIELD[pref], FIELD[pref].fallback).value;
    switch (id) {
      case "general":
        return get("schedule.autoRun") ? { tone: "good", text: `每 ${formatSettingValue(number("schedule.intervalHours"))} 小时自动检查` } : { tone: "off", text: "自动检查已关闭" };
      case "embedding": {
        const provider = PROVIDERS[get("embedding.provider")] ? get("embedding.provider") : "local";
        if (provider === "local") return { tone: "neutral", text: "本地模式" };
        const name = PROVIDERS[provider].name;
        if (String(get("embedding.apiKey") || "").trim()) return { tone: "good", text: `${name} · 已配置` };
        if (provider === "openai") return { tone: "neutral", text: `${name} · 未填写 API Key` };
        return { tone: "warning", text: `${name} · 未填写 API Key` };
      }
      case "apis": {
        const openalex = !!String(get("apis.openalexKey") || "").trim();
        const s2 = !!String(get("apis.semanticScholarKey") || "").trim();
        if (!openalex) return { tone: "warning", text: s2 ? "Semantic Scholar 已配置 · 缺少 OpenAlex Key" : "未配置 OpenAlex API Key" };
        return { tone: "good", text: s2 ? "OpenAlex、Semantic Scholar 已配置" : "OpenAlex 已配置" };
      }
      case "sources": {
        const names = [
          ["sources.crossref.enabled", "Crossref"],
          ["sources.arxiv.enabled", "arXiv"],
          ["sources.eartharxiv.enabled", "EarthArXiv"],
        ]
          .filter(([pref]) => get(pref))
          .map(([, name]) => name);
        return names.length ? { tone: "good", text: names.join("、") } : { tone: "warning", text: "未启用任何来源" };
      }
      case "journals": {
        if (ctx.whitelistError) return { tone: "critical", text: "期刊列表无法读取" };
        const count = ctx.whitelist?.count || 0;
        const auto = !!get("journals.autoFromLibrary");
        if (count) return { tone: "good", text: `${count} 个期刊${auto ? " · 并自动关注文库常见期刊" : ""}` };
        return auto ? { tone: "good", text: "自动关注文库常见期刊" } : { tone: "warning", text: "未设置关注期刊" };
      }
      case "followed": {
        if (!get("followed.enabled")) return { tone: "off", text: "未启用" };
        const authors = ctx.followed?.authors?.length || 0;
        const invalid = ctx.followed?.invalid?.length || 0;
        if (!authors) return { tone: "warning", text: "未填写有效的作者" };
        return invalid ? { tone: "warning", text: `${authors} 位作者，${invalid} 行无法识别` } : { tone: "good", text: `${authors} 位作者` };
      }
      case "flagship": {
        if (!get("flagship.enabled")) return { tone: "off", text: "未启用" };
        const issns = issnList(String(get("flagship.issns") || "").trim() ? get("flagship.issns") : defaultText(FIELD["flagship.issns"], ctx.auto)).length;
        const max = number("flagship.maxResults");
        return issns ? { tone: "good", text: `${issns} 个 ISSN · ${max ? `最多 ${max} 篇` : "不限篇数"}` } : { tone: "warning", text: "未填写有效的 ISSN" };
      }
      case "ranking": {
        const days = number("scoring.recentDays");
        return { tone: "neutral", text: `每次 ${number("scoring.topK")} 篇 · ${days ? `最近 ${days} 天` : "不限发表时间"}` };
      }
      case "ai": {
        if (!get("llm.enabled")) return { tone: "off", text: "未启用" };
        if (!String(get("llm.apiKey") || "").trim()) return { tone: "warning", text: "未填写 API Key" };
        return { tone: "good", text: String(get("llm.model") || "").trim() || "已启用" };
      }
      case "zotero": {
        const collection = String(get("output.collection") || "").trim();
        const tag = String(get("output.tag") || "").trim();
        const parts = [collection ? `分类「${collection}」` : "不放入分类", tag ? `标签 ${tag}` : null].filter(Boolean);
        return { tone: "neutral", text: parts.join(" · ") };
      }
      case "data":
        return { tone: "neutral", text: ctx.version ? `ZotWatch v${ctx.version}` : "数据目录与缓存" };
      default:
        return { tone: "neutral", text: "" };
    }
  }

  // ------------------------------------------------------------------ state

  const state = {
    section: null,
    index: null,
    auto: null,
    view: null,
    disposed: false,
    bindings: new Map(),
    pending: new Map(),
    conditional: [],
    advancedGroups: [],
    toggleCards: [],
    sectionNodes: new Map(),
    headings: new Map(),
    indexStatus: new Map(),
    testResults: [],
    viewTimer: null,
    // dirty: edited since it was loaded or saved (only then is the field written back)
    whitelist: {
      node: null,
      dirty: false,
      timer: null,
      buttons: [],
      fill: null,
      message: null,
      summary: null,
      invalid: null,
      invalidText: null,
      error: null,
      errorText: null,
      busy: false,
      // The review panel of 「从文库生成」 ({rows, button} while open)
      panel: null,
      review: null,
    },
    // Search boxes (their buttons follow the lists) and the missing-key notices under them
    searchBoxes: [],
    keyHints: [],
    // 顶刊 chips: journal names by ISSN ({title, issns} or null: unknown to OpenAlex)
    flagshipList: null,
    flagshipNames: new Map(),
    flagshipFailed: new Set(),
    flagshipLookup: null,
    flagshipError: null,
    // OpenAlex's latest known daily budget (学术数据接口)
    usage: null,
    followedPreview: null,
    presetChips: [],
    keyHelp: null,
    localCallout: null,
    schedule: null,
    data: {},
  };

  const getPref = (pref) => {
    try {
      return Zotero.Prefs.get(PREF_PREFIX + pref, true);
    } catch (e) {
      return undefined;
    }
  };

  /** Pref value as shown now: a control's unsaved text wins over the stored value. */
  const currentValue = (pref) => {
    const b = state.bindings.get(pref);
    return b && state.pending.has(pref) ? readControl(b) : getPref(pref);
  };

  const controlId = (pref) => `sp-${pref.replace(/\./g, "-")}`;
  const fallbackOf = (spec) => (spec.auto ? state.auto?.[spec.auto] ?? null : spec.fallback ?? null);

  function loadAutoValues() {
    try {
      return api()?.getAutoValues?.() || null;
    } catch (e) {
      Zotero.logError(e);
      return null;
    }
  }

  // --------------------------------------------------------------- binding

  function bind(spec, parts) {
    const binding = { spec, ...parts };
    state.bindings.set(spec.pref, binding);
    if (spec.when) state.conditional.push(binding);
    return binding;
  }

  function readControl(b) {
    switch (b.spec.kind) {
      case "switch":
        return !!b.control.checked;
      case "radio":
        return b.inputs.find((input) => input.checked)?.value ?? null;
      case "seg":
        return b.value ?? null;
      case "textarea":
        return b.control.value;
      default:
        return String(b.control.value || "").trim();
    }
  }

  function fillControl(b, value) {
    const { spec } = b;
    switch (spec.kind) {
      case "switch":
        b.control.checked = value === true;
        break;
      case "radio":
        for (const input of b.inputs) input.checked = input.value === value;
        break;
      case "seg":
        selectSeg(b, value);
        break;
      case "textarea":
        // Blank flagship texts mean "the default": show it, so it can be read and edited
        b.control.value = spec.restore && !String(value || "").trim() ? defaultText(spec, state.auto) : value ?? "";
        break;
      default:
        b.control.value = value ?? "";
    }
  }

  function selectSeg(b, value) {
    b.value = value;
    for (const button of b.buttons) {
      const on = button.getAttribute("data-value") === value;
      button.classList.toggle("active", on);
      button.setAttribute("aria-pressed", on ? "true" : "false");
    }
  }

  /** Write a binding's control value to its pref (when it changed); true when written. */
  function commit(b) {
    clearTimeout(state.pending.get(b.spec.pref));
    state.pending.delete(b.spec.pref);
    const raw = readControl(b);
    if (raw === null) return false;
    // Long flagship texts that equal the default are stored blank, so they follow the default
    const value = b.spec.restore && isDefaultText(b.spec, raw, state.auto) ? "" : raw;
    return writePref(b.spec.pref, value);
  }

  function writePref(pref, raw) {
    const current = getPref(pref);
    const value = coercePrefValue(raw, current);
    if (value === undefined) {
      Zotero.logError(new Error(`ZotWatch: no default for extensions.zotwatch.${pref}`));
      return false;
    }
    if (value === current) return false;
    try {
      Zotero.Prefs.set(PREF_PREFIX + pref, value, true);
    } catch (e) {
      Zotero.logError(e);
      toast(`保存失败：${e.message || e}`, { tone: "error", key: "settings-saved" });
      return false;
    }
    if (pref === "embedding.provider") state.auto = loadAutoValues();
    toast("已保存", { key: "settings-saved", duration: 1600 });
    updateDerived();
    scheduleViewRefresh();
    return true;
  }

  function schedule(b, delay = INPUT_DELAY) {
    clearTimeout(state.pending.get(b.spec.pref));
    state.pending.set(
      b.spec.pref,
      setTimeout(() => {
        if (!state.disposed) commit(b);
      }, delay),
    );
  }

  /** The sidebar and the other pages read some of these settings: reload the view shortly after edits. */
  function scheduleViewRefresh() {
    clearTimeout(state.viewTimer);
    state.viewTimer = setTimeout(() => {
      if (!state.disposed) refresh();
    }, VIEW_REFRESH_DELAY);
  }

  /** Save every edit still waiting for its debounce (the whitelist asynchronously). */
  function flush() {
    for (const pref of [...state.pending.keys()]) {
      const b = state.bindings.get(pref);
      if (b) commit(b);
    }
    return saveWhitelist();
  }

  /** Fill every control from the prefs (the whitelist from its file). */
  function load() {
    state.auto = loadAutoValues();
    for (const b of state.bindings.values()) {
      if (!state.pending.has(b.spec.pref)) fillControl(b, getPref(b.spec.pref));
    }
    const wl = state.whitelist;
    if (wl.node && !wl.dirty && !wl.busy) wl.node.value = api()?.getJournalWhitelist?.() ?? "";
    updateDerived();
  }

  // --------------------------------------------------------- derived views

  /** Everything that follows from the current values: visibility, placeholders, notes, previews, statuses. */
  function updateDerived() {
    const get = currentValue;
    const provider = PROVIDERS[get("embedding.provider")] ? get("embedding.provider") : "local";
    for (const b of state.conditional) b.row.hidden = !b.spec.when(get);
    for (const group of state.advancedGroups) group.node.hidden = group.rows.every((row) => row.hidden);
    for (const { pref, node } of state.toggleCards) node.classList.toggle("is-off", !get(pref));

    for (const b of state.bindings.values()) {
      const { spec } = b;
      if (spec.kind === "number") {
        const fallback = fallbackOf(spec);
        const placeholder = spec.auto ? (fallback === null ? "自动" : `自动（${formatSettingValue(fallback, spec)}）`) : formatSettingValue(fallback, spec);
        b.control.setAttribute("placeholder", placeholder);
        updateNote(b);
      } else if (spec.restore) {
        b.control.setAttribute("placeholder", defaultText(spec, state.auto));
        updateRestore(b);
      }
    }
    const model = state.bindings.get("embedding.model");
    if (model) model.control.setAttribute("placeholder", PROVIDERS[provider].model || "");
    const endpoint = state.bindings.get("embedding.baseURL");
    if (endpoint) endpoint.control.setAttribute("placeholder", PROVIDERS[provider].baseURL || "");
    if (state.keyHelp) state.keyHelp.textContent = PROVIDERS[provider].keyHelp || "";
    if (state.localCallout) state.localCallout.hidden = provider !== "local";
    const baseURL = String(get("llm.baseURL") || "").trim().replace(/\/+$/, "");
    for (const { chip, preset } of state.presetChips) {
      const on = baseURL === preset.baseURL;
      chip.classList.toggle("active", on);
      chip.setAttribute("aria-pressed", on ? "true" : "false");
    }
    const keyed = !!String(get("apis.openalexKey") || "").trim();
    for (const hint of state.keyHints) hint.hidden = keyed;
    renderFollowedPreview();
    renderFlagshipJournals();
    renderWhitelistSummary();
    renderStatuses();
    renderUsage();
    refreshSearchBoxes();
  }

  function updateNote(b) {
    let text = clampNote(readControl(b), b.spec, fallbackOf(b.spec));
    if (!text && b.spec.pref === "embedding.dimensions") {
      const { value } = effectiveNumber(readControl(b), b.spec, fallbackOf(b.spec));
      if (!DASHSCOPE_DIMENSIONS.includes(value)) text = `DashScope 不支持 ${value} 维，运行会失败`;
    }
    if (!text && b.spec.pref === "flagship.grayLow") {
      // src/settings.js caps the gray-zone floor at the acceptance threshold
      const minScore = FIELD["flagship.minScore"];
      const low = effectiveNumber(readControl(b), b.spec, fallbackOf(b.spec)).value;
      const high = effectiveNumber(currentValue("flagship.minScore"), minScore, fallbackOf(minScore)).value;
      if (Number.isFinite(low) && Number.isFinite(high) && low > high) text = `高于接受阈值，将按 ${formatSettingValue(high, b.spec)} 使用`;
    }
    b.note.textContent = text;
    b.note.hidden = !text;
    b.control.classList.toggle("invalid", !!text);
  }

  function updateRestore(b) {
    const text = b.control.value;
    b.meta.textContent = restoreMeta(b.spec, text, state.auto);
    b.restoreButton.disabled = isDefaultText(b.spec, text, state.auto);
  }

  function restoreDefault(b) {
    clearTimeout(state.pending.get(b.spec.pref));
    state.pending.delete(b.spec.pref);
    writePref(b.spec.pref, "");
    fillControl(b, "");
    updateRestore(b);
    toast("已恢复默认", { key: "settings-saved", duration: 1600 });
  }

  /** The followed-authors text parsed ({authors, invalid}); empty when the API is unavailable. */
  function parsedFollowed(text = state.bindings.get("followed.authors")?.control.value ?? currentValue("followed.authors")) {
    try {
      return api()?.parseFollowedAuthors?.(String(text || "")) || { authors: [], invalid: [] };
    } catch (e) {
      Zotero.logError(e);
      return { authors: [], invalid: [] };
    }
  }

  /** An icon-only 移除 button (the row's text stays the row's content). */
  function removeButton(name, onRemove, { disabled = false, title = "移除" } = {}) {
    const button = el("button", { type: "button", class: "btn ghost small icon-only sf-remove", title, "aria-label": `移除 ${name}` }, svgIcon("x"));
    button.disabled = disabled;
    button.addEventListener("click", onRemove);
    return button;
  }

  function renderFollowedPreview() {
    const host = state.followedPreview;
    const b = state.bindings.get("followed.authors");
    if (!host || !b) return;
    const { authors, invalid } = parsedFollowed(b.control.value);
    host.head.textContent = authors.length || invalid.length
      ? [`已识别 ${authors.length} 位作者`, invalid.length ? `${invalid.length} 行无法识别` : null].filter(Boolean).join(" · ")
      : "还没有关注的作者：在上面搜索后点「关注」，或在「编辑文本」中按行填写";
    host.head.classList.toggle("warning", invalid.length > 0);
    host.list.replaceChildren(
      ...authors.map((a) =>
        el(
          "li",
          { class: "sf-author" },
          el("span", { class: "dot followed" }),
          el("span", { class: "sf-author-name", text: a.name }),
          el("span", { class: "chip mono sf-author-id" }, el("span", { class: "chip-key", text: a.orcid ? "ORCID" : "OpenAlex" }), a.orcid || a.openalexId || ""),
          removeButton(a.name, () => removeFollowedLines((line) => parsedFollowed(line).authors[0]?.key === a.key)),
        ),
      ),
      ...invalid.map((line) =>
        el(
          "li",
          { class: "sf-invalid" },
          svgIcon("alert"),
          el("span", { class: "sf-invalid-text", text: `无法识别：${line}` }),
          removeButton(line, () => removeFollowedLines((l) => l.trim() === line)),
        ),
      ),
    );
    host.list.hidden = !authors.length && !invalid.length;
  }

  /** Change the followed-authors text (search results, 移除) and save it at once. */
  function setFollowedText(text) {
    const b = state.bindings.get("followed.authors");
    if (!b) return;
    b.control.value = text;
    commit(b);
    renderFollowedPreview();
    renderStatuses();
    refreshSearchBoxes();
  }

  /** Remove the lines matching `match` (an author's lines, an unparseable line) from the followed authors. */
  function removeFollowedLines(match) {
    const b = state.bindings.get("followed.authors");
    if (!b) return;
    setFollowedText(String(b.control.value || "").split(/\r?\n/).filter((line) => !match(line)).join("\n"));
  }

  /** Whether a searched author is followed already (by ORCID or OpenAlex id). */
  function isFollowed(author) {
    return parsedFollowed().authors.some((a) => (author.orcid && a.orcid === author.orcid) || (a.openalexId && a.openalexId === author.id));
  }

  /** 「关注」: append "Name, ORCID" (or the OpenAlex id without an ORCID). */
  async function followAuthor(author) {
    const b = state.bindings.get("followed.authors");
    if (!b || isFollowed(author)) return "";
    const text = String(b.control.value || "").replace(/\s+$/, "");
    const line = `${author.name}, ${author.orcid || author.id}`;
    setFollowedText(text ? `${text}\n${line}` : line);
    return currentValue("followed.enabled") ? `已关注 ${author.name}` : `已关注 ${author.name}（「关注指定作者的新论文」尚未开启）`;
  }

  // ------------------------------------------------------ 顶刊 (flagship ISSNs)

  /** The flagship ISSNs in effect: the field's list, or the defaults when it is blank. */
  function flagshipIssns() {
    const b = state.bindings.get("flagship.issns");
    const text = b ? b.control.value : currentValue("flagship.issns");
    return issnList(String(text || "").trim() ? text : defaultText(FIELD["flagship.issns"], state.auto));
  }

  /** Change the flagship ISSNs (search results, 移除) and save them at once (the defaults are stored blank). */
  function setFlagshipIssns(issns) {
    const b = state.bindings.get("flagship.issns");
    if (!b) return;
    b.control.value = issns.join("\n");
    commit(b);
    updateRestore(b);
    renderFlagshipJournals();
    renderStatuses();
    refreshSearchBoxes();
  }

  /** Look up the names of ISSNs without one (only while the 顶刊地学 section is shown); renders again when done. */
  function lookupFlagshipNames(issns) {
    const wanted = issns.filter((issn) => !state.flagshipNames.has(issn) && !state.flagshipFailed.has(issn));
    if (!wanted.length || state.flagshipLookup || state.section !== "flagship" || !api()?.lookupJournalNames) return;
    state.flagshipLookup = (async () => {
      try {
        const { names, error } = (await api().lookupJournalNames(wanted)) || { names: {}, error: null };
        for (const issn of wanted) {
          if (Object.hasOwn(names || {}, issn)) state.flagshipNames.set(issn, names[issn]);
          else state.flagshipFailed.add(issn);
        }
        state.flagshipError = error || null;
      } catch (e) {
        if (!e?.cancelled) Zotero.logError(e);
        for (const issn of wanted) state.flagshipFailed.add(issn);
      } finally {
        state.flagshipLookup = null;
        if (!state.disposed) renderFlagshipJournals();
      }
    })();
  }

  /** The flagship journals: one row per journal (name + ISSN chips + 移除); unknown ISSNs get a row each. */
  function renderFlagshipJournals() {
    const host = state.flagshipList;
    if (!host) return;
    const issns = flagshipIssns();
    lookupFlagshipNames(issns);
    const groups = [];
    const byTitle = new Map();
    for (const issn of issns) {
      const info = state.flagshipNames.get(issn);
      if (info?.title) {
        const key = info.title.toLowerCase();
        if (!byTitle.has(key)) {
          const group = { title: info.title, issns: [], known: true };
          byTitle.set(key, group);
          groups.push(group);
        }
        byTitle.get(key).issns.push(issn);
      } else {
        const pending = !state.flagshipNames.has(issn) && !state.flagshipFailed.has(issn);
        groups.push({ title: pending ? "正在识别期刊名称…" : "未识别的期刊", issns: [issn], known: false });
      }
    }
    const last = groups.length <= 1;
    host.list.replaceChildren(
      ...groups.map((g) =>
        el(
          "li",
          { class: "fj-row" },
          el("span", { class: "dot flagship" }),
          el("span", { class: `fj-name${g.known ? "" : " unknown"}`, text: g.title, title: g.title }),
          el("span", { class: "fj-chips" }, ...g.issns.map((issn) => el("span", { class: "fj-chip", text: issn }))),
          removeButton(g.known ? g.title : g.issns[0], () => setFlagshipIssns(flagshipIssns().filter((issn) => !g.issns.includes(issn))), {
            disabled: last,
            title: last ? "至少保留一种期刊；不需要时请关闭上面的开关" : "移除",
          }),
        ),
      ),
    );
    const b = state.bindings.get("flagship.issns");
    const usesDefault = b ? isDefaultText(b.spec, b.control.value, state.auto) : true;
    host.meta.textContent = `${groups.length} 种期刊 · ${issns.length} 个 ISSN · ${usesDefault ? "默认" : "已自定义"}`;
    host.error.hidden = !state.flagshipError;
    host.error.textContent = state.flagshipError ? `部分期刊名称未能识别：${state.flagshipError}` : "";
  }

  /** 「添加」 of the 顶刊 search: append the journal's ISSNs. */
  async function addFlagship(journal) {
    const current = flagshipIssns();
    const added = journal.issns.filter((issn) => !current.includes(issn));
    if (!added.length) return "";
    for (const issn of journal.issns) state.flagshipNames.set(issn, { title: journal.title, issns: journal.issns });
    setFlagshipIssns([...current, ...added]);
    return `已添加 ${journal.title}（${added.length} 个 ISSN）`;
  }

  function refreshSearchBoxes() {
    for (const box of state.searchBoxes) box.refresh();
  }

  function renderWhitelistSummary() {
    const wl = state.whitelist;
    if (!wl.node) return;
    let summary = { count: 0, withImpactFactor: 0, invalidLines: [] };
    try {
      summary = api()?.summarizeWhitelist?.(wl.node.value) || summary;
    } catch (e) {
      Zotero.logError(e);
    }
    const auto = !!currentValue("journals.autoFromLibrary");
    wl.summary.replaceChildren(
      el("span", { class: `dot ${summary.count ? "good" : auto ? "off" : "warning"}` }),
      el("span", {
        text: summary.count ? `已识别 ${summary.count} 个期刊，${summary.withImpactFactor} 个有影响因子` : auto ? "列表为空，只关注文库中的常见期刊" : "列表为空",
      }),
    );
    const invalid = summary.invalidLines || [];
    wl.invalid.hidden = !invalid.length;
    if (invalid.length) {
      const shown = invalid.slice(0, 3).join("；");
      wl.invalidText.textContent = `${invalid.length} 行无法识别（第一列须为 ISSN）：${shown}${invalid.length > 3 ? " 等" : ""}`;
    }
    const error = api()?.whitelistError || null;
    wl.error.hidden = !error;
    wl.errorText.textContent = error || "";
    if (wl.fill && !wl.busy) {
      const incomplete = summary.incomplete || 0;
      wl.fill.disabled = !incomplete;
      wl.fill.setAttribute("title", incomplete ? `${incomplete} 个期刊缺少学科、影响因子或刊名，从 OpenAlex 补全` : "所有期刊都已有学科和影响因子");
    }
  }

  function statusContext() {
    const text = state.whitelist.node ? state.whitelist.node.value : api()?.getJournalWhitelist?.() ?? "";
    const ctx = { auto: state.auto, version: api()?.version || state.view?.version || null, whitelistError: api()?.whitelistError || null };
    try {
      ctx.whitelist = api()?.summarizeWhitelist?.(text) || null;
      ctx.followed = api()?.parseFollowedAuthors?.(String(currentValue("followed.authors") || "")) || null;
    } catch (e) {
      Zotero.logError(e);
    }
    return ctx;
  }

  function renderStatuses() {
    if (!state.indexStatus.size) return;
    const ctx = statusContext();
    for (const [id, node] of state.indexStatus) {
      const { tone, text } = sectionStatus(id, currentValue, ctx);
      node.replaceChildren(el("span", { class: `dot ${tone}` }), el("span", { class: "sf-index-status-text", text }));
      node.setAttribute("title", text);
    }
  }

  /** 「今日额度」 of the 学术数据接口 section: OpenAlex's latest known daily budget. */
  function renderUsage() {
    const node = state.usage;
    if (!node) return;
    let status = null;
    try {
      status = api()?.getApiStatus?.() || null;
    } catch (e) {
      Zotero.logError(e);
    }
    const keyed = !!String(currentValue("apis.openalexKey") || "").trim();
    const usage = status?.openalexUsage || null;
    let tone = "off";
    let text = "运行一次或点击「测试接口」后显示";
    if (usage?.text) {
      tone = keyed ? "good" : "warning";
      const when = Number.isFinite(usage.at) ? formatRelative(usage.at) : "";
      text = when ? `${usage.text} · ${when}更新` : usage.text;
    } else if (!keyed) {
      tone = "warning";
      text = "未配置 Key：每日只有很少的免费额度";
    }
    node.replaceChildren(el("span", { class: `dot ${tone}` }), el("span", { class: "sf-usage-text", text }));
    node.setAttribute("title", text);
  }

  function renderSchedule() {
    const node = state.schedule;
    if (!node || !state.view) return;
    const status = runStatus(state.view);
    node.replaceChildren(el("span", { class: `dot ${status.tone}` }), el("span", { text: [status.label, status.next].filter(Boolean).join(" · ") }));
  }

  function renderData() {
    const { data } = state;
    if (!data.path) return;
    const dir = api()?.getDataDir?.() || null;
    data.path.textContent = dir || "（未知）";
    data.path.setAttribute("title", dir || "");
    data.reveal.disabled = !dir;
    const version = api()?.version || state.view?.version || "";
    data.version.textContent = version ? `v${version}` : "–";
    data.readme.setAttribute("href", README_URL(version || "0.4.0"));
    data.reset.disabled = !!state.view?.running;
    data.reset.setAttribute("title", state.view?.running ? "正在检查，请稍后再试" : "");
  }

  // -------------------------------------------------------------- controls

  function fieldRow(spec, control, { wide = false, extraHelp = [] } = {}) {
    const help = [].concat(spec.help || []);
    const text = el(
      "div",
      { class: "field-text" },
      el("label", { class: "field-label", for: controlId(spec.pref), text: spec.label }),
      ...help.map((line) => el("p", { class: "field-help", text: line })),
      ...extraHelp,
      spec.kind === "number" ? el("p", { class: "field-range", text: `范围 ${rangeText(spec)}${spec.auto ? "，留空自动" : ""}` }) : null,
    );
    // Narrow controls (switches, numbers, segments) leave the help text the rest of the row
    const compact = ["switch", "number", "seg"].includes(spec.kind);
    const classes = ["field", wide ? "wide" : null, spec.long ? "long" : null, compact ? "compact" : null, spec.kind === "switch" ? "toggle" : null].filter(Boolean).join(" ");
    return el("div", { class: classes, "data-field": spec.pref }, text, el("div", { class: "field-control" }, control));
  }

  function switchControl(spec) {
    const input = el("input", { type: "checkbox", role: "switch", class: "switch", id: controlId(spec.pref), "data-pref": spec.pref });
    const b = bind(spec, { control: input });
    input.addEventListener("change", () => commit(b));
    return { node: input, binding: b };
  }

  function numberControl(spec) {
    const input = el("input", {
      type: "text",
      inputmode: spec.int ? "numeric" : "decimal",
      class: "input sf-num",
      id: controlId(spec.pref),
      "data-pref": spec.pref,
      autocomplete: "off",
      spellcheck: "false",
    });
    const note = el("p", { class: "field-note", hidden: true });
    const b = bind(spec, { control: input, note });
    input.addEventListener("input", () => {
      updateNote(b);
      if (spec.pref === "flagship.minScore") {
        const low = state.bindings.get("flagship.grayLow");
        if (low) updateNote(low);
      }
      schedule(b);
    });
    input.addEventListener("change", () => commit(b));
    const node = el("div", { class: "sf-stack sf-stack-end" }, el("span", { class: "input-unit" }, input, el("span", { class: "sf-unit", text: spec.unit || "", "aria-hidden": spec.unit ? null : "true" })), note);
    return { node, binding: b };
  }

  function textControl(spec) {
    const type = spec.kind === "password" ? "password" : spec.inputType || "text";
    const input = el("input", {
      type,
      class: `input${spec.kind === "password" ? " mono" : ""}`,
      id: controlId(spec.pref),
      "data-pref": spec.pref,
      placeholder: spec.placeholder || null,
      autocomplete: "off",
      spellcheck: "false",
    });
    const b = bind(spec, { control: input });
    input.addEventListener("input", () => schedule(b));
    input.addEventListener("change", () => commit(b));
    if (spec.kind !== "password") return { node: input, binding: b };

    const eye = el("button", { type: "button", class: "sf-eye", title: "显示", "aria-label": `显示 ${spec.label}`, "aria-pressed": "false" }, svgIcon("eye"));
    eye.addEventListener("click", () => {
      const show = input.type === "password";
      input.type = show ? "text" : "password";
      eye.replaceChildren(svgIcon(show ? "eye-off" : "eye"));
      eye.setAttribute("title", show ? "隐藏" : "显示");
      eye.setAttribute("aria-label", `${show ? "隐藏" : "显示"} ${spec.label}`);
      eye.setAttribute("aria-pressed", show ? "true" : "false");
    });
    return { node: el("div", { class: "sf-secret" }, input, eye), binding: b };
  }

  function segControl(spec) {
    const buttons = spec.options.map((o) => el("button", { type: "button", "data-value": o.value, "aria-pressed": "false" }, o.title));
    const group = el("div", { class: "seg", role: "group", id: controlId(spec.pref), "data-pref": spec.pref, "aria-label": spec.label }, ...buttons);
    const b = bind(spec, { control: group, buttons, value: null });
    for (const button of buttons) {
      button.addEventListener("click", () => {
        selectSeg(b, button.getAttribute("data-value"));
        commit(b);
      });
    }
    return { node: group, binding: b };
  }

  function radioControl(spec) {
    const name = controlId(spec.pref);
    const inputs = [];
    const cards = spec.options.map((o) => {
      const input = el("input", { type: "radio", name, value: o.value, "data-pref": spec.pref });
      inputs.push(input);
      return el(
        "label",
        { class: "radio-card" },
        input,
        el(
          "span",
          { class: "radio-card-body" },
          el("span", { class: "radio-card-title" }, o.title, o.recommended ? el("span", { class: "pill sf-pill-accent", text: "推荐" }) : null),
          el("span", { class: "radio-card-text", text: o.text }),
        ),
      );
    });
    const group = el("div", { class: "radio-cards sf-radio-cards", role: "radiogroup", id: name, "aria-label": spec.label }, ...cards);
    const b = bind(spec, { control: group, inputs });
    for (const input of inputs) input.addEventListener("change", () => commit(b));
    return { node: group, binding: b };
  }

  function textareaControl(spec, extra = []) {
    const area = el("textarea", {
      class: `textarea${spec.mono ? " mono" : ""}`,
      id: controlId(spec.pref),
      "data-pref": spec.pref,
      rows: String(spec.rows || 3),
      spellcheck: "false",
      placeholder: spec.placeholder || null,
    });
    const b = bind(spec, { control: area });
    const parts = [area];
    if (spec.restore) {
      b.meta = el("span", { class: "sf-meta" });
      b.restoreButton = el("button", { type: "button", class: "btn ghost small" }, svgIcon("undo"), "恢复默认");
      b.restoreButton.addEventListener("click", () => restoreDefault(b));
      parts.push(el("div", { class: "sf-textarea-foot" }, b.meta, b.restoreButton));
    }
    parts.push(...extra);
    area.addEventListener("input", () => {
      if (spec.restore) updateRestore(b);
      if (spec.pref === "followed.authors") {
        renderFollowedPreview();
        renderStatuses();
        refreshSearchBoxes();
      }
      if (spec.pref === "flagship.issns") {
        renderFlagshipJournals();
        refreshSearchBoxes();
      }
      schedule(b);
    });
    area.addEventListener("change", () => commit(b));
    return { node: el("div", { class: "sf-stack" }, ...parts), binding: b };
  }

  /**
   * A field row for a pref (the control kind comes from FIELDS); `extraHelp`
   * nodes go under its help text, `extra` nodes under a textarea.
   */
  function row(pref, { extraHelp = [], extra = [] } = {}) {
    const spec = FIELD[pref];
    if (!spec) throw new Error(`Unknown setting ${pref}`);
    let built;
    let rowNode;
    switch (spec.kind) {
      case "switch":
        built = switchControl(spec);
        rowNode = fieldRow(spec, built.node, { extraHelp });
        break;
      case "number":
        built = numberControl(spec);
        rowNode = fieldRow(spec, built.node, { extraHelp });
        break;
      case "text":
      case "password":
        built = textControl(spec);
        rowNode = fieldRow(spec, built.node, { extraHelp });
        break;
      case "seg":
        built = segControl(spec);
        rowNode = fieldRow(spec, built.node, { extraHelp });
        break;
      case "textarea":
        built = textareaControl(spec, extra);
        rowNode = fieldRow(spec, built.node, { wide: true, extraHelp });
        break;
      case "radio":
        built = radioControl(spec);
        rowNode = el("div", { class: "field wide sf-radio-row", "data-field": spec.pref }, built.node);
        break;
      default:
        throw new Error(`Unknown kind ${spec.kind}`);
    }
    built.binding.row = rowNode;
    return rowNode;
  }

  /**
   * A settings card: a title (or the label of a `toggle` switch, shown in the
   * header) with an optional caption (a string or lines) and header `actions`,
   * then its rows. Cards with a toggle dim their rows while it is off (the
   * rows stay editable).
   */
  function card({ title = null, help = null, toggle = null, actions = null, className = "" }, ...body) {
    let titleNode;
    let caption = help;
    let toggleNode = null;
    if (toggle) {
      const spec = FIELD[toggle];
      const built = switchControl(spec);
      toggleNode = built.node;
      titleNode = el("label", { class: "card-title", for: controlId(toggle), text: spec.label });
      caption = help ?? spec.help;
    } else {
      titleNode = el("h3", { class: "card-title", text: title });
    }
    const items = body.flat().filter(Boolean);
    const header = el(
      "div",
      { class: `card-header sf-card-header${items.length ? "" : " solo"}` },
      el("div", { class: "sf-card-heading" }, titleNode, ...[].concat(caption || []).map((line) => el("p", { class: "sf-card-help", text: line }))),
      toggleNode,
      actions,
    );
    const node = el("section", { class: `card sf-card${className ? ` ${className}` : ""}` }, header, items.length ? el("div", { class: "card-body sf-card-body" }, ...items) : null);
    if (toggle) state.toggleCards.push({ pref: toggle, node });
    return node;
  }

  /** Collapsible group of advanced fields inside a card (hidden while none of them applies). */
  function advanced(...rows) {
    const node = el(
      "details",
      { class: "sf-advanced" },
      el("summary", { class: "sf-advanced-summary" }, svgIcon("chevron-right", "icon sf-chevron"), "高级选项"),
      el("div", { class: "sf-advanced-body" }, ...rows),
    );
    state.advancedGroups.push({ node, rows });
    return node;
  }

  /** A web link under a field's help text (opened in the browser by main.js). */
  const helpLink = (href, text) => el("p", { class: "field-help" }, el("a", { class: "sf-link", href }, text, svgIcon("external")));

  /** A paragraph of explanation inside a card. */
  const explainer = (text) => el("p", { class: "sf-explainer", text });

  /** A read-only row: label on the left, a value node on the right. */
  const infoRow = (label, value) =>
    el("div", { class: "field sf-info" }, el("div", { class: "field-text" }, el("span", { class: "field-label", text: label })), el("div", { class: "field-control" }, value));

  function callout(tone, icon, text, ...extra) {
    return el("div", { class: `callout ${tone} sf-callout` }, svgIcon(icon), el("div", { class: "callout-body" }, el("p", { class: "callout-text", text }), ...extra));
  }

  /** 「测试…」 button with the matching rows of checkConnections() shown next to it. */
  function testRow(label, keys, emptyText) {
    const result = el("div", { class: "sf-test-result", role: "status" });
    const button = el("button", { type: "button", class: "btn small" }, svgIcon("plug"), label);
    button.addEventListener("click", () => runCheck(keys, button, result, emptyText));
    state.testResults.push(result);
    return el("div", { class: "sf-test" }, button, result);
  }

  async function runCheck(keys, button, out, emptyText) {
    await flush();
    button.disabled = true;
    out.replaceChildren(el("span", { class: "sf-result running" }, el("span", { class: "dot running" }), el("span", { text: "正在测试…" })));
    try {
      const rows = (await api().checkConnections(keys)) || [];
      if (state.disposed) return;
      out.replaceChildren(...(rows.length ? rows.map(resultLine) : [el("span", { class: "sf-result muted", text: emptyText || "没有可测试的项目" })]));
      // The OpenAlex check reads today's budget
      renderUsage();
    } catch (e) {
      if (!state.disposed) out.replaceChildren(resultLine({ ok: false, name: "", detail: String(e?.message || e) }));
    } finally {
      if (!state.disposed) button.disabled = false;
    }
  }

  /** One checkConnections() row: status icon + text (never colour alone). */
  function resultLine(row) {
    // OpenAlex without a key answers, but with a tiny budget: a warning, not a failure
    const keyless = row.ok === null && row.key === "openalex";
    const tone = row.ok === true ? "good" : row.ok === false ? "critical" : keyless ? "warning" : "off";
    const icon = row.ok === true ? "check-circle" : row.ok === false || keyless ? "alert" : "info";
    const detail = row.detail || (row.ok === true ? "正常" : row.ok === false ? "失败" : "未启用");
    // Services tested together (publisher keys, the scholarly APIs) name themselves
    let text = row.ok === false ? `失败：${detail}` : detail;
    if (["elsevier", "springer", "openalex", "semanticscholar", "crossref"].includes(row.key) && row.name) text = `${row.name}：${detail}`;
    else if (row.key === "llm" && row.ok === true) text = `正常，模型回复：${detail}`;
    return el("span", { class: `sf-result ${tone}`, title: text }, svgIcon(icon), el("span", { class: "sf-result-text", text }));
  }

  // ------------------------------------------------------- journals (file)

  /** The status line under the journal buttons; a running OpenAlex action offers 「取消」. */
  function setJournalsMessage(text, tone = "info", { cancel = false } = {}) {
    const node = state.whitelist.message;
    node.hidden = !text;
    node.className = `sf-action-message ${tone}`;
    let stop = null;
    if (cancel) {
      stop = el("button", { type: "button", class: "link-btn sf-cancel", id: "sp-journals-cancel" }, "取消");
      stop.addEventListener("click", () => {
        stop.disabled = true;
        api()?.cancelJournalAction?.();
      });
    }
    // replaceChildren() would insert a null as the text "null"
    node.replaceChildren(...[svgIcon(tone === "error" ? "alert" : tone === "running" ? "clock" : "check-circle"), el("span", { text: text || "" }), stop].filter(Boolean));
  }

  async function saveWhitelist() {
    const wl = state.whitelist;
    clearTimeout(wl.timer);
    wl.timer = null;
    if (!wl.node || !wl.dirty || wl.busy || !api()) return;
    wl.dirty = false;
    const text = wl.node.value;
    if (text === api().getJournalWhitelist() && !api().whitelistError) return;
    try {
      await api().setJournalWhitelist(text);
      if (state.disposed) return;
      toast("已保存", { key: "settings-saved", duration: 1600 });
      renderWhitelistSummary();
      renderStatuses();
    } catch (e) {
      Zotero.logError(e);
      wl.dirty = true;
      if (!state.disposed) toast(`期刊列表保存失败：${e.message || e}`, { tone: "error", key: "settings-saved" });
    }
  }

  function lockJournals(on) {
    const wl = state.whitelist;
    wl.node.disabled = on;
    for (const button of wl.buttons) button.disabled = on;
  }

  /**
   * Run a task with the list and its buttons locked. A result with `text`
   * replaces the list (and is saved); its `message` is shown. `cancel` offers
   * 「取消」 (api().cancelJournalAction()) while it runs.
   */
  async function journalAction(pending, task, { cancel = false } = {}) {
    const wl = state.whitelist;
    if (wl.busy) return;
    await saveWhitelist();
    wl.busy = true;
    lockJournals(true);
    setJournalsMessage(pending, "running", { cancel });
    try {
      const result = await task();
      if (state.disposed) return;
      if (result && typeof result.text === "string" && result.text !== wl.node.value) {
        wl.node.value = result.text;
        // Until it is saved: a failed save is retried on the next blur, flush or section switch
        wl.dirty = true;
        await api().setJournalWhitelist(result.text);
        wl.dirty = false;
        toast("已保存", { key: "settings-saved", duration: 1600 });
      }
      setJournalsMessage(result?.message || "", result?.tone || "info");
    } catch (e) {
      if (state.disposed) return;
      if (e?.cancelled) {
        setJournalsMessage("已取消", "info");
      } else {
        Zotero.logError(e);
        setJournalsMessage(`出错：${e.message || e}`, "error");
      }
    } finally {
      wl.busy = false;
      if (!state.disposed) {
        lockJournals(false);
        renderWhitelistSummary();
        renderStatuses();
        refreshSearchBoxes();
      }
    }
  }

  /** ISSNs listed in whitelist CSV text (first column), for 「已在列表」. */
  function listedIssns(text) {
    const out = new Set();
    for (const raw of String(text || "").split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const issn = normalizeIssn(line.split(",")[0].replace(/"/g, ""));
      if (issn) out.add(issn);
    }
    return out;
  }

  const journalListed = (journal) => {
    const listed = listedIssns(state.whitelist.node?.value);
    return (journal.issns || []).some((issn) => listed.has(issn));
  };

  /**
   * Merge journals into the list (one row per ISSN; listed rows only get
   * blanks filled) and save it. Returns {added, filled} or null when the list is busy.
   */
  async function addJournals(journals) {
    const wl = state.whitelist;
    if (!wl.node || wl.busy) return null;
    const { text, added, filled } = api().mergeJournals(wl.node.value, journals);
    if (text !== wl.node.value) {
      wl.node.value = text;
      wl.dirty = true;
      await saveWhitelist();
    }
    renderWhitelistSummary();
    renderStatuses();
    refreshSearchBoxes();
    return { added, filled };
  }

  /** "近似 IF 5.0" from OpenAlex's two-year mean citedness. */
  const impactText = (value) => (Number.isFinite(value) ? `近似 IF ${value.toFixed(1)}` : null);

  /** A journal search result: name, then publisher · country · ISSNs · approximate IF · works. */
  function journalResultText(j) {
    return searchResultText(j.title, [
      [j.publisher, j.country].filter(Boolean).join(" · ") || null,
      (j.issns || []).join(" / ") || "无 ISSN",
      impactText(j.impactFactor),
      Number.isFinite(j.worksCount) ? `发文 ${searchCount(j.worksCount)} 篇` : null,
    ]);
  }

  /** The no-key notice under a search box (hidden once an OpenAlex key is set). */
  function keyHint() {
    const node = el(
      "p",
      { class: "sb-hint" },
      svgIcon("alert"),
      el("span", {}, "未配置 OpenAlex API Key：仍可搜索，但每天的免费额度很小。", el("a", { class: "sf-link", href: "#settings/apis" }, "去填写")),
    );
    node.hidden = !!String(currentValue("apis.openalexKey") || "").trim();
    state.keyHints.push(node);
    return node;
  }

  /** A labelled search box block (label above, results below). */
  function searchBlock(labelText, box, ...after) {
    state.searchBoxes.push(box);
    return el("div", { class: "sf-block" }, el("div", { class: "sf-stack" }, el("label", { class: "sb-label", for: box.input.getAttribute("id"), text: labelText }), box.node, ...after));
  }

  function journalSearch() {
    const box = createSearchBox({
      id: "sp-journal-search",
      label: "搜索期刊",
      placeholder: "刊名或 ISSN，如 Chemical Geology、0009-2541",
      search: (query) => api().searchJournals(query),
      render: journalResultText,
      action: (j) => (!j.issns?.length ? { label: "无 ISSN", disabled: true } : journalListed(j) ? { label: "已在列表", disabled: true } : { label: "添加" }),
      pick: async (j) => {
        const result = await addJournals([j]);
        return result ? `已添加 ${j.title}（${result.added} 行 ISSN${Number.isFinite(j.impactFactor) ? `，近似 IF ${j.impactFactor.toFixed(1)}` : ""}）` : "期刊列表正在更新，请稍后再试";
      },
      emptyText: "OpenAlex 中没有找到匹配的期刊",
      hint: keyHint(),
    });
    return searchBlock("添加期刊", box);
  }

  // --------------------------------------------------- 从文库生成: review panel

  /** The subject of a proposal: OpenAlex's subfield name (or the category) and a 中文核心 mark. */
  function subjectCell(p) {
    const chinese = /\(CN\)|_CN$/i.test(p.category || "");
    const name = p.subfield || String(p.category || "").replace(/\s*\(CN\)\s*$/i, "").replace(/_CN$/i, "").replace(/_/g, " ");
    return el(
      "span",
      { class: `jr-subject${name ? "" : " jr-subject-empty"}`, title: p.category || "" },
      name || "–",
      chinese ? el("span", { class: "pill jr-cn", text: "中文核心" }) : null,
    );
  }

  function closeReview() {
    const wl = state.whitelist;
    wl.review = null;
    if (!wl.panel) return;
    wl.panel.hidden = true;
    wl.panel.replaceChildren();
  }

  function updateReviewCount() {
    const review = state.whitelist.review;
    if (!review) return;
    const n = review.rows.filter((r) => r.check.checked && !r.check.disabled).length;
    review.button.textContent = `加入所选（${n}）`;
    review.button.disabled = !n;
  }

  /** Show the proposals of 「从文库生成」 for review: checkboxes, then 「加入所选」 / 「取消」. */
  function openReview(result) {
    const wl = state.whitelist;
    const listed = listedIssns(wl.node.value);
    const proposals = (result.proposals || []).map((p) => ({ ...p, inList: !!p.inList || (p.issns || []).some((issn) => listed.has(issn)) }));
    const rows = proposals.map((p) => {
      const check = el("input", { type: "checkbox", class: "jr-check", "aria-label": `选择 ${p.title}` });
      check.checked = !p.inList && !!p.checked;
      check.disabled = p.inList;
      check.addEventListener("change", updateReviewCount);
      const tr = el(
        "tr",
        { class: `jr-row${p.inList ? " is-listed" : ""}` },
        el("td", {}, check),
        el(
          "td",
          {},
          el("span", { class: "jr-name", text: p.title }),
          p.libraryTitle ? el("span", { class: "jr-sub", text: `文库中写作：${p.libraryTitle}` }) : null,
          !p.verified ? el("span", { class: "jr-sub", text: "OpenAlex 中未找到，学科与影响因子待补全" }) : null,
          p.inList ? el("span", { class: "pill jr-listed", text: "已在列表" }) : null,
        ),
        el("td", {}, el("span", { class: "jr-issns" }, ...(p.issns || []).map((issn) => el("span", { text: issn })))),
        el("td", {}, subjectCell(p)),
        el("td", { class: "num" }, Number.isFinite(p.impactFactor) ? p.impactFactor.toFixed(1) : "–"),
        el("td", { class: "num" }, p.origin === "ai" ? el("span", { class: "pill jr-ai", text: "AI 推荐" }) : `${p.libraryCount} 篇`),
      );
      return { proposal: p, check, tr };
    });
    const library = rows.filter((r) => r.proposal.origin !== "ai");
    const ai = rows.filter((r) => r.proposal.origin === "ai");
    const group = (text) => el("tr", { class: "jr-group" }, el("td", { colspan: "6", text }));
    const setAll = (on) => {
      for (const r of rows) if (!r.check.disabled) r.check.checked = on;
      updateReviewCount();
    };
    const selectAll = el("button", { type: "button", class: "btn ghost small" }, "全选");
    selectAll.addEventListener("click", () => setAll(true));
    const selectNone = el("button", { type: "button", class: "btn ghost small" }, "全不选");
    selectNone.addEventListener("click", () => setAll(false));
    const button = el("button", { type: "button", class: "btn primary small", id: "sp-journals-add" });
    button.addEventListener("click", () => addReviewed());
    const cancel = el("button", { type: "button", class: "btn ghost small", id: "sp-journals-review-cancel" }, "取消");
    cancel.addEventListener("click", () => {
      closeReview();
      setJournalsMessage("");
    });
    const listedCount = proposals.filter((p) => p.inList).length;
    const counts = [`文库期刊 ${library.length} 种`, ai.length ? `AI 推荐 ${ai.length} 种` : null, listedCount ? `${listedCount} 种已在列表` : null].filter(Boolean).join(" · ");
    const missing = result.missing || [];
    const llmOn = !!currentValue("llm.enabled") && !!String(currentValue("llm.apiKey") || "").trim();
    const notes = [
      "近似 IF 为 OpenAlex 的两年均被引（与 JCR 影响因子相近但不相同）；学科取自 OpenAlex 的主题分类。",
      missing.length ? `另有 ${missing.length} 种期刊没有找到 ISSN，无法加入：${missing.slice(0, 3).join("、")}${missing.length > 3 ? " 等" : ""}。` : null,
      !ai.length && !llmOn ? "在「AI 助手」中启用大模型后，还会推荐文库中尚未出现的相关期刊（经 OpenAlex 核实）。" : null,
    ].filter(Boolean);
    wl.review = { rows, button };
    wl.panel.replaceChildren(
      el(
        "div",
        { class: "jr-head" },
        el("div", { class: "jr-heading" }, el("span", { class: "jr-title", text: "期刊建议" }), el("span", { class: "jr-counts", text: counts })),
        el("div", { class: "jr-head-actions" }, selectAll, selectNone),
      ),
      ...(result.warnings || []).map((w) => el("p", { class: "jr-warning" }, svgIcon("alert"), el("span", { text: w }))),
      el(
        "div",
        { class: "jr-table-wrap" },
        el(
          "table",
          { class: "jr-table" },
          el(
            "colgroup",
            {},
            el("col", { class: "jr-col-check" }),
            el("col", { class: "jr-col-name" }),
            el("col", { class: "jr-col-issn" }),
            el("col", { class: "jr-col-subject" }),
            el("col", { class: "jr-col-if" }),
            el("col", { class: "jr-col-count" }),
          ),
          el(
            "thead",
            {},
            el(
              "tr",
              {},
              el("th", { scope: "col" }, el("span", { class: "visually-hidden", text: "选择" })),
              el("th", { scope: "col", text: "期刊" }),
              el("th", { scope: "col", text: "ISSN" }),
              el("th", { scope: "col", text: "学科" }),
              el("th", { scope: "col", class: "num", text: "近似 IF" }),
              el("th", { scope: "col", class: "num", text: "文库篇数" }),
            ),
          ),
          el(
            "tbody",
            {},
            ai.length && library.length ? group(`来自文库（${library.length}）`) : null,
            ...library.map((r) => r.tr),
            ai.length ? group(`AI 推荐 · 已通过 OpenAlex 核实（${ai.length}）`) : null,
            ...ai.map((r) => r.tr),
          ),
        ),
      ),
      ...notes.map((text) => el("p", { class: "jr-note", text })),
      el("div", { class: "jr-foot" }, button, cancel),
    );
    wl.panel.hidden = false;
    updateReviewCount();
    wl.panel.scrollIntoView?.({ block: "nearest" });
  }

  /** 「加入所选」: merge the checked proposals into the list and save it. */
  async function addReviewed() {
    const wl = state.whitelist;
    const review = wl.review;
    if (!review || wl.busy) return;
    const chosen = review.rows.filter((r) => r.check.checked && !r.check.disabled).map((r) => r.proposal);
    if (!chosen.length) return;
    review.button.disabled = true;
    const result = await addJournals(chosen);
    if (state.disposed) return;
    if (!result) {
      // The list was busy (an import or another action): nothing was added
      updateReviewCount();
      return;
    }
    closeReview();
    const filled = result.filled ? `，补全了 ${result.filled} 行的学科或影响因子` : "";
    setJournalsMessage(`已加入 ${chosen.length} 种期刊（${result.added} 行 ISSN）${filled}。`, "info");
  }

  /** 「从文库生成」: analyse the library (OpenAlex, optionally the LLM) and open the review panel. */
  function suggestJournals() {
    closeReview();
    const llmOn = !!currentValue("llm.enabled") && !!String(currentValue("llm.apiKey") || "").trim();
    return journalAction(
      llmOn ? "正在分析文库期刊，查询 OpenAlex 并请大模型推荐相关期刊…" : "正在分析文库期刊并查询 OpenAlex…",
      async () => {
        const result = await api().suggestJournals();
        if (state.disposed) return null;
        if (!result?.proposals?.length) {
          const missing = result?.missing?.length || 0;
          return { message: missing ? `文库中的 ${missing} 种期刊都没有找到 ISSN，无法生成列表。` : "文库中没有找到期刊论文。", tone: "info" };
        }
        openReview(result);
        return { message: "" };
      },
      { cancel: true },
    );
  }

  /** 「补全学科与影响因子」: fill blank categories / impact factors of the listed journals from OpenAlex. */
  function fillJournals() {
    return journalAction(
      "正在从 OpenAlex 查询学科与影响因子…",
      async () => {
        const r = await api().fillJournalMetadata(state.whitelist.node.value);
        const missing = r.notFound?.length ? `；${r.notFound.length} 个 ISSN 在 OpenAlex 中未找到：${r.notFound.slice(0, 3).join("、")}${r.notFound.length > 3 ? " 等" : ""}` : "";
        if (!r.incomplete) return { message: "所有期刊都已有学科和影响因子。" };
        return {
          text: r.text,
          message: r.filled ? `已补全 ${r.filled} 个期刊的学科或影响因子（近似 IF 为 OpenAlex 两年均被引）${missing}。` : `没有可补全的内容${missing}。`,
        };
      },
      { cancel: true },
    );
  }

  /** Replacing a non-empty list needs a confirmation. */
  function confirmReplace() {
    const count = api()?.summarizeWhitelist?.(state.whitelist.node.value)?.count || 0;
    return !count || confirmAction(`将替换当前列表中的 ${count} 个期刊，确定继续吗？`);
  }

  async function pickCsv() {
    const { FilePicker } = ChromeUtils.importESModule("chrome://zotero/content/modules/filePicker.mjs");
    const fp = new FilePicker();
    fp.init(window, "选择期刊白名单 CSV", fp.modeOpen);
    fp.appendFilter("CSV", "*.csv");
    fp.appendFilters(fp.filterAll);
    if ((await fp.show()) !== fp.returnOK) return null;
    return typeof fp.file === "string" ? fp.file : fp.file.path;
  }

  function journalButtons() {
    const suggest = el("button", { type: "button", class: "btn small", id: "sp-journals-suggest" }, svgIcon("database"), "从文库生成");
    const fill = el("button", { type: "button", class: "btn small", id: "sp-journals-fill" }, svgIcon("table"), "补全学科与影响因子");
    const importCsv = el("button", { type: "button", class: "btn small", id: "sp-journals-import" }, svgIcon("upload"), "导入 CSV…");
    const preset = el("button", { type: "button", class: "btn small", id: "sp-journals-preset" }, svgIcon("journal"), "载入地学期刊示例");
    state.whitelist.buttons = [suggest, fill, importCsv, preset];
    state.whitelist.fill = fill;
    suggest.addEventListener("click", () => suggestJournals());
    fill.addEventListener("click", () => fillJournals());
    importCsv.addEventListener("click", async () => {
      let path;
      try {
        path = await pickCsv();
      } catch (e) {
        Zotero.logError(e);
        setJournalsMessage(`出错：${e.message || e}`, "error");
        return;
      }
      if (!path || !confirmReplace()) return;
      await journalAction("正在导入…", async () => ({ text: await Zotero.File.getContentsAsync(path), message: `已导入：${path}` }));
    });
    preset.addEventListener("click", () => {
      if (!confirmReplace()) return;
      journalAction("正在载入…", async () => ({ text: await api().loadPresetJournals(), message: "已载入地学期刊示例列表（来自 AIZotWatch 项目）。" }));
    });
    return el("div", { class: "sf-actions" }, suggest, fill, importCsv, preset);
  }

  function whitelistField() {
    const wl = state.whitelist;
    wl.node = el("textarea", {
      class: "textarea mono",
      id: "sp-whitelist",
      rows: "10",
      spellcheck: "false",
      placeholder: "issn,title,category,impact_factor\n0016-7037,Geochimica et Cosmochimica Acta,GEOCHEMISTRY,5.0",
      "aria-label": "期刊白名单（CSV）",
    });
    wl.node.addEventListener("input", () => {
      wl.dirty = true;
      renderWhitelistSummary();
      clearTimeout(wl.timer);
      wl.timer = setTimeout(() => {
        if (!state.disposed) saveWhitelist();
      }, WHITELIST_DELAY);
    });
    wl.node.addEventListener("change", () => saveWhitelist());
    wl.summary = el("div", { class: "status sf-summary", role: "status" });
    wl.invalidText = el("span");
    wl.invalid = el("div", { class: "sf-inline-warning", hidden: true }, svgIcon("alert"), wl.invalidText);
    wl.errorText = el("p", { class: "callout-text" });
    wl.error = el("div", { class: "callout warning sf-callout", hidden: true }, svgIcon("alert"), el("div", { class: "callout-body" }, wl.errorText));
    wl.message = el("div", { class: "sf-action-message", role: "status", hidden: true });
    wl.panel = el("div", { class: "jr-panel", id: "sp-journals-review", role: "region", "aria-label": "期刊建议", hidden: true });
    return el("div", { class: "sf-block" }, el("div", { class: "sf-stack" }, wl.error, wl.summary, wl.node, wl.invalid, journalButtons(), wl.message, wl.panel));
  }

  /** The raw text of a list (a field row) under a collapsed 「编辑文本」. */
  function editText(fieldRow) {
    return el(
      "details",
      { class: "sf-advanced sf-edit-text" },
      el("summary", { class: "sf-advanced-summary" }, svgIcon("chevron-right", "icon sf-chevron"), "编辑文本"),
      el("div", { class: "sf-advanced-body" }, fieldRow),
    );
  }

  /** 顶刊: search to add, the journals grouped by name with 移除, the ISSN text under 「编辑文本」. */
  function flagshipJournalsBlock() {
    const box = createSearchBox({
      id: "sp-flagship-search",
      label: "搜索顶刊",
      placeholder: "刊名或 ISSN，如 Nature Geoscience",
      search: (query) => api().searchJournals(query),
      render: journalResultText,
      action: (j) => {
        const current = flagshipIssns();
        if (!j.issns?.length) return { label: "无 ISSN", disabled: true };
        return j.issns.some((issn) => current.includes(issn)) ? { label: "已添加", disabled: true } : { label: "添加" };
      },
      pick: addFlagship,
      emptyText: "OpenAlex 中没有找到匹配的期刊",
      hint: keyHint(),
    });
    const meta = el("span", { class: "sf-meta" });
    const list = el("ul", { class: "sf-rows fj-list", "aria-label": "顶刊列表" });
    const error = el("p", { class: "sb-status error", hidden: true });
    state.flagshipList = { list, meta, error };
    const head = el("div", { class: "sf-list-head" }, el("span", { class: "sf-list-title", text: "顶刊列表" }), meta);
    return [searchBlock("添加顶刊", box, head, list, error), editText(row("flagship.issns"))];
  }

  // --------------------------------------------------------------- sections

  const BUILDERS = {
    general: () => {
      state.schedule = el("span", { class: "status sf-run-status" });
      return [
        card({ toggle: "schedule.autoRun" }, row("schedule.intervalHours"), infoRow("运行状态", state.schedule)),
        card({ title: "推荐记录", help: "推荐保存在本地的收件箱中，已处理的论文不会重复推荐。" }, row("inbox.retentionDays")),
      ];
    },

    embedding: () => {
      state.keyHelp = el("p", { class: "field-help" });
      const apiKey = row("embedding.apiKey", { extraHelp: [state.keyHelp] });
      return [
        card(
          { title: "服务商", help: "向量化服务把标题和摘要转换为向量，用来计算论文与文库的语义相似度。" },
          row("embedding.provider"),
          apiKey,
          row("embedding.model"),
          row("embedding.baseURL"),
          advanced(row("embedding.dimensions")),
          testRow("测试向量化", ["embedding"]),
        ),
        callout("info", "info", "更换服务商或模型后，首次运行会重新向量化整个文库。向量缓存保存在 Zotero 数据目录的 zotwatch 文件夹中；本地模式不发送任何文献内容。"),
      ];
    },

    apis: () => {
      state.usage = el("span", { class: "status sf-usage", role: "status" });
      return [
        card(
          {
            title: "OpenAlex",
            help: [
              "开放学术图谱：关注作者、摘要补全以及期刊信息都从这里查询。",
              "自 2026 年 2 月起 OpenAlex 需要免费的 API Key：不填写时每天只有很少的免费额度，关注作者也只能获取第一页结果。",
            ],
          },
          row("apis.openalexKey", { extraHelp: [helpLink("https://openalex.org", "openalex.org")] }),
          infoRow("今日额度", state.usage),
        ),
        card(
          { title: "Semantic Scholar", help: "可选：补全其他来源找不到的摘要，并提供 TLDR（一句话总结）。入门级 Key 每秒只能请求 1 次，ZotWatch 会自动控制速率。" },
          row("apis.semanticScholarKey", { extraHelp: [helpLink("https://www.semanticscholar.org/product/api", "Semantic Scholar API 申请页")] }),
        ),
        card({ title: "Crossref", help: "Crossref 不需要 Key，填写联系邮箱即可进入更稳定的礼貌池。" }, row("sources.crossref.mailto")),
        card(
          { title: "连接测试", help: "每个服务一个 Key；ZotWatch 按服务自动限速（如 Crossref 每秒 3 次、OpenAlex 每秒 8 次），遇到限流时等待后重试。" },
          testRow("测试接口", ["openalex", "semanticscholar", "crossref"]),
          explainer("只发送 DOI、检索词和期刊 ISSN 等查询信息；Key 只保存在本机的 Zotero 设置中。"),
        ),
      ];
    },

    sources: () => [
      card({ toggle: "sources.crossref.enabled" }, row("sources.crossref.daysBack")),
      card({ toggle: "sources.arxiv.enabled" }, row("sources.arxiv.categories")),
      card({ toggle: "sources.eartharxiv.enabled" }),
      card(
        {
          title: "摘要补全",
          help: "论文缺少摘要时按 Crossref → Elsevier → Springer Nature → OpenAlex → Semantic Scholar 的顺序查找（Semantic Scholar 需在「学术数据接口」中填写 Key，并提供 TLDR 一句话总结），找到的摘要缓存 30 天，各来源都确认没有的 7 天内不再查询。",
        },
        row("abstracts.crossref"),
        row("sources.openalexAbstracts"),
        row("abstracts.elsevierKey"),
        row("abstracts.springerKey"),
        explainer("申请：dev.elsevier.com / dev.springernature.com。只发送 DOI。"),
        testRow("测试 API Key", ["elsevier", "springer"], "请先填写 Elsevier 或 Springer Nature 的 API Key"),
      ),
    ],

    journals: () => [
      card(
        {
          title: "期刊列表",
          help: [
            "CSV：issn,title,category,impact_factor（与 AIZotWatch 的 journal_whitelist.csv 相同）。影响因子参与打分（权重 20%），未列出的期刊按默认分计算。",
            "可以在 OpenAlex 中搜索添加，或「从文库生成」后挑选；学科和近似影响因子（OpenAlex 两年均被引）会自动填写。列表保存为数据目录下的 journals.csv，清除缓存时会保留。",
          ],
        },
        journalSearch(),
        whitelistField(),
      ),
      card({ toggle: "journals.autoFromLibrary" }, row("journals.autoTopN")),
    ],

    followed: () => {
      const head = el("p", { class: "sf-preview-head" });
      const list = el("ul", { class: "sf-preview-list sf-rows" });
      state.followedPreview = { head, list };
      const box = createSearchBox({
        id: "sp-author-search",
        label: "搜索作者",
        placeholder: "姓名、ORCID 或 OpenAlex 作者 ID",
        search: (query) => api().searchAuthors(query),
        nameOf: (a) => a.name,
        render: (a) =>
          searchResultText(
            a.name,
            [
              a.institution,
              Number.isFinite(a.worksCount) ? `${searchCount(a.worksCount)} 篇论文` : null,
              Number.isFinite(a.citedByCount) ? `被引 ${searchCount(a.citedByCount)}` : null,
              a.topic,
            ],
            a.orcid ? el("span", { class: "chip mono" }, el("span", { class: "chip-key", text: "ORCID" }), a.orcid) : el("span", { class: "pill", text: "无 ORCID" }),
          ),
        action: (a) => (isFollowed(a) ? { label: "已关注", disabled: true } : { label: "关注", title: a.orcid ? "按 ORCID 关注" : "没有 ORCID，按 OpenAlex 作者 ID 关注" }),
        pick: followAuthor,
        emptyText: "OpenAlex 中没有找到匹配的作者",
        hint: keyHint(),
      });
      const preview = el("div", { class: "sf-preview", "aria-live": "polite" }, head, list);
      return [
        card(
          { toggle: "followed.enabled" },
          row("followed.daysBack"),
          searchBlock("添加作者", box, preview),
          editText(row("followed.authors")),
        ),
      ];
    },

    flagship: () => {
      state.localCallout = callout(
        "warning",
        "alert",
        "当前为本地向量化模式：只比较字面词汇，阈值自动取 0.12 / 0.04。开启大模型复核时，除明显属于排除领域的文章外都交给大模型判断（请求更多）；不复核时大多数与描述用词不同的地学文章会被漏掉，建议改用 DashScope。",
        el("a", { class: "sf-callout-link", href: "#settings/embedding", text: "更换向量化服务" }),
      );
      return [
        card(
          { toggle: "flagship.enabled" },
          explainer(
            "这些期刊的文章不按文库相似度筛选，而是比较摘要与「地学领域描述」和各「排除领域」的语义相似度：离任一排除领域更近的不要，达到接受阈值的保留，介于灰区下限和接受阈值之间的交给大模型复核（未启用大模型时全部保留）。结果单独列为「顶刊地学速览」（每次取最新发表的 N 篇，窗口中按所选排序方式排列）。",
          ),
          row("flagship.maxResults"),
          row("flagship.llmFallback"),
          row("flagship.minScore"),
          row("flagship.grayLow"),
          explainer("阈值留空时按向量化服务自动取值：API 模式 0.30 / 0.22（按 DashScope text-embedding-v4 调校），本地模式 0.12 / 0.04。"),
          state.localCallout,
        ),
        card(
          { title: "期刊与领域", help: "留空或「恢复默认」时使用内置的默认值，与 Python 版 AIZotWatch 相同。" },
          flagshipJournalsBlock(),
          row("flagship.positiveAnchor"),
          row("flagship.negativeAnchors"),
          row("flagship.llmBoundary"),
        ),
      ];
    },

    ranking: () => [
      card(
        { title: "推荐", help: "打分 = 0.8 × 与文库的相似度 + 0.2 × 期刊分，按阈值分为「必读 / 推荐 / 一般」。" },
        row("scoring.topK"),
        row("scoring.recentDays"),
        row("scoring.minSimilarity"),
        row("scoring.maxPreprintRatio"),
        row("scoring.requireAbstract"),
        advanced(row("scoring.thresholdMode"), row("scoring.temporalHalflifeDays")),
      ),
      card({ title: "关键词过滤", help: "在打分之前按标题和摘要过滤候选论文。" }, row("scoring.includeKeywords"), row("scoring.excludeKeywords")),
    ],

    ai: () => {
      const chips = LLM_PRESETS.map((preset) => {
        const chip = el("button", { type: "button", class: "sf-preset", title: `填入 ${preset.title} 的接口地址和模型`, "aria-pressed": "false" }, preset.name);
        chip.addEventListener("click", () => applyPreset(preset));
        state.presetChips.push({ chip, preset });
        return chip;
      });
      return [
        card(
          { toggle: "llm.enabled" },
          el(
            "div",
            { class: "field long" },
            el("div", { class: "field-text" }, el("span", { class: "field-label", text: "快速填写" }), el("p", { class: "field-help", text: "填入常见服务商的接口地址和模型。" })),
            el("div", { class: "field-control" }, el("div", { class: "sf-presets", role: "group", "aria-label": "快速填写" }, ...chips)),
          ),
          row("llm.baseURL"),
          row("llm.apiKey"),
          row("llm.model"),
          testRow("测试大模型", ["llm"]),
        ),
        card({ title: "功能", help: "启用大模型并填写 API Key 后生效。" }, row("llm.translateTitles"), row("llm.summaryTopN"), row("llm.topicLabels")),
      ];
    },

    zotero: () => [
      card({ title: "加入文库", help: "在推荐中点「加入 Zotero」时使用。" }, row("output.collection"), row("output.tag"), row("output.useTranslator"), row("output.summaryNote")),
    ],

    data: () => {
      const data = state.data;
      data.path = el("code", { class: "sf-path" });
      data.reveal = el("button", { type: "button", class: "btn small" }, svgIcon("folder"), "在文件夹中显示");
      data.reveal.addEventListener("click", async () => {
        try {
          if (!(await api().revealDataDir())) toast("找不到数据目录", { tone: "error" });
        } catch (e) {
          Zotero.logError(e);
          toast(`无法打开：${e.message || e}`, { tone: "error" });
        }
      });
      data.reset = el("button", { type: "button", class: "btn danger small" }, svgIcon("trash"), "清除缓存与推荐记录…");
      data.reset.addEventListener("click", resetData);
      data.version = el("span", { class: "sf-value" });
      data.readme = el("a", { class: "sf-link", href: README_URL("0.4.0") }, "使用说明（README）", svgIcon("external"));
      return [
        card(
          { title: "数据目录", help: "向量缓存、推荐记录、研究方向标签、摘要缓存、主题调研记录和期刊列表都保存在这里。" },
          el("div", { class: "sf-path-row" }, data.path, data.reveal),
        ),
        card({
          title: "清除缓存与推荐记录",
          help: "删除向量缓存、聚类结果、全部推荐记录和主题调研的记录，不影响 Zotero 文库、期刊列表和已保存到 Zotero 的调研笔记；下次运行会重新向量化文库。",
          actions: data.reset,
        }),
        card(
          { title: "关于" },
          infoRow("版本", data.version),
          infoRow("文档", data.readme),
          explainer("更多高级参数（打分权重、k-NN 邻居数、抓取上限、使用的文库 ID 等）可在 Zotero 设置 → 高级 → 配置编辑器中搜索 extensions.zotwatch. 修改。"),
        ),
      ];
    },
  };

  function applyPreset(preset) {
    for (const [pref, value] of [
      ["llm.baseURL", preset.baseURL],
      ["llm.model", preset.model],
    ]) {
      const b = state.bindings.get(pref);
      clearTimeout(state.pending.get(pref));
      state.pending.delete(pref);
      if (b) fillControl(b, value);
      writePref(pref, value);
    }
    updateDerived();
  }

  async function resetData() {
    if (!confirmAction(RESET_CONFIRM)) return;
    const button = state.data.reset;
    button.disabled = true;
    try {
      await api().resetData();
      toast("已清除 ZotWatch 数据");
      refresh();
    } catch (e) {
      Zotero.logError(e);
      toast(`清除失败：${e.message || e}`, { tone: "error" });
    } finally {
      if (!state.disposed) button.disabled = !!state.view?.running;
    }
  }

  /** A section's heading; focusable from script, so a section switch can move focus to it. */
  function sectionTitle(section) {
    const node = el("h2", { class: "sf-section-title", tabindex: "-1", text: section.title });
    state.headings.set(section.id, node);
    return node;
  }

  function sectionHead(section) {
    return el(
      "header",
      { class: "sf-section-head" },
      el("a", { class: "sf-back", href: "#settings" }, svgIcon("chevron-left"), "全部设置"),
      el(
        "div",
        { class: "sf-section-title-row" },
        el("span", { class: "sf-icon-tile" }, svgIcon(section.icon)),
        el("div", { class: "sf-section-heading" }, sectionTitle(section), el("p", { class: "sf-section-desc", text: section.description })),
      ),
    );
  }

  function pager(index) {
    const link = (section, dir) =>
      section
        ? el(
            "a",
            { class: `sf-pager-link ${dir}`, href: `#settings/${section.id}` },
            dir === "prev" ? svgIcon("chevron-left") : null,
            el("span", { class: "sf-pager-text" }, el("span", { class: "sf-pager-dir", text: dir === "prev" ? "上一项" : "下一项" }), el("span", { class: "sf-pager-name", text: section.title })),
            dir === "next" ? svgIcon("chevron-right") : null,
          )
        : el("span");
    return el("nav", { class: "sf-pager", "aria-label": "设置分区" }, link(SETTINGS_SECTIONS[index - 1], "prev"), link(SETTINGS_SECTIONS[index + 1], "next"));
  }

  function buildSection(section, index) {
    const node = el(
      "section",
      { class: "sf-section", id: `settings-${section.id}`, "data-section": section.id, "aria-label": section.title, hidden: true },
      sectionHead(section),
      el("div", { class: "sf-cards" }, ...BUILDERS[section.id]()),
      pager(index),
    );
    state.sectionNodes.set(section.id, node);
    return node;
  }

  function buildIndex() {
    const items = SETTINGS_SECTIONS.map((s) => {
      const status = el("span", { class: "status sf-index-status" });
      state.indexStatus.set(s.id, status);
      return el(
        "a",
        { class: "sf-index-item", href: `#settings/${s.id}`, "data-section": s.id },
        el("span", { class: "sf-icon-tile" }, svgIcon(s.icon)),
        el("span", { class: "sf-index-body" }, el("span", { class: "sf-index-title", text: s.title }), el("span", { class: "sf-index-text", text: s.description }), status),
        svgIcon("chevron-right", "icon sf-index-chevron"),
      );
    });
    return el(
      "div",
      { class: "sf-index-view", id: "settings-index" },
      el("div", { class: "sf-index" }, ...items),
      el("p", { class: "sf-index-foot", text: "修改在下次检查时生效。更多高级参数可在 Zotero 设置 → 高级 → 配置编辑器中搜索 extensions.zotwatch. 修改。" }),
    );
  }

  // ----------------------------------------------------------- page hooks

  function init() {
    state.auto = loadAutoValues();
    const index = buildIndex();
    const sections = SETTINGS_SECTIONS.filter((s) => BUILDERS[s.id]).map((s) => buildSection(s, SETTINGS_SECTIONS.indexOf(s)));
    $("settings-root").replaceChildren(index, ...sections);
    state.index = index;
    window.addEventListener("unload", () => {
      state.disposed = true;
    });
  }

  function show({ section } = {}) {
    // Leaving a section saves what is still waiting for its debounce
    flush();
    state.section = section && state.sectionNodes.has(section) ? section : null;
    // Journal names OpenAlex could not be asked for are retried when 顶刊地学 is shown again
    if (state.section === "flagship") {
      state.flagshipFailed.clear();
      state.flagshipError = null;
    }
    state.index.hidden = !!state.section;
    for (const [id, node] of state.sectionNodes) node.hidden = id !== state.section;
    for (const result of state.testResults) result.replaceChildren();
    if (state.whitelist.message && !state.whitelist.busy) setJournalsMessage("");
    load();
    renderSchedule();
    renderData();
    const title = state.section ? SETTINGS_SECTIONS.find((s) => s.id === state.section).title : null;
    setSubtitle("settings", title ? `${title} · 所有修改都会自动保存` : "所有修改都会自动保存");
    setActiveSection(state.section);
    const pages = $("pages");
    if (pages && state.section) pages.scrollTop = 0;
  }

  function refreshView(view) {
    state.view = view;
    renderSchedule();
    renderData();
    renderStatuses();
    renderUsage();
  }

  /** Where focus goes after a switch that hid the focused control: the section's heading (null: the page title). */
  function heading() {
    return state.section ? state.headings.get(state.section) || null : null;
  }

  return {
    init,
    show,
    refresh: refreshView,
    flush,
    heading,
    // For tests: the field table and the pure helpers
    FIELDS,
    PROVIDERS,
    LLM_PRESETS,
    helpers: { coercePrefValue, effectiveNumber, clampNote, formatSettingValue, rangeText, issnList, textLines, isDefaultText, restoreMeta, sectionStatus, defaultText },
    state,
  };
})();

registerPage("settings", ZotWatchSettings);
