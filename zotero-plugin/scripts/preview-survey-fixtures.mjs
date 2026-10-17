// Topic-survey fixtures for scripts/preview.mjs (dev only): one finished
// survey with a realistic Chinese review citing [n] references, a candidate
// pool with scores and provenance, one made without an LLM (references grouped
// by cluster), a cancelled and a failed survey for the history, seed-picker
// results and the survey options (preview.mjs adds a running survey for the
// state "running"). Shapes are those of
// src/survey/ (built with its own helpers), so the 调研 page can be checked
// against real data; everything is relative to `now`.

import { SURVEY_DEPTHS, SURVEY_STAGES, surveyEstimate } from "../src/survey/presets.js";
import { NO_LLM_NOTICE, referenceOf } from "../src/survey/report.js";

const HOUR = 3600000;
const DAY = 86400000;
const ID = "mvq2k3a01x";

const person = (name) => {
  const parts = name.split(" ");
  return { given: parts.slice(0, -1).join(" "), family: parts.at(-1), name };
};

/**
 * Papers: [title, venue, year, citations, via, scores {total, sim, link, cites, recency}, theme (null: not read),
 * reading note fields or null, extra]. Order = rank.
 */
const PAPERS = [
  ["Copper isotope fractionation during magmatic sulfide saturation and porphyry ore formation", "Geochimica et Cosmochimica Acta", 2019, 142, ["seed", "search"], [0.93, 1, 1, 0.82, 0.6], 0,
    ["硫化物饱和如何改变残余熔体与流体的铜同位素组成？", "熔体包裹体与硫化物原位 LA-MC-ICP-MS 铜同位素分析", "智利 El Teniente 与秘鲁 Toquepala 斑岩矿床", "早期硫化物饱和使残余熔体 δ65Cu 升高约 0.4‰", "建立岩浆硫化物饱和与成矿流体铜同位素的定量联系", "样品集中于两个矿床，代表性有限", 5], { seed: 1, inLibrary: true, itemKey: "SEEDKEY1", authors: ["Li Wang", "Mei Chen", "Jeremy Richards"] }],
  ["Fluid exsolution controls copper isotope signatures of porphyry systems", "Earth and Planetary Science Letters", 2021, 88, ["cites:seed1", "search", "rec"], [0.88, 0.94, 1, 0.75, 0.73], 0,
    ["流体出溶过程是否造成铜同位素分馏？", "高温高压实验结合数值模拟", "含水安山质熔体–流体实验产物", "流体相相对熔体富集轻铜同位素 0.1–0.3‰", "量化了出溶阶段的分馏系数", "实验氧逸度范围较窄", 5], { authors: ["Sarah Gleeson", "Tao Liu"] }],
  ["Copper isotopes trace magmatic-hydrothermal evolution at Bingham Canyon", "Economic Geology", 2020, 61, ["ref:seed1", "search"], [0.84, 0.9, 0.5, 0.72, 0.67], 0,
    ["宾汉峡谷矿床不同蚀变带的铜同位素有何差异？", "系统采样与硫化物铜同位素分析", "宾汉峡谷矿床 120 件硫化物样品", "钾化带到青磐岩化带 δ65Cu 递增约 1‰", "首次在单一斑岩矿床刻画完整的铜同位素分带", "表生改造的影响未完全排除", 5], { authors: ["David Cooke", "Anna Smith"] }],
  ["Redox-dependent copper isotope fractionation between sulfide liquid and silicate melt", "Chemical Geology", 2018, 54, ["ref:seed1"], [0.78, 0.86, 0.5, 0.69, 0.53], 0,
    ["氧逸度如何影响硫化物熔体与硅酸盐熔体间的铜同位素分馏？", "活塞圆筒实验", "不同 fO2 条件下的实验产物", "还原条件下分馏系数增大约两倍", "提供岩浆体系氧化还原示踪的实验基础", "未覆盖高压条件", 4], { authors: ["Hans Keppler"] }],
  ["A global compilation of copper isotopes in arc magmas", "Geology", 2022, 37, ["search"], [0.74, 0.8, 0, 0.64, 0.8], 0,
    ["全球弧岩浆的铜同位素组成有何系统变化？", "全球数据汇编与统计分析", "68 个弧火山的 900 余个样品", "δ65Cu 与岩浆氧化程度和地壳厚度相关", "提供弧岩浆铜同位素的全球背景值", "数据来自不同实验室，存在系统偏差", 4], { authors: ["Cin-Ty Lee", "Ming Tang"] }],
  ["Spatial zoning of copper isotopes as an exploration vector in porphyry deposits", "Ore Geology Reviews", 2023, 21, ["cites:seed1", "search"], [0.81, 0.83, 0.5, 0.55, 0.87], 1,
    ["铜同位素空间分带能否用于斑岩矿床勘查？", "多矿床对比与地质统计", "安第斯与冈底斯 6 个斑岩矿床", "δ65Cu 高值指向矿化中心，可指示约 2 km 范围", "提出基于铜同位素的勘查矢量模型", "需要较密集的采样", 5], { authors: ["Qiang Zhang", "Yu Huang"] }],
  ["Copper isotope composition of supergene enrichment blankets", "Mineralium Deposita", 2016, 47, ["ref:seed1"], [0.69, 0.71, 0.5, 0.66, 0.4], 1,
    ["表生富集带的铜同位素特征如何？", "剖面采样与同位素分析", "智利北部 3 个表生富集带", "表生淋滤使残余硫化物 δ65Cu 显著降低", "区分了原生与表生信号", "仅讨论干旱气候区", 3], { authors: ["Martin Reich"] }],
  ["Copper isotopes in the Gangdese porphyry copper belt, southern Tibet", "Acta Petrologica Sinica", 2021, 15, ["search"], [0.72, 0.79, 0, 0.49, 0.73], 1,
    ["冈底斯斑岩铜矿带的铜同位素组成及其成因意义？", "辉钼矿–黄铜矿铜同位素与 Re–Os 年代学", "驱龙、甲玛等 4 个矿床", "碰撞后斑岩矿床 δ65Cu 变化范围较俯冲型更窄", "首次报道冈底斯带的系统铜同位素数据", "", 4], { authors: ["Zhiming Yang", "Zengqian Hou"] }],
  ["Machine-learning prediction of porphyry fertility from zircon and copper isotopes", "Journal of Geochemical Exploration", 2024, 6, ["search"], [0.62, 0.7, 0, 0.3, 1], 1,
    ["能否结合锆石与铜同位素预测斑岩含矿性？", "随机森林分类", "全球 2400 件锆石与 300 件铜同位素数据", "联合特征使分类准确率提高到 87%", "将铜同位素引入含矿性评价模型", "训练数据偏向安第斯地区", 3], { authors: ["Guoxiong Chen"] }],
  ["Experimental constraints on copper partitioning between brine and vapor", "Contributions to Mineralogy and Petrology", 2017, 73, ["ref:seed1", "rec"], [0.76, 0.75, 1, 0.71, 0.47], 2,
    ["卤水与蒸汽相之间铜的分配及同位素分馏？", "合成流体包裹体实验", "800 °C、100 MPa 条件下的 NaCl–H2O 体系", "蒸汽相富集重铜同位素约 0.2‰", "约束了流体沸腾对铜同位素的影响", "温压条件有限", 4], { authors: ["Thomas Driesner", "Christoph Heinrich"] }],
  ["Sulfur and copper isotope coupling during magmatic degassing", "Nature Geoscience", 2022, 39, ["rec", "search"], [0.71, 0.77, 0.5, 0.62, 0.8], 2,
    ["岩浆去气过程中硫与铜同位素是否耦合？", "火山气体与凝结物分析", "日本和意大利活火山的气体凝结物", "去气使熔体 δ34S 与 δ65Cu 同向变化", "提出硫–铜同位素联合示踪", "", 4], { authors: ["Marie Edmonds"] }],
  ["Copper diffusion in silicate melts: implications for isotope fractionation", "American Mineralogist", 2015, 58, ["ref:seed1"], [0.6, 0.62, 0.5, 0.67, 0.33], 2,
    ["扩散过程能否产生显著的铜同位素分馏？", "扩散偶实验", "玄武质与流纹质熔体", "扩散可产生高达 1‰ 的动力学分馏", "提醒解释数据时需排除动力学效应", "仅限无水体系", 3], { authors: ["Youxue Zhang"] }],
  // Candidates that were not read
  ["Copper isotopes in seafloor massive sulfides", "Chemical Geology", 2014, 92, ["search"], [0.52, 0.55, 0, 0.7, 0.27], null, null, {}],
  ["Zinc and copper isotopes in sediment-hosted ore deposits", "Ore Geology Reviews", 2019, 33, ["search"], [0.47, 0.5, 0, 0.58, 0.6], null, null, {}],
  ["Isotopic tracing of copper in mine drainage", "Environmental Science & Technology", 2020, 40, ["search"], [0.39, 0.38, 0, 0.62, 0.67], null, null, {}],
  ["Magmatic sulfide saturation recorded by apatite inclusions", "Journal of Petrology", 2023, 12, ["cites:seed1"], [0.58, 0.54, 0.5, 0.42, 0.87], null, null, { inLibrary: true, itemKey: "LIBKEY7" }],
  ["Sulfur isotopes of arc magmas", "Geology", 2020, 66, ["seed"], [0.66, 0.6, 1, 0.68, 0.67], null, null, { seed: 2, inLibrary: true, itemKey: "SEEDKEY2" }],
];

const THEMES = [
  ["岩浆–热液过程中的铜同位素分馏", "硫化物饱和、流体出溶与氧化还原条件如何控制铜同位素在熔体、流体和硫化物间的分配。", ["fractionation", "sulfide", "melt"]],
  ["铜同位素分带与勘查应用", "斑岩矿床内的铜同位素空间分带、表生改造及其在找矿勘查中的应用。", ["zoning", "exploration", "supergene"]],
  ["流体沸腾与去气的实验约束", "卤水–蒸汽分离、岩浆去气和扩散等过程对铜同位素的影响。", ["brine", "vapor", "degassing"]],
];

const REPORT = {
  title: "斑岩铜矿成矿流体铜同位素示踪研究进展",
  summary:
    "铜同位素（δ65Cu）已成为示踪斑岩铜矿岩浆–热液演化的新工具。现有研究表明，岩浆硫化物饱和会使残余熔体及其出溶流体富集重铜同位素 [1, 4]，而流体出溶本身仅带来 0.1–0.3‰ 的有限分馏 [2]，因此成矿流体的铜同位素主要继承自母岩浆的演化历史。\n\n" +
    "在矿床尺度上，宾汉峡谷等矿床显示从钾化带到青磐岩化带 δ65Cu 系统增高约 1‰ [3]，这一分带已被用于构建勘查矢量 [6]；但表生淋滤可以显著改造原生信号 [7]。全球弧岩浆数据则为解释矿床数据提供了背景值 [5]。",
  sections: [
    {
      heading: "岩浆–热液过程中的铜同位素分馏",
      theme: 0,
      paragraphs: [
        "多项研究将铜同位素分馏归因于岩浆硫化物饱和：硫化物优先结合轻铜同位素，使残余熔体 δ65Cu 升高约 0.4‰ [1]，而实验表明这一分馏在还原条件下可增大约两倍 [4]。相比之下，流体出溶阶段的分馏较小，流体相仅相对熔体富集轻铜同位素 0.1–0.3‰ [2]。",
        "全球弧岩浆数据显示 δ65Cu 与岩浆氧化程度和地壳厚度相关 [5]，这与氧化还原控制分馏的实验结果一致 [4]；不过不同实验室数据间的系统偏差仍需审慎对待 [5]。",
      ],
      unverified: false,
    },
    {
      heading: "铜同位素分带与勘查应用",
      theme: 1,
      paragraphs: [
        "宾汉峡谷矿床的系统采样首次刻画了从钾化带到青磐岩化带约 1‰ 的铜同位素递增 [3]。基于安第斯与冈底斯 6 个矿床的对比，δ65Cu 高值可指示约 2 km 范围内的矿化中心 [6, 8]，机器学习模型进一步将铜同位素纳入含矿性评价 [9]。",
        "需要注意的是，表生富集作用会使残余硫化物 δ65Cu 显著降低 [7]，在风化强烈地区应用勘查矢量前必须区分原生与表生信号。",
      ],
      unverified: false,
    },
    {
      heading: "流体沸腾与去气的实验约束",
      theme: 2,
      paragraphs: [
        "合成流体包裹体实验显示，流体沸腾时蒸汽相富集重铜同位素约 0.2‰ [10]；火山气体凝结物则记录了去气过程中硫与铜同位素的同向变化 [11]。扩散实验提醒，动力学效应可产生高达 1‰ 的分馏 [12]，解释高温样品数据时必须加以排除。",
      ],
      unverified: false,
    },
  ],
  debates: [
    "成矿流体的铜同位素主要继承自硫化物饱和后的熔体 [1]，还是在流体沸腾和运移中被显著改造 [10, 11]，目前尚无定论。",
    "碰撞后斑岩矿床较窄的 δ65Cu 变化范围 [8] 是否反映了不同的岩浆源区，仍有争议。",
  ],
  gaps: [
    "高压、高氧逸度条件下的分馏实验仍然缺乏 [4, 10]。",
    "多数研究集中于安第斯地区，青藏高原等碰撞造山带的数据较少 [8]。",
  ],
  future: [
    "结合铜、硫、铁等多同位素体系，区分岩浆与热液过程的贡献 [11]。",
    "建立统一的数据标准和实验室间比对，减小全球汇编数据的系统偏差 [5]。",
    "将铜同位素勘查矢量与地球物理、机器学习方法结合并在新区验证 [6, 9]。",
  ],
  reading: [5, 1, 2, 3, 6, 10, 7],
  partial: false,
};

function buildPool(now) {
  const read = [];
  const pool = PAPERS.map(([title, venue, year, citations, via, [total, sim, link, cites, recency], theme, note, extra], i) => {
    const doi = `10.5555/survey.${i + 1}`;
    const paper = {
      id: `doi:${doi}`,
      title,
      abstract: `${title}. This study reports new data and models on copper isotope behaviour in magmatic-hydrothermal systems and discusses the implications for porphyry ore formation.`,
      tldr: i % 3 === 0 ? `${title.split(" ").slice(0, 6).join(" ")} — key findings summarised.` : null,
      authors: (extra.authors || ["Ann Lee"]).map(person),
      authorCount: (extra.authors || ["Ann Lee"]).length + (i % 4),
      year,
      published: `${year}-06-01T00:00:00Z`,
      venue,
      issns: [],
      doi,
      url: `https://doi.org/${doi}`,
      type: "article",
      citations,
      openalexId: `W${4100000 + i}`,
      s2Id: i % 2 ? `s2preview${i}` : null,
      sources: via.includes("rec") ? ["openalex", "s2"] : ["openalex"],
      via,
      seed: extra.seed || 0,
      inLibrary: !!extra.inLibrary,
      itemKey: extra.itemKey || null,
      searchHits: via.includes("search") ? 2 : 0,
      scores: { total, sim, cosine: Math.round((0.35 + sim * 0.4) * 1000) / 1000, link, cites, recency },
      rank: i + 1,
    };
    if (note) {
      const [question, method, data, findings, contribution, limitations, relevance] = note;
      paper.read = { question, method, data, findings, contribution, limitations, relevance, ...(i === 0 ? { fullText: true } : {}) };
      paper.theme = theme;
      read.push(paper);
    }
    return paper;
  });
  // References: by theme, then score (the order the engine numbers them)
  const byTheme = THEMES.map((_, t) => read.filter((p) => p.theme === t));
  let n = 0;
  for (const group of byTheme) for (const p of group) p.n = ++n;
  const references = byTheme.flatMap((group, t) => group.map((p) => referenceOf(p, p.n, t)));
  const themes = THEMES.map(([label, summary, terms], t) => ({ label, summary, terms, paperIds: byTheme[t].map((p) => p.id) }));
  return { pool, references, themes, readIds: read.map((p) => p.id), now };
}

const stageSummaries = (at, pool, read) => ({
  seeds: { seeds: 2, missing: 0, openalex: 2, s2: 1, fullText: 1, at },
  queries: { queries: 6, source: "llm", at },
  expand: { found: pool + 4, pool: pool + 1, requests: 31, failed: 1, skipped: 0, beforeYear: 3, bySource: { OpenAlex: 24, "Semantic Scholar": 9, Crossref: 5 }, byVia: { search: 26, ref: 7, cites: 3, rec: 3 }, at },
  dedupe: { before: pool + 1, after: pool, merged: 1, inLibrary: 3, at },
  rank: { ranked: pool, embedder: "dashscope:text-embedding-v4:1024", at },
  enrich: { targets: pool, missing: 2, filled: 2, tldr: 6, stillMissing: 0, at },
  read: { selected: read, read: read - 2, cached: 2, failed: 0, at },
  themes: { themes: 3, papers: read, labelled: "llm", silhouette: 0.21, at },
  write: { kind: "review", sections: 3, failed: 0, unverified: 0, at },
});

export function createSurveyFixtures(now = Date.now()) {
  const created = now - 3 * HOUR;
  const { pool, references, themes, readIds } = buildPool(now);
  const input = {
    topic: "斑岩铜矿成矿流体中的铜同位素示踪",
    description: "关注岩浆–热液演化过程中铜同位素的分馏机制，以及铜同位素在斑岩铜矿勘查中的应用。",
    keywords: ["copper isotopes", "porphyry copper deposits", "ore-forming fluids"],
    seedKeys: ["SEEDKEY1", "SEEDKEY2"],
    depth: "standard",
    yearFrom: 2010,
    sources: { openalex: true, s2: true, crossref: true },
  };
  const done = {
    version: 1,
    id: ID,
    input,
    status: "done",
    stage: "done",
    createdAt: created,
    updatedAt: created + 9 * 60000,
    finishedAt: created + 9 * 60000,
    completed: SURVEY_STAGES.map((s) => s.key),
    stages: stageSummaries(created + 9 * 60000, pool.length, readIds.length),
    seeds: [
      { n: 1, key: "SEEDKEY1", title: PAPERS[0][0], abstract: "", doi: "10.5555/survey.1", year: 2019, creators: ["Li Wang", "Mei Chen"], tags: ["Copper isotopes", "Porphyry"], venue: "Geochimica et Cosmochimica Acta", itemType: "journalArticle", openalexId: "W4100000", s2Id: "s2preview0", references: [], fullText: 48210 },
      { n: 2, key: "SEEDKEY2", title: "Sulfur isotopes of arc magmas", abstract: "", doi: "10.5555/survey.17", year: 2020, creators: ["Li Wang"], tags: ["Sulfur"], venue: "Geology", itemType: "journalArticle", openalexId: "W4100016", s2Id: null, references: [], fullText: 0 },
    ],
    queries: ["copper isotope fractionation porphyry", "Cu isotopes magmatic hydrothermal fluids", "sulfide saturation copper isotopes", "copper isotope exploration vector", "brine vapor copper partitioning", "斑岩铜矿成矿流体中的铜同位素示踪"],
    terms: ["copper isotopes", "porphyry copper deposits", "sulfide saturation", "fluid exsolution", "supergene enrichment"],
    pool,
    readIds,
    themes,
    references,
    report: { kind: "review", ...REPORT, generatedAt: created + 9 * 60000, model: "deepseek/deepseek-chat" },
    errors: [{ stage: "expand", source: "Crossref", message: "检索 6/7失败：HTTP 504 for https://api.crossref.org/works?query.bibliographic=…" }],
    error: null,
    counts: { seeds: 2, candidates: pool.length, read: readIds.length, references: references.length },
    llm: true,
    model: "deepseek/deepseek-chat",
    saved: null,
    running: false,
    progress: null,
    interrupted: false,
  };
  const entry = (s, extra = {}) => ({
    id: s.id,
    topic: s.input.topic,
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
    status: s.status,
    depth: s.input.depth,
    stage: s.stage,
    counts: s.counts,
    error: s.error?.message || null,
    hasReport: !!s.report,
    llm: s.llm,
    interrupted: false,
    progress: null,
    ...extra,
  });
  const cancelled = { ...done, id: "mvn8w1k02b", input: { ...input, topic: "俯冲带流体的氧化还原状态", depth: "quick", seedKeys: [] }, status: "cancelled", stage: "read", createdAt: now - 2 * DAY, updatedAt: now - 2 * DAY + 4 * 60000, completed: SURVEY_STAGES.slice(0, 6).map((s) => s.key), report: null, references: [], themes: [], counts: { seeds: 0, candidates: 142, read: 6, references: 0 } };
  const failed = { ...done, id: "mvk3p9x07c", input: { ...input, topic: "Tibetan Plateau uplift and monsoon climate", depth: "deep" }, status: "failed", stage: "expand", createdAt: now - 6 * DAY, updatedAt: now - 6 * DAY + 60000, completed: ["seeds", "queries"], report: null, references: [], themes: [], pool: [], error: { stage: "expand", message: "所有数据源都请求失败：OpenAlex 免费额度已用完（HTTP 429）" }, counts: { seeds: 2, candidates: 0, read: 0, references: 0 } };

  // Made without an LLM: nothing read, the result is the candidate list grouped by cluster (labels = top terms)
  const listThemes = themes.map((t) => ({ ...t, label: t.terms.join(" · "), summary: "" }));
  const listPool = pool.map((p) => {
    const { read, ...rest } = p;
    return { ...rest, scores: { ...p.scores } };
  });
  const noLlm = {
    ...done,
    id: "mvl5t2r08d",
    input: { ...input, topic: "铜同位素勘查矢量（未启用大模型）", description: "", depth: "quick", yearFrom: null },
    createdAt: now - 26 * HOUR,
    updatedAt: now - 26 * HOUR + 3 * 60000,
    finishedAt: now - 26 * HOUR + 3 * 60000,
    pool: listPool,
    readIds: [],
    themes: listThemes,
    stages: { ...stageSummaries(now - 26 * HOUR, pool.length, 0), read: { skipped: true, at: now - 26 * HOUR }, themes: { themes: 3, papers: 12, labelled: "terms", at: now - 26 * HOUR } },
    report: { kind: "list", title: "铜同位素勘查矢量（未启用大模型）", notice: NO_LLM_NOTICE, groups: listThemes.map((t, i) => ({ label: t.label, terms: t.terms, refs: references.filter((r) => r.theme === i).map((r) => r.n) })), generatedAt: now - 26 * HOUR + 3 * 60000, model: null },
    errors: [],
    counts: { seeds: 2, candidates: pool.length, read: 0, references: references.length },
    llm: false,
    model: null,
  };

  const depths = Object.values(SURVEY_DEPTHS).map((d) => ({ key: d.key, label: d.label, pool: d.pool, deepRead: d.deepRead, estimate: surveyEstimate(d.key, { llm: true }) }));
  return {
    surveyOptions: { llm: true, openalexKey: true, semanticScholarKey: true, stages: SURVEY_STAGES.map((s) => ({ ...s })), depths, running: null },
    surveys: [entry(done), entry(noLlm), entry(cancelled), entry(failed)],
    surveyDetails: { [done.id]: done, [noLlm.id]: noLlm, [cancelled.id]: cancelled, [failed.id]: failed },
    // A running survey (state "running" of the preview): its live progress
    surveyProgress: { stage: "read", index: 6, label: "精读", message: "精读 17/40：Fluid exsolution controls copper isotope signatures of porphyry systems", current: 17, total: 40, counts: { seeds: 2, candidates: 371, read: 17, references: 0 } },
    libraryResults: [
      { key: "SEEDKEY1", title: PAPERS[0][0], year: 2019, creators: ["Li Wang", "Mei Chen", "Jeremy Richards"], doi: "10.5555/survey.1", itemType: "journalArticle" },
      { key: "SEEDKEY2", title: "Sulfur isotopes of arc magmas", year: 2020, creators: ["Li Wang"], doi: "10.5555/survey.17", itemType: "journalArticle" },
      { key: "LIBKEY3", title: "Porphyry copper deposits of the world: database and grade-tonnage models", year: 2008, creators: ["Dennis Singer"], doi: null, itemType: "report" },
    ],
    selectedItems: [{ key: "SEEDKEY1", title: PAPERS[0][0], year: 2019, creators: ["Li Wang", "Mei Chen"], doi: "10.5555/survey.1", itemType: "journalArticle", tags: ["Copper isotopes", "Porphyry"] }],
  };
}

/** No survey yet (a fresh install). */
export function createEmptySurveyFixtures(now = Date.now()) {
  const full = createSurveyFixtures(now);
  const depths = full.surveyOptions.depths.map((d) => ({ ...d, estimate: surveyEstimate(d.key, { llm: false, sources: { s2: false } }) }));
  return { ...full, surveyOptions: { ...full.surveyOptions, llm: false, openalexKey: false, semanticScholarKey: false, depths }, surveys: [], surveyDetails: {}, selectedItems: [] };
}
