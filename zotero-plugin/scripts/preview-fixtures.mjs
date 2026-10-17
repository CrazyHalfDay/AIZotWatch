// Fixture data for scripts/preview.mjs (dev only): a realistic inbox across the
// three tracks, a 14-run history, research directions, followed authors and a
// last run with one problem. Everything is relative to `now` so relative times
// ("2 小时前") look the same on every run.

import { createEmptySurveyFixtures, createSurveyFixtures } from "./preview-survey-fixtures.mjs";

const HOUR = 3600000;
const DAY = 86400000;

const SUMMARY = {
  research_question: "斑岩铜矿成矿过程中铜同位素如何随硫化物饱和而分馏？",
  methodology: "对智利 El Teniente 矿床不同阶段的岩浆包裹体和硫化物进行原位铜同位素分析，并结合热力学模拟。",
  key_findings: "早期硫化物饱和使残余熔体 δ65Cu 升高约 0.4‰，成矿流体继承了这一信号。",
  innovation: "首次将熔体包裹体铜同位素与流体演化直接对应。",
  relevance_note: "与你文库中关于斑岩成矿和硫化物饱和的研究高度相关。",
};

const authors = (...names) => names.map((name) => ({ name }));

/** Works: [track label, title, translated title, venue, source, score, similarity, days ago, extra]. */
const WORK_SPECS = [
  ["must_read", "Copper isotope fractionation during sulfide saturation in arc magmas", "弧岩浆硫化物饱和过程中的铜同位素分馏", "Geochimica et Cosmochimica Acta", "crossref", 0.91, 0.74, 1, { summary: SUMMARY, impactFactor: 5.0, topic: "斑岩铜矿成矿", followedAuthors: ["汪在聪"] }],
  ["must_read", "Slab-derived sulfur controls the oxidation state of arc magmas", "俯冲板片来源的硫控制弧岩浆的氧化状态", "Earth and Planetary Science Letters", "crossref", 0.88, 0.71, 2, { impactFactor: 5.3, topic: "俯冲带流体与氧化还原" }],
  ["must_read", "Melt inclusion constraints on volatile budgets beneath the Andes", "熔体包裹体对安第斯山下挥发分收支的约束", "Nature Geoscience", "crossref", 0.84, 0.69, 3, { impactFactor: 18.3, topic: "岩浆挥发分" }],
  ["consider", "Zircon trace elements as a fertility indicator for porphyry deposits", "锆石微量元素作为斑岩矿床含矿性指标", "Economic Geology", "crossref", 0.72, 0.6, 2, { impactFactor: 5.5, topic: "斑岩铜矿成矿" }],
  ["consider", "Thermodynamic modelling of chalcophile element partitioning in hydrous basalts", "含水玄武岩中亲铜元素分配的热力学模拟", "Contributions to Mineralogy and Petrology", "crossref", 0.69, 0.58, 4, { impactFactor: 3.5, topic: "岩浆挥发分" }],
  ["consider", "Rapid exhumation of the Gangdese batholith recorded by apatite fission tracks", "磷灰石裂变径迹记录的冈底斯岩基快速剥露", "Tectonics", "crossref", 0.66, 0.55, 5, { impactFactor: 4.2, topic: "青藏高原构造演化" }],
  ["consider", "Iron isotope systematics of mid-ocean ridge basalts revisited", "重新审视洋中脊玄武岩的铁同位素体系", "Chemical Geology", "crossref", 0.64, 0.54, 3, { impactFactor: 3.9, topic: "地幔部分熔融" }],
  ["consider", "Machine-learning classification of igneous zircon provenance", null, "arXiv", "arxiv", 0.61, 0.52, 1, { topic: "斑岩铜矿成矿" }],
  ["consider", "Sulfide liquid immiscibility in lunar basalts", "月球玄武岩中的硫化物熔体不混溶", "EarthArXiv", "eartharxiv", 0.58, 0.5, 6, { topic: "地幔部分熔融" }],
  ["ignore", "A regional geochemical survey of stream sediments in southern Tibet", "藏南水系沉积物区域地球化学调查", "Journal of Geochemical Exploration", "crossref", 0.47, 0.41, 5, { impactFactor: 3.4, topic: "青藏高原构造演化" }],
  ["ignore", "Clay mineral transformations in weathering profiles", "风化剖面中的黏土矿物转变", "Clays and Clay Minerals", "crossref", 0.44, 0.38, 6, {}],
  ["flagship", "Deep carbon recycling recorded by superdeep diamonds", "超深金刚石记录的深部碳循环", "Nature", "crossref", 0.36, 0.31, 2, { gate: { pos: 0.412, neg: 0.233, judged: false }, topic: "地幔部分熔融" }],
  ["flagship", "Early Cambrian oxygenation pulses inferred from thallium isotopes", "铊同位素揭示的早寒武世氧化脉冲", "Science", "crossref", 0.33, 0.27, 3, { gate: { pos: 0.338, neg: 0.251, judged: true } }],
  ["flagship", "A long-lived magma reservoir beneath Yellowstone imaged by seismic tomography", "地震层析成像揭示黄石下方长期存在的岩浆房", "Nature Communications", "crossref", 0.31, 0.25, 4, { gate: { pos: 0.391, neg: 0.214, judged: false } }],
  ["followed", "Melt inclusions record slab-derived sulfur in the Mariana arc", "熔体包裹体记录马里亚纳弧中来自板片的硫", "EarthArXiv", "openalex", 0, 0, 3, { followedAuthors: ["汪在聪"], venue: "EarthArXiv" }],
  // No abstract anywhere, but a Semantic Scholar TLDR (shown as one muted line)
  ["followed", "Metamorphic devolatilization of subducted carbonates", "俯冲碳酸盐的变质脱挥发分作用", "Journal of Petrology", "openalex", 0, 0, 9, {
    followedAuthors: ["Jay J. Ague"],
    abstract: null,
    tldr: "Carbonate devolatilization in subducting slabs releases far less CO2 than assumed, so most slab carbon reaches the deep mantle.",
  }],
];

/** The preview's inbox, last run, history and the derived dashboard. */
export function createFixtures(now = Date.now()) {
  const runAt = now - 2 * HOUR - 14 * 60000;
  const lastRunId = `run-${runAt}`;
  const works = WORK_SPECS.map(([label, title, translatedTitle, venue, source, score, similarity, daysAgo, extra], i) => ({
    id: `doi:10.5555/preview.${i + 1}`,
    source,
    title,
    translatedTitle: translatedTitle || undefined,
    abstract:
      "We combine in situ isotope measurements, trace-element systematics and thermodynamic modelling to constrain the processes that " +
      "control metal and volatile transfer in magmatic systems. The results show a consistent signal across several localities and " +
      "suggest that the timing of sulfide saturation, rather than the initial metal budget, governs the fertility of arc magmas.",
    authors: authors("Li Wang", "Maria González", "Kenji Sato", "A. Smith", "Zaicong Wang", "J. Ague", "P. Müller"),
    doi: `10.5555/preview.${i + 1}`,
    url: `https://doi.org/10.5555/preview.${i + 1}`,
    published: new Date(now - daysAgo * DAY).toISOString(),
    venue: source === "arxiv" || source === "eartharxiv" ? null : venue,
    label,
    score,
    similarity,
    status: "new",
    firstRunId: i < 9 ? lastRunId : "run-earlier",
    ...extra,
  }));
  // A few handled works for the 已加入 / 已忽略 tabs
  works.push(
    { ...works[3], id: "doi:10.5555/preview.added1", title: "Hydrothermal alteration footprints of porphyry systems", translatedTitle: "斑岩体系的热液蚀变印记", status: "added", itemID: 1, firstRunId: "run-earlier", summary: undefined },
    { ...works[6], id: "doi:10.5555/preview.added2", title: "Osmium isotopes in arc lavas", translatedTitle: "弧熔岩中的锇同位素", status: "added", itemID: 2, firstRunId: "run-earlier" },
    { ...works[10], id: "doi:10.5555/preview.dismissed1", title: "Soil carbon fluxes in alpine meadows", translatedTitle: "高寒草甸土壤碳通量", status: "dismissed", firstRunId: "run-earlier" },
  );

  const lastRun = {
    runId: lastRunId,
    runAt,
    finishedAt: runAt + 3 * 60000 + 12000,
    stats: {
      libraryItems: 2318,
      fetched: 1284,
      afterDedupe: 1201,
      afterSplit: 1037,
      afterKeywords: 1009,
      afterAbstract: 871,
      afterGate: 86,
      afterRecent: 41,
      recommended: 30,
      recentDays: 7,
      requireAbstract: true,
      newRecommendations: 23,
      newByLabel: { personal: 18, flagship: 3, followed: 2 },
      newMustRead: 4,
      mustRead: 6,
      journals: 52,
      personalIssns: 48,
      perSource: { Crossref: 1102, 顶刊: 164, arXiv: 18 },
      flagship: { candidates: 164, withAbstract: 151, recent: 120, accepted: 9, gray: 14, judged: 14, passed: 12, rejected: 108, recommended: 12 },
      followed: { authors: 3, fetched: 7, new: 2 },
      abstracts: { missing: 96, filled: 71, cacheHits: 20, crossref: 5, elsevier: 30, springer: 10, openalex: 20, s2: 6, tldr: 9 },
      topics: { clusters: 14, labeled: 14, generated: 0 },
      profile: { items: 2120, embedded: 12, reused: 2108, clustersRebuilt: false, clusters: 14 },
    },
    errors: [{ stage: "fetch", source: "arXiv", message: "HTTP 503 Service Unavailable" }],
  };

  const history = Array.from({ length: 14 }, (_, i) => {
    const at = runAt - (13 - i) * DAY;
    const wave = [3, 5, 2, 7, 4, 9, 6, 3, 8, 5, 11, 4, 6, 18][i];
    const by = { personal: wave, flagship: [1, 0, 2, 1, 0, 3, 1, 0, 2, 1, 2, 0, 1, 3][i], followed: [0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 2][i] };
    return {
      runId: `run-${at}`,
      runAt: at,
      finishedAt: at + 150000 + i * 4000,
      fetched: 900 + i * 31,
      recommended: 30,
      newByLabel: by,
      newMustRead: Math.round(wave / 4),
      errors: i === 13 ? 1 : i === 6 ? 2 : 0,
    };
  });
  history[13].runId = lastRunId;

  const pendingWorks = works.filter((w) => w.status === "new");
  const countTrack = (track) => pendingWorks.filter((w) => (w.label === "flagship" || w.label === "followed" ? w.label : "personal") === track).length;
  const schedule = { autoRun: true, intervalHours: 24, lastRun: lastRun.finishedAt, lastAttempt: runAt, nextRunAt: lastRun.finishedAt + 24 * HOUR };
  const followedAuthors = [
    { name: "汪在聪", orcid: "0000-0002-3584-1673", openalexId: null },
    { name: "Jay J. Ague", orcid: "0000-0001-8408-0279", openalexId: null },
    { name: "Cin-Ty Lee", orcid: null, openalexId: "A5023888391" },
  ];

  const view = {
    works,
    lastRun,
    history,
    running: false,
    progress: null,
    lastError: null,
    llmEnabled: true,
    provider: "dashscope",
    collection: "ZotWatch 推荐",
    version: "0.4.0",
    schedule,
  };

  const dashboard = {
    version: "0.4.0",
    library: { name: "我的文库", items: 2318, withAbstract: 2120 },
    embedding: { provider: "dashscope", model: "text-embedding-v4", label: "阿里云 DashScope · text-embedding-v4", local: false },
    llm: { enabled: true, configured: true, model: "deepseek/deepseek-chat", topicLabels: true },
    schedule,
    running: false,
    pending: {
      total: pendingWorks.length,
      personal: countTrack("personal"),
      flagship: countTrack("flagship"),
      followed: countTrack("followed"),
      mustRead: pendingWorks.filter((w) => w.label === "must_read").length,
    },
    added: works.filter((w) => w.status === "added").length,
    addedDays: 60,
    scoring: { recentDays: 7, requireAbstract: true },
    lastRun,
    lastError: null,
    history,
    topics: [
      ["斑岩铜矿成矿", 312],
      ["俯冲带流体与氧化还原", 268],
      ["岩浆挥发分", 221],
      ["地幔部分熔融", 198],
      ["青藏高原构造演化", 176],
      ["锆石 U-Pb 年代学", 143],
      ["稳定同位素地球化学", 131],
      ["变质作用与脱水", 102],
      ["沉积盆地演化", 77],
      ["矿物物理", 54],
    ].map(([label, size]) => ({ label, size })),
    topicCount: 14,
    clusters: 14,
    followed: {
      enabled: true,
      daysBack: 60,
      authors: followedAuthors.map((a) => ({ ...a, pending: pendingWorks.filter((w) => (w.followedAuthors || []).includes(a.name)).length })),
      invalid: ["张伟（没有 ORCID）"],
    },
    flagship: { enabled: true, issnCount: 12, stats: lastRun.stats.flagship },
    journals: { count: 52, withImpactFactor: 47, autoFromLibrary: true, error: null },
    sources: [
      { key: "crossref", name: "Crossref", enabled: true, count: 1102, detail: "48 个 ISSN", error: null, ran: true },
      { key: "flagship", name: "顶刊", enabled: true, count: 164, detail: "12 个 ISSN", error: null },
      { key: "arxiv", name: "arXiv", enabled: true, count: 18, detail: "physics.geo-ph", error: "HTTP 503 Service Unavailable" },
      { key: "eartharxiv", name: "EarthArXiv", enabled: false, count: null, detail: null, error: null },
      { key: "followed", name: "关注作者", enabled: true, count: 7, detail: "OpenAlex，3 位", error: null },
      { key: "abstracts", name: "摘要补全", enabled: true, count: 71, detail: "上次补全 71 / 96", error: null },
      { key: "openalex", name: "OpenAlex", enabled: true, count: null, detail: "今日剩余 $0.82", error: null, status: { tone: "good", label: "正常" } },
      { key: "semanticscholar", name: "Semantic Scholar", enabled: true, count: null, detail: "摘要与 TLDR", error: null, status: { tone: "good", label: "已配置" } },
      { key: "embedding", name: "向量化", enabled: true, count: null, detail: "阿里云 DashScope · text-embedding-v4", error: null },
      { key: "llm", name: "大模型", enabled: true, count: null, detail: "deepseek/deepseek-chat", error: null },
    ],
    abstracts: lastRun.stats.abstracts,
    apis: null, // set below (the same object getApiStatus() returns)
    dataDir: "/home/user/Zotero/zotwatch",
  };
  const apiStatus = {
    openalexKey: true,
    semanticScholarKey: true,
    mailto: true,
    openalexUsage: { remainingUsd: 0.82, limitUsd: 1, remaining: null, limit: null, keyed: true, at: now - 25 * 60000, text: "今日剩余 $0.82 / $1.00", short: "今日剩余 $0.82" },
  };
  dashboard.apis = apiStatus;

  const connections = [
    { key: "embedding", name: "向量化", target: "dashscope", ok: true, detail: "正常，维度 1024" },
    { key: "llm", name: "大模型", target: "deepseek/deepseek-chat", ok: true, detail: "OK" },
    { key: "crossref", name: "Crossref", target: null, ok: true, detail: "可访问" },
    { key: "openalex", name: "OpenAlex", target: null, ok: true, detail: "Key 有效，今日剩余 $0.82 / $1.00" },
    { key: "semanticscholar", name: "Semantic Scholar", target: null, ok: false, detail: "请求过于频繁（HTTP 429），请稍后再试" },
    { key: "elsevier", name: "Elsevier", target: null, ok: false, detail: "Key 无效（HTTP 401）" },
  ];

  const whitelist = [
    "issn,title,category,impact_factor",
    "0016-7037,Geochimica et Cosmochimica Acta,GEOCHEMISTRY,5.0",
    "0012-821X,Earth and Planetary Science Letters,GEOCHEMISTRY,5.3",
    "1752-0894,Nature Geoscience,GEOSCIENCES,18.3",
    "0009-2541,Chemical Geology,GEOCHEMISTRY,3.9",
    "0010-7999,Contributions to Mineralogy and Petrology,MINERALOGY,3.5",
    "0361-0128,Economic Geology,GEOLOGY,5.5",
    "0022-3530,Journal of Petrology,GEOCHEMISTRY,3.9",
    "0278-7407,Tectonics,GEOSCIENCES,4.2",
    // Written by 「从文库生成」 before 0.4.0: 「补全学科与影响因子」 fills them
    "1674-7313,Science China Earth Sciences,library:14,",
    "1000-0569,Acta Petrologica Sinica,library:6,",
  ].join("\n");

  // OpenAlex sources as sources/openalex-entities.js maps them (search results, proposals, names)
  const source = (openalexId, title, issns, impactFactor, subfield, extra = {}) => ({
    openalexId,
    title,
    issns,
    impactFactor,
    subfield,
    category: subfield.toUpperCase().replace(/[^A-Z0-9]+/g, "_"),
    publisher: "Elsevier BV",
    country: "NL",
    worksCount: 12000,
    ...extra,
  });
  const journalSearch = [
    source("S49861241", "Geochimica et Cosmochimica Acta", ["0016-7037", "1872-9533"], 5.0, "Geochemistry and Petrology", { worksCount: 28117 }),
    source("S95464858", "Geochemistry, Geophysics, Geosystems", ["1525-2027"], 3.6, "Geophysics", { publisher: "Wiley", country: "US", worksCount: 9930 }),
    source("S31283462", "Geochemical Perspectives Letters", ["2410-339X", "2410-3403"], 4.1, "Geochemistry and Petrology", { publisher: "European Association of Geochemistry", country: "GB", worksCount: 640 }),
    source("S12398740", "Applied Geochemistry", ["0883-2927", "1872-9134"], 3.1, "Pollution", { worksCount: 7354 }),
    source("S15342210", "地球化学", ["0379-1726"], 1.1, "Geochemistry and Petrology", { publisher: "科学出版社", country: "CN", worksCount: 4120, category: "GEOCHEMISTRY_AND_PETROLOGY (CN)" }),
  ];
  const authorSearch = [
    { id: "A5034567890", name: "Zaicong Wang", orcid: "0000-0002-3584-1673", institution: "China University of Geosciences", country: "CN", worksCount: 118, citedByCount: 3120, topic: "Platinum-group element geochemistry" },
    { id: "A5045678901", name: "Zaicong Wang", orcid: null, institution: "Chinese Academy of Sciences", country: "CN", worksCount: 12, citedByCount: 140, topic: "Mantle xenoliths" },
    { id: "A5056789012", name: "Z. C. Wang", orcid: "0000-0003-1234-5678", institution: "University of Cologne", country: "DE", worksCount: 37, citedByCount: 905, topic: "Osmium isotope systematics" },
  ];
  const proposals = {
    proposals: [
      { ...source("S49861241", "Geochimica et Cosmochimica Acta", ["0016-7037", "1872-9533"], 5.0, "Geochemistry and Petrology"), libraryCount: 64, origin: "library", libraryTitle: null, verified: true, inList: true, checked: false },
      { ...source("S30530123", "Lithos", ["0024-4937", "1872-6143"], 2.4, "Geochemistry and Petrology"), libraryCount: 23, origin: "library", libraryTitle: null, verified: true, inList: false, checked: true },
      { ...source("S41725839", "Precambrian Research", ["0301-9268"], 3.2, "Geology"), libraryCount: 17, origin: "library", libraryTitle: "Precambrian Res.", verified: true, inList: false, checked: true },
      { ...source("S16236278", "Acta Petrologica Sinica", ["1000-0569"], 1.3, "Geology", { publisher: "科学出版社", country: "CN", category: "GEOLOGY (CN)" }), libraryCount: 9, origin: "library", libraryTitle: null, verified: true, inList: true, checked: false },
      { ...source("S98201333", "Gondwana Research", ["1342-937X", "1878-0571"], 6.7, "Geology"), libraryCount: 6, origin: "library", libraryTitle: null, verified: true, inList: false, checked: true },
      { title: "Journal of Asian Earth Sciences", issns: ["1367-9120"], category: "", subfield: "", impactFactor: null, libraryCount: 1, origin: "library", libraryTitle: null, verified: false, inList: false, checked: false },
      { ...source("S59278812", "Earth-Science Reviews", ["0012-8252"], 10.2, "Geology"), libraryCount: 0, origin: "ai", libraryTitle: null, verified: true, inList: false, checked: true },
      { ...source("S28839020", "Geology", ["0091-7613", "1943-2682"], 4.8, "Geology", { publisher: "Geological Society of America", country: "US" }), libraryCount: 0, origin: "ai", libraryTitle: null, verified: true, inList: false, checked: true },
      { ...source("S43294512", "Mineralium Deposita", ["0026-4598", "1432-1866"], 4.4, "Geochemistry and Petrology", { publisher: "Springer" }), libraryCount: 0, origin: "ai", libraryTitle: null, verified: true, inList: false, checked: true },
      { ...source("S77188262", "Ore Geology Reviews", ["0169-1368"], 3.3, "Economic Geology"), libraryCount: 0, origin: "ai", libraryTitle: null, verified: true, inList: false, checked: true },
      { ...source("S11293940", "Journal of Metamorphic Geology", ["0263-4929"], 1.9, "Geology", { publisher: "Wiley" }), libraryCount: 0, origin: "ai", libraryTitle: null, verified: true, inList: false, checked: false },
    ],
    warnings: [],
    missing: ["Unpublished field notes", "Mineral Exploration Newsletter"],
    stats: {},
  };
  // Names of flagship ISSNs beyond the built-in ones (lookupJournalNames)
  const journalNames = {
    "0091-7613": { title: "Geology", issns: ["0091-7613", "1943-2682"] },
    "1943-2682": { title: "Geology", issns: ["0091-7613", "1943-2682"] },
    "1234-5679": null,
  };

  return { view, dashboard, connections, whitelist, apiStatus, journalSearch, authorSearch, proposals, journalNames, ...createSurveyFixtures(now), prefs: { "apis.openalexKey": "oa-preview-key", "apis.semanticScholarKey": "s2-preview-key", "sources.crossref.mailto": "li.wang@example.edu", "embedding.provider": "dashscope", "embedding.apiKey": "sk-preview", "llm.enabled": true, "llm.apiKey": "sk-preview", "followed.enabled": true, "followed.authors": "汪在聪, 0000-0002-3584-1673\nJay J. Ague, https://orcid.org/0000-0001-8408-0279\nCin-Ty Lee, https://openalex.org/A5023888391\n张伟（没有 ORCID）", "flagship.enabled": true, "flagship.issns": "0028-0836\n1476-4687\n0036-8075\n1095-9203\n2041-1723\n1752-0894\n1752-0908\n0091-7613\n1943-2682\n1234-5679", "sources.arxiv.enabled": true } };
}

/** A fresh install: nothing has run yet. */
export function createEmptyFixtures(now = Date.now()) {
  const full = createFixtures(now);
  const schedule = { autoRun: true, intervalHours: 24, lastRun: null, lastAttempt: null, nextRunAt: now - 1 };
  return {
    ...full,
    view: { ...full.view, works: [], lastRun: null, history: [], schedule, llmEnabled: false, provider: "local" },
    dashboard: {
      ...full.dashboard,
      embedding: { provider: "local", model: "", label: "本地模式", local: true },
      llm: { enabled: false, configured: false, model: "deepseek/deepseek-chat", topicLabels: true },
      schedule,
      pending: { total: 0, personal: 0, flagship: 0, followed: 0, mustRead: 0 },
      added: 0,
      lastRun: null,
      history: [],
      topics: [],
      topicCount: 0,
      clusters: null,
      followed: { enabled: false, daysBack: 60, authors: [], invalid: [] },
      flagship: { enabled: false, issnCount: 12, stats: null },
      // Default settings of a fresh install, before the first run
      sources: [
        { key: "crossref", name: "Crossref", enabled: true, count: null, detail: null, error: null },
        { key: "flagship", name: "顶刊", enabled: false, count: null, detail: "12 个 ISSN", error: null },
        { key: "arxiv", name: "arXiv", enabled: false, count: null, detail: "physics.geo-ph, astro-ph.EP", error: null },
        { key: "eartharxiv", name: "EarthArXiv", enabled: false, count: null, detail: null, error: null },
        { key: "followed", name: "关注作者", enabled: false, count: null, detail: "OpenAlex，0 位", error: null },
        { key: "abstracts", name: "摘要补全", enabled: true, count: null, detail: null, error: null },
        { key: "openalex", name: "OpenAlex", enabled: true, count: null, detail: "免费额度很小", error: null, status: { tone: "warning", label: "未配置 Key" } },
        { key: "semanticscholar", name: "Semantic Scholar", enabled: false, count: null, detail: "可选", error: null, status: { tone: "off", label: "未配置" } },
        { key: "embedding", name: "向量化", enabled: true, count: null, detail: "本地模式", error: null },
        { key: "llm", name: "大模型", enabled: false, count: null, detail: "未启用", error: null },
      ],
      journals: { count: 0, withImpactFactor: 0, autoFromLibrary: true, error: null },
      abstracts: null,
      apis: { openalexKey: false, semanticScholarKey: false, mailto: false, openalexUsage: null },
    },
    apiStatus: { openalexKey: false, semanticScholarKey: false, mailto: false, openalexUsage: null },
    ...createEmptySurveyFixtures(now),
    prefs: {},
  };
}
