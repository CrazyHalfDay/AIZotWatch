// Shared fixtures of the topic-survey tests: a small library, mocked
// OpenAlex / Semantic Scholar / Crossref / embedding / LLM services.

import { loadSettings } from "../src/settings.js";
import { embeddingHandler, mockHttp } from "./helpers.js";

export const LLM_URL = "https://llm.example/v1/chat/completions";

export function invert(text) {
  const index = {};
  text.split(" ").forEach((w, i) => (index[w] ||= []).push(i));
  return index;
}

export function oaWork(id, title, { doi = null, year = 2021, abstract = null, cites = 10, type = "article", venue = "Geochimica et Cosmochimica Acta" } = {}) {
  return {
    id: `https://openalex.org/${id}`,
    doi: doi ? `https://doi.org/${doi}` : null,
    display_name: title,
    publication_year: year,
    publication_date: `${year}-03-01`,
    type,
    is_paratext: false,
    authorships: [{ author: { display_name: "Ann Lee" } }, { author: { display_name: "Wei Chen" } }],
    primary_location: { source: { display_name: venue, issn: ["0016-7037"] }, landing_page_url: "https://example.org/x" },
    abstract_inverted_index: abstract === null ? null : invert(abstract),
    cited_by_count: cites,
  };
}

export function s2Paper(paperId, title, { doi = null, year = 2022, abstract = null, cites = 5, venue = "Nature Geoscience" } = {}) {
  return {
    paperId,
    externalIds: doi ? { DOI: doi } : {},
    title,
    abstract,
    year,
    publicationDate: `${year}-05-01`,
    venue,
    journal: { name: venue },
    authors: [{ name: "Bo Zhang" }],
    citationCount: cites,
  };
}

const ABS = (topic) => `${topic} ${"copper isotope fractionation records fluid evolution in porphyry systems and arc magmas".repeat(1)} with detailed measurements and modelling of the processes involved.`;

export const WORKS = {
  seed1: oaWork("W100", "Copper isotope fractionation in porphyry copper deposits", { doi: "10.1000/seed1", year: 2019, abstract: ABS("Copper isotope"), cites: 50 }),
  ref1: oaWork("W201", "Copper isotope fractionation during sulfide precipitation", { doi: "10.2000/ref1", year: 2015, abstract: ABS("Copper isotope copper sulfide"), cites: 120 }),
  lib3: oaWork("W202", "Mantle melting under ridges", { doi: "10.1000/lib3", year: 2012, abstract: "Mantle mantle peridotite melting beneath mid-ocean ridges and the generation of basalt magmas in the upper mantle.", cites: 300 }),
  battery: oaWork("W203", "Battery cathode degradation mechanisms", { doi: "10.2000/battery", year: 2020, abstract: "Battery battery cathode degradation in lithium ion cells studied with electron microscopy and electrochemistry methods.", cites: 40 }),
  cite1: oaWork("W301", "Copper isotope tracers of porphyry fluids", { doi: "10.3000/cite1", year: 2023, abstract: ABS("Copper isotope copper isotope tracers"), cites: 8 }),
  cite2: oaWork("W302", "Copper isotopes in arc magmas revisited", { doi: "10.3000/cite2", year: 2024, abstract: ABS("Copper isotopes arc magmas"), cites: 3 }),
  review: oaWork("W401", "Copper isotope systematics of porphyry deposits: a review", { doi: "10.4000/review1", year: 2022, abstract: ABS("Copper isotope review"), cites: 60, type: "review" }),
  degas: oaWork("W402", "Copper isotope behaviour during magmatic degassing", { year: 2018, abstract: null, cites: 20 }),
  galaxy: oaWork("W403", "Galaxy formation in the early universe", { doi: "10.4000/galaxy", year: 2021, abstract: "Galaxy galaxy formation and evolution in the early universe observed with deep surveys and telescopes.", cites: 90 }),
  paratext: { ...oaWork("W404", "Issue information", { doi: "10.4000/para" }), is_paratext: true },
};

export const S2 = {
  rec1: s2Paper("s2rec1", "Sulfur and copper isotope coupling in porphyry systems", { doi: "10.5000/rec1", year: 2021, abstract: ABS("Sulfur sulfur copper isotope coupling") }),
  ref1: s2Paper("s2ref1", "Copper isotope fractionation during sulfide precipitation", { doi: "10.2000/ref1", year: 2015, abstract: "short" }),
  cit2: s2Paper("s2cit2", "Copper isotopes in arc magmas revisited", { doi: "10.3000/cite2", year: 2024, abstract: ABS("Copper isotopes arc magmas revisited longer abstract from Semantic Scholar") }),
  pre1: s2Paper("s2pre1", "Copper Isotope Tracers of Porphyry Fluids", { year: 2022, abstract: ABS("preprint copper isotope tracers") }),
  seed2: s2Paper("s2seed2", "Sulfur isotopes of arc magmas", { year: 2020, abstract: "Sulfur" }),
};

export const CROSSREF_ITEM = {
  DOI: "10.6000/CR1",
  title: ["Isotope fractionation of copper in hydrothermal fluids"],
  author: [{ given: "Mei", family: "Liu" }],
  "container-title": ["Chemical Geology"],
  issued: { "date-parts": [[2020, 6, 2]] },
  type: "journal-article",
  "is-referenced-by-count": 12,
  ISSN: ["0009-2541"],
};

/** Plain library items (the watch adapter's shape plus seed metadata). */
export function libraryItems() {
  return [
    { key: "SEED1", itemType: "journalArticle", title: WORKS.seed1.display_name, abstract: ABS("Copper isotope"), doi: "10.1000/seed1", year: 2019, creators: ["Li Wang"], tags: ["Copper isotopes", "Porphyry"], venue: "Economic Geology" },
    { key: "SEED2", itemType: "journalArticle", title: "Sulfur isotopes of arc magmas", abstract: "Sulfur isotopes sulfur of arc magmas trace slab-derived fluids and the oxidation of the mantle wedge in subduction zones.", doi: "10.1000/seed2", year: 2020, creators: ["Li Wang"], tags: ["Sulfur"], venue: "Geology" },
    { key: "LIB3", itemType: "journalArticle", title: "Mantle melting under ridges", abstract: "Mantle melting abstract from the library", doi: "10.1000/lib3", year: 2012, creators: [], tags: [] },
    { key: "LIB4", itemType: "journalArticle", title: "Copper isotope behaviour during magmatic degassing", abstract: "Copper isotope copper behaviour during magmatic degassing of arc magmas, from the library item, long enough to read.", doi: "", year: 2018, creators: [], tags: [] },
  ];
}

export const FULL_TEXT =
  "Copper isotope fractionation in porphyry copper deposits\nAbstract\nWe measured copper isotopes.\n1. Introduction\n" +
  "Porphyry deposits are the main source of copper. ".repeat(200) +
  "\n5. Conclusions\nCopper isotopes trace fluid boiling.\nReferences\nSmith J. (2010) A paper. GCA.\n" +
  "Another reference line. ".repeat(400);

export function createLibrary(items = libraryItems(), { fullText = { SEED1: FULL_TEXT } } = {}) {
  const calls = { fullText: [] };
  return {
    calls,
    async getItems() {
      return items;
    },
    async getItemsByKeys(keys) {
      return keys.map((k) => items.find((i) => i.key === k)).filter(Boolean);
    },
    async getItemFullText(key, { maxChars }) {
      calls.fullText.push(key);
      return (fullText[key] || "").slice(0, maxChars);
    },
  };
}

/** Reply of the fake LLM for each survey prompt; `hooks` may override any kind. */
export function llmHandler(stats, hooks = {}) {
  return (req) => {
    const prompt = JSON.parse(req.body).messages[0].content;
    let kind = "other";
    if (prompt.includes("设计检索式")) kind = "queries";
    else if (prompt.includes("精读笔记")) kind = "read";
    else if (prompt.includes("按语义聚成的")) kind = "themes";
    else if (prompt.includes("撰写其中一节")) kind = "section";
    else if (prompt.includes("完成一篇关于")) kind = "overview";
    stats[kind] = (stats[kind] || 0) + 1;
    (stats.prompts ||= []).push({ kind, prompt });
    const reply = hooks[kind] ? hooks[kind](prompt, stats) : defaultReply(kind, prompt);
    if (reply && typeof reply === "object" && "status" in reply) return reply;
    return { choices: [{ message: { content: typeof reply === "string" ? reply : JSON.stringify(reply) } }] };
  };
}

export function defaultReply(kind, prompt) {
  if (kind === "queries") return { queries: ["copper isotope porphyry", "copper isotope \"fluid\" AND boiling", "copper isotope porphyry"], terms: ["copper isotopes", "porphyry deposits"] };
  if (kind === "read") {
    const title = prompt.match(/论文标题：(.*)/)[1];
    return `\`\`\`json\n${JSON.stringify({
      question: `${title} 的研究问题`,
      method: "同位素测试与热力学模拟",
      data: "斑岩矿床样品",
      findings: "铜同位素记录流体演化".repeat(10),
      contribution: "提出新的示踪方法",
      limitations: "",
      relevance: title.includes("Copper") ? 5 : 2,
    })}\n\`\`\``;
  }
  if (kind === "themes") {
    const k = Number(prompt.match(/按语义聚成的 (\d+) 组/)[1]);
    return Array.from({ length: k }, (_, i) => ({ label: `主题${i + 1}：铜同位素`, summary: `第 ${i + 1} 组的概括` }));
  }
  if (kind === "section") {
    const numbers = [...prompt.matchAll(/^\[(\d+)\] /gm)].map((m) => Number(m[1]));
    const label = prompt.match(/主题为「([^」]*)」/)[1];
    if (label.startsWith("主题2")) return { heading: "未引用的一节", paragraphs: ["这一段没有任何引用。"] };
    return { heading: `${label} 的进展`, paragraphs: [`研究表明铜同位素可以示踪成矿流体 [${numbers[0]}]，并与硫同位素耦合 [${numbers[0]}, 99]。`, `另一段讨论 [${numbers.join("-")}]【${numbers[0]}】。`] };
  }
  if (kind === "overview") {
    return { title: "斑岩铜矿中的铜同位素研究综述", summary: "总述第一段 [1]，以及不存在的文献 [120]。\n\n总述第二段 [2]。", debates: ["争议一 [1]", "[300]"], gaps: ["空白一"], future: ["方向一 [2]"], reading: [2, 1, 99, 2] };
  }
  return "OK";
}

/** Routes of every mocked service; `overrides` replaces the handler of a route by name. */
export function surveyHttp(counter = {}, overrides = {}) {
  const searchQueries = [];
  const s2Queries = [];
  const route = (name, matcher, handler) => [matcher, overrides[name] || handler];
  const url = (req) => new URL(req.url);
  const http = mockHttp([
    route("oaSeed", (req) => req.url.startsWith("https://api.openalex.org/works/"), (req) => {
      const path = decodeURIComponent(url(req).pathname);
      if (path === "/works/doi:10.1000/seed1") return { ...WORKS.seed1, referenced_works: ["https://openalex.org/W201", "https://openalex.org/W202", "https://openalex.org/W203"] };
      return { status: 404, data: { error: "not found" } };
    }),
    route("oaWorks", (req) => req.url.startsWith("https://api.openalex.org/works?"), (req) => {
      const params = url(req).searchParams;
      const filter = params.get("filter") || "";
      if (filter.startsWith("doi:")) return { meta: { count: 0 }, results: [] };
      if (/^openalex:/.test(filter)) {
        const ids = filter.split(",")[0].slice("openalex:".length).split("|");
        return { results: Object.values(WORKS).filter((w) => ids.includes(w.id.split("/").pop())) };
      }
      if (filter.includes("cites:W100")) return { results: [WORKS.cite1, WORKS.cite2] };
      if (params.get("search")) {
        searchQueries.push(params.get("search"));
        if (params.get("search").startsWith("Sulfur isotopes")) return { results: [] };
        return { results: [WORKS.review, WORKS.degas, WORKS.galaxy, WORKS.paratext, WORKS.seed1] };
      }
      return { results: [] };
    }),
    route("s2Batch", (req) => req.url.startsWith("https://api.semanticscholar.org/graph/v1/paper/batch"), (req) => {
      const ids = JSON.parse(req.body).ids;
      const fields = url(req).searchParams.get("fields");
      if (fields === "paperId,externalIds") return ids.map((id) => (id === "DOI:10.1000/seed1" ? { paperId: "s2seed1", externalIds: { DOI: "10.1000/seed1" } } : null));
      return ids.map((id) =>
        id === "DOI:10.6000/cr1"
          ? { paperId: "s2cr1", abstract: "Copper isotope fractionation in hydrothermal fluids measured in experiments at high temperature, from Semantic Scholar.", tldr: { text: "Copper fractionates in fluids." } }
          : id === "DOI:10.4000/review1"
            ? { paperId: "s2rev", tldr: { text: "A review of copper isotopes." } }
            : null,
      );
    }),
    route("s2Search", (req) => req.url.startsWith("https://api.semanticscholar.org/graph/v1/paper/search"), (req) => {
      const q = url(req).searchParams.get("query");
      s2Queries.push(q);
      if (q.startsWith("Sulfur isotopes")) return { total: 1, data: [S2.seed2] };
      return { total: 2, data: [S2.pre1, S2.cit2] };
    }),
    route("s2Edges", (req) => /api\.semanticscholar\.org\/graph\/v1\/paper\/[^/]+\/(references|citations)/.test(req.url), (req) => {
      if (req.url.includes("/s2seed1/references")) return { data: [{ citedPaper: S2.ref1, isInfluential: true }, { citedPaper: null }] };
      if (req.url.includes("/s2seed1/citations")) return { data: [{ citingPaper: S2.cit2 }] };
      return { data: [] };
    }),
    route("s2Recs", (req) => req.url.startsWith("https://api.semanticscholar.org/recommendations/v1/papers"), () => ({ recommendedPapers: [S2.rec1, { title: null }] })),
    route("crossrefSearch", (req) => req.url.startsWith("https://api.crossref.org/works?"), () => ({ message: { items: [CROSSREF_ITEM, { DOI: "10.6000/x", type: "journal-issue", title: ["Issue"] }] } })),
    route("crossrefWork", (req) => req.url.startsWith("https://api.crossref.org/works/"), () => ({ status: 404, data: { message: "not found" } })),
    route("embeddings", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)),
    route("llm", LLM_URL, (req) => ({ choices: [{ message: { content: "{}" } }] })),
  ]);
  http.searchQueries = searchQueries;
  http.s2Queries = s2Queries;
  return http;
}

export function surveySettings(prefs = {}) {
  const all = {
    "embedding.provider": "dashscope",
    "embedding.apiKey": "sk-test",
    "llm.enabled": true,
    "llm.apiKey": "sk-llm",
    "llm.baseURL": "https://llm.example/v1",
    "llm.model": "test-model",
    "apis.openalexKey": "oa-key",
    "apis.semanticScholarKey": "s2-key",
    "sources.crossref.mailto": "me@example.org",
    ...prefs,
  };
  return loadSettings((k) => all[k]);
}

export const SURVEY_INPUT = {
  topic: "Copper isotopes in porphyry systems",
  description: "铜同位素在斑岩铜矿成矿流体示踪中的应用",
  keywords: "copper isotope, porphyry",
  seedKeys: ["SEED1", "SEED2"],
  depth: "quick",
  yearFrom: null,
};
