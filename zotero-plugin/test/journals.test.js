// Journal list tooling of 0.4.0: OpenAlex entity lookups
// (sources/openalex-entities.js), the whitelist merge helpers
// (core/journals.js), the ported journal prompt (llm/journals.js) and the
// proposals / metadata fill of journal-proposals.js, with mocked HTTP.

import assert from "node:assert/strict";
import test from "node:test";

import { createCancelSignal } from "../src/core/http.js";
import {
  isEmptyCategory,
  isIncompleteJournal,
  journalRows,
  mergeWhitelistRows,
  parseWhitelist,
  summarizeWhitelist,
  whitelistIssns,
} from "../src/core/journals.js";
import { AI_DEFAULT_MIN_IF, abbreviates, fillJournalMetadata, proposeJournals, resolveVenueIssns, withChineseMark } from "../src/journal-proposals.js";
import { extractJsonObjects, journalGenerationPrompt, parseJournalReply } from "../src/llm/journals.js";
import {
  ISSN_BATCH_SIZE,
  categoryFromTopic,
  findSourceByTitle,
  isChineseSource,
  looksLikeIssn,
  lookupSourcesByIssn,
  mapOpenAlexAuthor,
  mapOpenAlexSource,
  openAlexErrorText,
  searchAuthors,
  searchSources,
} from "../src/sources/openalex-entities.js";
import { mockHttp, noSleep } from "./helpers.js";

const urls = (http) => http.calls.map((c) => decodeURIComponent(c.url));

/** An OpenAlex source record. */
const source = (id, name, issns, extra = {}) => ({
  id: `https://openalex.org/S${id}`,
  display_name: name,
  issn_l: issns[0] ?? null,
  issn: issns,
  host_organization_name: "Elsevier BV",
  country_code: "NL",
  type: "journal",
  works_count: 20000,
  cited_by_count: 900000,
  summary_stats: { "2yr_mean_citedness": 5.04, h_index: 300 },
  topics: [
    { id: "https://openalex.org/T1", display_name: "Geochemistry and Geochronology", count: 50, subfield: { display_name: "Oceanography" }, field: { display_name: "Earth and Planetary Sciences" } },
    { id: "https://openalex.org/T2", display_name: "Isotope systematics", count: 500, subfield: { display_name: "Geochemistry and Petrology" }, field: { display_name: "Earth and Planetary Sciences" } },
  ],
  ...extra,
});

/** A memory storage (readJSON/writeJSON) recording writes. */
function memoryStorage(files = {}) {
  const store = new Map(Object.entries(files));
  return {
    writes: [],
    async readJSON(name) {
      return store.has(name) ? JSON.parse(JSON.stringify(store.get(name))) : null;
    },
    async writeJSON(name, value) {
      this.writes.push(name);
      store.set(name, JSON.parse(JSON.stringify(value)));
    },
    get: (name) => store.get(name),
  };
}

const SETTINGS = {
  journals: { autoTopN: 40, autoMinItems: 2 },
  apis: { openalexKey: "oa-key", mailto: "me@example.org" },
  sources: { crossref: { mailto: "me@example.org" } },
};

// ----------------------------------------------------------- mapping

test("OpenAlex sources map to whitelist journals: category, approximate IF, ISSNs and (CN)", () => {
  const j = mapOpenAlexSource(source(123, "Geochimica et  Cosmochimica Acta", ["0016-7037", "1872-9533"], { abbreviated_title: "Geochim. Cosmochim. Acta", alternate_titles: ["GCA"] }));
  assert.equal(j.openalexId, "S123");
  assert.equal(j.title, "Geochimica et Cosmochimica Acta");
  assert.deepEqual(j.issns, ["0016-7037", "1872-9533"]);
  assert.equal(j.impactFactor, 5, "2yr_mean_citedness rounded to one decimal");
  assert.equal(j.category, "GEOCHEMISTRY_AND_PETROLOGY", "the most frequent topic's subfield, upper-cased");
  assert.equal(j.subfield, "Geochemistry and Petrology");
  assert.equal(j.chinese, false);
  assert.equal(j.publisher, "Elsevier BV");
  assert.equal(j.worksCount, 20000);
  assert.ok(j.titles.includes("geochim cosmochim acta"), "abbreviation kept for exact-title matching");
  assert.ok(j.titles.includes("gca"));

  // issn_l fallback, field fallback, "&", missing metrics
  const k = mapOpenAlexSource({ id: "S9", display_name: "Earth & Space", issn: null, issn_l: "1234-5679", topics: [{ field: { display_name: "Earth & Planetary Sciences" } }], summary_stats: null });
  assert.deepEqual(k.issns, ["1234-5679"]);
  assert.equal(k.category, "EARTH_AND_PLANETARY_SCIENCES");
  assert.equal(k.impactFactor, null);
  assert.equal(categoryFromTopic(null), "");
  assert.equal(mapOpenAlexSource({ id: "S1", topics: "junk", summary_stats: { "2yr_mean_citedness": "x" } }), null, "no name");
  assert.equal(mapOpenAlexSource({ display_name: "No topics", topics: [null, 4], summary_stats: { "2yr_mean_citedness": "2.345" } }).impactFactor, 2.3);
  // A citedness of 0 (a journal younger than two years) is unknown, not an impact factor of 0
  assert.equal(mapOpenAlexSource({ display_name: "Brand New Journal", summary_stats: { "2yr_mean_citedness": 0 } }).impactFactor, null);
  assert.equal(mapOpenAlexSource({ display_name: "Barely Cited", summary_stats: { "2yr_mean_citedness": 0.02 } }).impactFactor, 0.1);
  for (const odd of [null, 4, "x", [], {}]) assert.equal(mapOpenAlexSource(odd), null);

  // Chinese-language journals published in China get "(CN)"; English ones from China do not
  const cjk = mapOpenAlexSource(source(5, "岩石学报", ["1000-0569"], { country_code: "CN" }));
  assert.equal(cjk.category, "GEOCHEMISTRY_AND_PETROLOGY (CN)");
  assert.equal(cjk.chinese, true);
  assert.equal(isChineseSource(source(6, "Acta Petrologica Sinica", ["1000-0569"], { country_code: "cn", host_organization_name: "Science Press" })), true);
  assert.equal(isChineseSource(source(6, "Chinese Journal of Geophysics", ["0001-5733"], { country_code: "CN", host_organization_name: null })), true);
  assert.equal(isChineseSource(source(6, "Earth Science Frontiers", ["1005-2321"], { country_code: "CN", alternate_titles: ["Diqiu Kexue Qianyuan"] })), true, "a pinyin alternate title marks it whoever publishes it");
  // English journals from China with "Sinica" / "Chinese" in their title are not marked
  for (const [name, host] of [
    ["Acta Geologica Sinica - English Edition", null],
    ["Acta Geologica Sinica (English Edition)", "Wiley"],
    ["Acta Oceanologica Sinica", "Springer Science+Business Media"],
    ["Chinese Geographical Science", "Springer Science+Business Media"],
    ["Chinese Chemical Letters", "Elsevier BV"],
    ["Acta Pharmacologica Sinica", "Nature Portfolio"],
  ]) {
    assert.equal(isChineseSource(source(6, name, ["1000-9515"], { country_code: "CN", host_organization_name: host })), false, name);
  }
  assert.equal(isChineseSource(source(6, "Acta Geologica Sinica - English Edition", ["1000-9515"], { country_code: "CN", alternate_titles: ["地质学报"] })), false, "an English edition stays English");
  assert.equal(isChineseSource(source(7, "Geological Review", ["0371-5736"], { country_code: "CN", alternate_titles: ["地质论评"] })), true);
  assert.equal(isChineseSource(source(8, "Science China Earth Sciences", ["1674-7313"], { country_code: "CN" })), false);
  assert.equal(isChineseSource(source(9, "Acta Geologica Sinica", ["1000-9515"], { country_code: "US" })), false, "only journals published in China");
  assert.equal(mapOpenAlexSource({ display_name: "地质学报", country_code: "CN" }).category, "(CN)", "no topic: the mark alone");
});

test("OpenAlex authors map to search results", () => {
  const a = mapOpenAlexAuthor({
    id: "https://openalex.org/A5023888391",
    display_name: "Cin-Ty  Lee",
    orcid: "https://orcid.org/0000-0002-1825-009x",
    works_count: 250,
    cited_by_count: 18000,
    last_known_institutions: [{ display_name: "Rice University", country_code: "us" }],
    topics: [{ display_name: "Arc magmatism", count: 40 }],
    summary_stats: { h_index: 70 },
  });
  assert.deepEqual(a, {
    id: "A5023888391",
    name: "Cin-Ty Lee",
    orcid: "0000-0002-1825-009X",
    institution: "Rice University",
    country: "US",
    worksCount: 250,
    citedByCount: 18000,
    hIndex: 70,
    topic: "Arc magmatism",
  });
  const bare = mapOpenAlexAuthor({ id: "A1234", display_name: "No Orcid", last_known_institutions: null, topics: null });
  assert.equal(bare.orcid, null);
  assert.equal(bare.institution, null);
  for (const odd of [null, {}, { id: "A1234" }, { display_name: "x" }]) assert.equal(mapOpenAlexAuthor(odd), null);
});

// ----------------------------------------------------------- requests

test("ISSN lookups are batched, keyed, rate limited and keep the larger duplicate", async () => {
  const valid = Array.from({ length: 60 }, (_, i) => `1000-${1000 + i}`);
  const acquired = [];
  const http = mockHttp([
    ["https://api.openalex.org/sources", (req) => {
      const filter = new URL(req.url).searchParams.get("filter");
      const asked = filter.replace(/^issn:/, "").split("|");
      if (asked.includes("0016-7037")) {
        return {
          results: [
            source(1, "Small duplicate", ["0016-7037"], { works_count: 3 }),
            source(2, "Geochimica et Cosmochimica Acta", ["0016-7037", "1872-9533"]),
            null,
            "junk",
            source(3, "Not asked", ["9999-9999"]),
          ],
        };
      }
      return { results: null };
    }],
  ]);
  http.limiter = {
    async acquire(url) {
      acquired.push(url);
      return () => {};
    },
    pause() {},
  };
  const found = await lookupSourcesByIssn(http, ["0016-7037", "bad", ...valid, "0016-7037"], { apiKey: "oa-key", mailto: "me@example.org", retry: noSleep });
  assert.equal(http.calls.length, 2, "61 distinct ISSNs in batches of 50");
  assert.equal(ISSN_BATCH_SIZE, 50);
  const first = new URL(http.calls[0].url).searchParams;
  assert.equal(first.get("api_key"), "oa-key");
  assert.equal(first.get("mailto"), "me@example.org");
  assert.equal(first.get("filter").split("|").length, 50);
  assert.match(first.get("select"), /summary_stats/);
  assert.equal(acquired.length, 2, "every request waits for the limiter");
  assert.equal(found.get("0016-7037").title, "Geochimica et Cosmochimica Acta", "the record with more works wins");
  assert.equal(found.has("1872-9533"), false, "only requested ISSNs are keys");
  assert.equal(found.has("9999-9999"), false);
  assert.equal((await lookupSourcesByIssn(http, [], {})).size, 0);
  assert.equal(http.calls.length, 2, "nothing to look up: no request");
});

test("an unknown select field falls back to full records once; other errors throw", async () => {
  const http = mockHttp([
    [(req) => req.url.includes("select="), () => ({ status: 400, data: { error: "Invalid select field" } })],
    ["https://api.openalex.org/sources", () => ({ results: [source(1, "GCA", ["0016-7037"])] })],
  ]);
  const found = await lookupSourcesByIssn(http, ["0016-7037"], { retry: noSleep });
  assert.equal(found.get("0016-7037").title, "GCA");
  assert.equal(http.calls.length, 2);
  assert.doesNotMatch(http.calls[1].url, /select=/);

  const failing = mockHttp([["https://api.openalex.org/", () => ({ status: 429, data: {} })]]);
  await assert.rejects(searchSources(failing, "geochemistry", { retry: { retries: 0, ...noSleep } }), (e) => e.status === 429);
  assert.match(openAlexErrorText({ status: 429 }), /免费额度已用完.*API Key/);
  assert.match(openAlexErrorText({ status: 429 }, { keyed: true }), /今日额度已用完/);
  assert.match(openAlexErrorText({ status: 403 }), /Key 无效/);
  assert.equal(openAlexErrorText(new Error("网络连接失败：api.openalex.org")), "网络连接失败：api.openalex.org");
});

test("source search: by name with the journal filter, by ISSN with the ISSN filter, exact-title matching", async () => {
  const http = mockHttp([
    ["https://api.openalex.org/sources", (req) => {
      const p = new URL(req.url).searchParams;
      if (p.get("filter") === "issn:0016-7037") return { results: [source(2, "Geochimica et Cosmochimica Acta", ["0016-7037"])] };
      if (p.get("search") === "Geochim. Cosmochim. Acta") {
        return { results: [source(4, "Geochimica et Cosmochimica Acta Supplement", ["1111-1119"]), source(2, "Geochimica et Cosmochimica Acta", ["0016-7037"], { abbreviated_title: "Geochim. Cosmochim. Acta" })] };
      }
      return { results: [source(2, "Geochimica et Cosmochimica Acta", ["0016-7037"]), source(2, "Geochimica et Cosmochimica Acta", ["0016-7037"]), source(3, "Chemical Geology", ["0009-2541"])] };
    }],
  ]);
  assert.equal(looksLikeIssn("0016 703x"), true);
  assert.equal(looksLikeIssn("geochem 2024"), false);
  const byName = await searchSources(http, "  geochimica  ", { limit: 99 });
  let p = new URL(http.calls[0].url).searchParams;
  assert.equal(p.get("search"), "geochimica");
  assert.equal(p.get("filter"), "type:journal");
  assert.equal(p.get("per-page"), "25", "limit capped");
  assert.deepEqual(byName.map((j) => j.title), ["Geochimica et Cosmochimica Acta", "Chemical Geology"], "duplicates dropped");
  const byIssn = await searchSources(http, "0016-7037");
  p = new URL(http.calls[1].url).searchParams;
  assert.equal(p.get("filter"), "issn:0016-7037");
  assert.equal(p.get("search"), null);
  assert.equal(byIssn[0].openalexId, "S2");
  assert.deepEqual(await searchSources(http, "   "), []);
  assert.equal(http.calls.length, 2, "a blank query sends nothing");

  // Exact title (any normalisation of case, punctuation, a leading "The"), via the abbreviation too
  assert.equal((await findSourceByTitle(http, "Geochim. Cosmochim. Acta")).openalexId, "S2");
  assert.equal((await findSourceByTitle(http, "the geochimica et cosmochimica acta")).openalexId, "S2");
  assert.equal(await findSourceByTitle(http, "Geochimica"), null, "a prefix is no match");
});

test("author search: names, ORCID iDs and OpenAlex ids", async () => {
  const author = (id, name, orcid = null) => ({ id: `https://openalex.org/A${id}`, display_name: name, orcid, works_count: 10 });
  const http = mockHttp([
    ["https://api.openalex.org/authors/A5023888391", () => author(5023888391, "Cin-Ty Lee")],
    ["https://api.openalex.org/authors/A9999999", () => ({ status: 404, data: { error: "not found" } })],
    ["https://api.openalex.org/authors", (req) => {
      const p = new URL(req.url).searchParams;
      if (p.get("filter")) return { results: [author(1234567, "Jay J. Ague", "https://orcid.org/0000-0001-8408-0279")] };
      return { results: [author(111111, "Zaicong Wang"), author(111111, "Zaicong Wang"), { id: null }, author(222222, "Z. Wang")] };
    }],
  ]);
  const byName = await searchAuthors(http, "Zaicong Wang", { apiKey: "oa-key" });
  assert.deepEqual(byName.map((a) => a.id), ["A111111", "A222222"]);
  let p = new URL(http.calls[0].url).searchParams;
  assert.equal(p.get("search"), "Zaicong Wang");
  assert.equal(p.get("api_key"), "oa-key");
  assert.match(p.get("select"), /last_known_institutions/);

  const byOrcid = await searchAuthors(http, "https://orcid.org/0000-0001-8408-0279");
  p = new URL(http.calls[1].url).searchParams;
  assert.equal(p.get("filter"), "orcid:https://orcid.org/0000-0001-8408-0279");
  assert.equal(byOrcid[0].orcid, "0000-0001-8408-0279");

  const byId = await searchAuthors(http, "https://openalex.org/A5023888391");
  assert.equal(byId[0].name, "Cin-Ty Lee");
  assert.deepEqual(await searchAuthors(http, "A9999999", { retry: noSleep }), [], "an unknown id is no result, not an error");
});

test("a cancelled signal stops the lookups", async () => {
  const signal = createCancelSignal();
  const http = mockHttp([["https://api.openalex.org/", () => ({ results: [] })]]);
  http.signal = signal;
  signal.cancel();
  await assert.rejects(lookupSourcesByIssn(http, ["0016-7037"]), (e) => e.cancelled === true);
  await assert.rejects(searchSources(http, "geo"), (e) => e.cancelled === true);
  assert.equal(http.calls.length, 0);
});

// ----------------------------------------------------------- whitelist merge

test("whitelist rows merge without overwriting values; library:N counts as an empty category", () => {
  assert.equal(isEmptyCategory("library:14"), true);
  assert.equal(isEmptyCategory(" "), true);
  assert.equal(isEmptyCategory("GEOSCIENCES_CN"), false);
  assert.equal(isIncompleteJournal({ title: "X", category: "GEO", impactFactor: 1 }), false);
  assert.equal(isIncompleteJournal({ title: "X", category: "library:3", impactFactor: 1 }), true);
  assert.equal(isIncompleteJournal({ title: "X", category: "GEO", impactFactor: null }), true);

  const text = [
    "issn,title,category,impact_factor",
    "# my own list",
    "1674-7313,Science China Earth Sciences,library:14,",
    "0016-7037,GCA,GEOCHEM,5.0",
    "0012-821X,,GEO,",
    "",
  ].join("\n");
  assert.deepEqual(summarizeWhitelist(text).incomplete, 2);
  assert.deepEqual(whitelistIssns(text), ["1674-7313", "0016-7037", "0012-821X"]);
  const rows = journalRows([
    { title: "Science China Earth Sciences", issns: ["1674-7313", "1869-1897"], category: "GEOLOGY", impactFactor: 4.8 },
    { title: "Geochimica et Cosmochimica Acta", issns: ["0016-7037"], category: "GEOCHEMISTRY_AND_PETROLOGY", impactFactor: 4.9 },
    { title: "Earth and Planetary Science Letters", issns: ["0012-821x", "bad"], category: "GEOPHYSICS", impactFactor: 4.6 },
    { title: "Earth, Planets and Space", issns: ["1880-5981"], category: "", impactFactor: null },
  ]);
  assert.deepEqual(rows.map((r) => r.issn), ["1674-7313", "1869-1897", "0016-7037", "0012-821X", "1880-5981"]);
  const merged = mergeWhitelistRows(text, rows);
  assert.deepEqual(merged.text.split("\n"), [
    "issn,title,category,impact_factor",
    "# my own list",
    "1674-7313,Science China Earth Sciences,GEOLOGY,4.8",
    "0016-7037,GCA,GEOCHEM,5.0",
    "0012-821X,Earth and Planetary Science Letters,GEO,4.6",
    "1869-1897,Science China Earth Sciences,GEOLOGY,4.8",
    '1880-5981,"Earth, Planets and Space",,',
  ]);
  assert.equal(merged.added, 2);
  assert.equal(merged.filled, 2, "the complete GCA row is untouched");
  const map = parseWhitelist(merged.text);
  assert.equal(map.get("1880-5981").title, "Earth, Planets and Space");

  // Fill only, and an empty list gets a header
  const fillOnly = mergeWhitelistRows(text, rows, { append: false });
  assert.equal(fillOnly.added, 0);
  assert.equal(fillOnly.text.split("\n").length, 6, "trailing blank kept when nothing is appended");
  assert.deepEqual(mergeWhitelistRows("", rows.slice(0, 1)).text.split("\n"), ["issn,title,category,impact_factor", "1674-7313,Science China Earth Sciences,GEOLOGY,4.8"]);
  assert.deepEqual(mergeWhitelistRows(text, []), { text, added: 0, filled: 0 });
});

// ----------------------------------------------------------- LLM prompt

test("the journal prompt is the Python one and replies are parsed or salvaged", () => {
  const prompt = journalGenerationPrompt([{ title: "Geochim. $& Acta", count: 14 }], "斑岩铜矿成矿、俯冲带流体");
  assert.match(prompt, /^你是一位资深的学术期刊专家/);
  assert.match(prompt, /- Geochim\. \$& Acta \[journal\]: 14 篇/, "titles are inserted literally");
  assert.match(prompt, /斑岩铜矿成矿、俯冲带流体/);
  assert.match(prompt, /"journals": \[\n {4}\{"title": "Nature Geoscience"/);
  assert.match(journalGenerationPrompt([], ""), /未提供/);

  assert.deepEqual(parseJournalReply('```json\n{"journals": [{"title": " Lithos ", "category": "geochemistry & petrology", "impact_factor": "3.5", "is_chinese": false}, {"title": "lithos"}, {"title": ""}, 4]}\n```'), [
    { title: "Lithos", category: "GEOCHEMISTRY_PETROLOGY", impactFactor: 3.5, isChinese: false },
  ]);
  // Prose around a truncated reply: the complete entries survive
  const truncated = '好的，以下是名单：{"journals": [{"title": "岩石学报", "category": "GEOLOGY", "impact_factor": null, "is_chinese": true}, {"title": "Chemical Geology", "impact_factor": 3.9}, {"title": "Litho';
  assert.deepEqual(parseJournalReply(truncated).map((j) => [j.title, j.isChinese, j.impactFactor]), [
    ["岩石学报", true, null],
    ["Chemical Geology", false, 3.9],
  ]);
  assert.deepEqual(parseJournalReply("no json here"), []);
  assert.deepEqual(parseJournalReply(null), []);
  assert.equal(extractJsonObjects('{"a": "}{", "b": {"c": 1}}').length, 2, "braces inside strings are ignored");
});

// ----------------------------------------------------------- proposals

function libraryItems() {
  const item = (venue, issn = "", n = 1) => Array.from({ length: n }, () => ({ itemType: "journalArticle", venue, issn }));
  return [
    ...item("Geochimica et Cosmochimica Acta", "0016-7037", 6),
    ...item("Geochim. Cosmochim. Acta", "", 1), // the same journal under its abbreviation
    ...item("Earth and Planetary Science Letters", "", 3), // ISSN from Crossref
    ...item("Acta Petrologica Sinica", "", 2), // no ISSN anywhere but OpenAlex finds the title
    ...item("Unknown Society Newsletter", "", 2), // nowhere
    ...item("Lithos", "0024-4937", 1), // below autoMinItems: proposed unchecked
    ...item("Precambrian Res.", "", 1), // OpenAlex knows only the full name: the model's normalisation finds it
    { itemType: "book", venue: "A Book Series" },
  ];
}

function proposalHttp({ openalex = "ok", llm = null } = {}) {
  const sources = {
    "0016-7037": source(1, "Geochimica et Cosmochimica Acta", ["0016-7037", "1872-9533"], { abbreviated_title: "Geochim. Cosmochim. Acta" }),
    "0012-821X": source(2, "Earth and Planetary Science Letters", ["0012-821X", "1385-013X"], { summary_stats: { "2yr_mean_citedness": 4.62 } }),
    "0024-4937": source(3, "Lithos", ["0024-4937"], { summary_stats: { "2yr_mean_citedness": 1.93 } }),
  };
  const searches = {
    "Acta Petrologica Sinica": [source(4, "Acta Petrologica Sinica", ["1000-0569"], { country_code: "CN", host_organization_name: "Science Press", summary_stats: { "2yr_mean_citedness": 1.2 } })],
    "Chemical Geology": [source(5, "Chemical Geology", ["0009-2541", "1872-6836"], { summary_stats: { "2yr_mean_citedness": 3.94 } })],
    "Journal of Petrology": [source(6, "Journal of Petrology", ["0022-3530"], { publisher: "OUP", summary_stats: { "2yr_mean_citedness": 1.6 } })],
    "Earth and Planetary Science Letters": [sources["0012-821X"]],
    "Geochim. Cosmochim. Acta": [sources["0016-7037"]],
    "Precambrian Research": [source(7, "Precambrian Research", ["0301-9268"], { summary_stats: { "2yr_mean_citedness": 3.2 } })],
  };
  return mockHttp([
    ["https://api.crossref.org/journals", (req) => {
      const q = new URL(req.url).searchParams.get("query");
      return { message: { items: q === "Earth and Planetary Science Letters" ? [{ title: q, ISSN: ["0012-821X"] }] : [] } };
    }],
    ["https://api.openalex.org/sources", (req) => {
      if (openalex === "429") return { status: 429, data: { error: "Too many" } };
      const p = new URL(req.url).searchParams;
      if (p.get("filter")?.startsWith("issn:") && !p.get("search")) {
        return { results: p.get("filter").slice(5).split("|").map((issn) => sources[issn]).filter(Boolean) };
      }
      return { results: searches[p.get("search")] || [] };
    }],
    ["https://llm.example/v1/chat/completions", () => (typeof llm === "function" ? llm() : { choices: [{ message: { content: llm } }] })],
  ]);
}

const llmClient = (http) => ({
  async chat(prompt, opts) {
    const res = await http.request({ method: "POST", url: "https://llm.example/v1/chat/completions", body: JSON.stringify({ prompt, opts }) });
    if (res.status !== 200) throw new Error(`HTTP ${res.status}`);
    return res.data.choices[0].message.content;
  },
});

test("从文库生成 proposes library journals enriched from OpenAlex, without writing anything", async () => {
  const http = proposalHttp();
  const storage = memoryStorage();
  const whitelist = "issn,title,category,impact_factor\n0024-4937,Lithos,GEO,2.0";
  const { proposals, warnings, missing, stats } = await proposeJournals({ http, storage, items: libraryItems(), whitelist, settings: SETTINGS, retry: noSleep });
  assert.deepEqual(warnings, []);
  assert.deepEqual(missing, ["Unknown Society Newsletter", "Precambrian Res."], "only the model could name it (see the next test)");
  assert.deepEqual(
    proposals.map((p) => [p.title, p.libraryCount, p.issns.join(" "), p.category, p.impactFactor, p.origin, p.inList, p.checked]),
    [
      ["Geochimica et Cosmochimica Acta", 7, "0016-7037 1872-9533", "GEOCHEMISTRY_AND_PETROLOGY", 5, "library", false, true],
      ["Earth and Planetary Science Letters", 3, "0012-821X 1385-013X", "GEOCHEMISTRY_AND_PETROLOGY", 4.6, "library", false, true],
      ["Acta Petrologica Sinica", 2, "1000-0569", "GEOCHEMISTRY_AND_PETROLOGY (CN)", 1.2, "library", false, true],
      ["Lithos", 1, "0024-4937", "GEOCHEMISTRY_AND_PETROLOGY", 1.9, "library", true, false],
    ],
  );
  assert.equal(proposals[0].verified, true);
  assert.equal(proposals[0].openalexId, "S1");
  assert.equal(stats.enriched, 4);
  assert.equal(stats.venues, 7);
  // The Crossref title lookup is cached as before (shared with the watch run)
  assert.deepEqual(storage.writes, ["journals-cache.json"]);
  assert.deepEqual(storage.get("journals-cache.json").titles["earth and planetary science letters"].issns, ["0012-821X"]);
  assert.ok(urls(http).filter((u) => u.includes("api.openalex.org")).every((u) => u.includes("api_key=oa-key")));
  // Venues with an ISSN are never searched by title; the abbreviation merged into GCA by ISSN-less title search
  assert.equal(urls(http).filter((u) => u.includes("search=")).length, 4);
});

test("with the LLM on, suggestions are verified through OpenAlex and take its impact factor", async () => {
  const reply = JSON.stringify({
    journals: [
      { title: "Geochimica et Cosmochimica Acta", category: "GEOCHEMISTRY", impact_factor: 4.5, is_chinese: false },
      { title: "Acta Petrologica Sinica", category: "GEOLOGY", impact_factor: 1.0, is_chinese: true },
      { title: "Chemical Geology", category: "GEOCHEMISTRY", impact_factor: 99, is_chinese: false },
      { title: "Journal of Petrology", category: "PETROLOGY", impact_factor: 4.0, is_chinese: false },
      { title: "Made-up Journal of Everything", category: "GENERAL", impact_factor: 50, is_chinese: false },
      { title: "Precambrian Research", category: "GEOLOGY", impact_factor: 3.0, is_chinese: false },
    ],
  });
  const http = proposalHttp({ llm: reply });
  const { proposals, stats, missing } = await proposeJournals({ http, storage: memoryStorage(), items: libraryItems(), settings: SETTINGS, llm: llmClient(http), researchFocus: "斑岩铜矿成矿", retry: noSleep });
  // The model's "Precambrian Research" is the library's "Precambrian Res.": a library proposal
  const precambrian = proposals.find((p) => p.title === "Precambrian Research");
  assert.deepEqual([precambrian.origin, precambrian.libraryCount, precambrian.libraryTitle, precambrian.issns], ["library", 1, "Precambrian Res.", ["0301-9268"]]);
  assert.deepEqual(missing, ["Unknown Society Newsletter"]);
  const ai = proposals.filter((p) => p.origin === "ai");
  assert.deepEqual(ai.map((p) => [p.title, p.impactFactor, p.checked, p.libraryCount]), [
    ["Chemical Geology", 3.9, true, 0],
    ["Journal of Petrology", 1.6, false, 0],
  ], "IF from OpenAlex (never the model's); checked from IF >= 2");
  assert.equal(AI_DEFAULT_MIN_IF, 2);
  assert.equal(proposals.find((p) => p.title === "Acta Petrologica Sinica").category, "GEOCHEMISTRY_AND_PETROLOGY (CN)", "already marked, not twice");
  assert.deepEqual(stats, { venues: 7, enriched: 5, aiSuggested: 6, aiAdded: 2, aiUnverified: 1, aiSkipped: 0 });
  assert.equal(abbreviates("Geochim. Cosmochim. Acta", "Geochimica et Cosmochimica Acta"), true);
  assert.equal(abbreviates("J. Geophys. Res.", "Journal of Geophysical Research"), true);
  assert.equal(abbreviates("J. Geol.", "Journal of Geology and Geophysics"), false);
  assert.equal(abbreviates("", "Lithos"), false);
  const prompt = JSON.parse(http.calls.find((c) => c.url.startsWith("https://llm.example")).body).prompt;
  assert.match(prompt, /- Geochimica et Cosmochimica Acta \[journal\]: 6 篇/);
  assert.match(prompt, /斑岩铜矿成矿/);
  // Library journals are not searched again
  assert.equal(urls(http).filter((u) => /search=Geochimica/.test(u)).length, 0);
  assert.equal(withChineseMark("GEO"), "GEO (CN)");
  assert.equal(withChineseMark("GEOSCIENCES_CN"), "GEOSCIENCES_CN");
});

test("OpenAlex and LLM failures are isolated: the library proposals still come back", async () => {
  // OpenAlex refuses everything: library journals without category / IF, a warning, no AI suggestions
  const reply = JSON.stringify({ journals: [{ title: "Chemical Geology", is_chinese: false }] });
  const http = proposalHttp({ openalex: "429", llm: reply });
  const result = await proposeJournals({ http, storage: memoryStorage(), items: libraryItems(), settings: { ...SETTINGS, apis: { openalexKey: "", mailto: "" } }, llm: llmClient(http), retry: { retries: 0, ...noSleep } });
  assert.deepEqual(result.proposals.map((p) => [p.title, p.category, p.impactFactor, p.verified]), [
    ["Geochimica et Cosmochimica Acta", "", null, false],
    ["Earth and Planetary Science Letters", "", null, false],
    ["Lithos", "", null, false],
  ]);
  assert.equal(result.warnings.length, 2);
  assert.match(result.warnings[0], /OpenAlex 查询失败.*免费额度已用完/);
  assert.match(result.warnings[1], /需要通过 OpenAlex 核实/);
  assert.deepEqual(result.missing, ["Acta Petrologica Sinica", "Unknown Society Newsletter", "Geochim. Cosmochim. Acta", "Precambrian Res."], "in library order");
  assert.equal(urls(http).filter((u) => u.includes("api.openalex.org")).length, 1, "the first failure stops further OpenAlex lookups");

  // The model fails: proposals without AI suggestions and a warning
  const http2 = proposalHttp({ llm: () => ({ status: 401, data: { error: "bad key" } }) });
  const failedLlm = await proposeJournals({ http: http2, storage: memoryStorage(), items: libraryItems(), settings: SETTINGS, llm: llmClient(http2), retry: noSleep });
  assert.equal(failedLlm.proposals.length, 4);
  assert.deepEqual(failedLlm.warnings, ["大模型推荐失败：HTTP 401"]);
});

test("proposals can be cancelled; nothing is cached then", async () => {
  const signal = createCancelSignal();
  const http = proposalHttp();
  const request = http.request;
  http.signal = signal;
  http.request = async (req) => {
    if (req.url.includes("api.openalex.org")) signal.cancel();
    signal.throwIfCancelled();
    return request(req);
  };
  const storage = memoryStorage();
  await assert.rejects(proposeJournals({ http, storage, items: libraryItems(), settings: SETTINGS, retry: noSleep }), (e) => e.cancelled === true);

  // Cancelled during the Crossref title lookups: the cache is not written
  const signal2 = createCancelSignal();
  const http2 = proposalHttp();
  http2.signal = signal2;
  const request2 = http2.request;
  http2.request = async (req) => {
    if (req.url.includes("crossref")) signal2.cancel();
    signal2.throwIfCancelled();
    return request2(req);
  };
  const storage2 = memoryStorage();
  await assert.rejects(resolveVenueIssns(http2, storage2, [{ title: "EPSL", issns: [] }], { retry: noSleep }), (e) => e.cancelled === true);
  assert.deepEqual(storage2.writes, []);
});

test("补全学科与影响因子 fills blanks from OpenAlex and never overwrites", async () => {
  const text = [
    "issn,title,category,impact_factor",
    "1674-7313,Science China Earth Sciences,library:14,",
    "0016-7037,GCA,GEOCHEM,5.0",
    "0012-821X,EPSL,MY_OWN,",
    "9999-9999,Unknown,library:2,",
  ].join("\n");
  const http = mockHttp([
    ["https://api.openalex.org/sources", (req) => {
      const asked = new URL(req.url).searchParams.get("filter").slice(5).split("|");
      assert.deepEqual(asked, ["1674-7313", "0012-821X", "9999-9999"], "only incomplete rows are looked up");
      return {
        results: [
          source(1, "Science China Earth Sciences", ["1674-7313", "1869-1897"], { country_code: "CN", topics: [{ subfield: { display_name: "Geology" }, count: 5 }], summary_stats: { "2yr_mean_citedness": 4.84 } }),
          source(2, "Earth and Planetary Science Letters", ["0012-821X"], { summary_stats: { "2yr_mean_citedness": 4.62 } }),
        ],
      };
    }],
  ]);
  const result = await fillJournalMetadata({ http, text, settings: SETTINGS, retry: noSleep });
  assert.deepEqual(result.text.split("\n"), [
    "issn,title,category,impact_factor",
    "1674-7313,Science China Earth Sciences,GEOLOGY,4.8",
    "0016-7037,GCA,GEOCHEM,5.0",
    "0012-821X,EPSL,MY_OWN,4.6",
    "9999-9999,Unknown,library:2,",
  ]);
  assert.equal(result.filled, 2);
  assert.equal(result.incomplete, 3);
  assert.deepEqual(result.notFound, ["9999-9999"]);

  const complete = "issn,title,category,impact_factor\n0016-7037,GCA,GEOCHEM,5.0";
  const none = await fillJournalMetadata({ http, text: complete, settings: SETTINGS });
  assert.deepEqual(none, { text: complete, filled: 0, incomplete: 0, notFound: [] });
  assert.equal(http.calls.length, 1, "a complete list needs no request");

  // A citedness of 0 leaves the IF blank (the journal keeps the neutral score; a later 补全 fills it)
  const fresh = mockHttp([["https://api.openalex.org/sources", () => ({ results: [source(2, "Earth and Planetary Science Letters", ["0012-821X"], { summary_stats: { "2yr_mean_citedness": 0 } })] })]]);
  const zero = await fillJournalMetadata({ http: fresh, text: "issn,title,category,impact_factor\n0012-821X,EPSL,,", settings: SETTINGS, retry: noSleep });
  assert.equal(zero.text.split("\n")[1], "0012-821X,EPSL,GEOCHEMISTRY_AND_PETROLOGY,");

  const failing = mockHttp([["https://api.openalex.org/", () => ({ status: 503, data: "down" })]]);
  await assert.rejects(fillJournalMetadata({ http: failing, text, settings: SETTINGS, retry: { retries: 0, ...noSleep } }), (e) => e.status === 503);
});
