// Candidate sources of the topic survey: OpenAlex works (search, citing works,
// references, seed lookup), Semantic Scholar (search, references, citations,
// recommendations, batch lookup) and Crossref (bibliographic search), all
// mapped to one paper shape. Every request goes through core/http.js (per-host
// rate limiting, retries, cancellation) with the services' keys. Responses are
// parsed defensively: odd records are skipped, never thrown on; a failed
// request throws like the other sources (the survey engine records it as a
// non-fatal error of that source).

import { makeAuthor, parseAuthorName } from "../core/authors.js";
import { buildUrl, getJson } from "../core/http.js";
import { cleanHtml, cleanTitle, collapseWhitespace, isNonArticleTitle, normalizeDoi, normalizeIssn } from "../core/text.js";
import { openAlexAuth, reconstructAbstract } from "../sources/openalex.js";
import { mapS2Paper, s2Batch, s2Citations, s2DoiId, s2Recommendations, s2References, s2Search } from "../sources/semanticscholar.js";

const OPENALEX_WORKS = "https://api.openalex.org/works";
const CROSSREF_WORKS = "https://api.crossref.org/works";
/** Work fields requested from OpenAlex (an unknown field would fail the query). */
export const OPENALEX_WORK_SELECT =
  "id,doi,title,display_name,publication_year,publication_date,type,authorships,primary_location,abstract_inverted_index,cited_by_count,is_paratext";
// Seeds additionally need their reference lists
const SEED_SELECT = `${OPENALEX_WORK_SELECT},referenced_works`;
// Semantic Scholar fields of searches and citation edges (TLDRs come from the batch lookup of stage 6)
export const S2_SURVEY_FIELDS = "paperId,externalIds,title,abstract,year,publicationDate,venue,journal,authors,citationCount";
export const S2_TLDR_FIELDS = "paperId,externalIds,abstract,tldr";
const CROSSREF_SELECT = "DOI,title,author,abstract,container-title,issued,published,created,URL,type,is-referenced-by-count,ISSN";
const OPENALEX_ID_BATCH = 50;
const MAX_AUTHORS = 12;
// Work types kept: research articles, reviews and preprints
const OPENALEX_TYPES = new Set(["article", "review", "preprint", "letter"]);
const CROSSREF_TYPES = new Set(["journal-article", "posted-content", "proceedings-article", "review-article"]);
const REQUEST_TIMEOUT = 30000;

const isRecord = (value) => !!value && typeof value === "object" && !Array.isArray(value);
const resultsOf = (data) => (Array.isArray(data?.results) ? data.results : []);

/** OpenAlex work id ("W123") from its URL form; "" when absent. */
export function openAlexWorkId(value) {
  const m = String(value ?? "").match(/\bW\d+\b/i);
  return m ? m[0].toUpperCase() : "";
}

function isoDate(value) {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : null;
}

function validYear(value) {
  const n = Number(value);
  return Number.isInteger(n) && n > 1000 && n < 3000 ? n : null;
}

/** Paper id: DOI first, then the service ids. */
function paperId({ doi, openalexId, s2Id }) {
  if (doi) return `doi:${doi}`;
  if (openalexId) return `openalex:${openalexId}`;
  return s2Id ? `s2:${s2Id}` : "";
}

/**
 * Map an OpenAlex work to a survey paper; null for paratext, unkept types,
 * records without a title or without any identifier.
 */
export function mapOpenAlexPaper(work) {
  if (!isRecord(work) || work.is_paratext === true) return null;
  if (work.type && !OPENALEX_TYPES.has(work.type)) return null;
  const location = isRecord(work.primary_location) ? work.primary_location : {};
  const source = isRecord(location.source) ? location.source : null;
  const venue = typeof source?.display_name === "string" ? collapseWhitespace(source.display_name) || null : null;
  const title = cleanTitle([work.display_name, work.title].find((t) => typeof t === "string" && t.trim()) || "");
  if (!title || isNonArticleTitle(title, venue)) return null;
  const doi = normalizeDoi(work.doi) || null;
  const openalexId = openAlexWorkId(work.id) || null;
  if (!doi && !openalexId) return null;
  const authorships = Array.isArray(work.authorships) ? work.authorships.filter(isRecord) : [];
  const authors = authorships
    .slice(0, MAX_AUTHORS)
    .map((a) => parseAuthorName(a.author?.display_name || a.raw_author_name))
    .filter(Boolean);
  const landing = typeof location.landing_page_url === "string" && /^https?:\/\//i.test(location.landing_page_url) ? location.landing_page_url : null;
  const published = isoDate(work.publication_date);
  const citations = Number(work.cited_by_count);
  return {
    id: paperId({ doi, openalexId }),
    title,
    abstract: cleanHtml(reconstructAbstract(work.abstract_inverted_index)),
    tldr: null,
    authors,
    authorCount: authorships.length,
    year: validYear(work.publication_year) ?? (published ? Number(published.slice(0, 4)) : null),
    published,
    venue,
    issns: Array.isArray(source?.issn) ? [...new Set(source.issn.map(normalizeIssn).filter(Boolean))] : [],
    doi,
    url: doi ? `https://doi.org/${doi}` : landing || `https://openalex.org/${openalexId}`,
    type: typeof work.type === "string" ? work.type : null,
    citations: Number.isFinite(citations) && citations >= 0 ? citations : 0,
    openalexId,
    s2Id: null,
    sources: ["openalex"],
  };
}

/** Map a raw Semantic Scholar paper to a survey paper (see mapS2Paper); null when unusable. */
export function mapS2SurveyPaper(raw) {
  const p = mapS2Paper(raw);
  if (!p) return null;
  return {
    id: paperId({ doi: p.doi, s2Id: p.s2Id }),
    title: p.title,
    abstract: p.abstract,
    tldr: p.tldr,
    authors: p.authors.slice(0, MAX_AUTHORS),
    authorCount: p.authors.length,
    year: p.year,
    published: p.published,
    venue: p.venue,
    issns: [],
    doi: p.doi,
    url: p.url,
    type: null,
    citations: p.citations,
    openalexId: null,
    s2Id: p.s2Id,
    ...(p.arxivId ? { arxivId: p.arxivId } : {}),
    sources: ["s2"],
  };
}

/** Map a Crossref work to a survey paper; null for non-article records. */
export function mapCrossrefPaper(item) {
  if (!isRecord(item)) return null;
  if (item.type && !CROSSREF_TYPES.has(item.type)) return null;
  const rawTitle = Array.isArray(item.title) ? item.title[0] : item.title;
  const title = cleanTitle(typeof rawTitle === "string" ? rawTitle : "");
  const rawVenue = Array.isArray(item["container-title"]) ? item["container-title"][0] : null;
  const venue = typeof rawVenue === "string" ? rawVenue : null;
  if (!title || isNonArticleTitle(title, venue)) return null;
  const doi = normalizeDoi(item.DOI) || null;
  if (!doi) return null;
  const authors = (Array.isArray(item.author) ? item.author : [])
    .filter(isRecord)
    .slice(0, MAX_AUTHORS)
    .map((a) => (a.family || a.given ? makeAuthor(a.given, a.family) : typeof a.name === "string" ? { given: "", family: a.name, name: a.name } : null))
    .filter((a) => a && a.name);
  const parts = (key) => (Array.isArray(item[key]?.["date-parts"]?.[0]) ? item[key]["date-parts"][0] : null);
  const date = parts("issued") || parts("published") || parts("created");
  const year = date ? validYear(date[0]) : null;
  const pad = (n) => String(n).padStart(2, "0");
  const published = date && year ? `${year}-${pad(date[1] || 1)}-${pad(date[2] || 1)}T00:00:00Z` : null;
  const citations = Number(item["is-referenced-by-count"]);
  return {
    id: `doi:${doi}`,
    title,
    abstract: cleanHtml(item.abstract),
    tldr: null,
    authors,
    authorCount: Array.isArray(item.author) ? item.author.length : authors.length,
    year,
    published,
    venue: venue ? collapseWhitespace(venue) : null,
    issns: (Array.isArray(item.ISSN) ? item.ISSN : []).map(normalizeIssn).filter(Boolean),
    doi,
    url: `https://doi.org/${doi}`,
    type: typeof item.type === "string" ? item.type : null,
    citations: Number.isFinite(citations) && citations >= 0 ? citations : 0,
    openalexId: null,
    s2Id: null,
    sources: ["crossref"],
  };
}

const mapAll = (records, map) => records.map(map).filter(Boolean);

// ---------------------------------------------------------------- OpenAlex

function openAlexFilter({ yearFrom = null, extra = [] } = {}) {
  return [...extra, "type:article|review|preprint", ...(yearFrom ? [`from_publication_date:${yearFrom}-01-01`] : [])].join(",");
}

/**
 * A seed on OpenAlex, by DOI or by OpenAlex id: {paper, openalexId,
 * references: [W ids]}, or null when OpenAlex does not know it.
 */
export async function openAlexSeed(http, { doi = "", openalexId = "" } = {}, { apiKey = "", mailto = "", retry = {} } = {}) {
  const d = normalizeDoi(doi);
  const w = openAlexWorkId(openalexId);
  if (!d && !w) return null;
  const path = d ? `doi:${encodeURIComponent(d).replace(/%2F/gi, "/")}` : w;
  let data;
  try {
    data = await getJson(http, buildUrl(`${OPENALEX_WORKS}/${path}`, { select: SEED_SELECT, ...openAlexAuth({ mailto, apiKey }) }), { timeout: REQUEST_TIMEOUT, ...retry });
  } catch (e) {
    if (e?.status === 404) return null;
    throw e;
  }
  if (!isRecord(data)) return null;
  const references = (Array.isArray(data.referenced_works) ? data.referenced_works : []).map(openAlexWorkId).filter(Boolean);
  // A seed of any type keeps its id and references (mapOpenAlexPaper drops unkept types)
  const paper = mapOpenAlexPaper({ ...data, type: OPENALEX_TYPES.has(data.type) ? data.type : "article", is_paratext: false });
  return { paper, openalexId: openAlexWorkId(data.id) || null, references };
}

/** OpenAlex relevance search for one query: papers in relevance order. */
export async function openAlexSearch(http, query, { limit = 50, yearFrom = null, apiKey = "", mailto = "", retry = {} } = {}) {
  const url = buildUrl(OPENALEX_WORKS, {
    search: query,
    filter: openAlexFilter({ yearFrom }),
    sort: "relevance_score:desc",
    "per-page": Math.max(1, Math.min(200, limit)),
    select: OPENALEX_WORK_SELECT,
    ...openAlexAuth({ mailto, apiKey }),
  });
  return mapAll(resultsOf(await getJson(http, url, { timeout: REQUEST_TIMEOUT, ...retry })), mapOpenAlexPaper);
}

/** Works citing an OpenAlex work, most cited first. */
export async function openAlexCitingWorks(http, workId, { limit = 50, yearFrom = null, apiKey = "", mailto = "", retry = {} } = {}) {
  const id = openAlexWorkId(workId);
  if (!id) return [];
  const url = buildUrl(OPENALEX_WORKS, {
    filter: openAlexFilter({ yearFrom, extra: [`cites:${id}`] }),
    sort: "cited_by_count:desc",
    "per-page": Math.max(1, Math.min(200, limit)),
    select: OPENALEX_WORK_SELECT,
    ...openAlexAuth({ mailto, apiKey }),
  });
  return mapAll(resultsOf(await getJson(http, url, { timeout: REQUEST_TIMEOUT, ...retry })), mapOpenAlexPaper);
}

// Filter names OpenAlex accepts for work ids, tried in this order; the first that works is kept
const ID_FILTERS = ["openalex", "ids.openalex", "openalex_id"];
let idFilter = 0;

/** Reset the remembered id filter (tests). */
export function resetOpenAlexIdFilter() {
  idFilter = 0;
}

/**
 * Works by OpenAlex id, OPENALEX_ID_BATCH per request (an OR filter); the
 * order of `ids` is kept, unknown ids are left out. A filter name OpenAlex
 * rejects (HTTP 400) is replaced by the next known alias.
 */
export async function openAlexWorksByIds(http, ids, { apiKey = "", mailto = "", retry = {} } = {}) {
  const wanted = [...new Set((ids || []).map(openAlexWorkId).filter(Boolean))];
  const byId = new Map();
  for (let i = 0; i < wanted.length; i += OPENALEX_ID_BATCH) {
    http.signal?.throwIfCancelled?.();
    const batch = wanted.slice(i, i + OPENALEX_ID_BATCH);
    let data;
    for (;;) {
      const url = buildUrl(OPENALEX_WORKS, {
        filter: `${ID_FILTERS[idFilter]}:${batch.join("|")}`,
        "per-page": OPENALEX_ID_BATCH,
        select: OPENALEX_WORK_SELECT,
        ...openAlexAuth({ mailto, apiKey }),
      });
      try {
        data = await getJson(http, url, { timeout: REQUEST_TIMEOUT, ...retry });
        break;
      } catch (e) {
        if (e?.status !== 400 || idFilter >= ID_FILTERS.length - 1) throw e;
        idFilter++;
      }
    }
    for (const record of resultsOf(data)) {
      const paper = mapOpenAlexPaper(record);
      if (paper?.openalexId) byId.set(paper.openalexId, paper);
    }
  }
  return wanted.map((id) => byId.get(id)).filter(Boolean);
}

// ------------------------------------------------------- Semantic Scholar

/** Semantic Scholar ids ("paperId") of seeds by DOI, aligned with `dois` (null when unknown). */
export async function s2SeedIds(http, dois, { key = "", retry = {} } = {}) {
  const ids = dois.map((doi) => s2DoiId(doi));
  const lookup = ids.map((id, i) => [id, i]).filter(([id]) => id);
  const out = dois.map(() => null);
  if (!lookup.length) return out;
  const records = await s2Batch(http, lookup.map(([id]) => id), { key, fields: "paperId,externalIds", retry });
  lookup.forEach(([, i], j) => {
    const id = records[j]?.paperId;
    if (typeof id === "string" && id.trim()) out[i] = id.trim();
  });
  return out;
}

export async function s2SearchPapers(http, query, { limit = 50, yearFrom = null, key = "", retry = {} } = {}) {
  const { papers } = await s2Search(http, query, { key, limit, year: yearFrom || null, fields: S2_SURVEY_FIELDS, retry });
  return mapAll(papers, mapS2SurveyPaper);
}

export async function s2ReferencePapers(http, paperId, { limit = 50, key = "", retry = {} } = {}) {
  return mapAll(await s2References(http, paperId, { key, limit, fields: S2_SURVEY_FIELDS, retry }), mapS2SurveyPaper);
}

export async function s2CitingPapers(http, paperId, { limit = 50, key = "", retry = {} } = {}) {
  return mapAll(await s2Citations(http, paperId, { key, limit, fields: S2_SURVEY_FIELDS, retry }), mapS2SurveyPaper);
}

export async function s2RecommendedPapers(http, paperIds, { limit = 100, key = "", retry = {} } = {}) {
  return mapAll(await s2Recommendations(http, paperIds, { key, limit, fields: S2_SURVEY_FIELDS, retry }), mapS2SurveyPaper);
}

/**
 * TLDRs (and abstracts) of papers by DOI or Semantic Scholar id, one batch
 * request per 500: [{tldr, abstract, s2Id}|null] aligned with `papers`.
 */
export async function s2Tldrs(http, papers, { key = "", retry = {} } = {}) {
  const ids = papers.map((p) => (p.s2Id ? p.s2Id : s2DoiId(p.doi)));
  const lookup = ids.map((id, i) => [id, i]).filter(([id]) => id);
  const out = papers.map(() => null);
  if (!lookup.length) return out;
  const records = await s2Batch(http, lookup.map(([id]) => id), { key, fields: S2_TLDR_FIELDS, retry });
  lookup.forEach(([, i], j) => {
    const r = records[j];
    if (!r) return;
    const tldr = isRecord(r.tldr) && typeof r.tldr.text === "string" ? collapseWhitespace(r.tldr.text) || null : null;
    const abstract = typeof r.abstract === "string" ? cleanHtml(r.abstract) : null;
    out[i] = { tldr, abstract, s2Id: typeof r.paperId === "string" ? r.paperId : null };
  });
  return out;
}

// ----------------------------------------------------------------- Crossref

/** Crossref bibliographic search for one query (relevance order). */
export async function crossrefSearch(http, query, { limit = 25, yearFrom = null, mailto = "", retry = {} } = {}) {
  const filter = ["type:journal-article", "type:posted-content", ...(yearFrom ? [`from-pub-date:${yearFrom}-01-01`] : [])].join(",");
  const url = buildUrl(CROSSREF_WORKS, { "query.bibliographic": query, filter, rows: Math.max(1, Math.min(100, limit)), select: CROSSREF_SELECT, mailto });
  const data = await getJson(http, url, { timeout: REQUEST_TIMEOUT, ...retry });
  const items = Array.isArray(data?.message?.items) ? data.message.items : [];
  return mapAll(items, mapCrossrefPaper);
}
