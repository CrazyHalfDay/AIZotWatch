// Semantic Scholar Academic Graph and Recommendations API client
// (https://api.semanticscholar.org). Used by the abstract fallback (an extra
// tier with abstracts and TLDRs) and by the topic survey (search, references,
// citations, recommendations).
//
// The key travels in the `x-api-key` header. Introductory keys allow one
// request per second over all endpoints together; core/ratelimit.js paces
// api.semanticscholar.org accordingly and core/http.js retries its throttling
// answers (429, and 5xx, which the service also uses) with a longer delay.
// Every function throws on a failed request (like the other sources) and
// parses the response defensively: odd records are skipped, never thrown on.

import { parseAuthorName } from "../core/authors.js";
import { buildUrl, getJson, limitedRequest, postJson } from "../core/http.js";
import { cleanHtml, cleanTitle, collapseWhitespace, normalizeDoi } from "../core/text.js";

const S2_API = "https://api.semanticscholar.org";
const GRAPH = `${S2_API}/graph/v1`;
const RECOMMENDATIONS = `${S2_API}/recommendations/v1/papers`;
/** Paper fields requested unless the caller names others. */
export const S2_DEFAULT_FIELDS = "paperId,externalIds,title,abstract,tldr,year,publicationDate,venue,journal,authors,citationCount,openAccessPdf";
/** Ids per batch request (the service maximum). */
export const S2_BATCH_SIZE = 500;
const MAX_SEARCH_LIMIT = 100;
const MAX_EDGE_LIMIT = 1000;
const MAX_RECOMMENDATIONS = 500;
const REQUEST_TIMEOUT = 30000;
// Paper used by the key check (a 404 would still prove the key works)
const TEST_PAPER = "DOI:10.1038/nature14539";

function headersFor(key) {
  const k = String(key || "").trim();
  return k ? { "x-api-key": k } : {};
}

const isRecord = (value) => !!value && typeof value === "object" && !Array.isArray(value);

/** Clamp a requested result count to 1..max (default when not a number). */
function clampLimit(value, fallback, max) {
  const n = Math.floor(Number(value));
  return Number.isFinite(n) && n >= 1 ? Math.min(n, max) : fallback;
}

/** Search text the service matches well: hyphenated terms find nothing, so hyphens become spaces. */
export function s2Query(text) {
  return collapseWhitespace(String(text ?? "").replace(/[-‐‑–—]+/g, " "));
}

/** A paper id as a URL path segment ("DOI:10.1/x" keeps its ":" and "/", like the service's examples). */
export function s2PathId(id) {
  return encodeURIComponent(String(id ?? "").trim()).replace(/%3A/gi, ":").replace(/%2F/gi, "/");
}

/** "DOI:10.1016/…" for a DOI (any form), null when it is not one. */
export function s2DoiId(doi) {
  const d = normalizeDoi(doi);
  return d ? `DOI:${d}` : null;
}

function isoDate(value) {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : null;
}

/**
 * Map a Semantic Scholar paper to the plugin work shape; null for records
 * without a title or any identifier.
 * Extra fields: `tldr` (one-sentence summary), `s2Id`, `year`, `openAccessPdf` (URL).
 */
export function mapS2Paper(p) {
  if (!isRecord(p)) return null;
  const title = cleanTitle(p.title);
  if (!title) return null;
  const ids = isRecord(p.externalIds) ? p.externalIds : {};
  const doi = normalizeDoi(ids.DOI) || null;
  const s2Id = typeof p.paperId === "string" && p.paperId.trim() ? p.paperId.trim() : null;
  if (!doi && !s2Id) return null;
  const authors = (Array.isArray(p.authors) ? p.authors : [])
    .map((a) => (isRecord(a) && typeof a.name === "string" ? parseAuthorName(a.name) : null))
    .filter(Boolean);
  const journal = isRecord(p.journal) && typeof p.journal.name === "string" ? collapseWhitespace(p.journal.name) : "";
  const venue = journal || (typeof p.venue === "string" ? collapseWhitespace(p.venue) : "") || null;
  const year = Number.isInteger(p.year) && p.year > 1000 ? p.year : null;
  const tldr = isRecord(p.tldr) && typeof p.tldr.text === "string" ? collapseWhitespace(p.tldr.text) || null : null;
  const pdf = isRecord(p.openAccessPdf) && typeof p.openAccessPdf.url === "string" && /^https?:\/\//i.test(p.openAccessPdf.url) ? p.openAccessPdf.url : null;
  const arxivId = typeof ids.ArXiv === "string" && ids.ArXiv.trim() ? ids.ArXiv.trim() : null;
  const citations = Number(p.citationCount);
  return {
    id: doi ? `doi:${doi}` : `s2:${s2Id}`,
    source: "semanticscholar",
    title,
    abstract: typeof p.abstract === "string" ? cleanHtml(p.abstract) || null : null,
    tldr,
    authors,
    doi,
    url: doi ? `https://doi.org/${doi}` : `https://www.semanticscholar.org/paper/${s2Id}`,
    published: isoDate(p.publicationDate),
    year: year ?? (isoDate(p.publicationDate) ? Number(p.publicationDate.slice(0, 4)) : null),
    venue,
    type: null,
    citations: Number.isFinite(citations) && citations >= 0 ? citations : 0,
    s2Id,
    ...(arxivId ? { arxivId } : {}),
    openAccessPdf: pdf,
  };
}

/** A JSON array response, or an error for anything else (an error object, an HTML page). */
function expectArray(data, what) {
  if (!Array.isArray(data)) throw new Error(`Semantic Scholar ${what}: unexpected response`);
  return data;
}

/**
 * Papers by id (`DOI:…`, `CorpusId:…`, `ARXIV:…` or S2 paper ids), up to 500
 * per request. Returns raw paper records aligned with `ids` (null for unknown
 * ids or malformed records).
 * @param {object} [options] - {key, fields, retry, timeout}
 */
export async function s2Batch(http, ids, { key = "", fields = S2_DEFAULT_FIELDS, retry = {}, timeout = REQUEST_TIMEOUT } = {}) {
  const list = (ids || []).map((id) => String(id ?? "").trim());
  const out = [];
  for (let i = 0; i < list.length; i += S2_BATCH_SIZE) {
    const chunk = list.slice(i, i + S2_BATCH_SIZE);
    const data = expectArray(
      await postJson(http, buildUrl(`${GRAPH}/paper/batch`, { fields }), { ids: chunk }, { ...retry, headers: headersFor(key), timeout }),
      "batch",
    );
    for (let j = 0; j < chunk.length; j++) out.push(isRecord(data[j]) ? data[j] : null);
  }
  return out;
}

/**
 * Relevance search: {total, papers} with raw paper records (at most 100 per call).
 * `year` is a start year (2016 → "2016-") or a range string ("2016-2020").
 */
export async function s2Search(http, query, { key = "", limit = 20, offset = 0, year = null, fields = S2_DEFAULT_FIELDS, retry = {}, timeout = REQUEST_TIMEOUT } = {}) {
  const text = s2Query(query);
  if (!text) return { total: 0, papers: [] };
  const years = Number.isInteger(year) ? `${year}-` : typeof year === "string" && /^\d{4}(-(\d{4})?)?$/.test(year.trim()) ? year.trim() : null;
  const url = buildUrl(`${GRAPH}/paper/search`, {
    query: text,
    limit: clampLimit(limit, 20, MAX_SEARCH_LIMIT),
    offset: Number.isInteger(offset) && offset > 0 ? offset : null,
    fields,
    year: years,
  });
  const data = await getJson(http, url, { ...retry, headers: headersFor(key), timeout });
  const papers = Array.isArray(data?.data) ? data.data.filter(isRecord) : [];
  const total = Number(data?.total);
  return { total: Number.isFinite(total) ? total : papers.length, papers };
}

/** Papers on one side of a paper's citation edges, unwrapped (`influential` from isInfluential). */
async function s2Edges(http, paperId, side, wrapper, { key = "", limit = 100, fields = S2_DEFAULT_FIELDS, retry = {}, timeout = REQUEST_TIMEOUT } = {}) {
  const id = String(paperId ?? "").trim();
  if (!id) return [];
  const url = buildUrl(`${GRAPH}/paper/${s2PathId(id)}/${side}`, {
    fields: `${fields},isInfluential`,
    limit: clampLimit(limit, 100, MAX_EDGE_LIMIT),
  });
  const data = await getJson(http, url, { ...retry, headers: headersFor(key), timeout });
  const edges = Array.isArray(data?.data) ? data.data : [];
  return edges
    .filter((edge) => isRecord(edge) && isRecord(edge[wrapper]) && (edge[wrapper].paperId || edge[wrapper].title))
    .map((edge) => ({ ...edge[wrapper], influential: edge.isInfluential === true }));
}

/** Papers a paper cites (raw records from `citedPaper`). */
export function s2References(http, paperId, options = {}) {
  return s2Edges(http, paperId, "references", "citedPaper", options);
}

/** Papers citing a paper (raw records from `citingPaper`). */
export function s2Citations(http, paperId, options = {}) {
  return s2Edges(http, paperId, "citations", "citingPaper", options);
}

/** Papers recommended for a set of positive (and optional negative) paper ids. */
export async function s2Recommendations(http, positiveIds, { negativeIds = [], key = "", limit = 100, fields = S2_DEFAULT_FIELDS, retry = {}, timeout = REQUEST_TIMEOUT } = {}) {
  const positive = [...new Set((positiveIds || []).map((id) => String(id ?? "").trim()).filter(Boolean))];
  if (!positive.length) return [];
  const negative = [...new Set((negativeIds || []).map((id) => String(id ?? "").trim()).filter(Boolean))];
  const url = buildUrl(RECOMMENDATIONS, { limit: clampLimit(limit, 100, MAX_RECOMMENDATIONS), fields });
  const data = await postJson(http, url, { positivePaperIds: positive, negativePaperIds: negative }, { ...retry, headers: headersFor(key), timeout });
  return Array.isArray(data?.recommendedPapers) ? data.recommendedPapers.filter(isRecord) : [];
}

/** Author search: [{authorId, name, affiliations, paperCount, citationCount, hIndex, orcid}]. */
export async function s2AuthorSearch(http, query, { key = "", limit = 10, retry = {}, timeout = REQUEST_TIMEOUT } = {}) {
  const text = s2Query(query);
  if (!text) return [];
  const url = buildUrl(`${GRAPH}/author/search`, { query: text, limit: clampLimit(limit, 10, MAX_SEARCH_LIMIT), fields: "name,affiliations,paperCount,citationCount,hIndex,externalIds" });
  const data = await getJson(http, url, { ...retry, headers: headersFor(key), timeout });
  const count = (v) => (Number.isFinite(Number(v)) ? Number(v) : null);
  return (Array.isArray(data?.data) ? data.data : [])
    .filter((a) => isRecord(a) && typeof a.authorId === "string" && typeof a.name === "string")
    .map((a) => ({
      authorId: a.authorId,
      name: collapseWhitespace(a.name),
      affiliations: Array.isArray(a.affiliations) ? a.affiliations.filter((x) => typeof x === "string") : [],
      paperCount: count(a.paperCount),
      citationCount: count(a.citationCount),
      hIndex: count(a.hIndex),
      orcid: isRecord(a.externalIds) && typeof a.externalIds.ORCID === "string" ? a.externalIds.ORCID : null,
    }));
}

/**
 * Connection check ({key: "semanticscholar", name, ok, detail}): one paper
 * lookup with the key (2xx or 404 means accepted). Without a key nothing is
 * requested (ok null).
 */
export async function checkSemanticScholarRow(http, { key = "" } = {}) {
  const row = { key: "semanticscholar", name: "Semantic Scholar", ok: false, detail: "" };
  const k = String(key || "").trim();
  if (!k) return { ...row, ok: null, detail: "未配置 API Key" };
  try {
    const res = await limitedRequest(http, { url: buildUrl(`${GRAPH}/paper/${TEST_PAPER}`, { fields: "title" }), headers: headersFor(k), timeout: REQUEST_TIMEOUT });
    if ((res.status >= 200 && res.status < 300) || res.status === 404) return { ...row, ok: true, detail: "Key 有效" };
    if (res.status === 401 || res.status === 403) return { ...row, detail: `Key 无效（HTTP ${res.status}）` };
    if (res.status === 429) return { ...row, detail: "请求过于频繁（HTTP 429），请稍后再试" };
    return { ...row, detail: `HTTP ${res.status}` };
  } catch (e) {
    if (e?.cancelled) throw e;
    return { ...row, detail: String(e?.message || e) };
  }
}
