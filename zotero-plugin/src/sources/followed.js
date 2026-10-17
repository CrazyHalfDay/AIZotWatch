// Followed authors: recent works of chosen researchers from OpenAlex, found by
// ORCID iD or OpenAlex author id (port of Python sources/openalex.py
// OpenAlexAuthorFetcher). Unlike Python, every run searches the same rolling
// publication-date window (the engine keeps a seen-set), all authors share a
// few OR queries, non-research types and paratext are skipped, and malformed
// records never crash the run.

import { normalizeOpenAlexAuthorId, normalizeOrcid, parseAuthorName } from "../core/authors.js";
import { buildUrl, getJson } from "../core/http.js";
import { cleanHtml, cleanTitle, isNonArticleTitle, normalizeDoi, normalizeIssn } from "../core/text.js";
import { openAlexAuth, reconstructAbstract } from "./openalex.js";
import { isoDate } from "./window.js";

const OPENALEX_WORKS = "https://api.openalex.org/works";
const DAY = 86400000;
const PAGE_SIZE = 200;
const AUTHORS_PER_QUERY = 25;
// Singleton lookups per run for works whose (truncated) authorship list names no followed author
const MAX_ATTRIBUTION_LOOKUPS = 20;
const KEPT_TYPES = new Set(["article", "review", "letter", "preprint", "book-chapter", "book", "report", "dissertation"]);
// is_authors_truncated is not selected: an unknown select field fails the whole
// query, and an unmatched work of a multi-author query is looked up anyway
const SELECT = "id,doi,title,display_name,publication_date,type,authorships,primary_location,abstract_inverted_index,cited_by_count,is_paratext";

/** OpenAlex work id ("W123") from its URL form; "" when absent. */
function openalexWorkId(value) {
  const m = String(value ?? "").match(/\bW\d+\b/i);
  return m ? m[0].toUpperCase() : "";
}

function isoPublished(value) {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : null;
}

/** Names of the given authors, once each, in configuration order. */
function namesOf(authors) {
  return [...new Set(authors.map((a) => a.name))];
}

/** Names of configured authors listed in a work's authorships, in configuration order. */
function matchFollowed(authorships, authors) {
  const orcids = new Set();
  const ids = new Set();
  for (const a of authorships) {
    const orcid = normalizeOrcid(a.author?.orcid);
    const id = normalizeOpenAlexAuthorId(a.author?.id);
    if (orcid) orcids.add(orcid);
    if (id) ids.add(id);
  }
  return namesOf(authors.filter((a) => (a.orcid && orcids.has(a.orcid)) || (a.openalexId && ids.has(a.openalexId))));
}

/**
 * Map an OpenAlex work to the plugin work shape; null for records to skip.
 * `followedAuthors` names the configured authors listed in its authorships.
 * When none is listed, the query's author is certain only for a one-author
 * query or a record without authorships; otherwise (OpenAlex lists show the
 * first 100 authors only) it stays empty and `needsAttribution` is set, and
 * fetchFollowedWorks looks the full author list up.
 * @param {object[]} authors - configured authors of the query that returned it (fallback attribution)
 * @param {object[]} allAuthors - every configured author (attribution, in configuration order)
 */
export function parseOpenAlexWork(work, authors = [], allAuthors = authors) {
  if (!work || typeof work !== "object") return null;
  if (work.is_paratext === true) return null;
  if (!KEPT_TYPES.has(work.type)) return null;
  const location = work.primary_location && typeof work.primary_location === "object" ? work.primary_location : {};
  const source = location.source && typeof location.source === "object" ? location.source : null;
  const venue = typeof source?.display_name === "string" ? source.display_name : null;
  const title = cleanTitle(work.display_name || work.title);
  if (!title || isNonArticleTitle(title, venue)) return null;

  const doi = normalizeDoi(work.doi);
  const workId = openalexWorkId(work.id);
  if (!doi && !workId) return null;
  const authorships = Array.isArray(work.authorships) ? work.authorships.filter((a) => a && typeof a === "object") : [];
  const names = authorships.map((a) => parseAuthorName(a.author?.display_name || a.raw_author_name)).filter(Boolean);
  let followedAuthors = matchFollowed(authorships, allAuthors);
  let needsAttribution = false;
  if (!followedAuthors.length) {
    // OpenAlex matched the filter, so some queried author wrote it; never name
    // every author of a multi-author query (most of them did not)
    const queried = namesOf(authors);
    if (queried.length === 1 || !authorships.length) followedAuthors = queried;
    else needsAttribution = work.is_authors_truncated !== false;
  }
  const issns = Array.isArray(source?.issn) ? [...new Set(source.issn.map(normalizeIssn).filter(Boolean))] : null;
  const landing = typeof location.landing_page_url === "string" && /^https?:\/\//i.test(location.landing_page_url) ? location.landing_page_url : null;

  return {
    id: doi ? `doi:${doi}` : `openalex:${workId}`,
    source: "openalex",
    title,
    abstract: cleanHtml(reconstructAbstract(work.abstract_inverted_index)),
    authors: names,
    doi: doi || null,
    url: doi ? `https://doi.org/${doi}` : landing || `https://openalex.org/${workId}`,
    published: isoPublished(work.publication_date),
    venue,
    issns,
    type: work.type,
    citations: Number(work.cited_by_count) || 0,
    followedAuthors,
    openalexId: workId || null,
    ...(needsAttribution ? { needsAttribution } : {}),
  };
}

/**
 * Followed authors of a work from its full author list (the singleton
 * endpoint is not truncated); [] when the lookup fails or still names none.
 */
async function lookupFollowed(http, work, authors, { mailto, apiKey, retry, log }) {
  try {
    const data = await getJson(http, buildUrl(`${OPENALEX_WORKS}/${work.openalexId}`, { select: "id,authorships", ...openAlexAuth({ mailto, apiKey }) }), retry);
    const authorships = Array.isArray(data?.authorships) ? data.authorships.filter((a) => a && typeof a === "object") : [];
    return matchFollowed(authorships, authors);
  } catch (e) {
    if (e?.cancelled) throw e;
    log(`OpenAlex author lookup failed for ${work.openalexId}: ${e?.message || e}`);
    return [];
  }
}

function chunks(list, size) {
  const out = [];
  for (let i = 0; i < list.length; i += size) out.push(list.slice(i, i + size));
  return out;
}

/** Filter values per query: ORCID and OpenAlex-id authors use different filter keys. */
function authorQueries(authors) {
  const queries = [];
  for (const group of chunks(authors.filter((a) => a.orcid), AUTHORS_PER_QUERY)) {
    queries.push({ authors: group, filter: `author.orcid:${group.map((a) => `https://orcid.org/${a.orcid}`).join("|")}` });
  }
  for (const group of chunks(authors.filter((a) => !a.orcid && a.openalexId), AUTHORS_PER_QUERY)) {
    queries.push({ authors: group, filter: `authorships.author.id:${group.map((a) => a.openalexId).join("|")}` });
  }
  return queries;
}

/**
 * Fetch works published in the last `daysBack` days by the followed authors.
 * A work found by several authors (or queries) is returned once with all of
 * their names in `followedAuthors`. Throws when a query fails.
 * Without an OpenAlex API key only the first page of each query is read
 * (cursor paging needs a key); `onTruncated` is called once when a query had
 * more results than that page.
 * @param {{authors: {name: string, orcid?: string, openalexId?: string}[], daysBack?: number, maxResults?: number}} options
 *   `maxResults` caps the records read per query
 * @returns {Promise<object[]>} works, newest first
 */
export async function fetchFollowedWorks(
  http,
  { authors, daysBack = 60, mailto = "", apiKey = "", now = Date.now(), retry = {}, log = () => {}, maxResults = 500, onTruncated = () => {} },
) {
  const list = (authors || []).filter((a) => a && (a.orcid || a.openalexId));
  if (!list.length) return [];
  const key = String(apiKey || "").trim();
  const auth = openAlexAuth({ mailto, apiKey: key });
  const since = isoDate(now - daysBack * DAY);
  const byId = new Map();
  let records = 0;
  let truncated = false;
  for (const query of authorQueries(list)) {
    // Keyless requests read the first page only, without a cursor
    let cursor = key ? "*" : null;
    let read = 0;
    for (let page = 0; read < maxResults && (page === 0 || cursor); page++) {
      const url = buildUrl(OPENALEX_WORKS, {
        filter: `${query.filter},from_publication_date:${since}`,
        sort: "publication_date:desc",
        "per-page": PAGE_SIZE,
        cursor,
        select: SELECT,
        ...auth,
      });
      const data = await getJson(http, url, retry);
      const results = Array.isArray(data?.results) ? data.results : [];
      if (!key && Number(data?.meta?.count) > results.length && results.length < maxResults) truncated = true;
      if (!results.length) break;
      for (const raw of results.slice(0, maxResults - read)) {
        const work = parseOpenAlexWork(raw, query.authors, list);
        if (!work) continue;
        const seen = byId.get(work.id);
        if (seen) {
          const names = new Set([...seen.followedAuthors, ...work.followedAuthors]);
          seen.followedAuthors = namesOf(list).filter((n) => names.has(n));
          if (work.needsAttribution) seen.needsAttribution = true;
        } else {
          byId.set(work.id, work);
        }
      }
      read += results.length;
      cursor = key && typeof data?.meta?.next_cursor === "string" ? data.meta.next_cursor : null;
    }
    records += read;
  }
  if (truncated) {
    log("OpenAlex followed authors: no API key, only the first page of results was read");
    onTruncated();
  }
  // Works whose shown authorships name no followed author: read the full list
  let lookups = 0;
  for (const work of byId.values()) {
    if (!work.needsAttribution) continue;
    delete work.needsAttribution;
    if (!work.openalexId || lookups >= MAX_ATTRIBUTION_LOOKUPS) continue;
    lookups++;
    const found = await lookupFollowed(http, work, list, { mailto, apiKey: key, retry, log });
    const names = new Set([...work.followedAuthors, ...found]);
    work.followedAuthors = namesOf(list).filter((n) => names.has(n));
  }
  const works = [...byId.values()].sort((a, b) => (Date.parse(b.published || "") || 0) - (Date.parse(a.published || "") || 0));
  log(`OpenAlex followed authors: ${works.length} works (${records} records) for ${list.length} authors since ${since}`);
  return works;
}
