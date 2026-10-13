// Crossref source: recent works from the whitelisted journals (by ISSN).

import { makeAuthor } from "../core/authors.js";
import { buildUrl, getJson } from "../core/http.js";
import { cleanHtml, cleanTitle, isNonArticleTitle, normalizeDoi, normalizeIssn } from "../core/text.js";
import { isoDate, pastDaysWindow } from "./window.js";

const CROSSREF_WORKS = "https://api.crossref.org/works";
const PAGE_SIZE = 500;
const ISSNS_PER_QUERY = 60;
const SKIPPED_TYPES = new Set(["journal-issue", "journal-volume", "journal", "component", "peer-review"]);
const SELECT = "DOI,title,author,abstract,container-title,created,URL,type,is-referenced-by-count,publisher,ISSN";

export function parseCrossrefItem(item) {
  const title = cleanTitle((item.title || [])[0]);
  if (!title) return null;
  if (SKIPPED_TYPES.has(item.type)) return null;
  const venue = (item["container-title"] || [])[0] || null;
  if (isNonArticleTitle(title, venue)) return null;

  const doi = normalizeDoi(item.DOI);
  const authors = (item.author || [])
    .map((a) => (a.family || a.given ? makeAuthor(a.given, a.family) : a.name ? { given: "", family: a.name, name: a.name } : null))
    .filter((a) => a && a.name);
  return {
    id: doi ? `doi:${doi}` : `crossref:${item.URL || title}`,
    source: "crossref",
    title,
    abstract: cleanHtml(item.abstract),
    authors,
    doi: doi || null,
    url: item.URL || (doi ? `https://doi.org/${doi}` : null),
    published: item.created?.["date-time"] || null,
    venue,
    issns: (item.ISSN || []).map(normalizeIssn).filter(Boolean),
    type: item.type || null,
    citations: Number(item["is-referenced-by-count"] || 0),
    publisher: item.publisher || null,
  };
}

/**
 * Fetch works created in the past `daysBack` days from the given ISSNs.
 * @returns {Promise<object[]>} candidate works
 */
export async function fetchCrossref(http, { issns, daysBack = 7, maxResults = 3000, mailto = "", now = Date.now(), retry = {}, log = () => {} }) {
  const list = [...new Set((issns || []).map(normalizeIssn).filter(Boolean))];
  if (!list.length) {
    log("Crossref: no journal ISSNs configured, skipping");
    return [];
  }
  const { since, until } = pastDaysWindow(daysBack, now);
  const results = [];
  for (let i = 0; i < list.length && results.length < maxResults; i += ISSNS_PER_QUERY) {
    const chunk = list.slice(i, i + ISSNS_PER_QUERY);
    const filter = [`from-created-date:${isoDate(since)}`, `until-created-date:${isoDate(until)}`, ...chunk.map((s) => `issn:${s}`)].join(",");
    let offset = 0;
    while (results.length < maxResults) {
      const url = buildUrl(CROSSREF_WORKS, {
        filter,
        sort: "created",
        order: "desc",
        rows: Math.min(PAGE_SIZE, maxResults - results.length),
        offset,
        select: SELECT,
        mailto,
      });
      const data = await getJson(http, url, retry);
      const message = data?.message || {};
      const items = message.items || [];
      if (!items.length) break;
      for (const item of items) {
        const work = parseCrossrefItem(item);
        if (work) results.push(work);
        if (results.length >= maxResults) break;
      }
      offset += items.length;
      if (offset >= (message["total-results"] || 0)) break;
    }
  }
  log(`Crossref: ${results.length} works from ${list.length} ISSNs (${isoDate(since)} – ${isoDate(until)})`);
  return results;
}

/**
 * Resolve a journal title to ISSNs via the Crossref /journals endpoint
 * (accepts only an exact normalised title match).
 */
export async function resolveJournalIssns(http, title, { mailto = "", retry = {} } = {}) {
  const url = buildUrl("https://api.crossref.org/journals", { query: title, rows: 5, mailto });
  const data = await getJson(http, url, retry);
  const norm = (s) => (s || "").toLowerCase().replace(/^the\s+/, "").replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const target = norm(title);
  for (const item of data?.message?.items || []) {
    if (norm(item.title) === target) {
      return [...new Set((item.ISSN || []).map(normalizeIssn).filter(Boolean))];
    }
  }
  return [];
}
