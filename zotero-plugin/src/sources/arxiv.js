// arXiv source: recent submissions whose primary category is in the configured list.

import { parseAuthorName } from "../core/authors.js";
import { buildUrl, getText } from "../core/http.js";
import { cleanTitle, collapseWhitespace, normalizeDoi } from "../core/text.js";
import { parseXml } from "../core/xml.js";
import { isoDate, pastDaysWindow } from "./window.js";

const ARXIV_API = "https://export.arxiv.org/api/query";

export function arxivIdFromUrl(url) {
  const m = String(url || "").match(/arxiv\.org\/abs\/([^\s?#]+?)(v\d+)?$/i);
  return m ? m[1] : null;
}

export function parseArxivFeed(xml, categories) {
  const root = parseXml(xml);
  const allowed = new Set(categories);
  const works = [];
  for (const entry of root.childrenNamed("entry")) {
    const title = cleanTitle(entry.child("title")?.text);
    if (!title) continue;
    const primary = entry.child("primary_category")?.attr("term") || null;
    if (allowed.size && !allowed.has(primary)) continue; // drop cross-listed papers

    const idUrl = collapseWhitespace(entry.child("id")?.text);
    const arxivId = arxivIdFromUrl(idUrl);
    const alternate = entry.childrenNamed("link").find((l) => l.attr("rel") === "alternate");
    const doi = normalizeDoi(entry.child("doi")?.text);
    works.push({
      id: arxivId ? `arxiv:${arxivId}` : `arxiv:${idUrl || title}`,
      source: "arxiv",
      title,
      abstract: collapseWhitespace(entry.child("summary")?.text) || null,
      authors: entry
        .childrenNamed("author")
        .map((a) => parseAuthorName(a.child("name")?.text))
        .filter(Boolean),
      doi: doi || null,
      url: alternate?.attr("href") || idUrl || null,
      published: collapseWhitespace(entry.child("published")?.text) || null,
      venue: "arXiv",
      issns: [],
      arxivId,
      primaryCategory: primary,
    });
  }
  return works;
}

export async function fetchArxiv(http, { categories, daysBack = 7, maxResults = 500, now = Date.now(), retry = {}, log = () => {} }) {
  const cats = (categories || []).filter(Boolean);
  if (!cats.length) {
    log("arXiv: no categories configured, skipping");
    return [];
  }
  const { since, until } = pastDaysWindow(daysBack, now);
  const ymd = (ms) => isoDate(ms).replace(/-/g, "");
  const query = `(${cats.map((c) => `cat:${c}`).join(" OR ")}) AND submittedDate:[${ymd(since)}0000 TO ${ymd(until)}2359]`;
  const url = buildUrl(ARXIV_API, {
    search_query: query,
    sortBy: "submittedDate",
    sortOrder: "descending",
    max_results: Math.min(maxResults * 3, 2000), // extra room for cross-listed papers
  });
  const xml = await getText(http, url, retry);
  const works = parseArxivFeed(xml, cats).slice(0, maxResults);
  log(`arXiv: ${works.length} entries (${cats.join(", ")})`);
  return works;
}
