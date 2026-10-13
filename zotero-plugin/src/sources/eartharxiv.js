// EarthArXiv source via its OAI-PMH endpoint (oai_dc metadata).

import { parseAuthorName } from "../core/authors.js";
import { buildUrl, getText } from "../core/http.js";
import { cleanTitle, collapseWhitespace, normalizeDoi } from "../core/text.js";
import { parseXml } from "../core/xml.js";
import { isoDate, pastDaysWindow } from "./window.js";

const OAI_URL = "https://eartharxiv.org/api/oai/";

export function parseOaiPage(xml) {
  const root = parseXml(xml);
  const error = root.find("error");
  if (error) {
    const code = error.attr("code") || "unknown";
    if (code === "noRecordsMatch") return { works: [], token: null };
    throw new Error(`OAI-PMH error ${code}: ${collapseWhitespace(error.text)}`);
  }

  const works = [];
  for (const record of root.findAll("record")) {
    const header = record.child("header");
    if (header?.attr("status") === "deleted") continue;
    const dc = record.find("dc");
    if (!dc) continue;
    const title = cleanTitle(dc.child("title")?.text);
    if (!title) continue;

    let doi = "";
    let url = null;
    let numericId = null;
    for (const idEl of dc.childrenNamed("identifier")) {
      const text = collapseWhitespace(idEl.text);
      if (!text) continue;
      if (text.includes("doi.org/") || text.startsWith("10.")) doi = doi || normalizeDoi(text);
      else if (text.startsWith("http")) url = url || text;
      else if (/^\d+$/.test(text)) numericId = text;
    }
    if (!url && numericId) url = `https://eartharxiv.org/repository/view/${numericId}/`;
    const oaiId = collapseWhitespace(header?.child("identifier")?.text) || numericId || title;
    const date = collapseWhitespace(dc.child("date")?.text);

    works.push({
      id: doi ? `doi:${doi}` : `eartharxiv:${oaiId}`,
      source: "eartharxiv",
      title,
      abstract: collapseWhitespace(dc.child("description")?.text) || null,
      authors: dc
        .childrenNamed("creator")
        .map((c) => parseAuthorName(c.text))
        .filter(Boolean),
      doi: doi || null,
      url: url || (doi ? `https://doi.org/${doi}` : null),
      published: date ? (date.includes("T") ? date : `${date}T00:00:00Z`) : null,
      venue: "EarthArXiv",
      issns: [],
      subjects: dc.childrenNamed("subject").map((s) => collapseWhitespace(s.text)).filter(Boolean),
    });
  }
  const token = collapseWhitespace(root.find("resumptionToken")?.text) || null;
  return { works, token };
}

export async function fetchEartharxiv(http, { daysBack = 14, maxResults = 500, now = Date.now(), retry = {}, log = () => {} }) {
  const { since } = pastDaysWindow(daysBack, now);
  const results = [];
  let token = null;
  for (let page = 0; page < 50 && results.length < maxResults; page++) {
    const url = token
      ? buildUrl(OAI_URL, { verb: "ListRecords", resumptionToken: token })
      : buildUrl(OAI_URL, { verb: "ListRecords", metadataPrefix: "oai_dc", from: isoDate(since) });
    const xml = await getText(http, url, retry);
    const parsed = parseOaiPage(xml);
    results.push(...parsed.works);
    token = parsed.token;
    if (!token) break;
  }
  log(`EarthArXiv: ${Math.min(results.length, maxResults)} preprints since ${isoDate(since)}`);
  return results.slice(0, maxResults);
}
