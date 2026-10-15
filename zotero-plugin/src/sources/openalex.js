// OpenAlex abstracts: many publishers (e.g. Elsevier) do not deposit abstracts
// in Crossref, but OpenAlex stores them as an inverted index. Used by the last
// tier of the abstract fallback (sources/abstracts.js), one filter query per
// 50 DOIs, and by the followed-authors source.

import { buildUrl } from "../core/http.js";
import { collapseWhitespace, normalizeDoi } from "../core/text.js";

const OPENALEX_WORKS = "https://api.openalex.org/works";
// Abstracts are far shorter; larger positions are malformed and would make a huge sparse array
const MAX_ABSTRACT_WORDS = 20000;
/** DOIs per batched abstract lookup (OpenAlex allows up to 50 values in an OR filter). */
export const OPENALEX_BATCH_SIZE = 50;

export function reconstructAbstract(inverted) {
  if (!inverted || typeof inverted !== "object") return null;
  const words = [];
  for (const [word, positions] of Object.entries(inverted)) {
    if (!Array.isArray(positions)) continue;
    for (const pos of positions) {
      if (Number.isInteger(pos) && pos >= 0 && pos < MAX_ABSTRACT_WORDS) words[pos] = word;
    }
  }
  const text = collapseWhitespace(words.filter((w) => w !== undefined).join(" "));
  return text || null;
}

/** True when a DOI can go into an OpenAlex OR filter ("|" and "," are filter syntax). */
export function batchableDoi(doi) {
  return !/[|,]/.test(doi);
}

/** URL of one batched abstract lookup for up to OPENALEX_BATCH_SIZE normalized DOIs. */
export function openAlexAbstractsUrl(dois, mailto = "") {
  return buildUrl(OPENALEX_WORKS, {
    filter: `doi:${dois.map((doi) => `https://doi.org/${doi}`).join("|")}`,
    select: "doi,abstract_inverted_index",
    "per-page": OPENALEX_BATCH_SIZE,
    mailto,
  });
}

/** Reconstructed abstracts by normalized DOI from a batched lookup response; never throws. */
export function parseOpenAlexAbstracts(data) {
  const byDoi = new Map();
  const results = Array.isArray(data?.results) ? data.results : [];
  for (const record of results) {
    if (!record || typeof record !== "object") continue;
    const doi = normalizeDoi(record.doi);
    const abstract = reconstructAbstract(record.abstract_inverted_index);
    if (doi && abstract && !byDoi.has(doi)) byDoi.set(doi, abstract);
  }
  return byDoi;
}
