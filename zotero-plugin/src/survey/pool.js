// Candidate pool of the topic survey: merging the papers every source
// returned (stage 3, by DOI / OpenAlex id / Semantic Scholar id), choosing
// the papers that fit the pool cap, and the fuzzy dedupe with library
// matching of stage 4 (core/dedupe.js TitleIndex).

import { TitleIndex } from "../core/dedupe.js";
import { normalizeDoi } from "../core/text.js";

/**
 * Provenance tags (`via`): "seed" (a seed itself), "search" (a query hit),
 * "ref:seedN" (seed N cites the paper), "cites:seedN" (the paper cites seed
 * N) and "rec" (Semantic Scholar recommendation for the seeds).
 */
export const VIA_SEED = "seed";
export const VIA_SEARCH = "search";
export const VIA_REC = "rec";
export const viaRef = (n) => `ref:seed${n}`;
export const viaCites = (n) => `cites:seed${n}`;

// Share of the pool reserved for linked papers (references, citations,
// recommendations) when the searches found enough to fill the rest
export const LINKED_SHARE = 0.6;

const hasText = (v) => typeof v === "string" && v.trim() !== "";

/** Seed numbers (1-based) a paper is linked to through references or citations. */
export function linkedSeeds(via) {
  const seeds = new Set();
  for (const tag of via || []) {
    const m = /^(?:ref|cites):seed(\d+)$/.exec(tag);
    if (m) seeds.add(Number(m[1]));
  }
  return seeds;
}

/** Copy `from`'s fields into `into` where `into` lacks them; provenance, sources and query ranks are unions. */
export function mergePaper(into, from) {
  for (const key of ["doi", "openalexId", "s2Id", "arxivId", "venue", "published", "type", "url", "tldr"]) {
    if (!into[key] && from[key]) into[key] = from[key];
  }
  if (!Number.isInteger(into.year) && Number.isInteger(from.year)) into.year = from.year;
  // The longer abstract wins (one source may have a truncated or HTML-stripped one)
  if (hasText(from.abstract) && (!hasText(into.abstract) || from.abstract.length > into.abstract.length)) into.abstract = from.abstract;
  if ((!into.authors || !into.authors.length) && from.authors?.length) {
    into.authors = from.authors;
    into.authorCount = from.authorCount;
  }
  if (!into.issns?.length && from.issns?.length) into.issns = from.issns;
  into.citations = Math.max(Number(into.citations) || 0, Number(from.citations) || 0);
  into.via = [...new Set([...(into.via || []), ...(from.via || [])])];
  into.sources = [...new Set([...(into.sources || []), ...(from.sources || [])])];
  if (Number.isFinite(from.searchRank)) into.searchRank = Math.min(into.searchRank ?? Infinity, from.searchRank);
  into.searchHits = (into.searchHits || 0) + (from.searchHits || 0);
  if (from.seed && !into.seed) into.seed = from.seed;
  if (from.inLibrary) {
    into.inLibrary = true;
    into.itemKey ||= from.itemKey || null;
  }
  // A DOI is the preferred id (it may have been learnt from the other record)
  if (into.doi && !into.id.startsWith("doi:")) into.id = `doi:${normalizeDoi(into.doi)}`;
  return into;
}

/**
 * Papers merged by their identifiers (DOI, OpenAlex id, Semantic Scholar id);
 * fuzzy titles are left to stage 4 (dedupePool).
 */
export class PaperCollector {
  constructor() {
    this.papers = [];
    this.keys = new Map();
  }

  static keysOf(p) {
    const keys = [];
    const doi = normalizeDoi(p.doi);
    if (doi) keys.push(`doi:${doi}`);
    if (p.openalexId) keys.push(`oa:${p.openalexId}`);
    if (p.s2Id) keys.push(`s2:${p.s2Id}`);
    return keys;
  }

  /**
   * Add a paper found through `via` (a provenance tag); a known paper gains
   * the tag and missing fields. `searchRank` is its 0-based position in a result list.
   * @returns {object} the pooled paper
   */
  add(paper, via, { searchRank = null } = {}) {
    if (!paper || !hasText(paper.title)) return null;
    const incoming = {
      ...paper,
      doi: normalizeDoi(paper.doi) || null,
      via: [via],
      sources: paper.sources || [],
      ...(searchRank !== null ? { searchRank, searchHits: 1 } : {}),
    };
    const keys = PaperCollector.keysOf(incoming);
    const existing = keys.map((k) => this.keys.get(k)).find(Boolean);
    let target;
    if (existing) {
      target = mergePaper(existing, incoming);
    } else {
      target = { seed: 0, inLibrary: false, itemKey: null, searchHits: 0, ...incoming };
      this.papers.push(target);
    }
    for (const k of PaperCollector.keysOf(target)) if (!this.keys.has(k)) this.keys.set(k, target);
    return target;
  }
}

/**
 * The papers that fit the pool cap. Seeds always stay; linked papers
 * (references, citations, recommendations; most seed links first, then
 * citations) take up to LINKED_SHARE of the rest, search hits (best rank in
 * any result list, then the number of lists, then citations) the remainder,
 * and room one group leaves goes to the other.
 */
export function selectPool(papers, cap) {
  const seeds = papers.filter((p) => p.seed);
  const linkedScore = (p) => linkedSeeds(p.via).size + (p.via.includes(VIA_REC) ? 1 : 0);
  const linked = papers
    .filter((p) => !p.seed && linkedScore(p) > 0)
    .sort((a, b) => linkedScore(b) - linkedScore(a) || (b.searchHits || 0) - (a.searchHits || 0) || b.citations - a.citations);
  const linkedSet = new Set(linked);
  const searched = papers
    .filter((p) => !p.seed && !linkedSet.has(p))
    .sort((a, b) => (a.searchRank ?? Infinity) - (b.searchRank ?? Infinity) || (b.searchHits || 0) - (a.searchHits || 0) || b.citations - a.citations);
  const room = Math.max(0, cap - seeds.length);
  let linkedTake = Math.min(linked.length, Math.max(Math.floor(room * LINKED_SHARE), room - searched.length));
  const searchTake = Math.min(searched.length, room - linkedTake);
  linkedTake = Math.min(linked.length, room - searchTake);
  return [...seeds, ...linked.slice(0, linkedTake), ...searched.slice(0, searchTake)];
}

/**
 * Stage 4: merge duplicates left in the pool (normalised DOI, then fuzzy
 * title, see core/dedupe.js) and mark papers already in the library (by DOI
 * or title; `itemKey` names the library item). Library papers stay in the
 * pool: they are valuable context. A library abstract fills a missing one.
 * @param {object[]} papers
 * @param {{key, doi, title, abstract}[]} libraryItems
 * @returns {{papers: object[], merged: number, inLibrary: number}}
 */
export function dedupePool(papers, libraryItems, { threshold = 0.9 } = {}) {
  const kept = [];
  const byDoi = new Map();
  const titles = new TitleIndex(threshold);
  const byTitleId = new Map();
  let merged = 0;
  for (const p of papers) {
    const doi = normalizeDoi(p.doi);
    let target = doi ? byDoi.get(doi) : null;
    if (!target) {
      const id = titles.find(p.title);
      const candidate = id >= 0 ? byTitleId.get(id) : null;
      // Two different DOIs are two papers, however close their titles (a preprint and its article merge: one has no DOI)
      if (candidate && !(doi && candidate.doi && normalizeDoi(candidate.doi) !== doi)) target = candidate;
    }
    if (target) {
      mergePaper(target, p);
      merged++;
    } else {
      target = p;
      kept.push(p);
      const id = titles.add(p.title);
      if (id >= 0 && !byTitleId.has(id)) byTitleId.set(id, p);
    }
    const d = normalizeDoi(target.doi);
    if (d) byDoi.set(d, target);
  }

  const libraryByDoi = new Map();
  const libraryTitles = new TitleIndex(threshold);
  const libraryByTitleId = new Map();
  for (const item of libraryItems || []) {
    const doi = normalizeDoi(item.doi);
    if (doi && !libraryByDoi.has(doi)) libraryByDoi.set(doi, item);
    const id = libraryTitles.add(item.title);
    if (id >= 0 && !libraryByTitleId.has(id)) libraryByTitleId.set(id, item);
  }
  let inLibrary = 0;
  for (const p of kept) {
    const doi = normalizeDoi(p.doi);
    let item = doi ? libraryByDoi.get(doi) : null;
    if (!item) {
      const id = libraryTitles.find(p.title);
      item = id >= 0 ? libraryByTitleId.get(id) : null;
      // A title match with a different DOI is another paper
      if (item && doi && normalizeDoi(item.doi) && normalizeDoi(item.doi) !== doi) item = null;
    }
    if (item) {
      p.inLibrary = true;
      p.itemKey = item.key || p.itemKey || null;
      if (!hasText(p.abstract) && hasText(item.abstract)) p.abstract = item.abstract;
    } else {
      p.inLibrary = !!p.inLibrary;
      p.itemKey = p.itemKey || null;
    }
    if (p.inLibrary) inLibrary++;
  }
  return { papers: kept, merged, inLibrary };
}
