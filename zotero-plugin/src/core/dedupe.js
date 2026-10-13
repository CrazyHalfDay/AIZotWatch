// Duplicate detection against the library and within a batch, ported from
// zotwatch/pipeline/dedupe.py (DOI + exact title + fuzzy token-set ratio).
// An inverted token index keeps fuzzy matching cheap for large libraries.

import { normalizeDoi, normalizeTitle } from "./text.js";

const LENGTH_TOLERANCE = 0.3;
const MIN_TOKEN_LEN = 3;

/** Indel-based similarity ratio in [0, 100] (rapidfuzz `ratio`). */
export function ratio(a, b) {
  if (a === b) return 100;
  const n = a.length;
  const m = b.length;
  if (!n || !m) return 0;
  let prev = new Uint16Array(m + 1);
  let curr = new Uint16Array(m + 1);
  for (let i = 1; i <= n; i++) {
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= m; j++) {
      curr[j] = ca === b.charCodeAt(j - 1) ? prev[j - 1] + 1 : Math.max(prev[j], curr[j - 1]);
    }
    [prev, curr] = [curr, prev];
  }
  const lcs = prev[m];
  return (200 * lcs) / (n + m);
}

/** rapidfuzz `token_set_ratio`. */
export function tokenSetRatio(a, b) {
  const ta = new Set(a.split(" ").filter(Boolean));
  const tb = new Set(b.split(" ").filter(Boolean));
  if (!ta.size || !tb.size) return 0;
  const inter = [...ta].filter((t) => tb.has(t)).sort();
  const diffA = [...ta].filter((t) => !tb.has(t)).sort();
  const diffB = [...tb].filter((t) => !ta.has(t)).sort();
  // A subset relationship counts as a full match (rapidfuzz behaviour)
  if (inter.length && (!diffA.length || !diffB.length)) return 100;
  const sect = inter.join(" ");
  const combA = [sect, diffA.join(" ")].filter(Boolean).join(" ");
  const combB = [sect, diffB.join(" ")].filter(Boolean).join(" ");
  const scores = [ratio(combA, combB)];
  if (sect) scores.push(ratio(sect, combA), ratio(sect, combB));
  return Math.max(...scores);
}

function tokensOf(title) {
  return [...new Set(title.split(" ").filter((t) => t.length >= MIN_TOKEN_LEN))];
}

export class TitleIndex {
  constructor(threshold = 0.9) {
    this.threshold = threshold * 100;
    this.titles = [];
    this.exact = new Set();
    this.postings = new Map();
  }

  add(title) {
    const norm = normalizeTitle(title);
    if (!norm || this.exact.has(norm)) return;
    const id = this.titles.length;
    this.titles.push(norm);
    this.exact.add(norm);
    for (const t of tokensOf(norm)) {
      let list = this.postings.get(t);
      if (!list) this.postings.set(t, (list = []));
      list.push(id);
    }
  }

  has(title) {
    const norm = normalizeTitle(title);
    if (!norm) return false;
    if (this.exact.has(norm)) return true;
    const tokens = tokensOf(norm);
    if (!tokens.length) return false;
    // Count shared tokens per indexed title; only near-complete overlaps can pass
    const overlap = new Map();
    for (const t of tokens) {
      const list = this.postings.get(t);
      if (!list) continue;
      for (const id of list) overlap.set(id, (overlap.get(id) || 0) + 1);
    }
    const minLen = norm.length * (1 - LENGTH_TOLERANCE);
    const maxLen = norm.length * (1 + LENGTH_TOLERANCE);
    for (const [id, shared] of overlap) {
      const other = this.titles[id];
      if (other.length < minLen || other.length > maxLen) continue;
      if (shared < Math.ceil(tokens.length * 0.6)) continue;
      if (tokenSetRatio(norm, other) >= this.threshold) return true;
    }
    return false;
  }
}

/**
 * Remove works already in the library (by DOI or similar title) and duplicates
 * within the batch. `library` is a list of { doi, title }.
 */
export function dedupeAgainstLibrary(works, library, { threshold = 0.9, extraDois = [] } = {}) {
  const libraryDois = new Set(extraDois.map(normalizeDoi).filter(Boolean));
  const libraryTitles = new TitleIndex(threshold);
  for (const item of library) {
    const doi = normalizeDoi(item.doi);
    if (doi) libraryDois.add(doi);
    if (item.title) libraryTitles.add(item.title);
  }

  const seenIds = new Set();
  const batchTitles = new TitleIndex(threshold);
  const kept = [];
  for (const w of works) {
    const doi = normalizeDoi(w.doi);
    if (doi && (libraryDois.has(doi) || seenIds.has(doi))) continue;
    if (seenIds.has(w.id)) continue;
    if (libraryTitles.has(w.title) || batchTitles.has(w.title)) continue;
    kept.push(w);
    seenIds.add(w.id);
    if (doi) seenIds.add(doi);
    batchTitles.add(w.title);
  }
  return kept;
}
