// Candidate filters, ported from zotwatch/pipeline/filters.py.

import { PREPRINT_SOURCES } from "./scoring.js";

export function utcDayStart(now = Date.now()) {
  const d = new Date(now);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
}

/** Keep works published within `days` days (counted from today's UTC midnight). */
export function filterRecent(works, days, now = Date.now()) {
  if (!days || days <= 0) return works;
  const cutoff = utcDayStart(now) - days * 86400000;
  return works.filter((w) => {
    const t = w.published ? Date.parse(w.published) : NaN;
    return Number.isFinite(t) && t >= cutoff;
  });
}

/** Cap the share of preprints in an ordered list. */
export function limitPreprints(works, maxRatio) {
  if (!works.length || !(maxRatio > 0) || maxRatio >= 1) return works;
  const out = [];
  let preprints = 0;
  for (const w of works) {
    if (PREPRINT_SOURCES.has(w.source)) {
      if ((preprints + 1) / (out.length + 1) > maxRatio) continue;
      preprints++;
    }
    out.push(w);
  }
  return out;
}

export function filterWithoutAbstract(works) {
  return works.filter((w) => w.abstract && w.abstract.trim());
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Whole-word, case-insensitive keyword matcher (CJK keywords match as substrings). */
export function compileKeywords(keywords) {
  const list = (keywords || []).map((k) => k.trim()).filter(Boolean);
  if (!list.length) return null;
  const parts = list.map((k) => {
    const escaped = escapeRegExp(k.toLowerCase());
    return /[㐀-鿿]/.test(k) ? escaped : `(?<![\\p{L}\\p{N}])${escaped}(?![\\p{L}\\p{N}])`;
  });
  return new RegExp(parts.join("|"), "giu");
}

function haystack(work, fields) {
  return fields.map((f) => work[f] || "").join("\n");
}

export function excludeByKeywords(works, keywords, fields = ["title", "abstract"]) {
  const re = compileKeywords(keywords);
  if (!re) return works;
  return works.filter((w) => {
    re.lastIndex = 0;
    return !re.test(haystack(w, fields));
  });
}

export function includeByKeywords(works, keywords, { minMatches = 1, fields = ["title", "abstract"] } = {}) {
  const re = compileKeywords(keywords);
  if (!re) return works;
  return works.filter((w) => {
    const matches = new Set();
    for (const m of haystack(w, fields).matchAll(re)) matches.add(m[0].toLowerCase());
    return matches.size >= minMatches;
  });
}
