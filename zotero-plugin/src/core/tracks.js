// Recommendation tracks. One run can produce three kinds of inbox entries,
// told apart by `label` (mirroring the Python archive labels):
//   personal  - must_read | consider | ignore (library-similarity ranking)
//   flagship  - "flagship": geoscience articles from flagship journals (顶刊地学速览)
//   followed  - "followed": works found only through followed authors (关注作者)
// `followedAuthors` is an annotation that can sit on works of any track.

import { normalizeDoi } from "./text.js";

export const TRACKS = ["personal", "flagship", "followed"];
const TRACK_LABELS = new Set(["flagship", "followed"]);

/** Track of a work: "flagship", "followed" or "personal". */
export function trackOf(work) {
  return TRACK_LABELS.has(work?.label) ? work.label : "personal";
}

/**
 * Union of two name lists. With `order` (the configured names), names follow
 * that order and unknown names keep their first-seen order after them.
 */
export function unionNames(a, b, order = []) {
  const names = [...new Set([...(a || []), ...(b || [])].filter((n) => typeof n === "string" && n))];
  if (!order.length) return names;
  const rank = new Map(order.map((n, i) => [n, i]));
  const at = (n) => (rank.has(n) ? rank.get(n) : order.length);
  return names.map((n, i) => [n, i]).sort((x, y) => at(x[0]) - at(y[0]) || x[1] - y[1]).map(([n]) => n);
}

/**
 * Attach followed-author works to the personal/flagship works of the same run.
 * A followed work whose id or normalized DOI matches one of `works` adds its
 * author names to that work's `followedAuthors` (mutated in place) instead of
 * becoming a separate entry; the others are returned as followed-only works
 * (label "followed", score and similarity 0 so display code never sees NaN).
 */
export function attachFollowed(works, followed, order = []) {
  const byKey = new Map();
  for (const w of works) {
    byKey.set(`id:${w.id}`, w);
    const doi = normalizeDoi(w.doi);
    if (doi && !byKey.has(`doi:${doi}`)) byKey.set(`doi:${doi}`, w);
  }
  const only = [];
  for (const f of followed) {
    const doi = normalizeDoi(f.doi);
    const match = byKey.get(`id:${f.id}`) || (doi ? byKey.get(`doi:${doi}`) : null);
    if (match) {
      match.followedAuthors = unionNames(match.followedAuthors, f.followedAuthors, order);
    } else {
      only.push({ ...f, label: "followed", score: 0, similarity: 0 });
    }
  }
  return only;
}

const byPublishedDesc = (a, b) => (Date.parse(b.published || "") || 0) - (Date.parse(a.published || "") || 0);

/**
 * Works to summarize in one run: the `n` best personal works by score, plus up
 * to `n` flagship and `n` followed-only works (newest first). Works that
 * already have a summary are skipped before the limits apply.
 */
export function selectSummaryTargets(works, n) {
  if (!(n > 0)) return [];
  const pending = works.filter((w) => !w.summary);
  const pick = (track, compare) => pending.filter((w) => trackOf(w) === track).sort(compare).slice(0, n);
  return [
    ...pick("personal", (a, b) => (b.score || 0) - (a.score || 0)),
    ...pick("flagship", byPublishedDesc),
    ...pick("followed", byPublishedDesc),
  ];
}

/** Count works per track: {personal, flagship, followed}. */
export function countByTrack(works) {
  const counts = { personal: 0, flagship: 0, followed: 0 };
  for (const w of works) counts[trackOf(w)]++;
  return counts;
}
