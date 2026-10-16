// Flagship geoscience track (顶刊地学速览), ported from
// zotwatch/pipeline/flagship_filter.py and WatchPipeline._select_flagship.
// Articles from flagship journals (matched by ISSN) leave the personal
// pipeline and are gated on field relevance instead of library similarity:
// their embedding is compared with a positive anchor (solid earth +
// paleontology) and several negative anchors (other fields).

import { normalizeIssn } from "./text.js";

/**
 * Python FlagshipConfig defaults (config/settings.py) and the ISSN list of
 * config/config.yaml: Nature, Science, Nat Commun, Sci Adv, Commun Earth
 * Environ, PNAS, Nat Geosci, Nat Rev Earth Environ (print and online).
 */
export const FLAGSHIP_DEFAULTS = Object.freeze({
  issns: Object.freeze([
    "0028-0836",
    "1476-4687",
    "0036-8075",
    "1095-9203",
    "2041-1723",
    "2375-2548",
    "2662-4435",
    "1091-6490",
    "0027-8424",
    "1752-0894",
    "1752-0908",
    "2662-138X",
  ]),
  positiveAnchor:
    "Solid Earth geoscience: igneous, metamorphic and sedimentary petrology, geochemistry and isotope geochemistry, " +
    "mineralogy, tectonics and structural geology, geochronology, ore deposits and economic geology, volcanology, the " +
    "deep Earth, stratigraphy, and paleontology / paleobiology.",
  // One coherent field per anchor keeps each embedding sharp (a blended anchor is too blurry)
  negativeAnchors: Object.freeze([
    "Atmospheric science, meteorology, climate dynamics, air quality, weather and atmospheric circulation.",
    "Condensed matter and materials physics: ferroelectrics, semiconductors, nanomaterials, thin films, crystal device " +
      "structures, electron microscopy of engineered materials, quantum, topological and magnetic materials.",
    "Chemistry: synthesis, catalysis, electrochemistry, batteries and energy storage, polymers, molecular, organic and " +
      "analytical chemistry.",
    "Biology and biomedicine: genetics and genomics, molecular and cell biology, neuroscience, physiology, immunology, " +
      "microbiology, ecology and evolution.",
    "Astronomy, astrophysics, cosmology, exoplanets and the atmospheres of other planets.",
    "Engineering, computer science, machine learning, optics, photonics and applied device physics.",
  ]),
  llmBoundary:
    "ONLY solid-earth geoscience or paleontology (igneous/metamorphic/sedimentary petrology, geochemistry and isotope " +
    "geochemistry, mineralogy, tectonics and structural geology, geochronology, ore deposits, volcanology, stratigraphy, " +
    "paleontology/paleobiology, the deep Earth). EXCLUDE everything else, including materials science and " +
    "condensed-matter physics, chemistry, biology and biomedicine, astronomy and astrophysics, atmospheric " +
    "science/meteorology/climate, engineering and computer science — even when published in a general-science journal.",
  maxResults: 30,
});

/** Normalized ISSNs ("1234-567X"), invalid entries dropped, first occurrence kept. */
export function normalizeIssnList(list) {
  return [...new Set((list || []).map(normalizeIssn).filter(Boolean))];
}

/** One anchor per line (anchors contain commas): trimmed, blanks dropped, de-duplicated. */
export function parseAnchorLines(text) {
  return [
    ...new Set(
      String(text || "")
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter(Boolean),
    ),
  ];
}

/** Whether a work's ISSNs (print or online, as Crossref lists both) meet the flagship set. */
export function isFlagshipWork(work, issnSet) {
  if (!issnSet?.size) return false;
  return (work.issns || []).some((issn) => issnSet.has(normalizeIssn(issn)));
}

/** Split works into flagship-journal articles and the rest, keeping their order. */
export function splitFlagship(works, issnSet) {
  const flagship = [];
  const personal = [];
  for (const w of works) (isFlagshipWork(w, issnSet) ? flagship : personal).push(w);
  return { flagship, personal };
}

/** Cosine similarity; 0 when either vector is all zeros (e.g. a local-mode text without features). */
export function cosine(a, b) {
  let ab = 0;
  let aa = 0;
  let bb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa > 0 && bb > 0 ? ab / Math.sqrt(aa * bb) : 0;
}

/**
 * The geoscience gate (Python GeoscienceGate._partition). For each article
 * vector, pos = cosine to the positive anchor and neg = the highest cosine to
 * any negative anchor (0 without negatives). In this order:
 *   negatives present and neg >= pos -> rejected (a tie rejects)
 *   pos >= minScore                  -> accepted
 *   pos < grayLow                    -> rejected
 *   otherwise                        -> gray (judged by the LLM, or accepted)
 * @returns {{accepted: number[], gray: number[], rejected: number[], scores: {pos: number, neg: number}[]}} indices into `vectors`
 */
export function partitionByAnchors(vectors, positive, negatives, { minScore, grayLow }) {
  const out = { accepted: [], gray: [], rejected: [], scores: [] };
  const useNeg = negatives.length > 0;
  vectors.forEach((vec, i) => {
    const pos = cosine(vec, positive);
    const neg = useNeg ? Math.max(...negatives.map((n) => cosine(vec, n))) : 0;
    out.scores.push({ pos, neg });
    if (useNeg && neg >= pos) out.rejected.push(i);
    else if (pos >= minScore) out.accepted.push(i);
    else if (pos < grayLow) out.rejected.push(i);
    else out.gray.push(i);
  });
  return out;
}

/** neg - pos from which a local-mode article is clearly in another field (see lexicalGray). */
export const LEXICAL_NEGATIVE_MARGIN = 0.15;

/**
 * Lexical (local-mode) vectors only see word overlap with the one-sentence
 * anchors, so an on-topic abstract sharing no words with them scores as noise
 * and the bands reject it. When the LLM judges, local mode widens the gray zone
 * to every article not accepted, except those clearly closer to a negative
 * anchor (neg - pos >= margin).
 * @param {{accepted: number[], scores: {pos: number, neg: number}[]}} part - see partitionByAnchors
 * @returns {number[]} indices of the widened gray zone, in input order
 */
export function lexicalGray(part, margin = LEXICAL_NEGATIVE_MARGIN) {
  const accepted = new Set(part.accepted);
  return part.scores.map((_, i) => i).filter((i) => !accepted.has(i) && part.scores[i].neg - part.scores[i].pos < margin);
}

/** Newest first by publication date (unknown dates last, ties keep their order), capped at `max` (0 = no cap). */
export function newestFirst(works, max = 0) {
  const time = (w) => {
    const t = w.published ? Date.parse(w.published) : NaN;
    return Number.isFinite(t) ? t : -Infinity;
  };
  const sorted = works
    .map((w, i) => [w, time(w), i])
    .sort((a, b) => (b[1] === a[1] ? a[2] - b[2] : b[1] > a[1] ? 1 : -1))
    .map(([w]) => w);
  return max > 0 ? sorted.slice(0, max) : sorted;
}
