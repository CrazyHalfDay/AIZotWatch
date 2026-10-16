// Settings model. Preferences live under "extensions.zotwatch.*" (see
// addon/prefs.js); numbers are stored as strings so the preference pane can
// bind plain text inputs, and are parsed here with defaults.

import { parseFollowedAuthors } from "./core/authors.js";
import { FLAGSHIP_DEFAULTS, normalizeIssnList, parseAnchorLines } from "./core/flagship.js";
import { parseList } from "./core/text.js";

export const PREF_PREFIX = "extensions.zotwatch.";
// arXiv categories used when the field is left blank (the pref's default)
export const DEFAULT_ARXIV_CATEGORIES = ["physics.geo-ph", "astro-ph.EP"];

// Per-provider calibration: local TF-IDF cosines are much lower than neural ones.
// The flagship gate bands were tuned on DashScope text-embedding-v4 (the
// Python config); the local values are rough (lexical overlap with one sentence).
const PROVIDER_CALIBRATION = {
  local: { minSimilarity: 0.06, minMustRead: 0.22, minConsider: 0.12, flagshipMinScore: 0.12, flagshipGrayLow: 0.04 },
  default: { minSimilarity: 0.2, minMustRead: 0.55, minConsider: 0.35, flagshipMinScore: 0.3, flagshipGrayLow: 0.22 },
};

/**
 * Values used when the corresponding fields are left blank: the provider-calibrated
 * thresholds and the flagship defaults (placeholders and "restore default" in the UI).
 */
export function autoValues(provider) {
  const calib = PROVIDER_CALIBRATION[provider || "local"] || PROVIDER_CALIBRATION.default;
  return {
    provider: provider || "local",
    minSimilarity: calib.minSimilarity,
    flagshipMinScore: calib.flagshipMinScore,
    flagshipGrayLow: calib.flagshipGrayLow,
    flagship: {
      issns: [...FLAGSHIP_DEFAULTS.issns],
      positiveAnchor: FLAGSHIP_DEFAULTS.positiveAnchor,
      negativeAnchors: [...FLAGSHIP_DEFAULTS.negativeAnchors],
      llmBoundary: FLAGSHIP_DEFAULTS.llmBoundary,
      maxResults: FLAGSHIP_DEFAULTS.maxResults,
    },
  };
}

function num(value, fallback, { min = -Infinity, max = Infinity, int = false } = {}) {
  if (value === undefined || value === null || String(value).trim() === "") return fallback;
  const n = int ? parseInt(value, 10) : parseFloat(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function bool(value, fallback) {
  if (typeof value === "boolean") return value;
  if (value === "true") return true;
  if (value === "false") return false;
  return fallback;
}

function str(value, fallback = "") {
  return typeof value === "string" ? value.trim() : fallback;
}

/** Free text that falls back to `fallback` when unset or blank (clearing a field restores the default). */
function textOr(value, fallback) {
  return typeof value === "string" && value.trim() ? value : fallback;
}

/**
 * Build the settings object from a pref getter.
 * @param {(key: string) => any} get - returns the raw pref value for a key without the prefix
 */
export function loadSettings(get) {
  const provider = str(get("embedding.provider"), "local") || "local";
  const calib = PROVIDER_CALIBRATION[provider] || PROVIDER_CALIBRATION.default;
  const followedAuthors = parseFollowedAuthors(typeof get("followed.authors") === "string" ? get("followed.authors") : "");
  const flagshipMinScore = num(get("flagship.minScore"), calib.flagshipMinScore, { min: 0, max: 1 });
  const arxivCategories = parseList(get("sources.arxiv.categories"));

  return {
    embedding: {
      provider,
      apiKey: str(get("embedding.apiKey")),
      model: str(get("embedding.model")),
      baseURL: str(get("embedding.baseURL")),
      // Only DashScope uses it; text-embedding-v4 accepts 64 to 2048 (fixed sizes, see the settings page)
      dimensions: num(get("embedding.dimensions"), 1024, { min: 64, max: 2048, int: true }),
      batchSize: num(get("embedding.batchSize"), 0, { min: 0, max: 1000, int: true }),
      concurrency: num(get("embedding.concurrency"), 3, { min: 1, max: 8, int: true }),
    },
    llm: {
      enabled: bool(get("llm.enabled"), false),
      baseURL: str(get("llm.baseURL")),
      apiKey: str(get("llm.apiKey")),
      model: str(get("llm.model"), "deepseek/deepseek-chat") || "deepseek/deepseek-chat",
      translateTitles: bool(get("llm.translateTitles"), true),
      // Label the library's research directions (clusters) once; labels are reused across runs
      topicLabels: bool(get("llm.topicLabels"), true),
      summaryTopN: num(get("llm.summaryTopN"), 10, { min: 0, max: 100, int: true }),
      temperature: 0.3,
      maxTokens: 2048,
    },
    sources: {
      crossref: {
        enabled: bool(get("sources.crossref.enabled"), true),
        mailto: str(get("sources.crossref.mailto")),
        daysBack: num(get("sources.crossref.daysBack"), 7, { min: 1, max: 60, int: true }),
        maxResults: num(get("sources.crossref.maxResults"), 3000, { min: 1, max: 10000, int: true }),
      },
      arxiv: {
        enabled: bool(get("sources.arxiv.enabled"), false),
        // Blank means the default categories (as the field's placeholder says), not "fetch nothing"
        categories: arxivCategories.length ? arxivCategories : [...DEFAULT_ARXIV_CATEGORIES],
        daysBack: num(get("sources.arxiv.daysBack"), 7, { min: 1, max: 60, int: true }),
        maxResults: num(get("sources.arxiv.maxResults"), 500, { min: 1, max: 2000, int: true }),
      },
      eartharxiv: {
        enabled: bool(get("sources.eartharxiv.enabled"), false),
        daysBack: num(get("sources.eartharxiv.daysBack"), 14, { min: 1, max: 60, int: true }),
        maxResults: num(get("sources.eartharxiv.maxResults"), 500, { min: 1, max: 5000, int: true }),
      },
      openalexAbstracts: bool(get("sources.openalexAbstracts"), true),
    },
    // Abstract fallback tiers (sources/abstracts.js); mailto is sources.crossref.mailto
    abstracts: {
      crossref: bool(get("abstracts.crossref"), true),
      openalex: bool(get("sources.openalexAbstracts"), true),
      elsevierKey: str(get("abstracts.elsevierKey")),
      springerKey: str(get("abstracts.springerKey")),
    },
    followed: {
      enabled: bool(get("followed.enabled"), false),
      // Parsed lines (see parseFollowedAuthors); unparseable lines are reported by the run
      authors: followedAuthors.authors,
      invalid: followedAuthors.invalid,
      // Publication-date window searched on every run (a seen-set prevents repeats)
      daysBack: num(get("followed.daysBack"), 60, { min: 7, max: 365, int: true }),
    },
    // Flagship geoscience track (core/flagship.js); blank text fields use the defaults
    flagship: {
      enabled: bool(get("flagship.enabled"), false),
      issns: normalizeIssnList(parseList(textOr(get("flagship.issns"), FLAGSHIP_DEFAULTS.issns.join("\n")))),
      positiveAnchor: textOr(get("flagship.positiveAnchor"), FLAGSHIP_DEFAULTS.positiveAnchor).trim(),
      negativeAnchors: parseAnchorLines(textOr(get("flagship.negativeAnchors"), FLAGSHIP_DEFAULTS.negativeAnchors.join("\n"))),
      // Positive-anchor cosine >= minScore accepts, < grayLow rejects, in between is the gray zone
      minScore: flagshipMinScore,
      grayLow: Math.min(num(get("flagship.grayLow"), calib.flagshipGrayLow, { min: 0, max: 1 }), flagshipMinScore),
      llmFallback: bool(get("flagship.llmFallback"), true),
      llmBoundary: textOr(get("flagship.llmBoundary"), FLAGSHIP_DEFAULTS.llmBoundary).trim(),
      llmBatchSize: 20,
      // Articles shown per run, newest first (0 = no cap)
      maxResults: num(get("flagship.maxResults"), FLAGSHIP_DEFAULTS.maxResults, { min: 0, max: 200, int: true }),
    },
    journals: {
      whitelist: typeof get("journals.whitelist") === "string" ? get("journals.whitelist") : "",
      autoFromLibrary: bool(get("journals.autoFromLibrary"), true),
      autoTopN: num(get("journals.autoTopN"), 40, { min: 1, max: 300, int: true }),
      autoMinItems: num(get("journals.autoMinItems"), 2, { min: 1, max: 100, int: true }),
    },
    scoring: {
      topK: num(get("scoring.topK"), 30, { min: 1, max: 500, int: true }),
      recentDays: num(get("scoring.recentDays"), 7, { min: 0, max: 365, int: true }),
      maxPreprintRatio: num(get("scoring.maxPreprintRatio"), 0.7, { min: 0, max: 1 }),
      requireAbstract: bool(get("scoring.requireAbstract"), true),
      includeKeywords: parseList(get("scoring.includeKeywords")),
      excludeKeywords: parseList(get("scoring.excludeKeywords")),
      gateEnabled: true,
      minSimilarity: num(get("scoring.minSimilarity"), calib.minSimilarity, { min: 0, max: 1 }),
      microWeight: num(get("scoring.microWeight"), 0.65, { min: 0, max: 1 }),
      knn: num(get("scoring.knn"), 5, { min: 1, max: 50, int: true }),
      similarityWeight: num(get("scoring.similarityWeight"), 0.8, { min: 0, max: 1 }),
      ifWeight: num(get("scoring.ifWeight"), 0.2, { min: 0, max: 1 }),
      thresholds: {
        mode: str(get("scoring.thresholdMode"), "dynamic") === "fixed" ? "fixed" : "dynamic",
        mustRead: 0.75,
        consider: 0.55,
        mustReadPercentile: 95,
        considerPercentile: 70,
        minMustRead: calib.minMustRead,
        minConsider: calib.minConsider,
      },
      temporal: {
        halflifeDays: num(get("scoring.temporalHalflifeDays"), 0, { min: 0, max: 36500 }),
        minWeight: 0.05,
      },
      clustering: { minClusters: 5, maxClusters: 35, tolerance: 0.2 },
      journal: { preprintScore: 0.6, chineseCoreScore: 0.7, unknownScore: 0.3, logBase: 25 },
    },
    output: {
      collection: str(get("output.collection"), "ZotWatch 推荐"),
      tag: str(get("output.tag"), "ZotWatch"),
      summaryNote: bool(get("output.summaryNote"), true),
      useTranslator: bool(get("output.useTranslator"), true),
    },
    schedule: {
      autoRun: bool(get("schedule.autoRun"), true),
      intervalHours: num(get("schedule.intervalHours"), 24, { min: 1, max: 24 * 30 }),
    },
    inbox: {
      retentionDays: num(get("inbox.retentionDays"), 30, { min: 1, max: 365, int: true }),
    },
  };
}
