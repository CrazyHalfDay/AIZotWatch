// Topic survey (主题文献调研): stages, depth presets, input validation and the
// cost estimate shown next to the depth choice. Zotero-independent.

import { collapseWhitespace } from "../core/text.js";

/** The stages in run order, with the labels of the running view's stepper. */
export const SURVEY_STAGES = Object.freeze([
  { key: "seeds", label: "种子" },
  { key: "queries", label: "检索词" },
  { key: "expand", label: "扩展" },
  { key: "dedupe", label: "去重" },
  { key: "rank", label: "排序" },
  { key: "enrich", label: "补全" },
  { key: "read", label: "精读" },
  { key: "themes", label: "主题" },
  { key: "write", label: "综述" },
]);
export const STAGE_KEYS = SURVEY_STAGES.map((s) => s.key);

/**
 * Depth presets: `pool` caps the candidate pool (the best scored papers),
 * `scored` the papers found that are scored against the topic and the seeds
 * before that cut (it only bounds the embedding work), `deepRead` the papers
 * the LLM reads; the per-request limits size the expansion so that the pool
 * can fill up (search results per query and source, references / citations
 * per seed, recommendations).
 */
export const SURVEY_DEPTHS = Object.freeze({
  quick: { key: "quick", label: "快速", pool: 150, scored: 600, deepRead: 15, perQuery: 25, crossrefPerQuery: 15, perSeed: 25, recommendations: 50, maxQueries: 6 },
  standard: { key: "standard", label: "标准", pool: 400, scored: 1500, deepRead: 40, perQuery: 50, crossrefPerQuery: 25, perSeed: 50, recommendations: 100, maxQueries: 8 },
  deep: { key: "deep", label: "深度", pool: 1000, scored: 3000, deepRead: 80, perQuery: 100, crossrefPerQuery: 50, perSeed: 100, recommendations: 200, maxQueries: 10 },
});
export const DEFAULT_DEPTH = "standard";

export const MAX_TOPIC_CHARS = 200;
export const MAX_DESCRIPTION_CHARS = 2000;
export const MAX_KEYWORDS = 12;
export const MAX_KEYWORD_CHARS = 100;
export const MAX_SEEDS = 20;
// Theme sections of the review (k-means k range)
export const MIN_THEMES = 3;
export const MAX_THEMES = 6;

const cut = (text, max) => Array.from(text).slice(0, max).join("");

/** Keywords from a string (commas, semicolons, newlines, Chinese punctuation) or an array. */
export function parseKeywords(value) {
  const raw = Array.isArray(value) ? value : String(value ?? "").split(/[\n,;，；、]+/);
  const seen = new Set();
  const out = [];
  for (const entry of raw) {
    const k = cut(collapseWhitespace(typeof entry === "string" ? entry : ""), MAX_KEYWORD_CHARS);
    if (!k || seen.has(k.toLowerCase())) continue;
    seen.add(k.toLowerCase());
    out.push(k);
    if (out.length >= MAX_KEYWORDS) break;
  }
  return out;
}

/**
 * Validated survey input; throws an Error with a Chinese message for the form
 * when it is unusable (no topic, or neither keywords nor seeds).
 * @param {object} input - {topic, description, keywords (string or array), seedKeys, depth, yearFrom, sources}
 * @returns {{topic, description, keywords: string[], seedKeys: string[], depth, yearFrom: number|null, sources: {openalex, s2, crossref}}}
 */
export function normalizeSurveyInput(input, { now = Date.now() } = {}) {
  const src = input && typeof input === "object" ? input : {};
  const topic = cut(collapseWhitespace(typeof src.topic === "string" ? src.topic : ""), MAX_TOPIC_CHARS);
  if (!topic) throw new Error("请填写调研主题");
  const description = cut(String(typeof src.description === "string" ? src.description : "").trim(), MAX_DESCRIPTION_CHARS);
  const keywords = parseKeywords(src.keywords);
  const seedKeys = [
    ...new Set((Array.isArray(src.seedKeys) ? src.seedKeys : []).map((k) => (typeof k === "string" ? k.trim() : "")).filter((k) => /^[A-Za-z0-9]{1,32}$/.test(k))),
  ].slice(0, MAX_SEEDS);
  if (!keywords.length && !seedKeys.length) throw new Error("请至少填写一个关键词或选择一篇种子文献");
  const depth = Object.hasOwn(SURVEY_DEPTHS, src.depth) ? src.depth : DEFAULT_DEPTH;
  const year = Number.parseInt(src.yearFrom, 10);
  const thisYear = new Date(now).getUTCFullYear();
  const yearFrom = Number.isFinite(year) && year >= 1800 && year <= thisYear ? year : null;
  const s = src.sources && typeof src.sources === "object" ? src.sources : {};
  const sources = { openalex: s.openalex !== false, s2: s.s2 !== false, crossref: s.crossref !== false };
  if (!sources.openalex && !sources.s2 && !sources.crossref) throw new Error("请至少选择一个数据源");
  return { topic, description, keywords, seedKeys, depth, yearFrom, sources };
}

/**
 * Rough cost of a survey for the depth choice: AI calls, API requests and
 * minutes (all estimates; cached extractions and empty sources make it cheaper).
 * @param {string} depth - quick | standard | deep
 * @param {object} [options] - {seeds (default 3), llm (default true), sources, keywords}
 * @returns {{depth, pool, deepRead, aiCalls, apiRequests, minutes, text}}
 */
export function surveyEstimate(depth, { seeds = 3, llm = true, sources = {}, keywords = 3 } = {}) {
  const preset = SURVEY_DEPTHS[depth] || SURVEY_DEPTHS[DEFAULT_DEPTH];
  const s = Math.max(0, Math.min(MAX_SEEDS, Math.floor(Number(seeds)) || 0));
  const on = { openalex: sources.openalex !== false, s2: sources.s2 !== false, crossref: sources.crossref !== false };
  const queries = llm ? preset.maxQueries : Math.min(preset.maxQueries, 1 + Math.max(0, Math.floor(Number(keywords)) || 0));
  const enrich = Math.ceil(preset.deepRead * 1.2);
  let openalex = 0;
  let s2 = 0;
  let crossref = 0;
  if (on.openalex) openalex = s + queries + s * 3 + Math.ceil(enrich / 50);
  if (on.s2) s2 = (s ? 2 + s * 2 : 0) + queries + 2;
  if (on.crossref) crossref = queries;
  // The abstract chain asks Crossref once per DOI that still lacks an abstract (about half)
  crossref += Math.ceil(enrich / 2);
  const apiRequests = openalex + s2 + crossref;
  const themes = MAX_THEMES - 1;
  const aiCalls = llm ? 1 + preset.deepRead + 1 + themes + 1 : 0;
  // Seconds: Semantic Scholar is paced at 1/s, Crossref 3/s, OpenAlex 8/s; LLM calls about 10 s
  // each (two reads at a time) and 30 s per written section; embeddings in batches
  const seconds =
    s2 * 1.2 + crossref / 3 + openalex / 6 + preset.scored / 40 + (llm ? 10 + (preset.deepRead * 10) / 2 + 10 + (themes + 1) * 30 : 0);
  const minutes = Math.max(1, Math.round(seconds / 60));
  const text = llm ? `约 ${aiCalls} 次 AI 调用、${apiRequests} 次接口请求、预计 ${minutes} 分钟` : `约 ${apiRequests} 次接口请求、预计 ${minutes} 分钟（未启用大模型，不生成综述）`;
  return { depth: preset.key, pool: preset.pool, deepRead: preset.deepRead, aiCalls, apiRequests, minutes, text };
}
