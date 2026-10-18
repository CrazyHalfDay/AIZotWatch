// OpenAlex entity lookups for the settings page: journals (sources) by ISSN or
// by name, and authors by name, ORCID or OpenAlex id. Used by 「从文库生成」
// (journal proposals), 「补全学科与影响因子」 and the search boxes of 关注期刊,
// 关注作者 and 顶刊地学.
//
// Every request goes through core/http.js request() (per-host rate limiting,
// retries, cancellation) and carries the OpenAlex key as `api_key` when one is
// configured (see openAlexAuth). Responses are parsed defensively: odd records
// are skipped, never thrown on. A failed request throws (HttpError, network
// error, CancelledRequestError) like the other sources; openAlexErrorText()
// turns it into a short message for the UI.

import { normalizeOpenAlexAuthorId, normalizeOrcid } from "../core/authors.js";
import { buildUrl, getJson } from "../core/http.js";
import { venueKey } from "../core/journals.js";
import { collapseWhitespace, normalizeIssn } from "../core/text.js";
import { openAlexAuth } from "./openalex.js";

const OPENALEX = "https://api.openalex.org";
/** Source fields requested (an unknown field fails the query: see getOpenAlex). */
export const SOURCE_SELECT =
  "id,display_name,issn_l,issn,host_organization_name,country_code,type,works_count,cited_by_count,summary_stats,topics,apc_usd,alternate_titles,abbreviated_title";
export const AUTHOR_SELECT = "id,display_name,orcid,works_count,cited_by_count,last_known_institutions,topics,summary_stats";
/** ISSNs per batched source lookup (OpenAlex allows 50 values in an OR filter). */
export const ISSN_BATCH_SIZE = 50;
// Records per batched lookup: an ISSN can (rarely) belong to two source records
const ISSN_PAGE_SIZE = 100;
const MAX_SEARCH_RESULTS = 25;
const SEARCH_TIMEOUT = 20000;

const isRecord = (value) => !!value && typeof value === "object" && !Array.isArray(value);
const text = (value) => (typeof value === "string" ? collapseWhitespace(value) : "");
const count = (value) => (Number.isFinite(Number(value)) && value !== null && value !== "" ? Number(value) : null);
const CJK = /[㐀-鿿豈-﫿]/;
// Pinyin words of Chinese-language journal titles ("Dizhi Xuebao" = 地质学报)
const PINYIN_TITLE_WORDS = /\b(xuebao|kexue|dizhi|diqiu|zhongguo|tongbao|yanjiu|jinzhan|huaxue|kuangwu|yanshi|dili)\b/i;
// Latin / English words of Chinese-language titles ("Acta Petrologica Sinica" = 岩石学报) that
// English journals from China use as well ("Acta Oceanologica Sinica", "Chinese Chemical Letters")
const CHINESE_TITLE_HINTS = /\b(sinica|chinese)\b/i;
const ENGLISH_EDITION = /\bEnglish\s+(edition|series|version)\b/i;
// International publishers of English journals from China (co-published titles)
const INTERNATIONAL_PUBLISHER =
  /\b(springer|elsevier|wiley|taylor\s*(&|and)\s*francis|nature|oxford university press|cambridge university press|iop|keai|de gruyter|frontiers|mdpi|sage|ieee|american|royal society|bmc|hindawi|wolters kluwer)\b/i;

/** "S1234" / "A5023888391" from an OpenAlex URL or id; "" when absent. */
export function openAlexShortId(value, prefix) {
  const m = String(value ?? "").match(new RegExp(`\\b${prefix}(\\d+)\\b`, "i"));
  return m ? `${prefix}${m[1]}` : "";
}

/** Whether a query is an ISSN ("0016-7037", "0016 7037", "0016-703x"). */
export function looksLikeIssn(query) {
  return /^\s*\d{4}[\s-]?\d{3}[\dXx]\s*$/.test(String(query ?? ""));
}

/**
 * Whitelist category from an OpenAlex topic: its subfield (fallback: field)
 * upper-cased with "_" between words, like the CSV categories
 * ("Geochemistry and Petrology" -> "GEOCHEMISTRY_AND_PETROLOGY"); "" when none.
 */
export function categoryFromTopic(topic) {
  if (!isRecord(topic)) return "";
  const name = text(topic.subfield?.display_name) || text(topic.field?.display_name);
  return name
    .toUpperCase()
    .replace(/&/g, " AND ")
    .replace(/[^\p{L}\p{N}]+/gu, "_")
    .replace(/^_+|_+$/g, "");
}

/** The source's most frequent topic (OpenAlex lists them by count; sorted again defensively). */
function topTopic(topics) {
  const list = (Array.isArray(topics) ? topics : []).filter(isRecord);
  return list.reduce((best, t) => (best === null || (count(t.count) ?? 0) > (count(best.count) ?? 0) ? t : best), null);
}

/**
 * Heuristic for a Chinese-language journal (Chinese core, "(CN)" in the
 * whitelist category), as OpenAlex has no language field for sources:
 * published in China (country_code "CN"), not an "English Edition", and
 * - a title, abbreviation or alternate title in Chinese characters or pinyin
 *   ("岩石学报", "Dizhi Xuebao"), or
 * - a Latin title with "Sinica" / "Chinese" ("Acta Petrologica Sinica",
 *   "Chinese Journal of Geophysics") from a publisher other than the
 *   international ones that co-publish English journals from China
 *   (Springer, Elsevier, Wiley, ...: "Acta Oceanologica Sinica", "Chinese
 *   Chemical Letters" are English).
 * Other English journals published in China ("Science China Earth Sciences")
 * are not marked and keep their impact factor.
 */
export function isChineseSource(src) {
  if (!isRecord(src) || String(src.country_code || "").toUpperCase() !== "CN") return false;
  const names = [src.display_name, src.abbreviated_title, ...(Array.isArray(src.alternate_titles) ? src.alternate_titles : [])].map(text).filter(Boolean);
  if (names.some((name) => ENGLISH_EDITION.test(name))) return false;
  if (names.some((name) => CJK.test(name) || PINYIN_TITLE_WORDS.test(name))) return true;
  return !INTERNATIONAL_PUBLISHER.test(text(src.host_organization_name)) && names.some((name) => CHINESE_TITLE_HINTS.test(name));
}

/** Normalised distinct ISSNs of a source (its list, falling back to issn_l). */
function sourceIssns(src) {
  const list = (Array.isArray(src.issn) ? src.issn : []).map(normalizeIssn).filter(Boolean);
  const issnL = normalizeIssn(src.issn_l);
  if (issnL && !list.includes(issnL)) list.push(issnL);
  return [...new Set(list)];
}

/**
 * Map an OpenAlex source to a journal of the whitelist; null without a name.
 * `impactFactor` is OpenAlex's two-year mean citedness (summary_stats
 * "2yr_mean_citedness", an impact-factor-like metric) rounded to 1 decimal
 * (null when it is 0, i.e. unknown, so the journal keeps the neutral score);
 * `category` comes from the top topic, with " (CN)" for Chinese-language
 * journals (see isChineseSource).
 * @returns {{openalexId, title, issns, issnL, impactFactor, category, subfield, chinese, country, publisher, type, worksCount, citedByCount, titles}|null}
 */
export function mapOpenAlexSource(src) {
  if (!isRecord(src)) return null;
  const title = text(src.display_name);
  if (!title) return null;
  const citedness = count(isRecord(src.summary_stats) ? src.summary_stats["2yr_mean_citedness"] : null);
  const topic = topTopic(src.topics);
  const chinese = isChineseSource(src);
  const base = categoryFromTopic(topic);
  const alternates = (Array.isArray(src.alternate_titles) ? src.alternate_titles : []).map(text).filter(Boolean);
  return {
    openalexId: openAlexShortId(src.id, "S"),
    title,
    issns: sourceIssns(src),
    issnL: normalizeIssn(src.issn_l) || null,
    // 0 means "no citations linked yet" (journals younger than two years): unknown, not a zero IF
    impactFactor: citedness !== null && citedness > 0 ? Math.max(0.1, Math.round(citedness * 10) / 10) : null,
    category: chinese ? (base ? `${base} (CN)` : "(CN)") : base,
    subfield: text(topic?.subfield?.display_name) || text(topic?.field?.display_name),
    chinese,
    country: text(src.country_code).toUpperCase() || null,
    publisher: text(src.host_organization_name) || null,
    type: text(src.type) || null,
    worksCount: count(src.works_count),
    citedByCount: count(src.cited_by_count),
    // Normalised names the source is known by (exact-title matching)
    titles: [...new Set([title, text(src.abbreviated_title), ...alternates].filter(Boolean).map(venueKey).filter(Boolean))],
  };
}

/** Map an OpenAlex author to a search result; null without an id or a name. */
export function mapOpenAlexAuthor(a) {
  if (!isRecord(a)) return null;
  const id = openAlexShortId(a.id, "A");
  const name = text(a.display_name);
  if (!id || !name) return null;
  const institutions = (Array.isArray(a.last_known_institutions) ? a.last_known_institutions : []).filter(isRecord);
  const topic = topTopic(a.topics);
  return {
    id,
    name,
    orcid: normalizeOrcid(a.orcid) || null,
    institution: text(institutions[0]?.display_name) || null,
    country: text(institutions[0]?.country_code).toUpperCase() || null,
    worksCount: count(a.works_count),
    citedByCount: count(a.cited_by_count),
    hIndex: count(isRecord(a.summary_stats) ? a.summary_stats.h_index : null),
    topic: text(topic?.display_name) || null,
  };
}

const resultsOf = (data) => (Array.isArray(data?.results) ? data.results : []);

/**
 * GET an OpenAlex endpoint with the key. An unknown `select` field fails the
 * whole query with HTTP 400: then ask once more for full records.
 */
async function getOpenAlex(http, path, params, { apiKey = "", mailto = "", retry = {}, timeout } = {}) {
  const auth = openAlexAuth({ mailto, apiKey });
  const url = (p) => buildUrl(`${OPENALEX}${path}`, { ...p, ...auth });
  try {
    return await getJson(http, url(params), { timeout, ...retry });
  } catch (e) {
    if (e?.status !== 400 || !params.select) throw e;
    const { select, ...rest } = params;
    return getJson(http, url(rest), { timeout, ...retry });
  }
}

/** Of two records for the same ISSN, the one with more works (OpenAlex has a few duplicates). */
function better(a, b) {
  if (!a) return b;
  return (b.worksCount ?? 0) > (a.worksCount ?? 0) ? b : a;
}

/**
 * Journals by ISSN, in batches of ISSN_BATCH_SIZE: Map(issn -> journal) for
 * every requested ISSN OpenAlex knows (each of a journal's ISSNs maps to it).
 * Throws when a request fails.
 */
export async function lookupSourcesByIssn(http, issns, opts = {}) {
  const wanted = [...new Set((issns || []).map(normalizeIssn).filter(Boolean))];
  const found = new Map();
  for (let i = 0; i < wanted.length; i += ISSN_BATCH_SIZE) {
    http.signal?.throwIfCancelled?.();
    const batch = wanted.slice(i, i + ISSN_BATCH_SIZE);
    const data = await getOpenAlex(http, "/sources", { filter: `issn:${batch.join("|")}`, "per-page": ISSN_PAGE_SIZE, select: SOURCE_SELECT }, opts);
    const requested = new Set(batch);
    for (const record of resultsOf(data)) {
      const journal = mapOpenAlexSource(record);
      if (!journal) continue;
      for (const issn of journal.issns) {
        if (requested.has(issn)) found.set(issn, better(found.get(issn), journal));
      }
    }
  }
  return found;
}

const clampLimit = (limit, fallback = 10) => {
  const n = Math.floor(Number(limit));
  return Number.isFinite(n) && n >= 1 ? Math.min(n, MAX_SEARCH_RESULTS) : fallback;
};

/**
 * Search journals by name (OpenAlex `search`, type journal), or by ISSN when
 * the query is one. Results in OpenAlex's relevance order, distinct.
 */
export async function searchSources(http, query, { limit = 10, ...opts } = {}) {
  const q = collapseWhitespace(String(query ?? ""));
  if (!q) return [];
  const params = looksLikeIssn(q)
    ? { filter: `issn:${normalizeIssn(q)}`, "per-page": clampLimit(limit), select: SOURCE_SELECT }
    : { search: q, filter: "type:journal", "per-page": clampLimit(limit), select: SOURCE_SELECT };
  const data = await getOpenAlex(http, "/sources", params, { timeout: SEARCH_TIMEOUT, ...opts });
  const seen = new Set();
  const out = [];
  for (const record of resultsOf(data)) {
    const journal = mapOpenAlexSource(record);
    const key = journal?.openalexId || journal?.title;
    if (!journal || seen.has(key)) continue;
    seen.add(key);
    out.push(journal);
  }
  return out;
}

/**
 * The journal named exactly `title` (normalised like resolveJournalIssns:
 * case, punctuation and a leading "The" ignored), matched against OpenAlex's
 * name, abbreviation and alternate titles; null when no result matches.
 * Among several matches the main name wins, then the larger journal.
 */
export async function findSourceByTitle(http, title, opts = {}) {
  const target = venueKey(title);
  if (!target) return null;
  const results = await searchSources(http, title, { limit: 10, ...opts });
  const exact = results.filter((j) => j.titles.includes(target));
  if (!exact.length) return null;
  const main = exact.filter((j) => venueKey(j.title) === target);
  return (main.length ? main : exact).reduce((a, b) => better(a, b));
}

/**
 * Search authors by name; an ORCID iD or an OpenAlex author id (bare or as a
 * URL) looks that author up instead.
 */
export async function searchAuthors(http, query, { limit = 10, ...opts } = {}) {
  const q = collapseWhitespace(String(query ?? ""));
  if (!q) return [];
  const orcid = normalizeOrcid(q);
  const authorId = /^(?:https?:\/\/(?:api\.)?openalex\.org\/(?:authors\/)?)?A\d{4,}$/i.test(q) ? normalizeOpenAlexAuthorId(q) : "";
  let records;
  if (authorId) {
    try {
      records = [await getOpenAlex(http, `/authors/${authorId}`, { select: AUTHOR_SELECT }, { timeout: SEARCH_TIMEOUT, ...opts })];
    } catch (e) {
      if (e?.status === 404) return [];
      throw e;
    }
  } else {
    const params = orcid
      ? { filter: `orcid:https://orcid.org/${orcid}`, "per-page": clampLimit(limit), select: AUTHOR_SELECT }
      : { search: q, "per-page": clampLimit(limit), select: AUTHOR_SELECT };
    records = resultsOf(await getOpenAlex(http, "/authors", params, { timeout: SEARCH_TIMEOUT, ...opts }));
  }
  const seen = new Set();
  return records.map(mapOpenAlexAuthor).filter((a) => a && !seen.has(a.id) && seen.add(a.id));
}

/**
 * A failed OpenAlex request as a short message for the settings page; the
 * cancellation error is passed through unchanged by callers (check e.cancelled).
 */
export function openAlexErrorText(e, { keyed = false } = {}) {
  const status = e?.status;
  if (status === 401 || status === 403) return `OpenAlex API Key 无效（HTTP ${status}），请在「学术数据接口」中检查`;
  if (status === 429) {
    return keyed ? "OpenAlex 请求过于频繁或今日额度已用完（HTTP 429），请稍后再试" : "OpenAlex 免费额度已用完（HTTP 429），请在「学术数据接口」中填写 API Key";
  }
  if (Number.isFinite(status)) return `OpenAlex 出错（HTTP ${status}）`;
  return String(e?.message || e || "未知错误");
}
