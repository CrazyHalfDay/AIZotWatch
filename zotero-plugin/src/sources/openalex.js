// OpenAlex abstracts: many publishers (e.g. Elsevier) do not deposit abstracts
// in Crossref, but OpenAlex stores them as an inverted index. Used by the
// OpenAlex tier of the abstract fallback (sources/abstracts.js), one filter
// query per 50 DOIs, and by the followed-authors source.
//
// Since February 2026 OpenAlex expects a (free) API key: keyless clients get a
// very small daily budget and cannot page with cursors. Every request to
// api.openalex.org carries `api_key` when one is configured (see openAlexAuth);
// redactUrl() in core/http.js keeps it out of logs and error messages.

import { apiUsage, buildUrl, limitedRequest } from "../core/http.js";
import { collapseWhitespace, normalizeDoi } from "../core/text.js";

const OPENALEX_WORKS = "https://api.openalex.org/works";
const OPENALEX_RATE_LIMIT = "https://api.openalex.org/rate-limit";
const CHECK_TIMEOUT = 20000;
// Abstracts are far shorter; larger positions are malformed and would make a huge sparse array
const MAX_ABSTRACT_WORDS = 20000;
/** DOIs per batched abstract lookup (OpenAlex allows up to 50 values in an OR filter). */
export const OPENALEX_BATCH_SIZE = 50;
// Records per batched lookup (the OpenAlex maximum): several records may share a DOI
const OPENALEX_PAGE_SIZE = 200;

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

/**
 * Query parameters of every OpenAlex request: the polite-pool address and the
 * API key (blank values are left out by buildUrl).
 */
export function openAlexAuth({ mailto = "", apiKey = "" } = {}) {
  return { mailto, api_key: String(apiKey || "").trim() };
}

/** URL of one batched abstract lookup for up to OPENALEX_BATCH_SIZE normalized DOIs. */
export function openAlexAbstractsUrl(dois, mailto = "", apiKey = "") {
  return buildUrl(OPENALEX_WORKS, {
    filter: `doi:${dois.map((doi) => `https://doi.org/${doi}`).join("|")}`,
    select: "doi,abstract_inverted_index",
    "per-page": OPENALEX_PAGE_SIZE,
    ...openAlexAuth({ mailto, apiKey }),
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

// ------------------------------------------------------------- key and budget

const finite = (value) => {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

/**
 * Budget from the /rate-limit endpoint ({rate_limit: {daily_budget_usd,
 * daily_used_usd, daily_remaining_usd, resets_at, …}}) in the shape of
 * apiUsage.openalex; null when the body carries none of it.
 */
export function parseOpenAlexRateLimit(data, now = Date.now()) {
  const r = data && typeof data === "object" ? (data.rate_limit && typeof data.rate_limit === "object" ? data.rate_limit : data) : null;
  if (!r) return null;
  const usage = {
    remainingUsd: finite(r.daily_remaining_usd),
    limitUsd: finite(r.daily_budget_usd),
    remaining: finite(r.daily_remaining ?? r.credits_remaining),
    limit: finite(r.daily_limit ?? r.credits_limit),
  };
  if (Object.values(usage).every((v) => v === null)) return null;
  if (usage.remainingUsd === null && usage.limitUsd !== null && finite(r.daily_used_usd) !== null) {
    usage.remainingUsd = Math.max(0, usage.limitUsd - finite(r.daily_used_usd));
  }
  return { ...usage, creditsUsed: null, keyed: true, resetsAt: typeof r.resets_at === "string" ? r.resets_at : null, at: now };
}

function groupDigits(n) {
  return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * "今日剩余 $0.95 / $1.00" or "今日剩余 99,990 / 100,000 次额度" ("今日剩余 $0.95"
 * with `compact`, for narrow rows); "" when nothing is known.
 */
export function openAlexUsageText(usage, { compact = false } = {}) {
  if (!usage || typeof usage !== "object") return "";
  const usd = (v) => `$${v < 0.1 && v > 0 ? v.toFixed(3) : v.toFixed(2)}`;
  if (Number.isFinite(usage.remainingUsd)) {
    return `今日剩余 ${usd(usage.remainingUsd)}${!compact && Number.isFinite(usage.limitUsd) ? ` / ${usd(usage.limitUsd)}` : ""}`;
  }
  if (Number.isFinite(usage.remaining)) {
    return `今日剩余 ${groupDigits(usage.remaining)}${!compact && Number.isFinite(usage.limit) ? ` / ${groupDigits(usage.limit)}` : ""} 次额度`;
  }
  return "";
}

/**
 * Connection check of OpenAlex ({key: "openalex", name, ok, detail}): with a
 * key, the /rate-limit endpoint (key accepted, today's budget; a missing
 * endpoint falls back to a one-record query); without one, a one-record query
 * (ok null: reachable, but without a key the daily budget is tiny).
 */
export async function checkOpenAlexRow(http, { apiKey = "", mailto = "", now = Date.now() } = {}) {
  const key = String(apiKey || "").trim();
  const row = { key: "openalex", name: "OpenAlex", ok: false, detail: "" };
  const rejected = (status) => status === 401 || status === 403;
  try {
    if (key) {
      const res = await limitedRequest(http, { url: buildUrl(OPENALEX_RATE_LIMIT, { api_key: key }), timeout: CHECK_TIMEOUT });
      if (rejected(res.status)) return { ...row, detail: `Key 无效（HTTP ${res.status}）` };
      if (res.status >= 200 && res.status < 300) {
        const usage = parseOpenAlexRateLimit(res.data, now);
        if (usage) apiUsage.openalex = usage;
        const text = openAlexUsageText(usage || apiUsage.openalex);
        return { ...row, ok: true, detail: text ? `Key 有效，${text}` : "Key 有效" };
      }
      if (res.status !== 404) return { ...row, detail: res.status === 429 ? "今日额度已用完或请求过于频繁（HTTP 429）" : `HTTP ${res.status}` };
    }
    const res = await limitedRequest(http, { url: buildUrl(OPENALEX_WORKS, { "per-page": 1, select: "id", ...openAlexAuth({ mailto, apiKey: key }) }), timeout: CHECK_TIMEOUT });
    if (rejected(res.status)) return { ...row, detail: `Key 无效（HTTP ${res.status}）` };
    if (res.status === 429) return { ...row, detail: key ? "今日额度已用完或请求过于频繁（HTTP 429）" : "无 Key 的免费额度已用完（HTTP 429），请填写 API Key" };
    if (res.status < 200 || res.status >= 300) return { ...row, detail: `HTTP ${res.status}` };
    const text = openAlexUsageText(apiUsage.openalex);
    if (!key) return { ...row, ok: null, detail: "可访问，但未配置 API Key（每日免费额度很小）" };
    return { ...row, ok: true, detail: text ? `Key 有效，${text}` : "Key 有效" };
  } catch (e) {
    if (e?.cancelled) throw e;
    return { ...row, detail: String(e?.message || e) };
  }
}
