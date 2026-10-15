// Abstract fallback (摘要补全): fills in missing abstracts by DOI through
// Crossref -> Elsevier -> Springer Nature -> OpenAlex, the first abstract of at
// least 80 characters winning (port of Python infrastructure/enrichment/
// api_fallback.py and the API path of pipeline/enrich.py).
//
// Differences from Python, all deliberate:
// - transient failures (429, 5xx, network, timeouts) are never negative-cached,
//   and a negative entry remembers which tiers missed, so adding a publisher key
//   retries the DOIs it can serve
// - a rejected publisher key (401/403) disables that publisher for the run with
//   one reported error, and a provider failing repeatedly is skipped for the run
// - OpenAlex is queried in batches of 50 DOIs instead of once per DOI
// - lookups that would re-read the record a work was just fetched from (Crossref
//   for Crossref works, OpenAlex for OpenAlex works) are skipped
// - a leading "Abstract" label is removed before the length check

import { buildUrl, getJson } from "../core/http.js";
import { cleanHtml, normalizeDoi } from "../core/text.js";
import { OPENALEX_BATCH_SIZE, batchableDoi, openAlexAbstractsUrl, parseOpenAlexAbstracts } from "./openalex.js";

const DAY = 86400000;
/** Shorter texts are treated as missing (placeholders, truncated snippets). */
export const MIN_ABSTRACT_CHARS = 80;
/** Cache file in the plugin storage: {version: 1, entries: {<doi>: {a: string|null, s, exp}}}. */
export const ABSTRACT_CACHE_FILE = "abstracts-cache.json";
const POSITIVE_TTL = 30 * DAY;
const NEGATIVE_TTL = 7 * DAY;
const CACHE_MAX_ENTRIES = 5000;
const REQUEST_TIMEOUT = 20000;
// Per-DOI chains running at once; Elsevier and Springer share a smaller limit
const CONCURRENCY = 6;
const PUBLISHER_CONCURRENCY = 2;
const OPENALEX_CONCURRENCY = 2;
// A provider failing this many times in a row is skipped for the rest of the run
const MAX_CONSECUTIVE_FAILURES = 3;

const CROSSREF_WORK = "https://api.crossref.org/works/";
const ELSEVIER_ARTICLE = "https://api.elsevier.com/content/article/doi/";
const SPRINGER_META = "https://api.springernature.com/meta/v2/json";
// DOIs used by the key check of testConnections (a 404 still proves the key works)
const TEST_DOIS = { elsevier: "10.1016/j.gca.2019.01.001", springer: "10.1038/s41561-019-0300-9" };

/** Tiers in the order they are tried. */
export const TIERS = ["crossref", "elsevier", "springer", "openalex"];
const PROVIDER_NAMES = { crossref: "Crossref", elsevier: "Elsevier", springer: "Springer Nature", openalex: "OpenAlex" };
// Source of works whose record at this service was just fetched with its abstract
const SAME_RECORD_SOURCE = { crossref: "crossref", openalex: "openalex" };

/** DOI as a URL path: percent-encoded except for "/" (DOIs may contain "#", "?", "%" …). */
export function doiPath(doi) {
  return encodeURIComponent(doi).replace(/%2F/gi, "/");
}

/** Publisher API serving a DOI by its prefix ("elsevier" | "springer" | null). */
export function publisherOf(doi) {
  const d = String(doi || "").trim().toLowerCase();
  if (d.startsWith("10.1016/")) return "elsevier";
  if (d.startsWith("10.1007/") || d.startsWith("10.1038/")) return "springer";
  return null;
}

// "Abstract"/"Summary" label in front of the text: with punctuation, or followed by a
// capitalised sentence ("Abstract The …", "AbstractThe …") but not "Abstract reasoning …"
const LEADING_LABEL = /^(?:(?:Abstract|Summary)(?:\s*[:：.]\s*|\s*(?=[A-Z0-9]))|(?:ABSTRACT|SUMMARY)(?:\s*[:：.]\s*|\s+(?=[A-Z0-9])))/;

/** Cleaned abstract text (markup, entities, a leading label removed); null when too short or not a string. */
export function acceptAbstract(value) {
  if (typeof value !== "string") return null;
  const text = (cleanHtml(value) || "").replace(LEADING_LABEL, "").trim();
  return text.length >= MIN_ABSTRACT_CHARS ? text : null;
}

/** String leaves of a value joined with spaces (Springer may return a structured abstract). */
function textLeaves(value, depth = 0) {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object" || depth > 6) return "";
  return Object.values(value)
    .map((v) => textLeaves(v, depth + 1))
    .filter(Boolean)
    .join(" ");
}

export function crossrefWorkUrl(doi, mailto = "") {
  return buildUrl(`${CROSSREF_WORK}${doiPath(doi)}`, { mailto });
}

/** Elsevier Article Retrieval request (the key travels in a header). */
export function elsevierRequest(doi, key) {
  return {
    url: `${ELSEVIER_ARTICLE}${doiPath(doi)}?view=META_ABS`,
    headers: { "X-ELS-APIKey": key, Accept: "application/json" },
  };
}

export function springerUrl(doi, key) {
  return buildUrl(SPRINGER_META, { q: `doi:${doi}`, api_key: key });
}

export function parseCrossrefAbstract(data) {
  return acceptAbstract(data?.message?.abstract);
}

export function parseElsevierAbstract(data) {
  return acceptAbstract(data?.["full-text-retrieval-response"]?.coredata?.["dc:description"]);
}

export function parseSpringerAbstract(data) {
  const records = Array.isArray(data?.records) ? data.records : [];
  return acceptAbstract(textLeaves(records[0]?.abstract) || null);
}

function hasAbstract(work) {
  return typeof work.abstract === "string" ? work.abstract.trim() !== "" : !!work.abstract;
}

/** Tiers that can serve a DOI with the current settings, in chain order. */
function applicableTiers(doi, enabled) {
  const publisher = publisherOf(doi);
  return TIERS.filter((tier) => {
    if (!enabled[tier]) return false;
    if (tier === "elsevier" || tier === "springer") return publisher === tier;
    if (tier === "openalex") return batchableDoi(doi);
    return true;
  });
}

/** Concurrency limiter: run(fn) waits for a free slot; a released slot passes straight to the next waiter. */
function createLimiter(limit) {
  let active = 0;
  const queue = [];
  return async (fn) => {
    if (active < limit) active++;
    else await new Promise((resolve) => queue.push(resolve));
    try {
      return await fn();
    } finally {
      const next = queue.shift();
      if (next) next();
      else active--;
    }
  };
}

/**
 * Process items with bounded concurrency. Item functions handle their own
 * errors; anything they throw (a cancellation) stops new items, and the first
 * such error is rethrown once every worker has finished.
 */
async function runPool(items, limit, fn) {
  let next = 0;
  let stopped = false;
  const worker = async () => {
    while (!stopped && next < items.length) {
      const item = items[next++];
      try {
        await fn(item);
      } catch (e) {
        stopped = true;
        throw e;
      }
    }
  };
  const settled = await Promise.allSettled(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
  const failed = settled.find((r) => r.status === "rejected");
  if (failed) throw failed.reason;
}

function validEntry(entry, now) {
  if (!entry || typeof entry !== "object" || !(Number(entry.exp) > now)) return false;
  return typeof entry.a === "string" ? entry.a.length > 0 : entry.a === null && Array.isArray(entry.s);
}

async function loadCache(storage, now) {
  let data = null;
  try {
    data = await storage.readJSON(ABSTRACT_CACHE_FILE);
  } catch {
    // Corrupt file: start over (it is rewritten as soon as something new is found)
  }
  const raw = data?.version === 1 && data.entries && typeof data.entries === "object" ? data.entries : {};
  const entries = new Map();
  let dropped = 0;
  for (const [doi, entry] of Object.entries(raw)) {
    if (validEntry(entry, now)) entries.set(doi, entry);
    else dropped++;
  }
  return { entries, dirty: dropped > 0 };
}

/** Write live entries; over the cap, keep the ones that stay valid longest. Never throws. */
async function saveCache(storage, cache, now, log) {
  const live = [...cache.entries].filter(([, entry]) => entry.exp > now);
  if (live.length > CACHE_MAX_ENTRIES) {
    live.sort((a, b) => b[1].exp - a[1].exp);
    live.length = CACHE_MAX_ENTRIES;
  }
  try {
    await storage.writeJSON(ABSTRACT_CACHE_FILE, { version: 1, entries: Object.fromEntries(live) });
  } catch (e) {
    log(`Abstract cache save failed: ${e?.message || e}`);
  }
}

function shortReason(error) {
  return error?.status ? `HTTP ${error.status}` : String(error?.message || error).slice(0, 80);
}

/**
 * Fill in missing abstracts in place. Works without an abstract but with a DOI
 * are grouped by DOI; each DOI is looked up once per tier.
 * @param {object[]} works - works of any track; `abstract` is set on the ones filled
 * @param {object} options
 * @param {string} [options.mailto] - polite-pool address for Crossref and OpenAlex
 * @param {boolean} [options.crossref] - per-DOI Crossref tier
 * @param {boolean} [options.openalex] - batched OpenAlex tier
 * @param {string} [options.elsevierKey] - enables Elsevier for 10.1016/ DOIs
 * @param {string} [options.springerKey] - enables Springer Nature for 10.1007/ and 10.1038/ DOIs
 * @param {object|null} [options.cache] - storage ({readJSON, writeJSON}) keeping the cache file; null disables caching
 * @param {() => void} [options.checkpoint] - throws to stop (called before each lookup)
 * @param {(done: number, total: number) => void} [options.onProgress] - DOIs looked up so far
 * @returns {Promise<object>} counts of works {missing, noDoi, cacheHits, negativeSkips, crossref, elsevier,
 *   springer, openalex, filled, failed} plus `errors` [{source, message}] for disabled providers
 */
export async function enrichMissingAbstracts(
  http,
  works,
  {
    mailto = "",
    crossref = true,
    openalex = true,
    elsevierKey = "",
    springerKey = "",
    cache = null,
    now = Date.now(),
    retry = {},
    log = () => {},
    checkpoint = () => {},
    onProgress = () => {},
    concurrency = CONCURRENCY,
    publisherConcurrency = PUBLISHER_CONCURRENCY,
  } = {},
) {
  const counts = { missing: 0, noDoi: 0, cacheHits: 0, negativeSkips: 0, crossref: 0, elsevier: 0, springer: 0, openalex: 0, filled: 0, failed: 0 };
  const errors = [];
  const finish = () => {
    counts.filled = counts.cacheHits + counts.crossref + counts.elsevier + counts.springer + counts.openalex;
    counts.failed = counts.missing - counts.filled;
    return { ...counts, errors };
  };

  // Several works (e.g. of different tracks) may share a DOI: look it up once
  const groups = new Map();
  for (const work of new Set(works)) {
    if (!work || hasAbstract(work)) continue;
    const doi = normalizeDoi(work.doi);
    if (!doi) {
      counts.noDoi++;
      continue;
    }
    counts.missing++;
    if (!groups.has(doi)) groups.set(doi, { doi, works: [], outcomes: {}, abstract: null });
    groups.get(doi).works.push(work);
  }
  if (!groups.size) return finish();

  const keys = { elsevier: String(elsevierKey || "").trim(), springer: String(springerKey || "").trim() };
  const enabled = { crossref: !!crossref, elsevier: !!keys.elsevier, springer: !!keys.springer, openalex: !!openalex };
  const store = cache ? await loadCache(cache, now) : null;
  const remember = (doi, entry) => {
    if (!store) return;
    store.entries.set(doi, entry);
    store.dirty = true;
  };

  const pending = [];
  for (const group of groups.values()) {
    group.applicable = applicableTiers(group.doi, enabled);
    group.toQuery = group.applicable.filter((tier) => {
      const source = SAME_RECORD_SOURCE[tier];
      return !source || !group.works.every((w) => w.source === source);
    });
    const entry = store?.entries.get(group.doi);
    if (typeof entry?.a === "string") {
      for (const w of group.works) w.abstract = entry.a;
      counts.cacheHits += group.works.length;
    } else if (entry && group.applicable.every((tier) => entry.s.includes(tier))) {
      // Every tier that could serve this DOI missed it recently
      counts.negativeSkips += group.works.length;
    } else if (group.toQuery.length) {
      pending.push(group);
    }
  }

  const providers = Object.fromEntries(TIERS.map((tier) => [tier, { name: PROVIDER_NAMES[tier], disabled: false, failures: 0 }]));
  const disable = (provider, message) => {
    if (provider.disabled) return;
    provider.disabled = true;
    errors.push({ source: provider.name, message });
    log(`${provider.name} abstract lookups disabled for this run: ${message}`);
  };
  const requestOptions = { ...retry, retries: Math.min(retry.retries ?? 2, 1), timeout: REQUEST_TIMEOUT };

  // One provider request: {data} on success, {miss: true} for a definitive miss, {} when unavailable
  const call = async (tier, request) => {
    const provider = providers[tier];
    if (provider.disabled) return {};
    try {
      const data = await request();
      provider.failures = 0;
      return { data };
    } catch (e) {
      if (e?.cancelled) throw e;
      const status = e?.status;
      if (status === 404 || status === 410) {
        provider.failures = 0;
        return { miss: true };
      }
      if ((status === 401 || status === 403) && (tier === "elsevier" || tier === "springer")) {
        disable(provider, `API Key 无效或无权限（HTTP ${status}），本次运行已停用`);
        return {};
      }
      log(`${provider.name} abstract lookup failed: ${e?.message || e}`);
      if (++provider.failures >= MAX_CONSECUTIVE_FAILURES) {
        disable(provider, `连续 ${MAX_CONSECUTIVE_FAILURES} 次请求失败，本次运行已跳过（${shortReason(e)}）`);
      }
      return {};
    }
  };

  const publisherSlot = createLimiter(publisherConcurrency);
  const perDoi = {
    crossref: { request: (doi) => getJson(http, crossrefWorkUrl(doi, mailto), requestOptions), parse: parseCrossrefAbstract },
    elsevier: {
      request: (doi) => {
        const { url, headers } = elsevierRequest(doi, keys.elsevier);
        return getJson(http, url, { ...requestOptions, headers });
      },
      parse: parseElsevierAbstract,
      publisher: true,
    },
    springer: { request: (doi) => getJson(http, springerUrl(doi, keys.springer), requestOptions), parse: parseSpringerAbstract, publisher: true },
  };

  const found = (group, tier, abstract) => {
    group.abstract = abstract;
    for (const w of group.works) w.abstract = abstract;
    counts[tier] += group.works.length;
    remember(group.doi, { a: abstract, s: tier, exp: now + POSITIVE_TTL });
  };

  let done = 0;
  const step = (n) => {
    done += n;
    onProgress(done, pending.length);
  };

  try {
    // Per-DOI chain (Crossref, then the publisher APIs); OpenAlex follows in batches
    await runPool(pending, concurrency, async (group) => {
      checkpoint();
      for (const tier of group.toQuery) {
        if (tier === "openalex") continue;
        const spec = perDoi[tier];
        const attempt = () => call(tier, () => spec.request(group.doi));
        const res = spec.publisher ? await publisherSlot(attempt) : await attempt();
        const abstract = "data" in res ? spec.parse(res.data) : null;
        group.outcomes[tier] = abstract ? "hit" : "data" in res || res.miss ? "miss" : "unavailable";
        if (abstract) {
          found(group, tier, abstract);
          break;
        }
      }
      if (group.abstract || !group.toQuery.includes("openalex")) step(1);
    });

    const rest = pending.filter((group) => !group.abstract && group.toQuery.includes("openalex"));
    const batches = [];
    for (let i = 0; i < rest.length; i += OPENALEX_BATCH_SIZE) batches.push(rest.slice(i, i + OPENALEX_BATCH_SIZE));
    await runPool(batches, OPENALEX_CONCURRENCY, async (batch) => {
      checkpoint();
      const res = await call("openalex", () => getJson(http, openAlexAbstractsUrl(batch.map((g) => g.doi), mailto), requestOptions));
      const byDoi = "data" in res ? parseOpenAlexAbstracts(res.data) : null;
      for (const group of batch) {
        const abstract = byDoi ? acceptAbstract(byDoi.get(group.doi)) : null;
        group.outcomes.openalex = abstract ? "hit" : byDoi || res.miss ? "miss" : "unavailable";
        if (abstract) found(group, "openalex", abstract);
      }
      step(batch.length);
    });

    // Negative entries only for DOIs that every queried tier definitively missed
    let negatives = 0;
    for (const group of pending) {
      if (group.abstract || !group.toQuery.every((tier) => group.outcomes[tier] === "miss")) continue;
      remember(group.doi, { a: null, s: group.applicable, exp: now + NEGATIVE_TTL });
      negatives++;
    }
    const result = finish();
    log(
      `Abstracts: filled ${result.filled}/${result.missing} (cache ${result.cacheHits}, crossref ${result.crossref}, ` +
        `elsevier ${result.elsevier}, springer ${result.springer}, openalex ${result.openalex}; ` +
        `${result.negativeSkips} known misses skipped, ${negatives} new misses cached)`,
    );
    return result;
  } finally {
    // Also after a cancellation, so abstracts found so far are kept
    if (store?.dirty) await saveCache(cache, store, now, log);
  }
}

/**
 * Connection-test lines for the publisher keys that are set. Any 2xx or 404
 * means the key was accepted (the test DOI need not exist).
 */
export async function checkPublisherKeys(http, { elsevierKey = "", springerKey = "" } = {}) {
  const checks = [];
  if (elsevierKey) checks.push([PROVIDER_NAMES.elsevier, elsevierRequest(TEST_DOIS.elsevier, elsevierKey)]);
  if (springerKey) checks.push([PROVIDER_NAMES.springer, { url: springerUrl(TEST_DOIS.springer, springerKey) }]);
  const lines = [];
  for (const [name, req] of checks) {
    try {
      const res = await http.request({ ...req, timeout: REQUEST_TIMEOUT });
      if ((res.status >= 200 && res.status < 300) || res.status === 404) lines.push(`✅ ${name}：Key 有效`);
      else if (res.status === 401 || res.status === 403) lines.push(`❌ ${name}：Key 无效（HTTP ${res.status}）`);
      else lines.push(`❌ ${name}：HTTP ${res.status}`);
    } catch (e) {
      lines.push(`❌ ${name}：${e?.message || e}`);
    }
  }
  return lines;
}
