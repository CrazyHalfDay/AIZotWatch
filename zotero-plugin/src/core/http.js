// HTTP helpers on top of an injectable transport:
//   transport.request({ method, url, headers, body, responseType, timeout })
//     -> { status, data, retryAfter?, headers?: { get(name) } }
// In Zotero the transport wraps Zotero.HTTP.request; in tests it is a mock.
// A transport may carry `signal` (cancellation, see createCancelSignal) and
// `limiter` (per-host rate limiting, see core/ratelimit.js): request() waits
// for a slot of the URL's host before every attempt, retries included.

import { hostOf } from "./ratelimit.js";

export class HttpError extends Error {
  constructor(status, url, detail = "") {
    // The detail is server text, which may echo the query string (and its key)
    super(`HTTP ${status} for ${redactUrl(url)}${detail ? `: ${redactUrl(detail)}` : ""}`);
    this.name = "HttpError";
    this.status = status;
  }
}

/** Remove credentials from URLs before they end up in logs or error messages. */
export function redactUrl(url) {
  return String(url).replace(/([?&](api_key|apikey|key|token)=)[^&]+/gi, "$1***");
}

export function buildUrl(base, params = {}) {
  const query = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  if (!query) return base;
  return `${base}${base.includes("?") ? "&" : "?"}${query}`;
}

export const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function errorDetail(data) {
  if (!data) return "";
  if (typeof data === "string") return data.slice(0, 200);
  const msg = data.error?.message || data.message || data.error || data.detail;
  return typeof msg === "string" ? msg.slice(0, 200) : JSON.stringify(data).slice(0, 200);
}

/** Error thrown by transports when a request was cancelled; never retried. */
export class CancelledRequestError extends Error {
  constructor(message = "已取消") {
    super(message);
    this.name = "CancelledRequestError";
    this.cancelled = true;
  }
}

/**
 * Cancellation signal shared by every request of one run (AbortController is
 * not available in Zotero's plugin sandbox). Once cancelled, in-flight requests
 * are aborted through their registered cancellers, retry waits end early and
 * new requests fail immediately.
 */
export function createCancelSignal() {
  const listeners = new Set();
  const signal = {
    cancelled: false,
    cancel() {
      if (signal.cancelled) return;
      signal.cancelled = true;
      for (const fn of [...listeners]) {
        try {
          fn();
        } catch {
          // A failing canceller must not stop the others
        }
      }
      listeners.clear();
    },
    /** Register `fn` to run on cancel; returns an unregister function. */
    onCancel(fn) {
      if (signal.cancelled) {
        fn();
        return () => {};
      }
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    throwIfCancelled() {
      if (signal.cancelled) throw new CancelledRequestError();
    },
  };
  return signal;
}

/** Sleep that ends early with CancelledRequestError when `signal` is cancelled. */
async function cancellableSleep(sleep, ms, signal) {
  if (!signal) return sleep(ms);
  signal.throwIfCancelled();
  let unregister = () => {};
  const cancelled = new Promise((_, reject) => {
    unregister = signal.onCancel(() => reject(new CancelledRequestError()));
  });
  try {
    await Promise.race([sleep(ms), cancelled]);
  } finally {
    unregister();
  }
}

/** Parse a Retry-After header (seconds or HTTP date) into milliseconds. */
export function retryAfterMs(value, now = Date.now()) {
  if (value === undefined || value === null || value === "") return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  return Number.isFinite(date) ? Math.max(0, date - now) : null;
}

const MAX_RETRY_AFTER = 60000;

/**
 * Value of a response header, whatever the transport returned: a `headers`
 * object with get(name) (Zotero), a plain object (tests, any case), or the
 * legacy `retryAfter` field. null when absent.
 */
export function responseHeader(res, name) {
  const headers = res?.headers;
  let value = null;
  try {
    if (headers && typeof headers.get === "function") value = headers.get(name);
    else if (headers && typeof headers === "object") {
      const key = Object.keys(headers).find((k) => k.toLowerCase() === name.toLowerCase());
      value = key === undefined ? null : headers[key];
    }
  } catch {
    value = null;
  }
  if ((value === null || value === undefined) && /^retry-after$/i.test(name)) value = res?.retryAfter ?? null;
  return value === undefined || value === "" ? null : value;
}

// ------------------------------------------------------------- API usage

/**
 * Latest known usage of metered services, shared by every transport of the
 * addon instance (in memory only): `apiUsage.openalex = {remaining, limit,
 * remainingUsd, limitUsd, creditsUsed, keyed, at}` from OpenAlex's rate-limit
 * headers. Best effort: responses without these headers change nothing.
 */
export const apiUsage = {};

export function resetApiUsage() {
  for (const key of Object.keys(apiUsage)) delete apiUsage[key];
}

const finiteOrNull = (value) => {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

/** OpenAlex usage from a response's headers; null when it carries none. */
export function openAlexUsageFromHeaders(res, url = "", now = Date.now()) {
  const get = (name) => finiteOrNull(responseHeader(res, name));
  const usage = {
    remaining: get("X-RateLimit-Remaining"),
    limit: get("X-RateLimit-Limit"),
    remainingUsd: get("X-RateLimit-Remaining-USD"),
    limitUsd: get("X-RateLimit-Limit-USD"),
    creditsUsed: get("X-RateLimit-Credits-Used"),
  };
  if (Object.values(usage).every((v) => v === null)) return null;
  return { ...usage, keyed: /[?&]api_key=[^&]/.test(String(url)), at: now };
}

/** Record what a response tells about a metered service's remaining budget. */
export function recordApiUsage(url, res, now = Date.now()) {
  if (hostOf(url) !== "api.openalex.org") return;
  const usage = openAlexUsageFromHeaders(res, url, now);
  if (usage) apiUsage.openalex = usage;
}

// ---------------------------------------------------------------- request

/** Retry policy per host: Semantic Scholar throttles with 429 or 5xx and gets a longer base delay. */
function hostPolicy(url) {
  if (hostOf(url) === "api.semanticscholar.org") return { minBaseDelay: 3000, throttles: (status) => status === 429 || status >= 500 };
  return { minBaseDelay: 0, throttles: (status) => status === 429 };
}

/**
 * One transport request through the transport's limiter (when it has one),
 * recording API usage headers. Resolves with the raw response {status, data, …};
 * for callers that check a status themselves (connection tests).
 */
export async function limitedRequest(transport, req) {
  const { signal, limiter } = transport;
  signal?.throwIfCancelled();
  const release = limiter ? await limiter.acquire(req.url, signal) : null;
  try {
    const res = await transport.request({ method: "GET", responseType: "json", timeout: 60000, ...req });
    recordApiUsage(req.url, res);
    return res;
  } finally {
    release?.();
  }
}

/**
 * Perform a request, retrying network errors, 429 and 5xx with exponential backoff
 * (or the server's Retry-After, capped at one minute). Every attempt waits for a
 * slot of the URL's host when the transport has a limiter; a throttling answer
 * with Retry-After pauses that host for every request sharing the limiter.
 * Resolves with `data` for 2xx responses, throws HttpError otherwise.
 */
export async function request(transport, options, { retries = 2, baseDelay = 2000, sleep = defaultSleep } = {}) {
  const { signal, limiter } = transport;
  const policy = hostPolicy(options.url);
  const base = Math.max(baseDelay, policy.minBaseDelay);
  let lastError;
  let serverDelay = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) {
      const backoff = base * 2 ** (attempt - 1);
      await cancellableSleep(sleep, serverDelay !== null ? Math.min(Math.max(serverDelay, backoff), MAX_RETRY_AFTER) : backoff, signal);
    }
    let res;
    try {
      res = await limitedRequest(transport, options);
    } catch (e) {
      if (e?.cancelled) throw e;
      lastError = e;
      serverDelay = null;
      continue;
    }
    if (res.status >= 200 && res.status < 300) return res.data;
    lastError = new HttpError(res.status, options.url, errorDetail(res.data));
    if (res.status !== 429 && res.status < 500) throw lastError;
    serverDelay = retryAfterMs(responseHeader(res, "Retry-After"));
    if (serverDelay !== null && policy.throttles(res.status)) limiter?.pause(options.url, Math.min(serverDelay, MAX_RETRY_AFTER));
  }
  throw lastError;
}

export function getJson(transport, url, { headers, timeout, ...retry } = {}) {
  return request(transport, { method: "GET", url, headers, timeout, responseType: "json" }, retry);
}

export function getText(transport, url, { headers, timeout, ...retry } = {}) {
  return request(transport, { method: "GET", url, headers, timeout, responseType: "text" }, retry);
}

export function postJson(transport, url, body, { headers = {}, timeout, ...retry } = {}) {
  return request(
    transport,
    {
      method: "POST",
      url,
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
      timeout,
      responseType: "json",
    },
    retry,
  );
}

/**
 * Run async tasks with bounded concurrency, preserving result order. After the
 * first failure no new items are started (running ones finish on their own).
 */
export async function mapWithConcurrency(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  let failed = false;
  const workers = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (!failed && next < items.length) {
      const idx = next++;
      try {
        results[idx] = await fn(items[idx], idx);
      } catch (e) {
        failed = true;
        throw e;
      }
    }
  });
  await Promise.all(workers);
  return results;
}
