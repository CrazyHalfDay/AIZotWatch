// HTTP helpers on top of an injectable transport:
//   transport.request({ method, url, headers, body, responseType, timeout }) -> { status, data }
// In Zotero the transport wraps Zotero.HTTP.request; in tests it is a mock.

export class HttpError extends Error {
  constructor(status, url, detail = "") {
    super(`HTTP ${status} for ${redactUrl(url)}${detail ? `: ${detail}` : ""}`);
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
 * Perform a request, retrying network errors, 429 and 5xx with exponential backoff
 * (or the server's Retry-After, capped at one minute).
 * Resolves with `data` for 2xx responses, throws HttpError otherwise.
 */
export async function request(transport, options, { retries = 2, baseDelay = 2000, sleep = defaultSleep } = {}) {
  let lastError;
  let serverDelay = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) {
      const backoff = baseDelay * 2 ** (attempt - 1);
      await sleep(serverDelay !== null ? Math.min(Math.max(serverDelay, backoff), MAX_RETRY_AFTER) : backoff);
    }
    let res;
    try {
      res = await transport.request({ method: "GET", responseType: "json", timeout: 60000, ...options });
    } catch (e) {
      if (e?.cancelled) throw e;
      lastError = e;
      serverDelay = null;
      continue;
    }
    if (res.status >= 200 && res.status < 300) return res.data;
    lastError = new HttpError(res.status, options.url, errorDetail(res.data));
    if (res.status !== 429 && res.status < 500) throw lastError;
    serverDelay = retryAfterMs(res.retryAfter);
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

/** Run async tasks with bounded concurrency, preserving result order. */
export async function mapWithConcurrency(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (next < items.length) {
      const idx = next++;
      results[idx] = await fn(items[idx], idx);
    }
  });
  await Promise.all(workers);
  return results;
}
