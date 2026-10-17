// Per-host rate limiting for every HTTP request of the plugin (see request() in
// core/http.js). Each limited host has a token bucket refilled continuously at
// `rps` tokens per second (holding at most `burst` tokens, 1 by default, so
// requests are evenly spaced) and at most `concurrency` requests in flight.
// Waiters are served first in, first out; a waiter whose cancel signal fires
// leaves the queue at once. Hosts without an entry are not limited.
//
// One limiter is shared by all transports of an addon instance (the watch run,
// UI actions and later surveys), so that concurrent work never exceeds a
// service's limits together. Tests create their own limiter with a fake clock.

/**
 * Limits per host: requests per second and requests in flight. Hosts not
 * listed (LLM and embedding services, EarthArXiv) are unlimited.
 */
export const DEFAULT_RATE_LIMITS = Object.freeze({
  "api.crossref.org": { rps: 3, concurrency: 3 },
  "api.openalex.org": { rps: 8, concurrency: 4 },
  // Introductory keys allow one request per second over all endpoints together
  "api.semanticscholar.org": { rps: 1, concurrency: 1 },
  // arXiv asks for one request every three seconds
  "export.arxiv.org": { rps: 1 / 3, concurrency: 1 },
  "api.elsevier.com": { rps: 2, concurrency: 2 },
  "api.springernature.com": { rps: 2, concurrency: 2 },
});

/** Lower-case host name of a URL ("" when it has none). */
export function hostOf(url) {
  const m = String(url ?? "").match(/^[a-z][a-z0-9+.-]*:\/\/(?:[^@/?#]*@)?([^/?#:]+)/i);
  return m ? m[1].toLowerCase() : "";
}

/** A valid {rps, concurrency, burst} limit, or null for "unlimited". */
function validLimit(limit) {
  if (!limit || typeof limit !== "object") return null;
  const rps = Number(limit.rps);
  const concurrency = Math.floor(Number(limit.concurrency));
  const burst = limit.burst === undefined ? 1 : Number(limit.burst);
  const rateOk = Number.isFinite(rps) && rps > 0;
  const slotsOk = Number.isFinite(concurrency) && concurrency >= 1;
  if (!rateOk && !slotsOk) return null;
  return {
    rps: rateOk ? rps : Infinity,
    concurrency: slotsOk ? concurrency : Infinity,
    burst: Number.isFinite(burst) && burst >= 1 ? burst : 1,
  };
}

/**
 * Merge limit overrides (e.g. from a hidden pref holding JSON) into the
 * defaults: `{host: {rps, concurrency} | null}`, null removing a host's limit.
 * Anything malformed is ignored.
 */
export function mergeRateLimits(base = DEFAULT_RATE_LIMITS, overrides = null) {
  const out = { ...base };
  if (!overrides || typeof overrides !== "object" || Array.isArray(overrides)) return out;
  for (const [host, limit] of Object.entries(overrides)) {
    const key = String(host).trim().toLowerCase();
    if (!key) continue;
    if (limit === null) delete out[key];
    else if (validLimit(limit)) out[key] = limit;
  }
  return out;
}

/** The signal's own cancellation error (CancelledRequestError for createCancelSignal signals). */
function cancelError(signal) {
  try {
    signal.throwIfCancelled?.();
  } catch (e) {
    return e;
  }
  const error = new Error("已取消");
  error.cancelled = true;
  return error;
}

const realClock = {
  now: () => Date.now(),
  // Looked up at call time: the plugin sandbox provides its timers globally
  setTimer: (fn, ms) => setTimeout(fn, ms),
  clearTimer: (id) => clearTimeout(id),
};

/**
 * Create a limiter.
 * @param {object} [options]
 * @param {object} [options.limits] - {host: {rps, concurrency, burst?}}
 * @param {{now: () => number, setTimer: Function, clearTimer: Function}} [options.clock] - injectable for tests
 */
export function createRateLimiter({ limits = DEFAULT_RATE_LIMITS, clock = realClock } = {}) {
  let table = { ...limits };
  const queues = new Map();

  function queueOf(host) {
    let q = queues.get(host);
    if (q) return q;
    const limit = validLimit(table[host]);
    if (!limit) return null;
    q = { host, ...limit, tokens: limit.burst, last: clock.now(), notBefore: 0, active: 0, waiters: [], timer: null };
    queues.set(host, q);
    return q;
  }

  function refill(q, now) {
    if (q.rps === Infinity) {
      q.tokens = q.burst;
    } else if (now > q.last) {
      q.tokens = Math.min(q.burst, q.tokens + ((now - q.last) * q.rps) / 1000);
    }
    q.last = Math.max(q.last, now);
  }

  function makeRelease(q) {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      q.active--;
      pump(q);
    };
  }

  /** Grant slots to waiters while a slot and a token are available; arm a timer for the next token. */
  function pump(q) {
    const now = clock.now();
    refill(q, now);
    const unpaced = q.rps === Infinity;
    while (q.waiters.length && q.active < q.concurrency && now >= q.notBefore && (unpaced || q.tokens >= 1)) {
      const waiter = q.waiters.shift();
      if (!unpaced) q.tokens -= 1;
      q.active++;
      waiter.grant(makeRelease(q));
    }
    if (!q.waiters.length && q.timer !== null) {
      // Nobody waits any more (cancelled): leave no timer behind
      clock.clearTimer(q.timer);
      q.timer = null;
    }
    if (q.timer !== null || !q.waiters.length || q.active >= q.concurrency) return;
    // Waiting for time to pass (a token, or the end of a server-requested pause)
    const tokenWait = unpaced || q.tokens >= 1 ? 0 : ((1 - q.tokens) * 1000) / q.rps;
    const wait = Math.max(tokenWait, q.notBefore - now, 1);
    q.timer = clock.setTimer(() => {
      q.timer = null;
      pump(q);
    }, Math.ceil(wait));
  }

  return {
    /**
     * Wait for a request slot of `url`'s host. Resolves with a release
     * function (call it once the response arrived or the request failed);
     * rejects with CancelledRequestError when `signal` is cancelled first.
     */
    acquire(url, signal = null) {
      signal?.throwIfCancelled?.();
      const q = queueOf(hostOf(url));
      if (!q) return Promise.resolve(() => {});
      return new Promise((resolve, reject) => {
        let unregister = () => {};
        const waiter = {
          grant(release) {
            unregister();
            resolve(release);
          },
        };
        q.waiters.push(waiter);
        if (signal?.onCancel) {
          unregister = signal.onCancel(() => {
            const idx = q.waiters.indexOf(waiter);
            if (idx < 0) return;
            q.waiters.splice(idx, 1);
            reject(cancelError(signal));
            // The next waiter may now be due
            pump(q);
          });
        }
        pump(q);
      });
    },

    /**
     * Hold every request to `url`'s host back for `ms` (a 429 answer with
     * Retry-After): concurrent requests wait instead of failing as well.
     */
    pause(url, ms) {
      const q = queueOf(hostOf(url));
      if (!q || !(ms > 0)) return;
      q.notBefore = Math.max(q.notBefore, clock.now() + ms);
    },

    /** Replace the limits (hosts with queued requests keep their queue and get the new values). */
    setLimits(next) {
      table = { ...next };
      for (const [host, q] of queues) {
        const limit = validLimit(table[host]);
        if (limit) Object.assign(q, { rps: limit.rps, concurrency: limit.concurrency, burst: limit.burst, tokens: Math.min(q.tokens, limit.burst) });
        else Object.assign(q, { rps: Infinity, concurrency: Infinity });
        pump(q);
      }
    },

    limitOf(url) {
      return validLimit(table[hostOf(url)]);
    },

    /** Queue state per host, for tests and diagnostics: {host: {active, waiting}}. */
    stats() {
      return Object.fromEntries([...queues].map(([host, q]) => [host, { active: q.active, waiting: q.waiters.length }]));
    },
  };
}

let shared = null;

/** The limiter shared by every transport of this addon instance (created on first use). */
export function sharedRateLimiter() {
  return (shared ||= createRateLimiter());
}

/** Replace the shared limiter (tests; options as for createRateLimiter, null to drop it). */
export function resetSharedRateLimiter(options = {}) {
  shared = options === null ? null : createRateLimiter(options);
  return shared;
}
