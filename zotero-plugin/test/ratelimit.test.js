// Per-host rate limiter (core/ratelimit.js) and its use in core/http.js:
// token pacing on a fake clock, the concurrency cap, FIFO order, cancellation
// while queued, per-host isolation, server-requested pauses, retries through
// the limiter, Semantic Scholar's retry policy and OpenAlex usage headers.

import assert from "node:assert/strict";
import test from "node:test";

import { apiUsage, createCancelSignal, getJson, limitedRequest, openAlexUsageFromHeaders, request, resetApiUsage, responseHeader } from "../src/core/http.js";
import {
  DEFAULT_RATE_LIMITS,
  PUBLIC_CROSSREF_LIMIT,
  createRateLimiter,
  hostOf,
  mergeRateLimits,
  resetSharedRateLimiter,
  sharedRateLimiter,
} from "../src/core/ratelimit.js";

/** Manual clock: timers run only when advance() passes their due time. */
function fakeClock() {
  let now = 0;
  let nextId = 1;
  const timers = new Map();
  const flush = async () => {
    for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
  };
  return {
    now: () => now,
    setTimer(fn, ms) {
      const id = nextId++;
      timers.set(id, { fn, at: now + ms });
      return id;
    },
    clearTimer: (id) => timers.delete(id),
    pending: () => timers.size,
    flush,
    async advance(ms) {
      const until = now + ms;
      await flush();
      for (;;) {
        const due = [...timers.entries()].filter(([, t]) => t.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        timers.delete(due[0]);
        now = Math.max(now, due[1].at);
        due[1].fn();
        await flush();
      }
      now = until;
      await flush();
    },
  };
}

const CROSSREF = "https://api.crossref.org/works?rows=0";
const OPENALEX = "https://api.openalex.org/works?filter=doi:x";
const S2 = "https://api.semanticscholar.org/graph/v1/paper/batch";

/** Acquire `count` slots of `url`, recording when each was granted. */
function acquireMany(limiter, clock, url, count, { release = true, signal = null } = {}) {
  const granted = [];
  const releases = [];
  const promises = Array.from({ length: count }, (_, i) =>
    limiter.acquire(url, signal).then((done) => {
      granted.push([i, clock.now()]);
      releases.push(done);
      if (release) done();
      return done;
    }),
  );
  return { granted, releases, promises };
}

test("host names and limit overrides", () => {
  assert.equal(hostOf("https://API.OpenAlex.org/works?x=1"), "api.openalex.org");
  assert.equal(hostOf("https://user@api.crossref.org:443/works"), "api.crossref.org");
  assert.equal(hostOf("not a url"), "");
  // The documented table (README §工作原理)
  assert.deepEqual(DEFAULT_RATE_LIMITS, {
    "api.crossref.org": { rps: 3, concurrency: 3 },
    "api.openalex.org": { rps: 8, concurrency: 4 },
    "api.semanticscholar.org": { rps: 1, concurrency: 1 },
    "export.arxiv.org": { rps: 1 / 3, concurrency: 1 },
    "api.elsevier.com": { rps: 2, concurrency: 2 },
    "api.springernature.com": { rps: 2, concurrency: 2 },
  });
  assert.deepEqual(PUBLIC_CROSSREF_LIMIT, { rps: 3, concurrency: 1 });

  const merged = mergeRateLimits(DEFAULT_RATE_LIMITS, { "API.Crossref.org": { rps: 10, concurrency: 5 }, "export.arxiv.org": null, "bad.example.org": "fast", "x.org": { rps: -1 } });
  assert.deepEqual(merged["api.crossref.org"], { rps: 10, concurrency: 5 });
  assert.equal("export.arxiv.org" in merged, false, "null removes a limit");
  assert.equal("bad.example.org" in merged, false);
  assert.equal("x.org" in merged, false);
  assert.deepEqual(mergeRateLimits(DEFAULT_RATE_LIMITS, "garbage"), { ...DEFAULT_RATE_LIMITS });

  // A partial override keeps the other field: slowing arXiv down keeps its single connection
  const slower = mergeRateLimits(DEFAULT_RATE_LIMITS, { "export.arxiv.org": { rps: 0.2 }, "api.crossref.org": { concurrency: 1 }, "api.openalex.org": { rps: null } });
  assert.deepEqual(createRateLimiter({ limits: slower }).limitOf("https://export.arxiv.org/api/query"), { rps: 0.2, concurrency: 1, burst: 1 });
  assert.deepEqual(slower["api.crossref.org"], { rps: 3, concurrency: 1 }, "still paced");
  assert.deepEqual(createRateLimiter({ limits: slower }).limitOf(OPENALEX), { rps: Infinity, concurrency: 4, burst: 1 }, "an explicit null field drops only that field");
  assert.deepEqual(mergeRateLimits(DEFAULT_RATE_LIMITS, { "new.example.org": { rps: 2 } })["new.example.org"], { rps: 2 });
});

test("requests to a host are spaced by its rate", async () => {
  const clock = fakeClock();
  const limiter = createRateLimiter({ clock });
  const { granted, promises } = acquireMany(limiter, clock, CROSSREF, 4);
  await clock.advance(0);
  assert.deepEqual(granted, [[0, 0]], "the first request goes at once");
  await clock.advance(5000);
  await Promise.all(promises);
  assert.deepEqual(granted.map(([i]) => i), [0, 1, 2, 3], "first in, first out");
  const gaps = granted.slice(1).map(([, at], i) => at - granted[i][1]);
  for (const gap of gaps) assert.ok(gap >= 333 && gap <= 334, `3 per second: gap ${gap}`);

  // arXiv: one request every three seconds
  const arxiv = acquireMany(limiter, clock, "https://export.arxiv.org/api/query?x", 3);
  await clock.advance(10000);
  await Promise.all(arxiv.promises);
  const at = arxiv.granted.map(([, t]) => t - arxiv.granted[0][1]);
  assert.deepEqual(at, [0, 3000, 6000]);

  // An idle host has its token back
  await clock.advance(60000);
  const later = acquireMany(limiter, clock, CROSSREF, 1);
  await clock.advance(0);
  assert.equal(later.granted.length, 1);
  assert.equal(clock.pending(), 0, "no timer left behind");
});

test("at most `concurrency` requests of a host are in flight", async () => {
  const clock = fakeClock();
  const limiter = createRateLimiter({ clock });
  const { granted, releases } = acquireMany(limiter, clock, OPENALEX, 6, { release: false });
  await clock.advance(10000);
  assert.equal(granted.length, 4, "OpenAlex: four in flight");
  assert.deepEqual(limiter.stats()["api.openalex.org"], { active: 4, waiting: 2 });
  releases[0]();
  releases[0](); // releasing twice frees one slot only
  await clock.advance(1000);
  assert.equal(granted.length, 5);
  releases[1]();
  await clock.advance(1000);
  assert.equal(granted.length, 6);
  assert.deepEqual(limiter.stats()["api.openalex.org"], { active: 4, waiting: 0 });

  // Semantic Scholar: one at a time, one per second
  const s2 = acquireMany(limiter, clock, S2, 2, { release: false });
  await clock.advance(5000);
  assert.equal(s2.granted.length, 1);
  s2.releases[0]();
  await clock.advance(0);
  assert.equal(s2.granted.length, 2, "the token refilled while the first request was in flight");

  // A request arriving after the token refilled still waits for the slot of a slow request
  const clock2 = fakeClock();
  const limiter2 = createRateLimiter({ clock: clock2 });
  const slow = await limiter2.acquire(S2);
  await clock2.advance(2000);
  let second = null;
  limiter2.acquire(S2).then((done) => (second = done));
  await clock2.advance(0);
  assert.equal(second, null, "the cap holds at a fresh acquire");
  assert.deepEqual(limiter2.stats()["api.semanticscholar.org"], { active: 1, waiting: 1 });
  slow();
  await clock2.advance(0);
  assert.equal(typeof second, "function", "granted once the first request is done");
  second();
});

test("a queued request is cancelled without blocking the others", async () => {
  const clock = fakeClock();
  const limiter = createRateLimiter({ clock });
  const first = await limiter.acquire(S2);
  const signal = createCancelSignal();
  const waiting = limiter.acquire(S2, signal);
  const other = limiter.acquire(S2);
  let otherGranted = false;
  other.then(() => (otherGranted = true));
  await clock.advance(100);
  assert.deepEqual(limiter.stats()["api.semanticscholar.org"], { active: 1, waiting: 2 });

  signal.cancel();
  await assert.rejects(waiting, (e) => e.cancelled === true && e.name === "CancelledRequestError");
  // The last waiter leaving clears the pacing timer
  const lone = fakeClock();
  const quiet = createRateLimiter({ clock: lone });
  await quiet.acquire(CROSSREF);
  const loneSignal = createCancelSignal();
  const queued = quiet.acquire(CROSSREF, loneSignal);
  assert.equal(lone.pending(), 1);
  loneSignal.cancel();
  await assert.rejects(queued, (e) => e.cancelled === true);
  assert.equal(lone.pending(), 0, "no timer left after the queue emptied");
  assert.deepEqual(limiter.stats()["api.semanticscholar.org"], { active: 1, waiting: 1 });
  first();
  await clock.advance(2000);
  assert.equal(otherGranted, true);
  // A cancelled signal fails at once
  assert.throws(() => limiter.acquire(S2, signal), (e) => e.cancelled === true);
});

test("hosts are limited independently and unknown hosts not at all", async () => {
  const clock = fakeClock();
  const limiter = createRateLimiter({ clock });
  const busy = await limiter.acquire(S2);
  const queued = acquireMany(limiter, clock, S2, 1, { release: false });
  const crossref = acquireMany(limiter, clock, CROSSREF, 1);
  const llm = acquireMany(limiter, clock, "https://openrouter.ai/api/v1/chat/completions", 20);
  await clock.advance(0);
  assert.equal(crossref.granted.length, 1, "Crossref is not held up by Semantic Scholar");
  assert.equal(llm.granted.length, 20, "LLM requests are unlimited");
  assert.equal(queued.granted.length, 0);
  busy();
  await clock.advance(1000);
  assert.equal(queued.granted.length, 1);
});

test("a server-requested pause holds every request of that host", async () => {
  const clock = fakeClock();
  const limiter = createRateLimiter({ clock });
  limiter.pause(OPENALEX, 5000);
  limiter.pause("https://unlimited.example.org/", 5000); // no effect, no error
  const { granted } = acquireMany(limiter, clock, OPENALEX, 2);
  const other = acquireMany(limiter, clock, CROSSREF, 1);
  await clock.advance(4999);
  assert.equal(granted.length, 0);
  assert.equal(other.granted.length, 1);
  await clock.advance(1000);
  assert.equal(granted.length, 2);
  assert.equal(granted[0][1], 5000);
});

test("limits can be changed while requests wait", async () => {
  const clock = fakeClock();
  const limiter = createRateLimiter({ clock });
  const busy = await limiter.acquire(S2);
  const queued = acquireMany(limiter, clock, S2, 2, { release: false });
  await clock.advance(10);
  assert.equal(queued.granted.length, 0);
  limiter.setLimits(mergeRateLimits(DEFAULT_RATE_LIMITS, { "api.semanticscholar.org": null }));
  await clock.advance(0);
  assert.equal(queued.granted.length, 2, "an unlimited host serves its queue at once");
  busy();
  assert.equal(limiter.limitOf(S2), null);
  assert.deepEqual(limiter.limitOf(CROSSREF), { rps: 3, concurrency: 3, burst: 1 });
});

test("the shared limiter is one instance until reset", () => {
  const a = sharedRateLimiter();
  assert.equal(sharedRateLimiter(), a);
  const clock = fakeClock();
  const b = resetSharedRateLimiter({ clock });
  assert.notEqual(b, a);
  assert.equal(sharedRateLimiter(), b);
  resetSharedRateLimiter(null);
  assert.notEqual(sharedRateLimiter(), b);
});

// ------------------------------------------------------------ http.js

/** Transport answering from `respond(req, n)` with a limiter on a fake clock. */
function limitedTransport(clock, respond, { signal = null } = {}) {
  const calls = [];
  return {
    calls,
    signal,
    limiter: createRateLimiter({ clock }),
    async request(req) {
      calls.push({ url: req.url, at: clock.now() });
      return respond(req, calls.length);
    },
  };
}

test("request() waits for the limiter before every attempt, retries included", async () => {
  const clock = fakeClock();
  const http = limitedTransport(clock, (req, n) => (n === 1 ? { status: 503, data: null } : { status: 200, data: { ok: n } }));
  const sleeps = [];
  const pending = request(http, { url: CROSSREF }, { sleep: async (ms) => sleeps.push(ms) });
  await clock.advance(1000);
  assert.deepEqual(await pending, { ok: 2 });
  assert.deepEqual(http.calls.map((c) => c.at), [0, 334], "the retry waited for the next Crossref token");
  assert.deepEqual(sleeps, [2000]);
  assert.deepEqual(http.limiter.stats()["api.crossref.org"], { active: 0, waiting: 0 }, "slots are released after errors too");

  // Concurrent callers share the host's pace
  const many = Array.from({ length: 4 }, () => getJson(http, CROSSREF, { sleep: async () => {} }));
  await clock.advance(2000);
  await Promise.all(many);
  const times = http.calls.slice(2).map((c) => c.at);
  assert.equal(new Set(times).size, 4, "never two Crossref requests at the same moment");
});

test("a 429 with Retry-After pauses the host; Semantic Scholar retries 5xx with a longer delay", async () => {
  const clock = fakeClock();
  const http = limitedTransport(clock, (req, n) =>
    n === 1 ? { status: 429, data: {}, headers: { "retry-after": "7" } } : { status: 200, data: [] },
  );
  const sleeps = [];
  const first = request(http, { url: OPENALEX }, { sleep: async (ms) => sleeps.push(ms) });
  // Another request to the same host, started meanwhile, also waits for the pause
  await clock.advance(0);
  const second = getJson(http, OPENALEX, { sleep: async () => {} });
  await clock.advance(6000);
  assert.equal(http.calls.length, 1, "the retry and the other request both wait for the host pause");
  await clock.advance(2000);
  await Promise.all([first, second]);
  assert.deepEqual(sleeps, [7000], "Retry-After is honoured");
  assert.ok(http.calls.slice(1).every((c) => c.at >= 7000));

  // Semantic Scholar answers throttling with 500 as well: retried, base delay 3 s
  const clock2 = fakeClock();
  const s2 = limitedTransport(clock2, (req, n) => (n < 3 ? { status: 500, data: { message: "busy" } } : { status: 200, data: { ok: true } }));
  const s2Sleeps = [];
  const done = request(s2, { url: S2, method: "POST" }, { sleep: async (ms) => s2Sleeps.push(ms) });
  await clock2.advance(5000);
  assert.deepEqual(await done, { ok: true });
  assert.deepEqual(s2Sleeps, [3000, 6000]);
  assert.deepEqual(s2.calls.map((c) => c.at), [0, 1000, 2000], "one request per second");

  // ... and a Semantic Scholar 5xx with Retry-After pauses the host for other requests too
  const clock3 = fakeClock();
  const s2b = limitedTransport(clock3, (req, n) =>
    n === 1 ? { status: 503, data: {}, headers: { "retry-after": "5" } } : { status: 200, data: { ok: n } },
  );
  const firstS2 = request(s2b, { url: S2, method: "POST" }, { sleep: async () => {} });
  await clock3.advance(0);
  const otherS2 = getJson(s2b, `${S2}?other`, { sleep: async () => {} });
  await clock3.advance(4500);
  assert.equal(s2b.calls.length, 1, "nothing is sent to Semantic Scholar during the pause");
  await clock3.advance(3000);
  await Promise.all([firstS2, otherS2]);
  assert.ok(s2b.calls.slice(1).every((c) => c.at >= 5000));

  // Other hosts keep the 2 s base delay; a 4xx is not retried
  const plain = limitedTransport(fakeClock(), () => ({ status: 404, data: { message: "nope" } }));
  await assert.rejects(request(plain, { url: CROSSREF }), /HTTP 404/);
  assert.equal(plain.calls.length, 1);
});

test("cancelling a run ends a request waiting for its slot", async () => {
  const clock = fakeClock();
  const signal = createCancelSignal();
  const http = limitedTransport(clock, () => ({ status: 200, data: {} }), { signal });
  const busy = await http.limiter.acquire(S2);
  const pending = getJson(http, S2);
  await clock.advance(100);
  signal.cancel();
  await assert.rejects(pending, (e) => e.cancelled === true);
  assert.equal(http.calls.length, 0);
  busy();
});

test("OpenAlex rate-limit headers are recorded; keys never appear in errors", async () => {
  resetApiUsage();
  const headers = { "X-RateLimit-Limit": "100000", "X-RateLimit-Remaining": "99990", "X-RateLimit-Credits-Used": "10", "X-RateLimit-Remaining-USD": "0.95" };
  const http = {
    async request(req) {
      if (req.url.includes("fail")) return { status: 403, data: { message: `bad key ${req.url}` } };
      return { status: 200, data: { results: [] }, headers: req.url.includes("openalex") ? headers : {} };
    },
  };
  await getJson(http, "https://api.crossref.org/works?x=1");
  assert.equal(apiUsage.openalex, undefined, "only OpenAlex is metered");
  await getJson(http, "https://api.openalex.org/works?filter=x&api_key=secret");
  assert.equal(apiUsage.openalex.remaining, 99990);
  assert.equal(apiUsage.openalex.limit, 100000);
  assert.equal(apiUsage.openalex.creditsUsed, 10);
  assert.equal(apiUsage.openalex.remainingUsd, 0.95);
  assert.equal(apiUsage.openalex.limitUsd, null);
  assert.equal(apiUsage.openalex.keyed, true);
  assert.ok(Number.isFinite(apiUsage.openalex.at));

  // Responses without the headers keep the last known values
  await getJson({ request: async () => ({ status: 200, data: {} }) }, "https://api.openalex.org/works?x");
  assert.equal(apiUsage.openalex.remaining, 99990);
  assert.equal(openAlexUsageFromHeaders({ headers: { get: () => null } }), null);
  assert.equal(openAlexUsageFromHeaders({ headers: { get: () => {
    throw new Error("denied");
  } } }), null, "unreadable headers are ignored");

  await assert.rejects(getJson(http, "https://api.openalex.org/works?fail=1&api_key=secret"), (e) => {
    assert.doesNotMatch(e.message, /secret/);
    return /HTTP 403/.test(e.message);
  });

  // Header lookup: get(), plain objects in any case, and the legacy retryAfter field
  assert.equal(responseHeader({ headers: { get: (n) => (n === "Retry-After" ? "3" : null) } }, "Retry-After"), "3");
  assert.equal(responseHeader({ headers: { "retry-after": "4" } }, "Retry-After"), "4");
  assert.equal(responseHeader({ retryAfter: "5" }, "Retry-After"), "5");
  assert.equal(responseHeader({}, "X-Missing"), null);
  resetApiUsage();
});

test("limitedRequest returns the raw response through the limiter", async () => {
  const clock = fakeClock();
  const http = limitedTransport(clock, () => ({ status: 401, data: null }));
  const a = limitedRequest(http, { url: S2 });
  const b = limitedRequest(http, { url: S2 });
  await clock.advance(1500);
  assert.deepEqual((await a).status, 401);
  assert.deepEqual((await b).status, 401);
  assert.deepEqual(http.calls.map((c) => c.at), [0, 1000]);
});
