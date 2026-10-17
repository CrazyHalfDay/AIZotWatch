import assert from "node:assert/strict";
import test from "node:test";

import { CancelledRequestError, createCancelSignal, getJson, mapWithConcurrency, retryAfterMs } from "../src/core/http.js";
import { fetchArxiv, parseArxivFeed } from "../src/sources/arxiv.js";
import { fetchCrossref, parseCrossrefItem, resolveJournalIssns } from "../src/sources/crossref.js";
import { fetchEartharxiv, parseOaiPage } from "../src/sources/eartharxiv.js";
import { apiUsage, resetApiUsage } from "../src/core/http.js";
import {
  checkOpenAlexRow,
  openAlexAbstractsUrl,
  openAlexAuth,
  openAlexUsageText,
  parseOpenAlexAbstracts,
  parseOpenAlexRateLimit,
  reconstructAbstract,
} from "../src/sources/openalex.js";
import { pastDaysWindow } from "../src/sources/window.js";
import { fixture, mockHttp, noSleep } from "./helpers.js";

const NOW = Date.UTC(2024, 9, 10, 15, 30);

test("date window covers complete past days", () => {
  const { since, until } = pastDaysWindow(7, NOW);
  assert.equal(new Date(until).toISOString(), "2024-10-09T00:00:00.000Z");
  assert.equal(new Date(since).toISOString(), "2024-10-03T00:00:00.000Z");
});

function crossrefItem(n, extra = {}) {
  return {
    DOI: `10.1016/J.TEST.${n}`,
    title: [`Paper <i>${n}</i>`],
    author: [{ given: "Ann", family: `Author${n}` }, { name: "Consortium X" }],
    abstract: `<jats:p>Abstract ${n}</jats:p>`,
    "container-title": ["Test Journal"],
    created: { "date-time": "2024-10-08T10:00:00Z" },
    URL: `https://doi.org/10.1016/j.test.${n}`,
    type: "journal-article",
    "is-referenced-by-count": 3,
    ISSN: ["0016-7037"],
    ...extra,
  };
}

test("crossref item parsing", () => {
  const w = parseCrossrefItem(crossrefItem(1));
  assert.equal(w.id, "doi:10.1016/j.test.1");
  assert.equal(w.title, "Paper 1");
  assert.equal(w.abstract, "Abstract 1");
  assert.deepEqual(w.authors.map((a) => a.name), ["Ann Author1", "Consortium X"]);
  assert.deepEqual(w.issns, ["0016-7037"]);
  assert.equal(parseCrossrefItem(crossrefItem(2, { title: ["Table of Contents"] })), null);
  assert.equal(parseCrossrefItem(crossrefItem(3, { type: "peer-review" })), null);
  assert.equal(parseCrossrefItem(crossrefItem(4, { title: [] })), null);
});

test("crossref pagination, filter and max results", async () => {
  const pages = [
    { "total-results": 3, items: [crossrefItem(1), crossrefItem(2)] },
    { "total-results": 3, items: [crossrefItem(3)] },
  ];
  let page = 0;
  const http = mockHttp([["https://api.crossref.org/works", () => ({ message: pages[page++] })]]);
  const works = await fetchCrossref(http, { issns: ["00167037", "bad"], daysBack: 7, maxResults: 10, mailto: "me@x.org", now: NOW });
  assert.equal(works.length, 3);
  const url = decodeURIComponent(http.calls[0].url);
  assert.match(url, /from-created-date:2024-10-03,until-created-date:2024-10-09,issn:0016-7037/);
  assert.match(url, /mailto=me@x.org/);
  assert.match(decodeURIComponent(http.calls[1].url), /offset=2/);

  const none = await fetchCrossref(mockHttp([]), { issns: [], now: NOW });
  assert.deepEqual(none, []);
});

test("crossref retries server errors and gives up on 4xx", async () => {
  let n = 0;
  const flaky = mockHttp([["https://api.crossref.org/works", () => (n++ === 0 ? { status: 503, data: null } : { message: { "total-results": 1, items: [crossrefItem(1)] } })]]);
  const works = await fetchCrossref(flaky, { issns: ["0016-7037"], now: NOW, retry: noSleep });
  assert.equal(works.length, 1);
  assert.equal(flaky.calls.length, 2);

  const bad = mockHttp([["https://api.crossref.org/works", () => ({ status: 400, data: { message: "bad filter" } })]]);
  await assert.rejects(fetchCrossref(bad, { issns: ["0016-7037"], now: NOW, retry: noSleep }), /HTTP 400.*bad filter/);
  assert.equal(bad.calls.length, 1);
});

test("crossref journal title resolution requires an exact title match", async () => {
  const http = mockHttp([
    ["https://api.crossref.org/journals", () => ({ message: { items: [{ title: "Lithos Extra", ISSN: ["1111-1111"] }, { title: "Lithos", ISSN: ["0024-4937", "18726143"] }] } })],
  ]);
  assert.deepEqual(await resolveJournalIssns(http, "LITHOS"), ["0024-4937", "1872-6143"]);
});

test("arxiv feed parsing keeps primary-category matches only", () => {
  const works = parseArxivFeed(fixture("arxiv.xml"), ["physics.geo-ph"]);
  assert.equal(works.length, 1);
  const [w] = works;
  assert.equal(w.id, "arxiv:2410.01234");
  assert.equal(w.title, "Mantle plumes and subduction zone dynamics");
  assert.equal(w.abstract, "We model the interaction of mantle plumes with subducting slabs & arc magmatism.");
  assert.equal(w.doi, "10.1000/example.42");
  assert.equal(w.url, "http://arxiv.org/abs/2410.01234v2");
  assert.deepEqual(w.authors.map((a) => a.family), ["Doe", "Zhang"]);
});

test("arxiv fetch builds the date-bounded query", async () => {
  const http = mockHttp([["https://export.arxiv.org/api/query", () => fixture("arxiv.xml")]]);
  const works = await fetchArxiv(http, { categories: ["physics.geo-ph"], daysBack: 7, now: NOW });
  assert.equal(works.length, 1);
  const url = decodeURIComponent(http.calls[0].url.replace(/\+/g, " "));
  assert.match(url, /\(cat:physics\.geo-ph\) AND submittedDate:\[202410030000 TO 202410092359\]/);
  assert.equal(http.calls[0].responseType, "text");
});

test("eartharxiv OAI-PMH paging, deleted records and empty results", async () => {
  const page1 = parseOaiPage(fixture("oai-page1.xml"));
  assert.equal(page1.works.length, 1);
  assert.equal(page1.token, "page2token");
  const [w] = page1.works;
  assert.equal(w.doi, "10.31223/x5abc1");
  assert.equal(w.abstract, "Chalcophile <element> behaviour in arcs.");
  assert.equal(w.url, "https://eartharxiv.org/repository/view/7001/");
  assert.deepEqual(w.authors.map((a) => a.name), ["Anna Smith", "Ming Li"]);

  assert.deepEqual(parseOaiPage(fixture("oai-empty.xml")), { works: [], token: null });

  const http = mockHttp([
    [(r) => r.url.includes("resumptionToken=page2token"), () => fixture("oai-page2.xml")],
    ["https://eartharxiv.org/api/oai/", () => fixture("oai-page1.xml")],
  ]);
  const works = await fetchEartharxiv(http, { daysBack: 14, now: NOW });
  assert.equal(works.length, 2);
  assert.equal(works[1].id, "eartharxiv:oai:eartharxiv.org:id:7002");
  assert.equal(works[1].published, "2024-10-06T08:30:00Z");
});

test("openalex abstract reconstruction and batched lookups", () => {
  assert.equal(reconstructAbstract({ Zircon: [0], ages: [1, 3], "old,": [2] }), "Zircon ages old, ages");
  const url = decodeURIComponent(openAlexAbstractsUrl(["10.1234/a", "10.1234/c"], "me@example.org"));
  assert.match(url, /filter=doi:https:\/\/doi.org\/10.1234\/a\|https:\/\/doi.org\/10.1234\/c&select=doi,abstract_inverted_index&per-page=200&mailto=me@example.org$/, "50 DOIs, room for records sharing a DOI");
  const found = parseOpenAlexAbstracts({
    results: [{ doi: "https://doi.org/10.1234/A", abstract_inverted_index: { Hello: [0], world: [1] } }, null, { doi: 5 }, { doi: "10.1234/c" }],
  });
  assert.deepEqual([...found], [["10.1234/a", "Hello world"]]);
  // Malformed bodies yield nothing instead of throwing
  for (const body of [null, "<html>", { results: "x" }, { results: [{ doi: "10.1/x", abstract_inverted_index: { a: "b" } }] }]) {
    assert.equal(parseOpenAlexAbstracts(body).size, 0);
  }
});

test("retries honour Retry-After and never retry a cancelled request", async () => {
  assert.equal(retryAfterMs("7"), 7000);
  assert.equal(retryAfterMs("Wed, 21 Oct 2015 07:28:10 GMT", Date.parse("Wed, 21 Oct 2015 07:28:00 GMT")), 10000);
  assert.equal(retryAfterMs(null), null);
  assert.equal(retryAfterMs("soon"), null);

  const delays = [];
  let n = 0;
  const throttled = {
    async request() {
      return n++ === 0 ? { status: 429, data: null, retryAfter: "30" } : { status: 200, data: { ok: true } };
    },
  };
  assert.deepEqual(await getJson(throttled, "https://x.test/a", { sleep: async (ms) => delays.push(ms) }), { ok: true });
  assert.deepEqual(delays, [30000]);

  let calls = 0;
  const cancelling = {
    async request() {
      calls++;
      throw new CancelledRequestError();
    },
  };
  await assert.rejects(getJson(cancelling, "https://x.test/b", { sleep: async () => {} }), (e) => e.cancelled === true);
  assert.equal(calls, 1);
});

test("cancelling ends a retry wait and blocks later requests", async () => {
  const signal = createCancelSignal();
  let calls = 0;
  const transport = {
    signal,
    async request() {
      signal.throwIfCancelled();
      calls++;
      return { status: 429, data: null, retryAfter: "30" };
    },
  };
  // A real (long) sleep: only the cancel can end it
  const pending = getJson(transport, "https://x.test/c", { sleep: (ms) => new Promise((r) => setTimeout(r, ms)) });
  await new Promise((r) => setTimeout(r, 10));
  signal.cancel();
  await assert.rejects(pending, (e) => e.cancelled === true);
  assert.equal(calls, 1);
  await assert.rejects(getJson(transport, "https://x.test/d"), (e) => e.cancelled === true);
  assert.equal(calls, 1, "no request after cancel");

  const late = [];
  signal.onCancel(() => late.push("ran"));
  assert.deepEqual(late, ["ran"], "listeners added after cancel run immediately");
});

test("mapWithConcurrency stops handing out work after a failure", async () => {
  const started = [];
  await assert.rejects(
    mapWithConcurrency([1, 2, 3, 4, 5, 6], 2, async (n) => {
      started.push(n);
      await new Promise((r) => setTimeout(r, n === 1 ? 1 : 20));
      if (n === 1) throw new Error("boom");
      return n;
    }),
    /boom/,
  );
  await new Promise((r) => setTimeout(r, 60));
  assert.deepEqual(started, [1, 2]);
});

test("OpenAlex requests carry the API key when one is set", () => {
  assert.deepEqual(openAlexAuth({ mailto: "me@x.org", apiKey: " k " }), { mailto: "me@x.org", api_key: "k" });
  assert.deepEqual(openAlexAuth(), { mailto: "", api_key: "" });
  const url = decodeURIComponent(openAlexAbstractsUrl(["10.1234/a"], "me@example.org", "oa-key"));
  assert.match(url, /&per-page=200&mailto=me@example.org&api_key=oa-key$/);
  assert.doesNotMatch(decodeURIComponent(openAlexAbstractsUrl(["10.1234/a"], "", "")), /api_key|mailto/);
});

test("OpenAlex budget: the /rate-limit body, header-based usage and its text", () => {
  const usage = parseOpenAlexRateLimit({ rate_limit: { daily_budget_usd: 1, daily_used_usd: 0.25, resets_at: "2026-10-18T00:00:00Z" } }, 42);
  assert.deepEqual(usage, { remainingUsd: 0.75, limitUsd: 1, remaining: null, limit: null, creditsUsed: null, keyed: true, resetsAt: "2026-10-18T00:00:00Z", at: 42 });
  assert.equal(openAlexUsageText(usage), "今日剩余 $0.75 / $1.00");
  assert.equal(openAlexUsageText(usage, { compact: true }), "今日剩余 $0.75");
  assert.equal(openAlexUsageText({ remaining: 5, limit: 10 }, { compact: true }), "今日剩余 5 次额度");
  assert.equal(openAlexUsageText({ remainingUsd: 0.004 }), "今日剩余 $0.004");
  assert.equal(openAlexUsageText({ remaining: 99990, limit: 100000 }), "今日剩余 99,990 / 100,000 次额度");
  assert.equal(openAlexUsageText({ remaining: 7 }), "今日剩余 7 次额度");
  assert.equal(openAlexUsageText(null), "");
  assert.equal(openAlexUsageText({}), "");
  for (const body of [null, "x", [], {}, { rate_limit: "x" }, { rate_limit: { daily_budget_usd: "lots" } }]) {
    assert.equal(parseOpenAlexRateLimit(body), null, JSON.stringify(body));
  }
});

test("the OpenAlex check reads the budget with a key and reports keyless access as a notice", async () => {
  resetApiUsage();
  const answers = { rateLimit: { status: 200, data: { rate_limit: { daily_budget_usd: 1, daily_remaining_usd: 0.9 } } }, works: { status: 200, data: { results: [] } } };
  const http = mockHttp([
    ["https://api.openalex.org/rate-limit", () => answers.rateLimit],
    ["https://api.openalex.org/works", () => answers.works],
  ]);
  assert.deepEqual(await checkOpenAlexRow(http, { apiKey: "k", now: 7 }), { key: "openalex", name: "OpenAlex", ok: true, detail: "Key 有效，今日剩余 $0.90 / $1.00" });
  assert.equal(http.calls[0].url, "https://api.openalex.org/rate-limit?api_key=k");
  assert.equal(apiUsage.openalex.remainingUsd, 0.9);
  assert.equal(apiUsage.openalex.at, 7);

  answers.rateLimit = { status: 401, data: {} };
  assert.deepEqual(await checkOpenAlexRow(http, { apiKey: "bad" }), { key: "openalex", name: "OpenAlex", ok: false, detail: "Key 无效（HTTP 401）" });
  // No /rate-limit endpoint: a one-record query decides
  answers.rateLimit = { status: 404, data: {} };
  http.calls.length = 0;
  const fallback = await checkOpenAlexRow(http, { apiKey: "k", mailto: "me@x.org" });
  assert.equal(fallback.ok, true);
  assert.match(decodeURIComponent(http.calls[1].url), /^https:\/\/api\.openalex\.org\/works\?per-page=1&select=id&mailto=me@x\.org&api_key=k$/);
  answers.works = { status: 403, data: {} };
  assert.equal((await checkOpenAlexRow(http, { apiKey: "k" })).detail, "Key 无效（HTTP 403）");
  answers.rateLimit = { status: 429, data: {} };
  assert.match((await checkOpenAlexRow(http, { apiKey: "k" })).detail, /HTTP 429/);

  // Keyless: reachable is only a notice (ok null); a used-up budget says what to do
  answers.works = { status: 200, data: { results: [] } };
  http.calls.length = 0;
  assert.deepEqual(await checkOpenAlexRow(http, {}), { key: "openalex", name: "OpenAlex", ok: null, detail: "可访问，但未配置 API Key（每日免费额度很小）" });
  assert.equal(http.calls.length, 1);
  assert.doesNotMatch(http.calls[0].url, /rate-limit|api_key/);
  answers.works = { status: 429, data: {} };
  assert.match((await checkOpenAlexRow(http, {})).detail, /请填写 API Key/);
  answers.works = { status: 503, data: {} };
  assert.equal((await checkOpenAlexRow(http, {})).detail, "HTTP 503");
  const offline = { request: async () => { throw new Error("网络连接失败：api.openalex.org"); } };
  assert.equal((await checkOpenAlexRow(offline, { apiKey: "k" })).detail, "网络连接失败：api.openalex.org");
  resetApiUsage();
});
