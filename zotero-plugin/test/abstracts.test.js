// Abstract fallback (摘要补全): provider routing and parsing, the per-DOI chain,
// the positive/negative cache, provider error classes, concurrency limits, the
// publisher key check and the engine integration.

import assert from "node:assert/strict";
import test from "node:test";

import { CancelledRequestError } from "../src/core/http.js";
import { CancelledError, ZotWatchEngine } from "../src/engine.js";
import { loadSettings } from "../src/settings.js";
import {
  ABSTRACT_CACHE_FILE,
  MIN_ABSTRACT_CHARS,
  acceptAbstract,
  checkPublisherKeys,
  crossrefWorkUrl,
  doiPath,
  elsevierRequest,
  enrichMissingAbstracts,
  parseCrossrefAbstract,
  parseElsevierAbstract,
  parseSpringerAbstract,
  publisherOf,
  springerUrl,
} from "../src/sources/abstracts.js";
import { createMemoryStorage } from "../src/store.js";
import { abstractsStatusLine } from "../src/zotero/addon.js";
import { embeddingHandler, mockHttp, noSleep } from "./helpers.js";

const NOW = Date.UTC(2024, 9, 10, 12);
const DAY = 86400000;

/** An abstract long enough to be accepted, starting with `word`. */
const text = (word) => `${word} isotope systematics constrain the evolution of arc magmas and the sulfide reservoirs beneath them.`;

function invert(value) {
  const index = {};
  value.split(" ").forEach((word, i) => (index[word] ||= []).push(i));
  return index;
}

const elsevierBody = (description) => ({ "full-text-retrieval-response": { coredata: { "dc:description": description } } });
const work = (doi, source = "crossref", extra = {}) => ({ id: `doi:${doi}`, doi, source, title: `Work ${doi}`, abstract: null, ...extra });
const urls = (http) => http.calls.map((c) => decodeURIComponent(c.url));
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ------------------------------------------------------------------- units

test("abstract providers are routed by DOI prefix and DOIs are encoded in URL paths", () => {
  assert.equal(publisherOf("10.1016/j.gca.2024.01.001"), "elsevier");
  assert.equal(publisherOf("10.1016/J.EPSL.2024.1"), "elsevier");
  assert.equal(publisherOf("10.1007/s00410-024-0001-1"), "springer");
  assert.equal(publisherOf("10.1038/s41561-024-0001-1"), "springer");
  assert.equal(publisherOf("10.1029/2024GL000001"), null);
  assert.equal(publisherOf("10.10160/x"), null);
  assert.equal(publisherOf(""), null);

  assert.equal(doiPath("10.1002/(sici)1097#x?y%z;w"), "10.1002/(sici)1097%23x%3Fy%25z%3Bw");
  assert.equal(crossrefWorkUrl("10.31223/x5abc", "me@example.org"), "https://api.crossref.org/works/10.31223/x5abc?mailto=me%40example.org");
  assert.equal(crossrefWorkUrl("10.31223/x5abc"), "https://api.crossref.org/works/10.31223/x5abc");
  assert.deepEqual(elsevierRequest("10.1016/j.x#1", "KEY"), {
    url: "https://api.elsevier.com/content/article/doi/10.1016/j.x%231?view=META_ABS",
    headers: { "X-ELS-APIKey": "KEY", Accept: "application/json" },
  });
  assert.equal(decodeURIComponent(springerUrl("10.1007/abc", "KEY")), "https://api.springernature.com/meta/v2/json?q=doi:10.1007/abc&api_key=KEY");
});

test("abstract texts are cleaned, stripped of labels and need at least 80 characters", () => {
  const exact = "x".repeat(MIN_ABSTRACT_CHARS);
  assert.equal(acceptAbstract(exact), exact);
  assert.equal(acceptAbstract(exact.slice(1)), null);
  assert.equal(acceptAbstract("<p>Too short.</p>"), null);
  assert.equal(acceptAbstract(42), null);
  assert.equal(acceptAbstract(null), null);
  // The label alone does not count towards the minimum
  assert.equal(acceptAbstract(`Abstract: ${"y".repeat(MIN_ABSTRACT_CHARS - 2)}`), null);
  assert.equal(acceptAbstract(`Abstract ${"Y".repeat(MIN_ABSTRACT_CHARS - 2)}`), null);
  assert.equal(acceptAbstract(`Abstract reasoning ${"y".repeat(80)}`).slice(0, 18), "Abstract reasoning");

  assert.equal(parseCrossrefAbstract({ message: { abstract: `<jats:title>Abstract</jats:title><jats:p>${text("Copper")} &amp; more</jats:p>` } }), `${text("Copper")} & more`);
  assert.equal(parseElsevierAbstract(elsevierBody(`\n   Abstract\n   ${text("Zinc")}`)), text("Zinc"));
  assert.equal(parseElsevierAbstract(elsevierBody({ $: text("Zinc") })), null);
  assert.equal(parseSpringerAbstract({ records: [{ abstract: `Abstract${text("Iron")}` }] }), text("Iron"));
  assert.equal(parseSpringerAbstract({ records: [{ abstract: `ABSTRACT: ${text("Iron")}` }] }), text("Iron"));
  assert.equal(
    parseSpringerAbstract({ records: [{ abstract: { h1: "Abstract", p: ["Sulfur budgets of arcs.", text("Sulfur")] } }] }),
    `Sulfur budgets of arcs. ${text("Sulfur")}`,
  );
  // Malformed or empty bodies are misses, never exceptions
  for (const body of [null, undefined, "<html>", 5, { message: null }, { message: { abstract: ["x"] } }, { records: "x" }, { records: [] }, { records: [null] }, { "full-text-retrieval-response": null }]) {
    assert.equal(parseCrossrefAbstract(body), null);
    assert.equal(parseElsevierAbstract(body), null);
    assert.equal(parseSpringerAbstract(body), null);
  }
});

test("abstract settings default to Crossref and OpenAlex without publisher keys", () => {
  assert.deepEqual(loadSettings(() => undefined).abstracts, { crossref: true, openalex: true, elsevierKey: "", springerKey: "" });
  const prefs = { "abstracts.crossref": false, "sources.openalexAbstracts": "false", "abstracts.elsevierKey": " els ", "abstracts.springerKey": "spr" };
  assert.deepEqual(loadSettings((k) => prefs[k]).abstracts, { crossref: false, openalex: false, elsevierKey: "els", springerKey: "spr" });
});

// ------------------------------------------------------------------- chain

test("each DOI tries Crossref, Elsevier, Springer Nature and OpenAlex in order until one has an abstract", async () => {
  const http = mockHttp([
    ["https://api.crossref.org/works/10.31223/x5a?", () => ({ message: { abstract: `<jats:title>Abstract</jats:title><jats:p>${text("Alpha")}</jats:p>` } })],
    ["https://api.crossref.org/works/10.1016/j.b?", () => ({ message: { abstract: "<jats:p>Too short.</jats:p>" } })],
    ["https://api.elsevier.com/content/article/doi/10.1016/j.b?", () => elsevierBody(`Abstract ${text("Beta")}`)],
    ["https://api.elsevier.com/content/article/doi/10.1016/j.h?", () => elsevierBody(text("Eta"))],
    ["https://api.springernature.com/meta/v2/json", () => ({ records: [{ abstract: { h1: "Abstract", p: text("Gamma") } }] })],
    ["https://api.openalex.org/works", () => ({ results: [{ doi: "https://doi.org/10.1029/D", abstract_inverted_index: invert(text("Delta")) }] })],
  ]);
  const works = [
    work("10.31223/X5A", "eartharxiv"), // Crossref hit
    work("10.1016/j.b", "arxiv"), // Crossref too short, Elsevier hit
    work("10.1038/c"), // Crossref skipped (just fetched from Crossref), Springer hit
    work("10.1029/d"), // only OpenAlex
    work("10.5194/e", "openalex"), // Crossref 404, OpenAlex skipped (just fetched from OpenAlex)
    work("10.1016/j.f", "crossref", { abstract: "Has its own abstract" }),
    work(null, "eartharxiv"),
    work("10.1016/j.h"), // two works of different tracks share one DOI
    work("https://doi.org/10.1016/J.H", "openalex", { id: "openalex:W1" }),
  ];
  const storage = createMemoryStorage();
  const progress = [];
  const logs = [];
  const result = await enrichMissingAbstracts(http, works, {
    mailto: "me@example.org",
    elsevierKey: "els-key",
    springerKey: "spr-key",
    cache: storage,
    now: NOW,
    retry: noSleep,
    log: (m) => logs.push(m),
    onProgress: (done, total) => progress.push([done, total]),
  });

  assert.deepEqual(
    works.map((w) => w.abstract),
    [text("Alpha"), text("Beta"), text("Gamma"), text("Delta"), null, "Has its own abstract", null, text("Eta"), text("Eta")],
  );
  assert.deepEqual(result, {
    missing: 7,
    noDoi: 1,
    cacheHits: 0,
    negativeSkips: 0,
    crossref: 1,
    elsevier: 3,
    springer: 1,
    openalex: 1,
    filled: 6,
    failed: 1,
    errors: [],
  });
  const called = urls(http);
  assert.deepEqual(called.filter((u) => u.includes("api.crossref.org")).sort(), [
    "https://api.crossref.org/works/10.1016/j.b?mailto=me@example.org",
    "https://api.crossref.org/works/10.1016/j.h?mailto=me@example.org",
    "https://api.crossref.org/works/10.31223/x5a?mailto=me@example.org",
    "https://api.crossref.org/works/10.5194/e?mailto=me@example.org",
  ]);
  assert.equal(called.filter((u) => u.includes("api.elsevier.com")).length, 2, "one request per DOI");
  assert.deepEqual(called.filter((u) => u.includes("springernature")), ["https://api.springernature.com/meta/v2/json?q=doi:10.1038/c&api_key=spr-key"]);
  assert.deepEqual(called.filter((u) => u.includes("openalex")), [
    "https://api.openalex.org/works?filter=doi:https://doi.org/10.1029/d&select=doi,abstract_inverted_index&per-page=50&mailto=me@example.org",
  ]);
  const elsevier = http.calls.find((c) => c.url.includes("api.elsevier.com"));
  assert.deepEqual(elsevier.headers, { "X-ELS-APIKey": "els-key", Accept: "application/json" });
  assert.ok(http.calls.every((c) => c.timeout === 20000));
  assert.deepEqual(progress.at(-1), [6, 6]);
  assert.match(logs.at(-1), /^Abstracts: filled 6\/7 \(cache 0, crossref 1, elsevier 3, springer 1, openalex 1; 0 known misses skipped, 1 new misses cached\)$/);

  const cache = await storage.readJSON(ABSTRACT_CACHE_FILE);
  assert.equal(cache.version, 1);
  assert.deepEqual(cache.entries["10.1016/j.b"], { a: text("Beta"), s: "elsevier", exp: NOW + 30 * DAY });
  assert.deepEqual(cache.entries["10.5194/e"], { a: null, s: ["crossref", "openalex"], exp: NOW + 7 * DAY });
  assert.equal(Object.keys(cache.entries).length, 6);
});

test("cached abstracts and recent misses are reused, and expired entries are looked up again", async () => {
  const storage = createMemoryStorage();
  const http = mockHttp([
    ["https://api.crossref.org/works/10.31223/hit", () => ({ message: { abstract: text("Hit") } })],
    ["https://api.openalex.org/works", () => ({ results: [] })],
  ]);
  const batch = () => [work("10.31223/hit", "eartharxiv"), work("10.31223/miss", "eartharxiv"), work("10.31223/hit", "arxiv", { id: "arxiv:1" })];
  const run = (now, options = {}) => enrichMissingAbstracts(http, batch(), { cache: storage, now, retry: noSleep, ...options });

  const first = await run(NOW);
  assert.equal(first.crossref, 2);
  assert.equal(first.failed, 1);
  assert.equal(http.calls.length, 3); // two Crossref lookups and one OpenAlex batch

  http.calls.length = 0;
  const works = batch();
  const second = await enrichMissingAbstracts(http, works, { cache: storage, now: NOW + DAY, retry: noSleep });
  assert.equal(http.calls.length, 0, "everything answered from the cache");
  assert.equal(second.cacheHits, 2);
  assert.equal(second.negativeSkips, 1);
  assert.equal(second.filled, 2);
  assert.equal(works[2].abstract, text("Hit"));

  // A tier that was off when the miss was cached is tried once it is on
  http.calls.length = 0;
  await run(NOW + DAY, { crossref: false });
  assert.equal(http.calls.length, 0);
  const withKey = await run(NOW + DAY, { elsevierKey: "k" });
  assert.equal(withKey.negativeSkips, 1, "no Elsevier DOI here, so the miss still covers every tier");

  // The miss expires after 7 days, the abstract after 30
  http.calls.length = 0;
  const later = await run(NOW + 8 * DAY);
  assert.equal(later.cacheHits, 2);
  assert.deepEqual(urls(http).filter((u) => u.includes("crossref")), ["https://api.crossref.org/works/10.31223/miss"]);
  http.calls.length = 0;
  const much = await run(NOW + 31 * DAY);
  assert.equal(much.cacheHits, 0);
  assert.equal(much.crossref, 2);
});

test("a newly added publisher key retries DOIs cached as misses without it", async () => {
  const storage = createMemoryStorage();
  const http = mockHttp([
    ["https://api.elsevier.com/content/article/doi/", () => elsevierBody(text("Late"))],
    ["https://api.openalex.org/works", () => ({ results: [] })],
  ]);
  const first = await enrichMissingAbstracts(http, [work("10.1016/j.late")], { cache: storage, now: NOW, retry: noSleep });
  assert.equal(first.failed, 1);
  assert.deepEqual((await storage.readJSON(ABSTRACT_CACHE_FILE)).entries["10.1016/j.late"].s, ["crossref", "openalex"]);

  const works = [work("10.1016/j.late")];
  const second = await enrichMissingAbstracts(http, works, { cache: storage, now: NOW + DAY, elsevierKey: "k", retry: noSleep });
  assert.equal(second.elsevier, 1);
  assert.equal(works[0].abstract, text("Late"));
});

test("transient failures are never cached as misses", async () => {
  const storage = createMemoryStorage();
  let status = 503;
  const http = mockHttp([
    ["https://api.crossref.org/works/", () => ({ status, data: { message: "unavailable" } })],
    ["https://api.openalex.org/works", () => ({ status: 429, data: {} })],
  ]);
  const first = await enrichMissingAbstracts(http, [work("10.31223/flaky", "eartharxiv")], { cache: storage, now: NOW, retry: { ...noSleep, retries: 5 } });
  assert.equal(first.failed, 1);
  assert.deepEqual(first.errors, []);
  assert.equal(http.calls.length, 4, "retries are capped at one per request");
  assert.equal(await storage.readJSON(ABSTRACT_CACHE_FILE), null);

  status = 404;
  http.calls.length = 0;
  await enrichMissingAbstracts(http, [work("10.31223/flaky", "eartharxiv")], { cache: storage, now: NOW, retry: noSleep });
  assert.equal(http.calls.length, 3, "looked up again");
  assert.equal(await storage.readJSON(ABSTRACT_CACHE_FILE), null, "OpenAlex was still unavailable");
});

test("the cache tolerates corrupt files, prunes expired entries and keeps at most 5000", async () => {
  const http = mockHttp([["https://api.crossref.org/works/", () => ({ message: { abstract: text("New") } })]]);
  const broken = { ...createMemoryStorage(), readJSON: async () => { throw new SyntaxError("bad json"); } };
  const result = await enrichMissingAbstracts(http, [work("10.31223/new", "eartharxiv")], { cache: broken, openalex: false, now: NOW, retry: noSleep });
  assert.equal(result.crossref, 1);

  const storage = createMemoryStorage();
  const entries = {};
  for (let i = 0; i < 5003; i++) entries[`10.9/${i}`] = { a: `cached ${i}`, s: "openalex", exp: NOW + (i + 1) * 1000 };
  entries["10.9/expired"] = { a: null, s: ["crossref"], exp: NOW - 1 };
  entries["10.9/malformed"] = { a: 5, exp: NOW + DAY };
  await storage.writeJSON(ABSTRACT_CACHE_FILE, { version: 1, entries });
  await enrichMissingAbstracts(http, [work("10.31223/new", "eartharxiv")], { cache: storage, openalex: false, now: NOW, retry: noSleep });
  const saved = (await storage.readJSON(ABSTRACT_CACHE_FILE)).entries;
  assert.equal(Object.keys(saved).length, 5000);
  assert.ok(saved["10.31223/new"]);
  // The soonest-expiring entries go first
  for (const doi of ["10.9/0", "10.9/1", "10.9/2", "10.9/3", "10.9/expired", "10.9/malformed"]) assert.equal(saved[doi], undefined);
  assert.ok(saved["10.9/4"]);
});

// ---------------------------------------------------------- error classes

test("a rejected publisher key disables that publisher for the run with one error", async () => {
  const storage = createMemoryStorage();
  const http = mockHttp([
    ["https://api.elsevier.com/", () => ({ status: 401, data: { "service-error": { status: { statusText: "Invalid API Key" } } } })],
    ["https://api.springernature.com/", () => ({ status: 403, data: { message: "Forbidden" } })],
    ["https://api.openalex.org/works", () => ({ results: [] })],
  ]);
  const works = [work("10.1016/j.1"), work("10.1016/j.2"), work("10.1016/j.3"), work("10.1007/s1"), work("10.1007/s2")];
  const result = await enrichMissingAbstracts(http, works, { elsevierKey: "bad", springerKey: "bad", cache: storage, now: NOW, retry: noSleep, concurrency: 1 });
  assert.deepEqual(result.errors, [
    { source: "Elsevier", message: "API Key 无效或无权限（HTTP 401），本次运行已停用" },
    { source: "Springer Nature", message: "API Key 无效或无权限（HTTP 403），本次运行已停用" },
  ]);
  const called = urls(http);
  assert.equal(called.filter((u) => u.includes("elsevier")).length, 1);
  assert.equal(called.filter((u) => u.includes("springernature")).length, 1);
  assert.equal(called.filter((u) => u.includes("openalex")).length, 1, "OpenAlex still runs");
  // A disabled publisher is not a definitive miss: nothing is negative-cached
  assert.equal(await storage.readJSON(ABSTRACT_CACHE_FILE), null);
});

test("a provider failing again and again is skipped for the rest of the run", async () => {
  const http = mockHttp([
    ["https://api.springernature.com/", () => ({ status: 503, data: "Service Unavailable" })],
    ["https://api.crossref.org/works/", () => {
      throw new Error("网络连接失败：api.crossref.org");
    }],
  ]);
  const works = ["10.1038/a", "10.1038/b", "10.1038/c", "10.1038/d"].map((doi) => work(doi));
  const result = await enrichMissingAbstracts(http, works, { springerKey: "k", openalex: false, now: NOW, retry: noSleep, concurrency: 1 });
  assert.deepEqual(result.errors, [{ source: "Springer Nature", message: "连续 3 次请求失败，本次运行已跳过（HTTP 503）" }]);
  assert.equal(urls(http).length, 6, "three DOIs with one retry each, the fourth is skipped");
  assert.equal(result.failed, 4);

  const preprints = ["a", "b", "c", "d"].map((s) => work(`10.31223/${s}`, "eartharxiv"));
  const crossref = await enrichMissingAbstracts(http, preprints, { openalex: false, now: NOW, retry: { ...noSleep, retries: 0 }, concurrency: 1 });
  assert.deepEqual(crossref.errors, [{ source: "Crossref", message: "连续 3 次请求失败，本次运行已跳过（网络连接失败：api.crossref.org）" }]);
});

// ------------------------------------------------------------ concurrency

test("publisher calls share a limit of two while the per-DOI chain runs six at a time", async () => {
  const inFlight = { publisher: 0, crossref: 0 };
  const peak = { publisher: 0, crossref: 0 };
  const tracked = (kind, body) => async () => {
    inFlight[kind]++;
    peak[kind] = Math.max(peak[kind], inFlight[kind]);
    await delay(3);
    inFlight[kind]--;
    return body;
  };
  const http = mockHttp([
    ["https://api.elsevier.com/", tracked("publisher", elsevierBody(text("E")))],
    ["https://api.springernature.com/", tracked("publisher", { records: [{ abstract: text("S") }] })],
    ["https://api.crossref.org/works/", tracked("crossref", { message: { abstract: text("C") } })],
  ]);
  const works = [
    ...Array.from({ length: 8 }, (_, i) => work(`10.1016/j.${i}`)),
    ...Array.from({ length: 8 }, (_, i) => work(`10.1038/n${i}`)),
    ...Array.from({ length: 20 }, (_, i) => work(`10.31223/p${i}`, "eartharxiv")),
  ];
  const result = await enrichMissingAbstracts(http, works, { elsevierKey: "k", springerKey: "k", openalex: false, now: NOW, retry: noSleep });
  assert.equal(result.filled, 36);
  assert.equal(peak.publisher, 2);
  assert.ok(peak.crossref <= 6 && peak.crossref >= 4, `crossref peak ${peak.crossref}`);
});

// ----------------------------------------------------------- cancellation

test("cancellation stops the lookups, keeps the abstracts found so far and caches no misses", async () => {
  const storage = createMemoryStorage();
  const http = mockHttp([
    ["https://api.crossref.org/works/10.31223/first", () => ({ message: { abstract: text("First") } })],
    ["https://api.crossref.org/works/10.31223/second", () => {
      throw new CancelledRequestError();
    }],
    ["https://api.crossref.org/works/", () => ({ status: 404, data: {} })],
  ]);
  const works = ["first", "second", "third"].map((s) => work(`10.31223/${s}`, "eartharxiv"));
  await assert.rejects(
    enrichMissingAbstracts(http, works, { cache: storage, now: NOW, retry: noSleep, concurrency: 1 }),
    (e) => e.cancelled === true,
  );
  assert.equal(urls(http).length, 2, "nothing is requested after the cancellation");
  assert.deepEqual(Object.keys((await storage.readJSON(ABSTRACT_CACHE_FILE)).entries), ["10.31223/first"]);

  // The engine's checkpoint stops the pass between DOIs
  const idle = mockHttp([]);
  await assert.rejects(
    enrichMissingAbstracts(idle, works.slice(2), {
      now: NOW,
      checkpoint: () => {
        throw new CancelledError();
      },
    }),
    CancelledError,
  );
  assert.equal(idle.calls.length, 0);
});

// --------------------------------------------------------------- key check

test("publisher key checks tell accepted, rejected and unreachable keys apart", async () => {
  const none = mockHttp([]);
  assert.deepEqual(await checkPublisherKeys(none, { elsevierKey: "", springerKey: "" }), []);
  assert.equal(none.calls.length, 0);

  const ok = mockHttp([
    ["https://api.elsevier.com/", () => ({ status: 404, data: {} })],
    ["https://api.springernature.com/", () => ({ records: [] })],
  ]);
  assert.deepEqual(await checkPublisherKeys(ok, { elsevierKey: "e", springerKey: "s" }), ["✅ Elsevier：Key 有效", "✅ Springer Nature：Key 有效"]);
  assert.equal(ok.calls[0].headers["X-ELS-APIKey"], "e");
  assert.equal(ok.calls[0].timeout, 20000);
  assert.match(decodeURIComponent(ok.calls[1].url), /q=doi:10\.1038\/.+&api_key=s$/);

  const bad = mockHttp([
    ["https://api.elsevier.com/", () => ({ status: 401, data: {} })],
    ["https://api.springernature.com/", () => ({ status: 500, data: {} })],
  ]);
  assert.deepEqual(await checkPublisherKeys(bad, { elsevierKey: "e", springerKey: "s" }), ["❌ Elsevier：Key 无效（HTTP 401）", "❌ Springer Nature：HTTP 500"]);
  const offline = { request: async () => { throw new Error("网络连接失败：api.springernature.com"); } };
  assert.deepEqual(await checkPublisherKeys(offline, { springerKey: "s" }), ["❌ Springer Nature：网络连接失败：api.springernature.com"]);
});

test("the status line summarises the last run's abstract fallback", () => {
  assert.equal(
    abstractsStatusLine({ missing: 12, filled: 7, cacheHits: 3, crossref: 0, elsevier: 2, springer: 0, openalex: 2 }),
    "摘要补全：上次运行补全 7 / 12 篇（缓存 3，Elsevier 2，OpenAlex 2）",
  );
  assert.equal(abstractsStatusLine({ missing: 2, filled: 0 }), "摘要补全：上次运行补全 0 / 2 篇");
});

// ------------------------------------------------------------------ engine

function libraryItems() {
  const items = [];
  const topics = [
    ["mantle", "Mantle plume dynamics and mantle convection", "Mantle mantle upwelling"],
    ["copper", "Copper isotope systematics", "Copper isotope copper fractionation"],
  ];
  topics.forEach(([key, title, abstract]) => {
    for (let i = 0; i < 6; i++) {
      items.push({
        key: `${key.toUpperCase()}${i}`,
        itemType: "journalArticle",
        title: `${title} ${i}`,
        abstract: `${abstract} part ${i}`,
        doi: `10.9999/lib.${key}.${i}`,
        issn: "0016-7037",
        venue: "Geochimica et Cosmochimica Acta",
        dateAdded: "2024-01-01 00:00:00",
      });
    }
  });
  return items;
}

function crossrefWork(doi, title, abstract) {
  return {
    DOI: doi,
    title: [title],
    author: [{ given: "Ann", family: "Lee" }],
    abstract: abstract ? `<jats:p>${abstract}</jats:p>` : undefined,
    "container-title": ["Geochimica et Cosmochimica Acta"],
    created: { "date-time": "2024-10-08T10:00:00Z" },
    URL: `https://doi.org/${doi}`,
    type: "journal-article",
    ISSN: ["0016-7037"],
  };
}

// The copper paper has no abstract in Crossref; the keyword filter needs the one Elsevier provides
const ENRICHED = "Copper isotope fractionation records sulfide saturation in porphyry copper systems of continental arcs worldwide.";
const CANDIDATES = [
  crossrefWork("10.7777/cand.1", "Copper isotope fractionation in arc magmas", "Copper copper isotope sulfide isotope"),
  crossrefWork("10.1016/j.gca.2024.2", "Copper behaviour in arc crust", null),
  crossrefWork("10.1016/j.gca.2024.3", "Mantle wedge hydration", null),
];

function setup({ prefs = {}, elsevier = () => elsevierBody(ENRICHED), extraRoutes = [] } = {}) {
  const counter = {};
  const http = mockHttp([
    ...extraRoutes,
    ["https://api.crossref.org/works?", () => ({ message: { "total-results": CANDIDATES.length, items: CANDIDATES } })],
    ["https://api.elsevier.com/content/article/doi/10.1016/j.gca.2024.2?", elsevier],
    ["https://api.openalex.org/works", () => ({ results: [] })],
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)],
  ]);
  const allPrefs = {
    "embedding.provider": "dashscope",
    "embedding.apiKey": "sk-test",
    "journals.autoFromLibrary": true,
    "scoring.minSimilarity": "0.3",
    "scoring.includeKeywords": "sulfide",
    "abstracts.elsevierKey": "els-key",
    ...prefs,
  };
  const settings = loadSettings((k) => allPrefs[k]);
  const storage = createMemoryStorage();
  const library = { getItems: async () => libraryItems() };
  const engine = new ZotWatchEngine({ http, storage, library, settings, now: () => NOW, retry: noSleep });
  return { engine, http, storage };
}

test("the engine fills missing abstracts before the keyword filters and caches them for the next run", async () => {
  const { engine, http } = setup();
  const progress = [];
  const run = await engine.run({ onProgress: (p) => progress.push(p) });

  assert.deepEqual(run.errors, []);
  assert.deepEqual(run.stats.abstracts, {
    missing: 2,
    noDoi: 0,
    cacheHits: 0,
    negativeSkips: 0,
    crossref: 0,
    elsevier: 1,
    springer: 0,
    openalex: 0,
    filled: 1,
    failed: 1,
  });
  assert.equal(run.stats.abstractsEnriched, 1);
  assert.equal(run.stats.afterKeywords, 2, "the enriched abstract matches the include keyword");
  assert.equal(run.stats.afterAbstract, 2);
  const enrich = progress.filter((p) => p.stage === "enrich");
  assert.equal(enrich[0].message, "补全缺失摘要（2 篇）…");
  assert.deepEqual([enrich.at(-1).current, enrich.at(-1).total], [2, 2]);
  const stages = progress.map((p) => p.stage);
  assert.ok(stages.indexOf("dedupe") < stages.indexOf("enrich") && stages.indexOf("enrich") < stages.indexOf("embed"));
  // Crossref works skip the per-DOI Crossref lookup (their abstract was just requested)
  assert.ok(!urls(http).some((u) => u.startsWith("https://api.crossref.org/works/")));

  const inbox = await engine.loadInbox();
  assert.equal(inbox.works["doi:10.1016/j.gca.2024.2"]?.abstract, ENRICHED);

  http.calls.length = 0;
  const again = await engine.run();
  assert.equal(again.stats.abstracts.cacheHits, 1);
  assert.equal(again.stats.abstracts.negativeSkips, 1);
  assert.ok(!urls(http).some((u) => u.includes("elsevier") || u.includes("openalex")));
});

test("new followed-author works without an abstract go through the same pass", async () => {
  const followedWork = {
    id: "https://openalex.org/W9",
    doi: "https://doi.org/10.1016/j.epsl.2024.9",
    display_name: "Slab melts and arc copper",
    publication_date: "2024-10-01",
    type: "article",
    authorships: [{ author: { id: "https://openalex.org/A1", display_name: "Ann Lee", orcid: "https://orcid.org/0000-0002-1825-0097" } }],
    primary_location: { source: { display_name: "Earth and Planetary Science Letters", issn: ["0012-821X"] } },
    abstract_inverted_index: null,
  };
  const { engine, http } = setup({
    prefs: { "followed.enabled": true, "followed.authors": "Ann Lee, 0000-0002-1825-0097" },
    extraRoutes: [
      ["https://api.openalex.org/works?filter=author.orcid", () => ({ meta: { next_cursor: null }, results: [followedWork] })],
      ["https://api.elsevier.com/content/article/doi/10.1016/j.epsl.2024.9?", () => elsevierBody(text("Followed"))],
    ],
  });
  const run = await engine.run();
  assert.deepEqual(run.errors, []);
  assert.equal(run.stats.abstracts.missing, 3);
  assert.equal(run.stats.abstracts.elsevier, 2);
  const inbox = await engine.loadInbox();
  const followed = inbox.works["doi:10.1016/j.epsl.2024.9"];
  assert.equal(followed.label, "followed");
  assert.equal(followed.abstract, text("Followed"));
  // It came from OpenAlex, so the per-DOI Crossref lookup runs but the OpenAlex abstract batch skips it
  const called = urls(http);
  assert.ok(called.includes("https://api.crossref.org/works/10.1016/j.epsl.2024.9"));
  assert.ok(!called.some((u) => u.includes("filter=doi:") && u.includes("10.1016/j.epsl.2024.9")));
});

test("abstract provider problems are reported without failing the run", async () => {
  const { engine, storage } = setup({
    elsevier: () => ({ status: 401, data: {} }),
    extraRoutes: [
      ["https://api.openalex.org/works", () => ({ results: [{ doi: "https://doi.org/10.1016/j.gca.2024.3", abstract_inverted_index: invert(text("Wedge")) }] })],
    ],
  });
  const writeJSON = storage.writeJSON;
  storage.writeJSON = async (name, obj) => {
    if (name === ABSTRACT_CACHE_FILE) throw new Error("disk full");
    return writeJSON(name, obj);
  };
  const run = await engine.run();
  assert.deepEqual(run.errors, [{ stage: "enrich", source: "Elsevier", message: "API Key 无效或无权限（HTTP 401），本次运行已停用" }]);
  assert.equal(run.stats.abstracts.elsevier, 0);
  assert.equal(run.stats.abstracts.openalex, 1, "the next tier still runs; the failed cache write is only logged");
  assert.equal(await storage.readJSON(ABSTRACT_CACHE_FILE), null);
  assert.ok((await engine.loadInbox()).works["doi:10.7777/cand.1"]);
});

test("cancelling during abstract enrichment stops the run before anything is stored", async () => {
  const { engine, storage } = setup({
    elsevier: () => {
      throw new CancelledRequestError();
    },
  });
  await assert.rejects(engine.run(), (e) => e.cancelled === true);
  assert.equal(await storage.readJSON("inbox.json"), null);

  // The run's own cancel flag is honoured inside the pass too
  let cancelled = false;
  const other = setup();
  await assert.rejects(
    other.engine.run({
      onProgress: (p) => {
        if (p.stage === "enrich") cancelled = true;
      },
      isCancelled: () => cancelled,
    }),
    CancelledError,
  );
  assert.ok(!urls(other.http).some((u) => u.includes("elsevier")));
});

test("turning every abstract tier off skips the pass", async () => {
  const { engine, http } = setup({ prefs: { "abstracts.crossref": false, "sources.openalexAbstracts": false, "abstracts.elsevierKey": "" } });
  const stages = [];
  const run = await engine.run({ onProgress: (p) => stages.push(p.stage) });
  assert.ok(!stages.includes("enrich"));
  assert.equal(run.stats.abstracts, undefined);
  assert.equal(run.stats.afterKeywords, 1);
  assert.ok(!urls(http).some((u) => u.includes("elsevier") || u.includes("openalex")));
});
