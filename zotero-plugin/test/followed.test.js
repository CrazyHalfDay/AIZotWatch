// Followed authors (关注作者): line parsing, the OpenAlex source, the shared
// track model (merge rules, summary budget) and the engine integration.

import assert from "node:assert/strict";
import test from "node:test";

import { normalizeOpenAlexAuthorId, normalizeOrcid, parseFollowedAuthors } from "../src/core/authors.js";
import { CancelledRequestError } from "../src/core/http.js";
import { attachFollowed, countByTrack, selectSummaryTargets, trackOf, unionNames } from "../src/core/tracks.js";
import { CancelledError, FOLLOWED_KEYLESS_MESSAGE, ZotWatchEngine } from "../src/engine.js";
import { loadSettings } from "../src/settings.js";
import { fetchFollowedWorks, parseOpenAlexWork } from "../src/sources/followed.js";
import { createMemoryStorage } from "../src/store.js";
import { followedStatusLine, newRecommendationsNotice } from "../src/zotero/addon.js";
import { addWorkToZotero, summaryNoteHtml } from "../src/zotero/items.js";
import { embeddingHandler, mockHttp, noSleep } from "./helpers.js";

const NOW = Date.UTC(2024, 9, 10, 12);
const DAY = 86400000;
const ANN = { name: "Ann Lee", orcid: "0000-0002-1825-0097", key: "orcid:0000-0002-1825-0097" };
const WANG = { name: "汪在聪", orcid: "0000-0002-3584-1673", key: "orcid:0000-0002-3584-1673" };
const CHEN = { name: "Bo Chen", openalexId: "A5023888391", key: "openalex:A5023888391" };
const AUTHOR_LINES = `${ANN.name}, ${ANN.orcid}\n${WANG.name}，https://orcid.org/${WANG.orcid}`;

function invert(text) {
  const index = {};
  text.split(" ").forEach((word, i) => (index[word] ||= []).push(i));
  return index;
}

function oaWork(n, { doi = `10.8888/f.${n}`, title = `Followed paper ${n}`, date = "2024-10-01", type = "article", authors = [ANN], abstract = "Copper isotope systematics of arc magmas", ...extra } = {}) {
  return {
    id: `https://openalex.org/W${1000 + n}`,
    doi: doi ? `https://doi.org/${doi}` : null,
    title,
    display_name: title,
    publication_date: date,
    type,
    is_paratext: false,
    authorships: authors.map((a, i) => ({
      author: {
        id: `https://openalex.org/${a.openalexId || `A77${n}${i}`}`,
        display_name: a.name,
        orcid: a.orcid ? `https://orcid.org/${a.orcid}` : null,
      },
    })),
    primary_location: {
      landing_page_url: `https://publisher.example/${n}`,
      source: { display_name: "Earth and Planetary Science Letters", issn: ["0012-821X", "1385013X"] },
    },
    abstract_inverted_index: abstract ? invert(abstract) : null,
    cited_by_count: 3,
    ...extra,
  };
}

// ----------------------------------------------------------------- parsing

test("followed author lines accept names with ORCID iDs or OpenAlex ids in any form", () => {
  const { authors, invalid } = parseFollowedAuthors(
    [
      "汪在聪, 0000-0002-3584-1673",
      "# a comment, 0000-0001-0000-0000",
      "",
      "https://orcid.org/0000-0001-8408-0279 Jay J. Ague",
      "E.M. Stewart；0000000250923140",
      "Esther M. Schwarzenbach (ORCID: 0000-0002-4549-466x)",
      "Bo Chen | https://openalex.org/authors/a5023888391",
      "李继磊 ORCID：0000-0002-5847-4642",
      "Smith, John, Lee",
      "Second name for the same iD, https://orcid.org/0000-0002-3584-1673",
      "0000-0002-9340-9205",
      "  Jean-Pierre Dupont - A1234567  ",
    ].join("\r\n"),
  );
  assert.deepEqual(
    authors.map((a) => [a.name, a.key]),
    [
      ["汪在聪", "orcid:0000-0002-3584-1673"],
      ["Jay J. Ague", "orcid:0000-0001-8408-0279"],
      ["E.M. Stewart", "orcid:0000-0002-5092-3140"],
      ["Esther M. Schwarzenbach", "orcid:0000-0002-4549-466X"],
      ["Bo Chen", "openalex:A5023888391"],
      ["李继磊", "orcid:0000-0002-5847-4642"],
      ["0000-0002-9340-9205", "orcid:0000-0002-9340-9205"],
      ["Jean-Pierre Dupont", "openalex:A1234567"],
    ],
  );
  assert.equal(authors[0].orcid, "0000-0002-3584-1673");
  assert.equal(authors[4].openalexId, "A5023888391");
  assert.equal(authors[4].orcid, undefined);
  assert.deepEqual(invalid, ["Smith, John, Lee"]);

  assert.equal(normalizeOrcid("https://orcid.org/0000-0002-1825-009x"), "0000-0002-1825-009X");
  assert.equal(normalizeOrcid("no id"), "");
  assert.equal(normalizeOpenAlexAuthorId("https://openalex.org/A5023888391"), "A5023888391");
  assert.equal(normalizeOpenAlexAuthorId(null), "");
  assert.deepEqual(parseFollowedAuthors(undefined), { authors: [], invalid: [] });
});

test("followed settings are parsed with defaults and bounds", () => {
  const defaults = loadSettings(() => undefined).followed;
  assert.deepEqual(defaults, { enabled: false, authors: [], invalid: [], daysBack: 60 });
  const prefs = { "followed.enabled": true, "followed.authors": "Ann Lee, 0000-0002-1825-0097\nnobody", "followed.daysBack": "3" };
  const s = loadSettings((k) => prefs[k]).followed;
  assert.equal(s.enabled, true);
  assert.deepEqual(s.authors.map((a) => a.name), ["Ann Lee"]);
  assert.deepEqual(s.invalid, ["nobody"]);
  assert.equal(s.daysBack, 7, "clamped to at least a week");
  assert.equal(loadSettings((k) => ({ "followed.daysBack": "9999" })[k]).followed.daysBack, 365);
});

// ------------------------------------------------------------------ source

test("openalex works map to the plugin work shape", () => {
  const w = parseOpenAlexWork(
    oaWork(1, { title: "CO<sub>2</sub> in <i>arc</i> magmas", authors: [WANG, ANN], abstract: "Zircon <i>ages</i> &amp; melts" }),
    [ANN, WANG],
  );
  assert.equal(w.id, "doi:10.8888/f.1");
  assert.equal(w.source, "openalex");
  assert.equal(w.title, "CO2 in arc magmas");
  assert.equal(w.abstract, "Zircon ages & melts");
  assert.deepEqual(w.authors, [
    { given: "", family: "汪在聪", name: "汪在聪" },
    { given: "Ann", family: "Lee", name: "Ann Lee" },
  ]);
  assert.equal(w.doi, "10.8888/f.1");
  assert.equal(w.url, "https://doi.org/10.8888/f.1");
  assert.equal(w.published, "2024-10-01T00:00:00Z");
  assert.equal(w.venue, "Earth and Planetary Science Letters");
  assert.deepEqual(w.issns, ["0012-821X", "1385-013X"]);
  assert.equal(w.type, "article");
  assert.equal(w.citations, 3);
  assert.deepEqual(w.followedAuthors, ["Ann Lee", "汪在聪"], "configuration order, not authorship order");
  assert.equal(w.openalexId, "W1001");

  // No DOI: OpenAlex id and landing page; no date; preprints are kept
  const pre = parseOpenAlexWork(oaWork(2, { doi: null, date: null, type: "preprint" }), [ANN]);
  assert.equal(pre.id, "openalex:W1002");
  assert.equal(pre.url, "https://publisher.example/2");
  assert.equal(pre.published, null);
  assert.equal(pre.type, "preprint");
  const bare = parseOpenAlexWork({ ...oaWork(3, { doi: null }), primary_location: null }, [ANN]);
  assert.equal(bare.url, "https://openalex.org/W1003");
  assert.equal(bare.venue, null);
  assert.equal(bare.issns, null);

  // Skipped records
  assert.equal(parseOpenAlexWork(oaWork(4, { type: "dataset" }), [ANN]), null);
  assert.equal(parseOpenAlexWork(oaWork(5, { type: "peer-review" }), [ANN]), null);
  assert.equal(parseOpenAlexWork({ ...oaWork(6), is_paratext: true }, [ANN]), null);
  assert.equal(parseOpenAlexWork(oaWork(7, { title: "Erratum" }), [ANN]), null);
  assert.equal(parseOpenAlexWork(oaWork(8, { title: "  " }), [ANN]), null);
  assert.equal(parseOpenAlexWork({ ...oaWork(9, { doi: null }), id: null }, [ANN]), null);
  assert.equal(parseOpenAlexWork(null, [ANN]), null);
});

test("malformed openalex records never throw", () => {
  const raw = {
    ...oaWork(1),
    authorships: [null, { author: null }, { author: { display_name: null, orcid: null, id: null }, raw_author_name: "Raw Name" }],
    abstract_inverted_index: { Hello: null, world: 7, ok: [0], big: [1e9], neg: [-1], frac: [1.5] },
    cited_by_count: null,
  };
  const w = parseOpenAlexWork(raw, [ANN, WANG]);
  assert.deepEqual(w.authors.map((a) => a.name), ["Raw Name"]);
  assert.equal(w.abstract, "ok");
  assert.equal(w.citations, 0);
  assert.deepEqual(w.followedAuthors, [], "no authorship matches: not credited to every queried author");
  assert.equal(w.needsAttribution, true);
  // A one-author query can only have found its own author's work
  assert.deepEqual(parseOpenAlexWork(raw, [ANN]).followedAuthors, ["Ann Lee"]);
  assert.equal(parseOpenAlexWork(raw, [ANN]).needsAttribution, undefined);
  // A complete author list without a match needs no lookup
  assert.equal(parseOpenAlexWork({ ...raw, is_authors_truncated: false }, [ANN, WANG]).needsAttribution, undefined);
  const none = parseOpenAlexWork({ ...oaWork(2), authorships: null, primary_location: { source: { issn: "x" } } }, [ANN]);
  assert.deepEqual(none.authors, []);
  assert.deepEqual(none.followedAuthors, ["Ann Lee"]);
  assert.equal(none.issns, null);
  assert.deepEqual(parseOpenAlexWork({ ...oaWork(2), authorships: null }, [ANN, WANG]).followedAuthors, ["Ann Lee", "汪在聪"], "missing authorships");
});

test("works with a truncated author list are attributed from their full author list", async () => {
  const others = (n) => Array.from({ length: n }, (_, i) => ({ name: `Member ${i}`, orcid: `0000-0001-0000-${String(i).padStart(4, "0")}` }));
  // A consortium paper: OpenAlex lists show its first 100 authors only, without 汪在聪
  const big = oaWork(50, { authors: others(100), is_authors_truncated: true });
  const failing = oaWork(51, { authors: others(100), is_authors_truncated: true });
  const complete = oaWork(52, { authors: others(3), is_authors_truncated: false });
  const http = mockHttp([
    [(r) => decodeURIComponent(r.url).includes("author.orcid:"), () => ({ meta: { next_cursor: null }, results: [big, failing, complete] })],
    ["https://api.openalex.org/works/W1050?", () => ({ id: big.id, authorships: [...big.authorships, ...oaWork(50, { authors: [...others(30), WANG] }).authorships] })],
    ["https://api.openalex.org/works/W1051?", () => ({ status: 500, data: {} })],
  ]);
  const logs = [];
  const works = await fetchFollowedWorks(http, { authors: [ANN, WANG], mailto: "me@x.org", now: NOW, retry: noSleep, log: (m) => logs.push(m) });
  const byId = Object.fromEntries(works.map((w) => [w.id, w]));
  assert.deepEqual(byId["doi:10.8888/f.50"].followedAuthors, ["汪在聪"], "only the author actually on the paper");
  assert.deepEqual(byId["doi:10.8888/f.51"].followedAuthors, [], "a failed lookup names nobody and keeps the work");
  assert.deepEqual(byId["doi:10.8888/f.52"].followedAuthors, []);
  assert.ok(works.every((w) => !("needsAttribution" in w)));
  const lookups = http.calls.map((c) => decodeURIComponent(c.url)).filter((u) => u.startsWith("https://api.openalex.org/works/"));
  assert.deepEqual(lookups.map((u) => u.split("?")[0]), ["https://api.openalex.org/works/W1050", "https://api.openalex.org/works/W1051", "https://api.openalex.org/works/W1051", "https://api.openalex.org/works/W1051"]);
  assert.match(lookups[0], /\?select=id,authorships&mailto=me@x\.org$/);
  assert.ok(logs.some((m) => /author lookup failed for W1051/.test(m)));

  // With a key, the full-author lookups carry it too
  const keyed = mockHttp([
    [(r) => decodeURIComponent(r.url).includes("author.orcid:"), () => ({ meta: { next_cursor: null }, results: [big] })],
    ["https://api.openalex.org/works/W1050?", () => ({ id: big.id, authorships: big.authorships })],
  ]);
  await fetchFollowedWorks(keyed, { authors: [ANN, WANG], apiKey: "oa-key", now: NOW, retry: noSleep });
  const keyedLookup = keyed.calls.find((c) => c.url.startsWith("https://api.openalex.org/works/W1050"));
  assert.ok(keyedLookup, "the full author list was looked up");
  assert.ok(keyed.calls.every((c) => c.url.includes("api_key=oa-key")), "every OpenAlex request carries the key");

  // The note never shows an empty list of names
  assert.match(summaryNoteHtml({ label: "followed", score: 0, followedAuthors: [], summary: {} }), /<p>来源：关注作者<\/p>/);

  // Cancelling a lookup cancels the fetch
  const cancelled = mockHttp([
    [(r) => decodeURIComponent(r.url).includes("author.orcid:"), () => ({ meta: { next_cursor: null }, results: [big] })],
    ["https://api.openalex.org/works/W1050?", () => {
      throw new CancelledRequestError();
    }],
  ]);
  await assert.rejects(fetchFollowedWorks(cancelled, { authors: [ANN, WANG], now: NOW, retry: noSleep }), (e) => e.cancelled === true);
});

test("followed fetch queries OpenAlex by ORCID and author id, pages with cursors (with a key) and merges duplicates", async () => {
  const pages = {
    orcid: [
      { meta: { next_cursor: "page2" }, results: [oaWork(1, { authors: [ANN] }), oaWork(2, { authors: [WANG, ANN], date: "2024-09-20" })] },
      { meta: { next_cursor: null }, results: [oaWork(3, { type: "dataset" }), oaWork(4, { authors: [ANN], date: "2024-10-05" })] },
    ],
    // The same work found through the OpenAlex-id author as well
    id: [{ meta: { next_cursor: null }, results: [oaWork(2, { authors: [CHEN, WANG], date: "2024-09-20" })] }],
  };
  let orcidPage = 0;
  const http = mockHttp([
    [(r) => decodeURIComponent(r.url).includes("author.orcid:"), () => pages.orcid[orcidPage++]],
    [(r) => decodeURIComponent(r.url).includes("authorships.author.id:"), () => pages.id[0]],
  ]);
  const logs = [];
  let truncated = 0;
  const works = await fetchFollowedWorks(http, {
    authors: [ANN, WANG, CHEN],
    daysBack: 60,
    mailto: "me@x.org",
    apiKey: " oa-key ",
    now: NOW,
    retry: noSleep,
    log: (m) => logs.push(m),
    onTruncated: () => truncated++,
  });

  assert.equal(http.calls.length, 3);
  assert.equal(truncated, 0);
  assert.ok(http.calls.every((c) => c.url.endsWith("&api_key=oa-key")), "every OpenAlex request carries the key");
  const [first, second, third] = http.calls.map((c) => decodeURIComponent(c.url));
  assert.match(first, /^https:\/\/api\.openalex\.org\/works\?/);
  assert.match(first, /filter=author\.orcid:https:\/\/orcid\.org\/0000-0002-1825-0097\|https:\/\/orcid\.org\/0000-0002-3584-1673,from_publication_date:2024-08-11&/);
  assert.match(first, /sort=publication_date:desc/);
  assert.match(first, /per-page=200/);
  assert.match(first, /cursor=\*/);
  assert.match(first, /select=id,doi,title,display_name,publication_date,type,authorships,primary_location,abstract_inverted_index,cited_by_count,is_paratext&/);
  assert.match(first, /mailto=me@x.org/);
  assert.match(second, /cursor=page2/);
  assert.match(third, /filter=authorships\.author\.id:A5023888391,from_publication_date:2024-08-11&/);

  assert.deepEqual(works.map((w) => w.id), ["doi:10.8888/f.4", "doi:10.8888/f.1", "doi:10.8888/f.2"], "newest first, dataset skipped, duplicate merged");
  assert.deepEqual(works[2].followedAuthors, ["Ann Lee", "汪在聪", "Bo Chen"]);
  assert.match(logs.at(-1), /3 works/);

  // An empty page ends the paging even when it carries a cursor
  let emptyPages = 0;
  const empty = mockHttp([
    ["https://api.openalex.org/works", () => {
      if (++emptyPages > 1) throw new Error("paged past an empty page");
      return { meta: { next_cursor: "again" }, results: [] };
    }],
  ]);
  assert.deepEqual(await fetchFollowedWorks(empty, { authors: [ANN], now: NOW, retry: noSleep }), []);
  assert.equal(empty.calls.length, 1);

  // No usable author: no request at all
  const idle = mockHttp([]);
  assert.deepEqual(await fetchFollowedWorks(idle, { authors: [{ name: "x" }], now: NOW }), []);
  assert.equal(idle.calls.length, 0);
});

test("followed fetch splits large author lists and caps records per query", async () => {
  const many = Array.from({ length: 30 }, (_, i) => ({ name: `Author ${i}`, orcid: `0000-0000-0000-${String(i).padStart(4, "0")}` }));
  const http = mockHttp([["https://api.openalex.org/works", () => ({ meta: { next_cursor: "more" }, results: [1, 2, 3, 4, 5].map((n) => oaWork(n, { authors: [many[0]] })) })]]);
  const works = await fetchFollowedWorks(http, { authors: many, now: NOW, maxResults: 3 });
  assert.equal(http.calls.length, 2, "25 authors per query, no second page once the cap is reached");
  const filters = http.calls.map((c) => decodeURIComponent(c.url).match(/author\.orcid:([^,]+),/)[1].split("|").length);
  assert.deepEqual(filters, [25, 5]);
  assert.equal(works.length, 3);

  const failing = mockHttp([["https://api.openalex.org/works", () => ({ status: 400, data: { error: "Invalid query" } })]]);
  await assert.rejects(fetchFollowedWorks(failing, { authors: [ANN], now: NOW, retry: noSleep }), /HTTP 400.*Invalid query/);
});

test("without an OpenAlex key only the first page is read and the cut is reported", async () => {
  const page = (n, count) => ({ meta: { count, next_cursor: null }, results: Array.from({ length: n }, (_, i) => oaWork(i + 1, { authors: [ANN] })) });
  let answer = page(200, 950);
  const http = mockHttp([["https://api.openalex.org/works", () => answer]]);
  const logs = [];
  let truncated = 0;
  const works = await fetchFollowedWorks(http, { authors: [ANN], mailto: "me@x.org", now: NOW, retry: noSleep, log: (m) => logs.push(m), onTruncated: () => truncated++ });
  assert.equal(http.calls.length, 1, "no second page without a key");
  const url = decodeURIComponent(http.calls[0].url);
  assert.doesNotMatch(url, /cursor=/, "keyless requests do not page with cursors");
  assert.doesNotMatch(url, /api_key=/);
  assert.equal(works.length, 200);
  assert.equal(truncated, 1);
  assert.ok(logs.some((m) => /no API key, only the first page/.test(m)));

  // Everything fit on the first page: nothing to report
  answer = page(12, 12);
  truncated = 0;
  await fetchFollowedWorks(http, { authors: [ANN], now: NOW, retry: noSleep, onTruncated: () => truncated++ });
  assert.equal(truncated, 0);
});

// The engine turns the cut into a non-fatal error that names the setting
test("a keyless followed fetch that was cut short becomes a non-fatal run error", async () => {
  const { engine, state } = setup();
  state.followedHandler = () => ({ meta: { count: 450, next_cursor: null }, results: [oaWork(77, { authors: [WANG], date: "2024-10-05" })] });
  const run = await engine().run();
  const followedErrors = run.errors.filter((e) => e.stage === "followed");
  assert.deepEqual(followedErrors.map((e) => [e.source, e.message]), [["OpenAlex", FOLLOWED_KEYLESS_MESSAGE]]);
  assert.match(FOLLOWED_KEYLESS_MESSAGE, /学术数据接口/);
  assert.equal(run.stats.followed.fetched, 1, "the first page is still used");

  // With a key the same answer is complete (paging would continue) and nothing is reported
  const keyed = setup({ prefs: { "apis.openalexKey": "oa-key" } });
  keyed.state.followedHandler = state.followedHandler;
  const run2 = await keyed.engine().run();
  assert.deepEqual(run2.errors.filter((e) => e.stage === "followed"), []);
  assert.ok(keyed.http.calls.filter((c) => c.url.startsWith("https://api.openalex.org/")).every((c) => c.url.includes("api_key=oa-key")));
});

// -------------------------------------------------------------- track model

test("track helpers: labels, ordered unions, attachment and the summary budget", () => {
  assert.equal(trackOf({ label: "must_read" }), "personal");
  assert.equal(trackOf({ label: "flagship" }), "flagship");
  assert.equal(trackOf({ label: "followed" }), "followed");
  assert.equal(trackOf(undefined), "personal");
  assert.deepEqual(unionNames(["B", "X"], ["A", "B", ""], ["A", "B"]), ["A", "B", "X"]);
  assert.deepEqual(unionNames(undefined, ["B", "A"]), ["B", "A"]);

  const personal = { id: "arxiv:2410.00001", doi: "10.1000/A", label: "consider", score: 0.6 };
  const flagship = { id: "doi:10.1000/b", doi: "10.1000/b", label: "flagship", score: 0.2, followedAuthors: ["B"] };
  const only = attachFollowed(
    [personal, flagship],
    [
      { id: "doi:10.1000/a", doi: "https://doi.org/10.1000/a", followedAuthors: ["B"] },
      { id: "doi:10.1000/b", doi: "10.1000/b", followedAuthors: ["A"] },
      { id: "doi:10.1000/c", doi: "10.1000/c", followedAuthors: ["A"], score: 0.9 },
    ],
    ["A", "B"],
  );
  assert.deepEqual(personal.followedAuthors, ["B"], "matched by DOI");
  assert.deepEqual(flagship.followedAuthors, ["A", "B"], "matched by id, configuration order");
  assert.deepEqual(only, [{ id: "doi:10.1000/c", doi: "10.1000/c", followedAuthors: ["A"], score: 0, similarity: 0, label: "followed" }]);

  const works = [
    { id: "p1", label: "must_read", score: 0.9 },
    { id: "p2", label: "consider", score: 0.7, summary: { a: 1 } },
    { id: "p3", label: "ignore", score: 0.8 },
    { id: "f1", label: "flagship", score: 0.9, published: "2024-01-01T00:00:00Z" },
    { id: "f2", label: "flagship", score: 0.1, published: "2024-02-01T00:00:00Z" },
    { id: "a1", label: "followed", score: 0, published: null },
    { id: "a2", label: "followed", score: 0, published: "2024-03-01T00:00:00Z" },
  ];
  assert.deepEqual(selectSummaryTargets(works, 1).map((w) => w.id), ["p1", "f2", "a2"]);
  assert.deepEqual(selectSummaryTargets(works, 2).map((w) => w.id), ["p1", "p3", "f2", "f1", "a2", "a1"]);
  assert.deepEqual(selectSummaryTargets(works, 0), []);
  assert.deepEqual(countByTrack(works), { personal: 3, flagship: 2, followed: 2 });
});

test("notes, notices and status lines describe the track of a work", () => {
  const personal = summaryNoteHtml({ label: "consider", score: 0.8123, similarity: 0.7, summary: { methodology: "方法 <b>" }, followedAuthors: ["A", "B"] });
  assert.match(personal, /<strong>研究方法：<\/strong>方法 &lt;b&gt;/);
  assert.match(personal, /<p>推荐分数：81（相似度 70）<\/p>/);
  assert.match(personal, /<p>关注作者：A、B<\/p>/);
  const flagship = summaryNoteHtml({ label: "flagship", score: 0.3, similarity: 0.456, topic: "俯冲带 <流体>", summary: {} });
  assert.match(flagship, /<p>来源：顶刊地学速览（与文库相似度 46）<\/p>/);
  assert.match(flagship, /<p>研究方向：俯冲带 &lt;流体&gt;<\/p>/);
  assert.doesNotMatch(flagship, /推荐分数/);
  const followed = summaryNoteHtml({ label: "followed", score: 0, similarity: 0, followedAuthors: ["汪在聪", "<x>"], summary: {} });
  assert.match(followed, /<p>来源：关注作者（汪在聪、&lt;x&gt;）<\/p>/);
  assert.doesNotMatch(followed, /推荐分数|NaN/);
  assert.doesNotMatch(summaryNoteHtml({ label: "consider", score: 0.5, summary: {} }), /NaN/);

  assert.equal(
    newRecommendationsNotice({ newRecommendations: 5, mustRead: 3, newMustRead: 1, newByLabel: { personal: 2, flagship: 1, followed: 2 } }),
    "发现 5 篇新推荐（1 篇必读），顶刊地学 1 篇，关注作者 2 篇。在「工具 → ZotWatch 文献推荐」中查看。",
  );
  assert.equal(
    newRecommendationsNotice({ newRecommendations: 3, mustRead: 2, newMustRead: 0 }),
    "发现 3 篇新推荐。在「工具 → ZotWatch 文献推荐」中查看。",
    "must-read papers announced by an earlier run are not counted again",
  );
  assert.equal(followedStatusLine({ authors: [ANN, WANG], invalid: [], daysBack: 60 }), "关注作者：2 位（Ann Lee、汪在聪），检查最近 60 天");
  assert.equal(followedStatusLine({ authors: [], invalid: ["x", "y"], daysBack: 30 }), "关注作者：未填写有效的作者；2 行无法识别：x");
});

test("followed works that are not journal articles are saved with their own item type", async (t) => {
  const saved = [];
  class Item {
    constructor(itemType) {
      this.itemType = itemType;
      this.fields = {};
    }
    setField(field, value) {
      // Zotero rejects fields that the item type lacks
      const valid = { journalArticle: ["publicationTitle", "ISSN"], bookSection: ["bookTitle"], report: ["institution"], thesis: ["university"], book: [] };
      if (!["title", "abstractNote", "DOI", "url", "date"].includes(field) && !valid[this.itemType]?.includes(field)) throw new Error(`invalid field ${field}`);
      this.fields[field] = value;
    }
    setCreators(creators) {
      this.creators = creators;
    }
    async saveTx() {
      saved.push(this);
    }
  }
  globalThis.Zotero = { Item, ItemTypes: { getID: () => 31 }, Collections: {} };
  t.after(() => delete globalThis.Zotero);
  const save = (type, extra = {}) =>
    addWorkToZotero({ id: "openalex:W1", source: "openalex", type, title: `A ${type}`, venue: "ProQuest", issns: ["1234-5678"], authors: [], ...extra }, { libraryID: 1, useTranslator: false });

  const thesis = await save("dissertation");
  assert.equal(thesis.itemType, "thesis");
  assert.deepEqual(thesis.fields, { title: "A dissertation" }, "the repository is not the university");
  assert.equal((await save("report", { venue: "USGS" })).fields.institution, "USGS");
  assert.equal((await save("book-chapter", { venue: "Treatise on Geochemistry" })).fields.bookTitle, "Treatise on Geochemistry");
  assert.equal((await save("book")).itemType, "book");
  const article = await save("article", { venue: "EPSL" });
  assert.equal(article.itemType, "journalArticle");
  assert.deepEqual([article.fields.publicationTitle, article.fields.ISSN], ["EPSL", "1234-5678"]);
  assert.equal((await save("preprint")).itemType, "preprint");
  // Other sources keep the journal-article default
  assert.equal((await save("report", { source: "crossref" })).itemType, "journalArticle");
});

// ------------------------------------------------------------------ engine

function libraryItems() {
  const items = [];
  const topics = [
    ["mantle", "Mantle plume dynamics and mantle convection", "Mantle mantle upwelling"],
    ["subduction", "Subduction zone fluids", "Subduction subduction slab dehydration"],
    ["copper", "Copper isotope systematics", "Copper isotope copper fractionation"],
  ];
  for (const [key, title, abstract] of topics) {
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
  }
  return items;
}

function crossrefWork(n, title, abstract, { doi = `10.7777/cand.${n}`, created = "2024-10-08T10:00:00Z" } = {}) {
  return {
    DOI: doi,
    title: [title],
    author: [{ given: "Ann", family: "Lee" }],
    abstract: `<jats:p>${abstract}</jats:p>`,
    "container-title": ["Geochimica et Cosmochimica Acta"],
    created: { "date-time": created },
    URL: `https://doi.org/${doi}`,
    type: "journal-article",
    ISSN: ["0016-7037"],
  };
}

const CANDIDATES = [
  crossrefWork(1, "Copper isotope fractionation in arc magmas", "Copper copper isotope isotope"),
  crossrefWork(2, "Mantle convection revisited", "Mantle mantle"),
  crossrefWork(3, "Battery cathode chemistry", "Battery battery cathode"),
];

const FOLLOWED = [
  oaWork(1, { title: "A new copper isotope tracer", authors: [ANN], date: "2024-10-01" }),
  // Same paper as Crossref candidate 1: annotates the personal recommendation
  oaWork(2, { doi: "10.7777/cand.1", title: "Copper isotope fractionation in arc magmas", authors: [WANG] }),
  // Already in the library
  oaWork(3, { doi: "10.9999/lib.mantle.0", title: "Mantle plume dynamics and mantle convection 0", authors: [ANN] }),
  oaWork(4, { type: "dataset", title: "Supplementary data", authors: [ANN] }),
  // Preprint without DOI by both followed authors, off-topic and without abstract
  oaWork(5, { doi: null, type: "preprint", title: "Galaxy survey notes", authors: [WANG, ANN], date: "2024-09-20", abstract: null }),
];

function setup({ prefs = {}, llm } = {}) {
  const counter = {};
  const state = { candidates: CANDIDATES, followed: FOLLOWED, followedHandler: null };
  const isFollowedQuery = (r) => /author\.orcid:|authorships\.author\.id:/.test(decodeURIComponent(r.url));
  const routes = [
    ["https://api.crossref.org/works", () => ({ message: { "total-results": state.candidates.length, items: state.candidates } })],
    [isFollowedQuery, (req) => (state.followedHandler ? state.followedHandler(req) : { meta: { next_cursor: null }, results: state.followed })],
    ["https://api.openalex.org/works", () => ({ results: [] })],
    ["https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings", embeddingHandler(counter)],
  ];
  if (llm) routes.push(["https://llm.example/v1/chat/completions", llm]);
  const http = mockHttp(routes);
  const allPrefs = {
    "embedding.provider": "dashscope",
    "embedding.apiKey": "sk-test",
    "journals.autoFromLibrary": true,
    "scoring.minSimilarity": "0.3",
    "followed.enabled": true,
    "followed.authors": AUTHOR_LINES,
    ...prefs,
  };
  const storage = createMemoryStorage();
  const items = libraryItems();
  let now = NOW;
  const engine = (overrides = {}) =>
    new ZotWatchEngine({
      http,
      storage,
      library: { getItems: async () => items },
      settings: loadSettings((k) => ({ ...allPrefs, ...overrides })[k]),
      now: () => now,
      retry: noSleep,
    });
  return { engine, http, storage, state, counter, setNow: (v) => (now = v) };
}

test("followed works join the inbox once, annotate matching recommendations and are never scored", async () => {
  const { engine, http, storage, state, counter } = setup();
  const stages = [];
  const run = await engine().run({ onProgress: (p) => stages.push(p) });
  assert.deepEqual(run.errors, []);
  assert.ok(stages.some((p) => p.stage === "followed" && p.message === "获取关注作者的新论文…"));
  assert.deepEqual(run.stats.followed, { authors: 2, fetched: 4, new: 3 });
  assert.equal(run.stats.newByLabel.followed, 2);
  assert.equal(run.stats.newByLabel.flagship, 0);
  assert.equal(run.stats.newRecommendations, run.stats.newByLabel.personal + 2);
  assert.match(stages.at(-1).message, /，关注作者 2 篇$/);
  const query = decodeURIComponent(http.calls.find((c) => c.url.includes("author.orcid")).url);
  assert.match(query, /author\.orcid:https:\/\/orcid\.org\/0000-0002-1825-0097\|https:\/\/orcid\.org\/0000-0002-3584-1673,from_publication_date:2024-08-11/);

  const works = (await engine().loadInbox()).works;
  const followed = works["doi:10.8888/f.1"];
  assert.equal(followed.label, "followed");
  assert.equal(followed.score, 0);
  assert.equal(followed.similarity, 0);
  assert.equal(followed.status, "new");
  assert.equal(followed.source, "openalex");
  assert.deepEqual(followed.followedAuthors, ["Ann Lee"]);
  assert.equal(followed.openalexId, undefined, "only inbox fields are stored");
  const preprint = works["openalex:W1005"];
  assert.equal(preprint.label, "followed", "no abstract and off-topic: kept anyway");
  assert.deepEqual(preprint.followedAuthors, ["Ann Lee", "汪在聪"]);
  const personal = works["doi:10.7777/cand.1"];
  assert.notEqual(personal.label, "followed");
  assert.ok(personal.score > 0);
  assert.deepEqual(personal.followedAuthors, ["汪在聪"], "a personal recommendation gains the annotation");
  assert.ok(!works["doi:10.9999/lib.mantle.0"], "library duplicates are dropped");
  assert.equal(Object.values(works).filter((w) => w.title === "Copper isotope fractionation in arc magmas").length, 1);
  // Followed works are never embedded
  const embedded = counter.texts;
  const embeddedTexts = http.calls.filter((c) => c.url.startsWith("https://dashscope.aliyuncs.com/")).flatMap((c) => JSON.parse(c.body).input);
  assert.ok(embeddedTexts.length > 0);
  assert.ok(!embeddedTexts.some((t) => /A new copper isotope tracer|Galaxy survey notes/.test(t)), "followed-only works are not embedded");

  const seen = await storage.readJSON("followed.json");
  assert.equal(seen.version, 1);
  for (const id of ["doi:10.8888/f.1", "doi:10.7777/cand.1", "doi:10.9999/lib.mantle.0", "openalex:W1005", "openalex:W1001"]) {
    assert.equal(seen.seen[id], NOW, id);
  }

  // Second run: same responses, nothing new from the followed authors
  await engine().updateWork("doi:10.8888/f.1", { status: "dismissed" });
  const run2 = await engine().run();
  assert.deepEqual(run2.stats.followed, { authors: 2, fetched: 4, new: 0 });
  assert.equal(run2.stats.newByLabel.followed, 0);
  assert.equal(counter.texts, embedded, "nothing new embedded");
  const works2 = (await engine().loadInbox()).works;
  assert.equal(works2["doi:10.8888/f.1"].status, "dismissed");
  assert.deepEqual(works2["doi:10.7777/cand.1"].followedAuthors, ["汪在聪"], "annotation kept on re-merge");

  // A stored followed-only work that later ranks as a personal recommendation
  state.followed = [...FOLLOWED, oaWork(9, { doi: "10.8888/new.9", title: "Galaxy survey notes on copper isotope", authors: [ANN] })];
  const run3 = await engine().run();
  assert.equal(run3.stats.newByLabel.followed, 1);
  const stored = (await engine().loadInbox()).works["doi:10.8888/new.9"];
  assert.equal(stored.label, "followed");
  state.candidates = [...CANDIDATES, crossrefWork(9, "Galaxy survey notes on copper isotope", "Copper copper isotope isotope", { doi: "10.8888/new.9" })];
  // The followed fetch no longer returns it: the annotation can only come from the stored entry
  state.followed = FOLLOWED;
  const run4 = await engine().run();
  const promoted = (await engine().loadInbox()).works["doi:10.8888/new.9"];
  assert.notEqual(promoted.label, "followed");
  assert.ok(promoted.score > 0);
  assert.deepEqual(promoted.followedAuthors, ["Ann Lee"], "the annotation survives the overwrite");
  assert.equal(promoted.firstRunId, stored.firstRunId);
  assert.deepEqual(run4.stats.newByLabel, { personal: 0, flagship: 0, followed: 0 });

  // Found through another followed author: the stored names are kept (configuration order)
  state.followed = [...FOLLOWED, oaWork(9, { doi: "10.8888/new.9", title: "Galaxy survey notes on copper isotope", authors: [WANG] })];
  await engine().run();
  assert.deepEqual((await engine().loadInbox()).works["doi:10.8888/new.9"].followedAuthors, ["Ann Lee", "汪在聪"]);
});

test("a followed work already in the inbox only gains its authors", async () => {
  const { engine, state } = setup();
  state.followed = [];
  await engine().run();
  const before = (await engine().loadInbox()).works["doi:10.7777/cand.2"];
  assert.ok(before && before.label !== "followed");

  // Next run: candidate 2 is no longer fetched, but a followed author wrote it
  state.candidates = [CANDIDATES[0]];
  state.followed = [oaWork(20, { doi: "10.7777/CAND.2", title: "Mantle convection revisited", authors: [ANN] })];
  const run = await engine().run();
  assert.equal(run.stats.newByLabel.followed, 0);
  const after = (await engine().loadInbox()).works;
  assert.equal(after["doi:10.7777/cand.2"].label, before.label);
  assert.equal(after["doi:10.7777/cand.2"].score, before.score);
  assert.equal(after["doi:10.7777/cand.2"].status, "new");
  assert.equal(after["doi:10.7777/cand.2"].lastRunId, before.lastRunId);
  assert.deepEqual(after["doi:10.7777/cand.2"].followedAuthors, ["Ann Lee"]);

  // Matched by DOI when the stored entry has another id (e.g. an arXiv preprint)
  await engine().withInbox((inbox) => {
    inbox.works["arxiv:2410.00001"] = { id: "arxiv:2410.00001", source: "arxiv", title: "Preprint", doi: "10.5555/ARX.1", label: "consider", score: 0.5, status: "new", lastSeen: NOW };
  });
  state.followed = [oaWork(22, { doi: "10.5555/arx.1", title: "Slab melting and arc crust", authors: [WANG] })];
  const run2 = await engine().run();
  assert.equal(run2.stats.newByLabel.followed, 0);
  const inbox = (await engine().loadInbox()).works;
  assert.ok(!inbox["doi:10.5555/arx.1"]);
  assert.deepEqual(inbox["arxiv:2410.00001"].followedAuthors, ["汪在聪"]);
  assert.equal(inbox["arxiv:2410.00001"].label, "consider");
});

test("the notice counts only the must-read papers this run added", async () => {
  const { engine, state } = setup();
  state.followed = [];
  const first = await engine().run();
  assert.ok(first.stats.mustRead > 0);
  assert.equal(first.stats.newMustRead, first.stats.mustRead, "every must-read paper is new on the first run");

  // Only a followed-author paper is new; the must-read papers are still ranked
  state.followed = [FOLLOWED[0]];
  const second = await engine().run();
  assert.equal(second.stats.newRecommendations, 1);
  assert.equal(second.stats.mustRead, first.stats.mustRead);
  assert.equal(second.stats.newMustRead, 0);
  assert.equal(newRecommendationsNotice(second.stats), "发现 1 篇新推荐，关注作者 1 篇。在「工具 → ZotWatch 文献推荐」中查看。");
});

test("followed works fetched again annotate their stored entries in any status", async () => {
  const { engine, state, setNow } = setup({ prefs: { "followed.authors": `${ANN.name}, ${ANN.orcid}` } });
  state.followed = [
    oaWork(30, { title: "Joint copper paper", authors: [ANN, WANG] }),
    oaWork(31, { title: "Joint mantle paper", authors: [WANG, ANN] }),
  ];
  await engine().run();
  const before = (await engine().loadInbox()).works;
  assert.deepEqual(before["doi:10.8888/f.30"].followedAuthors, ["Ann Lee"]);
  await engine().updateWork("doi:10.8888/f.31", { status: "dismissed" });
  const dismissed = (await engine().loadInbox()).works["doi:10.8888/f.31"];

  // 汪在聪 is followed too from now on: the works were seen, so they are not new
  setNow(NOW + DAY);
  const run = await engine({ "followed.authors": AUTHOR_LINES }).run();
  assert.deepEqual(run.stats.followed, { authors: 2, fetched: 2, new: 0 });
  assert.equal(run.stats.newByLabel.followed, 0);
  const after = (await engine().loadInbox()).works;
  assert.deepEqual(after["doi:10.8888/f.30"].followedAuthors, ["Ann Lee", "汪在聪"]);
  assert.equal(after["doi:10.8888/f.30"].lastSeen, before["doi:10.8888/f.30"].lastSeen, "an annotation is not a sighting");
  assert.equal(after["doi:10.8888/f.30"].lastRunId, before["doi:10.8888/f.30"].lastRunId);
  assert.deepEqual(after["doi:10.8888/f.31"].followedAuthors, ["Ann Lee", "汪在聪"], "handled entries are annotated as well");
  assert.equal(after["doi:10.8888/f.31"].status, "dismissed");
  assert.equal(after["doi:10.8888/f.31"].statusAt, dismissed.statusAt);
});

test("invalid author lines and OpenAlex failures do not fail the run", async () => {
  const { engine, storage, state } = setup({ prefs: { "followed.authors": `${AUTHOR_LINES}\nJust a name` } });
  state.followedHandler = () => ({ status: 503, data: { message: "busy" } });
  const run = await engine().run();
  assert.deepEqual(run.errors.map((e) => [e.stage, e.source, e.message.replace(/HTTP 503.*$/, "HTTP 503")]), [
    ["followed", undefined, "无法识别的作者行：Just a name"],
    ["followed", "OpenAlex", "HTTP 503"],
  ]);
  assert.ok(run.stats.recommended > 0, "personal recommendations still arrive");
  assert.equal(run.stats.followed, undefined);
  assert.equal(run.stats.newByLabel.followed, 0);
  assert.equal(await storage.readJSON("followed.json"), null, "the seen-set is only written after a successful fetch");

  // Disabled: no OpenAlex author query and no errors even with bad lines
  const quiet = setup({ prefs: { "followed.enabled": false, "followed.authors": "Just a name" } });
  const run2 = await quiet.engine().run();
  assert.deepEqual(run2.errors, []);
  assert.ok(!quiet.http.calls.some((c) => decodeURIComponent(c.url).includes("author.orcid")));
});

test("cancelling during the followed stage stops the run before anything is stored", async () => {
  const { engine, storage, state } = setup();
  state.followedHandler = () => {
    throw new CancelledRequestError();
  };
  await assert.rejects(engine().run(), (e) => e.cancelled === true);
  assert.equal(await storage.readJSON("followed.json"), null);
  assert.equal((await engine().loadInbox()).lastRun, null);

  state.followedHandler = null;
  let stage = null;
  await assert.rejects(engine().run({ onProgress: (p) => (stage = p.stage), isCancelled: () => stage === "followed" }), CancelledError);
  assert.equal(await storage.readJSON("followed.json"), null);
  assert.deepEqual((await engine().loadInbox()).works, {});
});

test("the seen-set forgets works long after they left the search window", async () => {
  const { engine, storage, setNow } = setup();
  await storage.writeJSON("followed.json", {
    version: 1,
    seen: { "doi:old": NOW - 121 * DAY, "doi:recent": NOW - 100 * DAY, "doi:10.8888/f.1": NOW - 5 * DAY },
  });
  const run = await engine().run();
  assert.equal(run.stats.followed.new, 2, "f.1 was seen before");
  const { seen } = await storage.readJSON("followed.json");
  assert.equal(seen["doi:old"], undefined, "older than daysBack + 60 days");
  assert.equal(seen["doi:recent"], NOW - 100 * DAY);
  assert.equal(seen["doi:10.8888/f.1"], NOW, "refreshed when fetched again");

  // A garbled file is treated as empty; the inbox still prevents duplicates
  await storage.writeJSON("followed.json", { version: 1, seen: ["x"] });
  setNow(NOW + DAY);
  const run2 = await engine().run();
  assert.equal(run2.stats.followed.new, 3);
  assert.equal(run2.stats.newByLabel.followed, 1, "only the work hidden by the lost seen-set is added; stored ones are annotated");
});

test("the LLM budget covers every track: titles for all new works, summaries per track", async () => {
  const prompts = [];
  const llm = (req) => {
    const prompt = JSON.parse(req.body).messages[0].content;
    prompts.push(prompt);
    if (prompt.includes("翻译")) {
      const n = (prompt.match(/^\d+\. /gm) || []).length;
      return { choices: [{ message: { content: JSON.stringify(Array.from({ length: n }, (_, i) => `标题${i}`)) } }] };
    }
    return { choices: [{ message: { content: '{"research_question":"问题"}' } }] };
  };
  const { engine } = setup({
    // Topic labels have their own tests (test/topics.test.js); this mock answers translations and summaries only
    prefs: { "llm.enabled": true, "llm.apiKey": "k", "llm.baseURL": "https://llm.example/v1", "llm.model": "m", "llm.summaryTopN": "1", "llm.topicLabels": false },
    llm,
  });
  const run = await engine().run();
  assert.deepEqual(run.errors, []);
  const works = Object.values((await engine().loadInbox()).works);
  assert.ok(works.every((w) => w.translatedTitle), "every new work of every track is translated");
  const summarized = works.filter((w) => w.summary);
  assert.equal(summarized.length, 2, "one personal and one followed-only work");
  const best = works.filter((w) => w.label !== "followed").sort((a, b) => b.score - a.score)[0];
  assert.ok(best.summary);
  assert.ok(works.find((w) => w.id === "doi:10.8888/f.1").summary, "newest followed-only work");
  assert.ok(!works.find((w) => w.id === "openalex:W1005").summary);
  assert.equal(prompts.filter((p) => p.includes("翻译")).length, 1, "one translation batch for all tracks");
});
