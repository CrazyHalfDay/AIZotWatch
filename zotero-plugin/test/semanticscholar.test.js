// Semantic Scholar client (sources/semanticscholar.js): request shapes (key
// header, batch bodies, query parameters), defensive parsing of every
// endpoint, the work mapping and the key check.

import assert from "node:assert/strict";
import test from "node:test";

import {
  S2_BATCH_SIZE,
  S2_DEFAULT_FIELDS,
  checkSemanticScholarRow,
  mapS2Paper,
  s2AuthorSearch,
  s2Batch,
  s2Citations,
  s2DoiId,
  s2PathId,
  s2Query,
  s2Recommendations,
  s2References,
  s2Search,
} from "../src/sources/semanticscholar.js";
import { mockHttp, noSleep } from "./helpers.js";

const paper = (n, extra = {}) => ({
  paperId: `abc${n}`,
  externalIds: { DOI: `10.1016/J.GCA.${n}`, CorpusId: 1000 + n },
  title: `Copper isotopes <i>${n}</i>`,
  abstract: `Abstract ${n} about sulfide saturation in arc magmas &amp; porphyry deposits.`,
  tldr: { model: "tldr@v2.0.0", text: `A short summary ${n}.` },
  year: 2024,
  publicationDate: "2024-03-05",
  venue: "Geochim. Cosmochim. Acta",
  journal: { name: "Geochimica et Cosmochimica Acta", volume: "1" },
  authors: [{ authorId: "1", name: "Ann Lee" }, { authorId: "2", name: "Wang, Li" }, { name: null }, "junk"],
  citationCount: 12,
  openAccessPdf: { url: "https://example.org/a.pdf", status: "GREEN" },
  ...extra,
});

const urls = (http) => http.calls.map((c) => decodeURIComponent(c.url));

test("papers map to the plugin work shape and odd records are skipped", () => {
  const w = mapS2Paper(paper(1));
  assert.equal(w.id, "doi:10.1016/j.gca.1");
  assert.equal(w.source, "semanticscholar");
  assert.equal(w.title, "Copper isotopes 1");
  assert.equal(w.abstract, "Abstract 1 about sulfide saturation in arc magmas & porphyry deposits.");
  assert.equal(w.tldr, "A short summary 1.");
  assert.deepEqual(w.authors.map((a) => a.name), ["Ann Lee", "Li Wang"]);
  assert.equal(w.doi, "10.1016/j.gca.1");
  assert.equal(w.url, "https://doi.org/10.1016/j.gca.1");
  assert.equal(w.published, "2024-03-05T00:00:00Z");
  assert.equal(w.year, 2024);
  assert.equal(w.venue, "Geochimica et Cosmochimica Acta", "the journal name wins over the venue abbreviation");
  assert.equal(w.citations, 12);
  assert.equal(w.s2Id, "abc1");
  assert.equal(w.openAccessPdf, "https://example.org/a.pdf");

  // No DOI: an s2: id and the Semantic Scholar page; arXiv ids are kept
  const noDoi = mapS2Paper(paper(2, { externalIds: { ArXiv: "2401.00001" }, journal: null, venue: "arXiv.org", publicationDate: null, citationCount: "x", tldr: null, abstract: null, openAccessPdf: { url: "javascript:alert(1)" } }));
  assert.equal(noDoi.id, "s2:abc2");
  assert.equal(noDoi.url, "https://www.semanticscholar.org/paper/abc2");
  assert.equal(noDoi.arxivId, "2401.00001");
  assert.equal(noDoi.venue, "arXiv.org");
  assert.equal(noDoi.published, null);
  assert.equal(noDoi.year, 2024);
  assert.equal(noDoi.citations, 0);
  assert.equal(noDoi.tldr, null);
  assert.equal(noDoi.abstract, null);
  assert.equal(noDoi.openAccessPdf, null, "only http(s) links");

  for (const bad of [null, "x", [], {}, { title: "No ids" }, { paperId: "x", title: "  " }, { paperId: 5, title: "T", externalIds: "x" }]) {
    assert.equal(mapS2Paper(bad), null, JSON.stringify(bad));
  }
  assert.equal(mapS2Paper({ paperId: "p", title: "T", year: "2020", authors: "x", tldr: { text: 5 } }).year, null);
});

test("ids, queries and path segments", () => {
  assert.equal(s2DoiId("https://doi.org/10.1016/J.X"), "DOI:10.1016/j.x");
  assert.equal(s2DoiId("not a doi"), null);
  assert.equal(s2Query("  U-Pb   zircon — geochronology "), "U Pb zircon geochronology");
  assert.equal(s2PathId("DOI:10.1016/j.x#1"), "DOI:10.1016/j.x%231");
});

test("batch lookups send the key header, chunk by 500 and align results with the ids", async () => {
  const sent = [];
  const http = mockHttp([
    ["https://api.semanticscholar.org/graph/v1/paper/batch", (req) => {
      const body = JSON.parse(req.body);
      sent.push(body.ids.length);
      // Unknown ids come back as null; a malformed record counts as unknown
      return body.ids.map((id, i) => (id.endsWith(".404") ? null : i === 1 ? "junk" : { paperId: id, title: id }));
    }],
  ]);
  const ids = Array.from({ length: S2_BATCH_SIZE + 2 }, (_, i) => `DOI:10.1/x.${i === 0 ? "404" : i}`);
  const out = await s2Batch(http, ids, { key: " s2-key ", fields: "title,abstract", retry: noSleep });
  assert.deepEqual(sent, [500, 2]);
  assert.equal(out.length, ids.length);
  assert.equal(out[0], null);
  assert.equal(out[1], null);
  assert.equal(out[2].paperId, "DOI:10.1/x.2");
  assert.equal(out[S2_BATCH_SIZE].paperId, `DOI:10.1/x.${S2_BATCH_SIZE}`);
  const call = http.calls[0];
  assert.equal(call.method, "POST");
  assert.equal(decodeURIComponent(call.url), "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract");
  assert.deepEqual(call.headers, { "Content-Type": "application/json", "x-api-key": "s2-key" });

  // Default fields; no key, no header
  const plain = mockHttp([["https://api.semanticscholar.org/graph/v1/paper/batch", () => [null]]]);
  assert.deepEqual(await s2Batch(plain, ["CorpusId:1"]), [null]);
  assert.equal(decodeURIComponent(plain.calls[0].url).split("fields=")[1], S2_DEFAULT_FIELDS);
  assert.equal("x-api-key" in plain.calls[0].headers, false);
  assert.deepEqual(await s2Batch(plain, []), []);
  assert.equal(plain.calls.length, 1, "no request for no ids");

  // An error object instead of an array is a failure, not "all unknown"
  const odd = mockHttp([["https://api.semanticscholar.org/", () => ({ message: "Too Many Requests" })]]);
  await assert.rejects(s2Batch(odd, ["DOI:10.1/x"]), /unexpected response/);
  const failing = mockHttp([["https://api.semanticscholar.org/", () => ({ status: 403, data: { message: "Forbidden" } })]]);
  await assert.rejects(s2Batch(failing, ["DOI:10.1/x"], { retry: noSleep }), (e) => e.status === 403);
});

test("search, references, citations, recommendations and author search", async () => {
  const http = mockHttp([
    ["https://api.semanticscholar.org/graph/v1/paper/search", () => ({ total: 4321, offset: 0, next: 20, data: [paper(1), null, "x", paper(2)] })],
    ["https://api.semanticscholar.org/graph/v1/paper/abc1/references", () => ({
      data: [{ citedPaper: paper(3), isInfluential: true }, { citedPaper: { paperId: null, title: null } }, { citedPaper: null }, null, { citedPaper: paper(4) }],
    })],
    ["https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.x/citations", () => ({ data: [{ citingPaper: paper(5), isInfluential: false }] })],
    ["https://api.semanticscholar.org/recommendations/v1/papers", () => ({ recommendedPapers: [paper(6), 7, null] })],
    ["https://api.semanticscholar.org/graph/v1/author/search", () => ({
      total: 2,
      data: [
        { authorId: "123", name: " Ann  Lee ", affiliations: ["MIT", 5], paperCount: 40, citationCount: "900", hIndex: 12, externalIds: { ORCID: "0000-0002-1825-0097" } },
        { authorId: 9, name: "bad" },
      ],
    })],
  ]);
  const key = "k";
  const found = await s2Search(http, "copper-isotope fractionation", { key, limit: 500, year: 2016, retry: noSleep });
  assert.equal(found.total, 4321);
  assert.deepEqual(found.papers.map((p) => p.paperId), ["abc1", "abc2"]);
  const searchUrl = urls(http)[0];
  assert.match(searchUrl, /query=copper isotope fractionation&limit=100&fields=paperId,.*&year=2016-$/);
  assert.equal(http.calls[0].headers["x-api-key"], "k");
  await s2Search(http, "q", { year: "2010-2015", limit: 0, offset: 40 });
  assert.match(urls(http)[1], /query=q&limit=20&offset=40&fields=.*&year=2010-2015$/);
  assert.deepEqual(await s2Search(http, "  -  "), { total: 0, papers: [] }, "an empty query asks nothing");
  assert.equal(http.calls.length, 2);

  const refs = await s2References(http, "abc1", { key, limit: 5000 });
  assert.deepEqual(refs.map((p) => [p.paperId, p.influential]), [["abc3", true], ["abc4", false]]);
  assert.match(urls(http)[2], /\/paper\/abc1\/references\?fields=paperId,.*,isInfluential&limit=1000$/);
  const cits = await s2Citations(http, "DOI:10.1016/j.x");
  assert.deepEqual(cits.map((p) => p.paperId), ["abc5"]);
  assert.deepEqual(await s2References(http, ""), []);

  const recs = await s2Recommendations(http, ["abc1", "abc1", " ", "abc2"], { negativeIds: ["abc9"], key, limit: 50 });
  assert.deepEqual(recs.map((p) => p.paperId), ["abc6"]);
  const recCall = http.calls.at(-1);
  assert.equal(recCall.method, "POST");
  assert.match(decodeURIComponent(recCall.url), /^https:\/\/api\.semanticscholar\.org\/recommendations\/v1\/papers\?limit=50&fields=/);
  assert.deepEqual(JSON.parse(recCall.body), { positivePaperIds: ["abc1", "abc2"], negativePaperIds: ["abc9"] });
  assert.deepEqual(await s2Recommendations(http, []), []);

  const authors = await s2AuthorSearch(http, "Ann Lee", { key });
  assert.deepEqual(authors, [{ authorId: "123", name: "Ann Lee", affiliations: ["MIT"], paperCount: 40, citationCount: 900, hIndex: 12, orcid: "0000-0002-1825-0097" }]);

  // Malformed bodies give empty results
  const odd = mockHttp([["https://api.semanticscholar.org/", () => ({ data: "nope", recommendedPapers: {} })]]);
  assert.deepEqual(await s2Search(odd, "x"), { total: 0, papers: [] });
  assert.deepEqual(await s2Citations(odd, "abc"), []);
  assert.deepEqual(await s2Recommendations(odd, ["abc"]), []);
  assert.deepEqual(await s2AuthorSearch(odd, "x"), []);
});

test("the key check tells accepted, rejected, throttled and missing keys apart", async () => {
  const answer = { status: 200 };
  const http = mockHttp([["https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/nature14539", () => ({ status: answer.status, data: {} })]]);
  assert.deepEqual(await checkSemanticScholarRow(http, { key: "" }), { key: "semanticscholar", name: "Semantic Scholar", ok: null, detail: "未配置 API Key" });
  assert.equal(http.calls.length, 0);
  const row = await checkSemanticScholarRow(http, { key: "k" });
  assert.equal(row.ok, true);
  assert.equal(row.detail, "Key 有效");
  assert.equal(http.calls[0].headers["x-api-key"], "k");
  answer.status = 404;
  assert.equal((await checkSemanticScholarRow(http, { key: "k" })).ok, true);
  answer.status = 403;
  assert.deepEqual(await checkSemanticScholarRow(http, { key: "k" }), { key: "semanticscholar", name: "Semantic Scholar", ok: false, detail: "Key 无效（HTTP 403）" });
  answer.status = 429;
  assert.match((await checkSemanticScholarRow(http, { key: "k" })).detail, /HTTP 429/);
  answer.status = 502;
  assert.equal((await checkSemanticScholarRow(http, { key: "k" })).detail, "HTTP 502");
  const offline = { async request() {
    throw new Error("网络连接失败：api.semanticscholar.org");
  } };
  assert.match((await checkSemanticScholarRow(offline, { key: "k" })).detail, /网络连接失败/);
});
