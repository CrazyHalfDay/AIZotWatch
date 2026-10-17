import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { parseAuthorName, toZoteroCreator } from "../src/core/authors.js";
import { dedupeAgainstLibrary, ratio, tokenSetRatio } from "../src/core/dedupe.js";
import { excludeByKeywords, filterRecent, includeByKeywords, limitPreprints } from "../src/core/filters.js";
import { isChineseCoreCategory, libraryVenues, parseWhitelist, serializeWhitelist, splitCsvLine } from "../src/core/journals.js";
import {
  cleanHtml,
  cleanTitle,
  decodeEntities,
  extractDoi,
  extraFieldDoi,
  hashString,
  isNonArticleTitle,
  normalizeDoi,
  normalizeIssn,
  normalizeTitle,
  parseList,
} from "../src/core/text.js";
import { parseXml } from "../src/core/xml.js";

test("text helpers", () => {
  assert.equal(decodeEntities("a &amp; b &#x3B1; &#946; &unknown;"), "a & b α β &unknown;");
  assert.equal(cleanHtml("<jats:title>Abstract</jats:title><jats:p>Mantle  &amp; crust.</jats:p>"), "Mantle & crust.");
  assert.equal(cleanHtml("<p>Summary statistics of zircon</p>"), "Summary statistics of zircon");
  assert.equal(cleanHtml(""), null);
  assert.equal(cleanTitle("CO<sub>2</sub> in <i>arc</i>  magmas"), "CO2 in arc magmas");
  assert.equal(normalizeTitle("Zircon U–Pb Ages: A Review!"), "zircon u pb ages a review");
  assert.equal(normalizeDoi("https://doi.org/10.1016/J.GCA.2024.01.001."), "10.1016/j.gca.2024.01.001");
  assert.equal(normalizeDoi("doi:10.1038/s41561-024-0001-x"), "10.1038/s41561-024-0001-x");
  assert.equal(normalizeDoi("not a doi"), "");
  assert.equal(extractDoi("arXiv: 2401.1\nDOI: 10.5194/gmd-17-1-2024"), "10.5194/gmd-17-1-2024");
  assert.equal(extraFieldDoi("Original DOI: 10.1111/old\nDOI: 10.5194/GMD-17-1-2024"), "10.5194/gmd-17-1-2024");
  assert.equal(extraFieldDoi("Related: 10.1111/other"), "");
  assert.equal(normalizeIssn("0016 7037"), "0016-7037");
  assert.equal(normalizeIssn("2041-172x"), "2041-172X");
  assert.equal(normalizeIssn("123"), "");
  assert.deepEqual(parseList("a, b\nc；d;;"), ["a", "b", "c", "d"]);
  assert.equal(hashString("abc"), hashString("abc"));
  assert.notEqual(hashString("abc"), hashString("abd"));
});

test("non-article titles are filtered", () => {
  assert.ok(isNonArticleTitle("Table of Contents"));
  assert.ok(isNonArticleTitle("Information for Authors"));
  assert.ok(isNonArticleTitle("Geochimica et Cosmochimica Acta", "Geochimica et Cosmochimica Acta"));
  assert.ok(!isNonArticleTitle("Copper isotopes in porphyry deposits"));
});

test("author parsing", () => {
  assert.deepEqual(parseAuthorName("Smith, Anna"), { given: "Anna", family: "Smith", name: "Anna Smith" });
  assert.deepEqual(parseAuthorName("Jane Q. Doe"), { given: "Jane Q.", family: "Doe", name: "Jane Q. Doe" });
  assert.deepEqual(toZoteroCreator({ given: "", family: "IPCC", name: "IPCC" }), { creatorType: "author", name: "IPCC", fieldMode: 1 });
});

test("xml parser handles namespaces, CDATA, entities and strict mode", () => {
  const root = parseXml('<?xml version="1.0"?><a:root xmlns:a="x"><a:item k="v &amp; w">t&lt;1<![CDATA[<raw>]]></a:item><b/></a:root>');
  assert.equal(root.name, "root");
  const item = root.child("item");
  assert.equal(item.attr("k"), "v & w");
  assert.equal(item.text, "t<1<raw>");
  assert.equal(root.childrenNamed("b").length, 1);
  assert.throws(() => parseXml("<a><b></a>", { strict: true }));
  assert.throws(() => parseXml("<a><b>", { strict: true }));
  // What makes an XML parser (Gecko's expat) reject our XHTML: no DTD defines &nbsp;
  assert.throws(() => parseXml("<a>概&nbsp;览</a>", { strict: true }), /Undefined entity/);
  assert.throws(() => parseXml("<a>A & B</a>", { strict: true }), /bare "&"/);
  assert.throws(() => parseXml('<a title="A & B"/>', { strict: true }), /attribute title/);
  assert.throws(() => parseXml("<a><span class=x>y</span></a>", { strict: true }), /Malformed attributes/);
  assert.throws(() => parseXml('<a id="x" id="y"/>', { strict: true }), /Duplicate attribute id/);
  assert.equal(parseXml('<a title="&#160;&amp;&lt;&quot;">&#xA0;&gt;&apos;</a>', { strict: true }).attr("title"), " &<\"");
  // Lenient mode (feeds from the network) keeps going
  assert.equal(parseXml("<a>A & B</a>").text, "A & B");
});

test("fuzzy ratios match rapidfuzz semantics", () => {
  assert.equal(ratio("abc", "abc"), 100);
  assert.equal(Math.round(ratio("this is a test", "this is a test!")), 97);
  assert.equal(tokenSetRatio("fuzzy wuzzy was a bear", "fuzzy fuzzy was a bear"), 100);
  assert.ok(tokenSetRatio("mantle plume dynamics", "slab rollback in arcs") < 50);
});

test("dedupe against library by DOI, exact and fuzzy title, and within batch", () => {
  const library = [
    { doi: "10.5555/a", title: "Old paper" },
    { doi: "", title: "Copper isotope fractionation during magmatic sulfide segregation" },
  ];
  const works = [
    { id: "doi:10.5555/a", doi: "10.5555/A", title: "Different title" },
    { id: "doi:10.5555/b", doi: "10.5555/b", title: "Copper isotope fractionation during magmatic sulphide segregation" },
    { id: "doi:10.5555/c", doi: "10.5555/c", title: "Brand new subduction study" },
    { id: "arxiv:1", doi: null, title: "Brand new subduction study." },
    { id: "doi:10.5555/d", doi: "10.5555/d", title: "Dismissed earlier" },
  ];
  const kept = dedupeAgainstLibrary(works, library, { extraDois: ["10.5555/d"] });
  assert.deepEqual(kept.map((w) => w.id), ["doi:10.5555/c"]);
});

test("filters", () => {
  const now = Date.UTC(2024, 9, 10, 12);
  const works = [
    { source: "crossref", published: "2024-10-09T00:00:00Z" },
    { source: "crossref", published: "2024-09-01T00:00:00Z" },
    { source: "crossref", published: null },
  ];
  assert.equal(filterRecent(works, 7, now).length, 1);
  assert.equal(filterRecent(works, 0, now).length, 3);

  const mixed = ["arxiv", "arxiv", "crossref", "arxiv", "crossref"].map((source, i) => ({ source, i }));
  assert.deepEqual(limitPreprints(mixed, 0.5).map((w) => w.i), [2, 3, 4]);
  assert.equal(limitPreprints(mixed, 1).length, 5);

  const kw = [
    { title: "Lithium-ion battery cathode", abstract: "" },
    { title: "Mantle melting", abstract: "Batteryless sensors" },
    { title: "地幔熔融", abstract: "" },
  ];
  assert.deepEqual(excludeByKeywords(kw, ["battery"]).map((w) => w.title), ["Mantle melting", "地幔熔融"]);
  assert.deepEqual(includeByKeywords(kw, ["mantle", "地幔"]).map((w) => w.title), ["Mantle melting", "地幔熔融"]);
  assert.equal(includeByKeywords(kw, []).length, 3);
});

test("journal whitelist parsing and library venues", () => {
  assert.deepEqual(splitCsvLine('1234-5678,"Journal, The",GEO,"1.5"'), ["1234-5678", "Journal, The", "GEO", "1.5"]);
  const map = parseWhitelist("issn,title,category,impact_factor\n# comment\n0016-7037,GCA,GEOCHEM,5.0\n1869-1897,Sci China,GEOSCIENCES (CN),NA\nbad,row");
  assert.equal(map.size, 2);
  assert.equal(map.get("0016-7037").impactFactor, 5);
  assert.equal(map.get("1869-1897").isChineseCore, true);
  assert.equal(parseWhitelist(serializeWhitelist([...map.values()])).size, 2);
  assert.equal(map.get("0016-7037").isChineseCore, false);

  // Chinese core journals: "(CN)" anywhere or a "_CN" suffix (the preset and the Python data)
  const cn = parseWhitelist("1869-1897,Science China Earth Sciences,GEOSCIENCES_CN,5\n1000-0569,岩石学报,geology_cn,\n0016-7037,GCA,CNS_JOURNALS,5\n1674-7313,Sci China,library:14,");
  assert.deepEqual([...cn.values()].map((j) => j.isChineseCore), [true, true, false, false]);
  for (const [category, expected] of [["GEOSCIENCES (CN)", true], ["(cn) 地学", true], ["GEOSCIENCES_CN ", true], ["CN_GEO", false], ["", false], [null, false]]) {
    assert.equal(isChineseCoreCategory(category), expected, String(category));
  }
  // The bundled preset marks its Chinese journals with _CN
  const preset = parseWhitelist(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "../addon/content/presets/geoscience-journals.csv"), "utf8"));
  assert.equal(preset.get("1869-1897").isChineseCore, true);

  const venues = libraryVenues([
    { itemType: "journalArticle", venue: "Geochimica et Cosmochimica Acta", issn: "0016-7037" },
    { itemType: "journalArticle", venue: "Geochimica et cosmochimica acta", issn: "" },
    { itemType: "journalArticle", venue: "Nature", issn: "0028-0836, 1476-4687" },
    { itemType: "book", venue: "Ignored", issn: "" },
  ]);
  assert.equal(venues[0].count, 2);
  assert.deepEqual(venues[0].issns, ["0016-7037"]);
  assert.deepEqual(venues[1].issns, ["0028-0836", "1476-4687"]);
  assert.equal(venues.length, 2);
});
