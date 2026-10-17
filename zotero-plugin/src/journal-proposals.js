// Journal list proposals (关注期刊 →「从文库生成」) and 「补全学科与影响因子」,
// independent of Zotero like engine.js: http, storage and the LLM client are
// injected.
//
// 「从文库生成」 proposes, it does not write: the library's most frequent
// venues (ISSNs from the items, else a Crossref title lookup, cached), each
// enriched from OpenAlex (by ISSN, or by exact title when the venue has no
// ISSN) with a subject category and OpenAlex's two-year mean citedness as an
// approximate impact factor; with the LLM on, the model also normalises the
// names and suggests related journals, and every suggestion is verified
// through OpenAlex (unverifiable ones are dropped; the impact factor always
// comes from OpenAlex). The settings page shows the proposals for review and
// merges the chosen ones (core/journals.js mergeWhitelistRows).
//
// OpenAlex or LLM failures never fail the whole proposal: they end up in
// `warnings` and the proposals carry what is known. Cancellation (the
// transport's signal) is never swallowed.

import { isChineseCoreCategory, isIncompleteJournal, libraryVenues, mergeWhitelistRows, parseWhitelist, venueKey } from "./core/journals.js";
import { suggestJournalsWithLlm } from "./llm/journals.js";
import { resolveJournalIssns } from "./sources/crossref.js";
import { findSourceByTitle, lookupSourcesByIssn, openAlexErrorText } from "./sources/openalex-entities.js";

const DAY = 86400000;
export const JOURNAL_CACHE_FILE = "journals-cache.json";
export const JOURNAL_CACHE_TTL = 90 * DAY;
// OpenAlex title searches run a few at a time (the rate limiter paces them further)
const LOOKUP_CONCURRENCY = 3;
// Library venues without any ISSN looked up by title, and AI suggestions verified, per run
export const MAX_TITLE_SEARCHES = 40;
export const MAX_AI_SUGGESTIONS = 30;
// Venues shown to the model
const LLM_VENUES = 40;
/** AI suggestions start checked from this approximate impact factor. */
export const AI_DEFAULT_MIN_IF = 2;

/**
 * Fill the ISSNs of venues without any from the Crossref /journals endpoint
 * (exact title match), through the shared title cache (journals-cache.json,
 * 90 days). Mutates `venues` (v.issns). A failed lookup is logged and leaves
 * the venue without ISSNs; a cancellation is rethrown (the cache is then not written).
 */
export async function resolveVenueIssns(http, storage, venues, { mailto = "", retry = {}, now = () => Date.now(), log = () => {} } = {}) {
  if (!venues.some((v) => !v.issns.length)) return venues;
  const stored = await storage.readJSON(JOURNAL_CACHE_FILE).catch(() => null);
  const cache = stored && typeof stored === "object" && stored.titles && typeof stored.titles === "object" ? stored : { titles: {} };
  let dirty = false;
  for (const v of venues) {
    if (v.issns.length) continue;
    const key = venueKey(v.title);
    const hit = cache.titles[key];
    if (hit && Array.isArray(hit.issns) && now() - hit.at < JOURNAL_CACHE_TTL) {
      v.issns = hit.issns;
      continue;
    }
    try {
      v.issns = await resolveJournalIssns(http, v.title, { mailto, retry });
      cache.titles[key] = { issns: v.issns, at: now() };
      dirty = true;
    } catch (e) {
      if (e?.cancelled) throw e;
      log(`Journal lookup failed for "${v.title}": ${e.message}`);
    }
  }
  if (dirty) await storage.writeJSON(JOURNAL_CACHE_FILE, cache);
  return venues;
}

// Words ISO 4 abbreviations leave out ("J. Geophys. Res." = Journal of Geophysical Research)
const TITLE_STOP_WORDS = new Set(["of", "the", "and", "for", "in", "on", "a", "an", "et", "de", "des", "du", "la", "le", "und", "der", "die", "das", "y"]);

/**
 * Whether `short` abbreviates `full` word by word, ISO 4 style ("Precambrian
 * Res." -> "Precambrian Research", "Geochim. Cosmochim. Acta" -> "Geochimica
 * et Cosmochimica Acta"): the same number of words (articles and
 * prepositions left out), each a prefix of the full word.
 */
export function abbreviates(short, full) {
  const words = (title) => venueKey(title).split(" ").filter((w) => w && !TITLE_STOP_WORDS.has(w));
  const a = words(short);
  const b = words(full);
  return a.length > 0 && a.length === b.length && a.every((w, i) => b[i].startsWith(w));
}

/** Add the Chinese-core mark to a category ("GEOLOGY" -> "GEOLOGY (CN)"). */
export function withChineseMark(category) {
  const value = String(category || "").trim();
  if (isChineseCoreCategory(value)) return value;
  return value ? `${value} (CN)` : "(CN)";
}

/** A proposal from an OpenAlex journal (null when OpenAlex does not know it) and what the library says. */
function proposalOf(journal, { title, issns, libraryCount, origin }) {
  return {
    key: journal?.openalexId ? `openalex:${journal.openalexId}` : `issn:${issns[0]}`,
    title: journal?.title || title,
    // The venue name as written in the library (when it differs from OpenAlex's)
    libraryTitle: origin === "library" && journal && journal.title !== title ? title : null,
    issns,
    category: journal?.category || "",
    subfield: journal?.subfield || "",
    impactFactor: journal?.impactFactor ?? null,
    libraryCount,
    origin,
    verified: !!journal,
    openalexId: journal?.openalexId || null,
    publisher: journal?.publisher || null,
    country: journal?.country || null,
    worksCount: journal?.worksCount ?? null,
    // Internal: normalised names for matching the model's titles
    names: new Set([title, ...(journal ? [journal.title] : [])].map(venueKey).concat(journal?.titles || []).filter(Boolean)),
  };
}

/**
 * Journal proposals for the review panel.
 * @param {object} deps
 * @param {object} deps.http - transport (its signal cancels everything)
 * @param {object} deps.storage - for the Crossref title cache
 * @param {object[]} deps.items - library items ({itemType, venue, issn})
 * @param {string} deps.whitelist - the current CSV (proposals already listed are marked inList)
 * @param {object} deps.settings - see settings.js (journals, apis, sources.crossref.mailto)
 * @param {object|null} [deps.llm] - chat client; null skips the AI suggestions
 * @param {string} [deps.researchFocus] - research directions for the prompt
 * @returns {Promise<{proposals: object[], warnings: string[], missing: string[], stats: object}>}
 *   proposals: [{key, title, libraryTitle, issns, category, subfield, impactFactor, libraryCount,
 *   origin: "library"|"ai", verified, inList, checked, openalexId, publisher, country, worksCount}]
 *   (library journals by item count, then AI suggestions by impact factor); missing: library
 *   venues without any ISSN (they cannot be listed).
 */
export async function proposeJournals({ http, storage, items, whitelist = "", settings, llm = null, researchFocus = "", log = () => {}, now = () => Date.now(), retry = {} }) {
  const cfg = settings.journals;
  const apiKey = settings.apis?.openalexKey || "";
  const openalex = { apiKey, mailto: settings.apis?.mailto || "", retry };
  const warnings = [];
  const stats = { venues: 0, enriched: 0, aiSuggested: 0, aiAdded: 0, aiUnverified: 0, aiSkipped: 0 };
  let openalexError = null;
  const failed = (e) => {
    if (e?.cancelled) throw e;
    if (!openalexError) {
      openalexError = e;
      warnings.push(`OpenAlex 查询失败，部分期刊未能补全学科与影响因子：${openAlexErrorText(e, { keyed: !!apiKey })}`);
    }
  };
  /** Run OpenAlex lookups a few at a time; after the first failure (a 429, a bad key) the rest are skipped. */
  const eachLookup = async (list, fn) => {
    let next = 0;
    const worker = async () => {
      while (!openalexError && next < list.length) {
        const entry = list[next++];
        try {
          await fn(entry);
        } catch (e) {
          failed(e);
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(LOOKUP_CONCURRENCY, list.length) }, worker));
  };

  // 1. The library's venues, ISSNs from the items or Crossref
  const venues = libraryVenues(items).slice(0, cfg.autoTopN);
  stats.venues = venues.length;
  await resolveVenueIssns(http, storage, venues, { mailto: settings.sources?.crossref?.mailto || "", retry, now, log });

  // 2. OpenAlex: by ISSN in batches, then by exact title for venues without ISSNs
  const sourceOf = new Map();
  try {
    const byIssn = await lookupSourcesByIssn(http, venues.flatMap((v) => v.issns), openalex);
    for (const v of venues) {
      const journal = v.issns.map((issn) => byIssn.get(issn)).find(Boolean);
      if (journal) sourceOf.set(v, journal);
    }
  } catch (e) {
    failed(e);
  }
  const untitled = venues.filter((v) => !v.issns.length).slice(0, MAX_TITLE_SEARCHES);
  await eachLookup(untitled, async (v) => {
    const journal = await findSourceByTitle(http, v.title, openalex);
    if (journal) sourceOf.set(v, journal);
  });

  // 3. Library proposals (venues naming the same journal are merged)
  const proposals = [];
  const missing = [];
  const byIssn = new Map();
  const findListed = (issns) => issns.map((issn) => byIssn.get(issn)).find(Boolean) || null;
  const register = (p) => {
    proposals.push(p);
    for (const issn of p.issns) if (!byIssn.has(issn)) byIssn.set(issn, p);
  };
  for (const v of venues) {
    const journal = sourceOf.get(v) || null;
    const issns = [...new Set([...(journal?.issns || []), ...v.issns])];
    if (!issns.length) {
      missing.push(v);
      continue;
    }
    const same = findListed(issns);
    if (same) {
      same.libraryCount += v.count;
      same.names.add(venueKey(v.title));
      for (const issn of issns) {
        if (!same.issns.includes(issn)) same.issns.push(issn);
        if (!byIssn.has(issn)) byIssn.set(issn, same);
      }
      continue;
    }
    register(proposalOf(journal, { title: v.title, issns, libraryCount: v.count, origin: "library" }));
  }
  stats.enriched = proposals.filter((p) => p.verified).length;

  // 4. AI suggestions, each verified through OpenAlex
  if (llm && venues.length) {
    let suggestions = [];
    try {
      suggestions = await suggestJournalsWithLlm(
        llm,
        venues.slice(0, LLM_VENUES).map((v) => ({ title: v.title, count: v.count })),
        researchFocus,
      );
    } catch (e) {
      if (e?.cancelled) throw e;
      warnings.push(`大模型推荐失败：${e.message || e}`);
    }
    stats.aiSuggested = suggestions.length;
    const known = (title) => {
      const key = venueKey(title);
      return proposals.find((p) => p.names.has(key)) || null;
    };
    const toVerify = [];
    for (const s of suggestions) {
      const listed = known(s.title);
      // A normalised name of a library journal: only its Chinese-core flag is new
      if (listed) {
        if (s.isChinese) listed.category = withChineseMark(listed.category);
      } else {
        toVerify.push(s);
      }
    }
    const capped = toVerify.slice(0, MAX_AI_SUGGESTIONS);
    stats.aiSkipped = toVerify.length - capped.length;
    if (openalexError && capped.length) {
      stats.aiSkipped += capped.length;
      warnings.push("大模型推荐的期刊需要通过 OpenAlex 核实，本次已略过");
    }
    await eachLookup(capped, async (s) => {
      const journal = await findSourceByTitle(http, s.title, openalex);
      if (!journal || !journal.issns.length) {
        stats.aiUnverified++;
        return;
      }
      const same = findListed(journal.issns) || known(journal.title);
      if (same) {
        if (s.isChinese) same.category = withChineseMark(same.category);
        return;
      }
      // The model's full name of a library venue OpenAlex could not match by its (abbreviated) name
      const venue = missing.find((v) => abbreviates(v.title, journal.title) || journal.titles.includes(venueKey(v.title)));
      if (venue) {
        missing.splice(missing.indexOf(venue), 1);
        const p = proposalOf(journal, { title: venue.title, issns: journal.issns, libraryCount: venue.count, origin: "library" });
        if (s.isChinese) p.category = withChineseMark(p.category);
        register(p);
        stats.enriched++;
        return;
      }
      const p = proposalOf(journal, { title: journal.title, issns: journal.issns, libraryCount: 0, origin: "ai" });
      if (!p.category && s.category) p.category = s.category;
      if (s.isChinese) p.category = withChineseMark(p.category);
      register(p);
      stats.aiAdded++;
    });
  }

  // 5. Already listed and default selection
  const listed = new Set(parseWhitelist(whitelist).keys());
  const byCount = (a, b) => b.libraryCount - a.libraryCount || a.title.localeCompare(b.title);
  const byImpact = (a, b) => (b.impactFactor ?? -1) - (a.impactFactor ?? -1) || a.title.localeCompare(b.title);
  const ordered = [...proposals.filter((p) => p.origin === "library").sort(byCount), ...proposals.filter((p) => p.origin === "ai").sort(byImpact)];
  return {
    proposals: ordered.map(({ names, ...p }) => {
      const inList = p.issns.some((issn) => listed.has(issn));
      const checked = !inList && (p.origin === "library" ? p.libraryCount >= cfg.autoMinItems : (p.impactFactor ?? 0) >= AI_DEFAULT_MIN_IF);
      return { ...p, inList, checked };
    }),
    warnings,
    missing: missing.map((v) => v.title),
    stats,
  };
}

/**
 * 「补全学科与影响因子」: fill the blank (or `library:N`) categories, missing
 * impact factors and blank titles of the listed journals from OpenAlex; never
 * overwrites a value. Throws when OpenAlex cannot be reached (nothing else to do then).
 * @returns {Promise<{text, filled: number, incomplete: number, notFound: string[]}>}
 */
export async function fillJournalMetadata({ http, text, settings, retry = {} }) {
  const targets = [...parseWhitelist(text).values()].filter(isIncompleteJournal).map((j) => j.issn);
  if (!targets.length) return { text: String(text ?? ""), filled: 0, incomplete: 0, notFound: [] };
  const found = await lookupSourcesByIssn(http, targets, { apiKey: settings.apis?.openalexKey || "", mailto: settings.apis?.mailto || "", retry });
  const rows = targets
    .filter((issn) => found.has(issn))
    .map((issn) => {
      const journal = found.get(issn);
      return { issn, title: journal.title, category: journal.category, impactFactor: journal.impactFactor };
    });
  const merged = mergeWhitelistRows(text, rows, { append: false });
  return { text: merged.text, filled: merged.filled, incomplete: targets.length, notFound: targets.filter((issn) => !found.has(issn)) };
}
