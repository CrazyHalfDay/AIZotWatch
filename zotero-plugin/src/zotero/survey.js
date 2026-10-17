// Zotero adapters of the topic survey: library search and selection for the
// seed picker, seed metadata and full text, and saving a survey as a note
// (with its papers) in a collection. Only this file and its siblings in
// src/zotero/ touch Zotero globals.

/* global Zotero, IOUtils */

import { collapseWhitespace, extraFieldDoi, normalizeDoi } from "../core/text.js";
import { reportCitations, surveyNoteHtml } from "../survey/report.js";
import { addWorkToZotero, getOrCreateCollection } from "./items.js";
import { profileLibraryID } from "./env.js";

/** Attachment types with indexed full text (PDF, EPUB, HTML snapshots), in order of preference. */
const FULL_TEXT_TYPES = ["application/pdf", "application/epub+zip", "text/html"];
export const SURVEY_COLLECTION_PREFIX = "ZotWatch 调研 · ";
const MAX_COLLECTION_TOPIC = 60;

const text = (value) => (typeof value === "string" ? value : "");

function field(item, name, baseMapped = false) {
  try {
    return text(item.getField(name, true, baseMapped));
  } catch {
    return "";
  }
}

/** Display names of an item's creators ("First Last", single-field names as they are). */
function creatorNames(item) {
  let creators = [];
  try {
    creators = typeof item.getCreators === "function" ? item.getCreators() : [];
  } catch {
    creators = [];
  }
  return (Array.isArray(creators) ? creators : [])
    .map((c) => collapseWhitespace(c?.fieldMode === 1 ? text(c.lastName) : `${text(c?.firstName)} ${text(c?.lastName)}`))
    .filter(Boolean);
}

/**
 * Plain metadata of a regular item for the seed picker and the survey:
 * {key, libraryID, itemType, title, year, creators, doi, venue, abstract, tags}.
 */
export function itemSummary(item) {
  const date = field(item, "date");
  const year = Number((date.match(/\b(1[5-9]\d\d|2\d\d\d)\b/) || [])[1]) || null;
  let tags = [];
  try {
    tags = typeof item.getTags === "function" ? item.getTags().map((t) => t?.tag).filter((t) => typeof t === "string" && t) : [];
  } catch {
    tags = [];
  }
  return {
    key: item.key,
    libraryID: item.libraryID ?? null,
    itemType: item.itemType || null,
    title: collapseWhitespace(field(item, "title", true)),
    year,
    creators: creatorNames(item),
    doi: normalizeDoi(field(item, "DOI")) || extraFieldDoi(field(item, "extra")) || null,
    venue: collapseWhitespace(field(item, "publicationTitle", true)) || null,
    abstract: field(item, "abstractNote"),
    tags,
  };
}

async function regularItems(libraryID) {
  await Zotero.Libraries.get(libraryID)?.waitForDataLoad?.("item");
  return (await Zotero.Items.getAll(libraryID, true, false)).filter((i) => i.isRegularItem());
}

/** An item of a library by key (null when missing or deleted). */
export async function itemByKey(key, libraryID = profileLibraryID()) {
  if (typeof key !== "string" || !key) return null;
  let item = null;
  try {
    item =
      typeof Zotero.Items.getByLibraryAndKeyAsync === "function"
        ? await Zotero.Items.getByLibraryAndKeyAsync(libraryID, key)
        : Zotero.Items.getByLibraryAndKey(libraryID, key);
  } catch {
    item = null;
  }
  return item && !item.deleted ? item : null;
}

/**
 * Regular items of the profile library matching a query: every word must
 * appear in the title, a creator name or the year; a DOI matches exactly.
 * Best matches first (title hits, then newer), at most `limit`.
 * @returns {Promise<{key, title, year, creators, doi, itemType}[]>}
 */
export async function searchLibrary(query, { limit = 20 } = {}) {
  const q = collapseWhitespace(String(query ?? "")).toLowerCase();
  if (!q) return [];
  const max = Math.max(1, Math.min(100, Math.floor(Number(limit)) || 20));
  const doi = /10\.\d{4,9}\//.test(q) ? normalizeDoi(q) : "";
  const words = q.split(" ").filter(Boolean);
  const hits = [];
  for (const item of await regularItems(profileLibraryID())) {
    if (doi) {
      const itemDoi = normalizeDoi(field(item, "DOI")) || extraFieldDoi(field(item, "extra"));
      if (itemDoi === doi) hits.push({ item, score: 100 });
      continue;
    }
    const title = field(item, "title", true).toLowerCase();
    const creators = creatorNames(item).join(" ").toLowerCase();
    const year = (field(item, "date").match(/\b\d{4}\b/) || [""])[0];
    let score = title.includes(q) ? 10 : 0;
    let all = true;
    for (const w of words) {
      if (title.includes(w)) score += 2;
      else if (creators.includes(w)) score += 1;
      else if (year && year === w) score += 1;
      else {
        all = false;
        break;
      }
    }
    if (all) hits.push({ item, score, year: Number(year) || 0 });
  }
  hits.sort((a, b) => b.score - a.score || (b.year || 0) - (a.year || 0));
  return hits.slice(0, max).map(({ item }) => {
    const s = itemSummary(item);
    return { key: s.key, title: s.title, year: s.year, creators: s.creators, doi: s.doi, itemType: s.itemType };
  });
}

/**
 * Regular items selected in Zotero's main window, from the profile library
 * (a selected attachment or note stands for its parent item); [] without a main window.
 */
export function getSelectedItems() {
  let selected = [];
  try {
    const pane = typeof Zotero.getActiveZoteroPane === "function" ? Zotero.getActiveZoteroPane() : Zotero.getMainWindow()?.ZoteroPane;
    selected = pane?.getSelectedItems?.() || [];
  } catch {
    selected = [];
  }
  const libraryID = profileLibraryID();
  const out = new Map();
  for (let item of selected) {
    if (item && !item.isRegularItem?.() && item.parentItemID) item = Zotero.Items.get(item.parentItemID);
    if (!item || !item.isRegularItem?.() || item.libraryID !== libraryID || out.has(item.key)) continue;
    out.set(item.key, itemSummary(item));
  }
  return [...out.values()];
}

/** Seed metadata (see itemSummary) of the profile library's items with these keys, in key order. */
export async function getItemsByKeys(keys) {
  const out = [];
  for (const key of keys || []) {
    const item = await itemByKey(key);
    if (item?.isRegularItem?.()) out.push(itemSummary(item));
  }
  return out;
}

/**
 * Text of an item's best full-text attachment (PDF, then EPUB, then HTML),
 * at most `maxChars`; "" when there is none. Uses `attachment.attachmentText`
 * (Zotero 7+: the indexed text); its absence or failure is not an error.
 */
export async function getItemFullText(key, { maxChars = 20000 } = {}) {
  const item = await itemByKey(key);
  if (!item) return "";
  let attachments = [];
  try {
    if (item.isAttachment?.()) attachments = [item];
    else if (typeof item.getBestAttachments === "function") attachments = await item.getBestAttachments();
    else if (typeof item.getAttachments === "function") attachments = Zotero.Items.get(item.getAttachments()) || [];
  } catch {
    attachments = [];
  }
  const ranked = (attachments || [])
    .filter((a) => a && FULL_TEXT_TYPES.includes(a.attachmentContentType) && "attachmentText" in a)
    .sort((a, b) => FULL_TEXT_TYPES.indexOf(a.attachmentContentType) - FULL_TEXT_TYPES.indexOf(b.attachmentContentType));
  for (const attachment of ranked) {
    let value = "";
    try {
      value = await attachment.attachmentText;
    } catch (e) {
      Zotero.debug?.(`[ZotWatch] Full text of ${attachment.key} unavailable: ${e}`);
      continue;
    }
    if (typeof value === "string" && value.trim()) return value.trim().slice(0, Math.max(0, maxChars));
  }
  return "";
}

/** Collection name of a survey: 「ZotWatch 调研 · <topic>」. */
export function surveyCollectionName(survey) {
  const topic = Array.from(collapseWhitespace(survey.input?.topic || "")).slice(0, MAX_COLLECTION_TOPIC).join("");
  return `${SURVEY_COLLECTION_PREFIX}${topic || survey.id}`;
}

function inCollection(item, collectionID) {
  try {
    if (typeof item.inCollection === "function") return item.inCollection(collectionID);
    return (item.getCollections?.() || []).includes(collectionID);
  } catch {
    return false;
  }
}

/** The work shape of items.js addWorkToZotero for a survey paper (or reference). */
export function workOfPaper(p) {
  return {
    title: p.title,
    abstract: p.abstract || null,
    doi: p.doi || null,
    url: p.url || null,
    published: p.published || (Number.isInteger(p.year) ? `${p.year}-01-01T00:00:00Z` : null),
    venue: p.venue || null,
    issns: p.issns || [],
    type: p.type || null,
    source: Array.isArray(p.sources) && p.sources.includes("openalex") ? "openalex" : "survey",
    arxivId: p.arxivId || null,
    authors: (p.authors || []).map((a) => (typeof a === "string" ? { given: "", family: a, name: a } : a)).filter((a) => a && a.name),
  };
}

/**
 * The references `addPapers` names: "read" (every reference), "cited" (the
 * ones the report cites; every listed one for a candidate list) or "none".
 */
export function papersToAdd(survey, addPapers) {
  if (addPapers === "none") return [];
  const refs = survey.references || [];
  if (addPapers === "read" || survey.report?.kind !== "review") return refs;
  const cited = new Set(reportCitations(survey.report));
  return refs.filter((r) => cited.has(r.n));
}

/**
 * Save a survey in Zotero: a collection 「ZotWatch 调研 · <topic>」 (created or
 * reused) holding a standalone note with the report (escaped HTML; an earlier
 * note of this survey is updated instead) and, with `addPapers`, its papers:
 * library items are added to the collection, missing ones imported through
 * addWorkToZotero (DOI translator first), one after another. A paper that
 * fails is counted, not fatal. Mutates the survey's references and pool
 * (inLibrary, itemKey) and `survey.saved`; the caller persists it.
 * @returns {Promise<{collectionName, collectionID, noteID, noteKey, total, linked, imported, failed, skipped, cancelled}>}
 */
export async function saveSurveyToZotero(
  survey,
  { libraryID, collectionName = surveyCollectionName(survey), addPapers = "cited", tag = "", useTranslator = true, onProgress = () => {}, isCancelled = () => false, log = () => {} },
) {
  const collection = await getOrCreateCollection(libraryID, collectionName);
  let note = survey.saved?.noteKey && survey.saved.libraryID === libraryID ? await itemByKey(survey.saved.noteKey, libraryID) : null;
  if (note && !(typeof note.isNote === "function" ? note.isNote() : note.itemType === "note")) note = null;
  if (!note) {
    note = new Zotero.Item("note");
    note.libraryID = libraryID;
  }
  note.setNote(surveyNoteHtml(survey));
  if (!inCollection(note, collection.id)) note.addToCollection(collection.id);
  await note.saveTx();

  const refs = papersToAdd(survey, addPapers);
  const byId = new Map((survey.pool || []).map((p) => [p.id, p]));
  const sameLibrary = libraryID === profileLibraryID();
  const result = { collectionName, collectionID: collection.id, noteID: note.id, noteKey: note.key, total: refs.length, linked: 0, imported: 0, failed: 0, skipped: 0, cancelled: false };
  for (const [i, ref] of refs.entries()) {
    if (isCancelled()) {
      result.cancelled = true;
      break;
    }
    onProgress({ current: i, total: refs.length, title: ref.title });
    const paper = byId.get(ref.id);
    try {
      if (ref.inLibrary && ref.itemKey) {
        if (!sameLibrary) {
          // The library item lives in another library than the collection
          result.skipped++;
          continue;
        }
        const item = await itemByKey(ref.itemKey, libraryID);
        if (item) {
          if (!inCollection(item, collection.id)) {
            item.addToCollection(collection.id);
            await item.saveTx();
          }
          result.linked++;
          continue;
        }
      }
      const item = await addWorkToZotero(workOfPaper(paper || ref), { libraryID, collectionName, tag, withNote: false, useTranslator, log });
      for (const target of [ref, paper]) {
        if (!target) continue;
        target.inLibrary = true;
        target.itemKey = item.key;
      }
      result.imported++;
    } catch (e) {
      result.failed++;
      log(`Survey paper import failed (${ref.title}): ${e?.message || e}`);
    }
  }
  onProgress({ current: refs.length, total: refs.length, title: null });
  survey.saved = { at: Date.now(), libraryID, noteKey: note.key, collectionName, collectionID: collection.id };
  return result;
}

/** Write a UTF-8 text file (the Markdown export). */
export async function writeTextFile(path, content) {
  await IOUtils.writeUTF8(path, String(content ?? ""));
}
