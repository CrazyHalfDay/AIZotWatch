// Saving recommendations as Zotero items.

/* global Zotero */

import { toZoteroCreator } from "../core/authors.js";
import { SUMMARY_FIELDS } from "../llm/llm.js";

const PREPRINT_SOURCES = new Set(["arxiv", "eartharxiv"]);

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function summaryNoteHtml(work) {
  const rows = SUMMARY_FIELDS.filter(([key]) => work.summary?.[key])
    .map(([key, label]) => `<p><strong>${label}：</strong>${escapeHtml(work.summary[key])}</p>`)
    .join("\n");
  const score = typeof work.score === "number" ? `<p>推荐分数：${(work.score * 100).toFixed(0)}（相似度 ${(work.similarity * 100).toFixed(0)}）</p>` : "";
  return `<h2>ZotWatch AI 摘要</h2>\n${rows}\n${score}`;
}

// In-flight lookups, so two quick "add" clicks never create two collections
const collectionLocks = new Map();

export function getOrCreateCollection(libraryID, name) {
  const lockKey = `${libraryID}\u0000${name}`;
  if (!collectionLocks.has(lockKey)) {
    const pending = (async () => {
      // Recursive: also find the collection if the user moved it under another one
      const existing = Zotero.Collections.getByLibrary(libraryID, true).find((c) => c.name === name);
      if (existing) return existing;
      const collection = new Zotero.Collection({ name, libraryID });
      await collection.saveTx();
      return collection;
    })();
    collectionLocks.set(lockKey, pending);
    pending.finally(() => collectionLocks.delete(lockKey)).catch(() => {});
  }
  return collectionLocks.get(lockKey);
}

/** Import full metadata through Zotero's own DOI / arXiv lookup ("Add Item by Identifier"). */
async function importByIdentifier(work, libraryID, collection, log) {
  const identifier = work.doi ? { DOI: work.doi } : work.arxivId ? { arXiv: work.arxivId } : null;
  if (!identifier) return null;
  try {
    const translate = new Zotero.Translate.Search();
    translate.setIdentifier(identifier);
    const translators = await translate.getTranslators();
    if (!translators || !translators.length) return null;
    translate.setTranslator(translators);
    const items = await translate.translate({
      libraryID,
      collections: collection ? [collection.id] : false,
      saveAttachments: false,
    });
    return items && items.length ? items[0] : null;
  } catch (e) {
    log(`Identifier lookup failed for ${JSON.stringify(identifier)}: ${e}`);
    return null;
  }
}

function createItemFromWork(work, libraryID) {
  const preprint = PREPRINT_SOURCES.has(work.source) && Zotero.ItemTypes.getID("preprint");
  const item = new Zotero.Item(preprint ? "preprint" : "journalArticle");
  item.libraryID = libraryID;
  const set = (field, value) => {
    if (!value) return;
    try {
      item.setField(field, value);
    } catch (e) {
      // Field not valid for this item type
    }
  };
  set("title", work.title);
  set("abstractNote", work.abstract);
  set("DOI", work.doi);
  set("url", work.url);
  set("date", work.published ? work.published.slice(0, 10) : "");
  if (preprint) {
    set("repository", work.venue);
    if (work.arxivId) set("archiveID", `arXiv:${work.arxivId}`);
  } else {
    set("publicationTitle", work.venue);
    set("ISSN", (work.issns || []).join(" "));
  }
  item.setCreators((work.authors || []).map((a) => toZoteroCreator(a)));
  return item;
}

/**
 * Save a recommendation to the library.
 * @returns {Promise<Zotero.Item>}
 */
export async function addWorkToZotero(work, { libraryID, collectionName, tag, withNote, useTranslator, log = () => {} }) {
  const collection = collectionName ? await getOrCreateCollection(libraryID, collectionName) : null;

  let item = useTranslator ? await importByIdentifier(work, libraryID, collection, log) : null;
  if (item) {
    // Translators sometimes omit the abstract; keep the one we already have
    if (!item.getField("abstractNote") && work.abstract) item.setField("abstractNote", work.abstract);
  } else {
    item = createItemFromWork(work, libraryID);
    if (collection) item.addToCollection(collection.id);
  }
  if (tag) item.addTag(tag);
  await item.saveTx();
  // Newer Zotero versions list recently used collections under "Add to Collection"
  if (collection && typeof Zotero.Collections.addToRecent === "function") {
    Zotero.Collections.addToRecent(collection);
  }

  if (withNote && work.summary) {
    const note = new Zotero.Item("note");
    note.libraryID = libraryID;
    note.parentID = item.id;
    note.setNote(summaryNoteHtml(work));
    await note.saveTx();
  }
  return item;
}
