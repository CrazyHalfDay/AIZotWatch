// Persistence of topic surveys in the plugin storage (<data dir>/zotwatch/):
//   surveys.json          index [{id, topic, createdAt, updatedAt, status, depth, counts, stage, error}],
//                         newest first, at most SURVEY_LIMIT entries
//   survey-<id>.json      one survey: input, stage summaries, the candidate pool,
//                         themes, references and the report
//   survey-cache.json     reading notes shared by all surveys (by DOI or title, and model)
// All of it is a cache of external data and AI output: resetData() removes it.

import { hashString, normalizeDoi, normalizeTitle } from "../core/text.js";
import { EXTRACTION_FIELDS } from "./prompts.js";

export const SURVEY_INDEX_FILE = "surveys.json";
export const SURVEY_CACHE_FILE = "survey-cache.json";
export const SURVEY_LIMIT = 30;
export const SURVEY_VERSION = 1;
const CACHE_LIMIT = 3000;
export const STATUSES = new Set(["running", "done", "cancelled", "failed"]);

/** File of one survey (ids are generated here: lower-case letters and digits). */
export function surveyFile(id) {
  if (!isSurveyId(id)) throw new Error("无效的调研 ID");
  return `survey-${id}.json`;
}

export function isSurveyId(id) {
  return typeof id === "string" && /^[a-z0-9]{6,40}$/.test(id);
}

let idCounter = 0;
/** A new survey id: time in base 36 plus a counter and random digits. */
export function newSurveyId(now = Date.now()) {
  idCounter = (idCounter + 1) % 1296;
  const random = Math.floor(Math.random() * 1296).toString(36).padStart(2, "0");
  return `${Math.floor(now).toString(36)}${idCounter.toString(36).padStart(2, "0")}${random}`;
}

// One write queue per storage backend: stage saves, deletes and imports may overlap
const locks = new WeakMap();

/** Run `task` after every earlier locked task of this storage. */
export async function withSurveyLock(storage, task) {
  const previous = locks.get(storage) || Promise.resolve();
  let release;
  const current = new Promise((resolve) => (release = resolve));
  locks.set(storage, previous.then(() => current));
  await previous;
  try {
    return await task();
  } finally {
    release();
  }
}

const isRecord = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const count = (v) => (Number.isFinite(v) && v >= 0 ? v : 0);

/** Valid index entries of stored data (a damaged file yields []). */
export function validIndex(data) {
  const list = Array.isArray(data) ? data : Array.isArray(data?.surveys) ? data.surveys : [];
  return list.filter((e) => isRecord(e) && isSurveyId(e.id) && typeof e.topic === "string" && Number.isFinite(e.createdAt));
}

/** Index entry of a survey. */
export function indexEntry(survey) {
  const c = survey.counts || {};
  return {
    id: survey.id,
    topic: survey.input?.topic || "",
    createdAt: survey.createdAt,
    updatedAt: survey.updatedAt,
    status: STATUSES.has(survey.status) ? survey.status : "failed",
    depth: survey.input?.depth || "standard",
    stage: survey.stage || null,
    counts: { seeds: count(c.seeds), candidates: count(c.candidates), read: count(c.read), references: count(c.references) },
    error: survey.error?.message || null,
    hasReport: !!survey.report,
    llm: survey.llm !== false,
  };
}

export async function readIndex(storage) {
  let data = null;
  try {
    data = await storage.readJSON(SURVEY_INDEX_FILE);
  } catch {
    data = null;
  }
  return validIndex(data);
}

async function removeFile(storage, name) {
  try {
    if (typeof storage.remove === "function") await storage.remove(name);
    else await storage.writeJSON(name, null);
  } catch {
    // Best effort: an orphaned file is only wasted space
  }
}

/**
 * Write a survey file and its index entry (newest first, capped: the files of
 * surveys falling off the index are removed). Returns the ids removed.
 */
export async function saveSurvey(storage, survey) {
  return withSurveyLock(storage, () => writeSurvey(storage, survey));
}

/** saveSurvey's work; the caller holds the lock. */
async function writeSurvey(storage, survey) {
  await storage.writeJSON(surveyFile(survey.id), survey);
  const index = (await readIndex(storage)).filter((e) => e.id !== survey.id);
  index.sort((a, b) => b.createdAt - a.createdAt);
  // The survey being saved always stays; the oldest others make room
  const dropped = index.splice(SURVEY_LIMIT - 1);
  index.push(indexEntry(survey));
  index.sort((a, b) => b.createdAt - a.createdAt);
  await storage.writeJSON(SURVEY_INDEX_FILE, index);
  for (const e of dropped) await removeFile(storage, surveyFile(e.id));
  return dropped.map((e) => e.id);
}

/**
 * Change a stored survey in place: load it, apply `mutate(survey)` and write
 * it, all under the write lock, so that a copy loaded long before (e.g. at the
 * start of a save to Zotero) never overwrites newer stages. Resolves with the
 * written survey, or null (nothing written) when it no longer exists.
 */
export async function updateSurvey(storage, id, mutate) {
  return withSurveyLock(storage, async () => {
    const survey = await loadSurvey(storage, id);
    if (!survey) return null;
    mutate(survey);
    await writeSurvey(storage, survey);
    return survey;
  });
}

/**
 * Copy what saving papers to Zotero changes from `source` (a working copy)
 * into `target` (the stored survey): `saved` and the library flags of the
 * papers and references (inLibrary, itemKey, itemLibraryID), matched by id.
 */
export function mergeLibraryState(target, source) {
  if (isRecord(source.saved)) target.saved = source.saved;
  for (const key of ["pool", "references"]) {
    const byId = new Map((Array.isArray(source[key]) ? source[key] : []).map((p) => [p?.id, p]));
    for (const p of Array.isArray(target[key]) ? target[key] : []) {
      const from = byId.get(p.id);
      if (!from?.inLibrary || !from.itemKey) continue;
      p.inLibrary = true;
      p.itemKey = from.itemKey;
      if (from.itemLibraryID !== undefined) p.itemLibraryID = from.itemLibraryID;
    }
  }
  target.updatedAt = Math.max(Number(target.updatedAt) || 0, Number(source.updatedAt) || 0) || Date.now();
  return target;
}

/** A stored survey, or null when missing or damaged. */
export async function loadSurvey(storage, id) {
  if (!isSurveyId(id)) return null;
  let data = null;
  try {
    data = await storage.readJSON(surveyFile(id));
  } catch {
    return null;
  }
  if (!isRecord(data) || data.id !== id || !isRecord(data.input)) return null;
  data.pool = Array.isArray(data.pool) ? data.pool.filter(isRecord) : [];
  data.references = Array.isArray(data.references) ? data.references.filter(isRecord) : [];
  data.themes = Array.isArray(data.themes) ? data.themes.filter(isRecord) : [];
  data.stages = isRecord(data.stages) ? data.stages : {};
  data.completed = Array.isArray(data.completed) ? data.completed.filter((s) => typeof s === "string") : [];
  data.errors = Array.isArray(data.errors) ? data.errors.filter(isRecord) : [];
  return data;
}

/** Remove a survey's file and index entry; true when it was listed. */
export async function deleteSurvey(storage, id) {
  if (!isSurveyId(id)) return false;
  return withSurveyLock(storage, async () => {
    const index = await readIndex(storage);
    const kept = index.filter((e) => e.id !== id);
    await removeFile(storage, surveyFile(id));
    if (kept.length !== index.length) await storage.writeJSON(SURVEY_INDEX_FILE, kept);
    return kept.length !== index.length;
  });
}

// ---------------------------------------------------------- reading cache

/**
 * Cache key of a paper's reading note: its DOI (else its normalised title)
 * and the model. The note's relevance depends on the survey topic, so it is
 * stored per topic (see readCachedNote).
 */
export function noteCacheKey(paper, model) {
  const id = normalizeDoi(paper.doi) || `t:${normalizeTitle(paper.title)}`;
  return hashString(`${id}|${model || ""}`);
}

export const topicKey = (topic) => hashString(String(topic || "").trim().toLowerCase());

export async function loadNoteCache(storage) {
  let data = null;
  try {
    data = await storage.readJSON(SURVEY_CACHE_FILE);
  } catch {
    data = null;
  }
  const entries = isRecord(data?.entries) && data.version === 1 ? data.entries : {};
  const map = new Map();
  for (const [key, entry] of Object.entries(entries)) {
    if (isRecord(entry) && isRecord(entry.e) && Number.isFinite(entry.at)) map.set(key, entry);
  }
  return { entries: map, dirty: false };
}

/** The cached note of a paper for a topic: {note, relevanceKnown} or null. */
export function readCachedNote(cache, key, topic) {
  const entry = cache.entries.get(key);
  if (!entry) return null;
  const note = {};
  for (const [field] of EXTRACTION_FIELDS) note[field] = typeof entry.e[field] === "string" ? entry.e[field] : "";
  const r = isRecord(entry.r) ? entry.r[topicKey(topic)] : undefined;
  note.relevance = Number.isInteger(r) && r >= 1 && r <= 5 ? r : null;
  return { note, relevanceKnown: note.relevance !== null };
}

export function writeCachedNote(cache, key, topic, note, now = Date.now()) {
  const previous = cache.entries.get(key);
  const e = {};
  for (const [field] of EXTRACTION_FIELDS) e[field] = note[field] || "";
  const r = { ...(isRecord(previous?.r) ? previous.r : {}) };
  if (Number.isInteger(note.relevance)) r[topicKey(topic)] = note.relevance;
  cache.entries.set(key, { e, r, at: now });
  cache.dirty = true;
}

/** Write the cache (newest CACHE_LIMIT entries). Never throws. */
export async function saveNoteCache(storage, cache, log = () => {}) {
  if (!cache.dirty) return;
  const entries = [...cache.entries].sort((a, b) => b[1].at - a[1].at).slice(0, CACHE_LIMIT);
  try {
    await storage.writeJSON(SURVEY_CACHE_FILE, { version: 1, entries: Object.fromEntries(entries) });
    cache.dirty = false;
  } catch (e) {
    log(`Survey cache save failed: ${e?.message || e}`);
  }
}
