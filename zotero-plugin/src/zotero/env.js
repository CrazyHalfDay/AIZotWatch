// Adapters that bind the engine's injectable dependencies to Zotero APIs.
// Only this directory touches Zotero globals (Zotero, IOUtils, PathUtils).

/* global Zotero, IOUtils, PathUtils */

import { extractDoi, normalizeDoi } from "../core/text.js";
import { PREF_PREFIX, loadSettings } from "../settings.js";

/** HTTP transport over Zotero.HTTP.request (honours Zotero's proxy settings). */
export const zoteroTransport = {
  async request({ method = "GET", url, headers = {}, body, responseType = "json", timeout = 60000 }) {
    const xhr = await Zotero.HTTP.request(method, url, {
      headers,
      body,
      responseType: responseType === "json" ? "json" : "text",
      timeout,
      successCodes: false, // status handling and retries live in core/http.js
      errorDelayMax: 0,
      logBodyLength: 0,
    });
    if (!xhr.status) throw new Error(`网络连接失败：${url.split("?")[0]}`);
    return { status: xhr.status, data: xhr.response };
  },
};

/** File storage in <Zotero data directory>/zotwatch/. */
export function createZoteroStorage() {
  const dir = PathUtils.join(Zotero.DataDirectory.dir, "zotwatch");
  let ready = null;
  const ensureDir = () => (ready ||= IOUtils.makeDirectory(dir, { ignoreExisting: true, createAncestors: true }));
  const path = (name) => PathUtils.join(dir, name);

  return {
    dir,
    async readJSON(name) {
      await ensureDir();
      return (await IOUtils.exists(path(name))) ? IOUtils.readJSON(path(name)) : null;
    },
    async writeJSON(name, obj) {
      await ensureDir();
      await IOUtils.writeJSON(path(name), obj, { tmpPath: `${path(name)}.tmp` });
    },
    async readBinary(name) {
      await ensureDir();
      return (await IOUtils.exists(path(name))) ? IOUtils.read(path(name)) : null;
    },
    async writeBinary(name, bytes) {
      await ensureDir();
      await IOUtils.write(path(name), bytes, { tmpPath: `${path(name)}.tmp` });
    },
    async clear() {
      await IOUtils.remove(dir, { recursive: true, ignoreAbsent: true });
      ready = null;
    },
  };
}

export function getPref(key) {
  return Zotero.Prefs.get(PREF_PREFIX + key, true);
}

export function setPref(key, value) {
  Zotero.Prefs.set(PREF_PREFIX + key, value, true);
}

export function readSettings() {
  return loadSettings(getPref);
}

export function profileLibraryID() {
  const raw = parseInt(getPref("profile.libraryID"), 10);
  if (Number.isFinite(raw) && Zotero.Libraries.get(raw)) return raw;
  return Zotero.Libraries.userLibraryID;
}

function sqlDateToIso(value) {
  return value ? `${String(value).replace(" ", "T")}Z` : null;
}

function itemToPlain(item) {
  const field = (name, baseMapped = false) => {
    const v = item.getField(name, true, baseMapped);
    return typeof v === "string" ? v : "";
  };
  return {
    key: item.key,
    itemType: item.itemType,
    title: field("title", true),
    abstract: field("abstractNote"),
    doi: normalizeDoi(field("DOI")) || extractDoi(field("extra")),
    issn: field("ISSN"),
    venue: field("publicationTitle", true),
    dateAdded: sqlDateToIso(item.dateAdded),
  };
}

/** Library adapter: plain metadata of all regular (non-note, non-attachment) items. */
export function createZoteroLibrary() {
  return {
    async getItems() {
      const items = (await Zotero.Items.getAll(profileLibraryID(), true, false)).filter((i) => i.isRegularItem());
      try {
        return items.map(itemToPlain);
      } catch (e) {
        // Item data is normally preloaded; load it explicitly if it isn't
        await Zotero.Items.loadDataTypes(items);
        return items.map(itemToPlain);
      }
    },
  };
}

export function log(message) {
  Zotero.debug(`[ZotWatch] ${message}`);
}
