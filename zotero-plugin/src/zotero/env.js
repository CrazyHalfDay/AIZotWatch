// Adapters that bind the engine's injectable dependencies to Zotero APIs.
// Only this directory touches Zotero globals (Zotero, IOUtils, PathUtils).

/* global Zotero, IOUtils, PathUtils */

import { CancelledRequestError } from "../core/http.js";
import { extraFieldDoi, normalizeDoi } from "../core/text.js";
import { PREF_PREFIX, loadSettings } from "../settings.js";

/**
 * HTTP transport over Zotero.HTTP.request (honours Zotero's proxy settings).
 * With a `signal` (see createCancelSignal), cancelling aborts in-flight
 * requests and makes later ones fail at once; core/http.js also ends retry
 * waits early through `transport.signal`.
 */
export function createZoteroTransport({ signal = null } = {}) {
  return {
    signal,
    async request({ method = "GET", url, headers = {}, body, responseType = "json", timeout = 60000 }) {
      signal?.throwIfCancelled();
      const unregisters = [];
      let xhr;
      try {
        xhr = await Zotero.HTTP.request(method, url, {
          headers,
          body,
          responseType: responseType === "json" ? "json" : "text",
          timeout,
          successCodes: false, // status handling and retries live in core/http.js
          errorDelayMax: 0,
          logBodyLength: 0,
          anon: true, // never send Zotero's cookies to third-party APIs (Zotero 9+; ignored before)
          cancellerReceiver: (cancel) => {
            if (signal) unregisters.push(signal.onCancel(cancel));
          },
        });
      } catch (e) {
        if (e instanceof Zotero.HTTP.CancelledException) throw new CancelledRequestError();
        throw e;
      } finally {
        for (const unregister of unregisters) unregister();
      }
      if (!xhr.status) {
        // Status 0: let Zotero explain certificate/proxy problems, otherwise report the channel error
        try {
          Zotero.HTTP.checkSecurity(xhr.channel);
        } catch (e) {
          if (e instanceof Zotero.HTTP.SecurityException) throw e;
        }
        const code = xhr.channel?.status;
        const host = url.replace(/^https?:\/\/([^/?#]+).*$/, "$1");
        throw new Error(`网络连接失败：${host}${code ? `（0x${(code >>> 0).toString(16)}）` : ""}`);
      }
      return { status: xhr.status, data: xhr.response, retryAfter: xhr.getResponseHeader("Retry-After") };
    },
  };
}

/** File storage in <Zotero data directory>/zotwatch/. */
export function createZoteroStorage() {
  const dir = PathUtils.join(Zotero.DataDirectory.dir, "zotwatch");
  let ready = null;
  const ensureDir = () => (ready ||= IOUtils.makeDirectory(dir, { ignoreExisting: true, createAncestors: true }));
  const path = (name) => PathUtils.join(dir, name);

  return {
    dir,
    /** Create the directory if needed (e.g. before revealing it). */
    async ensureDir() {
      await ensureDir();
      return dir;
    },
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
    async readText(name) {
      await ensureDir();
      return (await IOUtils.exists(path(name))) ? IOUtils.readUTF8(path(name)) : null;
    },
    async writeText(name, text) {
      await ensureDir();
      await IOUtils.writeUTF8(path(name), text, { tmpPath: `${path(name)}.tmp` });
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

/** Library the profile is built from (hidden pref; feeds are never valid). */
export function profileLibraryID() {
  const raw = parseInt(getPref("profile.libraryID"), 10);
  const library = Number.isFinite(raw) ? Zotero.Libraries.get(raw) : null;
  if (library && library.libraryType !== "feed") return raw;
  return Zotero.Libraries.userLibraryID;
}

/** Library new items are saved to: the profile library if editable, else My Library. */
export function targetLibraryID() {
  const id = profileLibraryID();
  return Zotero.Libraries.get(id)?.editable ? id : Zotero.Libraries.userLibraryID;
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
    doi: normalizeDoi(field("DOI")) || extraFieldDoi(field("extra")),
    issn: field("ISSN"),
    venue: field("publicationTitle", true),
    dateAdded: sqlDateToIso(item.dateAdded),
    // Manual and automatic tags (cluster keywords for research-direction labels)
    tags: typeof item.getTags === "function" ? item.getTags().map((t) => t?.tag).filter((t) => typeof t === "string" && t) : [],
  };
}

/** Library adapter: plain metadata of all regular (non-note, non-attachment) items. */
export function createZoteroLibrary() {
  return {
    async getItems() {
      const libraryID = profileLibraryID();
      // Same call ZoteroPane uses; joins a load that is already in progress
      await Zotero.Libraries.get(libraryID)?.waitForDataLoad("item");
      const items = (await Zotero.Items.getAll(libraryID, true, false)).filter((i) => i.isRegularItem());
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
