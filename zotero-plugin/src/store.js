// Persistence helpers over an injectable storage backend:
//   storage.readJSON(name) / writeJSON(name, obj) / readBinary(name) / writeBinary(name, bytes)
//   (plus readText / writeText, and remove(name) for single files)
// In Zotero these map to files in <Zotero data dir>/zotwatch/.

import { bytesToFloat32, float32ToBytes } from "./core/vector.js";

/**
 * Load a vector store: `${name}.json` holds { signature, dim, ids, meta },
 * `${name}.bin` holds the Float32 rows. Returns null when missing or inconsistent.
 */
export async function loadVectorStore(storage, name) {
  let index;
  let bytes;
  try {
    index = await storage.readJSON(`${name}.json`);
    bytes = index ? await storage.readBinary(`${name}.bin`) : null;
  } catch {
    return null;
  }
  if (!index || !bytes || !Array.isArray(index.ids) || !index.dim) return null;
  const vectors = bytesToFloat32(bytes);
  if (vectors.length !== index.ids.length * index.dim) return null;
  return { signature: index.signature, dim: index.dim, ids: index.ids, meta: index.meta || [], vectors };
}

export async function saveVectorStore(storage, name, { signature, dim, ids, meta, vectors }) {
  // Write vectors first so a crash never leaves an index pointing at stale data
  await storage.writeBinary(`${name}.bin`, float32ToBytes(vectors));
  await storage.writeJSON(`${name}.json`, { version: 1, signature, dim, ids, meta });
}

/** In-memory storage, used by tests and as a fallback. */
export function createMemoryStorage() {
  const files = new Map();
  return {
    files,
    async readJSON(name) {
      return files.has(name) ? JSON.parse(files.get(name)) : null;
    },
    async writeJSON(name, obj) {
      files.set(name, JSON.stringify(obj));
    },
    async readBinary(name) {
      return files.has(name) ? files.get(name).slice() : null;
    },
    async writeBinary(name, bytes) {
      files.set(name, Uint8Array.from(bytes));
    },
    async readText(name) {
      return files.has(name) ? String(files.get(name)) : null;
    },
    async writeText(name, text) {
      files.set(name, String(text));
    },
    async remove(name) {
      files.delete(name);
    },
  };
}
