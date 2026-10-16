// Journal whitelist handling: CSV parsing (same format as data/journal_whitelist.csv)
// and deriving target journals from the venues already in the user's library.

import { collapseWhitespace, normalizeIssn } from "./text.js";

/** Split one CSV line, honouring double-quoted fields. */
export function splitCsvLine(line) {
  const out = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      out.push(field);
      field = "";
    } else {
      field += ch;
    }
  }
  out.push(field);
  return out.map((f) => f.trim());
}

/**
 * Parse whitelist CSV text: `issn,title,category,impact_factor`.
 * The header row is optional; lines starting with "#" are comments.
 * @returns {Map<string, {issn, title, category, impactFactor, isChineseCore}>}
 */
export function parseWhitelist(text) {
  const journals = new Map();
  if (!text) return journals;
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const [issnRaw, title = "", category = "", ifRaw = ""] = splitCsvLine(line);
    const issn = normalizeIssn(issnRaw);
    if (!issn) continue; // also skips the header row
    const ifNum = parseFloat(ifRaw);
    journals.set(issn, {
      issn,
      title,
      category,
      impactFactor: Number.isFinite(ifNum) ? ifNum : null,
      isChineseCore: category.includes("(CN)"),
    });
  }
  return journals;
}

function csvField(value) {
  const s = value == null ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * Summary of whitelist CSV text for live previews: journals (unique ISSNs), how
 * many have an impact factor, and the data lines that yield no journal (the
 * header row, blank lines and "#" comments are not counted as invalid).
 */
export function summarizeWhitelist(text) {
  const journals = parseWhitelist(text);
  const invalidLines = [];
  let first = true;
  for (const rawLine of String(text ?? "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const isHeader = first && /^"?issn"?$/i.test(splitCsvLine(line)[0]);
    first = false;
    if (!isHeader && !normalizeIssn(splitCsvLine(line)[0])) invalidLines.push(line);
  }
  const values = [...journals.values()];
  return { count: values.length, withImpactFactor: values.filter((j) => j.impactFactor !== null).length, invalidLines };
}

export function serializeWhitelist(entries) {
  const lines = ["issn,title,category,impact_factor"];
  for (const e of entries) {
    lines.push([e.issn, e.title, e.category || "", e.impactFactor ?? ""].map(csvField).join(","));
  }
  return lines.join("\n");
}

export function venueKey(title) {
  return collapseWhitespace((title || "").toLowerCase().replace(/^the\s+/, "").replace(/[^\p{L}\p{N}]+/gu, " "));
}

/**
 * Rank the journals in the library by number of items.
 * @param {Array<{itemType, venue, issn}>} items
 * @returns {Array<{title, count, issns: string[]}>}
 */
export function libraryVenues(items) {
  const venues = new Map();
  for (const item of items) {
    if (item.itemType && item.itemType !== "journalArticle") continue;
    const key = venueKey(item.venue);
    if (!key) continue;
    let v = venues.get(key);
    if (!v) {
      v = { title: collapseWhitespace(item.venue), count: 0, issns: new Set() };
      venues.set(key, v);
    }
    v.count++;
    for (const part of String(item.issn || "").split(/[\s,;]+/)) {
      const issn = normalizeIssn(part);
      if (issn) v.issns.add(issn);
    }
  }
  return [...venues.values()]
    .map((v) => ({ title: v.title, count: v.count, issns: [...v.issns] }))
    .sort((a, b) => b.count - a.count || a.title.localeCompare(b.title));
}
