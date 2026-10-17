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
 * Whether a whitelist category marks a Chinese core journal: "(CN)" anywhere
 * ("GEOCHEMISTRY (CN)") or a "_CN" suffix ("GEOSCIENCES_CN", the style of the
 * bundled preset and the Python data).
 */
export function isChineseCoreCategory(category) {
  const text = String(category ?? "").trim();
  return /\(CN\)/i.test(text) || /_CN$/i.test(text);
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
      isChineseCore: isChineseCoreCategory(category),
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
  return {
    count: values.length,
    withImpactFactor: values.filter((j) => j.impactFactor !== null).length,
    // Journals 「补全学科与影响因子」 can complete (empty or library:N category, no impact factor, no title)
    incomplete: values.filter(isIncompleteJournal).length,
    invalidLines,
  };
}

/**
 * Whether a whitelist category carries no subject: blank, or the
 * `library:N` placeholder (item count) written by 「从文库生成」 before 0.4.0.
 */
export function isEmptyCategory(category) {
  const value = String(category ?? "").trim();
  return !value || /^library:\s*\d*$/i.test(value);
}

/** Whether a parsed whitelist journal lacks a title, a category or an impact factor. */
export function isIncompleteJournal(journal) {
  return !String(journal?.title ?? "").trim() || isEmptyCategory(journal?.category) || !Number.isFinite(journal?.impactFactor);
}

/** Normalised ISSNs of the data rows of whitelist CSV text, in order. */
export function whitelistIssns(text) {
  return [...parseWhitelist(text).keys()];
}

/**
 * Whitelist rows of journals (one row per ISSN, so either ISSN of a journal
 * matches Crossref records): [{issn, title, category, impactFactor}].
 */
export function journalRows(journals) {
  const rows = [];
  const seen = new Set();
  for (const j of journals || []) {
    for (const raw of j?.issns || []) {
      const issn = normalizeIssn(raw);
      if (!issn || seen.has(issn)) continue;
      seen.add(issn);
      rows.push({ issn, title: j.title || "", category: j.category || "", impactFactor: Number.isFinite(j.impactFactor) ? j.impactFactor : null });
    }
  }
  return rows;
}

const HEADER = "issn,title,category,impact_factor";

/**
 * Merge journal rows into whitelist CSV text without disturbing it: comment
 * lines, the order and hand-written rows stay as they are. Rows already listed
 * only fill what is missing there (a blank title, a blank or `library:N`
 * category, a missing impact factor) and never overwrite a value; with
 * `append` new ISSNs are added at the end (with a header when the text was
 * empty). Returns the new text, the number of added rows and of rows filled.
 * @param {{issn, title?, category?, impactFactor?}[]} rows
 */
export function mergeWhitelistRows(text, rows, { append = true } = {}) {
  const byIssn = new Map();
  for (const row of rows || []) {
    const issn = normalizeIssn(row?.issn);
    if (issn && !byIssn.has(issn)) byIssn.set(issn, row);
  }
  const listed = new Set();
  let filled = 0;
  const lines = String(text ?? "")
    .split(/\r?\n/)
    .map((raw) => {
      const line = raw.trim();
      if (!line || line.startsWith("#")) return raw;
      const fields = splitCsvLine(line);
      const issn = normalizeIssn(fields[0]);
      if (!issn) return raw;
      listed.add(issn);
      const row = byIssn.get(issn);
      if (!row) return raw;
      const [issnText, title = "", category = "", ifText = "", ...rest] = fields;
      const next = [issnText, title, category, ifText];
      if (!title && row.title) next[1] = row.title;
      if (isEmptyCategory(category) && row.category) next[2] = row.category;
      if (!Number.isFinite(parseFloat(ifText)) && Number.isFinite(row.impactFactor)) next[3] = String(row.impactFactor);
      if (next.every((value, i) => value === [issnText, title, category, ifText][i])) return raw;
      filled++;
      return [...next, ...rest].map(csvField).join(",");
    });
  let added = 0;
  const fresh = append ? [...byIssn.entries()].filter(([issn]) => !listed.has(issn)) : [];
  if (fresh.length) {
    while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
    if (!lines.length) lines.push(HEADER);
    for (const [issn, row] of fresh) {
      lines.push([issn, row.title || "", row.category || "", Number.isFinite(row.impactFactor) ? row.impactFactor : ""].map(csvField).join(","));
      added++;
    }
  }
  return { text: lines.join("\n"), added, filled };
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
