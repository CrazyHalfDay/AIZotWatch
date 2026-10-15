// Author name helpers. Authors are stored as { given, family, name }, where
// `name` is the display form and given/family may be empty for single-field names.

import { collapseWhitespace } from "./text.js";

export function makeAuthor(given, family) {
  const g = collapseWhitespace(given);
  const f = collapseWhitespace(family);
  const name = collapseWhitespace(`${g} ${f}`);
  return { given: g, family: f, name };
}

/**
 * Parse a single-string author name.
 * "Smith, John" -> given "John", family "Smith"; "John Smith" -> given "John", family "Smith".
 */
export function parseAuthorName(raw) {
  const value = collapseWhitespace(raw);
  if (!value) return null;
  if (value.includes(",")) {
    const [family, ...rest] = value.split(",");
    const given = rest.join(" ");
    if (family.trim() && given.trim()) return makeAuthor(given, family);
  }
  const parts = value.split(" ");
  if (parts.length === 1) return { given: "", family: value, name: value };
  const family = parts.pop();
  return makeAuthor(parts.join(" "), family);
}

/** Convert an author to a Zotero creator JSON object. */
export function toZoteroCreator(author, creatorType = "author") {
  if (author.family && author.given) {
    return { creatorType, firstName: author.given, lastName: author.family };
  }
  return { creatorType, name: author.name || author.family || author.given, fieldMode: 1 };
}

// ------------------------------------------------------- followed authors

const ORCID_RE = /\b(\d{4})-?(\d{4})-?(\d{4})-?(\d{3}[\dXx])\b/;
const OPENALEX_AUTHOR_RE = /\b[Aa](\d{4,})\b/;
// Profile URLs are removed as a whole so their host never ends up in the name
const PROFILE_URL_RE = /(?:https?:\/\/)?(?:www\.|api\.)?(?:orcid|openalex)\.org\/[^\s,，;；|]*/gi;
const EDGE_SEPARATORS = /^[\s,，;；|:：-]+|[\s,，;；|:：-]+$/g;
// A leftover "ORCID:" / "OpenAlex ID" label next to the removed identifier
const ID_LABEL = /(?:^|[\s,，;；|])(?:orcid|openalex)(?:\s*id)?\s*[:：]?\s*$/i;
// Brackets that held the identifier: "()" or "(ORCID: )"
const EMPTY_BRACKETS = /[(（[【]\s*(?:(?:orcid|openalex)(?:\s*id)?\s*[:：]?)?\s*[)）\]】]/gi;

/** ORCID iD in canonical "0000-0002-1825-009X" form (from a bare iD or an orcid.org URL); "" when absent. */
export function normalizeOrcid(value) {
  const m = String(value ?? "").match(ORCID_RE);
  return m ? `${m[1]}-${m[2]}-${m[3]}-${m[4].toUpperCase()}` : "";
}

/** OpenAlex author id in canonical "A5023888391" form (from a bare id or an openalex.org URL); "" when absent. */
export function normalizeOpenAlexAuthorId(value) {
  const m = String(value ?? "").match(OPENALEX_AUTHOR_RE);
  return m ? `A${m[1]}` : "";
}

/**
 * Parse the followed-authors setting: one author per line, a name plus an
 * ORCID iD (preferred) or an OpenAlex author id, in any order and with any
 * separator ("汪在聪, 0000-0002-3584-1673", "https://orcid.org/... Jay J. Ague").
 * Lines are split on newlines only: names may contain commas. Blank lines and
 * lines starting with "#" are ignored. Authors are deduplicated by identifier
 * (the first name wins).
 * @returns {{authors: {name: string, orcid?: string, openalexId?: string, key: string}[], invalid: string[]}}
 */
export function parseFollowedAuthors(text) {
  const authors = [];
  const invalid = [];
  const byKey = new Map();
  for (const raw of String(text ?? "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const orcid = normalizeOrcid(line);
    const openalexId = orcid ? "" : normalizeOpenAlexAuthorId(line);
    if (!orcid && !openalexId) {
      invalid.push(line);
      continue;
    }
    const idRe = orcid ? ORCID_RE : OPENALEX_AUTHOR_RE;
    let name = line.replace(PROFILE_URL_RE, " ").replace(idRe, " ").replace(EMPTY_BRACKETS, " ");
    name = collapseWhitespace(name.replace(EDGE_SEPARATORS, "").replace(ID_LABEL, "").replace(EDGE_SEPARATORS, ""));
    const key = orcid ? `orcid:${orcid}` : `openalex:${openalexId}`;
    if (byKey.has(key)) continue;
    const author = { name: name || orcid || openalexId, key };
    if (orcid) author.orcid = orcid;
    else author.openalexId = openalexId;
    byKey.set(key, author);
    authors.push(author);
  }
  return { authors, invalid };
}
