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
