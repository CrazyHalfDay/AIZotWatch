// Text helpers shared by sources, dedupe and scoring.

const NAMED_ENTITIES = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
  ndash: "–",
  mdash: "—",
  hellip: "…",
  lsquo: "‘",
  rsquo: "’",
  ldquo: "“",
  rdquo: "”",
  deg: "°",
  plusmn: "±",
  times: "×",
  minus: "−",
  micro: "µ",
  alpha: "α",
  beta: "β",
  gamma: "γ",
  delta: "δ",
  epsilon: "ε",
  lambda: "λ",
  mu: "μ",
  pi: "π",
  sigma: "σ",
  tau: "τ",
  omega: "ω",
  Delta: "Δ",
  permil: "‰",
};

export function decodeEntities(value) {
  if (!value) return "";
  return value.replace(/&(#x[0-9a-fA-F]+|#[0-9]+|[a-zA-Z][a-zA-Z0-9]*);/g, (match, body) => {
    if (body[0] === "#") {
      const code = body[1] === "x" || body[1] === "X" ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      if (Number.isFinite(code) && code > 0 && code <= 0x10ffff) {
        try {
          return String.fromCodePoint(code);
        } catch {
          return match;
        }
      }
      return match;
    }
    return Object.prototype.hasOwnProperty.call(NAMED_ENTITIES, body) ? NAMED_ENTITIES[body] : match;
  });
}

export function collapseWhitespace(value) {
  return (value || "").replace(/\s+/g, " ").trim();
}

/** Strip HTML/JATS markup and entities; returns null for empty input. */
export function cleanHtml(value) {
  if (!value) return null;
  // Crossref JATS abstracts usually start with an "Abstract" heading element
  let text = String(value).replace(/<([\w:]*title)\b[^>]*>\s*(abstract|summary)\s*<\/\1>/gi, " ");
  text = text.replace(/<[^>]+>/g, " ");
  text = collapseWhitespace(decodeEntities(text));
  return text || null;
}

/** Clean a title: drop inline markup (<i>, <sub>, ...) and normalise whitespace. */
export function cleanTitle(value) {
  if (!value) return "";
  return collapseWhitespace(decodeEntities(String(value).replace(/<[^>]+>/g, "")));
}

/** Normalise a title for duplicate detection. */
export function normalizeTitle(title) {
  return collapseWhitespace(
    (title || "")
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "")
      .replace(/[^\p{L}\p{N}]+/gu, " "),
  );
}

/** Normalise a DOI: lowercase, without resolver prefix. Returns "" when invalid. */
export function normalizeDoi(value) {
  if (!value) return "";
  let doi = String(value).trim().toLowerCase();
  doi = doi.replace(/^(https?:\/\/)?(dx\.)?doi\.org\//, "").replace(/^doi:\s*/, "");
  const match = doi.match(/10\.\d{4,9}\/\S+/);
  if (!match) return "";
  return match[0].replace(/[.,;]+$/, "");
}

/** Extract a DOI from free text such as Zotero's "Extra" field. */
export function extractDoi(text) {
  if (!text) return "";
  const match = String(text).match(/\b(10\.\d{4,9}\/[^\s"<>]+)/i);
  return match ? normalizeDoi(match[1]) : "";
}

/** DOI from a "DOI: ..." line in Zotero's Extra field (older schemas kept it there). */
export function extraFieldDoi(extra) {
  const match = String(extra || "").match(/^\s*DOI:\s*(\S+)/im);
  return match ? normalizeDoi(match[1]) : "";
}

/** Normalise an ISSN to the canonical "1234-567X" form; "" when invalid. */
export function normalizeIssn(value) {
  if (!value) return "";
  const compact = String(value).toUpperCase().replace(/[^0-9X]/g, "");
  if (compact.length !== 8) return "";
  return `${compact.slice(0, 4)}-${compact.slice(4)}`;
}

/** Split a free-form list (newlines, commas, semicolons) into trimmed entries. */
export function parseList(text) {
  if (!text) return [];
  return String(text)
    .split(/[\n,;，；]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

const NON_ARTICLE_PATTERNS = [
  /^table of contents$/i,
  /^masthead$/i,
  /^editorial board$/i,
  /^errat(a|um)$/i,
  /^correction(s)?$/i,
  /^retraction$/i,
  /^connect\. support\. inspire\.$/i,
  /information for authors/i,
  /publication information/i,
  /author guidelines/i,
  /instructions for authors/i,
  /guide for authors/i,
  /^ieee transactions on [a-z\s]+$/i,
  /^ieee journal of [a-z\s]+$/i,
  /^proceedings of the ieee$/i,
  /^ieee [a-z\s]+ magazine$/i,
  /^issue information$/i,
  /^front cover$/i,
  /^back cover$/i,
  /^cover image$/i,
];

/** True when a title looks like journal metadata (TOC, masthead, ...), not an article. */
export function isNonArticleTitle(title, venue) {
  if (!title) return true;
  const clean = title.trim();
  if (NON_ARTICLE_PATTERNS.some((re) => re.test(clean))) return true;
  if (venue && clean.toLowerCase() === String(venue).trim().toLowerCase()) return true;
  return false;
}

/** Fast non-cryptographic 53-bit string hash (cyrb53), returned as hex. */
export function hashString(str, seed = 0) {
  let h1 = 0xdeadbeef ^ seed;
  let h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  const value = 4294967296 * (2097151 & h2) + (h1 >>> 0);
  return value.toString(16).padStart(14, "0");
}

/** Text used for embedding, identical to the Python pipeline. */
export function contentForEmbedding(title, abstract) {
  return `Title: ${title || ""}\nAbstract: ${abstract || ""}`;
}

/** Format author list for display: "A, B, C et al." */
export function formatAuthors(authors, max = 5) {
  const names = (authors || []).map((a) => (typeof a === "string" ? a : a?.name || "")).filter(Boolean);
  if (names.length <= max) return names.join(", ");
  return `${names.slice(0, max).join(", ")} et al.`;
}
