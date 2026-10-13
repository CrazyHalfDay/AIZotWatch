// OpenAlex abstract fallback: many publishers (e.g. Elsevier) do not deposit
// abstracts in Crossref, but OpenAlex stores them as an inverted index.

import { buildUrl, getJson, mapWithConcurrency } from "../core/http.js";
import { collapseWhitespace, normalizeDoi } from "../core/text.js";

const OPENALEX_WORKS = "https://api.openalex.org/works";

export function reconstructAbstract(inverted) {
  if (!inverted || typeof inverted !== "object") return null;
  const words = [];
  for (const [word, positions] of Object.entries(inverted)) {
    for (const pos of positions || []) words[pos] = word;
  }
  const text = collapseWhitespace(words.filter((w) => w !== undefined).join(" "));
  return text || null;
}

/**
 * Fill in missing abstracts in place for works with DOIs.
 * @returns {Promise<number>} number of abstracts added
 */
export async function enrichAbstracts(http, works, { mailto = "", batchSize = 50, concurrency = 2, retry = {}, log = () => {} } = {}) {
  const missing = works.filter((w) => !w.abstract && normalizeDoi(w.doi) && !/[|,]/.test(w.doi));
  if (!missing.length) return 0;
  const batches = [];
  for (let i = 0; i < missing.length; i += batchSize) batches.push(missing.slice(i, i + batchSize));

  let added = 0;
  await mapWithConcurrency(batches, concurrency, async (batch) => {
    const url = buildUrl(OPENALEX_WORKS, {
      filter: `doi:${batch.map((w) => `https://doi.org/${w.doi}`).join("|")}`,
      select: "doi,abstract_inverted_index",
      "per-page": batchSize,
      mailto,
    });
    let data;
    try {
      data = await getJson(http, url, retry);
    } catch (e) {
      log(`OpenAlex batch failed: ${e.message}`);
      return;
    }
    const byDoi = new Map();
    for (const r of data?.results || []) {
      const abstract = reconstructAbstract(r.abstract_inverted_index);
      if (abstract) byDoi.set(normalizeDoi(r.doi), abstract);
    }
    for (const w of batch) {
      const abstract = byDoi.get(normalizeDoi(w.doi)) || null;
      if (abstract) {
        w.abstract = abstract;
        added++;
      }
    }
  });
  log(`OpenAlex: filled ${added}/${missing.length} missing abstracts`);
  return added;
}
