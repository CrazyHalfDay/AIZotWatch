// Local embedding without any API: hashed TF-IDF over unigrams + bigrams.
// Quality is below neural embeddings, but it lets the plugin work out of the
// box. IDF is fitted on the library each run, so vectors are never cached.

import { normalizeInPlace } from "../core/vector.js";

const STOPWORDS = new Set(
  (
    "a about above after again against all also am an and any are as at be because been before being below between both but by " +
    "can could did do does doing down during each few for from further had has have having he her here hers him his how however i if in into is it its " +
    "itself just me more most my no nor not now of off on once only or other our ours out over own same she should so some such than that the their " +
    "theirs them then there these they this those through to too under until up very was we were what when where which while who whom why will with " +
    "would you your yours title abstract using use used based new study studies show shows shown result results paper here we via within without " +
    "however thus therefore among across et al may might must also et one two three"
  ).split(/\s+/),
);

const CJK = /[㐀-鿿豈-﫿]/;
const TOKEN_RE = /[㐀-鿿豈-﫿]+|[^\s\p{P}\p{S}㐀-鿿豈-﫿]+/gu;

function stem(word) {
  if (word.length > 5 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (word.length > 4 && word.endsWith("es") && /(ss|x|ch|sh)es$/.test(word)) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith("s") && !word.endsWith("ss") && !word.endsWith("us") && !word.endsWith("is")) return word.slice(0, -1);
  return word;
}

export function tokenize(text) {
  const tokens = [];
  for (const m of String(text || "").toLowerCase().matchAll(TOKEN_RE)) {
    const w = m[0];
    if (CJK.test(w)) {
      if (w.length === 1) tokens.push(w);
      for (let i = 0; i < w.length - 1; i++) tokens.push(w.slice(i, i + 2));
      continue;
    }
    if (w.length < 3 || STOPWORDS.has(w) || /^\d+$/.test(w)) continue;
    tokens.push(stem(w));
  }
  return tokens;
}

function features(text) {
  const tokens = tokenize(text);
  const counts = new Map();
  const add = (f, w) => counts.set(f, (counts.get(f) || 0) + w);
  for (let i = 0; i < tokens.length; i++) {
    add(tokens[i], 1);
    if (i + 1 < tokens.length) add(`${tokens[i]} ${tokens[i + 1]}`, 0.5);
  }
  return counts;
}

function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export function createLocalEmbedder({ dim = 1024 } = {}) {
  let docFreq = new Map();
  let docCount = 0;

  return {
    signature: `local:hash-tfidf-v1:${dim}`,
    dim,
    cacheable: false,
    requiresFit: true,

    /** Fit IDF statistics on a corpus (the library). */
    fit(texts) {
      docFreq = new Map();
      docCount = texts.length;
      for (const t of texts) {
        for (const f of features(t).keys()) docFreq.set(f, (docFreq.get(f) || 0) + 1);
      }
    },

    async embed(texts) {
      return texts.map((t) => {
        const vec = new Float32Array(dim);
        for (const [f, tf] of features(t)) {
          const idf = Math.log((docCount + 1) / ((docFreq.get(f) || 0) + 1)) + 1;
          const h = fnv1a(f);
          const sign = h & 0x80000000 ? -1 : 1;
          vec[h % dim] += sign * (1 + Math.log(tf)) * idf;
        }
        return normalizeInPlace(vec);
      });
    },
  };
}
