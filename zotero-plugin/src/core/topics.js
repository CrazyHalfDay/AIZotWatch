// Research-direction topics of the library clusters (Python ClusterInfo +
// ClusterLabeler): keywords from library tags, and LLM labels carried over
// between reclusterings by member-key Jaccard so that a recluster (every run
// in local mode) never relabels directions that did not really change.

export const MAX_CLUSTER_KEYWORDS = 10;
export const MAX_CLUSTER_TITLES = 5;
// A stored label is reused when its members overlap the new cluster this much
export const LABEL_CARRY_OVER_JACCARD = 0.5;

const HAS_WORD = /[\p{L}\p{N}]/u;

/**
 * Whether a library tag can serve as a cluster keyword. Excludes the plugin's
 * own output tag (added to every saved recommendation), workflow tags such as
 * "/unread" or "_todo", and marker tags without any letter, digit or CJK
 * character ("⭐", "✔️", "⏳"), mirroring the Python exclude_tags config.
 */
export function isKeywordTag(tag, outputTag = "") {
  const t = typeof tag === "string" ? tag.trim() : "";
  if (!t || t.startsWith("/") || t.startsWith("_") || !HAS_WORD.test(t)) return false;
  const own = String(outputTag || "").trim().toLowerCase();
  return !own || t.toLowerCase() !== own;
}

/**
 * The most frequent keyword tags among `items` (each `{tags: string[]}`),
 * counted once per item; ties keep their first-seen order (Python Counter.most_common).
 */
export function topKeywords(items, { outputTag = "", limit = MAX_CLUSTER_KEYWORDS } = {}) {
  const counts = new Map();
  for (const item of items) {
    const tags = new Set((item?.tags || []).filter((t) => isKeywordTag(t, outputTag)).map((t) => t.trim()));
    for (const t of tags) counts.set(t, (counts.get(t) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([t]) => t);
}

/** Jaccard similarity of two key lists (or sets). */
export function jaccard(a, b) {
  const setA = a instanceof Set ? a : new Set(a);
  const setB = b instanceof Set ? b : new Set(b);
  if (!setA.size && !setB.size) return 0;
  let inter = 0;
  for (const k of setA) if (setB.has(k)) inter++;
  return inter / (setA.size + setB.size - inter);
}

/** Valid entries of a stored label file: [{keys: string[], label: string}]. */
export function storedLabelEntries(data) {
  const list = Array.isArray(data?.clusters) ? data.clusters : [];
  return list.filter(
    (c) => c && typeof c.label === "string" && c.label.trim() && Array.isArray(c.keys) && c.keys.length && c.keys.every((k) => typeof k === "string"),
  );
}

/**
 * Carry stored labels over to the current clusters: each cluster takes the
 * label of the stored cluster with the highest member-key Jaccard (at least
 * `minJaccard`), one-to-one, assigned greedily from the best overlap down.
 * @param {{memberKeys?: string[]}[]} clusters - current clusters
 * @param {{keys: string[], label: string}[]} stored - see storedLabelEntries
 * @returns {(string|null)[]} label per cluster (null: unlabeled)
 */
export function matchStoredLabels(clusters, stored, options = {}) {
  return matchStoredEntries(clusters, stored, options).map((j) => (j === null ? null : stored[j].label.trim()));
}

/**
 * Index of the stored entry each current cluster takes (see matchStoredLabels).
 * @returns {(number|null)[]} index into `stored` per cluster (null: unlabeled)
 */
export function matchStoredEntries(clusters, stored, { minJaccard = LABEL_CARRY_OVER_JACCARD } = {}) {
  const taken = clusters.map(() => null);
  const storedSets = stored.map((s) => new Set(s.keys));
  const pairs = [];
  clusters.forEach((c, i) => {
    if (!Array.isArray(c.memberKeys) || !c.memberKeys.length) return;
    const keys = new Set(c.memberKeys);
    storedSets.forEach((s, j) => {
      const score = jaccard(keys, s);
      if (score >= minJaccard) pairs.push({ i, j, score });
    });
  });
  pairs.sort((a, b) => b.score - a.score || a.i - b.i || a.j - b.j);
  const usedStored = new Set();
  for (const { i, j } of pairs) {
    if (taken[i] !== null || usedStored.has(j)) continue;
    taken[i] = j;
    usedStored.add(j);
  }
  return taken;
}

/** Stored label file for the labeled current clusters. */
export function labelStoreOf(clusters, labels) {
  return {
    version: 1,
    clusters: clusters
      .map((c, i) => (labels[i] && Array.isArray(c.memberKeys) && c.memberKeys.length ? { keys: c.memberKeys, label: labels[i] } : null))
      .filter(Boolean),
  };
}

/** Whether two label stores hold the same clusters and labels (to skip needless writes). */
export function sameLabelStore(a, b) {
  const x = storedLabelEntries(a);
  const y = storedLabelEntries(b);
  if (x.length !== y.length) return false;
  return x.every((c, i) => c.label === y[i].label && c.keys.length === y[i].keys.length && c.keys.every((k, j) => k === y[i].keys[j]));
}

/** Topics for display: [{label, size}] by size, largest first. */
export function topicsOf(data) {
  return storedLabelEntries(data)
    .map((c) => ({ label: c.label.trim(), size: c.keys.length }))
    .sort((a, b) => b.size - a.size);
}
