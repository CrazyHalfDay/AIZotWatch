import assert from "node:assert/strict";
import test from "node:test";

import { chooseClusterCount, clusterSearchBounds, silhouetteFromDistances, sphericalKMeans } from "../src/core/kmeans.js";
import {
  ProfileRanker,
  assignLabel,
  buildClusters,
  computeTemporalWeights,
  computeThresholds,
  journalScore,
  percentile,
} from "../src/core/scoring.js";
import { Matrix, createRng, gaussian, normalizeInPlace } from "../src/core/vector.js";
import { createLocalEmbedder, tokenize } from "../src/embedding/local.js";
import { loadSettings } from "../src/settings.js";

/** Three well-separated blobs on the unit sphere. */
function blobs(perCluster = 30, dim = 32, seed = 1) {
  const rng = createRng(seed);
  const centers = [0, 1, 2].map(() => normalizeInPlace(Float32Array.from({ length: dim }, () => gaussian(rng))));
  const rows = [];
  const labels = [];
  centers.forEach((c, ci) => {
    for (let i = 0; i < perCluster; i++) {
      rows.push(normalizeInPlace(Float32Array.from(c, (x) => x + 0.08 * gaussian(rng))));
      labels.push(ci);
    }
  });
  return { matrix: Matrix.fromRows(rows), labels, centers };
}

test("matrix top-k returns sorted best rows", () => {
  const m = Matrix.fromRows([[1, 0], [0, 1], [0.8, 0.6], [-1, 0]]);
  const top = m.topK(Float32Array.from([1, 0]), 2);
  assert.deepEqual(top.map((t) => t.index), [0, 2]);
  assert.deepEqual(m.topK(Float32Array.from([1, 0]), 10).map((t) => t.index), [0, 2, 1, 3]);
  assert.deepEqual(m.topK(Float32Array.from([1, 0]), 0), []);
});

test("spherical k-means recovers separated clusters", async () => {
  const { matrix, labels } = blobs();
  const { assignments } = await sphericalKMeans(matrix, 3, { seed: 3 });
  // Every true cluster maps to exactly one predicted cluster
  for (const c of [0, 1, 2]) {
    const predicted = new Set(labels.map((l, i) => (l === c ? assignments[i] : null)).filter((x) => x !== null));
    assert.equal(predicted.size, 1);
  }
});

test("silhouette is high for good clusterings and null for one cluster", () => {
  const { matrix, labels } = blobs(10, 16);
  const n = matrix.rows;
  const dist = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0;
    for (let d = 0; d < matrix.dim; d++) s += matrix.data[i * matrix.dim + d] * matrix.data[j * matrix.dim + d];
    dist[i * n + j] = i === j ? 0 : 1 - s;
  }
  assert.ok(silhouetteFromDistances(dist, n, Int32Array.from(labels), 3) > 0.7);
  assert.equal(silhouetteFromDistances(dist, n, new Int32Array(n), 1), null);
});

test("cluster search bounds follow the Python rules", () => {
  assert.equal(clusterSearchBounds(4), null);
  assert.deepEqual(clusterSearchBounds(50), { minK: 5, maxK: 8 });
  assert.deepEqual(clusterSearchBounds(2000), { minK: 14, maxK: 35 });
  assert.deepEqual(clusterSearchBounds(400, { minClusters: 5, maxClusters: 10 }), { minK: 6, maxK: 10 });
});

test("k selection and cluster summaries", async () => {
  const { matrix } = blobs(40, 32);
  const { k, scores } = await chooseClusterCount(matrix, { minClusters: 2, maxClusters: 6, searchDim: 16, tolerance: 0.05 });
  assert.ok(scores.length > 0);
  assert.equal(k, 3);

  const weights = new Float32Array(matrix.rows).fill(1);
  const { clusters } = await buildClusters(matrix, weights, { minClusters: 2, maxClusters: 6, tolerance: 0.05 });
  assert.equal(clusters.length, 3);
  assert.equal(clusters.reduce((s, c) => s + c.memberCount, 0), matrix.rows);
  for (const c of clusters) assert.ok(Math.abs(Math.hypot(...c.centroid) - 1) < 1e-4);
});

test("temporal weights decay with age and respect the floor", () => {
  const now = Date.UTC(2024, 0, 1);
  const w = computeTemporalWeights(["2024-01-01T00:00:00Z", "2023-07-05T00:00:00Z", "2000-01-01T00:00:00Z", null], { halflifeDays: 180, now });
  assert.equal(w[0], 1);
  assert.ok(Math.abs(w[1] - 0.5) < 0.01);
  assert.ok(Math.abs(w[2] - 0.05) < 1e-6);
  assert.equal(w[3], 1);
  assert.deepEqual(Array.from(computeTemporalWeights(["2000-01-01"], { halflifeDays: 0, now })), [1]);
});

test("percentile thresholds match numpy and respect floors", () => {
  assert.equal(percentile([1, 2, 3, 4], 50), 2.5);
  assert.equal(percentile([5], 95), 5);
  const cfg = loadSettings(() => undefined).scoring.thresholds;
  const scores = Array.from({ length: 101 }, (_, i) => i / 100);
  const t = computeThresholds(scores, cfg);
  assert.equal(t.mode, "dynamic");
  assert.ok(Math.abs(t.mustRead - 0.95) < 1e-9);
  assert.ok(Math.abs(t.consider - 0.7) < 1e-9);
  const low = computeThresholds([0.01, 0.02, 0.03], { ...cfg, minMustRead: 0.5, minConsider: 0.5 });
  assert.ok(low.mustRead > low.consider);
  assert.equal(computeThresholds([0.4], cfg).mode, "fixed");
  assert.equal(assignLabel(0.99, t), "must_read");
  assert.equal(assignLabel(0.8, t), "consider");
  assert.equal(assignLabel(0.1, t), "ignore");
});

test("journal score", () => {
  const cfg = { preprintScore: 0.6, chineseCoreScore: 0.7, unknownScore: 0.3, logBase: 25 };
  const journals = new Map([
    ["0016-7037", { impactFactor: 5, isChineseCore: false }],
    ["1000-0569", { impactFactor: null, isChineseCore: true }],
    ["0028-0836", { impactFactor: 64.8, isChineseCore: false }],
  ]);
  assert.equal(journalScore({ source: "arxiv" }, journals, cfg).ifScore, 0.6);
  assert.ok(Math.abs(journalScore({ source: "crossref", issns: ["0016-7037"] }, journals, cfg).ifScore - Math.log(6) / Math.log(25)) < 1e-9);
  assert.equal(journalScore({ source: "crossref", issns: ["0028-0836"] }, journals, cfg).ifScore, 1);
  assert.equal(journalScore({ source: "crossref", issns: ["1000-0569"] }, journals, cfg).isChineseCore, true);
  assert.equal(journalScore({ source: "crossref", issns: ["9999-9999"] }, journals, cfg).ifScore, 0.3);
});

test("ranker fuses micro and macro scores, gates and labels", async () => {
  const { matrix, centers } = blobs(20, 16, 5);
  const weights = new Float32Array(matrix.rows).fill(1);
  const { clusters } = await buildClusters(matrix, weights, { minClusters: 2, maxClusters: 5 });
  const settings = loadSettings((k) => (k === "embedding.provider" ? "dashscope" : undefined));
  const ranker = new ProfileRanker({ matrix, weights, clusters }, new Map(), settings.scoring);

  const rng = createRng(9);
  const near = { id: "near", source: "crossref", issns: [], vector: centers[0] };
  const far = { id: "far", source: "crossref", issns: [], vector: normalizeInPlace(Float32Array.from(centers[0], (x) => -x + 0.01 * gaussian(rng))) };
  const { ranked, gated } = await ranker.rank([far, near]);
  assert.equal(gated, 1);
  assert.equal(ranked.length, 1);
  assert.equal(ranked[0].id, "near");
  assert.ok(ranked[0].similarity > 0.9);
  assert.ok(ranked[0].micro > 0.9 && ranked[0].macro > 0.5);
  assert.ok(Math.abs(ranked[0].score - (0.8 * ranked[0].similarity + 0.2 * 0.3)) < 1e-6);

  const cold = new ProfileRanker({ matrix: new Matrix(0, 16), weights: new Float32Array(0), clusters: [] }, new Map(), settings.scoring);
  const coldResult = await cold.rank([near, far]);
  assert.equal(coldResult.thresholds.mode, "random");
  assert.ok(coldResult.ranked.every((w) => w.label === "consider"));
});

test("local embedder separates topics", async () => {
  assert.deepEqual(tokenize("Copper isotopes in the mantle 2024 地幔熔融"), ["copper", "isotope", "mantle", "地幔", "幔熔", "熔融"]);
  const corpus = [
    "Copper isotope fractionation in subduction zone magmas",
    "Chalcophile elements and sulfide saturation in arc magmas",
    "Mantle melting beneath mid-ocean ridges",
    "Deep learning for image classification",
  ];
  const emb = createLocalEmbedder({ dim: 512 });
  emb.fit(corpus);
  const [a, b, c] = await emb.embed([
    "Copper isotopes record sulfide saturation in subduction magmas",
    "Copper isotope behaviour in arc magmas",
    "Convolutional networks for image classification",
  ]);
  const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
  assert.ok(dot(a, b) > dot(a, c) + 0.1);
  assert.ok(Math.abs(dot(a, a) - 1) < 1e-5);
});
