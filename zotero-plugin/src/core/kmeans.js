// Spherical k-means with Silhouette-based k selection, ported from the Python
// ProfileClusterer (FAISS + sklearn). Vectors are expected to be L2-normalised.

import { Matrix, createRng, dot, gaussian, normalizeInPlace, yieldToEventLoop } from "./vector.js";

// Long loops yield to the event loop roughly every this many dot products,
// so clustering never freezes the Zotero UI for long.
const WORK_PER_SLICE = 2e7;

/** k-means++ seeding on the unit sphere (distance = 1 - cosine). */
async function seedCentroids(matrix, k, rng, cooperative) {
  const { rows, dim } = matrix;
  const centroids = new Float32Array(k * dim);
  const first = Math.floor(rng() * rows);
  centroids.set(matrix.row(first), 0);
  const minDist = new Float64Array(rows).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let total = 0;
    for (let i = 0; i < rows; i++) {
      const d = Math.max(0, 1 - dot(matrix.data, centroids, i * dim, (c - 1) * dim, dim));
      if (d < minDist[i]) minDist[i] = d;
      total += minDist[i] * minDist[i];
    }
    let pick = rows - 1;
    if (total > 0) {
      let r = rng() * total;
      for (let i = 0; i < rows; i++) {
        r -= minDist[i] * minDist[i];
        if (r <= 0) {
          pick = i;
          break;
        }
      }
    } else {
      pick = Math.floor(rng() * rows);
    }
    centroids.set(matrix.row(pick), c * dim);
    if (cooperative) await yieldToEventLoop();
  }
  return centroids;
}

async function assign(matrix, centroids, k, assignments, cooperative) {
  const { rows, dim } = matrix;
  const rowsPerSlice = Math.max(1, Math.floor(WORK_PER_SLICE / (k * dim)));
  let changed = 0;
  for (let i = 0; i < rows; i++) {
    if (cooperative && i > 0 && i % rowsPerSlice === 0) await yieldToEventLoop();
    let best = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < k; c++) {
      const s = dot(matrix.data, centroids, i * dim, c * dim, dim);
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (assignments[i] !== best) {
      assignments[i] = best;
      changed++;
    }
  }
  return changed;
}

/**
 * Run spherical k-means.
 * @returns {{ centroids: Float32Array, assignments: Int32Array, k: number }}
 */
export async function sphericalKMeans(matrix, k, { iterations = 20, seed = 42, cooperative = false } = {}) {
  const { rows, dim } = matrix;
  k = Math.max(1, Math.min(k, rows));
  const rng = createRng(seed);
  const centroids = await seedCentroids(matrix, k, rng, cooperative);
  const assignments = new Int32Array(rows).fill(-1);

  for (let iter = 0; iter < iterations; iter++) {
    const changed = await assign(matrix, centroids, k, assignments, cooperative);
    if (changed === 0 && iter > 0) break;

    centroids.fill(0);
    const counts = new Int32Array(k);
    for (let i = 0; i < rows; i++) {
      const c = assignments[i];
      counts[c]++;
      const off = c * dim;
      const rowOff = i * dim;
      for (let d = 0; d < dim; d++) centroids[off + d] += matrix.data[rowOff + d];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        // Re-seed an empty cluster with a random point
        centroids.set(matrix.row(Math.floor(rng() * rows)), c * dim);
      } else {
        normalizeInPlace(centroids, c * dim, dim);
      }
    }
    if (cooperative) await yieldToEventLoop();
  }
  await assign(matrix, centroids, k, assignments, cooperative);
  return { centroids, assignments, k };
}

/** Mean Silhouette coefficient with cosine distance, from a precomputed distance matrix. */
export function silhouetteFromDistances(distances, n, labels, k) {
  const sums = new Float64Array(k);
  const counts = new Int32Array(k);
  for (let i = 0; i < n; i++) counts[labels[i]]++;
  if (counts.filter((c) => c > 0).length < 2) return null;

  let total = 0;
  for (let i = 0; i < n; i++) {
    sums.fill(0);
    const row = i * n;
    for (let j = 0; j < n; j++) {
      if (j !== i) sums[labels[j]] += distances[row + j];
    }
    const own = labels[i];
    if (counts[own] <= 1) continue; // sklearn: s(i) = 0 for singleton clusters
    const a = sums[own] / (counts[own] - 1);
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c !== own && counts[c] > 0) b = Math.min(b, sums[c] / counts[c]);
    }
    const denom = Math.max(a, b);
    total += denom > 0 ? (b - a) / denom : 0;
  }
  return total / n;
}

/** Random Gaussian projection to `targetDim` dims (re-normalised), used to speed up k search. */
export async function randomProjection(matrix, targetDim, seed = 7) {
  if (matrix.dim <= targetDim) return matrix;
  const rng = createRng(seed);
  const proj = new Float32Array(matrix.dim * targetDim);
  for (let i = 0; i < proj.length; i++) proj[i] = gaussian(rng);
  const out = new Matrix(matrix.rows, targetDim);
  const rowsPerSlice = Math.max(1, Math.floor(WORK_PER_SLICE / (matrix.dim * targetDim)));
  for (let r = 0; r < matrix.rows; r++) {
    if (r > 0 && r % rowsPerSlice === 0) await yieldToEventLoop();
    const rowOff = r * matrix.dim;
    for (let t = 0; t < targetDim; t++) {
      let s = 0;
      for (let d = 0; d < matrix.dim; d++) s += matrix.data[rowOff + d] * proj[d * targetDim + t];
      out.data[r * targetDim + t] = s;
    }
  }
  return out.normalizeRows();
}

/** Cluster count search bounds, identical to the Python implementation. */
export function clusterSearchBounds(n, { minClusters = 5, maxClusters = 35 } = {}) {
  if (n < 5) return null;
  const theoreticalMax = Math.floor(Math.sqrt(n));
  const maxKLimit = n < 100 ? Math.max(theoreticalMax, Math.min(Math.floor(n / 5), 8)) : Math.min(35, theoreticalMax);
  const maxK = Math.min(maxKLimit, Math.floor(n / 2), maxClusters);
  if (maxK < 2) return null;
  const adaptiveMin = Math.max(2, Math.floor(Math.sqrt(n) / 3));
  const minK = Math.min(Math.max(minClusters, adaptiveMin), maxK);
  return { minK, maxK };
}

/**
 * Choose k with the biased Silhouette rule: the largest k whose score is within
 * `tolerance` (relative) of the best score.
 */
export async function chooseClusterCount(
  matrix,
  { minClusters = 5, maxClusters = 35, tolerance = 0.2, sampleSize = 600, searchDim = 128, seed = 42 } = {},
) {
  const bounds = clusterSearchBounds(matrix.rows, { minClusters, maxClusters });
  if (!bounds) return { k: 1, silhouette: null, scores: [] };

  // Subsample + project for speed; silhouette needs O(s^2) distances.
  const rng = createRng(seed);
  let sample = matrix;
  if (matrix.rows > sampleSize) {
    const idx = Array.from({ length: matrix.rows }, (_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    sample = matrix.selectRows(idx.slice(0, sampleSize));
  }
  sample = await randomProjection(sample, searchDim, seed);

  const n = sample.rows;
  const distances = new Float32Array(n * n);
  const rowsPerSlice = Math.max(1, Math.floor(WORK_PER_SLICE / (n * sample.dim)));
  for (let i = 0; i < n; i++) {
    if (i > 0 && i % rowsPerSlice === 0) await yieldToEventLoop();
    for (let j = i; j < n; j++) {
      const d = i === j ? 0 : Math.max(0, 1 - dot(sample.data, sample.data, i * sample.dim, j * sample.dim, sample.dim));
      distances[i * n + j] = d;
      distances[j * n + i] = d;
    }
  }
  await yieldToEventLoop();

  const scores = [];
  for (let k = bounds.minK; k <= bounds.maxK; k++) {
    const { assignments } = await sphericalKMeans(sample, k, { iterations: 10, seed });
    const s = silhouetteFromDistances(distances, n, assignments, k);
    if (s !== null && Number.isFinite(s)) scores.push({ k, score: s });
    await yieldToEventLoop();
  }
  if (!scores.length) return { k: 1, silhouette: null, scores };

  const best = Math.max(...scores.map((s) => s.score));
  const delta = Math.abs(best) * tolerance;
  const within = scores.filter((s) => s.score >= best - delta);
  const chosen = within.reduce((a, b) => (b.k > a.k ? b : a));
  return { k: chosen.k, silhouette: chosen.score, scores };
}
