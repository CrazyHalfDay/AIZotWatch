// Dense vector helpers. Matrices are row-major Float32Arrays with a known dim.

export function dot(a, b, aOffset = 0, bOffset = 0, dim = a.length) {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let i = 0;
  const end4 = dim - (dim % 4);
  for (; i < end4; i += 4) {
    s0 += a[aOffset + i] * b[bOffset + i];
    s1 += a[aOffset + i + 1] * b[bOffset + i + 1];
    s2 += a[aOffset + i + 2] * b[bOffset + i + 2];
    s3 += a[aOffset + i + 3] * b[bOffset + i + 3];
  }
  for (; i < dim; i++) s0 += a[aOffset + i] * b[bOffset + i];
  return s0 + s1 + s2 + s3;
}

/** L2-normalise in place and return the same array. */
export function normalizeInPlace(vec, offset = 0, dim = vec.length) {
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[offset + i] * vec[offset + i];
  norm = Math.sqrt(norm);
  if (norm > 1e-12) {
    for (let i = 0; i < dim; i++) vec[offset + i] /= norm;
  }
  return vec;
}

export function toFloat32(values) {
  return values instanceof Float32Array ? values : Float32Array.from(values);
}

export class Matrix {
  constructor(rows, dim, data) {
    this.rows = rows;
    this.dim = dim;
    this.data = data || new Float32Array(rows * dim);
    if (this.data.length !== rows * dim) {
      throw new Error(`Matrix data length ${this.data.length} != ${rows}x${dim}`);
    }
  }

  static fromRows(rows, dim) {
    const d = dim ?? (rows[0] ? rows[0].length : 0);
    const m = new Matrix(rows.length, d);
    rows.forEach((row, i) => {
      if (row.length !== d) throw new Error(`Row ${i} has dim ${row.length}, expected ${d}`);
      m.data.set(row, i * d);
    });
    return m;
  }

  row(i) {
    return this.data.subarray(i * this.dim, (i + 1) * this.dim);
  }

  normalizeRows() {
    for (let i = 0; i < this.rows; i++) normalizeInPlace(this.data, i * this.dim, this.dim);
    return this;
  }

  /** Dot products of `vec` against every row. */
  dotAll(vec, out = new Float32Array(this.rows)) {
    for (let i = 0; i < this.rows; i++) out[i] = dot(this.data, vec, i * this.dim, 0, this.dim);
    return out;
  }

  /** Top-k rows by dot product with `vec`: [{ index, score }] sorted descending. */
  topK(vec, k) {
    if (k <= 0) return [];
    const best = [];
    for (let i = 0; i < this.rows; i++) {
      const score = dot(this.data, vec, i * this.dim, 0, this.dim);
      if (best.length < k) {
        best.push({ index: i, score });
        if (best.length === k) best.sort((a, b) => b.score - a.score);
      } else if (score > best[k - 1].score) {
        let pos = k - 1;
        while (pos > 0 && best[pos - 1].score < score) {
          best[pos] = best[pos - 1];
          pos--;
        }
        best[pos] = { index: i, score };
      }
    }
    if (best.length < k) best.sort((a, b) => b.score - a.score);
    return best;
  }

  /**
   * Top-k rows for a block of query vectors at once. Each library row is read
   * once per block instead of once per query, which keeps the scan cache-friendly.
   * @returns {Array<Array<{index, score}>>}
   */
  topKBlock(queries, k) {
    const { rows, dim, data } = this;
    const q = queries.length;
    const best = queries.map(() => []);
    const sums = new Float64Array(q);
    for (let r = 0; r < rows; r++) {
      const off = r * dim;
      sums.fill(0);
      for (let j = 0; j < q; j++) {
        const vec = queries[j];
        let s0 = 0;
        let s1 = 0;
        let d = 0;
        for (; d + 1 < dim; d += 2) {
          s0 += data[off + d] * vec[d];
          s1 += data[off + d + 1] * vec[d + 1];
        }
        if (d < dim) s0 += data[off + d] * vec[d];
        sums[j] = s0 + s1;
      }
      for (let j = 0; j < q; j++) insertTopK(best[j], r, sums[j], k);
    }
    return best;
  }

  selectRows(indices) {
    const m = new Matrix(indices.length, this.dim);
    indices.forEach((idx, i) => m.data.set(this.row(idx), i * this.dim));
    return m;
  }
}

/** Insert into a descending top-k list in place. */
function insertTopK(list, index, score, k) {
  if (list.length === k && score <= list[k - 1].score) return;
  let pos = list.length < k ? list.length : k - 1;
  if (list.length < k) list.push(null);
  while (pos > 0 && list[pos - 1].score < score) {
    list[pos] = list[pos - 1];
    pos--;
  }
  list[pos] = { index, score };
}

/** Deterministic PRNG (mulberry32) so clustering is reproducible. */
export function createRng(seed = 42) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Yield to the event loop so long computations don't freeze the Zotero UI. */
export function yieldToEventLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** Float32Array <-> bytes, for binary persistence. */
export function float32ToBytes(arr) {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

export function bytesToFloat32(bytes) {
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(copy);
}
