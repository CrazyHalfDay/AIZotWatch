// Embedding providers. All return L2-normalised Float32Array vectors in input order.

import { mapWithConcurrency, postJson } from "../core/http.js";
import { normalizeInPlace } from "../core/vector.js";
import { createLocalEmbedder } from "./local.js";

export const EMBEDDING_PRESETS = {
  dashscope: {
    label: "阿里云 DashScope",
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: "text-embedding-v4",
    batchSize: 10,
    sendDimensions: true,
  },
  voyage: {
    label: "Voyage AI",
    baseURL: "https://api.voyageai.com/v1",
    model: "voyage-3.5",
    batchSize: 128,
    sendDimensions: false,
  },
  openai: {
    label: "OpenAI 兼容接口",
    baseURL: "https://api.openai.com/v1",
    model: "text-embedding-3-small",
    batchSize: 64,
    sendDimensions: false,
  },
};

// Long abstracts are truncated so batches stay well within token limits
const MAX_CHARS = 6000;

function prepare(text) {
  const t = (text || "").trim();
  return (t || "[untitled]").slice(0, MAX_CHARS);
}

function vectorsFromResponse(data, expected) {
  const items = Array.isArray(data?.data) ? data.data : null;
  if (!items || items.length !== expected) {
    throw new Error(`Embedding response has ${items ? items.length : "no"} vectors, expected ${expected}`);
  }
  const sorted = [...items].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  return sorted.map((d) => normalizeInPlace(Float32Array.from(d.embedding)));
}

/**
 * Create the embedder configured in settings.embedding.
 * @param {object} cfg - { provider, apiKey, model, baseURL, dimensions, batchSize, concurrency }
 */
export function createEmbedder(cfg, http, { retry = {} } = {}) {
  const provider = cfg.provider || "local";
  if (provider === "local") return createLocalEmbedder({ dim: 1024 });

  const preset = EMBEDDING_PRESETS[provider];
  if (!preset) throw new Error(`Unknown embedding provider: ${provider}`);
  if (!cfg.apiKey && provider !== "openai") {
    throw new Error(`请先在 ZotWatch 设置中填写 ${preset.label} 的 API Key`);
  }
  const model = cfg.model || preset.model;
  const baseURL = (cfg.baseURL || preset.baseURL).replace(/\/+$/, "");
  const batchSize = Math.max(1, cfg.batchSize || preset.batchSize);
  const dimensions = preset.sendDimensions ? cfg.dimensions || 1024 : undefined;
  const headers = cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {};

  async function embedBatch(batch, inputType) {
    const body = { model, input: batch.map(prepare) };
    // Voyage embeds queries (e.g. the flagship anchors) differently from documents
    if (provider === "voyage") body.input_type = inputType || "document";
    if (dimensions) {
      body.dimensions = dimensions;
      body.encoding_format = "float";
    }
    const data = await postJson(http, `${baseURL}/embeddings`, body, { headers, timeout: 120000, ...retry });
    return vectorsFromResponse(data, batch.length);
  }

  return {
    signature: `${provider}:${model}:${dimensions || "native"}`,
    dim: dimensions || null,
    cacheable: true,
    requiresFit: false,

    /**
     * @param {string[]} texts
     * @param {object} [options] - `onProgress(done, total)`; `inputType` "query" or "document" (default; Voyage only, others ignore it)
     */
    async embed(texts, { onProgress, inputType } = {}) {
      const batches = [];
      for (let i = 0; i < texts.length; i += batchSize) batches.push(texts.slice(i, i + batchSize));
      let done = 0;
      const results = await mapWithConcurrency(batches, Math.max(1, cfg.concurrency || 3), async (batch) => {
        const vectors = await embedBatch(batch, inputType);
        done += batch.length;
        onProgress?.(done, texts.length);
        return vectors;
      });
      return results.flat();
    },
  };
}
