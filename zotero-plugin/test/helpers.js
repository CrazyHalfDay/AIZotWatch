import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

export function fixture(name) {
  return readFileSync(join(here, "fixtures", name), "utf8");
}

/**
 * Mock transport. `routes` is a list of [matcher, handler]; a matcher is a
 * string (URL prefix) or a function(req). Handlers return { status, data }
 * or plain data (status 200). Every request is recorded in `calls`.
 */
export function mockHttp(routes) {
  const calls = [];
  return {
    calls,
    async request(req) {
      calls.push(req);
      for (const [matcher, handler] of routes) {
        const ok = typeof matcher === "string" ? req.url.startsWith(matcher) : matcher(req);
        if (!ok) continue;
        const res = await handler(req);
        if (res && typeof res === "object" && "status" in res && ("data" in res || res.status !== 200)) return res;
        return { status: 200, data: res };
      }
      return { status: 404, data: { message: `no mock for ${req.url}` } };
    },
  };
}

export const noSleep = { sleep: async () => {} };

/** Fake OpenAI-compatible embedding endpoint based on keyword topics (deterministic). */
export const TOPICS = ["mantle", "subduction", "copper", "isotope", "sulfur", "battery", "neuron", "galaxy"];

export function topicVector(text) {
  const lower = text.toLowerCase();
  const v = TOPICS.map((t) => (lower.split(t).length - 1) + 0.01);
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
}

export function embeddingHandler(counter) {
  return (req) => {
    const body = JSON.parse(req.body);
    counter.batches = (counter.batches || 0) + 1;
    counter.texts = (counter.texts || 0) + body.input.length;
    return { data: body.input.map((t, index) => ({ index, embedding: topicVector(t) })).reverse() };
  };
}
