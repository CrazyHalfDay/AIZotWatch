// Backup model of createLlmClient: a failed call is retried on the backup,
// which then serves the following calls; the main model alone has no fallback.

import assert from "node:assert/strict";
import test from "node:test";

import { createLlmClient } from "../src/llm/llm.js";
import { mockHttp, noSleep } from "./helpers.js";

const reply = (text) => ({ choices: [{ message: { content: text } }] });

test("a failing main model hands over to the backup model", async () => {
  const http = mockHttp([
    ["https://main.example/v1/chat/completions", () => ({ status: 402, data: { error: { message: "Insufficient balance" } } })],
    ["https://backup.example/v1/chat/completions", (req) => reply(`backup:${JSON.parse(req.body).model}`)],
  ]);
  const cfg = { apiKey: "k1", baseURL: "https://main.example/v1", model: "m1", backup: { baseURL: "https://backup.example/v1", apiKey: "k2", model: "m2" } };
  const llm = createLlmClient(cfg, http, { retry: { retries: 0, ...noSleep } });
  assert.equal(await llm.chat("hi"), "backup:m2");
  assert.equal(await llm.chat("again"), "backup:m2");
  const main = http.calls.filter((c) => c.url.startsWith("https://main.example"));
  assert.equal(main.length, 1, "the main model is not asked again in this run");
  assert.equal(http.calls.at(-1).headers.Authorization, "Bearer k2");

  // Without a backup the error stays the main model's
  const alone = createLlmClient({ ...cfg, backup: { model: "" } }, http, { retry: { retries: 0, ...noSleep } });
  await assert.rejects(alone.chat("hi"), /HTTP 402/);
  // Both failing: one error naming both
  const both = createLlmClient({ ...cfg, backup: { model: "m3" } }, http, { retry: { retries: 0, ...noSleep } });
  await assert.rejects(both.chat("hi"), /HTTP 402.*备用大模型也失败/);
});
