// OpenAI-compatible chat client (OpenRouter, DeepSeek, Kimi, DashScope, ...)
// plus the title translation and paper summary tasks.

import { postJson } from "../core/http.js";
import { formatAuthors } from "../core/text.js";

export const LLM_PRESETS = {
  openrouter: "https://openrouter.ai/api/v1",
  deepseek: "https://api.deepseek.com/v1",
  kimi: "https://api.moonshot.cn/v1",
  dashscope: "https://dashscope.aliyuncs.com/compatible-mode/v1",
};

export const SUMMARY_FIELDS = [
  ["research_question", "研究问题"],
  ["methodology", "研究方法"],
  ["key_findings", "主要发现"],
  ["innovation", "创新点"],
  ["relevance_note", "相关性"],
];

/** Extract the first JSON object/array from an LLM reply (tolerates ```json fences and chatter). */
export function parseJsonReply(text) {
  if (!text) throw new Error("Empty LLM reply");
  const cleaned = String(text)
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/```(?:json)?/gi, "")
    .trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    // fall through to bracket matching
  }
  const start = cleaned.search(/[[{]/);
  if (start < 0) throw new Error("No JSON found in LLM reply");
  const open = cleaned[start];
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  for (let i = start; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (inString) {
      if (ch === "\\") i++;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === open) depth++;
    else if (ch === close && --depth === 0) return JSON.parse(cleaned.slice(start, i + 1));
  }
  throw new Error("Unterminated JSON in LLM reply");
}

/**
 * Chat client of the configured model. With a backup model (`cfg.backup.model`;
 * its interface address and API key default to the main model's), a call that
 * fails on one model is retried on the other, and the model that answered
 * serves the following calls of this client (one run).
 */
export function createLlmClient(cfg, http, { retry = {} } = {}) {
  if (!cfg.apiKey) throw new Error("请先在 ZotWatch 设置中填写大模型 API Key");
  const main = chatClient(cfg, http, retry);
  const backup = cfg.backup?.model ? { ...cfg, baseURL: cfg.backup.baseURL || cfg.baseURL, apiKey: cfg.backup.apiKey || cfg.apiKey, model: cfg.backup.model } : null;
  if (!backup) return { chat: main };
  const clients = [main, chatClient(backup, http, retry)];
  const names = ["主模型", "备用大模型"];
  let current = 0;

  async function chat(prompt, options) {
    try {
      return await clients[current](prompt, options);
    } catch (e) {
      if (e?.cancelled) throw e;
      const other = 1 - current;
      try {
        const reply = await clients[other](prompt, options);
        current = other;
        return reply;
      } catch (e2) {
        if (e2?.cancelled) throw e2;
        throw new Error(`${e?.message || e}（${names[other]}也失败：${e2?.message || e2}）`);
      }
    }
  }

  return { chat };
}

/** Chat function of one OpenAI-compatible model. */
function chatClient(cfg, http, retry) {
  const baseURL = (cfg.baseURL || LLM_PRESETS.openrouter).replace(/\/+$/, "");
  const reasoning = /reasoner|thinking|-r1\b/i.test(cfg.model || "");
  // Reasoning tokens count against max_tokens: a small budget would end the
  // reply before any content (Python DeepSeekClient / KimiClient minimums)
  const minTokens = reasoning ? (/thinking/i.test(cfg.model || "") ? 16000 : 8192) : 0;

  async function chat(prompt, { maxTokens = cfg.maxTokens || 2048, temperature = cfg.temperature ?? 0.3 } = {}) {
    const body = {
      model: cfg.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: Math.max(maxTokens, minTokens),
    };
    if (!reasoning) body.temperature = temperature;
    const data = await postJson(http, `${baseURL}/chat/completions`, body, {
      headers: {
        Authorization: `Bearer ${cfg.apiKey}`,
        "HTTP-Referer": "https://github.com/CrazyHalfDay/AIZotWatch",
        "X-Title": "ZotWatch for Zotero",
      },
      timeout: reasoning ? 180000 : 90000,
      ...retry,
    });
    const content = data?.choices?.[0]?.message?.content;
    if (!content) throw new Error("LLM returned no content");
    return content;
  }

  return chat;
}

/** Translate titles to Chinese in batches; returns translations aligned with input. */
export async function translateTitles(llm, titles, { batchSize = 20 } = {}) {
  const out = new Array(titles.length).fill(null);
  for (let start = 0; start < titles.length; start += batchSize) {
    const batch = titles.slice(start, start + batchSize);
    const list = batch.map((t, i) => `${i + 1}. ${t}`).join("\n");
    const prompt =
      "请将以下英文学术论文标题准确翻译为简体中文，保留专业术语、化学式和缩写。\n" +
      `返回一个长度为 ${batch.length} 的 JSON 字符串数组，顺序与输入一致，不要添加任何其他文字。\n\n${list}`;
    try {
      const parsed = parseJsonReply(await llm.chat(prompt, { maxTokens: 4096, temperature: 0.1 }));
      const arr = Array.isArray(parsed) ? parsed : parsed.translations || parsed.titles || [];
      arr.slice(0, batch.length).forEach((t, i) => {
        if (typeof t === "string" && t.trim()) out[start + i] = t.trim();
      });
    } catch (e) {
      // Translation is best-effort; leave titles untranslated on failure, but never swallow a cancel
      if (e?.cancelled) throw e;
    }
  }
  return out;
}

/** Five-point structured summary, same prompt as the Python summarizer. */
export async function summarizePaper(llm, work) {
  const prompt = `请分析以下学术论文并提供简明扼要的摘要。

论文标题：${work.title}
摘要：${work.abstract || "（无摘要）"}
作者：${formatAuthors(work.authors, 8)}
期刊/会议：${work.venue || "未知"}

请提供以下 5 个要点的总结：
1. **研究问题**：本文解决什么问题？
2. **研究方法**：采用了什么方法或技术？
3. **主要发现**：核心研究结果是什么？
4. **创新点**：本文的创新之处在哪里？
5. **相关性说明**：为什么这项研究对相关领域的研究者有价值？

请以 JSON 格式返回，包含以下键：research_question, methodology, key_findings, innovation, relevance_note
每个值应为简洁的单句话（不超过 50 字）。

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。`;
  const parsed = parseJsonReply(await llm.chat(prompt, { maxTokens: 1024 }));
  const summary = {};
  for (const [key] of SUMMARY_FIELDS) {
    if (typeof parsed[key] === "string" && parsed[key].trim()) summary[key] = parsed[key].trim();
  }
  if (!Object.keys(summary).length) throw new Error("LLM summary had none of the expected fields");
  return summary;
}

/**
 * Relevance prompt of the Python PaperRelevanceFilter (RELEVANCE_FILTER_PROMPT,
 * llm/prompts.py), verbatim, except that papers are named P1, P2, … instead of
 * by DOI: models echo short ids reliably but sometimes rewrite DOIs.
 */
export function relevancePrompt(boundary, works) {
  const papers = works
    .map((w, i) => `- P${i + 1}\n  Title: ${w.title || ""}\n  Abstract: ${w.abstract || "No abstract available"}`)
    .join("\n");
  return `You are an academic research assistant. Determine whether each paper is relevant to the user's research interests.

User research interests:
${boundary}

Papers:
${papers}

Return a JSON object with:
{
  "relevant_ids": ["paper_id_1", "paper_id_2", ...],
  "irrelevant_ids": ["paper_id_3", "paper_id_4", ...]
}

Rules:
- Use the title and abstract meaning (semantic relevance), not just keyword overlap.
- If a paper is borderline or uncertain, classify it as irrelevant.
- Return only JSON (no markdown, no extra text).
`;
}

/**
 * Indices (0-based) of the papers a relevance reply lists in `relevant_ids`
 * ("P3", "p3", 3 or "3"; unknown ids are ignored). Throws when the reply holds
 * no usable verdict.
 */
export function parseRelevantIds(reply, count) {
  const parsed = parseJsonReply(reply);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Relevance reply is not a JSON object");
  let ids = parsed.relevant_ids ?? [];
  if (typeof ids === "string") ids = ids.split(/[\s,;，；]+/);
  if (!Array.isArray(ids)) throw new Error("Relevance reply has no relevant_ids list");
  const out = new Set();
  for (const id of ids) {
    const match = String(id ?? "").trim().match(/^P?\s*(\d+)$/i);
    const n = match ? Number(match[1]) : 0;
    if (n >= 1 && n <= count) out.add(n - 1);
  }
  return out;
}

/**
 * Ask the LLM which works match `boundary` (the flagship gray zone), in
 * batches. A batch whose reply cannot be parsed, or whose request fails, is
 * kept whole (inclusive, like having no LLM); once the very first request
 * fails, the remaining batches are kept without asking (likely a
 * configuration problem). An empty boundary keeps everything (Python parity).
 * Cancellation propagates.
 * @returns {Promise<{kept: object[], judged: number, failedBatches: number, error: string|null, confirmed: Set<object>}>}
 *   kept in input order; `judged` counts works with a verdict; `confirmed` holds the works the LLM accepted
 */
export async function judgeRelevance(llm, works, boundary, { batchSize = 20 } = {}) {
  const result = { kept: [], judged: 0, failedBatches: 0, error: null, confirmed: new Set() };
  const text = String(boundary || "").trim();
  if (!text) {
    result.kept = [...works];
    return result;
  }
  const size = Math.max(1, Math.floor(batchSize) || 1);
  let answered = false;
  let stopped = false;
  for (let start = 0; start < works.length; start += size) {
    const batch = works.slice(start, start + size);
    const keepBatch = (message) => {
      result.kept.push(...batch);
      result.failedBatches++;
      result.error ||= message;
    };
    if (stopped) {
      keepBatch(null);
      continue;
    }
    let reply;
    try {
      reply = await llm.chat(relevancePrompt(text, batch), { maxTokens: 1024, temperature: 0.3 });
    } catch (e) {
      if (e?.cancelled) throw e;
      keepBatch(e?.message || String(e));
      if (!answered) stopped = true;
      continue;
    }
    answered = true;
    let relevant;
    try {
      relevant = parseRelevantIds(reply, batch.length);
    } catch (e) {
      keepBatch(`无法解析大模型的回复（${e.message}）`);
      continue;
    }
    result.judged += batch.length;
    batch.forEach((w, i) => {
      if (!relevant.has(i)) return;
      result.kept.push(w);
      result.confirmed.add(w);
    });
  }
  return result;
}

// Longest topic label kept (the card chip and note line stay short)
export const MAX_TOPIC_LABEL_CHARS = 30;

/**
 * Batch label prompt of the Python ClusterLabeler (BATCH_CLUSTER_LABEL_PROMPT,
 * llm/prompts.py, and the cluster blocks of label_clusters_batch), verbatim:
 * up to 3 representative titles and 5 keywords per cluster.
 * @param {{memberCount: number, titles?: string[], keywords?: string[]}[]} clusters
 */
export function clusterLabelPrompt(clusters) {
  const info = clusters
    .map((c, i) => {
      const titles = (c.titles || []).slice(0, 3).map((t) => `"${t}"`).join(", ");
      const keywords = c.keywords?.length ? c.keywords.slice(0, 5).join(", ") : "N/A";
      return `Cluster ${i + 1} (${c.memberCount} papers):\n  Titles: ${titles}\n  Keywords: ${keywords}`;
    })
    .join("\n\n");
  return `基于以下多个论文簇，为每个簇生成一个简洁的研究方向标签（仅中文，不超过 10 个词）。

## Clusters:
${info}

## Instructions:
1. 为每个簇识别共同的研究主题或方向
2. 用中文生成能概括每个簇核心的简洁标签
3. 适当使用领域术语，保持自然流畅；常用英文缩写可直接使用（如 LLM, SAR 等）
4. 按输入顺序返回 JSON 数组，元素为中文标签字符串

## Output（JSON array of strings, 中文标签）:
`;
}

/**
 * Labels from a batch label reply. Unlike Python (which pads with "Cluster N"
 * and parses stray lines), only a JSON array of exactly `count` non-empty
 * strings is accepted, so a bad reply is retried later instead of being
 * stored as a label. Surrounding quotes are trimmed and labels capped.
 * @throws {Error} with a Chinese message for the run's error list
 */
export function parseClusterLabels(reply, count) {
  let parsed;
  try {
    parsed = parseJsonReply(reply);
  } catch (e) {
    throw new Error(`无法解析大模型的回复（${e.message}）`);
  }
  if (!Array.isArray(parsed)) throw new Error("大模型的回复不是 JSON 数组");
  if (parsed.length !== count) throw new Error(`大模型返回了 ${parsed.length} 个标签，应为 ${count} 个`);
  return parsed.map((value, i) => {
    const label = typeof value === "string" ? value.replace(/\s+/g, " ").replace(/^[\s"'“”‘’]+|[\s"'“”‘’]+$/g, "") : "";
    if (!label) throw new Error(`第 ${i + 1} 个标签为空`);
    return Array.from(label).slice(0, MAX_TOPIC_LABEL_CHARS).join("").trim();
  });
}

/** One batch LLM call labelling every given cluster (aligned with input); throws on a bad reply. */
export async function labelClusters(llm, clusters) {
  if (!clusters.length) return [];
  const reply = await llm.chat(clusterLabelPrompt(clusters), { maxTokens: 1024 });
  return parseClusterLabels(reply, clusters.length);
}
