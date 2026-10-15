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

export function createLlmClient(cfg, http, { retry = {} } = {}) {
  if (!cfg.apiKey) throw new Error("请先在 ZotWatch 设置中填写大模型 API Key");
  const baseURL = (cfg.baseURL || LLM_PRESETS.openrouter).replace(/\/+$/, "");
  const reasoning = /reasoner|thinking|-r1\b/i.test(cfg.model || "");

  async function chat(prompt, { maxTokens = cfg.maxTokens || 2048, temperature = cfg.temperature ?? 0.3 } = {}) {
    const body = {
      model: cfg.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
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

  return { chat };
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
