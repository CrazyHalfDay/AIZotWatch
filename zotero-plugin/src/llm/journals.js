// Journal suggestions for 「从文库生成」: a port of the Python
// JournalRecommender (zotwatch/llm/journal_recommender.py) and its
// JOURNAL_GENERATION_PROMPT (zotwatch/llm/prompts.py). The model normalises
// the library's venue names and suggests related journals; it never supplies
// ISSNs or the impact factor used by the plugin: every title is verified
// through OpenAlex (sources/openalex-entities.js), which also provides the
// impact-factor-like metric, so all rows share one metric.

import { parseJsonReply } from "./llm.js";

/** Python JOURNAL_GENERATION_PROMPT, verbatim ({venues_list} and {research_focus} filled in). */
export const JOURNAL_GENERATION_PROMPT = `你是一位资深的学术期刊专家。下面是某位研究者的 Zotero 文献库中出现频率最高的期刊/会议列表（含收录篇数）：

{venues_list}

研究者的主要研究领域/关键词：
{research_focus}

请基于以上信息，生成一份"目标期刊监控名单"。要求：
1. 将上述期刊名称规范化为期刊的官方全称（展开缩写、纠正不规范写法）。
2. 在研究者已有期刊的基础上，**补充推荐**同领域内其他高水平、值得长期监控的权威期刊（即使文献库中尚未出现）。
3. 为每本期刊给出以下字段：
   - title: 期刊官方全称（英文期刊用英文全称，中文期刊用中文全称）
   - category: 该刊所属学科领域，用大写英文标识（如 GEOSCIENCES、GENERAL、REMOTE_SENSING 等）
   - impact_factor: 最近一年的影响因子（JCR）数值，给出你已知的最佳估计；若确实未知则填 null
   - is_chinese: 是否为中文核心期刊（true / false）
4. 排除明显不是学术期刊的条目（如个人笔记、技术报告、数据集等）。
5. 按 ISSN 去重，按学科相关度从高到低排序。

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式：
{
  "journals": [
    {"title": "Nature Geoscience", "category": "GEOSCIENCES", "impact_factor": 15.8, "is_chinese": false}
  ]
}
`;

/** The prompt for the library's top venues ({title, count}) and an optional research focus. */
export function journalGenerationPrompt(venues, researchFocus = "") {
  const list = venues.map((v) => `- ${v.title} [journal]: ${v.count} 篇`).join("\n");
  const focus = String(researchFocus || "").trim() || "未提供";
  // Replacer functions: titles may contain "$" patterns
  return JOURNAL_GENERATION_PROMPT.replace("{venues_list}", () => list).replace("{research_focus}", () => focus);
}

/**
 * Every balanced {...} JSON object in arbitrary text (Python
 * _extract_json_objects): recovers the complete entries of a reply wrapped in
 * prose or truncated mid-array.
 */
export function extractJsonObjects(text) {
  const objects = [];
  const stack = [];
  let inString = false;
  let escape = false;
  const source = String(text ?? "");
  for (let i = 0; i < source.length; i++) {
    const ch = source[i];
    if (inString) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") stack.push(i);
    else if (ch === "}" && stack.length) {
      const start = stack.pop();
      try {
        const value = JSON.parse(source.slice(start, i + 1));
        if (value && typeof value === "object" && !Array.isArray(value)) objects.push(value);
      } catch {
        // not a complete object
      }
    }
  }
  return objects;
}

/** Upper-case identifier form of a category ("remote sensing" -> "REMOTE_SENSING"). */
function categoryOf(value) {
  return typeof value === "string"
    ? value
        .trim()
        .toUpperCase()
        .replace(/[^\p{L}\p{N}()]+/gu, "_")
        .replace(/^_+|_+$/g, "")
    : "";
}

/**
 * Journals of a reply: [{title, category, impactFactor, isChinese}] (the
 * model's impact factor is kept for reference only). Never throws: a reply
 * without any usable entry yields [].
 */
export function parseJournalReply(content) {
  let entries = [];
  try {
    const parsed = parseJsonReply(content);
    if (Array.isArray(parsed?.journals)) entries = parsed.journals;
    else if (Array.isArray(parsed)) entries = parsed;
  } catch {
    entries = [];
  }
  if (!entries.length) entries = extractJsonObjects(content).filter((o) => "title" in o);
  const out = [];
  const seen = new Set();
  for (const entry of entries) {
    if (!entry || typeof entry !== "object") continue;
    const title = typeof entry.title === "string" ? entry.title.replace(/\s+/g, " ").trim() : "";
    if (!title || seen.has(title.toLowerCase())) continue;
    seen.add(title.toLowerCase());
    const ifValue = entry.impact_factor === null || entry.impact_factor === undefined || entry.impact_factor === "" ? NaN : Number(entry.impact_factor);
    out.push({
      title,
      category: categoryOf(entry.category),
      impactFactor: Number.isFinite(ifValue) ? ifValue : null,
      isChinese: entry.is_chinese === true || entry.is_chinese === "true",
    });
  }
  return out;
}

/** Ask the model for the journal list (journal lists are long: a 4096-token budget, like Python). */
export async function suggestJournalsWithLlm(llm, venues, researchFocus = "") {
  if (!venues.length) return [];
  const reply = await llm.chat(journalGenerationPrompt(venues, researchFocus), { maxTokens: 4096, temperature: 0.2 });
  return parseJournalReply(reply);
}
