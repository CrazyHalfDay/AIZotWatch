// LLM prompts of the topic survey and the parsers of their replies. Every
// prompt asks for JSON only, Chinese output (search queries excepted: the
// databases are English) and forbids knowledge or references beyond the
// papers it lists; the parsers never trust the reply: odd shapes are rejected
// or trimmed, and citations are checked against the numbers the prompt gave
// (validateCitations).

import { collapseWhitespace } from "../core/text.js";
import { extractJsonObjects } from "../llm/journals.js";
import { parseJsonReply } from "../llm/llm.js";

/** Characters per extracted field (the design's limit; longer values are cut with "…"). */
export const MAX_FIELD_CHARS = 60;
/** Full-text excerpt given to the reading prompt. */
export const MAX_EXCERPT_CHARS = 6000;
const MAX_ABSTRACT_CHARS = 3000;
const MAX_SEED_ABSTRACT_CHARS = 600;
const MAX_QUERY_CHARS = 120;
const MAX_LABEL_CHARS = 20;
const MAX_THEME_SUMMARY_CHARS = 100;
const MAX_LIST_ITEMS = 6;
const MAX_READING = 12;

/** Fields of a reading note, with their labels (UI, note, Markdown). */
export const EXTRACTION_FIELDS = Object.freeze([
  ["question", "研究问题"],
  ["method", "研究方法"],
  ["data", "数据与样本"],
  ["findings", "主要发现"],
  ["contribution", "贡献"],
  ["limitations", "局限"],
]);

const chars = (text) => Array.from(String(text ?? ""));
/** `text` cut to `max` characters (code points), "…" marking the cut. */
export function clip(text, max) {
  const list = chars(text);
  return list.length > max ? `${list.slice(0, max - 1).join("")}…` : list.join("");
}
const str = (value) => (typeof value === "string" ? collapseWhitespace(value) : "");
const yearOf = (p) => (Number.isInteger(p.year) ? String(p.year) : "年份不详");

// ------------------------------------------------------------------ queries

/**
 * Stage 2: English search queries and key terms from the topic, the
 * description, the keywords and the seed papers.
 * @param {object} input - normalised survey input
 * @param {{title, year, abstract, tags}[]} seeds
 */
export function queryPrompt(input, seeds = []) {
  const seedLines = seeds.length
    ? seeds
        .map((s, i) => {
          const tags = (s.tags || []).slice(0, 8).join("; ");
          const abstract = s.abstract ? `\n   摘要：${clip(collapseWhitespace(s.abstract), MAX_SEED_ABSTRACT_CHARS)}` : "";
          return `${i + 1}. ${s.title}（${yearOf(s)}）${tags ? `\n   标签：${tags}` : ""}${abstract}`;
        })
        .join("\n")
    : "（无）";
  return `你是一位资深的科研文献检索专家。请为下面的文献调研设计检索式，用于在 OpenAlex、Semantic Scholar 等英文学术数据库中查找相关论文。

调研主题：${input.topic}
补充描述：${input.description || "（无）"}
用户关键词：${input.keywords.length ? input.keywords.join("; ") : "（无）"}
种子文献（用户认为与主题最相关的论文）：
${seedLines}

要求：
1. 给出 4–8 条简洁的英文检索式，每条 2–6 个词，覆盖主题的核心概念、研究对象、主要方法和常见的同义说法或缩写；不要使用布尔运算符、引号、通配符或字段限定。
2. 各条检索式应相互补充，不要只是同一说法的轻微改写；中文主题请先译为规范的英文学术术语。
3. 另给出 5–12 个能概括该主题的英文关键术语（terms）。
4. 只返回 JSON 对象，不要添加任何其他文字或 markdown 格式：
{"queries": ["...", "..."], "terms": ["...", "..."]}
`;
}

/** A search query as the databases take it: no quotes, boolean operators or field prefixes. */
export function cleanQuery(value) {
  const text = str(value)
    .replace(/["“”'‘’`]/g, " ")
    .replace(/\b(AND|OR|NOT)\b/g, " ")
    .replace(/[()[\]{}*?:|]+/g, " ");
  const q = collapseWhitespace(text);
  return q.length >= 2 ? clip(q, MAX_QUERY_CHARS) : "";
}

/** {queries, terms} from a query reply (an object, or a bare array of queries); throws when it has no query. */
export function parseQueryReply(reply, { maxQueries = 8 } = {}) {
  const parsed = parseJsonReply(reply);
  const rawQueries = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.queries) ? parsed.queries : [];
  const rawTerms = !Array.isArray(parsed) && Array.isArray(parsed?.terms) ? parsed.terms : [];
  const distinct = (list, clean, max) => {
    const seen = new Set();
    const out = [];
    for (const v of list) {
      const t = clean(v);
      if (!t || seen.has(t.toLowerCase())) continue;
      seen.add(t.toLowerCase());
      out.push(t);
      if (out.length >= max) break;
    }
    return out;
  };
  const queries = distinct(rawQueries, cleanQuery, maxQueries);
  if (!queries.length) throw new Error("大模型没有给出可用的检索式");
  return { queries, terms: distinct(rawTerms, (v) => clip(str(v), 60), 12) };
}

// -------------------------------------------------------------- extraction

/**
 * Stage 7: reading note of one paper, in Chinese, from the text given only.
 * @param {object} input - survey input (topic, description)
 * @param {object} paper - {title, year, venue, abstract, tldr}
 * @param {string} [excerpt] - full-text excerpt (≤ MAX_EXCERPT_CHARS)
 */
export function extractionPrompt(input, paper, excerpt = "") {
  const abstract = paper.abstract ? clip(collapseWhitespace(paper.abstract), MAX_ABSTRACT_CHARS) : "（无）";
  const text = excerpt ? clip(String(excerpt).trim(), MAX_EXCERPT_CHARS) : "";
  return `你是一位严谨的科研助理，正在为主题「${input.topic}」的文献综述做精读笔记。请只根据下面提供的论文信息提取要点：不要使用论文以外的知识，不要编造数据或结论；信息不足的字段填空字符串 ""。

调研主题：${input.topic}
主题说明：${input.description || "（无）"}

论文标题：${paper.title}
发表年份：${yearOf(paper)}
期刊/来源：${paper.venue || "未知"}
摘要：${abstract}${paper.tldr ? `\nTLDR：${paper.tldr}` : ""}${text ? `\n全文节选：\n${text}` : ""}

请用简体中文填写以下字段（专业术语可保留英文或缩写），每个字段不超过 ${MAX_FIELD_CHARS} 个字：
- question：研究问题
- method：研究方法
- data：数据与样本（研究区、样品、数据集等）
- findings：主要发现
- contribution：主要贡献或创新
- limitations：局限性（论文未提及时填 ""）
- relevance：与调研主题的相关度，1–5 的整数（5 = 直接研究该主题，1 = 基本无关）

只返回 JSON 对象，不要添加任何其他文字或 markdown 格式：
{"question": "", "method": "", "data": "", "findings": "", "contribution": "", "limitations": "", "relevance": 3}
`;
}

/**
 * A reading note from a reply: the six text fields (cut to MAX_FIELD_CHARS)
 * and `relevance` (1–5, or null). Throws when neither the question nor the
 * findings were extracted.
 */
export function parseExtraction(reply) {
  let parsed;
  try {
    parsed = parseJsonReply(reply);
  } catch (e) {
    parsed = extractJsonObjects(reply).find((o) => "findings" in o || "question" in o) || null;
    if (!parsed) throw new Error(`无法解析大模型的回复（${e.message}）`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("大模型的回复不是 JSON 对象");
  const note = {};
  for (const [key] of EXTRACTION_FIELDS) note[key] = clip(str(parsed[key]), MAX_FIELD_CHARS);
  if (!note.question && !note.findings) throw new Error("大模型没有提取出研究问题或主要发现");
  const r = Math.round(Number(parsed.relevance));
  note.relevance = Number.isFinite(r) && r >= 1 && r <= 5 ? r : null;
  return note;
}

// ------------------------------------------------------------------ themes

/**
 * Stage 8: one call naming every theme (a short Chinese label and one
 * sentence), from each theme's titles and findings.
 * @param {object} input - survey input
 * @param {{papers: object[]}[]} themes - papers carry `title` and their reading note `read`
 */
export function themePrompt(input, themes) {
  const blocks = themes
    .map((t, i) => {
      const lines = t.papers
        .slice(0, 8)
        .map((p) => `- ${p.title}${p.read?.findings ? `：${p.read.findings}` : ""}`)
        .join("\n");
      return `第 ${i + 1} 组（${t.papers.length} 篇）：\n${lines}`;
    })
    .join("\n\n");
  return `下面是围绕主题「${input.topic}」精读的论文按语义聚成的 ${themes.length} 组。请为每一组起一个简洁的中文主题名（不超过 15 个字，可保留常用英文缩写），并用一句话（不超过 60 个字）概括这一组论文共同关注的问题。只能依据所列论文，不要引入其他内容。

${blocks}

按输入顺序返回 JSON 数组，每组一个对象，不要添加任何其他文字或 markdown 格式：
[{"label": "主题名", "summary": "一句话概括"}]
`;
}

/** [{label, summary}] aligned with the themes; throws unless the reply names exactly `count` themes. */
export function parseThemeReply(reply, count) {
  let parsed;
  try {
    parsed = parseJsonReply(reply);
  } catch (e) {
    throw new Error(`无法解析大模型的回复（${e.message}）`);
  }
  const list = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.themes) ? parsed.themes : null;
  if (!list) throw new Error("大模型的回复不是 JSON 数组");
  if (list.length !== count) throw new Error(`大模型返回了 ${list.length} 个主题，应为 ${count} 个`);
  return list.map((entry, i) => {
    const label = clip(str(typeof entry === "string" ? entry : entry?.label).replace(/^["'“”‘’]+|["'“”‘’]+$/g, ""), MAX_LABEL_CHARS);
    if (!label) throw new Error(`第 ${i + 1} 个主题名为空`);
    return { label, summary: clip(str(entry?.summary), MAX_THEME_SUMMARY_CHARS) };
  });
}

// ----------------------------------------------------------------- writing

const NO_INVENTION =
  "只能依据下面列出的文献撰写，不得引入列表以外的文献、数据或结论，也不要编造参考文献；每个论断后用方括号标注其依据的文献编号，如 [3] 或 [3, 7]，只能使用列表中出现的编号。";

/** One line per paper with its reading note, for the writing prompts ("[3] Title（2021，Venue）"). */
function paperBlock(p) {
  const note = p.read
    ? EXTRACTION_FIELDS.filter(([key]) => p.read[key])
        .map(([key, label]) => `${label}：${p.read[key]}`)
        .join("；")
    : p.abstract
      ? `摘要：${clip(collapseWhitespace(p.abstract), 400)}`
      : "";
  return `[${p.n}] ${p.title}（${yearOf(p)}${p.venue ? `，${p.venue}` : ""}）${note ? `\n    ${note}` : ""}`;
}

/**
 * Stage 9, one call per theme: the section on that theme, citing only the
 * theme's papers by their reference numbers (`n`).
 * @param {object} input - survey input
 * @param {{label, summary}} theme
 * @param {object[]} papers - the theme's papers with `n` and `read`
 */
export function sectionPrompt(input, theme, papers) {
  return `你是一位资深学者，正在用中文撰写关于「${input.topic}」的学术综述。现在请撰写其中一节，主题为「${theme.label}」${theme.summary ? `（${theme.summary}）` : ""}。

调研主题说明：${input.description || "（无）"}

本节可以引用的全部文献如下，方括号中的数字是它们在综述参考文献列表中的编号：
${papers.map(paperBlock).join("\n")}

写作要求：
1. ${NO_INVENTION}
2. 比较和归纳不同研究的问题、方法与结论，指出共识与分歧，不要逐篇罗列摘要。
3. 使用简体中文和学术语体，专业术语可保留英文缩写；写 2–4 段，每段 120–300 字。
4. 只返回 JSON 对象，不要添加任何其他文字或 markdown 格式：
{"heading": "本节标题", "paragraphs": ["第一段……[${papers[0]?.n ?? 1}]", "第二段……"]}
`;
}

/** {heading, paragraphs} of a section reply (citations not yet checked); throws without any paragraph. */
export function parseSectionReply(reply, fallbackHeading = "") {
  let parsed;
  try {
    parsed = parseJsonReply(reply);
  } catch (e) {
    throw new Error(`无法解析大模型的回复（${e.message}）`);
  }
  const paragraphs = (Array.isArray(parsed?.paragraphs) ? parsed.paragraphs : typeof parsed?.paragraphs === "string" ? [parsed.paragraphs] : [])
    .map((p) => (typeof p === "string" ? p.trim() : ""))
    .filter(Boolean)
    .slice(0, 8);
  if (!paragraphs.length) throw new Error("大模型没有写出本节内容");
  return { heading: clip(str(parsed.heading), 40) || fallbackHeading, paragraphs };
}

/**
 * Stage 9, the final call: title, summary, debates, gaps, future directions
 * and a reading order, from the written sections and the numbered references.
 * @param {object} input - survey input
 * @param {{heading, paragraphs}[]} sections
 * @param {object[]} references - papers with `n`, `title`, `year` and `read`
 */
export function overviewPrompt(input, sections, references) {
  const sectionText = sections
    .map((s) => `### ${s.heading}\n${clip(s.paragraphs.join("\n"), 900)}`)
    .join("\n\n");
  const refs = references
    .map((p) => `[${p.n}] ${p.title}（${yearOf(p)}）${p.read?.contribution ? `——${p.read.contribution}` : ""}`)
    .join("\n");
  return `你是一位资深学者，正在用中文完成一篇关于「${input.topic}」的学术综述。各主题小节已经写好，下面给出各小节的内容和全部参考文献。请撰写综述的标题和总述，归纳主要争议、研究空白和未来方向，并给出推荐阅读顺序。

调研主题说明：${input.description || "（无）"}

各小节：
${sectionText || "（无）"}

参考文献（方括号中为编号）：
${refs}

要求：
1. ${NO_INVENTION}
2. title：综述标题，不超过 30 个字。
3. summary：总述，1–2 段（段间用空行分隔），概括该主题的研究现状、主要进展和整体格局，300–500 字。
4. debates：2–5 条主要争议或尚无定论的问题；gaps：2–5 条研究空白或不足；future：2–5 条未来研究方向的建议。每条一到两句，注明涉及的文献编号。
5. reading：推荐阅读顺序，按先后列出 5–10 个文献编号（数字）：先读综述性、奠基性的工作，再读具体进展。
6. 使用简体中文和学术语体。只返回 JSON 对象，不要添加任何其他文字或 markdown 格式：
{"title": "", "summary": "", "debates": [""], "gaps": [""], "future": [""], "reading": [1, 2]}
`;
}

/** The overview reply ({title, summary, debates, gaps, future, reading}); citations not yet checked. */
export function parseOverviewReply(reply) {
  let parsed;
  try {
    parsed = parseJsonReply(reply);
  } catch (e) {
    throw new Error(`无法解析大模型的回复（${e.message}）`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("大模型的回复不是 JSON 对象");
  const list = (value) =>
    (Array.isArray(value) ? value : typeof value === "string" && value.trim() ? [value] : [])
      .map((v) => (typeof v === "string" ? v.trim() : ""))
      .filter(Boolean)
      .slice(0, MAX_LIST_ITEMS);
  const summary = Array.isArray(parsed.summary) ? parsed.summary.filter((p) => typeof p === "string").join("\n\n") : typeof parsed.summary === "string" ? parsed.summary : "";
  const reading = (Array.isArray(parsed.reading) ? parsed.reading : [])
    .map((n) => Number((String(n).trim().match(/^\[?(\d{1,3})\]?$/) || [])[1]))
    .filter((n) => Number.isInteger(n) && n > 0);
  const out = { title: clip(str(parsed.title), 60), summary: summary.trim(), debates: list(parsed.debates), gaps: list(parsed.gaps), future: list(parsed.future), reading };
  if (!out.summary && !out.debates.length && !out.gaps.length && !out.future.length) throw new Error("大模型没有写出总述");
  return out;
}

// --------------------------------------------------------------- citations

// A bracketed group of reference numbers: [3], [3, 7], [3-5], 【3】, ［3，7］ (four digits, such as
// a year in brackets, are not a reference number and are left alone); adjacent groups form one run
const GROUP = String.raw`[[【［]\s*(\d{1,3}(?:\s*(?:[-–—~～]|[,，、;；]|and|和)\s*\d{1,3})*)\s*[\]】］]`;
const CITATION_GROUP = new RegExp(GROUP, "g");
const CITATION_RUN = new RegExp(`${GROUP}(?:[ \t]*${GROUP})*`, "g");
const MAX_RANGE = 12;

/** Reference numbers of one group's body ("3, 5-7" -> 3, 5, 6, 7). */
function groupNumbers(body) {
  const numbers = [];
  for (const part of body.split(/\s*(?:[,，、;；]|and|和)\s*/)) {
    const range = part.match(/^(\d+)\s*[-–—~～]\s*(\d+)$/);
    if (range) {
      const [a, b] = [Number(range[1]), Number(range[2])];
      if (b >= a && b - a < MAX_RANGE) for (let n = a; n <= b; n++) numbers.push(n);
    } else if (/^\d+$/.test(part)) numbers.push(Number(part));
  }
  return numbers;
}

/**
 * Check the [n] citations of a text against the numbers it may cite: invalid
 * numbers are dropped (a group without any valid number disappears), ranges
 * are expanded (up to MAX_RANGE numbers), and every group, or run of adjacent
 * groups ("[3][5]"), is rewritten as one "[3, 5]". Non-numeric brackets are left alone.
 * @param {string} text
 * @param {Set<number>} allowed
 * @returns {{text: string, cited: number[]}} cited numbers in order of first use
 */
export function validateCitations(text, allowed) {
  const cited = [];
  const out = String(text ?? "").replace(CITATION_RUN, (run) => {
    const numbers = [...run.matchAll(CITATION_GROUP)].flatMap((m) => groupNumbers(m[1]));
    const valid = [...new Set(numbers.filter((n) => allowed.has(n)))];
    for (const n of valid) if (!cited.includes(n)) cited.push(n);
    return valid.length ? `[${valid.join(", ")}]` : "";
  });
  // Removed groups leave spaces before punctuation
  const tidy = out.replace(/[ \t]+([，。；：、,.;:！？!?）)])/g, "$1").replace(/[ \t]{2,}/g, " ").trim();
  return { text: tidy, cited };
}

/** Unique [n] numbers in a text (already validated), in order of first use. */
export function citedNumbers(text) {
  const out = [];
  for (const m of String(text ?? "").matchAll(/\[(\d+(?:, \d+)*)\]/g)) {
    for (const n of m[1].split(", ").map(Number)) if (!out.includes(n)) out.push(n);
  }
  return out;
}

export const UNVERIFIED_MARK = "（未引用文献，需核实）";

// ---------------------------------------------------------------- full text

// Headings of a paper's conclusions and of its reference list (full text of a PDF has one line per heading)
const CONCLUSIONS = /\n[ \t]*(?:\d+(?:\.\d+)*\.?[ \t]*)?(?:conclusions?|concluding remarks|summary and conclusions|discussion and conclusions?|结论|结语|结论与展望)[ \t]*\n/i;
const REFERENCES = /\n[ \t]*(?:references|bibliography|literature cited|参考文献)[ \t]*\n/gi;

/**
 * Excerpt of a paper's full text for the reading prompt, at most `max`
 * characters: the reference list is cut off, and a long text keeps its
 * beginning (abstract, introduction) and the start of its conclusions.
 */
export function fullTextExcerpt(text, max = MAX_EXCERPT_CHARS) {
  let body = String(text ?? "");
  let refAt = -1;
  for (const m of body.matchAll(REFERENCES)) refAt = m.index;
  if (refAt > body.length * 0.5) body = body.slice(0, refAt);
  const clean = (s) => s.replace(/\s+/g, " ").trim();
  const all = clean(body);
  if (all.length <= max) return all;
  const conclusionAt = body.search(CONCLUSIONS);
  if (conclusionAt <= body.length * 0.3) return `${all.slice(0, max - 1)}…`;
  const headLength = Math.floor(max * 0.7);
  const tail = clean(body.slice(conclusionAt)).slice(0, max - headLength - 3);
  return `${all.slice(0, headLength)} … ${tail}`;
}
