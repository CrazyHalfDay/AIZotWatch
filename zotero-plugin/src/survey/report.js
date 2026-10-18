// Survey output: the numbered reference list, and the report rendered as
// Markdown (「导出 Markdown」) and as Zotero note HTML (「保存到 Zotero」). The
// report is structured data written by the LLM; it is rendered as text only:
// everything is escaped, and links are built here from DOIs and http(s) URLs.

import { formatAuthors, normalizeDoi } from "../core/text.js";
import { EXTRACTION_FIELDS, citedNumbers } from "./prompts.js";

export const AI_NOTICE = "本综述由 AI 根据所列文献的摘要、TLDR 和全文节选自动生成，可能存在遗漏或错误，引用前请核对原文。";
export const NO_LLM_NOTICE = "未启用大模型，因此没有生成 AI 综述；下面按语义聚类列出候选文献（组名为自动提取的关键词）。在设置中开启大模型后可以「重新生成综述」。";

/** Reference entry of a paper numbered `n`. */
export function referenceOf(p, n, theme = null) {
  return {
    n,
    id: p.id,
    title: p.title,
    authors: (p.authors || []).map((a) => (typeof a === "string" ? a : a?.name)).filter(Boolean),
    authorCount: p.authorCount || (p.authors || []).length,
    year: Number.isInteger(p.year) ? p.year : null,
    venue: p.venue || null,
    doi: normalizeDoi(p.doi) || null,
    url: p.url || (p.doi ? `https://doi.org/${normalizeDoi(p.doi)}` : null),
    inLibrary: !!p.inLibrary,
    itemKey: p.itemKey || null,
    theme,
  };
}

// Characters a URL may not hold as they are (old Wiley SICI DOIs contain "<" and ">"):
// percent-encoded, so that a Markdown <autolink> or an href stays one link
const UNSAFE_URL_CHARS = /[<>\s"`{}|\\^]/g;
const safeUrl = (url) => url.replace(UNSAFE_URL_CHARS, (c) => encodeURIComponent(c));

/** Link of a reference: its DOI, else an http(s) URL, else null (unsafe characters percent-encoded). */
export function referenceLink(ref) {
  const doi = normalizeDoi(ref.doi);
  if (doi) return safeUrl(`https://doi.org/${doi}`);
  return /^https?:\/\//i.test(ref.url || "") ? safeUrl(ref.url) : null;
}

/** "Wang L, Li M et al. Title. Venue, 2021." style citation text of a reference. */
export function referenceText(ref) {
  const names = ref.authors.length ? formatAuthors(ref.authors, 3) : "";
  const more = ref.authorCount > ref.authors.length && ref.authors.length <= 3 && !/et al\.$/.test(names) ? `${names} et al.` : names;
  const parts = [more, ref.title, [ref.venue, ref.year].filter(Boolean).join(", ")].map((p) => String(p || "").replace(/\.+$/, "")).filter(Boolean);
  return parts.length ? `${parts.join(". ")}.` : "";
}

/** Reference numbers the report cites (sections, summary and lists), in order of first use. */
export function reportCitations(report) {
  if (!report || report.kind !== "review") return [];
  const texts = [report.summary, ...(report.sections || []).flatMap((s) => s.paragraphs || []), ...(report.debates || []), ...(report.gaps || []), ...(report.future || [])];
  const out = [];
  for (const t of texts) for (const n of citedNumbers(t)) if (!out.includes(n)) out.push(n);
  return out;
}

const dateText = (ms) => {
  const d = new Date(ms);
  return Number.isFinite(d.getTime()) ? `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}` : "";
};

/** "生成于 2026-10-17 · 模型 deepseek-chat · 精读 40 篇 · 候选 386 篇". */
export function reportMetaLine(survey) {
  const report = survey.report || {};
  const counts = survey.counts || {};
  return [
    report.generatedAt ? `生成于 ${dateText(report.generatedAt)}` : "",
    report.model ? `模型 ${report.model}` : "",
    Number.isFinite(counts.read) && counts.read ? `精读 ${counts.read} 篇` : "",
    Number.isFinite(counts.candidates) ? `候选 ${counts.candidates} 篇` : "",
  ]
    .filter(Boolean)
    .join(" · ");
}

const paragraphsOf = (text) =>
  String(text || "")
    .split(/\n\s*\n|\n/)
    .map((p) => p.trim())
    .filter(Boolean);

// ---------------------------------------------------------------- Markdown

/** Escape Markdown control characters of plain text (the text stays readable). */
export function mdEscape(text) {
  return String(text ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/([*_`<>#|])/g, "\\$1")
    .replace(/\[(?!\d+(?:, \d+)*\])/g, "\\[")
    .replace(/^(\s*)([-+]|\d+\.)(\s)/gm, "$1\\$2$3");
}

/** The survey as a Markdown document (report, reading order, references). */
export function surveyMarkdown(survey) {
  const report = survey.report;
  const refs = survey.references || [];
  const lines = [];
  const title = report?.title || survey.input?.topic || "文献调研";
  lines.push(`# ${mdEscape(title)}`, "");
  const meta = reportMetaLine(survey);
  if (meta) lines.push(`> ${mdEscape(meta)}`, ">", `> ${mdEscape(report?.kind === "list" ? NO_LLM_NOTICE : AI_NOTICE)}`, "");
  lines.push(`**调研主题**：${mdEscape(survey.input?.topic || "")}`, "");
  if (report?.kind === "review") {
    if (report.summary) {
      lines.push("## 总述", "");
      for (const p of paragraphsOf(report.summary)) lines.push(mdEscape(p), "");
    }
    for (const s of report.sections || []) {
      lines.push(`## ${mdEscape(s.heading)}`, "");
      for (const p of s.paragraphs || []) lines.push(mdEscape(p), "");
    }
    for (const [key, heading] of [["debates", "主要争议"], ["gaps", "研究空白"], ["future", "未来方向"]]) {
      if (!report[key]?.length) continue;
      lines.push(`## ${heading}`, "", ...report[key].map((t) => `- ${mdEscape(t)}`), "");
    }
    if (report.reading?.length) {
      const byN = new Map(refs.map((r) => [r.n, r]));
      lines.push("## 推荐阅读顺序", "", ...report.reading.filter((n) => byN.has(n)).map((n, i) => `${i + 1}. [${n}] ${mdEscape(byN.get(n).title)}`), "");
    }
  } else if (report?.kind === "list") {
    for (const g of report.groups || []) {
      lines.push(`## ${mdEscape(g.label)}`, "");
      const byN = new Map(refs.map((r) => [r.n, r]));
      for (const n of g.refs || []) if (byN.has(n)) lines.push(`- [${n}] ${mdEscape(byN.get(n).title)}`);
      lines.push("");
    }
  }
  if (refs.length) {
    lines.push("## 参考文献", "");
    for (const r of refs) {
      const link = referenceLink(r);
      lines.push(`${r.n}. ${mdEscape(referenceText(r))}${link ? ` <${link}>` : ""}${r.inLibrary ? "（已在文库）" : ""}`);
    }
    lines.push("");
  }
  return `${lines.join("\n").replace(/\n{3,}/g, "\n\n").trim()}\n`;
}

/** File name for the Markdown export ("ZotWatch 调研 - <topic> - 2026-10-17.md", without unsafe characters). */
export function markdownFileName(survey, now = Date.now()) {
  const topic = String(survey.input?.topic || "调研").replace(/[\\/:*?"<>|\u0000-\u001f]+/g, " ").replace(/\s+/g, " ").trim();
  return `ZotWatch 调研 - ${Array.from(topic).slice(0, 40).join("")} - ${dateText(now)}.md`;
}

// --------------------------------------------------------------- note HTML

export function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * The survey as Zotero note HTML: headings, paragraphs and lists of escaped
 * text, numbered references with DOI links, the generation date and model,
 * and a line saying it was AI-generated.
 */
export function surveyNoteHtml(survey) {
  const report = survey.report;
  const refs = survey.references || [];
  const byN = new Map(refs.map((r) => [r.n, r]));
  const out = [];
  const p = (text) => out.push(`<p>${escapeHtml(text)}</p>`);
  out.push(`<h1>${escapeHtml(report?.title || survey.input?.topic || "文献调研")}</h1>`);
  const meta = reportMetaLine(survey);
  out.push(`<p><em>${escapeHtml(`ZotWatch 主题调研：${survey.input?.topic || ""}${meta ? ` · ${meta}` : ""}`)}</em></p>`);
  out.push(`<p><em>${escapeHtml(report?.kind === "list" ? NO_LLM_NOTICE : AI_NOTICE)}</em></p>`);
  if (report?.kind === "review") {
    if (report.summary) {
      out.push("<h2>总述</h2>");
      paragraphsOf(report.summary).forEach(p);
    }
    for (const s of report.sections || []) {
      out.push(`<h2>${escapeHtml(s.heading)}</h2>`);
      (s.paragraphs || []).forEach(p);
    }
    for (const [key, heading] of [["debates", "主要争议"], ["gaps", "研究空白"], ["future", "未来方向"]]) {
      if (!report[key]?.length) continue;
      out.push(`<h2>${heading}</h2>`, `<ul>${report[key].map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul>`);
    }
    const reading = (report.reading || []).filter((n) => byN.has(n));
    if (reading.length) {
      out.push("<h2>推荐阅读顺序</h2>", `<ol>${reading.map((n) => `<li>${escapeHtml(`[${n}] ${byN.get(n).title}`)}</li>`).join("")}</ol>`);
    }
  } else if (report?.kind === "list") {
    for (const g of report.groups || []) {
      out.push(`<h2>${escapeHtml(g.label)}</h2>`);
      out.push(`<ul>${(g.refs || []).filter((n) => byN.has(n)).map((n) => `<li>${escapeHtml(`[${n}] ${byN.get(n).title}`)}</li>`).join("")}</ul>`);
    }
  }
  if (refs.length) {
    out.push("<h2>参考文献</h2>");
    // Explicit numbers: a reference list may skip none, but the note editor renumbers <ol> on its own
    for (const r of refs) {
      const link = referenceLink(r);
      const linkHtml = link ? ` <a href="${escapeHtml(link)}">${escapeHtml(r.doi ? `doi:${r.doi}` : link)}</a>` : "";
      out.push(`<p>[${r.n}] ${escapeHtml(referenceText(r))}${linkHtml}${r.inLibrary ? " （已在文库）" : ""}</p>`);
    }
  }
  return out.join("\n");
}

/** Reading-note lines of a paper ("研究问题：…"), for UIs and exports. */
export function readingNoteLines(read) {
  if (!read) return [];
  return EXTRACTION_FIELDS.filter(([key]) => read[key]).map(([key, label]) => `${label}：${read[key]}`);
}
