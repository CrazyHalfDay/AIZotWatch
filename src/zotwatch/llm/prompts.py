"""Prompt templates for LLM summarization."""

BULLET_SUMMARY_PROMPT = """请分析以下学术论文并提供简明扼要的摘要。

论文标题：{title}
摘要：{abstract}
作者：{authors}
期刊/会议：{venue}

请提供以下 5 个要点的总结：
1. **研究问题**：本文解决什么问题？
2. **研究方法**：采用了什么方法或技术？
3. **主要发现**：核心研究结果是什么？
4. **创新点**：本文的创新之处在哪里？
5. **相关性说明**：为什么这项研究对相关领域的研究者有价值？

请以 JSON 格式返回，包含以下键：research_question, methodology, key_findings, innovation, relevance_note
每个值应为简洁的单句话（不超过 50 字）。

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。"""

DETAILED_ANALYSIS_PROMPT = """请对以下学术论文进行详细分析。

论文标题：{title}
摘要：{abstract}
作者：{authors}
期刊/会议：{venue}

请撰写全面的分析，涵盖以下方面：

1. **研究背景**：本研究的背景和动机（2-3 句话）
2. **方法详情**：研究方法的详细说明（3-4 句话）
3. **研究结果**：主要发现及其意义（3-4 句话）
4. **局限性**：已知的局限性或潜在问题（2-3 句话）
5. **未来方向**：可能的后续研究方向（1-2 句话）
6. **研究相关性**：为什么这篇论文可能与研究类似课题的研究者相关

请以 JSON 格式返回，包含以下键：background, methodology_details, results, limitations, future_directions, relevance_to_interests

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。"""

INTEREST_REFINEMENT_PROMPT = """You are an academic research assistant. Based on the user's research interest description, generate an optimized search query for finding relevant academic papers.

User's research interests:
{user_interests}

Please output a JSON object with the following fields:
{{
  "refined_query": "A refined English query for retrieving relevant papers (50-100 words, covering key research topics, methods, and applications)",
  "include_keywords": ["keyword1", "keyword2", ...],
  "exclude_keywords": ["exclude1", "exclude2", ...]
}}

Guidelines:
- The refined_query should be a comprehensive English description suitable for semantic search
- include_keywords should contain 5-10 important technical terms
- exclude_keywords should contain terms the user explicitly wants to avoid
- If no exclusions are mentioned, return an empty array for exclude_keywords

Important: Only return the JSON object, no additional text or markdown formatting."""

OVERALL_SUMMARY_PROMPT = """请根据以下学术论文列表，按研究主题进行分组并撰写总结。

论文数量：{paper_count}
论文类型：{section_type}

论文列表：
{papers_list}

请完成以下任务：
1. 根据论文内容，将论文分成若干个研究主题（数量由你根据内容自动判断）
2. 撰写一句概述性的开头，说明各主题的论文数量分布
3. 对每个主题，用 1-2 句话描述该主题下论文的关键研究要点

请以 JSON 格式返回：
{{
  "overview": "本期推荐涵盖了X篇关于主题A的论文、Y篇关于主题B的论文...",
  "topics": [
    {{
      "topic_name": "主题名称（2-6字）",
      "paper_count": 数量,
      "description": "1-2句话描述该主题的研究重点和关键方法"
    }}
  ]
}}

注意：
- 主题分组应覆盖所有论文
- overview 应简洁说明各主题论文数量分布
- description 应具体描述研究内容，而非泛泛而谈

重要：只返回 JSON 对象，不要添加任何额外文字或 markdown 格式。"""


__all__ = [
    "BULLET_SUMMARY_PROMPT",
    "DETAILED_ANALYSIS_PROMPT",
    "INTEREST_REFINEMENT_PROMPT",
    "OVERALL_SUMMARY_PROMPT",
]
