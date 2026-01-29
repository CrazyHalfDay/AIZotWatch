"""LLM-based paper domain classifier for semantic categorization."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from zotwatch.core.models import CandidateWork, RankedWork

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Default research domains for geoscience focus
DEFAULT_DOMAINS = [
    "地球物理学",
    "地质学",
    "地震学",
    "大地测量学",
    "遥感与GIS",
    "水文地质学",
    "海洋地质",
    "古气候与古环境",
    "地球化学",
    "矿物学与岩石学",
    "构造地质学",
    "自然灾害",
    "环境科学",
    "大气科学",
    "其他",
]

DOMAIN_CLASSIFICATION_PROMPT = """请根据论文标题和摘要，将这些论文分类到相应的研究领域。

可选领域列表：
{domains_list}

论文列表：
{papers_list}

请返回 JSON 格式：
{{
  "classifications": [
    {{"id": "paper_id_1", "domain": "领域名称", "confidence": 0.9}},
    {{"id": "paper_id_2", "domain": "领域名称", "confidence": 0.8}},
    ...
  ]
}}

分类规则：
1. 每篇论文只能分配一个主要领域
2. confidence 表示分类置信度 (0.0-1.0)
3. 如果论文明显不属于任何领域，使用"其他"
4. 优先根据研究方法和主题，而非应用场景来分类
5. 只返回 JSON，不要添加任何额外文字"""


class PaperDomainClassifier:
    """Classify papers into research domains using LLM.

    Supports batch classification with concurrent processing for efficiency.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        model: str | None = None,
        domains: list[str] | None = None,
        batch_size: int = 20,
        max_workers: int = 3,
    ):
        """Initialize the domain classifier.

        Args:
            llm: LLM provider instance.
            model: Optional model name to use.
            domains: Custom domain list. Uses DEFAULT_DOMAINS if not provided.
            batch_size: Number of papers per LLM call.
            max_workers: Maximum concurrent LLM calls.
        """
        self.llm = llm
        self.model = model
        self.domains = domains or DEFAULT_DOMAINS
        self.batch_size = batch_size
        self.max_workers = max_workers

    def classify_papers(
        self,
        papers: list[RankedWork] | list[CandidateWork],
    ) -> dict[str, str]:
        """Classify papers into research domains.

        Args:
            papers: List of papers to classify.

        Returns:
            Dict mapping paper identifier to domain name.
        """
        if not papers:
            return {}

        # Split into batches
        batches = [
            papers[i : i + self.batch_size]
            for i in range(0, len(papers), self.batch_size)
        ]

        results: dict[str, str] = {}
        total_batches = len(batches)

        logger.info(
            "Classifying %d papers into domains using %d batches",
            len(papers),
            total_batches,
        )

        if total_batches == 1:
            # Single batch, no need for concurrency
            batch_result = self._classify_batch(batches[0])
            results.update(batch_result)
        else:
            # Use concurrent processing for multiple batches
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._classify_batch, batch): i
                    for i, batch in enumerate(batches)
                }

                for future in as_completed(futures):
                    batch_idx = futures[future]
                    try:
                        batch_result = future.result()
                        results.update(batch_result)
                        logger.debug(
                            "Completed batch %d/%d (%d papers classified)",
                            batch_idx + 1,
                            total_batches,
                            len(batch_result),
                        )
                    except Exception as e:
                        logger.warning("Batch %d classification failed: %s", batch_idx, e)

        logger.info("Domain classification complete: %d papers classified", len(results))
        return results

    def _classify_batch(
        self,
        papers: list[RankedWork] | list[CandidateWork],
    ) -> dict[str, str]:
        """Classify a single batch of papers.

        Args:
            papers: Batch of papers to classify.

        Returns:
            Dict mapping paper identifier to domain name.
        """
        papers_list = self._format_papers(papers)
        domains_list = "\n".join(f"- {d}" for d in self.domains)

        prompt = DOMAIN_CLASSIFICATION_PROMPT.format(
            domains_list=domains_list,
            papers_list=papers_list,
        )

        try:
            response = self.llm.complete(prompt, model=self.model)
            return self._parse_response(response.content)
        except Exception as e:
            logger.warning("Domain classification batch failed: %s", e)
            return {}

    def _format_papers(
        self,
        papers: list[RankedWork] | list[CandidateWork],
    ) -> str:
        """Format papers for the classification prompt.

        Args:
            papers: Papers to format.

        Returns:
            Formatted string for the prompt.
        """
        lines = []
        for paper in papers:
            paper_id = paper.identifier
            title = paper.title or "无标题"
            abstract = (paper.abstract or "")[:500]  # Limit abstract length

            lines.append(f"ID: {paper_id}")
            lines.append(f"标题: {title}")
            if abstract:
                lines.append(f"摘要: {abstract}")
            lines.append("")

        return "\n".join(lines)

    def _parse_response(self, content: str) -> dict[str, str]:
        """Parse LLM response into classification results.

        Args:
            content: Raw LLM response content.

        Returns:
            Dict mapping paper identifier to domain name.
        """
        try:
            content = self._clean_json_response(content)
            data = json.loads(content)

            results = {}
            classifications = data.get("classifications", [])

            for item in classifications:
                paper_id = item.get("id")
                domain = item.get("domain")

                if paper_id and domain:
                    # Validate domain is in allowed list
                    if domain in self.domains:
                        results[paper_id] = domain
                    else:
                        # Map to closest domain or "其他"
                        results[paper_id] = self._find_closest_domain(domain)

            return results

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse classification response: %s", e)
            return {}

    def _find_closest_domain(self, domain: str) -> str:
        """Find the closest matching domain from the allowed list.

        Args:
            domain: Domain name to match.

        Returns:
            Closest matching domain or "其他".
        """
        domain_lower = domain.lower()

        for allowed in self.domains:
            if allowed.lower() in domain_lower or domain_lower in allowed.lower():
                return allowed

        return "其他"

    def _clean_json_response(self, content: str) -> str:
        """Clean JSON response by removing markdown formatting.

        Args:
            content: Raw response content.

        Returns:
            Cleaned JSON string.
        """
        content = content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
        return content


__all__ = ["PaperDomainClassifier", "DEFAULT_DOMAINS"]
