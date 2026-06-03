"""Field-level geoscience gate for the flagship-journal track.

Articles from flagship journals are kept based on similarity to a field anchor
(solid earth + paleontology) rather than to the user's personal library, so a
high-value venue can surface all on-topic articles. A negative anchor filters
out atmospheric science, and gray-zone articles are optionally judged by an LLM.
"""

import logging

import numpy as np

from zotwatch.config.settings import ScoringConfig
from zotwatch.core.models import CandidateWork
from zotwatch.infrastructure.embedding.base import BaseEmbeddingProvider
from zotwatch.llm.base import BaseLLMProvider
from zotwatch.llm.relevance_filter import PaperRelevanceFilter

logger = logging.getLogger(__name__)


def _l2norm(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows for cosine similarity via dot product."""
    mat = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


class GeoscienceGate:
    """Decides which flagship-journal articles are on-topic geoscience."""

    def __init__(
        self,
        config: "ScoringConfig.FlagshipConfig",
        vectorizer: BaseEmbeddingProvider,
        llm: BaseLLMProvider | None = None,
        model: str | None = None,
    ):
        self.config = config
        self.vectorizer = vectorizer
        self.llm = llm
        self.model = model

    def _negative_anchors(self) -> list[str]:
        """Collect negative anchors (multi-discipline), incl. legacy single anchor."""
        cfg = self.config
        anchors = list(getattr(cfg, "negative_anchors", None) or [])
        single = (cfg.negative_anchor or "").strip()
        if single:
            anchors.append(single)
        # De-duplicate while preserving order, dropping blanks.
        seen: set[str] = set()
        result: list[str] = []
        for a in (s.strip() for s in anchors):
            if a and a not in seen:
                seen.add(a)
                result.append(a)
        return result

    def select(self, candidates: list[CandidateWork]) -> list[CandidateWork]:
        """Return the subset of candidates that pass the geoscience gate."""
        if not candidates:
            return []

        cfg = self.config
        negatives = self._negative_anchors()
        use_neg = bool(negatives)
        anchors = [cfg.positive_anchor] + negatives

        anchor_vecs = _l2norm(self.vectorizer.encode_query(anchors))
        cand_vecs = _l2norm(self.vectorizer.encode([c.content_for_embedding() for c in candidates]))
        sims = cand_vecs @ anchor_vecs.T  # (n, 1 + k)
        sim_pos = sims[:, 0]
        # Reject against the CLOSEST negative anchor, not an averaged one.
        sim_neg = sims[:, 1:].max(axis=1) if use_neg else np.zeros(len(candidates), dtype=np.float32)

        accepted, gray = self._partition(candidates, sim_pos, sim_neg, use_neg)

        if gray and cfg.llm_fallback and self.llm is not None:
            kept = self._llm_judge(gray)
            keep_ids = {id(c) for c in accepted} | {id(c) for c in kept}
            # Preserve original order across the accepted + LLM-kept sets.
            accepted = [c for c in candidates if id(c) in keep_ids]
        elif gray:
            # No LLM available: keep the gray zone (inclusive by design).
            keep_ids = {id(c) for c in accepted} | {id(c) for c in gray}
            accepted = [c for c in candidates if id(c) in keep_ids]

        logger.info(
            "Flagship geoscience gate: %d/%d articles passed (%d gray-zone judged)",
            len(accepted),
            len(candidates),
            len(gray),
        )
        return accepted

    def _partition(
        self,
        candidates: list[CandidateWork],
        sim_pos: np.ndarray,
        sim_neg: np.ndarray,
        use_neg: bool,
    ) -> tuple[list[CandidateWork], list[CandidateWork]]:
        """Split candidates into clearly-accepted and gray-zone lists.

        Rejected candidates are dropped. Separated from embedding for testing.
        """
        accepted: list[CandidateWork] = []
        gray: list[CandidateWork] = []
        for candidate, sp, sn in zip(candidates, sim_pos, sim_neg):
            sp = float(sp)
            sn = float(sn)
            # Closer to the atmospheric anchor than to geoscience -> reject.
            if use_neg and sn >= sp:
                continue
            if sp >= self.config.min_score:
                accepted.append(candidate)
            elif sp < self.config.gray_low:
                continue
            else:
                gray.append(candidate)
        return accepted, gray

    def _llm_judge(self, gray: list[CandidateWork]) -> list[CandidateWork]:
        """Judge gray-zone articles against the field boundary with the LLM."""
        relevance = PaperRelevanceFilter(
            self.llm,
            model=self.model,
            batch_size=self.config.llm_batch_size,
            max_candidates=0,  # judge all gray-zone candidates
        )
        kept, _ = relevance.filter_candidates(gray, user_interests=self.config.llm_boundary)
        return kept


__all__ = ["GeoscienceGate"]
