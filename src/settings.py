import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ZoteroApiConfig(BaseModel):
    user_id: str
    api_key: str
    page_size: int = 100
    polite_delay_ms: int = 200


class ZoteroConfig(BaseModel):
    mode: str = "api"
    api: ZoteroApiConfig = Field(default_factory=ZoteroApiConfig)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        allowed = {"api", "bbt"}
        if value not in allowed:
            raise ValueError(f"Unsupported Zotero mode '{value}'. Allowed: {sorted(allowed)}")
        return value


class OpenAlexConfig(BaseModel):
    enabled: bool = True
    mailto: str = "you@example.com"
    days_back: int = 7


class CrossRefConfig(BaseModel):
    enabled: bool = True
    mailto: str = "you@example.com"
    days_back: int = 7


class ArxivConfig(BaseModel):
    enabled: bool = True
    categories: List[str] = Field(default_factory=lambda: ["cs.LG"])
    days_back: int = 7
    max_results: int = 500


class BioRxivConfig(BaseModel):
    enabled: bool = True
    days_back: int = 7


class MedRxivConfig(BaseModel):
    enabled: bool = False
    days_back: int = 7


class SourcesConfig(BaseModel):
    openalex: OpenAlexConfig = Field(default_factory=OpenAlexConfig)
    crossref: CrossRefConfig = Field(default_factory=CrossRefConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    biorxiv: BioRxivConfig = Field(default_factory=BioRxivConfig)
    medrxiv: MedRxivConfig = Field(default_factory=MedRxivConfig)


class ScoreWeights(BaseModel):
    similarity: float = 0.50
    recency: float = 0.15
    citations: float = 0.15
    journal_quality: float = 0.09
    author_bonus: float = 0.02
    venue_bonus: float = 0.09

    def normalized(self) -> "ScoreWeights":
        total = sum(self.model_dump().values())
        if not total:
            raise ValueError("Score weights sum to zero; at least one positive weight is required.")
        normalized = {k: v / total for k, v in self.model_dump().items()}
        return ScoreWeights(**normalized)


class Thresholds(BaseModel):
    must_read: float = 0.75
    consider: float = 0.5


class ScoringConfig(BaseModel):
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    decay_days: Dict[str, int] = Field(default_factory=lambda: {"fast": 30, "medium": 60, "slow": 180})
    whitelist_authors: List[str] = Field(default_factory=list)
    whitelist_venues: List[str] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    model: str = "voyage-3.5"
    api_key: str
    input_type: str = "document"
    batch_size: int = 128


class Settings(BaseModel):
    zotero: ZoteroConfig
    sources: SourcesConfig
    scoring: ScoringConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


def _expand_env_vars(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    if isinstance(data, str):
        return os.path.expandvars(data)
    return data


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data = _expand_env_vars(data)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
    return data


def load_settings(base_dir: Path | str) -> Settings:
    base = Path(base_dir)
    config_path = base / "config" / "config.yaml"
    config = _load_yaml(config_path)
    return Settings(
        zotero=ZoteroConfig(**config.get("zotero", {})),
        sources=SourcesConfig(**config.get("sources", {})),
        scoring=ScoringConfig(**config.get("scoring", {})),
        embedding=EmbeddingConfig(**config.get("embedding", {})),
    )


__all__ = [
    "Settings",
    "load_settings",
    "ZoteroConfig",
    "SourcesConfig",
    "ScoringConfig",
    "EmbeddingConfig",
]
