"""Validated YAML settings and environment-only credentials; no network or dotenv loading."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator

from ecsa.schemas import UTCDatetime


def _environment_secret(name: str) -> SecretStr | None:
    value = os.environ.get(name, "").strip()
    return SecretStr(value) if value else None


class Settings(BaseModel):
    """Application configuration; durations are seconds and request rates are requests/second.

    Use :meth:`from_yaml` to resolve paths relative to the project's config directory.
    Credentials are captured from the process environment at construction, omitted from
    serialization, and cannot be passed as constructor or YAML fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    storage_timezone: Literal["UTC"] = "UTC"
    market_timezone: Literal["America/New_York"] = "America/New_York"
    data_dir: Path = Path("data")
    tickers: list[str] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None
    benchmark: str = Field(default="SPY", min_length=1)
    finbert_model_id: str = Field(default="ProsusAI/finbert", min_length=1)
    finbert_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    finbert_batch_size: int = Field(default=32, gt=0)
    finbert_max_tokens: int = Field(default=512, gt=0, le=512)
    llm_enabled: bool = False
    llm_model: str = "gpt-4o-mini-2024-07-18"
    llm_available_at: UTCDatetime = datetime(2024, 7, 18, tzinfo=UTC)
    llm_input_cost_per_million: float = Field(default=0.15, ge=0)
    llm_output_cost_per_million: float = Field(default=0.60, ge=0)
    finbert_available_at: UTCDatetime = datetime(2023, 5, 23, 12, 43, 35, tzinfo=UTC)
    nlp_backend: Literal["finbert", "demo"] = "finbert"
    fls_backend: Literal["rules", "model"] = "rules"
    fls_model_id: str = "yiyanghkust/finbert-fls"
    fls_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    fls_available_at: UTCDatetime | None = None
    dictionary_path: Path | None = None
    positive_phrases: list[str] = Field(
        default_factory=lambda: [
            "growth",
            "expansion",
            "record revenue",
            "strong demand",
            "margin expansion",
            "raised guidance",
        ]
    )
    negative_phrases: list[str] = Field(
        default_factory=lambda: [
            "costs",
            "margin pressure",
            "uncertainty",
            "headwinds",
            "slowdown",
            "lowered guidance",
            "impairment",
        ]
    )
    uncertainty_phrases: list[str] = Field(default_factory=lambda: ["uncertainty"])
    financial_stop_words: list[str] = Field(
        default_factory=lambda: [
            "company",
            "quarter",
            "year",
            "thank",
            "operator",
            "million",
            "billion",
        ]
    )
    topic_count: int = Field(default=4, ge=1)
    estimation_start: int = -250
    estimation_end: int = -30
    estimation_min_days: int = Field(default=120, ge=3)
    event_start: int = -5
    event_end: int = 10
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "finbert_net_weighted": 1.0,
            "sentiment_surprise": 1.0,
            "lexicon_net": 1.0,
            "fls_net": 1.0,
            "uncertainty_per_1k": -1.0,
        }
    )
    ranking_days: int = Field(default=90, gt=0)
    minimum_ranking_events: int = Field(default=20, ge=2)
    long_quantile: float = Field(default=0.8, gt=0.5, lt=1)
    short_quantile: float = Field(default=0.2, gt=0, lt=0.5)
    holding_days: int = Field(default=10, gt=0)
    max_name_weight: float = Field(default=0.05, gt=0, le=1)
    gross_target: float = Field(default=1.0, gt=0, le=2)
    weighting: Literal["equal", "inverse_vol"] = "equal"
    sector_neutral: bool = False
    sectors: dict[str, str] = Field(default_factory=dict)
    volatility_days: int = Field(default=60, ge=2)
    transaction_cost_bps: float = Field(default=10, ge=0)
    borrow_cost_bps_annual: float = Field(default=0, ge=0)
    placebo_seeds: int = Field(default=100, ge=1)
    variants_tried: int = Field(default=1, ge=1)
    learned_model: Literal["none", "ridge", "logistic"] = "none"
    embargo_days: int = Field(default=10, ge=10)
    minimum_training_events: int = Field(default=30, ge=5)
    include_edgar: bool = False
    include_mdna: bool = False
    price_backend: Literal["yfinance", "local"] = "yfinance"

    @model_validator(mode="after")
    def validate_research_windows(self) -> Self:
        """Reject settings that leak event returns into fitting or lack pinned FLS metadata."""
        if not self.estimation_start < self.estimation_end < self.event_start <= -5:
            raise ValueError("Estimation must finish before the pre-event window")
        if self.event_end < max(10, self.holding_days):
            raise ValueError("Event window must cover the holding period and +10 days")
        if not self.score_weights or not any(self.score_weights.values()):
            raise ValueError("At least one score weight must be nonzero")
        if self.fls_backend == "model" and (not self.fls_revision or not self.fls_available_at):
            raise ValueError("FLS model requires an immutable revision and availability timestamp")
        return self

    sec_requests_per_second: float = Field(default=8.0, gt=0, lt=10)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    request_max_retries: int = Field(default=4, ge=0)
    retry_backoff_seconds: float = Field(default=1.0, gt=0)
    cache_ttl_seconds: int = Field(default=86400, gt=0)

    _sec_user_agent: SecretStr | None = PrivateAttr(
        default_factory=lambda: _environment_secret("SEC_USER_AGENT")
    )
    _openai_api_key: SecretStr | None = PrivateAttr(
        default_factory=lambda: _environment_secret("OPENAI_API_KEY")
    )
    _transcript_api_key: SecretStr | None = PrivateAttr(
        default_factory=lambda: _environment_secret("TRANSCRIPT_API_KEY")
    )

    @classmethod
    def from_yaml(cls, path: str | Path = "config/settings.yaml") -> Self:
        """Load YAML; resolve relative data paths against ``path.parent.parent``.

        The file convention is ``<project>/config/settings.yaml``. Loading does
        not create directories, contact services, or require SEC credentials.
        """
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Settings YAML must contain a mapping of configuration fields.")
        settings = cls.model_validate(payload)
        data_dir = settings.data_dir.expanduser()
        if not data_dir.is_absolute():
            data_dir = config_path.parent.parent / data_dir
        dictionary = settings.dictionary_path
        if dictionary is not None and not dictionary.is_absolute():
            dictionary = config_path.parent.parent / dictionary
        return settings.model_copy(
            update={"data_dir": data_dir.resolve(), "dictionary_path": dictionary}
        )

    @property
    def raw_dir(self) -> Path:
        """Directory for raw Parquet data and licensed input files."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Directory for intermediate Parquet tables."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Directory for processed Parquet tables consumed by the app."""
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        """Directory for client response and revision-aware model caches."""
        return self.data_dir / "cache"

    @property
    def sec_user_agent(self) -> SecretStr | None:
        """SEC identity from SEC_USER_AGENT; required by the future EdgarClient."""
        return self._sec_user_agent

    @property
    def openai_api_key(self) -> SecretStr | None:
        """Optional OPENAI_API_KEY from the process environment."""
        return self._openai_api_key

    @property
    def transcript_api_key(self) -> SecretStr | None:
        """Optional licensed vendor credential from TRANSCRIPT_API_KEY."""
        return self._transcript_api_key
