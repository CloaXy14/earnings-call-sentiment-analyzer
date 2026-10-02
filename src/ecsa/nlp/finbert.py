"""Pinned FinBERT inference and an explicitly synthetic offline demo scorer."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ecsa.data.cache import DiskCache, HttpClient
from ecsa.settings import Settings


@dataclass(frozen=True)
class SentimentResult:
    """Probabilities are fractions; net is positive minus negative in [-1,1]."""

    p_pos: float
    p_neg: float
    p_neu: float

    @property
    def net(self) -> float:
        """Signed sentiment, in [-1,1]."""
        return self.p_pos - self.p_neg


class SentimentModel(Protocol):
    """Common sentence scorer contract; provenance includes the artifact availability."""

    available_at: datetime
    model_id: str
    revision: str

    def score(self, sentences: list[str]) -> list[SentimentResult]: ...


class ModelClient:
    """Download only pinned model files through the rate-limited cached HTTP client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = HttpClient(
            DiskCache(settings.cache_dir / "model_http", float("inf")),
            rate=2,
            retries=settings.request_max_retries,
            timeout=settings.request_timeout_seconds,
            backoff=settings.retry_backoff_seconds,
        )

    def snapshot(self, model_id: str, revision: str) -> Path:
        """Cache a tokenizer/config/weights snapshot keyed by model ID and immutable SHA."""
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Model revision must be an immutable 40-character commit")
        path = self.settings.cache_dir / "models" / model_id.replace("/", "--") / revision
        if (path / "READY").exists():
            return path
        metadata = self.http.json(
            f"https://huggingface.co/api/models/{model_id}/revision/{revision}"
        )
        files = {entry["rfilename"] for entry in metadata["siblings"]}
        weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
        if weight not in files:
            raise ValueError("No supported pretrained weights in pinned model snapshot")
        selected = {
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "tokenizer.json",
            weight,
        } & files
        path.mkdir(parents=True, exist_ok=True)
        for name in sorted(selected):
            data = self.http.request(f"https://huggingface.co/{model_id}/resolve/{revision}/{name}")
            temporary = path / (name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path / name)
        (path / "READY").write_text(revision)
        return path


class FinBertScorer:
    """Batched deterministic inference; model label order always comes from id2label."""

    def __init__(self, settings: Settings, *, model: Any = None, tokenizer: Any = None) -> None:
        import torch

        self.settings = settings
        self.model_id, self.revision = settings.finbert_model_id, settings.finbert_revision
        self.available_at = settings.finbert_available_at
        self.cache = DiskCache(settings.cache_dir / "sentiment", float("inf"))
        torch.manual_seed(settings.seed)
        torch.use_deterministic_algorithms(True)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if model is None or tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            path = ModelClient(settings).snapshot(self.model_id, self.revision)
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        self.model, self.tokenizer = model.to(self.device).eval(), tokenizer
        self.labels = {
            str(label).lower(): int(index) for index, label in self.model.config.id2label.items()
        }
        if set(self.labels) != {"positive", "negative", "neutral"}:
            raise ValueError("FinBERT id2label must map positive/negative/neutral")

    def score(self, sentences: list[str]) -> list[SentimentResult]:
        """Return probabilities and net; cache by text hash, model ID, revision and truncation."""
        import torch

        keys = [
            {
                "text": hashlib.sha256(s.encode()).hexdigest(),
                "revision": self.revision,
                "max_tokens": self.settings.finbert_max_tokens,
            }
            for s in sentences
        ]
        values = [self.cache.get_json(self.model_id, key) for key in keys]
        missing = [i for i, value in enumerate(values) if value is None]
        for offset in range(0, len(missing), self.settings.finbert_batch_size):
            indices = missing[offset : offset + self.settings.finbert_batch_size]
            tokens = self.tokenizer(
                [sentences[i] for i in indices],
                padding=True,
                truncation=True,
                max_length=self.settings.finbert_max_tokens,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(**tokens).logits, dim=-1).cpu().tolist()
            for i, probs in zip(indices, probabilities, strict=True):
                prediction = SentimentResult(
                    probs[self.labels["positive"]],
                    probs[self.labels["negative"]],
                    probs[self.labels["neutral"]],
                )
                values[i] = asdict(prediction)
                self.cache.put_json(self.model_id, values[i], keys[i])
        return [SentimentResult(**value) for value in values]


class DemoScorer:
    """Synthetic lexical fixture scorer. This is NOT FinBERT or evidence of NLP quality."""

    available_at = datetime(2000, 1, 1, tzinfo=UTC)
    model_id, revision = "synthetic-demo-scorer", "demo-v1"

    def score(self, sentences: list[str]) -> list[SentimentResult]:
        """Deterministic toy sentiment for exercising the offline pipeline."""
        output = []
        for sentence in sentences:
            pos = len(re.findall(r"growth|strong|record|expansion|raised|improve", sentence, re.I))
            neg = len(
                re.findall(
                    r"pressure|uncertain|headwind|slowdown|lowered|impairment|costs", sentence, re.I
                )
            )
            net = math.tanh((pos - neg) / 2)
            output.append(SentimentResult((1 + net) * 0.45, (1 - net) * 0.45, 0.1))
        return output
