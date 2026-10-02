"""Forward-looking statements on already cleaned sentences, with optional pinned model."""

from __future__ import annotations

import re
from typing import Any

from ecsa.nlp.finbert import ModelClient
from ecsa.settings import Settings

CUES = re.compile(
    r"\b(?:expect\w*|anticipat\w*|guidance|outlook|will|plan\w*|intend\w*|target\w*)\b"
    r"|\bnext quarter\b|\bfull year\b|\bfiscal (?:year )?20\d{2}\b",
    re.I,
)


def is_forward_looking(text: str) -> bool:
    """Classify future/modal business cues; input must exclude safe-harbor boilerplate."""
    return bool(CUES.search(text))


class FlsDetector:
    """Optional FLS model loaded from a pinned local snapshot; rules are the default."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any = None
        self.tokenizer: Any = None
        if settings.fls_backend == "model":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            path = ModelClient(settings).snapshot(settings.fls_model_id, str(settings.fls_revision))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def classify(self, sentences: list[str]) -> list[bool]:
        """Return one FLS decision per sentence; model semantics must be explicit in labels."""
        if self.model is None:
            return [is_forward_looking(text) for text in sentences]
        import torch

        results: list[bool] = []
        labels = {int(i): str(label).lower() for i, label in self.model.config.id2label.items()}
        positive = {
            i
            for i, label in labels.items()
            if label in {"specific fls", "non-specific fls", "fls", "forward-looking"}
        }
        if not positive:
            raise ValueError(
                "Unrecognized FLS id2label semantics; configure a supported labeled checkpoint"
            )
        for i in range(0, len(sentences), self.settings.finbert_batch_size):
            inputs = self.tokenizer(
                sentences[i : i + self.settings.finbert_batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.inference_mode():
                predictions = self.model(**inputs).logits.argmax(dim=-1).tolist()
            results.extend(int(label) in positive for label in predictions)
        return results
