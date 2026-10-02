"""Synthetic NLP assertions including pretrained wrapper stubs and future invariance."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from ecsa.nlp.cleaning import clean_document
from ecsa.nlp.features import add_surprise, analyze_documents
from ecsa.nlp.finbert import DemoScorer, FinBertScorer
from ecsa.nlp.fls import is_forward_looking
from ecsa.nlp.lexicon import Lexicon
from ecsa.nlp.segment import Sentence, segment
from ecsa.nlp.themes import CorpusDocument, TemporalThemes
from ecsa.schemas import Document, Section, Utterance
from ecsa.settings import Settings

CONFIG = Path(__file__).resolve().parents[1] / "config/settings.yaml"


def sentence(text: str) -> Sentence:
    return Sentence("sentence", "document", 0, "qa", "Executive", "executive", text)


def test_safe_harbor_removed_but_business_guidance_retained(document_payload: dict) -> None:
    text = (
        "Safe harbor: forward-looking statements involve risks and actual results "
        "may differ.\n\nWe expect strong demand next quarter."
    )
    doc = Document.model_validate(
        {
            **document_payload,
            "sections": [
                Section(
                    name="prepared_remarks",
                    utterances=[Utterance(speaker="CEO", role="executive", text=text)],
                )
            ],
        }
    )
    cleaned = clean_document(doc)
    sentences = segment(cleaned)
    assert len(cleaned.removed) == 1 and cleaned.removed[0]["reason"] == "safe_harbor"
    assert len(sentences) == 1 and "We expect" in sentences[0].text
    assert is_forward_looking(sentences[0].text)
    assert sentences[0].role == "executive"


def test_negation_and_lemmatized_multiword_matching() -> None:
    lex = Lexicon(Settings.from_yaml(CONFIG))
    metrics, matches = lex.analyze([sentence("We had no margin pressures and raised guidance.")])
    assert metrics["lexicon_net"] == pytest.approx(2 / 3)
    assert len(matches) == 2 and any(m["negated"] for m in matches)
    assert {m["category"] for m in matches} == {"positive"}


class StubModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={0: "neutral", 1: "positive", 2: "negative"})
        self.calls = 0

    def to(self, device: str) -> "StubModel":
        return self

    def eval(self) -> "StubModel":
        return self

    def __call__(self, **tokens: torch.Tensor) -> object:
        assert not torch.is_grad_enabled()
        self.calls += 1
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0, -1.0]] * len(tokens["input_ids"])))


def test_finbert_mapping_inference_batch_and_revision_cache(tmp_path: Path) -> None:
    settings = Settings.from_yaml(CONFIG).model_copy(
        update={"data_dir": tmp_path, "finbert_batch_size": 2}
    )

    def tokenize(texts: list[str], **kw: object) -> dict[str, torch.Tensor]:
        assert kw["max_length"] == 512 and kw["truncation"] is True
        return {"input_ids": torch.ones(len(texts), 2, dtype=torch.int64)}

    model = StubModel()
    scorer = FinBertScorer(settings, model=model, tokenizer=tokenize)
    results = scorer.score(["Good.", "Bad.", "Neutral."])
    expected = torch.softmax(torch.tensor([0.0, 2.0, -1.0]), dim=0).tolist()
    assert results[0].p_pos == pytest.approx(expected[1])
    assert results[0].net == pytest.approx(expected[1] - expected[2])
    scorer.score(["Good."])
    assert model.calls == 2
    other = StubModel()
    FinBertScorer(
        settings.model_copy(update={"finbert_revision": "a" * 40}), model=other, tokenizer=tokenize
    ).score(["Good."])
    assert other.calls == 1


def test_surprise_uses_previous_four_known_events() -> None:
    times = pd.date_range("2024-01-01", periods=7, freq="MS", tz="UTC")
    frame = pd.DataFrame(
        {
            "ticker": ["TEST"] * 7,
            "event_datetime": times,
            "available_at": times,
            "finbert_net_mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )
    expected = (0.5 - np.mean([0.1, 0.2, 0.3, 0.4])) / np.std([0.1, 0.2, 0.3, 0.4], ddof=1)
    baseline = add_surprise(frame)
    assert baseline.loc[4, "sentiment_surprise"] == pytest.approx(expected)
    assert np.isnan(baseline.loc[1, "sentiment_surprise"])
    frame.loc[5:, "finbert_net_mean"] = -999
    changed = add_surprise(frame.sample(frac=1, random_state=42))
    assert changed.loc[4, "sentiment_surprise"] == baseline.loc[4, "sentiment_surprise"]
    frame.loc[0, "available_at"] = times[6]
    assert np.isnan(add_surprise(frame).loc[2, "sentiment_surprise"])


def test_themes_excludes_future_corpus() -> None:
    model = TemporalThemes(Settings.from_yaml(CONFIG))
    past = [
        CorpusDocument(
            "Strong revenue growth and margin expansion.", datetime(2023, 1, i, tzinfo=UTC)
        )
        for i in (1, 2)
    ]
    future = CorpusDocument("Future secret guidance collapse.", datetime(2025, 1, 1, tzinfo=UTC))
    args = ([sentence("Strong revenue growth.")], past, datetime(2024, 1, 1, tzinfo=UTC))
    baseline = model.extract(*args)
    changed = model.extract(args[0], past + [future], args[2])
    assert baseline == changed and baseline[1]


def test_feature_aggregation_and_metadata(document_payload: dict, tmp_path: Path) -> None:
    doc = Document.model_validate(document_payload)
    settings = Settings.from_yaml(CONFIG).model_copy(update={"data_dir": tmp_path})
    output = analyze_documents([doc], DemoScorer(), settings)
    assert len(output.features) == 1
    assert "finbert_net_mean_qa" in output.features
    assert output.features.iloc[0]["model_id"] == "synthetic-demo-scorer"
    assert set(output.sentences["role"]) == {"executive", "analyst"}


def test_lm_dictionary_categories_and_surface_evidence(tmp_path: Path) -> None:
    dictionary = tmp_path / "lm.csv"
    dictionary.write_text(
        "Word,Positive,Negative,Uncertainty,Litigious\nPROFIT,2009,0,0,0\nUNCERTAIN,0,0,2009,0\nLAWSUIT,0,0,0,2009\n"
    )
    settings = Settings.from_yaml(CONFIG).model_copy(update={"dictionary_path": dictionary})
    metrics, matches = Lexicon(settings).analyze(
        [sentence("Profits remain uncertain after lawsuits; we raised guidance.")]
    )
    assert metrics["uncertainty_per_1k"] > 0 and metrics["litigious_per_1k"] > 0
    assert any(m["matched_text"] == "raised guidance" for m in matches)


def test_same_vendor_event_id_is_namespaced_by_ticker(
    document_payload: dict, tmp_path: Path
) -> None:
    one = Document.model_validate({**document_payload, "event_id": "vendor-call-1"})
    two = Document.model_validate(
        {**document_payload, "ticker": "SECOND", "event_id": "vendor-call-1"}
    )
    settings = Settings.from_yaml(CONFIG).model_copy(update={"data_dir": tmp_path})
    result = analyze_documents([one, two], DemoScorer(), settings)
    assert result.features.event_id.nunique() == 2
