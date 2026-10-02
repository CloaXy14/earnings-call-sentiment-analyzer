"""Event feature aggregation with point-in-time surprise and evidence tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ecsa.data.transcripts import parse_transcript
from ecsa.nlp.cleaning import clean_document
from ecsa.nlp.finbert import SentimentModel, SentimentResult
from ecsa.nlp.fls import FlsDetector
from ecsa.nlp.lexicon import Lexicon, words
from ecsa.nlp.segment import Sentence, segment
from ecsa.nlp.themes import CorpusDocument, TemporalThemes
from ecsa.schemas import Document, EventFeatures
from ecsa.settings import Settings


@dataclass
class Analysis:
    """Event features and auditable sentence/phrase/disclaimer evidence tables."""

    features: pd.DataFrame
    sentences: pd.DataFrame
    phrases: pd.DataFrame
    removed: pd.DataFrame


def add_surprise(frame: pd.DataFrame) -> pd.DataFrame:
    """Company prior-four-event z-score; fewer than two/zero std gives NaN.

    Prior events must precede both the event itself and the information cutoff;
    tied availability timestamps are excluded. Future row permutations cannot matter.
    """
    result = frame.copy()
    result["sentiment_surprise"] = np.nan
    for index, row in result.iterrows():
        past = (
            result[
                (result.ticker == row.ticker)
                & (result.event_datetime < row.event_datetime)
                & (result.available_at < row.available_at)
            ]
            .sort_values("event_datetime")
            .tail(4)
        )
        values = past.finbert_net_mean.dropna()
        if len(values) >= 2 and values.std(ddof=1) > 1e-12:
            result.loc[index, "sentiment_surprise"] = (
                row.finbert_net_mean - values.mean()
            ) / values.std(ddof=1)
    return result


def _sentiment(sentences: list[Sentence], scores: list[SentimentResult]) -> dict[str, float]:
    if not sentences:
        return dict.fromkeys(
            ["finbert_net_mean", "finbert_net_weighted", "pos_share", "neg_share"], float("nan")
        )
    weights = [max(1, len(words(s.text))) for s in sentences]
    return {
        "finbert_net_mean": float(np.mean([s.net for s in scores])),
        "finbert_net_weighted": float(np.average([s.net for s in scores], weights=weights)),
        "pos_share": float(np.mean([s.p_pos > max(s.p_neg, s.p_neu) for s in scores])),
        "neg_share": float(np.mean([s.p_neg > max(s.p_pos, s.p_neu) for s in scores])),
    }


def analyze_documents(
    documents: list[Document], scorer: SentimentModel, settings: Settings
) -> Analysis:
    """Aggregate one row per explicitly linked (ticker,event), with supporting evidence."""
    groups: dict[tuple[str, str], list[Document]] = {}
    for doc in documents:
        key = (doc.ticker, f"{doc.ticker}:{doc.event_id or doc.event_datetime.isoformat()}")
        groups.setdefault(key, []).append(doc)
    rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    phrase_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    history: list[CorpusDocument] = []
    lexicon, detector, themes = Lexicon(settings), FlsDetector(settings), TemporalThemes(settings)
    for (ticker, event_id), docs in sorted(
        groups.items(), key=lambda item: (max(d.available_at for d in item[1]), item[0])
    ):
        cutoff = max(d.available_at for d in docs)
        known_at = max(
            cutoff,
            scorer.available_at,
            settings.fls_available_at
            if settings.fls_backend == "model" and settings.fls_available_at
            else cutoff,
        )
        sentences: list[Sentence] = []
        for doc in sorted(docs, key=lambda d: d.doc_id):
            if not doc.sections:
                doc = doc.model_copy(update={"sections": parse_transcript(doc.raw_text)})
            cleaned = clean_document(doc)
            sentences.extend(segment(cleaned))
            removed_rows.extend({**row, "event_id": event_id} for row in cleaned.removed)
        if not sentences:
            continue
        scores = scorer.score([s.text for s in sentences])
        fls = detector.classify([s.text for s in sentences])
        if len(scores) != len(sentences):
            raise ValueError("Scorer returned wrong number of predictions")
        metrics, matches = lexicon.analyze(sentences)
        topics, phrases = themes.extract(sentences, history, cutoff)
        row: dict[str, Any] = {
            "event_id": event_id,
            "ticker": ticker,
            "event_datetime": min(d.event_datetime for d in docs),
            "available_at": known_at,
            "source_available_at": known_at,
            "document_available_at": cutoff,
            "session_timing": docs[0].session_timing,
            "doc_type": "+".join(sorted({d.doc_type for d in docs})),
            "doc_ids": json.dumps([d.doc_id for d in docs]),
            "model_id": scorer.model_id,
            "model_revision": scorer.revision,
            "themes": json.dumps(topics),
            "key_phrases": json.dumps(phrases),
            **_sentiment(sentences, scores),
            **metrics,
            "fls_share": float(np.mean(fls)),
            "fls_net": float(np.mean([s.net for s, flag in zip(scores, fls, strict=True) if flag]))
            if any(fls)
            else float("nan"),
        }
        for split in ("prepared_remarks", "qa", "executive", "analyst"):
            indices = [i for i, s in enumerate(sentences) if s.section == split or s.role == split]
            row.update(
                {
                    key + "_" + split: value
                    for key, value in _sentiment(
                        [sentences[i] for i in indices], [scores[i] for i in indices]
                    ).items()
                }
            )
        EventFeatures.model_validate(
            {
                key: (None if isinstance(value, float) and np.isnan(value) else value)
                for key, value in row.items()
                if key in EventFeatures.model_fields
            }
        )
        rows.append(row)
        phrase_rows.extend({**match, "event_id": event_id, "ticker": ticker} for match in matches)
        for ordinal, (sentence, score, forward) in enumerate(
            zip(sentences, scores, fls, strict=True)
        ):
            sentence_rows.append(
                {
                    **asdict(sentence),
                    **asdict(score),
                    "net": score.net,
                    "forward_looking": forward,
                    "event_ordinal": ordinal,
                    "event_id": event_id,
                    "ticker": ticker,
                    "available_at": known_at,
                }
            )
        history.append(CorpusDocument(" ".join(s.text for s in sentences), cutoff))
    features = (
        add_surprise(pd.DataFrame(rows))
        if rows
        else pd.DataFrame(columns=["event_id", "ticker", "event_datetime", "available_at"])
    )
    return Analysis(
        features,
        pd.DataFrame(sentence_rows),
        pd.DataFrame(
            phrase_rows,
            columns=[
                "sentence_id",
                "phrase",
                "category",
                "negated",
                "sentence",
                "event_id",
                "ticker",
            ],
        ),
        pd.DataFrame(removed_rows, columns=["doc_id", "text", "reason", "event_id"]),
    )
