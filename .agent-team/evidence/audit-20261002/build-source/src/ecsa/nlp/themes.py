"""Temporal TF-IDF/NMF themes: fit only on documents available before the current call."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from ecsa.nlp.segment import Sentence
from ecsa.settings import Settings
from ecsa.storage import utc


@dataclass(frozen=True)
class CorpusDocument:
    """Text corpus member with UTC availability, independent of realized returns."""

    text: str
    available_at: datetime


class TemporalThemes:
    """Small expanding corpus model; current/future documents never enter fit."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(
        self, sentences: list[Sentence], history: list[CorpusDocument], as_of: datetime
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Return fitted topic labels and ten TF-IDF phrases with supporting sentences."""
        texts = [doc.text for doc in history if utc(doc.available_at) < utc(as_of)]
        if len(texts) < 2 or not sentences:
            return [], []
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), stop_words=self.settings.financial_stop_words, max_features=3000
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return [], []
        names = vectorizer.get_feature_names_out()
        current = vectorizer.transform([" ".join(s.text for s in sentences)])
        components = min(self.settings.topic_count, matrix.shape[0], matrix.shape[1])
        model = NMF(
            n_components=components, init="nndsvda", random_state=self.settings.seed, max_iter=500
        )
        model.fit(matrix)
        weights = model.transform(current)[0]
        topics = []
        for topic in np.argsort(weights)[::-1][:3]:
            if weights[topic] > 0:
                topics.append(", ".join(names[np.argsort(model.components_[topic])[::-1][:3]]))
        phrases = []
        values = current.toarray()[0]
        for i in np.argsort(values)[::-1]:
            phrase = str(names[i])
            support = [s for s in sentences if phrase in s.text.lower()]
            if values[i] > 0 and support:
                best = max(support, key=lambda s: s.text.lower().count(phrase))
                phrases.append(
                    {"phrase": phrase, "sentence_id": best.sentence_id, "sentence": best.text}
                )
            if len(phrases) == 10:
                break
        return topics, phrases
