"""Licensed LM dictionary and lemmatized phrase matching with three-token negation."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import simplemma

from ecsa.nlp.segment import Sentence
from ecsa.settings import Settings


@lru_cache(maxsize=50000)
def lemma(word: str) -> str:
    """English dictionary lemma; packaged simplemma resources require no downloads."""
    return str(simplemma.lemmatize(word.lower(), lang="en"))


def words(text: str) -> list[str]:
    """Tokenize financial prose into English words without punctuation."""
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


@dataclass(frozen=True)
class PhraseMatch:
    """Phrase evidence with original sentence, category and negation flag."""

    sentence_id: str
    phrase: str
    category: str
    negated: bool
    sentence: str
    matched_text: str


class Lexicon:
    """User-supplied LM CSV (license respected) plus configurable custom phrases."""

    def __init__(self, settings: Settings) -> None:
        self.phrases: dict[str, set[tuple[str, ...]]] = {
            key: set() for key in ("positive", "negative", "uncertainty", "litigious")
        }
        for category, entries in (
            ("positive", settings.positive_phrases),
            ("negative", settings.negative_phrases),
            ("uncertainty", settings.uncertainty_phrases),
        ):
            self.phrases[category].update(tuple(lemma(t) for t in words(p)) for p in entries)
        if settings.dictionary_path:
            self.load_dictionary(settings.dictionary_path)

    def load_dictionary(self, path: Path) -> None:
        """Read LM Word/Positive/Negative/Uncertainty/Litigious columns; nonzero means member."""
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                normalized = {k.lower(): v for k, v in row.items()}
                token = lemma(normalized["word"])
                for category in self.phrases:
                    if float(normalized.get(category, "0") or 0) != 0:
                        self.phrases[category].add((token,))

    def analyze(
        self, sentences: list[Sentence]
    ) -> tuple[dict[str, float], list[dict[str, object]]]:
        """Counts per 1,000 words; net=(positive-negative)/(positive+negative+1)."""
        counts = dict.fromkeys(self.phrases, 0)
        matches = []
        total = 0
        for sentence in sentences:
            spans = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence.text))
            tokens = [lemma(t.group()) for t in spans]
            total += len(tokens)
            for category, phrases in self.phrases.items():
                # Longest matches win within a category; avoid counting 'expansion' twice.
                occupied: set[int] = set()
                for phrase in sorted(phrases, key=lambda p: (-len(p), p)):
                    for i in range(len(tokens) - len(phrase) + 1):
                        positions = set(range(i, i + len(phrase)))
                        if positions & occupied or tuple(tokens[i : i + len(phrase)]) != phrase:
                            continue
                        occupied |= positions
                        negated = bool(
                            set(tokens[max(0, i - 3) : i])
                            & {"no", "not", "never", "without", "less"}
                        )
                        polarity = (
                            {"positive": "negative", "negative": "positive"}.get(category, category)
                            if negated
                            else category
                        )
                        counts[polarity] += 1
                        matches.append(
                            asdict(
                                PhraseMatch(
                                    sentence.sentence_id,
                                    " ".join(phrase),
                                    polarity,
                                    negated,
                                    sentence.text,
                                    sentence.text[
                                        spans[i].start() : spans[i + len(phrase) - 1].end()
                                    ],
                                )
                            )
                        )
        positive, negative = counts["positive"], counts["negative"]
        metrics = {f"{key}_per_1k": value * 1000 / max(total, 1) for key, value in counts.items()}
        metrics["lexicon_net"] = (positive - negative) / (positive + negative + 1)
        return metrics, matches
