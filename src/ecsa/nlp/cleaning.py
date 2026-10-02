"""Unicode cleanup and auditable removal of safe-harbor boilerplate."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ecsa.schemas import Document, Section, Utterance


@dataclass(frozen=True)
class CleanedDocument:
    """Cleaned sections and removed text/reasons; source identity is unchanged."""

    document: Document
    sections: list[Section]
    removed: list[dict[str, str]]


_DASHES = str.maketrans(dict.fromkeys("–—‑‐", "-"))


def normalize(text: str) -> str:
    """Normalize Unicode and intra-paragraph whitespace, retaining paragraph boundaries."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").translate(_DASHES)
    return "\n\n".join(
        re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()
    )


def clean_document(document: Document) -> CleanedDocument:
    """Remove legal disclaimer paragraphs before forward-looking detection."""
    removed, sections = [], []
    for section in document.sections:
        turns = []
        for turn in section.utterances:
            kept = []
            for paragraph in normalize(turn.text).split("\n\n"):
                disclaimer = bool(
                    re.search(r"safe[ -]harbor|forward[ -]looking statements?", paragraph, re.I)
                ) and bool(
                    re.search(
                        r"safe[ -]harbor|risks?|actual results|uncertaint|securities|no obligation",
                        paragraph,
                        re.I,
                    )
                )
                housekeeping = turn.role == "operator" and bool(
                    re.search(
                        r"welcome|recorded|instructions|hold|question|conference|please|thank",
                        paragraph,
                        re.I,
                    )
                )
                if disclaimer or housekeeping:
                    removed.append(
                        {
                            "doc_id": document.doc_id,
                            "text": paragraph,
                            "reason": "safe_harbor" if disclaimer else "operator_housekeeping",
                        }
                    )
                elif paragraph:
                    kept.append(paragraph)
            if kept:
                turns.append(
                    Utterance(speaker=turn.speaker, role=turn.role, text="\n\n".join(kept))
                )
        if turns:
            sections.append(Section(name=section.name, utterances=turns))
    return CleanedDocument(document, sections, removed)
