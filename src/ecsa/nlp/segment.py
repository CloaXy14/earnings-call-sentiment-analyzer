"""Sentence segmentation with stable provenance and speaker metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pysbd

from ecsa.nlp.cleaning import CleanedDocument
from ecsa.schemas import SectionName, SpeakerRole


@dataclass(frozen=True)
class Sentence:
    """One sentence; ordinal is zero-based within its source document."""

    sentence_id: str
    doc_id: str
    ordinal: int
    section: SectionName
    speaker: str
    role: SpeakerRole
    text: str


def segment(document: CleanedDocument) -> list[Sentence]:
    """Use pysbd without discarding attribution or prepared/Q&A boundaries."""
    splitter = pysbd.Segmenter(language="en", clean=False)
    result: list[Sentence] = []
    for section in document.sections:
        for turn in section.utterances:
            for text in splitter.segment(turn.text):
                text = text.strip()
                if text:
                    sid = hashlib.sha256(
                        f"{document.document.doc_id}:{len(result)}:{text}".encode()
                    ).hexdigest()
                    result.append(
                        Sentence(
                            sid,
                            document.document.doc_id,
                            len(result),
                            section.name,
                            turn.speaker,
                            turn.role,
                            text,
                        )
                    )
    return result
