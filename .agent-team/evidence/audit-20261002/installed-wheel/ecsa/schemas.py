"""Research records with explicit information availability and UTC instants.

Returns and portfolio weights are decimal fractions, never percentages. Calendar
dates label exchange sessions; they are not timestamps. All timestamp fields reject
naive datetimes and normalize aware inputs to UTC.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_validator,
)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UTCDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
NonemptyString = Annotated[str, Field(min_length=1)]
SentimentScore = Annotated[FiniteFloat, Field(ge=-1, le=1)]
Fraction = Annotated[FiniteFloat, Field(ge=0, le=1)]
PositivePrice = Annotated[FiniteFloat, Field(gt=0)]
DocType = Literal["transcript", "8k_press_release", "presentation", "10q_mdna", "10k_mdna"]
SessionTiming = Literal["BMO", "DURING", "AMC", "UNKNOWN"]
SectionName = Literal["prepared_remarks", "qa", "mdna", "other"]
SpeakerRole = Literal["executive", "analyst", "operator", "unknown"]


class Record(BaseModel):
    """Shared strict field names and immutable scalar fields for research records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Utterance(Record):
    """One speaker turn; text is the original unscored content."""

    speaker: str
    role: SpeakerRole = "unknown"
    text: str


class Section(Record):
    """An ordered group of utterances in a document section."""

    name: SectionName
    utterances: list[Utterance] = Field(default_factory=list)


def document_id(source: str, url: str, text: str) -> str:
    """Return a SHA-256 hex ID of the exact UTF-8 source, URL and text.

    A compact JSON array makes the three-field boundary unambiguous. Text is not
    normalized here: changing original source text creates a different document.
    """
    payload = json.dumps([source, url, text], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Document(Record):
    """Source document; ``available_at`` is when the full text was first knowable.

    Both instants are UTC. Availability must be supplied explicitly: call start
    time alone does not establish when its completed transcript was available.
    CIK is a ten-digit identifier, or None when not yet resolved.
    """

    doc_id: str = ""
    event_id: str | None = None
    event_time_known: bool = True
    ticker: NonemptyString
    cik: Annotated[str, Field(pattern=r"^[0-9]{10}$")] | None = None
    doc_type: DocType
    event_datetime: UTCDatetime
    available_at: UTCDatetime
    session_timing: SessionTiming = "UNKNOWN"
    source: NonemptyString
    url: str
    raw_text: str
    sections: list[Section] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Generate an omitted ID; reject an ID that does not match the source bytes."""
        if not self.event_time_known and self.session_timing != "UNKNOWN":
            raise ValueError("An unknown call time requires UNKNOWN session timing")
        expected = document_id(self.source, self.url, self.raw_text)
        if self.doc_id and self.doc_id != expected:
            raise ValueError("doc_id must match the hash of source, url and raw_text")
        object.__setattr__(self, "doc_id", expected)
        return self


class PriceBar(Record):
    """Daily bar: prices in quote currency/share; volume in shares.

    ``date`` labels the New York exchange session, while ``available_at`` is the
    UTC instant the bar was knowable. Adjusted close can differ from the OHLC range.
    """

    ticker: NonemptyString
    date: date
    available_at: UTCDatetime
    open: PositivePrice
    high: PositivePrice
    low: PositivePrice
    close: PositivePrice
    adj_close: PositivePrice
    volume: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_ohlc(self) -> Self:
        """Require low <= open/close <= high in the bar's quote currency."""
        if self.low > min(self.open, self.close, self.high) or self.high < max(
            self.open, self.close, self.low
        ):
            raise ValueError("OHLC prices must lie between low and high")
        return self


class EventFeatures(Record):
    """One (ticker, event) feature row, with explicit source availability.

    Scores are in [-1, 1], shares are fractions, uncertainty is per 1,000 words,
    and surprise is a z-score. None denotes an uncomputed value at scaffold stage;
    the Step 2 calculation will emit NaN for an undefined sentiment surprise.
    ``source_available_at`` is the latest availability of all inputs, including
    fitted artifacts; it cannot be later than the feature's ``available_at``.
    """

    event_id: NonemptyString
    ticker: NonemptyString
    event_datetime: UTCDatetime
    available_at: UTCDatetime
    source_available_at: UTCDatetime
    finbert_net_mean: SentimentScore | None = None
    finbert_net_weighted: SentimentScore | None = None
    pos_share: Fraction | None = None
    neg_share: Fraction | None = None
    lexicon_net: SentimentScore | None = None
    uncertainty_per_1k: Annotated[FiniteFloat, Field(ge=0)] | None = None
    fls_share: Fraction | None = None
    fls_net: SentimentScore | None = None
    sentiment_surprise: FiniteFloat | None = None

    @model_validator(mode="after")
    def validate_information_time(self) -> Self:
        """Reject a feature timestamp that precedes its latest input's availability."""
        if self.source_available_at > self.available_at:
            raise ValueError("Features cannot use source data from the future")
        if self.pos_share is not None and self.neg_share is not None:
            if self.pos_share + self.neg_share > 1 + 1e-12:
                raise ValueError("Positive and negative shares must sum to at most one")
        return self


class Signal(Record):
    """Trading intent; score is unitless and weight is unsigned gross NAV fraction.

    ``side`` supplies the direction. All ``*_date`` fields are UTC instants at
    their relevant closes. ``feature_available_at`` records the latest input
    availability and must not exceed signal formation. Dates describe the planned
    holding period, not future realized prices or returns.
    """

    event_id: NonemptyString
    signal_date: UTCDatetime
    feature_available_at: UTCDatetime
    ticker: NonemptyString
    score: FiniteFloat
    side: Literal["LONG", "SHORT", "NONE"]
    weight: Fraction
    entry_date: UTCDatetime
    exit_date: UTCDatetime

    @model_validator(mode="after")
    def validate_trade(self) -> Self:
        """Enforce information timing, chronological execution and consistent exposure."""
        if self.feature_available_at > self.signal_date:
            raise ValueError("Signals cannot use features from the future")
        if not self.signal_date <= self.entry_date < self.exit_date:
            raise ValueError("Require signal_date <= entry_date < exit_date")
        if (self.side == "NONE") != (self.weight == 0):
            raise ValueError("NONE requires zero weight; LONG/SHORT require positive weight")
        return self
