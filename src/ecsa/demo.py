"""Reproducible synthetic source fixtures; not real earnings or investment evidence."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ecsa.events.calendar import TradingCalendar
from ecsa.schemas import Document, Section, Utterance
from ecsa.storage import write_parquet


def generate_fixtures(directory: Path, seed: int = 42) -> None:
    """Generate eight synthetic firms, twelve events each and 560 NYSE daily bars."""
    directory.mkdir(parents=True, exist_ok=True)
    dates = [
        s.date() for s in xcals.get_calendar("XNYS", start="2022-01-01", end="2025-01-01").sessions
    ][:560]
    calendar = TradingCalendar(dates)
    rng = np.random.default_rng(seed)
    market = rng.normal(0.00015, 0.008, len(dates))
    bars, documents = [], []
    for ticker_index, ticker in enumerate(["SPY", *[f"SYN{i:02d}" for i in range(8)]]):
        values = (
            market.copy()
            if ticker == "SPY"
            else 0.00005 + (0.8 + 0.05 * ticker_index) * market + rng.normal(0, 0.004, len(dates))
        )
        if ticker != "SPY":
            for event_number, index in enumerate(range(280, 521, 21)):
                tone = (event_number + 2 * ticker_index) % 7 - 3
                values[index + 1 : index + 11] += tone * 0.0004
                # Alternate BMO and AMC, all with day0 at index and text ready before entry.
                amc = event_number % 2 == 1
                event = (
                    calendar.close(dates[index - 1]) + timedelta(hours=1)
                    if amc
                    else calendar.close(dates[index]) - timedelta(hours=8)
                )
                positive = "Strong demand supported growth and margin expansion. " * max(tone, 0)
                negative = "Headwinds caused margin pressure and uncertainty. " * max(-tone, 0)
                if tone == 0:
                    positive = "Revenue was stable in the reporting period."
                guidance = (
                    "We expect strong demand next quarter and raised guidance."
                    if tone > 0
                    else "We anticipate a slowdown next quarter and lowered guidance."
                    if tone < 0
                    else "Our full year outlook is unchanged."
                )
                sections = [
                    Section(
                        name="prepared_remarks",
                        utterances=[
                            Utterance(
                                speaker="Operator",
                                role="operator",
                                text="Welcome to the conference call.",
                            ),
                            Utterance(
                                speaker="CEO",
                                role="executive",
                                text=(
                                    "Safe harbor: forward-looking statements involve risks "
                                    "and actual results may differ.\n\n"
                                )
                                + positive
                                + negative
                                + guidance,
                            ),
                        ],
                    ),
                    Section(
                        name="qa",
                        utterances=[
                            Utterance(
                                speaker="Analyst",
                                role="analyst",
                                text="What is your outlook for next quarter?",
                            ),
                            Utterance(speaker="CFO", role="executive", text=guidance),
                        ],
                    ),
                ]
                text = "\n\n".join(
                    f"{u.speaker}: {u.text}" for section in sections for u in section.utterances
                )
                documents.append(
                    Document(
                        event_id=f"{ticker}-{event_number}",
                        ticker=ticker,
                        cik=str(ticker_index).zfill(10),
                        doc_type="transcript",
                        event_datetime=event,
                        available_at=event + timedelta(hours=1),
                        session_timing="AMC" if amc else "BMO",
                        source="synthetic_fixture",
                        url=f"local://synthetic/{ticker}/{event_number}",
                        raw_text=text,
                        sections=sections,
                    ).model_dump(mode="json")
                )
        closes = 100 * np.cumprod(1 + values)
        for day, close in zip(dates, closes, strict=True):
            bars.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "adj_close": close,
                    "volume": 1000000,
                    "available_at": calendar.close(day),
                }
            )
    (directory / "sample_documents.json").write_text(json.dumps(documents, indent=2) + "\n")
    write_parquet(pd.DataFrame(bars), directory / "sample_prices.parquet")
