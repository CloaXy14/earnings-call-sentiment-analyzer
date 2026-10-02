#!/usr/bin/env python3
"""Smoke-run the sentiment pipeline over verified local sample transcripts.

This is a manual smoke check, not an accuracy benchmark and not a regression test
(pytest's testpaths does not collect it). Its contract is strict (F5):

* any pipeline exception propagates and the process exits nonzero;
* empty feature output is a FAILURE, not a silent success;
* the unverified staging artifact produced by ``download_data.py`` is rejected,
  because scoring fabricated issuer/event metadata would be meaningless.

Point ``--data-dir`` at a directory of licensed documents whose ticker, event time
and availability have been verified (the format ``LocalFileSource`` expects).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from ecsa.data.transcripts import LocalFileSource
from ecsa.nlp.features import analyze_documents
from ecsa.nlp.finbert import FinBertScorer
from ecsa.settings import Settings


def _discover_tickers(data_dir: Path) -> set[str]:
    """Collect tickers from verified JSON/CSV sample files, ignoring staging artifacts."""
    tickers: set[str] = set()
    for path in sorted(data_dir.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("metadata_status") == "unverified":
            continue
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            ticker = row.get("ticker") if isinstance(row, dict) else None
            if ticker:
                tickers.add(str(ticker))
    return tickers


def run(data_dir: Path, config: Path) -> int:
    """Return 0 only on a successful run that produced at least one feature row."""
    staging = data_dir / "real_sample_transcripts.json"
    if staging.exists():
        try:
            payload = json.loads(staging.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("metadata_status") == "unverified":
            print(
                f"Error: {staging} holds unverified metadata (ticker/event time are "
                "null). Enrich it with verified provenance before scoring.",
                file=sys.stderr,
            )
            return 1

    source = LocalFileSource(data_dir)
    settings = Settings.from_yaml(config)
    tickers = _discover_tickers(data_dir)
    if not tickers:
        print(
            f"Error: no verified documents with a ticker found in {data_dir}.",
            file=sys.stderr,
        )
        return 1
    documents = []
    for ticker in sorted(tickers):
        documents.extend(source.fetch(ticker=ticker, start=date(1900, 1, 1), end=date(2100, 1, 1)))
    if not documents:
        print(f"Error: no verified documents loaded from {data_dir}.", file=sys.stderr)
        return 1
    print(f"Loaded {len(documents)} document(s) for tickers: {', '.join(sorted(tickers))}")

    scorer = FinBertScorer(settings)
    result = analyze_documents(documents, scorer, settings)
    if result.features.empty:
        print("Error: pipeline produced no feature rows.", file=sys.stderr)
        return 1

    print(f"--- SENTIMENT SCORES ({len(result.features)} rows) ---")
    for _, row in result.features.iterrows():
        print(f"Ticker: {row['ticker']} | Event: {row['event_datetime']}")
        print(f"  FinBERT net mean: {row.get('finbert_net_mean', 'N/A')}")
        print(f"  Lexicon net:      {row.get('lexicon_net', 'N/A')}")
        print(f"  FLS net:          {row.get('fls_net', 'N/A')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    args = parser.parse_args(argv)
    return run(args.data_dir, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
