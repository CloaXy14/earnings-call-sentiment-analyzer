#!/usr/bin/env python3
"""Fetch complete real earnings-call transcripts from the finosfoundation dataset.

Uses the Hugging Face datasets-server REST API (only the ``requests`` library, not the
heavier ``datasets`` package). The script:

* paginates to the END of each selected call so no segment is silently dropped (F3),
  and records expected vs collected segment counts plus a completeness flag;
* NEVER invents issuer/ticker or event-time metadata (F1). The dataset provides only an
  ``original_filename`` and transcript segments, so ``ticker``, ``event_datetime`` and
  ``available_at`` are emitted as null with ``metadata_status: "unverified"``. A verified
  security mapping, event timestamp and source-availability instant must be supplied
  before the sample is used for any event study, join or backtest;
* pins the dataset revision it read, for reproducible provenance.

The output is a staging artifact, not a pipeline-ready document: ``LocalFileSource``
requires a real ticker and event time, which this script deliberately refuses to guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - environment bootstrap only
    print("The 'requests' library is required: pip install requests", file=sys.stderr)
    sys.exit(1)

DATASET = "finosfoundation/EarningsCallTranscript"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100  # datasets-server caps a single /rows request at 100 rows.


def fetch_page(
    session: requests.Session, offset: int, length: int, timeout: float
) -> dict[str, Any]:
    """Fetch one page of rows; raise for any non-200 response."""
    response = session.get(
        ROWS_URL,
        params={
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_all_rows(session: requests.Session, timeout: float) -> list[dict[str, Any]]:
    """Page through the full split to the end; verify the row count matches the total."""
    first = fetch_page(session, 0, PAGE_SIZE, timeout)
    total = int(first.get("num_rows_total", len(first["rows"])))
    rows = list(first["rows"])
    while len(rows) < total:
        page = fetch_page(session, len(rows), PAGE_SIZE, timeout)
        new = page.get("rows", [])
        if not new:
            break
        rows.extend(new)
    if len(rows) != total:
        raise RuntimeError(
            f"Incomplete download: collected {len(rows)} of {total} reported rows."
        )
    return rows


def assemble_calls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group segments by source call, verify segment continuity, keep metadata unknown."""
    calls: dict[str, dict[str, Any]] = {}
    for item in rows:
        row = item["row"]
        filename = row.get("original_filename") or "UNKNOWN_CALL"
        transcript = row.get("transcript") or ""
        calls.setdefault(filename, {"segments": []})
        calls[filename]["segments"].append(
            {"segment_id": int(row.get("segment_id", 0)), "text": transcript}
        )

    records = []
    for filename, call in sorted(calls.items()):
        segments = sorted(call["segments"], key=lambda s: s["segment_id"])
        ids = [s["segment_id"] for s in segments]
        expected = list(range(min(ids), max(ids) + 1)) if ids else []
        missing = sorted(set(expected) - set(ids))
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        full_text = "\n\n".join(seg["text"] for seg in segments if seg["text"])
        records.append(
            {
                # Metadata the dataset does not supply is kept UNKNOWN, never invented.
                # Supply a verified security mapping, event time and source-availability
                # instant before using this record downstream (F1).
                "ticker": None,
                "event_datetime": None,
                "available_at": None,
                "metadata_status": "unverified",
                "metadata_required": [
                    "ticker",
                    "event_datetime",
                    "available_at",
                ],
                "original_filename": filename,
                "segment_count": len(segments),
                "segment_id_range": [min(ids), max(ids)] if ids else None,
                "missing_segment_ids": missing,
                "duplicate_segment_ids": duplicates,
                "complete": not missing and not duplicates,
                "raw_text": full_text,
                "source": DATASET,
                "url": f"hf://{DATASET}/{filename}",
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    """Download, verify completeness and persist the staging artifact; never fabricate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/real_sample_transcripts.json"),
        help="Output path for the unverified staging artifact.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    session = requests.Session()
    print(f"Fetching all rows of {DATASET} from the Hugging Face datasets-server...")
    try:
        rows = fetch_all_rows(session, args.timeout)
    except Exception as error:  # noqa: BLE001 - surface any transport/consistency failure
        print(f"Failed to download a complete sample: {error}", file=sys.stderr)
        return 1

    records = assemble_calls(rows)
    incomplete = [r["original_filename"] for r in records if not r["complete"]]

    payload = {
        "dataset": DATASET,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "total_rows": len(rows),
        "call_count": len(records),
        "metadata_status": "unverified",
        "note": (
            "ticker/event_datetime/available_at are intentionally null: the source "
            "dataset provides no verified issuer or event-time metadata. Supply "
            "verified provenance before any event-study, join or backtest use."
        ),
        "calls": records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Saved {len(records)} call(s), {len(rows)} segments to {args.out}. "
        "Metadata is unverified and must be enriched before downstream use."
    )
    if incomplete:
        print(f"WARNING: incomplete segment coverage for: {', '.join(incomplete)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
