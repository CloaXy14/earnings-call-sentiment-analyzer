"""Atomic Parquet storage and UTC conversion helpers."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ecsa.schemas import Document


def utc(value: Any) -> datetime:
    """Parse a timezone-aware instant and normalize to UTC; reject naive values."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required")
    return cast(datetime, timestamp.to_pydatetime().astimezone(UTC))


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a DataFrame without an index; returns remain decimal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
    os.close(fd)
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def save_documents(documents: list[Document], path: Path) -> None:
    """Store UTC timestamps as Parquet timestamps and nested sections as JSON."""
    rows = []
    for doc in documents:
        row = doc.model_dump()
        row["sections"] = json.dumps([s.model_dump() for s in doc.sections])
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(Document.model_fields))
    for column in ("event_datetime", "available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    write_parquet(frame, path)


def load_documents(path: Path) -> list[Document]:
    """Load and validate persisted source identities and timestamps."""
    frame = pd.read_parquet(path)
    return [
        Document.model_validate({**row, "sections": json.loads(row["sections"])})
        for row in frame.to_dict("records")
    ]
