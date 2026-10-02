"""Licensed local transcripts and the explicitly unconfigured vendor interface."""

from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from ecsa.schemas import Document, Section, SectionName, SessionTiming, SpeakerRole, Utterance
from ecsa.storage import utc


class TranscriptSource(Protocol):
    """Source of licensed, timestamped call documents."""

    def fetch(self, ticker: str, start: date, end: date) -> list[Document]: ...


def session_timing(timestamp: datetime | None) -> SessionTiming:
    """Classify call time in New York; 16:00 inclusive is DURING; missing is UNKNOWN."""
    if timestamp is None:
        return "UNKNOWN"
    local = utc(timestamp).astimezone(ZoneInfo("America/New_York")).time()
    return "BMO" if local < time(9, 30) else "DURING" if local <= time(16) else "AMC"


def parse_transcript(text: str) -> list[Section]:
    """Parse common participant lists and speaker turns, retaining Q&A context."""
    roles: dict[str, SpeakerRole] = {}
    participant_role: SpeakerRole | None = None
    in_call = False
    section: SectionName = "prepared_remarks"
    sections: dict[SectionName, list[Utterance]] = {"prepared_remarks": [], "qa": []}
    speaker = "Unknown"
    role: SpeakerRole = "unknown"
    parts: list[str] = []

    def flush() -> None:
        if parts:
            sections[section].append(Utterance(speaker=speaker, role=role, text=" ".join(parts)))
            parts.clear()

    for line in (line.strip() for line in text.splitlines()):
        if not line:
            if parts:
                parts.append("\n\n")
            continue
        lower = line.lower()
        if re.search(r"^(corporate|company|executive) participants", lower):
            participant_role = "executive"
            continue
        if re.search(r"^(conference call participants|analysts|analyst participants)", lower):
            participant_role = "analyst"
            continue
        qa = bool(
            re.search(
                r"question[ -]and[ -]answer|^q\s*&\s*a|(?:first|next) question "
                r"(?:comes|is from)|open.*(?:questions|q&a)",
                lower,
            )
        )
        if qa:
            flush()
            section, in_call, participant_role = "qa", True, None
            if not re.match(r"^(operator|[^:]+):", line):
                continue
        if lower in {"presentation", "prepared remarks", "prepared_remarks", "transcript"}:
            in_call, participant_role = True, None
            continue
        if participant_role is not None and not in_call:
            name = re.split(r"\s+[-–—]\s+|,", line)[0].strip()
            roles[name.casefold()] = participant_role
            continue
        match = re.match(r"^([^:]{1,90}):\s*(.*)$", line)
        standalone = line.casefold() in roles or lower == "operator"
        if match or standalone:
            flush()
            in_call = True
            name, rest = (match.group(1), match.group(2)) if match else (line, "")
            speaker = re.split(r"\s+[-–—]\s+", name)[0].strip()
            role = roles.get(
                speaker.casefold(), "operator" if speaker.lower() == "operator" else "unknown"
            )
            if rest:
                parts.append(rest)
        else:
            parts.append(line)
    flush()
    return [Section(name=name, utterances=turns) for name, turns in sections.items() if turns]


class LocalFileSource:
    """Read JSON objects/lists, CSV rows, and TXT plus a same-stem .meta.json sidecar.

    Required columns: ticker, event_datetime, available_at, raw_text (TXT uses its
    contents). UTC offsets are required; optional event_id links multiple sources.
    Date bounds refer to the event's New York calendar date, inclusively.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def fetch(self, ticker: str, start: date, end: date) -> list[Document]:
        """Load licensed local documents, without inference of publication time.

        Malformed files and rows raise ValueError naming the file and 1-based row.
        """
        result = []
        for path in sorted(self.directory.glob("*")):
            rows: list[dict[str, Any]] = []
            if path.name.endswith(".meta.json"):
                continue
            if path.suffix == ".json":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path.name}: invalid JSON ({error.msg})") from error
                rows = value if isinstance(value, list) else [value]
            elif path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            elif path.suffix == ".txt":
                metadata = path.with_suffix(".meta.json")
                if not metadata.exists():
                    raise ValueError(
                        f"{path.name} requires a .meta.json timestamp/licensing sidecar"
                    )
                rows = [{**json.loads(metadata.read_text()), "raw_text": path.read_text()}]
            for number, row in enumerate(rows, start=1):
                try:
                    document = self._document(row, path, ticker, start, end)
                except (KeyError, TypeError, ValueError) as error:
                    detail = f"missing field {error}" if isinstance(error, KeyError) else error
                    raise ValueError(f"{path.name} row {number}: {detail}") from error
                if document is not None:
                    result.append(document)
        return list({d.doc_id: d for d in result}.values())

    @staticmethod
    def _document(
        row: dict[str, Any], path: Path, ticker: str, start: date, end: date
    ) -> Document | None:
        """Build one Document, or None when the row is for another ticker or outside the dates."""
        if str(row["ticker"]).upper() != ticker.upper():
            return None
        if not row.get("event_datetime"):
            if not row.get("event_date"):
                raise ValueError("Local documents need event_datetime or event_date")
            event_date = date.fromisoformat(str(row.pop("event_date")))
            # Noon is a date anchor, not a guessed call time; UNKNOWN forces next session.
            row["event_datetime"] = datetime.combine(
                event_date, time(12), ZoneInfo("America/New_York")
            )
            row["session_timing"] = "UNKNOWN"
            row["event_time_known"] = False
        event = utc(row["event_datetime"])
        if not start <= event.astimezone(ZoneInfo("America/New_York")).date() <= end:
            return None
        row = {k: v for k, v in row.items() if v != "" or k in {"raw_text", "url"}}
        row.setdefault("doc_type", "transcript")
        row.setdefault("source", "local_licensed")
        row.setdefault("url", path.resolve().as_uri())
        row.setdefault("session_timing", session_timing(event))
        row.setdefault("sections", [s.model_dump() for s in parse_transcript(row["raw_text"])])
        return Document.model_validate(row)


class VendorAPISource:
    """Extension point for an explicitly licensed transcript vendor."""

    def fetch(self, ticker: str, start: date, end: date) -> list[Document]:
        """Require vendor implementation, endpoint, contract and licensed API key."""
        raise NotImplementedError(
            "Configure a licensed transcript API and TRANSCRIPT_API_KEY, then "
            "implement this adapter using HttpClient. No scraping fallback is "
            "permitted."
        )
