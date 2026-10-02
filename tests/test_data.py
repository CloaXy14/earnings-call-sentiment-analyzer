"""Offline collection, cache, transcript and market-session contract checks."""

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from ecsa.data.cache import DiskCache, HttpClient
from ecsa.data.edgar import EdgarClient, extract_mdna, pad_cik, parse_submissions
from ecsa.data.prices import daily_returns
from ecsa.data.ratelimit import TokenBucket
from ecsa.data.transcripts import LocalFileSource, VendorAPISource, parse_transcript, session_timing
from ecsa.events.calendar import TradingCalendar
from ecsa.settings import Settings
from ecsa.storage import load_documents, save_documents

CONFIG = Path(__file__).resolve().parents[1] / "config/settings.yaml"
TRANSCRIPT = """Corporate Participants
Alice Example - CEO
Analysts
Bob Example - Research
Presentation
Operator: Welcome to the call.
Alice Example: Record revenue grew this quarter.
Question-and-Answer Session
Bob Example: What is your guidance?
Alice Example: We expect strong demand next quarter.
"""


@pytest.mark.parametrize(
    "value,expected", [("320193", "0000320193"), (1, "0000000001"), ("0000320193", "0000320193")]
)
def test_cik(value: str | int, expected: str) -> None:
    assert pad_cik(value) == expected


def test_saved_submissions() -> None:
    payload = json.loads((Path(__file__).parent / "fixtures/submissions.json").read_text())
    rows = parse_submissions(payload, "1")
    assert rows[0].items == ("2.02", "9.01")
    assert rows[0].acceptance_datetime == datetime(2024, 7, 25, 20, 30, tzinfo=UTC)
    assert rows[1].items == ("5.02",)


def test_edgar_pagination_and_item_filter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Synthetic Tester test@example.invalid")
    recent = json.loads((Path(__file__).parent / "fixtures/submissions.json").read_text())
    recent["filings"]["files"] = [
        {"name": "older.json", "filingFrom": "2023-01-01", "filingTo": "2023-12-31"}
    ]
    older = {
        "accessionNumber": ["0001-23-000001"],
        "form": ["10-K"],
        "filingDate": ["2023-02-01"],
        "acceptanceDateTime": ["2023-02-01T20:00:00Z"],
        "primaryDocument": ["annual.htm"],
    }
    calls = []

    def response(url: str) -> object:
        calls.append(url)
        return older if url.endswith("older.json") else recent

    http = SimpleNamespace(json=response, request=lambda url: b"<html></html>")
    client = EdgarClient(Settings.from_yaml(CONFIG), http=http)
    filings = client.list_filings("1", ["10-K", "8-K"], date(2023, 1, 1), date(2024, 12, 31))
    assert len(filings) == 3
    assert any(url.endswith("older.json") for url in calls)
    assert client.earnings_8ks("1", date(2024, 1, 1), date(2024, 12, 31)) == []


def test_mdna_requires_real_unambiguous_headings() -> None:
    body = "Management discusses revenue and expenses for the reporting period. " * 10
    html = (
        f"<h2>Item 2. Management’s Discussion and Analysis</h2><p>{body}</p>"
        "<h2>Item 3. Market Risk</h2>"
    )
    assert body.strip() in extract_mdna(html, "10-Q")
    assert extract_mdna("<p>Item 2. Management Discussion</p><p>Item 3. Risk</p>", "10-Q") is None
    assert (
        extract_mdna("<p>Management discussed expenses without item boundaries.</p>", "10-Q")
        is None
    )
    assert extract_mdna(html + html, "10-Q") is None


def test_cache_ttl_bytes_json_and_parameter_order(tmp_path: Path) -> None:
    now = [100.0]
    cache = DiskCache(tmp_path, 10, clock=lambda: now[0])
    cache.put_json("test", {"a": 2}, {"b": 2, "a": 1})
    assert cache.get_json("test", {"a": 1, "b": 2}) == {"a": 2}
    cache.put_bytes("bytes", b"\x00\xff")
    assert cache.get_bytes("bytes") == b"\x00\xff"
    now[0] = 110
    assert cache.get_json("test", {"a": 1, "b": 2}) is None


def test_token_bucket_known_elapsed_time() -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    limiter = TokenBucket(8, clock=lambda: now[0], sleep=sleep)
    for _ in range(9):
        limiter.acquire()
    assert now[0] == pytest.approx(1)


def test_http_retries_then_caches(tmp_path: Path) -> None:
    calls = []
    waits = []

    def request(*args: object, **kwargs: object) -> object:
        calls.append(args)
        return SimpleNamespace(
            status_code=429 if len(calls) == 1 else 200,
            headers={"Retry-After": "2"},
            content=b'{"ok":true}',
            raise_for_status=lambda: None,
        )

    http = HttpClient(
        DiskCache(tmp_path), session=SimpleNamespace(request=request), sleep=waits.append
    )
    http.limiter = SimpleNamespace(acquire=lambda: None)
    assert http.json("https://example.invalid") == {"ok": True}
    assert http.json("https://example.invalid") == {"ok": True}
    assert len(calls) == 2 and waits == [2]


def test_http_does_not_retry_404(tmp_path: Path) -> None:
    def fail() -> None:
        raise requests.HTTPError("404")

    http = HttpClient(
        DiskCache(tmp_path),
        session=SimpleNamespace(
            request=lambda *a, **kw: SimpleNamespace(status_code=404, raise_for_status=fail)
        ),
        sleep=lambda x: None,
    )
    with pytest.raises(requests.HTTPError):
        http.request("https://example.invalid/missing")


def test_transcript_roles_qa_and_operator_cue() -> None:
    sections = parse_transcript(TRANSCRIPT)
    assert [s.name for s in sections] == ["prepared_remarks", "qa"]
    assert sections[0].utterances[1].role == "executive"
    assert sections[1].utterances[0].role == "analyst"
    assert (
        parse_transcript(
            TRANSCRIPT.replace(
                "Question-and-Answer Session", "Operator: Our first question comes from Bob."
            )
        )[1].name
        == "qa"
    )


@pytest.mark.parametrize(
    "hour,minute,expected", [(9, 29, "BMO"), (9, 30, "DURING"), (16, 0, "DURING"), (16, 1, "AMC")]
)
@pytest.mark.parametrize("day,offset", [("2024-01-25", "-05:00"), ("2024-07-25", "-04:00")])
def test_session_boundaries(hour: int, minute: int, expected: str, day: str, offset: str) -> None:
    assert (
        session_timing(datetime.fromisoformat(f"{day}T{hour:02}:{minute:02}:00{offset}"))
        == expected
    )
    assert session_timing(None) == "UNKNOWN"


def test_local_source_and_parquet_roundtrip(tmp_path: Path) -> None:
    payload = {
        "ticker": "TEST",
        "event_datetime": "2024-07-25T20:30:00Z",
        "available_at": "2024-07-25T22:00:00Z",
        "raw_text": TRANSCRIPT,
    }
    (tmp_path / "call.json").write_text(json.dumps(payload))
    docs = LocalFileSource(tmp_path).fetch("TEST", date(2024, 7, 25), date(2024, 7, 25))
    assert len(docs) == 1 and docs[0].session_timing == "AMC"
    save_documents(docs, tmp_path / "docs.parquet")
    assert load_documents(tmp_path / "docs.parquet") == docs
    with pytest.raises(NotImplementedError, match="licensed"):
        VendorAPISource().fetch("TEST", date.today(), date.today())


def test_day0_weekend_holiday_and_half_day() -> None:
    cal = TradingCalendar([date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8)])
    assert cal.day0(datetime(2024, 7, 3, 12, tzinfo=UTC), "BMO") == date(2024, 7, 3)
    assert cal.day0(datetime(2024, 7, 3, 22, tzinfo=UTC), "AMC") == date(2024, 7, 5)
    assert cal.day0(datetime(2024, 7, 6, 12, tzinfo=UTC), "BMO") == date(2024, 7, 8)
    assert cal.close(date(2024, 7, 3)) == datetime(2024, 7, 3, 17, tzinfo=UTC)


def test_simple_returns_are_decimal_without_fill() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["TEST"] * 3,
            "date": [date(2024, 1, i) for i in (2, 3, 4)],
            "adj_close": [100.0, 110.0, 99.0],
        }
    )
    assert daily_returns(frame)["return"].iloc[1:].tolist() == pytest.approx([0.1, -0.1])


def test_ex99_only_downloaded_for_item_202(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Synthetic Test test@example.invalid")
    payload = json.loads((Path(__file__).parent / "fixtures/submissions.json").read_text())
    calls = []
    index = (
        b'<table class="tableFile"><tr><td>2</td><td>Press release</td>'
        b'<td><a href="release.htm">release</a></td><td>EX-99.1</td></tr></table>'
    )

    def request(url: str) -> bytes:
        calls.append(url)
        return index if url.endswith("-index.html") else b"<p>Revenue increased.</p>"

    http = SimpleNamespace(json=lambda url: payload, request=request)
    docs = EdgarClient(Settings.from_yaml(CONFIG), http=http).earnings_8ks(
        "1", date(2024, 1, 1), date(2024, 12, 31), "TEST"
    )
    assert len(docs) == 1 and docs[0].doc_type == "8k_press_release"
    assert len(calls) == 2 and "000124000001/release.htm" in calls[1]
    assert docs[0].available_at == datetime(2024, 7, 25, 20, 30, tzinfo=UTC)


def test_txt_and_csv_licensed_sources(tmp_path: Path) -> None:
    payload = {
        "ticker": "TEST",
        "event_datetime": "2024-07-25T20:00:00Z",
        "available_at": "2024-07-25T22:00:00Z",
    }
    (tmp_path / "call.txt").write_text("CEO: Strong demand.")
    (tmp_path / "call.meta.json").write_text(json.dumps(payload))
    (tmp_path / "calls.csv").write_text(
        "ticker,event_datetime,available_at,raw_text\n"
        "TEST,2024-07-26T13:00:00Z,2024-07-26T14:00:00Z,CEO: Raised guidance.\n"
    )
    docs = LocalFileSource(tmp_path).fetch("TEST", date(2024, 7, 1), date(2024, 7, 31))
    assert len(docs) == 2 and all(d.sections for d in docs)


def test_yfinance_adapter_cache_avoids_second_download(tmp_path: Path) -> None:
    from ecsa.data.prices import YFinancePriceSource

    calls = []

    def download(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Adj Close": [50.0],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex(["2024-07-05"], name="Date"),
        )

    settings = Settings.from_yaml(CONFIG).model_copy(update={"data_dir": tmp_path})
    source = YFinancePriceSource(settings, downloader=download)
    first = source.daily_bars("TEST", date(2024, 7, 5), date(2024, 7, 5))
    second = source.daily_bars("TEST", date(2024, 7, 5), date(2024, 7, 5))
    assert len(calls) == 1 and calls[0]["end"] == "2024-07-06"
    pd.testing.assert_frame_equal(first, second)


def test_date_only_transcript_is_marked_unknown(tmp_path: Path) -> None:
    (tmp_path / "unknown.json").write_text(
        json.dumps(
            {
                "ticker": "TEST",
                "event_date": "2024-07-05",
                "available_at": "2024-07-06T12:00:00Z",
                "raw_text": "CEO: Stable results.",
            }
        )
    )
    doc = LocalFileSource(tmp_path).fetch("TEST", date(2024, 7, 5), date(2024, 7, 5))[0]
    assert doc.session_timing == "UNKNOWN" and not doc.event_time_known
    cal = TradingCalendar([date(2024, 7, 5), date(2024, 7, 8)])
    assert cal.day0(doc.event_datetime, doc.session_timing) == date(2024, 7, 8)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("bad.json", "{not json", "bad.json: invalid JSON"),
        ("rows.csv", "event_datetime,raw_text\n2024-07-26T13:00:00Z,CEO: Hi.\n", "rows.csv row 1"),
        (
            "one.json",
            '{"event_datetime": "2024-07-26T13:00:00Z", "raw_text": "x"}',
            "one.json row 1",
        ),
    ],
)
def test_local_source_names_the_file_for_malformed_input(
    tmp_path: Path, name: str, content: str, message: str
) -> None:
    (tmp_path / name).write_text(content)
    with pytest.raises(ValueError, match=re.escape(message)):
        LocalFileSource(tmp_path).fetch("TEST", date(2024, 7, 1), date(2024, 7, 31))


def test_local_source_ignores_unrelated_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("notes")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    assert LocalFileSource(tmp_path).fetch("TEST", date(2024, 7, 1), date(2024, 7, 31)) == []
