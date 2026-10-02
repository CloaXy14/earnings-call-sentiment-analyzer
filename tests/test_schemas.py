"""Record identities, units, timezone normalization and information-time boundaries."""

import hashlib
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from ecsa.schemas import Document, EventFeatures, PriceBar, Section, Signal, document_id


def test_document_id_has_unambiguous_field_boundaries() -> None:
    expected = hashlib.sha256(b'["local","file:///sample.txt","Revenue grew."]').hexdigest()
    assert document_id("local", "file:///sample.txt", "Revenue grew.") == expected
    assert document_id("ab", "c", "d") != document_id("a", "bc", "d")


def test_document_round_trip_preserves_roles_and_identity(document_payload: dict[str, Any]) -> None:
    document = Document.model_validate(document_payload)
    assert document.doc_id == document_id(document.source, document.url, document.raw_text)
    assert document.sections[0].utterances[0].role == "executive"
    assert document.sections[1].name == "qa"
    assert document.sections[1].utterances[0].role == "analyst"
    assert Document.model_validate_json(document.model_dump_json()) == document


@pytest.mark.parametrize("field", ["source", "url", "raw_text"])
def test_document_hash_changes_with_each_source_field(
    field: str, document_payload: dict[str, Any]
) -> None:
    original = Document.model_validate(document_payload)
    changed = Document.model_validate({**document_payload, field: document_payload[field] + "x"})
    assert changed.doc_id != original.doc_id
    with pytest.raises(ValidationError, match="doc_id must match"):
        Document.model_validate({**changed.model_dump(), "doc_id": original.doc_id})


@pytest.mark.parametrize(
    ("input_time", "expected"),
    [
        ("2024-01-25T16:30:00-05:00", datetime(2024, 1, 25, 21, 30, tzinfo=UTC)),
        ("2024-07-25T16:30:00-04:00", datetime(2024, 7, 25, 20, 30, tzinfo=UTC)),
    ],
)
def test_document_normalizes_winter_and_summer_offsets(
    input_time: str, expected: datetime, document_payload: dict[str, Any]
) -> None:
    document = Document.model_validate({**document_payload, "event_datetime": input_time})
    assert document.event_datetime == expected
    assert document.event_datetime.tzinfo is UTC
    assert document.available_at == datetime(2024, 7, 25, 22, tzinfo=UTC)


@pytest.mark.parametrize("field", ["event_datetime", "available_at"])
def test_document_rejects_naive_timestamp(field: str, document_payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        Document.model_validate({**document_payload, field: "2024-07-25T16:30:00"})


def test_document_availability_cannot_be_inferred_from_call_start(
    document_payload: dict[str, Any],
) -> None:
    del document_payload["available_at"]
    with pytest.raises(ValidationError, match="available_at"):
        Document.model_validate(document_payload)


def test_records_do_not_share_mutable_lists() -> None:
    one = Section(name="qa")
    two = Section(name="qa")
    assert one.utterances is not two.utterances


@pytest.mark.parametrize(
    ("field", "value"),
    [("cik", "1"), ("doc_type", "scraped_article"), ("session_timing", "ET"), ("ticker", "")],
)
def test_document_invalid_values_are_rejected(
    field: str, value: str, document_payload: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        Document.model_validate({**document_payload, field: value})


def feature_payload() -> dict[str, Any]:
    """Return a synthetic feature record available at day-0 close (UTC)."""
    return {
        "event_id": "test-event",
        "ticker": "TEST",
        "event_datetime": "2024-07-26T13:00:00Z",
        "available_at": "2024-07-26T20:00:00Z",
        "source_available_at": "2024-07-26T19:00:00Z",
        "finbert_net_mean": 0.25,
        "pos_share": 0.5,
        "neg_share": 0.25,
    }


def test_features_reject_future_inputs_but_accept_known_at_formation() -> None:
    payload = feature_payload()
    payload["source_available_at"] = payload["available_at"]
    assert EventFeatures.model_validate(payload).finbert_net_mean == 0.25
    payload["source_available_at"] = "2024-07-26T20:00:01Z"
    with pytest.raises(ValidationError, match="future"):
        EventFeatures.model_validate(payload)


@pytest.mark.parametrize("field", ["event_datetime", "available_at", "source_available_at"])
def test_feature_timestamps_reject_naive_values(field: str) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        EventFeatures.model_validate({**feature_payload(), field: "2024-07-26T20:00:00"})


@pytest.mark.parametrize(
    "patch",
    [
        {"finbert_net_mean": 25},
        {"fls_share": 50},
        {"uncertainty_per_1k": -1},
        {"pos_share": 0.8, "neg_share": 0.4},
    ],
)
def test_feature_units_and_ranges(patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EventFeatures.model_validate({**feature_payload(), **patch})


def signal_payload() -> dict[str, Any]:
    """Return a planned long with 5% gross exposure, formed at day-0 close."""
    return {
        "event_id": "test-event",
        "ticker": "TEST",
        "signal_date": "2024-07-26T20:00:00Z",
        "feature_available_at": "2024-07-26T19:59:00Z",
        "score": 1.5,
        "side": "LONG",
        "weight": 0.05,
        "entry_date": "2024-07-26T20:00:00Z",
        "exit_date": "2024-08-09T20:00:00Z",
    }


def test_signal_preserves_decimal_weight_and_rejects_future_information() -> None:
    payload = signal_payload()
    signal = Signal.model_validate(payload)
    assert signal.weight == 0.05
    assert signal.signal_date.tzinfo is UTC
    payload["feature_available_at"] = signal.signal_date
    assert Signal.model_validate(payload).weight == 0.05
    payload["feature_available_at"] = signal.signal_date + timedelta(seconds=1)
    with pytest.raises(ValidationError, match="future"):
        Signal.model_validate(payload)


@pytest.mark.parametrize(
    "field", ["signal_date", "feature_available_at", "entry_date", "exit_date"]
)
def test_signal_timestamps_reject_naive_values(field: str) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        Signal.model_validate({**signal_payload(), field: "2024-07-26T20:00:00"})


@pytest.mark.parametrize(
    "patch",
    [
        {"entry_date": "2024-07-25T20:00:00Z"},
        {"exit_date": "2024-07-26T20:00:00Z"},
        {"weight": 5},
        {"weight": 0},
        {"side": "NONE"},
        {"score": float("inf")},
    ],
)
def test_signal_execution_and_exposure_constraints(patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Signal.model_validate({**signal_payload(), **patch})


def test_no_trade_signal_has_zero_exposure() -> None:
    signal = Signal.model_validate({**signal_payload(), "side": "NONE", "weight": 0})
    assert signal.weight == 0


def price_payload() -> dict[str, Any]:
    """Synthetic prices in dollars/share and volume in shares."""
    return {
        "ticker": "TEST",
        "date": "2024-07-26",
        "available_at": "2024-07-26T16:00:00-04:00",
        "open": 100,
        "high": 103,
        "low": 99,
        "close": 102,
        "adj_close": 51,
        "volume": 1000,
    }


def test_price_bar_preserves_session_date_and_normalizes_availability() -> None:
    bar = PriceBar.model_validate(price_payload())
    assert bar.date == date(2024, 7, 26)
    assert bar.available_at == datetime(2024, 7, 26, 20, tzinfo=UTC)
    assert bar.available_at.tzinfo is UTC
    assert bar.adj_close == 51  # A split adjustment may fall outside raw OHLC.


@pytest.mark.parametrize(
    "patch",
    [
        {"available_at": "2024-07-26T16:00:00"},
        {"high": 99},
        {"low": 103},
        {"adj_close": 0},
        {"volume": -1},
        {"close": float("nan")},
    ],
)
def test_price_bar_rejects_invalid_data(patch: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PriceBar.model_validate({**price_payload(), **patch})
