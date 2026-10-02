"""Synthetic fixtures; pytest-socket disables socket use across the entire suite."""

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def document_payload() -> dict[str, Any]:
    """Return a fresh synthetic document, with no licensed or live source data."""
    return dict(json.loads((Path(__file__).parent / "fixtures" / "document.json").read_text()))


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep test results independent of any real credentials in the parent process."""
    for key in ("SEC_USER_AGENT", "OPENAI_API_KEY", "TRANSCRIPT_API_KEY"):
        monkeypatch.delenv(key, raising=False)
