"""CLI acceptance through the installed ecsa module; no models or data downloaded."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from ecsa.pipeline import build_parser, main

STAGES = ["ingest", "analyze", "event-study", "signals", "all"]


def test_module_help_works_without_config_from_any_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ecsa.pipeline", "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert all(stage in result.stdout for stage in STAGES)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_has_help(stage: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([stage, "--help"])
    assert exc.value.code == 0
    assert "--config" in capsys.readouterr().out


@pytest.mark.parametrize("stage", STAGES)
def test_missing_inputs_fail_clearly(
    stage: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([stage])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "arguments", [["--config", "custom.yaml", "all"], ["all", "--config", "custom.yaml"]]
)
def test_config_supported_before_and_after_stage(arguments: list[str]) -> None:
    assert build_parser().parse_args(arguments).config == Path("custom.yaml")


def test_ingest_parser_accepts_requested_shape() -> None:
    args = build_parser().parse_args(
        ["ingest", "--tickers", "AAPL", "MSFT", "--start", "2019-01-01", "--end", "2025-12-31"]
    )
    assert args.tickers == ["AAPL", "MSFT"]
    assert args.start.isoformat() == "2019-01-01"
    assert args.end.isoformat() == "2025-12-31"


def test_all_pipeline_modules_import() -> None:
    modules = {
        "data": ["edgar", "transcripts", "prices", "cache", "ratelimit"],
        "nlp": ["cleaning", "segment", "finbert", "llm", "lexicon", "fls", "themes", "features"],
        "events": ["calendar", "market_model", "study", "stats"],
        "signals": ["scoring", "model", "portfolio", "backtest", "metrics"],
    }
    for package, names in modules.items():
        for name in names:
            importlib.import_module(f"ecsa.{package}.{name}")
