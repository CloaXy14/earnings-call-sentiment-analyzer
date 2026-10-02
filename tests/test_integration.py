"""Full offline sample pipeline and Streamlit interactions; sockets remain disabled."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ecsa.dashboard import highlighted, signal_explanation
from ecsa.nlp.llm import LlmScorer
from ecsa.pipeline import analyze, event_study, ingest, signals
from ecsa.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings = Settings.from_yaml(ROOT / "config/settings.yaml").model_copy(
        update={
            "data_dir": tmp_path_factory.mktemp("demo"),
            "nlp_backend": "demo",
            "placebo_seeds": 3,
        }
    )
    assert ingest(settings, None, None, None, demo=True)["documents"] == 96
    assert analyze(settings)["events"] == 96
    assert event_study(settings)["event_study_rows"] == 96
    assert signals(settings)["signals"] > 0
    return settings


def test_offline_pipeline_artifacts_and_availability(sample_run: Settings) -> None:
    directory = sample_run.processed_dir
    expected = {
        "features",
        "sentences",
        "phrases",
        "removed_text",
        "event_study",
        "event_paths",
        "event_buckets",
        "bucket_paths",
        "window_tests",
        "information_coefficients",
        "regression",
        "signals",
        "rejected_signals",
        "backtest",
        "placebo",
        "placebo_seeds",
        "strategy_metrics",
        "strategy_ic",
        "run_metadata",
    }
    assert expected <= {p.stem for p in directory.glob("*.parquet")}
    signal = pd.read_parquet(directory / "signals.parquet")
    assert (signal.feature_available_at <= signal.signal_date).all()
    assert (signal.entry_date < signal.exit_date).all()
    assert signal.weight.max() <= 0.05
    assert pd.read_parquet(directory / "backtest.parquet")["return"].notna().all()
    assert pd.read_parquet(directory / "run_metadata.parquet").iloc[0].demo
    assert len(pd.read_parquet(directory / "placebo_seeds.parquet")) == 3


def test_streamlit_app_controls(sample_run: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Settings, "from_yaml", classmethod(lambda cls, path: sample_run))
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert len(app.get("plotly_chart")) == 3
    assert len(app.selectbox(key="ticker").options) == 8
    app.selectbox(key="ticker").select("SYN03").run()
    assert not app.exception
    app.selectbox(key="score").select("lexicon_net").run()
    app.selectbox(key="bucket").select("trailing_history").run()
    assert not app.exception and len(app.metric) == 4
    assert any(c.value.startswith("Signal ") for c in app.caption)


def test_signal_explanation_states_reason_and_composite_score() -> None:
    ranked = pd.DataFrame([{"side": "LONG", "score": 1.234, "reason": "ranked"}])
    assert signal_explanation(ranked) == (
        "Signal LONG: composite score +1.23; "
        "ranked in the top or bottom quintile of the trailing window."
    )
    middle = pd.DataFrame([{"side": "NONE", "score": float("nan"), "reason": "new_reason"}])
    assert signal_explanation(middle) == "Signal NONE: new reason."
    assert signal_explanation(pd.DataFrame()).startswith("Signal: no record")


def test_html_evidence_is_escaped() -> None:
    result = highlighted("<script>alert(1)</script> strong demand", "strong demand")
    assert "<script>" not in result and "<mark>strong demand</mark>" in result


def test_optional_llm_schema_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    settings = Settings.from_yaml(ROOT / "config/settings.yaml").model_copy(
        update={"llm_enabled": True}
    )

    def response(url: str, *, body: dict) -> dict:
        assert body["temperature"] == 0 and body["response_format"]["json_schema"]["strict"]
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"sentiment":0.5,"themes":["growth"],'
                            '"forward_looking":["next quarter"]}'
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    scorer = LlmScorer(settings, http=SimpleNamespace(json=response))
    assert scorer.score(["We expect growth next quarter."])[0].net == 0.5
    assert scorer.details[0].themes == ["growth"]
