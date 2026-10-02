"""Known-return event studies and point-in-time strategy tests."""

from datetime import date, timedelta
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from ecsa.events.calendar import TradingCalendar
from ecsa.events.market_model import fit_market_model
from ecsa.events.stats import decision_available, information_coefficient, window_tests
from ecsa.events.study import run_study
from ecsa.settings import Settings
from ecsa.signals.backtest import backtest, placebo
from ecsa.signals.metrics import performance
from ecsa.signals.model import walk_forward
from ecsa.signals.portfolio import build_signals
from ecsa.signals.scoring import composite_scores

CONFIG = Path(__file__).resolve().parents[1] / "config/settings.yaml"


def synthetic_prices(count: int = 310) -> tuple[pd.DataFrame, TradingCalendar]:
    dates = [
        s.date() for s in xcals.get_calendar("XNYS", start="2022-01-01", end="2024-12-31").sessions
    ][:count]
    cal = TradingCalendar(dates)
    market = np.random.default_rng(7).normal(0.0002, 0.01, count)
    stock = 0.0003 + 1.2 * market
    stock[270] += 0.03
    stock[271:281] += 0.001
    rows = []
    for ticker, values in [("SPY", market), ("TEST", stock)]:
        closes = 100 * np.cumprod(1 + values)
        for day, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "adj_close": close,
                    "volume": 1000,
                    "available_at": cal.close(day),
                }
            )
    return pd.DataFrame(rows), cal


def test_market_model_known_alpha_beta_car_and_future_fit_invariance() -> None:
    prices, cal = synthetic_prices()
    day = cal.dates[270]
    stamp = cal.close(day) - timedelta(hours=7)
    events = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "event_id": "e",
                "event_datetime": stamp,
                "available_at": stamp,
                "session_timing": "BMO",
                "finbert_net_mean": 0.7,
            }
        ]
    )
    table, paths = run_study(events, prices, Settings.from_yaml(CONFIG))
    assert table.iloc[0].alpha == pytest.approx(0.0003, abs=1e-12)
    assert table.iloc[0].beta == pytest.approx(1.2, abs=1e-12)
    assert table.iloc[0].CAR_reaction == pytest.approx(0.03, abs=1e-12)
    assert table.iloc[0].CAR_drift == pytest.approx(0.01, abs=1e-12)
    assert paths.iloc[-1].car == pytest.approx(0.04, abs=1e-12)
    assert table.iloc[0].label_available_at == cal.close(cal.dates[280])
    prices.loc[prices.date > day, "adj_close"] *= 3
    changed, _ = run_study(events, prices, Settings.from_yaml(CONFIG))
    assert changed.iloc[0].beta == table.iloc[0].beta


def test_insufficient_estimation_falls_back_and_missing_windows_stay_nan() -> None:
    prices, cal = synthetic_prices()
    day = cal.dates[50]
    event = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "event_id": "e",
                "event_datetime": cal.close(day) - timedelta(hours=7),
                "available_at": cal.close(day),
                "session_timing": "BMO",
            }
        ]
    )
    table, _ = run_study(event, prices, Settings.from_yaml(CONFIG))
    assert table.iloc[0].method == "market_adjusted" and table.iloc[0].beta == 1
    assert np.isnan(table.iloc[0].SCAR_drift)
    prices = prices[~((prices.ticker == "TEST") & (prices.date == cal.dates[55]))]
    table, _ = run_study(event, prices, Settings.from_yaml(CONFIG))
    assert np.isnan(table.iloc[0].CAR_drift) and pd.isna(table.iloc[0].label_available_at)


def test_car_forecast_variance_includes_parameter_covariance() -> None:
    index = pd.date_range("2023-01-01", periods=130, freq="D").date
    market = pd.Series(np.linspace(-0.02, 0.02, 130), index=index)
    asset = 0.001 + market * 1.1 + np.random.default_rng(4).normal(0, 0.005, 130)
    model = fit_market_model(asset, market, cutoff=index[-1])
    rm = np.array([0.01, 0.02])
    expected = model.residual_variance * (
        2 + 4 / 130 + (rm.sum() - 2 * model.market_mean) ** 2 / model.market_sxx
    )
    assert model.car_variance(rm) == pytest.approx(expected)


def ranking_events(cal: TradingCalendar) -> pd.DataFrame:
    rows = []
    for i, score in zip([260, 261, 262, 270], [-2.0, 0.0, 1.0, 3.0], strict=True):
        rows.append(
            {
                "event_id": str(i),
                "ticker": "TEST",
                "day0": cal.dates[i],
                "available_at": cal.close(cal.dates[i]) - timedelta(hours=2),
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def test_ranking_future_price_perturbation_holding_period_and_capacity() -> None:
    prices, cal = synthetic_prices()
    settings = Settings.from_yaml(CONFIG).model_copy(update={"minimum_ranking_events": 3})
    events = ranking_events(cal)
    baseline, _ = build_signals(events, prices, cal, settings)
    last = baseline[baseline.event_id == "270"].iloc[0]
    assert last.exit_date == cal.close(cal.dates[280])
    # Previous TEST sleeve still occupies its 5% cap: no second overlapping allocation.
    assert last.side == "NONE"
    prices.loc[prices.date > cal.dates[270], "adj_close"] *= 10
    future = {
        "event_id": "future",
        "ticker": "TEST",
        "day0": cal.dates[285],
        "available_at": cal.close(cal.dates[285]),
        "score": -1000,
    }
    changed, _ = build_signals(
        pd.concat([events, pd.DataFrame([future])], ignore_index=True), prices, cal, settings
    )
    pd.testing.assert_frame_equal(
        baseline.reset_index(drop=True),
        changed[changed.event_id != "future"].reset_index(drop=True),
    )


def test_late_features_rejected_not_backdated() -> None:
    prices, cal = synthetic_prices()
    events = ranking_events(cal)
    events.loc[3, "available_at"] = cal.close(cal.dates[270]) + timedelta(seconds=1)
    signals, rejected = build_signals(events, prices, cal, Settings.from_yaml(CONFIG))
    assert "270" not in set(signals.event_id)
    assert rejected.iloc[0].reason == "features_unavailable_at_day0_close"


def test_past_only_expanding_scores() -> None:
    times = pd.date_range("2024-01-01", periods=6, tz="UTC")
    events = pd.DataFrame({"available_at": times, "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    settings = Settings.from_yaml(CONFIG).model_copy(update={"score_weights": {"x": 1}})
    baseline = composite_scores(events, settings)
    events.loc[4:, "x"] = -1000
    changed = composite_scores(events, settings)
    assert baseline.loc[3, "score"] == changed.loc[3, "score"] == pytest.approx(2.0)


def test_walk_forward_purges_labels_and_future_targets() -> None:
    _, cal = synthetic_prices()
    rows = []
    for i in range(220, 280):
        rows.append(
            {
                "day0": cal.dates[i],
                "available_at": cal.close(cal.dates[i]),
                "label_available_at": cal.close(cal.dates[i + 10]),
                "x": float(i % 7),
                "CAR_drift": 0.001 * (i % 7),
            }
        )
    events = pd.DataFrame(rows)
    settings = Settings.from_yaml(CONFIG).model_copy(
        update={"learned_model": "ridge", "minimum_training_events": 5, "score_weights": {"x": 1}}
    )
    baseline = walk_forward(events, cal, settings)
    target = 40
    assert baseline.loc[target, "training_latest_label"] < cal.close(cal.dates[250])
    events.loc[target:, "CAR_drift"] = 999
    changed = walk_forward(events, cal, settings)
    assert baseline.loc[target, "model_score"] == changed.loc[target, "model_score"]


def test_costs_and_entry_exit_timing_on_flat_portfolio() -> None:
    prices, cal = synthetic_prices(290)
    prices["adj_close"] = 100.0
    signals = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "side": "LONG",
                "weight": 0.05,
                "entry_date": cal.close(cal.dates[270]),
                "exit_date": cal.close(cal.dates[280]),
            }
        ]
    )
    daily = backtest(signals, prices, Settings.from_yaml(CONFIG))
    assert daily.transaction_cost.sum() == pytest.approx(2 * 0.05 * 0.001)
    assert daily.gross_return.sum() == 0
    assert daily.iloc[270]["return"] == pytest.approx(-0.00005)
    assert daily.iloc[280]["return"] == pytest.approx(-0.00005)
    assert daily.iloc[281].gross_exposure == 0


def test_bmp_and_ic_known_cross_section() -> None:
    table = pd.DataFrame(
        {
            "day0": [date(2024, 1, i) for i in range(1, 7)],
            "score": np.arange(6),
            "CAR_drift": np.arange(6) / 100,
        }
    )
    for window in ["pre", "reaction", "reaction_01", "full"]:
        table["CAR_" + window] = table.CAR_drift
    for window in ["pre", "reaction", "reaction_01", "drift", "full"]:
        table["SCAR_" + window] = table.CAR_drift / 0.01
    tests = window_tests(table)
    assert tests.loc[tests.test == "BMP", "statistic"].iloc[0] == pytest.approx(
        np.mean(np.arange(6)) / (np.std(np.arange(6), ddof=1) / np.sqrt(6))
    )
    assert information_coefficient(table, "score").iloc[0].ic == pytest.approx(1)


def test_inverse_vol_signal_ignores_future_prices() -> None:
    prices, cal = synthetic_prices()
    events = pd.DataFrame(
        [
            {
                "event_id": str(i),
                "ticker": "HISTORY" if i < 24 else "TEST",
                "day0": cal.dates[220 + i],
                "available_at": cal.close(cal.dates[220 + i]),
                "score": float(i),
            }
            for i in range(25)
        ]
    )
    settings = Settings.from_yaml(CONFIG).model_copy(
        update={"minimum_ranking_events": 25, "weighting": "inverse_vol"}
    )
    baseline, _ = build_signals(events, prices, cal, settings)
    assert baseline.iloc[-1].side == "LONG" and baseline.iloc[-1].weight == 0.05
    prices.loc[prices.date > cal.dates[244], "adj_close"] *= 999
    changed, _ = build_signals(events, prices, cal, settings)
    pd.testing.assert_frame_equal(baseline, changed)


def test_sector_neutrality_and_name_caps_with_simultaneous_events() -> None:
    prices, cal = synthetic_prices()
    events = pd.DataFrame(
        [
            {
                "event_id": str(i),
                "ticker": "TEST" if i < 2 else "OTHER",
                "day0": cal.dates[270],
                "available_at": cal.close(cal.dates[270]),
                "score": 2.0 if i < 2 else -2.0,
            }
            for i in range(3)
        ]
    )
    settings = Settings.from_yaml(CONFIG).model_copy(
        update={
            "minimum_ranking_events": 3,
            "sector_neutral": True,
            "sectors": {"TEST": "tech", "OTHER": "tech"},
        }
    )
    selected, _ = build_signals(events, prices, cal, settings)
    assert selected[selected.side == "LONG"].weight.sum() == pytest.approx(
        selected[selected.side == "SHORT"].weight.sum()
    )
    assert selected.groupby("ticker").weight.sum().max() <= 0.05


def test_100_seed_placebo_gross_returns_center_near_zero() -> None:
    dates = [
        s.date() for s in xcals.get_calendar("XNYS", start="2024-01-01", end="2024-06-01").sessions
    ][:45]
    cal = TradingCalendar(dates)
    rows = []
    for ticker, daily in [("SPY", 0.0), ("UP", 0.001), ("DOWN", -0.001)]:
        for i, day in enumerate(dates):
            close = 100 * (1 + daily) ** i
            rows.append(
                {"ticker": ticker, "date": day, "adj_close": close, "available_at": cal.close(day)}
            )
    prices = pd.DataFrame(rows)
    events = pd.DataFrame(
        [
            {
                "event_id": f"{i}-{ticker}",
                "ticker": ticker,
                "day0": dates[i],
                "available_at": cal.close(dates[i]),
                "score": score,
            }
            for i in (5, 20, 33)
            for ticker, score in [("UP", 1.0), ("DOWN", -1.0)]
        ]
    )
    settings = Settings.from_yaml(CONFIG).model_copy(update={"minimum_ranking_events": 2})
    _, results = placebo(events, prices, cal, settings)
    assert len(results) == 100 and results.total_return.nunique() > 1
    assert abs(results.gross_mean_return.mean()) < 1e-5


def test_known_metrics_include_initial_nav_in_drawdown() -> None:
    from ecsa.signals.metrics import performance

    daily = pd.DataFrame(
        {
            "return": [0.1, -0.1, 0.0],
            "turnover": [0.05, 0.0, 0.05],
            "long_return": [0.1, -0.1, 0.0],
            "short_return": [0.0, 0.0, 0.0],
        }
    )
    prices = pd.DataFrame({"date": [date(2024, 1, 2)], "ticker": ["TEST"], "adj_close": [100.0]})
    settings = Settings.from_yaml(CONFIG).model_copy(update={"variants_tried": 3})
    values = (
        performance(daily, pd.DataFrame(columns=["side"]), prices, pd.DataFrame(), settings)
        .set_index("metric")
        .value
    )
    assert values["max_drawdown"] == pytest.approx(-0.1)
    assert values["annualized_return"] == pytest.approx(0.99**84 - 1)
    assert values["variants_tried"] == 3


def test_f2_delayed_previous_close_does_not_leak_into_label() -> None:
    """F2: a return is only available once BOTH of its closes are available.

    The day-250 bar becomes available only on day 309. The day +1 return of a day-250
    event uses that close, so the drift label cannot mature before day 309, and
    perturbing the still-unavailable price must not change the learned prediction made
    at a later event. Mirrors the audit's delayed-historical-price reproducer.
    """
    prices, cal = synthetic_prices()
    settings = Settings.from_yaml(CONFIG).model_copy(
        update={"learned_model": "ridge", "minimum_training_events": 5, "score_weights": {"x": 1.0}}
    )
    events = pd.DataFrame(
        [
            {
                "event_id": str(i),
                "ticker": "TEST",
                "session_timing": "BMO",
                "event_datetime": cal.close(cal.dates[i]) - timedelta(hours=7),
                "available_at": cal.close(cal.dates[i]) - timedelta(hours=1),
                "x": float(i - 200),
            }
            for i in [200, 205, 210, 215, 220, 225, 230, 240, 250, 280]
        ]
    )
    mask = (prices.ticker == "TEST") & (prices.date == cal.dates[250])
    prices.loc[mask, "available_at"] = cal.close(cal.dates[309])

    def predict(frame: pd.DataFrame) -> tuple[float, object]:
        table, _ = run_study(events, frame, settings)
        prediction = walk_forward(table, cal, settings)
        label = table[table.event_id == "250"].iloc[0].label_available_at
        return float(prediction.iloc[-1].model_score), label

    score_before, label_before = predict(prices)
    # The day-250 close is unavailable until day 309, so the label matures no earlier.
    assert pd.notna(label_before)
    assert pd.Timestamp(label_before) >= cal.close(cal.dates[309])
    bumped = prices.copy()
    bumped.loc[mask, "adj_close"] *= 2
    score_after, label_after = predict(bumped)
    # Perturbing the unavailable price changes neither the prediction nor the label time.
    assert score_before == pytest.approx(score_after, abs=1e-10)
    assert pd.Timestamp(label_after) == pd.Timestamp(label_before)


def test_f4_missing_benchmark_bar_invalidates_window_without_shifting_horizon() -> None:
    """F4: a missing SPY bar keeps the +10 session scheduled and voids the window.

    Removing one benchmark bar must not slide day +10 onto a later observed session;
    the horizon stays anchored to the exchange schedule, the drift window becomes
    NaN, and the drift label does not mature.
    """
    prices, cal = synthetic_prices()
    settings = Settings.from_yaml(CONFIG)
    event = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "event_id": "e",
                "event_datetime": cal.close(cal.dates[270]) - timedelta(hours=7),
                "available_at": cal.close(cal.dates[270]),
                "session_timing": "BMO",
                "finbert_net_mean": 0.5,
            }
        ]
    )
    full, _ = run_study(event, prices, settings)
    assert full.iloc[0].CAR_drift == pytest.approx(0.01, abs=1e-9)
    gap = prices[~((prices.ticker == "SPY") & (prices.date == cal.dates[275]))]
    changed, path = run_study(event, gap, settings)
    day_ten = path.loc[path.relative_day == 10, "date"].iloc[0]
    assert day_ten == cal.dates[280]
    assert pd.isna(changed.iloc[0].CAR_drift)
    assert pd.isna(changed.iloc[0].label_available_at)


def test_f6_predictive_ic_excludes_features_available_after_outcome() -> None:
    """F6: features knowable only after the outcome are dropped from predictive IC.

    Four perfectly ranked scores arrive at day 309, well after their outcomes. They
    are untradeable, so the predictive information coefficient excludes them (NaN),
    even though the retrospective correlation over all events is a perfect 1.0.
    """
    prices, cal = synthetic_prices()
    settings = Settings.from_yaml(CONFIG)
    events = pd.DataFrame(
        [
            {
                "event_id": str(i),
                "ticker": "TEST",
                "day0": cal.dates[260 + i],
                "available_at": cal.close(cal.dates[309]),
                "score": float(i),
                "CAR_drift": i / 100,
            }
            for i in range(4)
        ]
    )
    signals, rejected = build_signals(events, prices, cal, settings)
    assert signals[signals.side != "NONE"].empty
    assert len(rejected) == 4
    daily = backtest(signals, prices, settings)
    metrics = performance(daily, signals, prices, events, settings).set_index("metric").value
    assert np.isnan(metrics["ic"])
    assert metrics["ic_retrospective"] == pytest.approx(1.0)
    assert decision_available(events, cal).empty


def test_f7_hit_rate_matches_daily_rebalanced_ledger() -> None:
    """F7: trade statistics come from the same daily ledger as the equity curve.

    A 5% short over 100 -> 200 -> 99 loses money once the position is rebalanced to
    target daily, so hit rate is 0 and the average loss equals the equity change —
    not the +1% "win" that endpoint buy-and-hold accounting reported.
    """
    _, cal = synthetic_prices()
    settings = Settings.from_yaml(CONFIG).model_copy(
        update={"transaction_cost_bps": 0.0, "borrow_cost_bps_annual": 0.0}
    )
    dates = cal.dates[270:281]
    prices = pd.DataFrame(
        [
            {"ticker": ticker, "date": day, "adj_close": value, "available_at": cal.close(day)}
            for ticker, values in [("SPY", [100.0] * 11), ("TEST", [100.0, 200.0] + [99.0] * 9)]
            for day, value in zip(dates, values, strict=True)
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "side": "SHORT",
                "weight": 0.05,
                "entry_date": cal.close(cal.dates[270]),
                "exit_date": cal.close(cal.dates[280]),
            }
        ]
    )
    daily = backtest(signals, prices, settings)
    metrics = (
        performance(daily, signals, prices, pd.DataFrame(), settings).set_index("metric").value
    )
    equity_change = float(daily.equity.iloc[-1]) - 1
    assert equity_change < 0
    assert metrics["completed_trades"] == 1
    assert metrics["hit_rate"] == 0.0
    assert metrics["average_loss"] == pytest.approx(equity_change)
