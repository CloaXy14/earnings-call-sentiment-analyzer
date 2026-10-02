"""Daily target-weight backtest with overlapping sleeves, rebalancing and all-side costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecsa.events.calendar import TradingCalendar
from ecsa.settings import Settings
from ecsa.signals.portfolio import build_signals


def backtest(signals: pd.DataFrame, prices: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Daily decimal returns; enter at close, earn next session, exit after +10 return.

    Sleeves maintain their target NAV weights with daily close rebalancing. Costs
    apply to every change from drifted holdings, including entry and exit; short
    borrow accrues over actual calendar-day gaps. Cash earns zero. Missing held
    prices are an error, never silently converted to a zero return.
    """
    benchmark = prices[prices.ticker == settings.benchmark].sort_values("date")
    calendar = TradingCalendar(list(benchmark.date))
    adjusted = prices.pivot(index="date", columns="ticker", values="adj_close").reindex(
        calendar.dates
    )
    returns = adjusted.pct_change(fill_method=None)
    dates = calendar.dates
    closes = pd.DatetimeIndex([calendar.close(day) for day in dates])
    tickers = list(adjusted.columns)
    lookup = {ticker: i for i, ticker in enumerate(tickers)}
    targets = np.zeros(adjusted.shape)
    for _, signal in signals.iterrows():
        if signal.side != "NONE":
            if signal.ticker not in lookup:
                raise ValueError(f"Missing prices for {signal.ticker}")
            active = (closes >= signal.entry_date) & (closes < signal.exit_date)
            targets[active, lookup[signal.ticker]] += signal.weight * (
                1 if signal.side == "LONG" else -1
            )
    available = prices.pivot(index="date", columns="ticker", values="available_at").reindex(
        index=dates, columns=tickers
    )
    known = np.column_stack(
        [pd.to_datetime(available[t], utc=True).to_numpy() <= closes.to_numpy() for t in tickers]
    )
    raw_values, price_values = returns.to_numpy(), adjusted.to_numpy()
    holdings = np.zeros(len(tickers))
    output = []
    equity, benchmark_equity = 1.0, 1.0
    benchmark_index = lookup[settings.benchmark]
    for i, day in enumerate(dates):
        active = holdings != 0
        if (np.isnan(raw_values[i]) & active).any() or ((~known[i]) & active).any():
            raise ValueError(f"Missing or unavailable held return at {day}")
        target = targets[i]
        if ((np.isnan(price_values[i]) | (~known[i])) & (target != 0)).any():
            raise ValueError(f"Missing or unavailable entry price at {day}")
        values = np.nan_to_num(raw_values[i], nan=0.0)
        long_return = float(np.sum(np.clip(holdings, 0, None) * values))
        short_return = float(np.sum(np.clip(holdings, None, 0) * values))
        gross = long_return + short_return
        if gross <= -1:
            raise ValueError("Portfolio insolvent")
        drifted = holdings * (1 + values) / (1 + gross)
        turnover = float(np.sum(np.abs(target - drifted)))
        transaction_cost = turnover * settings.transaction_cost_bps / 10000
        elapsed = (day - dates[i - 1]).days if i else 1
        borrow = (
            float(-np.clip(holdings, None, 0).sum())
            * settings.borrow_cost_bps_annual
            / 10000
            * elapsed
            / 365
        )
        net = gross - transaction_cost - borrow
        equity *= 1 + net
        spy = float(values[benchmark_index])
        benchmark_equity *= 1 + spy
        output.append(
            {
                "date": day,
                "timestamp": closes[i],
                "return": net,
                "gross_return": gross,
                "long_return": long_return,
                "short_return": short_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "borrow_cost": borrow,
                "gross_exposure": float(np.abs(target).sum()),
                "equity": equity,
                "benchmark_return": spy,
                "benchmark_equity": benchmark_equity,
            }
        )
        holdings = target
    return pd.DataFrame(output)


def placebo(
    events: pd.DataFrame, prices: pd.DataFrame, calendar: TradingCalendar, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle only within contemporaneous day0 cross-sections, never across future dates."""
    paths, summaries = [], []
    for seed in range(settings.placebo_seeds):
        shuffled = events.copy()
        rng = np.random.default_rng(settings.seed + seed)
        for day, indices in shuffled.groupby("day0").groups.items():
            indices = [i for i in indices if shuffled.loc[i, "available_at"] <= calendar.close(day)]
            shuffled.loc[indices, "score"] = rng.permutation(
                shuffled.loc[indices, "score"].to_numpy()
            )
        signals, _ = build_signals(shuffled, prices, calendar, settings)
        daily = backtest(signals, prices, settings)
        daily["seed"] = seed
        paths.append(daily[["date", "seed", "equity"]])
        summaries.append(
            {
                "seed": seed,
                "total_return": float(daily.equity.iloc[-1] - 1),
                "gross_mean_return": float(daily.gross_return.mean()),
            }
        )
    frame = pd.concat(paths, ignore_index=True)
    aggregate = (
        frame.groupby("date")
        .equity.agg(
            placebo_equity="mean",
            placebo_lower=lambda s: s.quantile(0.025),
            placebo_upper=lambda s: s.quantile(0.975),
        )
        .reset_index()
    )
    return aggregate, pd.DataFrame(summaries)
