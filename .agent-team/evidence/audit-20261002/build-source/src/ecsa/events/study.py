"""Event-aligned abnormal returns and forecast-error-standardized CAR windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ecsa.events.calendar import TradingCalendar
from ecsa.events.market_model import fit_market_model
from ecsa.settings import Settings
from ecsa.storage import utc

WINDOWS = {
    "pre": (-5, -1),
    "reaction": (0, 0),
    "reaction_01": (0, 1),
    "drift": (1, 10),
    "full": (-5, 10),
}
INTERPRETATION = (
    "[-5,-1] measures anticipation; [0,0] and [0,+1] measure reaction. "
    "Only [+1,+10] tests a tradeable prediction for a signal known by day-0 close. "
    "Quarter buckets and outcome regressions are retrospective, never signal inputs."
)


def run_study(
    features: pd.DataFrame, prices: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return event-wide and long path tables; AR/CAR are decimal returns.

    Missing daily bars remain missing. Cumulative paths/windows are NaN if any
    constituent observation is absent. Labels carry their actual availability.
    """
    if features.empty:
        return pd.DataFrame(), pd.DataFrame()
    benchmark = prices[prices.ticker == settings.benchmark].sort_values("date")
    calendar = TradingCalendar(list(benchmark.date))
    close = prices.pivot(index="date", columns="ticker", values="adj_close").reindex(calendar.dates)
    returns = close.pct_change(fill_method=None)
    availability = prices.pivot(index="date", columns="ticker", values="available_at").reindex(
        calendar.dates
    )
    rows, paths = [], []
    for _, event in features.iterrows():
        day0 = calendar.day0(utc(event.event_datetime), event.session_timing)
        if day0 is None or event.ticker not in returns:
            continue
        position = calendar.index[day0]
        entry = calendar.close(day0)
        indices = list(
            range(
                max(0, position + settings.estimation_start),
                max(0, position + settings.estimation_end + 1),
            )
        )
        paired = returns.iloc[indices][[event.ticker, settings.benchmark]].copy()
        known = availability.iloc[indices][[event.ticker, settings.benchmark]]
        paired = paired.where(known.le(pd.Timestamp(entry)))
        cutoff = calendar.offset(day0, settings.estimation_end)
        if cutoff is None:
            paired = paired.iloc[:0]
            cutoff = day0
        model = fit_market_model(
            paired[event.ticker],
            paired[settings.benchmark],
            cutoff=cutoff,
            min_days=settings.estimation_min_days,
        )
        row: dict[str, Any] = {
            **event.to_dict(),
            "day0": day0,
            "entry_close": entry,
            "alpha": model.alpha,
            "beta": model.beta,
            "method": model.method,
            "estimation_n": model.n,
            "residual_variance": model.residual_variance,
            "interpretation": INTERPRETATION,
        }
        ar, rm, raw, label_times = {}, {}, {}, []
        for relative in range(settings.event_start, settings.event_end + 1):
            day = calendar.offset(day0, relative)
            asset_return = float(returns.loc[day, event.ticker]) if day else float("nan")
            market_return = float(returns.loc[day, settings.benchmark]) if day else float("nan")
            abnormal = asset_return - model.alpha - model.beta * market_return
            raw[relative], ar[relative], rm[relative] = asset_return, abnormal, market_return
            row[f"AR_{relative:+d}"] = abnormal
            if 1 <= relative <= 10 and day:
                for ticker in (event.ticker, settings.benchmark):
                    when = availability.loc[day, ticker]
                    if pd.notna(when):
                        label_times.append(utc(when))
        for name, (left, right) in WINDOWS.items():
            values = np.array([ar[t] for t in range(left, right + 1)])
            market_values = np.array([rm[t] for t in range(left, right + 1)])
            car = float(values.sum())
            variance = model.car_variance(market_values)
            row["CAR_" + name] = car
            row["SCAR_" + name] = car / np.sqrt(variance) if variance > 1e-20 else float("nan")
        row["label_available_at"] = (
            max(label_times) if len(label_times) == 20 and np.isfinite(row["CAR_drift"]) else pd.NaT
        )
        rows.append(row)
        cumulative, gross = 0.0, 1.0
        for relative in range(settings.event_start, settings.event_end + 1):
            cumulative += ar[relative]
            gross *= 1 + raw[relative]
            paths.append(
                {
                    "event_id": event.event_id,
                    "ticker": event.ticker,
                    "day0": day0,
                    "relative_day": relative,
                    "date": calendar.offset(day0, relative),
                    "ar": ar[relative],
                    "car": cumulative,
                    "cumulative_return": gross - 1,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(paths)
