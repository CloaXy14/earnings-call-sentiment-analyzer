"""Point-in-time event ranking and capped overlapping positions."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ecsa.events.calendar import TradingCalendar
from ecsa.schemas import Signal
from ecsa.settings import Settings

SIGNAL_COLUMNS = [*Signal.model_fields, "reason", "sector"]


def build_signals(
    events: pd.DataFrame, prices: pd.DataFrame, calendar: TradingCalendar, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank events against a trailing calendar-day window; use no future prices/outcomes.

    Target gross weights are capped across overlapping events for a name. Sector
    neutrality, when requested, requires both sides in each sector in the new batch.
    Missing sector metadata is an error. Late features are audited, never backdated.
    """
    rows: list[dict[str, Any]] = []
    rejected = []
    for day, batch in events.groupby("day0", sort=True):
        now = calendar.close(day)
        history = events[
            (events.available_at <= now)
            & (events.day0 <= day)
            & (events.day0 >= day - timedelta(days=settings.ranking_days))
            & events.score.notna()
        ]
        lower, upper = (
            (
                history.score.quantile(settings.short_quantile),
                history.score.quantile(settings.long_quantile),
            )
            if len(history) >= settings.minimum_ranking_events
            else (np.nan, np.nan)
        )
        candidates = []
        for _, event in batch.sort_values(["ticker", "event_id"]).iterrows():
            if event.available_at > now:
                rejected.append(
                    {
                        "event_id": event.event_id,
                        "ticker": event.ticker,
                        "day0": day,
                        "reason": "features_unavailable_at_day0_close",
                    }
                )
                continue
            if not np.isfinite(event.score):
                rejected.append(
                    {
                        "event_id": event.event_id,
                        "ticker": event.ticker,
                        "day0": day,
                        "reason": "insufficient_score_history",
                    }
                )
                continue
            side, reason = "NONE", "middle_quantile"
            if len(history) < settings.minimum_ranking_events:
                reason = "insufficient_ranking_events"
            elif upper > lower:
                side = (
                    "LONG" if event.score >= upper else "SHORT" if event.score <= lower else "NONE"
                )
            sector = settings.sectors.get(event.ticker, "UNKNOWN")
            if settings.sector_neutral and sector == "UNKNOWN":
                raise ValueError(f"Missing sector for {event.ticker}")
            inverse_vol = 1.0
            if side != "NONE" and settings.weighting == "inverse_vol":
                known = (
                    prices[
                        (prices.ticker == event.ticker)
                        & (prices.date <= day)
                        & (prices.available_at <= now)
                    ]
                    .sort_values("date")
                    .tail(settings.volatility_days + 1)
                )
                values = known.adj_close.pct_change(fill_method=None).dropna()
                if len(values) < settings.volatility_days or values.std(ddof=1) <= 1e-12:
                    side, reason = "NONE", "insufficient_volatility_history"
                else:
                    inverse_vol = 1 / float(values.std(ddof=1))
            exit_day = calendar.planned_offset(day, settings.holding_days)
            candidates.append(
                {
                    "event_id": event.event_id,
                    "signal_date": now,
                    "feature_available_at": event.available_at,
                    "ticker": event.ticker,
                    "score": float(event.score),
                    "side": side,
                    "weight": 0.0,
                    "entry_date": now,
                    "exit_date": calendar.scheduled_close(exit_day),
                    "reason": reason,
                    "sector": sector,
                    "inverse_vol": inverse_vol,
                }
            )
        active = [row for row in rows if row["side"] != "NONE" and row["exit_date"] > now]
        used = sum(row["weight"] for row in active)
        eligible = [row for row in candidates if row["side"] != "NONE"]
        denominator = sum(row["inverse_vol"] for row in eligible)
        for row in eligible:
            name_used = sum(r["weight"] for r in active if r["ticker"] == row["ticker"])
            weight = min(
                settings.max_name_weight - name_used,
                max(0, settings.gross_target - used) * row["inverse_vol"] / max(denominator, 1e-12),
            )
            row["weight"] = max(0.0, weight)
        # Simultaneous events for the same name share its remaining capacity.
        for ticker in {r["ticker"] for r in eligible}:
            name_rows = [r for r in eligible if r["ticker"] == ticker]
            total = sum(r["weight"] for r in name_rows)
            capacity = settings.max_name_weight - sum(
                r["weight"] for r in active if r["ticker"] == ticker
            )
            for row in name_rows:
                row["weight"] *= min(1.0, max(0.0, capacity) / max(total, 1e-12))
        if settings.sector_neutral:
            for sector in {r["sector"] for r in eligible}:
                sector_rows = [r for r in eligible if r["sector"] == sector]
                longs = sum(r["weight"] for r in sector_rows if r["side"] == "LONG")
                shorts = sum(r["weight"] for r in sector_rows if r["side"] == "SHORT")
                for row in sector_rows:
                    row["weight"] *= min(longs, shorts) / max(
                        longs if row["side"] == "LONG" else shorts, 1e-12
                    )
        for row in candidates:
            row.pop("inverse_vol")
            if row["weight"] <= 1e-15:
                row["side"], row["weight"] = "NONE", 0.0
                if row["reason"] == "middle_quantile":
                    row["reason"] = "middle_quantile_or_capacity"
            else:
                row["reason"] = "ranked"
            Signal.model_validate(
                {key: value for key, value in row.items() if key in Signal.model_fields}
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=SIGNAL_COLUMNS), pd.DataFrame(
        rejected, columns=["event_id", "ticker", "day0", "reason"]
    )
