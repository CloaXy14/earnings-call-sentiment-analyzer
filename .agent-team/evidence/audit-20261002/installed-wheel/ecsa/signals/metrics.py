"""Strategy diagnostics in decimal return units, with explicit trial-count disclosure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecsa.events.stats import information_coefficient
from ecsa.settings import Settings


def performance(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    events: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Annualized returns/volatility use 252 sessions; Sharpe assumes zero cash rate."""
    values = daily["return"].to_numpy()
    years = len(values) / 252
    terminal = float(np.prod(1 + values))
    annual = terminal ** (1 / years) - 1 if years and terminal > 0 else float("nan")
    volatility = float(np.std(values, ddof=1) * np.sqrt(252)) if len(values) > 1 else float("nan")
    sharpe = float(np.mean(values) * 252 / volatility) if volatility > 0 else float("nan")
    equity = np.r_[1.0, np.cumprod(1 + values)]
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    adjusted = prices.pivot(index="date", columns="ticker", values="adj_close")
    wins = []
    for _, signal in signals[signals.side != "NONE"].iterrows():
        entry, exit_ = signal.entry_date.date(), signal.exit_date.date()
        if entry in adjusted.index and exit_ in adjusted.index:
            raw = float(adjusted.loc[exit_, signal.ticker] / adjusted.loc[entry, signal.ticker] - 1)
            net = (
                raw * (1 if signal.side == "LONG" else -1)
                - 2 * settings.transaction_cost_bps / 10000
            )
            if signal.side == "SHORT":
                net -= settings.borrow_cost_bps_annual / 10000 * (exit_ - entry).days / 365
            wins.append(net)
    trades = np.array(wins)
    ic = information_coefficient(events, "score") if not events.empty else pd.DataFrame()
    quarterly = ic[ic.quarter != "overall"].ic.dropna() if not ic.empty else pd.Series(dtype=float)
    ic_t = (
        float(quarterly.mean() / (quarterly.std(ddof=1) / np.sqrt(len(quarterly))))
        if len(quarterly) > 1 and quarterly.std(ddof=1) > 0
        else float("nan")
    )
    metrics = {
        "annualized_return": annual,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "hit_rate": float(np.mean(trades > 0)) if len(trades) else float("nan"),
        "average_win": float(trades[trades > 0].mean()) if (trades > 0).any() else float("nan"),
        "average_loss": float(trades[trades < 0].mean()) if (trades < 0).any() else float("nan"),
        "annualized_turnover": float(daily.turnover.sum() / years) if years else float("nan"),
        "ic": float(ic.iloc[0].ic) if not ic.empty else float("nan"),
        "quarterly_ic_t_stat": ic_t,
        "long_leg_arithmetic_contribution": float(daily.long_return.sum()),
        "short_leg_arithmetic_contribution": float(daily.short_return.sum()),
        "variants_tried": float(settings.variants_tried),
        "completed_trades": float(len(trades)),
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
