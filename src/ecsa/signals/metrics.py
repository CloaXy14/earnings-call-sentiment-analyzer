"""Strategy diagnostics in decimal return units, with explicit trial-count disclosure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ecsa.events.calendar import TradingCalendar
from ecsa.events.stats import decision_available, information_coefficient
from ecsa.settings import Settings


def _trade_returns(
    signals: pd.DataFrame, prices: pd.DataFrame, settings: Settings
) -> NDArray[np.float64]:
    """Per-trade net return from the same daily-rebalanced ledger as the backtest (F7).

    Hit rate and average win/loss previously used endpoint buy-and-hold returns, which
    disagree with the strategy's daily target-weight rebalancing and all-side costs.
    Each trade is replayed as a single isolated sleeve through the production
    ``backtest`` accounting, so trade statistics reconcile with the equity curve.
    """
    from ecsa.signals.backtest import backtest

    returns = []
    traded = signals[signals.side != "NONE"]
    for _, signal in traded.iterrows():
        sleeve = pd.DataFrame([signal])
        ledger = backtest(sleeve, prices, settings)
        returns.append(float(ledger.equity.iloc[-1] - 1))
    return np.array(returns)


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
    trades = _trade_returns(signals, prices, settings) if not signals.empty else np.array([])
    # Predictive IC uses only events tradeable at the day-0 close (F6); the
    # retrospective IC over every event is reported separately and labeled.
    predictive_events = (
        decision_available(
            events, TradingCalendar(list(prices[prices.ticker == settings.benchmark].date))
        )
        if not events.empty
        else events
    )
    ic = (
        information_coefficient(predictive_events, "score")
        if not predictive_events.empty
        else pd.DataFrame()
    )
    retrospective = (
        information_coefficient(events, "score") if not events.empty else pd.DataFrame()
    )
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
        "ic_retrospective": float(retrospective.iloc[0].ic)
        if not retrospective.empty
        else float("nan"),
        "quarterly_ic_t_stat": ic_t,
        "long_leg_arithmetic_contribution": float(daily.long_return.sum()),
        "short_leg_arithmetic_contribution": float(daily.short_return.sum()),
        "variants_tried": float(settings.variants_tried),
        "completed_trades": float(len(trades)),
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])
