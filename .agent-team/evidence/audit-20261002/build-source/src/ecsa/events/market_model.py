"""Past-window OLS expected returns, with explicit market-adjusted fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class MarketModel:
    """Daily decimal alpha and residual variance; beta is dimensionless."""

    alpha: float
    beta: float
    method: str
    n: int
    residual_variance: float
    market_mean: float
    market_sxx: float

    def car_variance(self, market_returns: NDArray[np.float64]) -> float:
        """Forecast-error-adjusted CAR variance, including shared estimation covariance."""
        length = len(market_returns)
        if self.method != "market_model" or self.n < 3 or self.market_sxx <= 0:
            return float("nan")
        inflation = (
            length
            + length * length / self.n
            + (float(np.sum(market_returns)) - length * self.market_mean) ** 2 / self.market_sxx
        )
        return self.residual_variance * inflation


def fit_market_model(
    asset: pd.Series, market: pd.Series, *, cutoff: date, min_days: int = 120
) -> MarketModel:
    """OLS on supplied estimation-window rows at/before cutoff; returns are decimals."""
    frame = pd.concat([asset.rename("asset"), market.rename("market")], axis=1).dropna()
    frame = frame[[pd.Timestamp(day).date() <= cutoff for day in frame.index]]
    n = len(frame)
    x, y = frame.market.to_numpy(dtype=float), frame.asset.to_numpy(dtype=float)
    sxx = float(np.sum((x - x.mean()) ** 2)) if n else 0.0
    if n < min_days or sxx < 1e-14:
        return MarketModel(
            0.0,
            1.0,
            "market_adjusted",
            n,
            float(np.var(y - x, ddof=1)) if n > 1 else float("nan"),
            float(x.mean()) if n else float("nan"),
            sxx,
        )
    design = np.column_stack([np.ones(n), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([alpha, beta])
    return MarketModel(
        float(alpha),
        float(beta),
        "market_model",
        n,
        float(residual @ residual / (n - 2)),
        float(x.mean()),
        sxx,
    )
