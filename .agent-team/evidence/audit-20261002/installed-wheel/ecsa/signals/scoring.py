"""Composite sentiment from expanding standardizations using strictly earlier events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecsa.settings import Settings


def composite_scores(features: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Weighted mean of past-only z-scores; unavailable/zero-variance components omitted."""
    result = features.copy()
    result["score"] = np.nan
    for feature in settings.score_weights:
        if feature not in result:
            raise ValueError(f"Unknown scoring feature {feature}")
        result["z_" + feature] = np.nan
    for index, row in result.iterrows():
        history = result[result.available_at < row.available_at]
        numerator, denominator = 0.0, 0.0
        for feature, weight in settings.score_weights.items():
            values = history[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if (
                weight == 0
                or len(values) < 2
                or values.std(ddof=1) <= 1e-12
                or not np.isfinite(row[feature])
            ):
                continue
            z = (row[feature] - values.mean()) / values.std(ddof=1)
            result.loc[index, "z_" + feature] = z
            numerator += weight * z
            denominator += abs(weight)
        if denominator:
            result.loc[index, "score"] = numerator / denominator
    return result
