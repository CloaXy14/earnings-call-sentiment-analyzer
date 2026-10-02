"""Retrospective event inference; these outcome tables must never feed live signals."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ecsa.events.study import WINDOWS

SCORES = [
    "finbert_net_mean",
    "finbert_net_weighted",
    "sentiment_surprise",
    "lexicon_net",
    "fls_net",
]


def quarter_buckets(events: pd.DataFrame, score: str) -> pd.DataFrame:
    """Retrospective quarter quintiles; ties share a bucket; tiny quarters are neutral."""
    result = events.copy()
    result["quarter"] = pd.to_datetime(result.day0).dt.to_period("Q").astype(str)
    result["bucket"] = 2
    for _, group in result.groupby("quarter"):
        values = group[score].dropna()
        if len(values) >= 5 and values.nunique() >= 5:
            edges = values.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            result.loc[values.index, "bucket"] = np.searchsorted(
                edges, values.to_numpy(), side="left"
            )
    return result


def bucket_membership(
    events: pd.DataFrame,
    score: str,
    *,
    method: str = "quarter_retrospective",
    ranking_days: int = 90,
    minimum_events: int = 20,
) -> pd.DataFrame:
    """Persist peer membership separately from causal trading decisions."""
    valid = events.dropna(subset=[score]).copy()
    if method == "quarter_retrospective":
        labels = quarter_buckets(valid, score)[["event_id", "bucket"]]
    elif method == "trailing_history":
        valid["bucket"] = 2
        for index, event_row in valid.iterrows():
            past = valid[
                (valid.available_at < event_row.available_at)
                & (valid.day0 < event_row.day0)
                & (valid.day0 >= event_row.day0 - timedelta(days=ranking_days))
            ]
            if len(past) >= minimum_events:
                valid.loc[index, "bucket"] = min(
                    4, int(float((past[score] <= event_row[score]).mean()) * 5)
                )
        labels = valid[["event_id", "bucket"]]
    else:
        raise ValueError("Unknown bucket method")
    labels = labels.copy()
    labels["score_definition"] = score
    labels["bucket_method"] = method
    return labels


def bucket_paths(
    events: pd.DataFrame,
    paths: pd.DataFrame,
    score: str,
    *,
    method: str = "quarter_retrospective",
    ranking_days: int = 90,
    minimum_events: int = 20,
) -> pd.DataFrame:
    """Average AR/CAR paths with date-cluster robust 95% intervals for AR and CAR."""
    labels = bucket_membership(
        events, score, method=method, ranking_days=ranking_days, minimum_events=minimum_events
    )
    merged = paths.merge(labels, on="event_id", validate="many_to_one")
    rows = []
    for (bucket, day), group in merged.groupby(["bucket", "relative_day"]):
        row: dict[str, Any] = {
            "score_definition": score,
            "bucket_method": method,
            "bucket": bucket,
            "relative_day": day,
            "n": len(group),
        }
        for field in ("ar", "car"):
            clean = group.dropna(subset=[field])
            mean = float(clean[field].mean())
            groups = clean.day0.astype(str)
            clusters = groups.nunique()
            se = float("nan")
            if len(clean) > 1 and clusters > 1:
                fit = sm.OLS(clean[field].to_numpy(), np.ones((len(clean), 1))).fit(
                    cov_type="cluster", cov_kwds={"groups": groups.to_numpy()}
                )
                se = float(fit.bse[0])
            margin = float(stats.t.ppf(0.975, clusters - 1)) * se if clusters > 1 else float("nan")
            row.update(
                {
                    "mean_" + field: mean,
                    field + "_lower": mean - margin,
                    field + "_upper": mean + margin,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def window_tests(events: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional t and BMP standardized t per window; undefined tests remain NaN.

    BMP uses SCAR with market-model forecast-error variance and excludes fallback
    events. It allows event-induced variance but does not fix cross-firm dependence.
    """
    rows = []
    for window in WINDOWS:
        for name, column in (("cross_sectional_t", "CAR_" + window), ("BMP", "SCAR_" + window)):
            values = events[column].replace([np.inf, -np.inf], np.nan).dropna()
            statistic, pvalue = (float("nan"), float("nan"))
            if len(values) > 1 and values.std(ddof=1) > 1e-15:
                test = stats.ttest_1samp(values, 0)
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            rows.append(
                {
                    "window": window,
                    "test": name,
                    "n": len(values),
                    "statistic": statistic,
                    "p_value": pvalue,
                    "mean": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def information_coefficient(events: pd.DataFrame, score: str) -> pd.DataFrame:
    """Spearman score versus CAR[+1,+10], overall and by calendar quarter."""
    frame = events.copy()
    frame["quarter"] = pd.to_datetime(frame.day0).dt.to_period("Q").astype(str)
    rows = []
    for quarter, group in [("overall", frame), *list(frame.groupby("quarter"))]:
        clean = group[[score, "CAR_drift"]].dropna()
        ic, pvalue = float("nan"), float("nan")
        if len(clean) > 2 and clean[score].nunique() > 1 and clean.CAR_drift.nunique() > 1:
            result = stats.spearmanr(clean[score], clean.CAR_drift)
            ic, pvalue = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "score_definition": score,
                "quarter": str(quarter),
                "n": len(clean),
                "ic": ic,
                "p_value": pvalue,
            }
        )
    return pd.DataFrame(rows)


def drift_regression(events: pd.DataFrame, score: str) -> pd.DataFrame:
    """Retrospective OLS drift ~ score + anticipation + reaction; SE clustered by day0."""
    predictors = [score, "CAR_pre", "CAR_reaction_01"]
    if "earnings_surprise" in events:
        predictors.append("earnings_surprise")
    frame = events.dropna(subset=predictors + ["CAR_drift", "day0"])
    columns = ["term", "coefficient", "std_error", "p_value", "n", "clusters", "status"]
    if len(frame) <= len(predictors) + 1 or frame.day0.nunique() < 2:
        return pd.DataFrame(
            [
                [
                    "all",
                    np.nan,
                    np.nan,
                    np.nan,
                    len(frame),
                    frame.day0.nunique(),
                    "insufficient_observations",
                ]
            ],
            columns=columns,
        )
    design = sm.add_constant(frame[predictors], has_constant="add")
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        return pd.DataFrame(
            [["all", np.nan, np.nan, np.nan, len(frame), frame.day0.nunique(), "rank_deficient"]],
            columns=columns,
        )
    fit = sm.OLS(frame.CAR_drift, design).fit(
        cov_type="cluster", cov_kwds={"groups": frame.day0.astype(str).to_numpy()}, use_t=True
    )
    return pd.DataFrame(
        [
            [
                term,
                float(fit.params[term]),
                float(fit.bse[term]),
                float(fit.pvalues[term]),
                len(frame),
                frame.day0.nunique(),
                "ok",
            ]
            for term in fit.params.index
        ],
        columns=columns,
    )
