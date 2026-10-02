"""Optional expanding walk-forward learning with matured labels and trading-day embargo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecsa.events.calendar import TradingCalendar
from ecsa.settings import Settings


def walk_forward(
    events: pd.DataFrame, calendar: TradingCalendar, settings: Settings
) -> pd.DataFrame:
    """Fit before each test close; label end must precede a ten-session embargo.

    All predictions are out of sample. Normalization and imputation fit only on
    training rows. No in-sample IC is returned or used for selection.
    """
    output = events.copy()
    output["model_score"] = np.nan
    output["training_n"] = 0
    output["training_latest_label"] = pd.Series(
        pd.NaT, index=output.index, dtype="datetime64[ns, UTC]"
    )
    columns = list(settings.score_weights)
    for day, test in output.groupby("day0", sort=True):
        embargo_day = calendar.offset(day, -settings.embargo_days)
        if embargo_day is None:
            continue
        deadline = calendar.close(embargo_day)
        train = output[
            (output.label_available_at < deadline)
            & (output.available_at < deadline)
            & (output.day0 < day)
        ].dropna(subset=["CAR_drift"])
        test = test[test.available_at <= calendar.close(day)]
        if len(train) < settings.minimum_training_events or test.empty:
            continue
        usable = [col for col in columns if train[col].notna().any()]
        target = (
            train.CAR_drift
            if settings.learned_model == "ridge"
            else (train.CAR_drift > 0).astype(int)
        )
        if settings.learned_model == "logistic" and target.nunique() < 2:
            continue
        estimator = (
            Ridge(alpha=1)
            if settings.learned_model == "ridge"
            else LogisticRegression(random_state=settings.seed, max_iter=1000)
        )
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
        model.fit(train[usable], target)
        predictions = (
            model.predict(test[usable])
            if settings.learned_model == "ridge"
            else model.decision_function(test[usable])
        )
        output.loc[test.index, "model_score"] = predictions
        output.loc[test.index, "training_n"] = len(train)
        output.loc[test.index, "training_latest_label"] = train.label_available_at.max()
    return output
