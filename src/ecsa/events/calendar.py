"""Benchmark-derived trading sessions, with NYSE early-close times."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ecsa.schemas import SessionTiming
from ecsa.storage import utc


class TradingCalendar:
    """Observed benchmark session dates; no fabricated future return observations."""

    def __init__(self, dates: list[date]) -> None:
        self.dates = sorted(set(dates))
        if not self.dates:
            raise ValueError("Benchmark calendar is empty")
        self.index = {day: i for i, day in enumerate(self.dates)}
        self.exchange = xcals.get_calendar(
            "XNYS",
            start=str(self.dates[0] - timedelta(days=10)),
            end=str(self.dates[-1] + timedelta(days=730)),
        )
        self.planned_dates: list[date] = [session.date() for session in self.exchange.sessions]
        self.planned_index = {day: i for i, day in enumerate(self.planned_dates)}

    def day0(self, event: datetime, timing: SessionTiming) -> date | None:
        """BMO/DURING roll on-or-after; AMC/UNKNOWN strictly after the local event date."""
        local = utc(event).astimezone(ZoneInfo("America/New_York")).date()
        for day in self.dates:
            if day > local or (day == local and timing in {"BMO", "DURING"}):
                return day
        return None

    def offset(self, day: date, days: int) -> date | None:
        """Offset a session by an integer number of observed trading days."""
        position = self.index[day] + days
        return self.dates[position] if 0 <= position < len(self.dates) else None

    def close(self, day: date) -> datetime:
        """UTC exchange close, including NYSE half-days; synthetic weekdays default 16 ET."""
        if self.exchange.is_session(pd.Timestamp(day)):
            return utc(self.exchange.session_close(pd.Timestamp(day)))
        return datetime.combine(day, time(16), ZoneInfo("America/New_York")).astimezone(UTC)

    def planned_offset(self, day: date, days: int) -> date:
        """Plan an exit from the exchange calendar, independent of future price coverage."""
        return self.planned_dates[self.planned_index[day] + days]

    def planned_offset_or_none(self, day: date, days: int) -> date | None:
        """Offset by scheduled sessions; None when the target falls outside the plan.

        Unlike :meth:`offset`, this walks the fixed exchange schedule, so a missing
        observed price bar never compresses the event horizon onto a later session.
        """
        position = self.planned_index.get(day)
        if position is None:
            return None
        target = position + days
        return self.planned_dates[target] if 0 <= target < len(self.planned_dates) else None

    def scheduled_close(self, day: date) -> datetime:
        """UTC scheduled NYSE close, including exits beyond the observed price sample."""
        return utc(self.exchange.session_close(pd.Timestamp(day)))
