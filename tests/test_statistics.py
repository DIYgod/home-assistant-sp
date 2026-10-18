"""The imported electricity statistic keeps one running sum across polls.

Each poll recomputes the cumulative series from the window it read and writes
part of it. Two things the AMI feeds do broke that: the half-hour feed lags the
meter by hours, so a slot can land behind the newest start already written,
and the 13-month daily window drops a month on the 1st, which lowers every sum
after it. These replay polls the way the coordinator writes them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from custom_components.sp_group.const import (
    AMI_DAILY_MONTHS,
    AMI_HALF_HOUR_DAYS,
    STATISTICS_REOPEN,
)
from custom_components.sp_group.history import (
    CumulativePoint,
    continued,
    cumulative_points,
)
from custom_components.sp_group.mapper import electricity_view
from custom_components.sp_group.models import (
    SG_TZ,
    PeriodReading,
    PremiseInfo,
    UsageReadings,
    UtilitySeries,
)

SLOT_KWH = 0.25
HOUR_KWH = 2 * SLOT_KWH
DAY_KWH = 48 * SLOT_KWH
# How far the half-hour feed trails the clock: the poll after midnight sees
# slots up to the afternoon before, as Jarvis serves them.
FEED_LAG = timedelta(hours=8, minutes=27)

PREMISE = PremiseInfo(
    id="2001590888",
    address=None,
    account_number=None,
    account_status=None,
    account_type=None,
    premise_type=None,
    utilities=(),
    ami_elec=True,
    retailer_name=None,
    ppms_exists=False,
)
BILLED = UtilitySeries(
    total=DAY_KWH,
    unit="kWh",
    periods=(PeriodReading(start=datetime(2025, 1, 1, tzinfo=SG_TZ), amount=DAY_KWH),),
    average=None,
    comparison=None,
)


def _usage_at(now: datetime) -> UsageReadings:
    """What a poll at ``now`` reads, in the windows the client asks Jarvis for."""
    month = now.month - (AMI_DAILY_MONTHS - 1)
    year = now.year
    while month <= 0:
        month += 12
        year -= 1
    month_start = now.replace(year=year, month=month, day=1, hour=0, minute=0)
    day_start = (now - timedelta(days=AMI_HALF_HOUR_DAYS - 1)).replace(hour=0, minute=0)
    published = now - FEED_LAG
    hourly: list[PeriodReading] = []
    slot = day_start
    while slot < now:
        # The feed pads the slots it has not published with zeros.
        amount = SLOT_KWH if slot + timedelta(minutes=30) <= published else 0.0
        hourly.append(PeriodReading(start=slot, amount=amount))
        slot += timedelta(minutes=30)
    daily: list[PeriodReading] = []
    day = month_start
    while day < now:
        if day + timedelta(days=1) <= published:
            amount = DAY_KWH
        else:
            amount = sum(
                item.amount
                for item in hourly
                if item.start.date() == day.date() and item.amount
            )
        daily.append(PeriodReading(start=day, amount=amount))
        day += timedelta(days=1)
    return UsageReadings(
        premise=PREMISE,
        electricity=BILLED,
        water=None,
        gas=None,
        ami_hourly=tuple(hourly),
        ami_daily=tuple(daily),
    )


@dataclass
class Recorder:
    """One statistic's rows, written the way the coordinator writes them."""

    rows: dict[datetime, float] = field(default_factory=dict)
    written: dict[datetime, float] = field(default_factory=dict)

    def poll(self, now: datetime) -> list[CumulativePoint]:
        usage = _usage_at(now)
        points = cumulative_points(electricity_view(usage, now).graph)
        oldest = min(point.start for point in points)
        for start in [start for start in self.written if start < oldest]:
            del self.written[start]
        fresh = continued(points, self.written, STATISTICS_REOPEN)
        for point in fresh:
            self.rows[point.start] = point.cumulative
            self.written[point.start] = point.cumulative
        return fresh

    def restart(self) -> None:
        """Home Assistant restarted: the rows stay, the in-memory record goes."""
        self.written = {}

    def change(self, local_hour: datetime) -> float | None:
        """The usage the recorder holds for the hour starting at ``local_hour``."""
        start = local_hour.astimezone(UTC)
        if start not in self.rows:
            return None
        before = [moment for moment in self.rows if moment < start]
        if not before:
            return None
        return self.rows[start] - self.rows[max(before)]


def _polls(start: datetime, end: datetime) -> list[datetime]:
    """Every 30-minute poll from ``start`` up to ``end``, at :27 and :57."""
    moment = start.replace(minute=27)
    out: list[datetime] = []
    while moment <= end:
        out.append(moment)
        moment += timedelta(minutes=30)
    return out


def test_the_evening_slots_reach_the_statistics_after_midnight() -> None:
    """The slots published after midnight land behind today's daily point.

    Before, the poll after midnight wrote today's daily point at 00:00, the
    evening's slots arrived behind it, and none of them were written until the
    next restart: 16:00 to 23:00 stayed empty and 01:00 held their whole sum.
    """
    midnight = datetime(2026, 10, 18, tzinfo=SG_TZ)
    recorder = Recorder()
    for now in _polls(midnight - timedelta(hours=2), midnight + timedelta(hours=12)):
        recorder.poll(now)

    evening = [midnight - timedelta(hours=hours) for hours in range(8, 0, -1)]
    morning = [midnight + timedelta(hours=hours) for hours in range(3)]
    for hour in (*evening, *morning):
        assert recorder.change(hour) == pytest.approx(HOUR_KWH), hour


def test_the_monthly_window_move_keeps_the_running_sum() -> None:
    """On the 1st the daily window drops a month and every sum after it falls.

    Before, the first hour written after that read as the dropped month's usage
    in reverse, about a month of negative consumption on the Energy dashboard.
    """
    first = datetime(2026, 11, 1, tzinfo=SG_TZ)
    recorder = Recorder()
    for now in _polls(first - timedelta(days=1), first + timedelta(hours=12)):
        recorder.poll(now)

    # From well before the rows the rollover poll rewrote to the newest full hour.
    hours = [first - timedelta(hours=hours) for hours in range(96, -3, -1)]
    for hour in hours:
        assert recorder.change(hour) == pytest.approx(HOUR_KWH), hour


def test_a_restart_sends_the_whole_series_again() -> None:
    """With nothing in memory the import is the series as computed, as before."""
    now = datetime(2026, 10, 18, 9, 27, tzinfo=SG_TZ)
    recorder = Recorder()
    first = recorder.poll(now)
    recorder.restart()
    again = recorder.poll(now)

    assert again == first
    assert len(again) == len(
        cumulative_points(electricity_view(_usage_at(now), now).graph)
    )


def test_continued_sends_the_whole_series_first() -> None:
    points = [
        CumulativePoint(start=datetime(2026, 9, 21, hour, tzinfo=UTC), cumulative=hour)
        for hour in range(3)
    ]
    assert continued(points, {}, STATISTICS_REOPEN) == points


def test_continued_rewrites_the_reopened_rows_and_appends_the_new() -> None:
    newest = datetime(2026, 9, 21, 12, tzinfo=UTC)
    points = [
        CumulativePoint(start=newest + timedelta(hours=hour), cumulative=100.0 + hour)
        for hour in range(-60, 3)
    ]
    written = {
        point.start: point.cumulative for point in points if point.start <= newest
    }

    fresh = continued(points, written, timedelta(days=2))

    assert [point.start for point in fresh] == [
        point.start for point in points if point.start > newest - timedelta(days=2)
    ]
    # Nothing moved, so nothing is shifted.
    assert [point.cumulative for point in fresh] == [
        point.cumulative for point in points if point.start > newest - timedelta(days=2)
    ]


def test_continued_shifts_a_moved_series_back_onto_the_written_sums() -> None:
    newest = datetime(2026, 9, 21, 12, tzinfo=UTC)
    reopen = timedelta(hours=2)
    written = {newest - timedelta(hours=hour): 500.0 - hour for hour in range(6)}
    # The same hours a month of usage lower, plus one new hour.
    moved = [
        CumulativePoint(start=start, cumulative=value - 400.0)
        for start, value in sorted(written.items())
    ] + [CumulativePoint(start=newest + timedelta(hours=1), cumulative=101.0)]

    fresh = continued(moved, written, reopen)

    assert [(point.start, point.cumulative) for point in fresh] == [
        (newest - timedelta(hours=1), 499.0),
        (newest, 500.0),
        (newest + timedelta(hours=1), 501.0),
    ]


def test_continued_without_an_anchor_does_not_shift() -> None:
    """No written row at or before the cut is in the series: nothing to align to."""
    newest = datetime(2026, 9, 21, 12, tzinfo=UTC)
    written = {newest: 50.0}
    points = [CumulativePoint(start=newest + timedelta(hours=1), cumulative=7.0)]

    assert continued(points, written, timedelta(hours=2)) == points


def test_continued_leaves_a_fallback_series_older_than_the_cut_alone() -> None:
    """Both AMI feeds down used to fall back to the billed months.

    Those points all start before the cut, so none of them is written over the
    AMI hours the recorder already holds.
    """
    newest = datetime(2026, 10, 18, 3, tzinfo=UTC)
    written = {newest - timedelta(hours=hour): 3000.0 - hour for hour in range(72)}
    billed = [
        CumulativePoint(start=datetime(2026, month, 1, tzinfo=UTC), cumulative=month)
        for month in range(1, 10)
    ]

    assert continued(billed, written, STATISTICS_REOPEN) == []


def test_usage_fixture_matches_the_shape_jarvis_serves() -> None:
    """The replay is only worth something if its series looks like the real one."""
    now = datetime(2026, 10, 18, 0, 27, tzinfo=SG_TZ)
    usage = _usage_at(now)
    view = electricity_view(usage, now)

    assert view.last_slot is not None
    assert view.last_slot.start == datetime(2026, 10, 17, 15, 30, tzinfo=SG_TZ)
    # Today has no published slot yet, so the graph ends on today's daily point.
    assert view.graph[-1].start == datetime(2026, 10, 18, tzinfo=SG_TZ)
    assert dataclasses.replace(usage, ami_hourly=()).ami_daily == usage.ami_daily
