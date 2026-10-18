"""Shape usage and bill periods into the series written to HA statistics.

Billed Jarvis periods become cumulative points for the recorder, AMI half-hours
are folded to clock hours, and bills collapse to one point per calendar month.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from .const import DOMAIN
from .models import SG_TZ, BillInfo, PeriodReading


class HasStart(Protocol):
    """Anything a statistic row is timestamped by."""

    @property
    def start(self) -> datetime: ...


def unimported[Started: HasStart](
    points: Sequence[Started], imported_through: datetime | None
) -> list[Started]:
    """The points the recorder does not have yet.

    Every poll recomputes the whole series from the loaded window, so without
    this filter the poll would rewrite all of it on every cycle. The recorder
    keys a row by (statistic id, start), so the rewrite is invisible work, not
    a wrong number, and filtering keeps one series to the readings that are
    actually new.

    ``imported_through`` is the caller's in-memory record of the last start it
    sent, which starts empty on every Home Assistant restart: the first import
    after one is the whole series. Nothing here reads the recorder back to
    discover what is already there.
    """
    if imported_through is None:
        return list(points)
    return [point for point in points if point.start > imported_through]


@dataclass(frozen=True)
class CumulativePoint:
    start: datetime
    cumulative: float


def continued(
    points: Sequence[CumulativePoint],
    written: Mapping[datetime, float],
    reopen: timedelta,
) -> list[CumulativePoint]:
    """The points to write so a cumulative series keeps one running sum.

    ``written`` is the sum the caller last wrote at each start of this series.
    It starts empty on every Home Assistant start, and the first import after
    one is the whole series as computed, which is what ``unimported`` sends.

    After that, two things differ from cutting at the newest start written.

    The points from ``reopen`` before the newest start written are sent again.
    The half-hour feed publishes a slot hours after it ends, and by then the
    next day's daily point has already moved the newest start past it. Cut at
    the newest start, the slots of each evening never reached the recorder
    until a restart re-sent everything, and the first hour written after them
    carried their whole sum. The recorder keys a row by its start, so a row
    sent again is replaced, not doubled.

    And every point sent is shifted so the newest point at or before the cut,
    which this series wrote before, keeps the sum it was written with. A poll
    sums its series from the start of the window it read, and that start moves:
    on the 1st a month of daily points leaves the window, and every sum after
    it falls by that month's usage. Unshifted, the first hour written after
    the move read as a month of negative usage.
    """
    if not written:
        return list(points)
    cut = max(written) - reopen
    anchor: CumulativePoint | None = None
    for point in points:
        if (
            point.start <= cut
            and point.start in written
            and (anchor is None or point.start > anchor.start)
        ):
            anchor = point
    offset = 0.0 if anchor is None else written[anchor.start] - anchor.cumulative
    return [
        CumulativePoint(start=point.start, cumulative=point.cumulative + offset)
        for point in points
        if point.start > cut
    ]


def cost_points(
    points: list[CumulativePoint] | tuple[CumulativePoint, ...], price: float
) -> list[CumulativePoint]:
    """Cumulative cost for a fixed unit price, e.g. SGD per kWh."""
    return [
        CumulativePoint(
            start=point.start, cumulative=round(point.cumulative * price, 4)
        )
        for point in points
    ]


def external_statistic_id(premise_id: str, key: str) -> str:
    """Recorder id for imported history, e.g. ``sp_group:2001590888_electricity``.

    Imported history must not share an id with a sensor: the recorder seeds
    its own running sum from short-term rows only, so a sensor with a
    ``total_increasing`` state class would get a second, conflicting sum.
    """
    return f"{DOMAIN}:{premise_id}_{key}"


def cumulative_points(
    periods: tuple[PeriodReading, ...] | list[PeriodReading],
) -> list[CumulativePoint]:
    ordered = sorted(periods, key=lambda item: item.start)
    total = 0.0
    points: list[CumulativePoint] = []
    for item in ordered:
        total += item.amount
        hour = item.start.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        if points and points[-1].start == hour:
            points[-1] = CumulativePoint(start=hour, cumulative=total)
        else:
            points.append(CumulativePoint(start=hour, cumulative=total))
    return points


def trim_unreported(periods: tuple[PeriodReading, ...]) -> tuple[PeriodReading, ...]:
    """Drop trailing zero slots the AMI feed has not filled yet."""
    ordered = sorted(periods, key=lambda item: item.start)
    last_idx = -1
    for index, item in enumerate(ordered):
        if item.amount:
            last_idx = index
    if last_idx < 0:
        return ()
    return tuple(ordered[: last_idx + 1])


def fold_half_hours(periods: tuple[PeriodReading, ...]) -> tuple[PeriodReading, ...]:
    """Sum 30-minute AMI slots into SGT clock hours for Energy statistics."""
    buckets: dict[datetime, float] = {}
    for item in periods:
        hour = item.start.astimezone(SG_TZ).replace(minute=0, second=0, microsecond=0)
        buckets[hour] = buckets.get(hour, 0.0) + item.amount
    return tuple(
        PeriodReading(start=start, amount=amount)
        for start, amount in sorted(buckets.items())
    )


def monthly_bill_points(bills: tuple[BillInfo, ...]) -> tuple[PeriodReading, ...]:
    """One point per calendar month in SGT, using the issued bill amount."""
    by_month: dict[datetime, float] = {}
    for bill in bills:
        if bill.issued_at is None:
            continue
        month = bill.issued_at.astimezone(SG_TZ).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        by_month[month.astimezone(UTC)] = bill.amount_sgd
    return tuple(
        PeriodReading(start=start, amount=amount)
        for start, amount in sorted(by_month.items())
    )


def merge_ami_periods(
    daily: tuple[PeriodReading, ...],
    hourly: tuple[PeriodReading, ...],
) -> tuple[PeriodReading, ...]:
    """Prefer half-hourly AMI on days it covers; keep the daily point for the rest."""
    hourly_days = {item.start.astimezone(SG_TZ).date() for item in hourly}
    merged = [
        item for item in daily if item.start.astimezone(SG_TZ).date() not in hourly_days
    ]
    merged.extend(hourly)
    return tuple(sorted(merged, key=lambda item: item.start))
