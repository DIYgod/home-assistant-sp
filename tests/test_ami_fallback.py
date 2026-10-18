"""A failed AMI read stands on the previous answer instead of dropping sensors.

Jarvis answers the chart call with a 500 now and then, and the next poll reads
fine. Before, each such poll published no half-hour slots: today's total and
the last slot went unknown for 30 minutes and the cumulative sensor switched
to the daily or the billed series and back.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse

import pytest

from custom_components.sp_group.client import HttpResponse
from custom_components.sp_group.const import (
    AMI_FALLBACK_MAX_AGE,
    AMI_GROUPED_BY_DAILY,
    AMI_GROUPED_BY_HALF_HOUR,
    B2C_HOST,
    JARVIS_AMI_PATH,
    SENSOR_KEY_ELECTRICITY,
    SENSOR_KEY_ELECTRICITY_HOUR,
    SENSOR_KEY_ELECTRICITY_TODAY,
)
from custom_components.sp_group.mapper import sensors_from_usage
from custom_components.sp_group.models import SG_TZ

from .conftest import FIXED_NOW, FixedClock, FixtureTransport, fixture_client

AMI_LABEL = f"POST {B2C_HOST}{JARVIS_AMI_PATH}"
SERVER_ERROR = b'{"message": "Sorry, there was a problem. Please try again later."}'


@dataclass
class AmiOutage(FixtureTransport):
    """Fixture transport whose AMI chart call answers ``status`` for some feeds."""

    failing: set[str] = field(default_factory=set)
    status: int = 500
    half_hour_body: bytes | None = None

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        *,
        timeout: int | None = None,
    ) -> HttpResponse:
        if method == "POST" and urlparse(url).path == JARVIS_AMI_PATH and body:
            grouped = json.loads(body.decode("utf-8"))["grouped_by"]
            if grouped in self.failing:
                return HttpResponse(self.status, SERVER_ERROR)
            if grouped == AMI_GROUPED_BY_HALF_HOUR and self.half_hour_body:
                return HttpResponse(200, self.half_hour_body)
        return super().request(method, url, headers, body, timeout=timeout)


def _values(client_now: datetime, usage: object) -> dict[str, object]:
    return {
        spec.key: spec.native_value
        for spec in sensors_from_usage(usage, client_now)  # type: ignore[arg-type]
    }


def test_a_failed_half_hour_read_keeps_the_previous_slots(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = AmiOutage()
    client = fixture_client(transport)
    good = client.fetch_usage()
    assert good.ami_hourly, "the fixture poll must carry half-hour slots"

    transport.failing = {AMI_GROUPED_BY_HALF_HOUR}
    with caplog.at_level(logging.WARNING):
        failed = client.fetch_usage()

    assert failed.ami_hourly == good.ami_hourly
    now = client.clock.now()
    before, after = _values(now, good), _values(now, failed)
    for key in (SENSOR_KEY_ELECTRICITY_TODAY, SENSOR_KEY_ELECTRICITY_HOUR):
        assert key in after
        assert after[key] == before[key]
    # Standing on the old slots does not hide that the read failed.
    assert "HTTP 500" in caplog.text
    assert "HTTP 500" in client.read_failures[AMI_LABEL]


def test_both_feeds_failing_keep_the_cumulative_sensor_on_its_series() -> None:
    """Both feeds down used to publish the billed total as the cumulative state.

    A total_increasing sensor read that drop and the climb back as a meter
    reset, which added the billed total to its long-term sum.
    """
    transport = AmiOutage()
    client = fixture_client(transport)
    good = client.fetch_usage()

    transport.failing = {AMI_GROUPED_BY_HALF_HOUR, AMI_GROUPED_BY_DAILY}
    failed = client.fetch_usage()

    assert failed.ami_hourly == good.ami_hourly
    assert failed.ami_daily == good.ami_daily
    now = client.clock.now()
    assert (
        _values(now, failed)[SENSOR_KEY_ELECTRICITY]
        == _values(now, good)[SENSOR_KEY_ELECTRICITY]
    )


def test_a_failed_read_with_no_previous_answer_reads_nothing() -> None:
    transport = AmiOutage(failing={AMI_GROUPED_BY_HALF_HOUR})
    usage = fixture_client(transport).fetch_usage()

    assert usage.ami_hourly == ()
    assert SENSOR_KEY_ELECTRICITY_TODAY not in _values(FIXED_NOW, usage)


def test_an_answer_older_than_the_limit_is_not_used() -> None:
    clock = FixedClock()
    transport = AmiOutage()
    client = fixture_client(transport, clock)
    assert client.fetch_usage().ami_hourly

    transport.failing = {AMI_GROUPED_BY_HALF_HOUR}
    clock.instant = FIXED_NOW + AMI_FALLBACK_MAX_AGE
    assert client.fetch_usage().ami_hourly, "at the limit the answer still stands"
    clock.instant = FIXED_NOW + AMI_FALLBACK_MAX_AGE + timedelta(minutes=1)
    assert client.fetch_usage().ami_hourly == ()


def test_a_host_that_says_there_is_nothing_is_not_overruled() -> None:
    """A 404 is an answer, not a failure, so the old slots do not come back."""
    transport = AmiOutage()
    client = fixture_client(transport)
    assert client.fetch_usage().ami_hourly

    transport.failing = {AMI_GROUPED_BY_HALF_HOUR}
    transport.status = 404
    assert client.fetch_usage().ami_hourly == ()


def test_a_body_with_no_usable_slot_stands_on_the_previous_answer() -> None:
    """A 200 the parser finds nothing in is a failed read, and is logged as one."""
    transport = AmiOutage()
    client = fixture_client(transport)
    good = client.fetch_usage()

    transport.half_hour_body = b'{"history": []}'
    assert client.fetch_usage().ami_hourly == good.ami_hourly


def test_the_kept_slots_stay_inside_the_window_the_poll_asked_for() -> None:
    """At midnight the half-hour window drops a day, and so does the stand-in."""
    first_day = datetime(2026, 7, 3, tzinfo=SG_TZ)
    newest = datetime(2026, 8, 2, 1, 0, tzinfo=SG_TZ)
    body = {
        "history": [
            {
                "data": [
                    {"date": first_day.isoformat(), "consumption": 0.3},
                    {"date": newest.isoformat(), "consumption": 0.5},
                ]
            }
        ]
    }
    clock = FixedClock(datetime(2026, 8, 2, 23, 0, tzinfo=SG_TZ))
    transport = AmiOutage(half_hour_body=json.dumps(body).encode())
    client = fixture_client(transport, clock)
    assert [item.start for item in client.fetch_usage().ami_hourly] == [
        first_day,
        newest,
    ]

    transport.failing = {AMI_GROUPED_BY_HALF_HOUR}
    clock.instant = datetime(2026, 8, 3, 1, 0, tzinfo=SG_TZ)
    assert [item.start for item in client.fetch_usage().ami_hourly] == [newest]


def test_each_premise_stands_only_on_its_own_answer() -> None:
    """The stand-in is keyed by premise, so a re-auth to another one starts empty."""
    transport = AmiOutage()
    client = fixture_client(transport)
    good = client.fetch_usage()
    assert good.ami_hourly

    feeds = {premise for premise, _ in client._ami_answers}
    assert feeds == {good.premise_id}


def test_a_failure_with_nothing_to_stand_on_is_flagged_for_the_statistics() -> None:
    """The import must not write a series with a hole where a feed goes."""
    transport = AmiOutage(failing={AMI_GROUPED_BY_DAILY})
    client = fixture_client(transport)

    assert client.fetch_usage().ami_read_failed is True
    transport.failing = set()
    assert client.fetch_usage().ami_read_failed is False


def test_standing_on_the_previous_answer_is_not_a_failure() -> None:
    transport = AmiOutage()
    client = fixture_client(transport)
    assert client.fetch_usage().ami_read_failed is False

    transport.failing = {AMI_GROUPED_BY_HALF_HOUR, AMI_GROUPED_BY_DAILY}
    assert client.fetch_usage().ami_read_failed is False


def test_an_absent_feed_is_not_a_failure() -> None:
    transport = AmiOutage(failing={AMI_GROUPED_BY_HALF_HOUR}, status=404)
    assert fixture_client(transport).fetch_usage().ami_read_failed is False
