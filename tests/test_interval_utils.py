from __future__ import annotations

from pathlib import Path

import pytest


from bot_core.data.intervals import interval_to_milliseconds, normalize_interval_token


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        (" 1d ", "1d"),
        ("D1", "1d"),
        ("h4", "4h"),
        ("H4", "4h"),
    ],
)
def test_normalize_interval_token(raw, expected):
    assert normalize_interval_token(raw) == expected


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1s", 1000),
        ("5m", 5 * 60 * 1000),
        ("2H", 2 * 3600 * 1000),
        ("1d", 24 * 3600 * 1000),
        ("3W", 3 * 7 * 24 * 3600 * 1000),
    ],
)
def test_interval_to_milliseconds(interval: str, expected: int):
    assert interval_to_milliseconds(interval) == expected


@pytest.mark.parametrize("bad_interval", ["x", "", "foo", "99q"])
def test_interval_to_milliseconds_invalid(bad_interval: str):
    with pytest.raises(ValueError):
        interval_to_milliseconds(bad_interval)
