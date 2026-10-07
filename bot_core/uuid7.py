"""Shared UUIDv7 arithmetic and CSPRNG minting; these helpers confer no authority."""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone


class UUID7Error(ValueError):
    """The supplied reservation instant cannot be represented as uint48 milliseconds."""


def reservation_epoch_milliseconds(reservation_now: datetime) -> int:
    """Floor one UTC instant to integer Unix milliseconds without float rounding."""
    if not isinstance(reservation_now, datetime) or reservation_now.utcoffset() != timedelta(0):
        raise UUID7Error("INVALID_UUID7_RESERVATION_TIMESTAMP")
    elapsed = reservation_now - datetime(1970, 1, 1, tzinfo=timezone.utc)
    millis = elapsed.days * 86_400_000 + elapsed.seconds * 1000 + elapsed.microseconds // 1000
    if not 0 <= millis < 1 << 48:
        raise UUID7Error("INVALID_UUID7_RESERVATION_TIMESTAMP")
    return millis


def mint_uuid7(prefix: str, millis: int) -> str:
    """Mint the RFC variant and version with independent CSPRNG rand_a and rand_b."""
    if (
        type(prefix) is not str
        or not prefix
        or type(millis) is not int
        or not 0 <= millis < 1 << 48
    ):
        raise UUID7Error("INVALID_UUID7_RESERVATION_TIMESTAMP")
    raw_id = (millis << 80) | (7 << 76) | (secrets.randbits(12) << 64)
    raw_id |= (2 << 62) | secrets.randbits(62)
    return prefix + str(uuid.UUID(int=raw_id))
