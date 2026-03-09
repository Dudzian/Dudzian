from __future__ import annotations

import datetime as dt
import json

from scripts.validate_long_poll_snapshots import prepare_snapshots


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _base_payload(collected_at: str) -> dict[str, object]:
    return {
        "collected_at": collected_at,
        "snapshots": [
            {"labels": {"adapter": "deribit_futures", "scope": "private", "environment": "paper"}},
            {"labels": {"adapter": "deribit_futures", "scope": "private", "environment": "live"}},
            {"labels": {"adapter": "bitmex_futures", "scope": "public", "environment": "paper"}},
            {"labels": {"adapter": "bitmex_futures", "scope": "public", "environment": "live"}},
        ],
    }


def test_prepare_snapshots_accepts_recent_input_without_timestamp_rewrite(tmp_path) -> None:
    now = dt.datetime.now(tz=dt.timezone.utc)
    ts = _iso_utc(now - dt.timedelta(minutes=2))
    source = tmp_path / "input.json"
    output = tmp_path / "out" / "long_poll_snapshots.json"
    source.write_text(json.dumps(_base_payload(ts)), encoding="utf-8")

    prepare_snapshots(
        source_path=source,
        output_path=output,
        max_age_minutes=720.0,
        future_grace_minutes=5.0,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["collected_at"] == ts
    assert len(payload["snapshots"]) == 4
    assert all(snapshot.get("collected_at") == ts for snapshot in payload["snapshots"])


def test_prepare_snapshots_fails_when_required_profile_missing(tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps({"snapshots": [{"labels": {"adapter": "deribit_futures", "scope": "private", "environment": "paper"}}]}),
        encoding="utf-8",
    )

    try:
        prepare_snapshots(
            source_path=source,
            output_path=tmp_path / "out.json",
            max_age_minutes=720.0,
            future_grace_minutes=5.0,
        )
    except RuntimeError as exc:
        assert "Brak wymaganych snapshotów" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Oczekiwano RuntimeError dla brakujących profili")


def test_prepare_snapshots_fails_when_required_snapshots_are_stale(tmp_path) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_base_payload("2025-12-01T18:01:10.893977Z")), encoding="utf-8")

    try:
        prepare_snapshots(
            source_path=source,
            output_path=tmp_path / "out.json",
            max_age_minutes=60.0,
            future_grace_minutes=5.0,
        )
    except RuntimeError as exc:
        assert "walidacji świeżości" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Oczekiwano RuntimeError dla nieświeżych snapshotów")


def test_prepare_snapshots_fails_when_collected_at_far_in_future(tmp_path) -> None:
    future_ts = _iso_utc(dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(minutes=20))
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_base_payload(future_ts)), encoding="utf-8")

    try:
        prepare_snapshots(
            source_path=source,
            output_path=tmp_path / "out.json",
            max_age_minutes=720.0,
            future_grace_minutes=5.0,
        )
    except RuntimeError as exc:
        assert "now+grace" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Oczekiwano RuntimeError dla collected_at zbyt daleko w przyszłości")


def test_prepare_snapshots_skip_freshness_accepts_stale_and_future_timestamp(tmp_path) -> None:
    future_ts = _iso_utc(dt.datetime.now(tz=dt.timezone.utc) + dt.timedelta(days=2))
    source = tmp_path / "input.json"
    source.write_text(json.dumps(_base_payload(future_ts)), encoding="utf-8")

    prepare_snapshots(
        source_path=source,
        output_path=tmp_path / "out.json",
        max_age_minutes=1.0,
        future_grace_minutes=0.0,
        skip_freshness=True,
    )
