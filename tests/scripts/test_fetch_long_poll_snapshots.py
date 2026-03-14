from __future__ import annotations

import json

import pytest

from scripts import fetch_long_poll_snapshots as fetcher


def _fake_snapshot(adapter: str, environment: str) -> dict[str, object]:
    return {
        "labels": {
            "adapter": adapter,
            "scope": "public",
            "environment": environment,
            "region": "eu",
        },
        "requestLatency": {"count": 2, "p95": 0.5},
        "deliveryLag": {"count": 2, "p95": 0.7},
        "httpErrors": {"total": 0, "byStatusCode": {}},
        "reconnects": {"attempts": 0, "success": 0, "failure": 0},
    }


def test_fetch_snapshots_writes_all_required_profiles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _stub_collect_single_snapshot(**kwargs):
        return _fake_snapshot(kwargs["adapter"], kwargs["environment"])

    monkeypatch.setattr(fetcher, "_collect_single_snapshot", _stub_collect_single_snapshot)

    output = tmp_path / "long_poll_snapshots.json"
    fetcher.fetch_snapshots(
        base_url="http://127.0.0.1:8765",
        output_path=output,
        channels=("ticker",),
        timeout_seconds=1.0,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "collected_at" in payload
    assert len(payload["snapshots"]) == 4
    adapters_envs = {
        (item["labels"]["adapter"], item["labels"]["environment"]) for item in payload["snapshots"]
    }
    assert adapters_envs == {
        ("deribit_futures", "paper"),
        ("deribit_futures", "live"),
        ("bitmex_futures", "paper"),
        ("bitmex_futures", "live"),
    }


def test_fetch_snapshots_fails_when_required_profile_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _stub_collect_single_snapshot(**kwargs):
        adapter = kwargs["adapter"]
        environment = kwargs["environment"]
        if adapter == "bitmex_futures" and environment == "live":
            return _fake_snapshot("bitmex_futures", "paper")
        return _fake_snapshot(adapter, environment)

    monkeypatch.setattr(fetcher, "_collect_single_snapshot", _stub_collect_single_snapshot)

    with pytest.raises(RuntimeError, match="Brak wymaganych snapshotów"):
        fetcher.fetch_snapshots(
            base_url="http://127.0.0.1:8765",
            output_path=tmp_path / "long_poll_snapshots.json",
            channels=("ticker",),
            timeout_seconds=1.0,
        )


def test_collect_single_snapshot_uses_stream_gateway_path_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyStream:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            return None

        def export_metrics_snapshot(self):
            return {
                "labels": {"adapter": "deribit_futures", "scope": "public", "environment": "paper"},
                "requestLatency": {"count": 1, "p95": 0.2},
            }

        def close(self):
            return None

    monkeypatch.setattr(fetcher, "LocalLongPollStream", _DummyStream)

    snapshot = fetcher._collect_single_snapshot(
        base_url="http://127.0.0.1:8765",
        adapter="deribit_futures",
        environment="paper",
        scope="public",
        channels=("ticker",),
        timeout_seconds=1.0,
    )

    assert captured["path"] == "/stream/deribit_futures/public"
    assert captured["environment"] == "paper"
    assert captured["scope"] == "public"
    assert snapshot["labels"]["environment"] == "paper"
