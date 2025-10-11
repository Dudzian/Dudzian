from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import io
import json
import logging
import sys
import textwrap

import pytest

from scripts.verify_decision_log import main as verify_main


def _write_risk_profile_file(tmp_path, name: str = "ops", severity: str = "error"):
    profiles_path = tmp_path / "telemetry_profiles.json"
    payload = {"risk_profiles": {name: {"severity_min": severity}}}
    profiles_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return profiles_path


def _write_core_config(tmp_path, *, profiles_path, profile_name: str = "ops"):
    config_path = tmp_path / "core.yaml"
    profiles_str = str(profiles_path)
    config_path.write_text(
        textwrap.dedent(
            f"""
            risk_profiles:
              conservative:
                max_daily_loss_pct: 0.05
                max_position_pct: 0.10
                target_volatility: 0.2
                max_leverage: 3.0
                stop_loss_atr_multiple: 1.5
                max_open_positions: 5
                hard_drawdown_pct: 0.25
            environments: {{}}
            runtime:
              metrics_service:
                ui_alerts_risk_profile: {profile_name}
                ui_alerts_risk_profiles_file: {profiles_str}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _signed_entry(payload: dict[str, object], *, key: bytes, key_id: str | None) -> dict[str, object]:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(key, canonical, hashlib.sha256).digest()
    signature = {
        "algorithm": "HMAC-SHA256",
        "value": base64.b64encode(digest).decode("ascii"),
    }
    if key_id is not None:
        signature["key_id"] = key_id
    signed = dict(payload)
    signed["signature"] = signature
    return signed


def _signed_summary_payload(summary: dict[str, object], *, key: bytes, key_id: str | None) -> dict[str, object]:
    payload = {"summary": summary}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(key, canonical, hashlib.sha256).digest()
    signature = {
        "algorithm": "HMAC-SHA256",
        "value": base64.b64encode(digest).decode("ascii"),
    }
    if key_id is not None:
        signature["key_id"] = key_id
    payload["signature"] = signature
    return payload


def test_verify_success(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    log_path = tmp_path / "decision.jsonl"
    key = b"secret"
    key_id = "ops-key"

    entries = [
        _signed_entry(
            {
                "kind": "metadata",
                "timestamp": "2024-02-01T00:00:00+00:00",
                "metadata": {"mode": "grpc"},
            },
            key=key,
            key_id=key_id,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2024-02-01T00:00:01+00:00",
                "source": "grpc",
                "event": "reduce_motion",
                "severity": "warning",
                "fps": 45.0,
                "screen": {"name": "Main"},
                "notes": {"severity": "warning"},
            },
            key=key,
            key_id=key_id,
        ),
    ]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    exit_code = verify_main([
        str(log_path),
        "--hmac-key",
        key.decode("utf-8"),
        "--hmac-key-id",
        key_id,
    ])

    assert exit_code == 0
    assert any("OK: zweryfikowano" in message for message in caplog.messages)


def test_verify_fails_without_signature(tmp_path):
    log_path = tmp_path / "decision.jsonl"
    log_path.write_text(
        json.dumps({"kind": "snapshot", "timestamp": "2024-02-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main([str(log_path)])
    assert exit_code == 2


def test_verify_allows_unsigned_when_flag_set(tmp_path):
    log_path = tmp_path / "decision.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"kind": "metadata", "timestamp": "2024", "metadata": {}}),
                json.dumps({"kind": "snapshot", "timestamp": "2024"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main([str(log_path), "--allow-unsigned"])
    assert exit_code == 0


def test_verify_supports_gzip(tmp_path):
    log_path = tmp_path / "decision.jsonl.gz"
    key = b"secret"
    signed = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-02-01T00:00:00Z",
            "event": "reduce_motion",
            "source": "jsonl",
            "severity": "info",
            "fps": 59.0,
            "screen": {},
            "notes": {"severity": "info"},
        },
        key=key,
        key_id=None,
    )
    with gzip.open(log_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(signed, ensure_ascii=False) + "\n")

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8")])
    assert exit_code == 0


def test_verify_reads_from_stdin(monkeypatch, tmp_path):
    key = b"k"
    signed = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024",
            "event": "reduce_motion",
            "severity": "info",
            "source": "stdin",
            "fps": 60,
            "screen": {},
            "notes": {"severity": "info"},
        },
        key=key,
        key_id=None,
    )
    buffer = io.StringIO(json.dumps(signed) + "\n")
    monkeypatch.setattr(sys, "stdin", buffer)

    exit_code = verify_main(["-", "--hmac-key", key.decode("utf-8")])
    assert exit_code == 0


def test_print_risk_profiles_cli(tmp_path, capsys):
    profiles_path = _write_risk_profile_file(tmp_path, name="desk", severity="notice")

    exit_code = verify_main([
        "--print-risk-profiles",
        "--risk-profiles-file",
        str(profiles_path),
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["risk_profiles"]["desk"]["severity_min"] == "notice"
    assert payload["risk_profiles"]["desk"]["origin"].startswith("verify:")


def test_print_risk_profiles_env(monkeypatch, tmp_path, capsys):
    profiles_path = _write_risk_profile_file(tmp_path, name="desk", severity="warning")
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_PRINT_RISK_PROFILES", "1")
    monkeypatch.setenv(
        "BOT_CORE_VERIFY_DECISION_LOG_RISK_PROFILES_FILE",
        str(profiles_path),
    )

    exit_code = verify_main([])

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["risk_profiles"]["desk"]["severity_min"] == "warning"

def test_verify_env_configuration(monkeypatch, tmp_path):
    log_path = tmp_path / "decision.jsonl"
    key = b"secret"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024",
            "event": "reduce_motion",
            "severity": "info",
            "source": "env",
            "fps": 30,
            "screen": {},
            "notes": {"severity": "info"},
        },
        key=key,
        key_id="ops",
    )
    log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_PATH", str(log_path))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY", key.decode("utf-8"))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY_ID", "ops")

    exit_code = verify_main([])
    assert exit_code == 0


def test_verify_creates_report_output(tmp_path):
    log_path = tmp_path / "decision.jsonl"
    report_path = tmp_path / "report.json"
    key = b"secret"
    key_id = "ops"
    entries = [
        _signed_entry(
            {
                "kind": "metadata",
                "timestamp": "2024-03-01T00:00:00Z",
                "metadata": {"mode": "grpc", "filters": {"event": "reduce_motion"}},
            },
            key=key,
            key_id=key_id,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2024-03-01T00:00:01Z",
                "event": "reduce_motion",
                "severity": "warning",
                "source": "grpc",
                "fps": 48.5,
                "screen": {"index": 0, "name": "Main"},
                "notes": {"severity": "warning"},
            },
            key=key,
            key_id=key_id,
        ),
    ]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
            "--report-output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_version"] == 1
    assert payload["metadata"]["mode"] == "grpc"
    assert payload["summary"]["total_snapshots"] == 1
    assert "summary_validation" not in payload


def test_verify_report_output_env(monkeypatch, tmp_path):
    log_path = tmp_path / "decision.jsonl"
    report_path = tmp_path / "report.json"
    key = b"secret"
    entries = [
        _signed_entry(
            {
                "kind": "metadata",
                "timestamp": "2024-04-01T00:00:00Z",
                "metadata": {"mode": "jsonl"},
            },
            key=key,
            key_id=None,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2024-04-01T00:00:01Z",
                "event": "overlay_budget",
                "severity": "info",
                "source": "jsonl",
                "screen": {"index": 1, "name": "Secondary"},
                "fps": 55.0,
                "notes": {"severity": "info"},
            },
            key=key,
            key_id=None,
        ),
    ]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_PATH", str(log_path))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY", key.decode("utf-8"))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_REPORT_OUTPUT", str(report_path))

    exit_code = verify_main([])

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["mode"] == "jsonl"
    assert payload["summary"]["events"]["overlay_budget"]["count"] == 1


def test_verify_rejects_signature_mismatch(tmp_path):
    log_path = tmp_path / "decision.jsonl"
    entry = {
        "kind": "snapshot",
        "timestamp": "2024",
        "event": "reduce_motion",
        "severity": "info",
        "source": "grpc",
        "fps": 30,
        "screen": {},
        "notes": {"severity": "info"},
        "signature": {"algorithm": "HMAC-SHA256", "value": base64.b64encode(b"wrong").decode("ascii")},
    }
    log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    exit_code = verify_main([str(log_path), "--hmac-key", "secret"])
    assert exit_code == 2


def test_verify_summary_validation(tmp_path):
    key = b"secret"
    key_id = "ops"
    log_path = tmp_path / "decision.jsonl"
    summary_path = tmp_path / "summary.json"

    entries = [
        _signed_entry(
            {
                "kind": "metadata",
                "timestamp": "2024-02-01T00:00:00+00:00",
                "metadata": {"mode": "jsonl", "summary_enabled": True},
            },
            key=key,
            key_id=key_id,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2024-02-01T00:00:01+00:00",
                "source": "jsonl",
                "event": "reduce_motion",
                "severity": "warning",
                "fps": 45.0,
                "screen": {"index": 0, "name": "Main"},
                "notes": {"event": "reduce_motion", "severity": "warning"},
            },
            key=key,
            key_id=key_id,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2024-02-01T00:00:02+00:00",
                "source": "jsonl",
                "event": "overlay_budget",
                "severity": "error",
                "fps": 30.0,
                "screen": {"index": 1, "name": "Aux"},
                "notes": {"event": "overlay_budget", "severity": "error"},
            },
            key=key,
            key_id=key_id,
        ),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary_payload = {
        "summary": {
            "total_snapshots": 2,
            "first_timestamp": "2024-02-01T00:00:01+00:00",
            "last_timestamp": "2024-02-01T00:00:02+00:00",
            "severity_counts": {"error": 1, "warning": 1},
            "events": {
                "overlay_budget": {
                    "count": 1,
                    "fps": {"min": 30.0, "max": 30.0, "avg": 30.0, "samples": 1},
                    "screens": [{"index": 1, "name": "Aux"}],
                    "severity": {"counts": {"error": 1}},
                    "first_timestamp": "2024-02-01T00:00:02+00:00",
                    "last_timestamp": "2024-02-01T00:00:02+00:00",
                },
                "reduce_motion": {
                    "count": 1,
                    "fps": {"min": 45.0, "max": 45.0, "avg": 45.0, "samples": 1},
                    "screens": [{"index": 0, "name": "Main"}],
                    "severity": {"counts": {"warning": 1}},
                    "first_timestamp": "2024-02-01T00:00:01+00:00",
                    "last_timestamp": "2024-02-01T00:00:01+00:00",
                },
            },
        }
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False), encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0


def test_verify_summary_mismatch(tmp_path):
    key = b"secret"
    log_path = tmp_path / "decision.jsonl"
    summary_path = tmp_path / "summary.json"

    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-02-01T00:00:01+00:00",
            "source": "jsonl",
            "event": "reduce_motion",
            "severity": "warning",
            "fps": 45.0,
            "screen": {"index": 0, "name": "Main"},
            "notes": {"event": "reduce_motion", "severity": "warning"},
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_snapshots": 2,
                    "events": {"reduce_motion": {"count": 2}},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = verify_main(
        [str(log_path), "--hmac-key", key.decode("utf-8"), "--summary-json", str(summary_path)]
    )

    assert exit_code == 2


def test_verify_summary_signature_success(tmp_path):
    key = b"supersecret"
    key_id = "ops-2025"
    log_path = tmp_path / "signed.jsonl"
    summary_path = tmp_path / "summary.json"

    metadata = {
        "mode": "jsonl",
        "summary_enabled": True,
        "summary_signature": {"algorithm": "HMAC-SHA256", "key_id": key_id},
    }
    entries = [
        _signed_entry(
            {"kind": "metadata", "timestamp": "2024-06-01T00:00:00+00:00", "metadata": metadata},
            key=key,
            key_id=key_id,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2024-06-01T00:00:01+00:00",
                "source": "jsonl",
                "event": "reduce_motion",
                "severity": "warning",
                "fps": 50.0,
                "screen": {"index": 0, "name": "Ops"},
                "notes": {"event": "reduce_motion", "severity": "warning"},
            },
            key=key,
            key_id=key_id,
        ),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary_body = {
        "total_snapshots": 1,
        "first_timestamp": "2024-06-01T00:00:01+00:00",
        "last_timestamp": "2024-06-01T00:00:01+00:00",
        "severity_counts": {"warning": 1},
        "events": {
            "reduce_motion": {
                "count": 1,
                "fps": {"min": 50.0, "max": 50.0, "avg": 50.0, "samples": 1},
                "screens": [{"index": 0, "name": "Ops"}],
                "severity": {"counts": {"warning": 1}},
                "first_timestamp": "2024-06-01T00:00:01+00:00",
                "last_timestamp": "2024-06-01T00:00:01+00:00",
            }
        },
    }
    summary_payload = _signed_summary_payload(summary_body, key=key, key_id=key_id)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False), encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0


def test_verify_summary_signature_missing(tmp_path):
    key = b"supersecret"
    key_id = "ops-2025"
    log_path = tmp_path / "missing_sig.jsonl"
    summary_path = tmp_path / "summary.json"

    metadata = {
        "mode": "jsonl",
        "summary_enabled": True,
        "summary_signature": {"algorithm": "HMAC-SHA256", "key_id": key_id},
    }
    entries = [
        _signed_entry(
            {"kind": "metadata", "timestamp": "2024-06-01T00:00:00+00:00", "metadata": metadata},
            key=key,
            key_id=key_id,
        )
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary_payload = {"summary": {"total_snapshots": 0, "events": {}}}
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False), encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 2


def test_verify_summary_signature_mismatch(tmp_path):
    key = b"supersecret"
    log_path = tmp_path / "mismatch.jsonl"
    summary_path = tmp_path / "summary.json"

    metadata = {
        "mode": "jsonl",
        "summary_enabled": True,
        "summary_signature": {"algorithm": "HMAC-SHA256", "key_id": "ops-a"},
    }
    entries = [
        _signed_entry(
            {"kind": "metadata", "timestamp": "2024-06-01T00:00:00+00:00", "metadata": metadata},
            key=key,
            key_id="ops-a",
        )
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary_body = {"total_snapshots": 0, "events": {}}
    summary_payload = _signed_summary_payload(summary_body, key=key, key_id="ops-b")
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False), encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            "ops-a",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 2

def test_verify_metadata_expectations(tmp_path):
    key = b"secret"
    key_id = "ops"
    log_path = tmp_path / "audit.jsonl"

    metadata_entry = _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-02-01T00:00:00+00:00",
            "metadata": {
                "mode": "jsonl",
                "input_file": str(log_path),
                "summary_enabled": True,
                "filters": {
                    "event": "reduce_motion",
                    "severity_min": "warning",
                },
            },
        },
        key=key,
        key_id=key_id,
    )
    snapshot_entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-02-01T00:00:01+00:00",
            "event": "reduce_motion",
            "severity": "warning",
            "source": "jsonl",
            "fps": 55.0,
            "screen": {"index": 0},
            "notes": {"severity": "warning"},
        },
        key=key,
        key_id=key_id,
    )
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in (metadata_entry, snapshot_entry)) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
            "--expect-mode",
            "jsonl",
            "--expect-summary-enabled",
            "--expect-filter",
            "event=reduce_motion",
            "--expect-filter",
            "severity_min=warning",
            "--expect-input-file",
            str(log_path),
        ]
    )

    assert exit_code == 0


def test_verify_metadata_expectation_failure(tmp_path):
    key = b"secret"
    log_path = tmp_path / "audit.jsonl"
    metadata_entry = _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "metadata": {"mode": "jsonl"},
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(metadata_entry) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [str(log_path), "--hmac-key", key.decode("utf-8"), "--expect-mode", "grpc"]
    )

    assert exit_code == 2


def test_verify_metadata_env_expectations(monkeypatch, tmp_path):
    key = b"secret"
    key_id = "ops"
    log_path = tmp_path / "grpc.jsonl"
    metadata_entry = _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-05-01T00:00:00+00:00",
            "metadata": {
                "mode": "grpc",
                "endpoint": "localhost:50051",
                "use_tls": True,
                "auth_token_provided": True,
                "summary_enabled": True,
                "filters": {"event": "overlay_budget"},
            },
        },
        key=key,
        key_id=key_id,
    )
    snapshot_entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-05-01T00:00:01+00:00",
            "event": "overlay_budget",
            "severity": "critical",
            "source": "grpc",
            "fps": 30.0,
            "screen": {"index": 1},
            "notes": {"severity": "critical"},
        },
        key=key,
        key_id=key_id,
    )
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in (metadata_entry, snapshot_entry)) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_PATH", str(log_path))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY", key.decode("utf-8"))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY_ID", key_id)
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_EXPECT_MODE", "grpc")
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_EXPECT_ENDPOINT", "localhost:50051")
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_EXPECT_SUMMARY_ENABLED", "true")
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_REQUIRE_TLS", "1")
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_REQUIRE_AUTH_TOKEN", "yes")
    monkeypatch.setenv(
        "BOT_CORE_VERIFY_DECISION_LOG_EXPECT_FILTERS_JSON",
        json.dumps({"event": "overlay_budget"}),
    )

    exit_code = verify_main([])
    assert exit_code == 0


def test_verify_metadata_filters_mismatch(tmp_path):
    key = b"secret"
    log_path = tmp_path / "audit.jsonl"
    metadata_entry = _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-03-01T00:00:00+00:00",
            "metadata": {
                "mode": "jsonl",
                "filters": {"event": "reduce_motion"},
            },
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(metadata_entry) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--expect-filter",
            "event=jank",
        ]
    )

    assert exit_code == 2


def test_verify_requires_metadata_when_expected(tmp_path):
    key = b"secret"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-04-01T00:00:00+00:00",
            "event": "reduce_motion",
            "severity": "warning",
            "source": "jsonl",
            "fps": 58.0,
            "screen": {},
            "notes": {"severity": "warning"},
        },
        key=key,
        key_id=None,
    )
    log_path = tmp_path / "missing_meta.jsonl"
    log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [str(log_path), "--hmac-key", key.decode("utf-8"), "--expect-mode", "jsonl"]
    )

    assert exit_code == 2


def test_verify_requires_screen_info(tmp_path):
    key = b"secret"
    log_path = tmp_path / "missing_screen.jsonl"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-04-01T00:00:00+00:00",
            "event": "reduce_motion",
            "severity": "warning",
            "source": "jsonl",
            "notes": {"severity": "warning"},
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-screen-info",
        ]
    )

    assert exit_code == 2


def test_verify_env_requires_screen_info(monkeypatch, tmp_path):
    key = b"secret"
    log_path = tmp_path / "screen.jsonl"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-04-01T00:00:00+00:00",
            "event": "reduce_motion",
            "severity": "warning",
            "source": "jsonl",
            "fps": 58.0,
            "screen": {"index": 0, "refresh_hz": 60.0},
            "notes": {"severity": "warning", "screen": {"index": 0}},
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_PATH", str(log_path))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY", key.decode("utf-8"))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_REQUIRE_SCREEN_INFO", "1")

    exit_code = verify_main([])

    assert exit_code == 0


def _write_signed_log(entries: list[dict[str, object]], path):
    path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _metadata_entry(
    *,
    key: bytes,
    key_id: str | None,
    metadata: dict[str, object],
) -> dict[str, object]:
    return _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-06-01T00:00:00+00:00",
            "metadata": metadata,
        },
        key=key,
        key_id=key_id,
    )


def _snapshot_entry(
    *,
    key: bytes,
    key_id: str | None,
    timestamp: str,
    event: str,
    severity: str,
    screen: dict[str, object],
) -> dict[str, object]:
    return _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": timestamp,
            "event": event,
            "severity": severity,
            "source": "jsonl",
            "fps": 58.0,
            "screen": screen,
            "notes": {"severity": severity},
        },
        key=key,
        key_id=key_id,
    )


def test_verify_fails_when_severity_list_not_matched(tmp_path):
    key = b"sec"
    key_id = "ops"
    log_path = tmp_path / "severity.jsonl"
    metadata = _metadata_entry(
        key=key,
        key_id=key_id,
        metadata={
            "mode": "jsonl",
            "filters": {"severity": ["critical", "error"]},
        },
    )
    snapshot = _snapshot_entry(
        key=key,
        key_id=key_id,
        timestamp="2024-06-01T00:00:01+00:00",
        event="reduce_motion",
        severity="warning",
        screen={"index": 0, "name": "Main"},
    )
    _write_signed_log([metadata, snapshot], log_path)

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8"), "--hmac-key-id", key_id])

    assert exit_code == 2


def test_verify_fails_when_severity_min_not_respected(tmp_path):
    key = b"sec"
    log_path = tmp_path / "severity_min.jsonl"
    metadata = _metadata_entry(
        key=key,
        key_id=None,
        metadata={
            "mode": "jsonl",
            "filters": {"severity_min": "warning"},
        },
    )
    snapshot = _snapshot_entry(
        key=key,
        key_id=None,
        timestamp="2024-06-01T00:00:01+00:00",
        event="reduce_motion",
        severity="info",
        screen={"index": 0},
    )
    _write_signed_log([metadata, snapshot], log_path)

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8")])

    assert exit_code == 2


def test_verify_fails_when_time_window_not_respected(tmp_path):
    key = b"sec"
    log_path = tmp_path / "time.jsonl"
    metadata = _metadata_entry(
        key=key,
        key_id=None,
        metadata={
            "mode": "jsonl",
            "filters": {
                "since": "2024-06-01T00:00:05+00:00",
                "until": "2024-06-01T00:00:10+00:00",
            },
        },
    )
    snapshot = _snapshot_entry(
        key=key,
        key_id=None,
        timestamp="2024-06-01T00:00:02+00:00",
        event="reduce_motion",
        severity="warning",
        screen={"index": 0},
    )
    _write_signed_log([metadata, snapshot], log_path)

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8")])

    assert exit_code == 2


def test_verify_fails_when_screen_or_event_filters_not_matched(tmp_path):
    key = b"sec"
    log_path = tmp_path / "screen_event.jsonl"
    metadata = _metadata_entry(
        key=key,
        key_id=None,
        metadata={
            "mode": "jsonl",
            "filters": {
                "event": "reduce_motion",
                "screen_index": 1,
                "screen_name": "Main",
            },
        },
    )
    snapshot = _snapshot_entry(
        key=key,
        key_id=None,
        timestamp="2024-06-01T00:00:02+00:00",
        event="overlay_budget",
        severity="warning",
        screen={"index": 0, "name": "Aux monitor"},
    )
    _write_signed_log([metadata, snapshot], log_path)

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8")])

    assert exit_code == 2


def test_verify_passes_when_filters_respected(tmp_path):
    key = b"sec"
    key_id = "ops"
    log_path = tmp_path / "ok.jsonl"
    metadata = _metadata_entry(
        key=key,
        key_id=key_id,
        metadata={
            "mode": "jsonl",
            "filters": {
                "event": "reduce_motion",
                "severity": ["warning", "error"],
                "severity_min": "warning",
                "screen_index": 0,
                "screen_name": "Main",
                "since": "2024-06-01T00:00:00+00:00",
                "until": "2024-06-01T00:00:10+00:00",
                "limit": 5,
            },
        },
    )
    snapshot = _snapshot_entry(
        key=key,
        key_id=key_id,
        timestamp="2024-06-01T00:00:02+00:00",
        event="reduce_motion",
        severity="warning",
        screen={"index": 0, "name": "Main Display"},
    )
    _write_signed_log([metadata, snapshot], log_path)

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8"), "--hmac-key-id", key_id])

    assert exit_code == 0


def test_verify_enforces_event_limit(tmp_path):
    key = b"limit"
    log_path = tmp_path / "limit.jsonl"
    entries = [
        _metadata_entry(key=key, key_id=None, metadata={"mode": "jsonl"}),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-06-01T00:00:02+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0},
        ),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-06-01T00:00:03+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0},
        ),
    ]
    _write_signed_log(entries, log_path)

    exit_code = verify_main([
        str(log_path),
        "--hmac-key",
        key.decode("utf-8"),
        "--max-event-count",
        "reduce_motion=1",
    ])

    assert exit_code == 2


def test_verify_event_limit_from_env(monkeypatch, tmp_path):
    key = b"limit"
    log_path = tmp_path / "limit_env.jsonl"
    entries = [
        _metadata_entry(key=key, key_id=None, metadata={"mode": "jsonl"}),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-06-01T00:00:02+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0},
        ),
    ]
    _write_signed_log(entries, log_path)

    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_MAX_EVENT_COUNTS_JSON", json.dumps({"reduce_motion": 1}))

    exit_code = verify_main([
        str(log_path),
        "--hmac-key",
        key.decode("utf-8"),
    ])

    assert exit_code == 0


def test_verify_enforces_min_event_count(tmp_path):
    key = b"min"
    log_path = tmp_path / "min.jsonl"
    entries = [
        _metadata_entry(key=key, key_id=None, metadata={"mode": "jsonl"}),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-06-01T00:00:02+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0},
        ),
    ]
    _write_signed_log(entries, log_path)

    exit_code = verify_main([
        str(log_path),
        "--hmac-key",
        key.decode("utf-8"),
        "--min-event-count",
        "reduce_motion=2",
    ])

    assert exit_code == 2


def test_verify_min_event_count_from_env(monkeypatch, tmp_path):
    key = b"min"
    log_path = tmp_path / "min_env.jsonl"
    entries = [
        _metadata_entry(key=key, key_id=None, metadata={"mode": "jsonl"}),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-06-01T00:00:02+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0},
        ),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-06-01T00:00:03+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0},
        ),
    ]
    _write_signed_log(entries, log_path)

    monkeypatch.setenv(
        "BOT_CORE_VERIFY_DECISION_LOG_MIN_EVENT_COUNTS_JSON",
        json.dumps({"reduce_motion": 2}),
    )

    exit_code = verify_main([
        str(log_path),
        "--hmac-key",
        key.decode("utf-8"),
    ])

    assert exit_code == 0


def test_verify_risk_profile_conservative(tmp_path):
    key = b"risk"
    key_id = "ops"
    log_path = tmp_path / "risk_profile.jsonl"
    entries = [
        _metadata_entry(
            key=key,
            key_id=key_id,
            metadata={"mode": "jsonl", "summary_enabled": True},
        ),
        _snapshot_entry(
            key=key,
            key_id=key_id,
            timestamp="2024-07-01T00:00:01+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0, "name": "Main", "manufacturer": "ACME"},
        ),
    ]
    _write_signed_log(entries, log_path)

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
            "--risk-profile",
            "conservative",
        ]
    )

    assert exit_code == 0


def test_verify_risk_profile_rejects_low_severity(tmp_path):
    key = b"risk"
    key_id = "ops"
    log_path = tmp_path / "risk_profile_fail.jsonl"
    entries = [
        _metadata_entry(
            key=key,
            key_id=key_id,
            metadata={"mode": "jsonl", "summary_enabled": True},
        ),
        _snapshot_entry(
            key=key,
            key_id=key_id,
            timestamp="2024-07-01T00:00:01+00:00",
            event="reduce_motion",
            severity="info",
            screen={"index": 0, "name": "Main", "manufacturer": "ACME"},
        ),
    ]
    _write_signed_log(entries, log_path)

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
            "--risk-profile",
            "conservative",
        ]
    )

    assert exit_code == 2


def test_verify_risk_profile_file_cli(tmp_path):
    key = b"risk"
    log_path = tmp_path / "custom_profile.jsonl"
    report_path = tmp_path / "report.json"
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps({"risk_profiles": {"ops": {"severity_min": "error"}}}, ensure_ascii=False)
    )

    entries = [
        _metadata_entry(
            key=key,
            key_id="ops",
            metadata={"mode": "jsonl", "summary_enabled": True},
        ),
        _snapshot_entry(
            key=key,
            key_id="ops",
            timestamp="2024-07-01T00:00:01+00:00",
            event="reduce_motion",
            severity="error",
            screen={"index": 0, "name": "Ops"},
        ),
    ]
    _write_signed_log(entries, log_path)

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--risk-profiles-file",
            str(profiles_path),
            "--risk-profile",
            "ops",
            "--report-output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["risk_profile"]["name"] == "ops"
    assert report_payload["risk_profile"]["origin"].startswith("verify:")
    assert report_payload["risk_profile"]["summary"]["name"] == "ops"
    assert report_payload["risk_profile_summary"]["name"] == "ops"


def test_verify_risk_profile_file_env(monkeypatch, tmp_path):
    key = b"risk"
    log_path = tmp_path / "env_profile.jsonl"
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps({"risk_profiles": {"desk": {"severity_min": "notice"}}}, ensure_ascii=False)
    )

    entries = [
        _metadata_entry(
            key=key,
            key_id=None,
            metadata={"mode": "jsonl", "summary_enabled": True},
        ),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-07-01T00:00:01+00:00",
            event="reduce_motion",
            severity="notice",
            screen={"index": 0, "name": "Desk"},
        ),
    ]
    _write_signed_log(entries, log_path)

    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_RISK_PROFILES_FILE", str(profiles_path))
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_RISK_PROFILE", "desk")
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_HMAC_KEY", key.decode("utf-8"))

    exit_code = verify_main([str(log_path)])

    assert exit_code == 0
def test_verify_risk_profile_env_and_report(monkeypatch, tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk_profile_env.jsonl"
    report_path = tmp_path / "report.json"
    entries = [
        _metadata_entry(
            key=key,
            key_id=None,
            metadata={"mode": "jsonl", "summary_enabled": True},
        ),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-07-01T00:00:01+00:00",
            event="reduce_motion",
            severity="notice",
            screen={"index": 0, "name": "Main", "manufacturer": "ACME"},
        ),
    ]
    _write_signed_log(entries, log_path)

    monkeypatch.delenv("BOT_CORE_VERIFY_DECISION_LOG_RISK_PROFILES_FILE", raising=False)
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_RISK_PROFILE", "balanced")
    monkeypatch.setenv("BOT_CORE_VERIFY_DECISION_LOG_REPORT_OUTPUT", str(report_path))

    exit_code = verify_main([
        str(log_path),
        "--hmac-key",
        key.decode("utf-8"),
    ])

    assert exit_code == 0
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["risk_profile"]["name"] == "balanced"
    assert report_payload["risk_profile"]["severity_min"] == "notice"
    assert report_payload["risk_profile"]["origin"] == "builtin"
    assert report_payload["risk_profile"]["summary"]["name"] == "balanced"
    assert (
        report_payload["risk_profile"]["summary"]["severity_min"]
        == "notice"
    )
    assert report_payload["risk_profile_summary"]["name"] == "balanced"


def test_verify_core_config_risk_profile(tmp_path):
    key = b"risk"
    log_path = tmp_path / "core_profile.jsonl"
    report_path = tmp_path / "report.json"

    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    core_config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        profile_name="ops",
    )

    entries = [
        _metadata_entry(
            key=key,
            key_id=None,
            metadata={
                "mode": "jsonl",
                "summary_enabled": True,
            },
        ),
        _snapshot_entry(
            key=key,
            key_id=None,
            timestamp="2024-07-01T00:00:01+00:00",
            event="reduce_motion",
            severity="warning",
            screen={"index": 0, "name": "Ops", "manufacturer": "ACME"},
        ),
    ]
    _write_signed_log(entries, log_path)

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--core-config",
            str(core_config_path),
            "--report-output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["risk_profile"]["name"] == "ops"
    assert report_payload["risk_profile"]["origin"].startswith("verify:")
    assert report_payload["risk_profile"].get("source") == "core_config"
    assert report_payload["risk_profile"]["summary"]["name"] == "ops"
    assert report_payload["risk_profile_summary"]["name"] == "ops"
    assert report_payload["core_config"]["path"] == str(core_config_path)
    assert report_payload["core_config"]["metrics_service"]["risk_profile"] == "ops"
    assert (
        report_payload["core_config"]["metrics_service"]["risk_profiles_file"]
        == str(profiles_path)
    )
