from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import importlib.util
import io
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _format_env(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


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


def _load_verify_module_without_jsonschema(monkeypatch):
    module_name = "scripts.verify_decision_log_no_jsonschema"
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "scripts" / "verify_decision_log.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


def test_list_schema_aliases_outputs_builtin_aliases(capsys):
    exit_code = verify_main(["--list-schema-aliases"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "schemas" in payload
    schemas = {entry["canonical"]: entry for entry in payload["schemas"]}
    assert "builtin:decision_log_v2" in schemas
    info = schemas["builtin:decision_log_v2"]
    assert "decision_log_v2" in info["aliases"]
    assert any(alias.startswith("https://") for alias in info["aliases"])


def test_describe_schema_alias_outputs_details(capsys):
    exit_code = verify_main(["--describe-schema-alias", "builtin:decision_log_v2"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["canonical"] == "builtin:decision_log_v2"
    assert payload["supports_builtin_validator"] is True
    requirements = payload.get("requirements")
    assert requirements is not None
    assert set(requirements["required_fields"]) == {
        "artefacts",
        "runtime_flags",
        "signatures",
        "stage",
        "status",
        "timestamp",
    }
    assert "signature_pattern" in requirements


def test_describe_schema_alias_unknown(caplog):
    caplog.set_level(logging.ERROR)
    exit_code = verify_main(["--describe-schema-alias", "builtin:unknown"])
    assert exit_code == 2
    assert any("nieznany" in message for message in caplog.messages)


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


def test_verify_schema_success(tmp_path):
    schema_path = Path(__file__).resolve().parents[1] / "docs/schemas/decision_log_v2.json"
    log_path = tmp_path / "decision_schema.jsonl"
    key = b"schema-key"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2025-01-01T12:00:00Z",
            "stage": "paper",
            "status": "paper_ready",
            "artefacts": {
                "config_hash": "sha384:deadbeef",
                "paper_labs_report": "reports/paper_labs/2025-01-01.pdf"
            },
            "runtime_flags": {
                "StrategyContext.require_demo_mode": False,
                "runtime.compliance_confirmed": False
            },
            "signatures": {"owner": "hmac:deadbeef"},
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--schema",
            str(schema_path),
        ]
    )

    assert exit_code == 0


def test_verify_schema_failure(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    schema_path = Path(__file__).resolve().parents[1] / "docs/schemas/decision_log_v2.json"
    log_path = tmp_path / "decision_schema_invalid.jsonl"
    key = b"schema-bad"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2025-01-01T12:00:00Z",
            "stage": "demo",
            "status": "demo_ready",
            "artefacts": {"config_hash": "sha384:bead"},
            "runtime_flags": {
                "StrategyContext.require_demo_mode": "yes"
            },
            "signatures": {"owner": "hmac:bad"},
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--schema",
            str(schema_path),
        ]
    )

    assert exit_code == 2
    assert any("schematu JSON" in message for message in caplog.messages)
    assert any(
        "ścieżka: $.runtime_flags['StrategyContext.require_demo_mode']" in message
        for message in caplog.messages
    )


def test_verify_schema_success_without_jsonschema(tmp_path, monkeypatch):
    module = _load_verify_module_without_jsonschema(monkeypatch)
    assert module.jsonschema is None

    schema_path = Path(__file__).resolve().parents[1] / "docs/schemas/decision_log_v2.json"
    log_path = tmp_path / "decision_schema_builtin.jsonl"
    key = b"builtin"
    key_id = "builtin-key"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-03-01T00:00:00Z",
            "stage": "paper",
            "status": "paper_ready",
            "artefacts": {
                "config_hash": "sha384:abc123",
                "report": "reports/paper_labs/latest.pdf",
                "run_id": 42,
                "flag": False,
                "optional": None,
            },
            "runtime_flags": {
                "StrategyContext.require_demo_mode": True,
                "runtime.compliance_confirmed": False,
            },
            "signatures": {"ops": "hmac:ABCdef123_-"},
        },
        key=key,
        key_id=key_id,
    )
    log_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code = module.main(
        [
            str(log_path),
            "--schema",
            str(schema_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
        ]
    )

    assert exit_code == 0


def test_verify_schema_failure_without_jsonschema(tmp_path, caplog, monkeypatch):
    module = _load_verify_module_without_jsonschema(monkeypatch)
    assert module.jsonschema is None

    caplog.set_level(logging.ERROR)
    schema_path = Path(__file__).resolve().parents[1] / "docs/schemas/decision_log_v2.json"
    log_path = tmp_path / "decision_schema_builtin_invalid.jsonl"
    key = b"builtin-bad"
    key_id = "builtin-key"
    invalid_entry = {
        "kind": "snapshot",
        "timestamp": "2024-03-01T00:00:00Z",
        "stage": "paper",
        "status": "paper_ready",
        "artefacts": {
            "config_hash": "sha384:def456",
            "report": "reports/paper_labs/latest.pdf",
        },
        "runtime_flags": {
            "runtime.compliance_confirmed": "false",
        },
        "signatures": {"ops": "hmac:XYZabc789"},
    }
    log_path.write_text(
        json.dumps(_signed_entry(invalid_entry, key=key, key_id=key_id), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            str(log_path),
            "--schema",
            str(schema_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
        ]
    )

    assert exit_code == 2
    assert any("runtime_flags" in message and "schematu JSON" in message for message in caplog.messages)


def test_verify_schema_builtin_alias_success(tmp_path):
    log_path = tmp_path / "decision_schema_builtin_alias.jsonl"
    key = b"alias"
    key_id = "alias-key"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2025-05-01T10:00:00Z",
            "stage": "demo",
            "status": "demo_ready",
            "artefacts": {
                "config_hash": "sha384:0123456789abcdef",
                "report": "reports/demo/latest.pdf",
            },
            "runtime_flags": {
                "StrategyContext.require_demo_mode": True,
            },
            "signatures": {"ops": "hmac:XYZabc012_-"},
        },
        key=key,
        key_id=key_id,
    )
    log_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--schema",
            "builtin:decision_log_v2",
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
        ]
    )

    assert exit_code == 0


def test_verify_schema_builtin_alias_success_without_jsonschema(tmp_path, monkeypatch):
    module = _load_verify_module_without_jsonschema(monkeypatch)
    assert module.jsonschema is None

    log_path = tmp_path / "decision_schema_builtin_alias_fallback.jsonl"
    key = b"alias-fallback"
    key_id = "alias-fallback-key"
    entry = _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2025-05-02T10:00:00Z",
            "stage": "paper",
            "status": "paper_ready",
            "artefacts": {
                "config_hash": "sha384:fallback",
            },
            "runtime_flags": {
                "runtime.compliance_confirmed": False,
            },
            "signatures": {"ops": "hmac:XYZabc987_-"},
        },
        key=key,
        key_id=key_id,
    )
    log_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    exit_code = module.main(
        [
            str(log_path),
            "--schema",
            "builtin:decision_log_v2",
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
        ]
    )

    assert exit_code == 0


def test_verify_schema_builtin_alias_failure_without_jsonschema(tmp_path, caplog, monkeypatch):
    module = _load_verify_module_without_jsonschema(monkeypatch)
    assert module.jsonschema is None

    caplog.set_level(logging.ERROR)
    log_path = tmp_path / "decision_schema_builtin_alias_invalid.jsonl"
    key = b"alias-bad"
    key_id = "alias-bad-key"
    invalid_entry = {
        "kind": "snapshot",
        "timestamp": "2025-05-03T10:00:00Z",
        "stage": "demo",
        "status": "demo_ready",
        "artefacts": {
            "config_hash": "sha384:badalias",
        },
        "signatures": {"ops": "hmac:XYZabc654_-"},
    }
    log_path.write_text(
        json.dumps(_signed_entry(invalid_entry, key=key, key_id=key_id), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            str(log_path),
            "--schema",
            "builtin:decision_log_v2",
            "--hmac-key",
            key.decode("utf-8"),
            "--hmac-key-id",
            key_id,
        ]
    )

    assert exit_code == 2
    assert any("runtime_flags" in message for message in caplog.messages)


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


def test_verify_snippet_validation_success(tmp_path):
    key = b"sig"
    log_path = tmp_path / "decision.jsonl"
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.json"
    env_snippet = tmp_path / "balanced.env"
    yaml_snippet = tmp_path / "balanced.json"

    entries = [
        _signed_entry(
            {
                "kind": "metadata",
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {"mode": "jsonl", "summary_enabled": True},
            },
            key=key,
            key_id=None,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2025-01-01T00:00:01Z",
                "source": "jsonl",
                "event": "reduce_motion",
                "severity": "notice",
                "notes": {"event": "reduce_motion", "severity": "notice"},
                "screen": {
                    "index": 0,
                    "name": "primary",
                    "refresh_hz": 120,
                },
            },
            key=key,
            key_id=None,
        ),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    recommended_overrides = {
        "ui_alerts_reduce_mode": "enable",
        "ui_alerts_overlay_mode": "enable",
        "ui_alerts_jank_mode": "enable",
        "ui_alerts_reduce_active_severity": "notice",
        "ui_alerts_overlay_exceeded_severity": "notice",
        "ui_alerts_jank_spike_severity": "notice",
        "ui_alerts_overlay_critical_threshold": 2,
    }
    risk_profile_summary = {
        "name": "balanced",
        "severity_min": "notice",
        "limits": {
            "overlay_budget": 2,
            "jank": 1,
            "reduce_motion": 5,
        },
        "requirements": {
            "expect_summary_enabled": True,
            "require_screen_info": True,
        },
        "recommended_overrides": recommended_overrides,
    }

    summary_timestamp = "2025-01-01T00:00:01+00:00"
    summary_payload = {
        "summary": {
            "total_snapshots": 1,
            "first_timestamp": summary_timestamp,
            "last_timestamp": summary_timestamp,
            "events": {
                "reduce_motion": {
                    "count": 1,
                    "first_timestamp": summary_timestamp,
                    "last_timestamp": summary_timestamp,
                    "screens": [
                        {
                            "index": 0,
                            "name": "primary",
                            "refresh_hz": 120,
                        }
                    ],
                    "severity": {"counts": {"notice": 1}},
                }
            },
            "severity_counts": {"notice": 1},
        },
        "metadata": {
            "mode": "jsonl",
            "summary_enabled": True,
            "risk_profile_summary": risk_profile_summary,
        },
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False), encoding="utf-8")

    env_lines = [
        f"RUN_TRADING_STUB_METRICS_{key.upper()}={_format_env(value)}"
        for key, value in recommended_overrides.items()
    ]
    env_snippet.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    yaml_snippet.write_text(
        json.dumps({"metrics_service_overrides": recommended_overrides}, ensure_ascii=False),
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--summary-json",
            str(summary_path),
            "--report-output",
            str(report_path),
            "--risk-profile",
            "balanced",
            "--risk-profile-env-snippet",
            str(env_snippet),
            "--risk-profile-yaml-snippet",
            str(yaml_snippet),
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    validations = payload.get("risk_profile_snippet_validation")
    assert validations, "Raport powinien zawierać sekcję walidacji snippetów"
    statuses = {entry["type"]: entry["status"] for entry in validations}
    assert statuses.get("env") == "ok"
    assert statuses.get("yaml") == "ok"


def test_verify_snippet_validation_mismatch(tmp_path):
    key = b"sig"
    log_path = tmp_path / "decision.jsonl"
    summary_path = tmp_path / "summary.json"
    env_snippet = tmp_path / "balanced.env"

    entries = [
        _signed_entry(
            {
                "kind": "metadata",
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {"mode": "jsonl", "summary_enabled": True},
            },
            key=key,
            key_id=None,
        ),
        _signed_entry(
            {
                "kind": "snapshot",
                "timestamp": "2025-01-01T00:00:01Z",
                "source": "jsonl",
                "event": "reduce_motion",
                "severity": "notice",
                "notes": {"event": "reduce_motion", "severity": "notice"},
            },
            key=key,
            key_id=None,
        ),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary_payload = {
        "summary": {
            "total_snapshots": 1,
            "events": {"reduce_motion": {"count": 1}},
        },
        "metadata": {
            "mode": "jsonl",
            "summary_enabled": True,
            "risk_profile_summary": {
                "name": "balanced",
                "recommended_overrides": {
                    "ui_alerts_reduce_mode": "enable",
                },
            },
        },
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False), encoding="utf-8")

    env_snippet.write_text(
        "RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_MODE=disable\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--summary-json",
            str(summary_path),
            "--risk-profile",
            "balanced",
            "--risk-profile-env-snippet",
            str(env_snippet),
        ]
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
                "input_files": [str(log_path)],
                "input_sources": [str(log_path)],
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
                "auth_token_scope_checked": True,
                "auth_token_scope_match": True,
                "auth_token_scopes": ["metrics.read"],
                "auth_token_scope_required": "metrics.read",
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


def test_verify_metadata_require_auth_scope(tmp_path):
    key = b"secret"
    log_path = tmp_path / "auth_scope.jsonl"
    metadata_entry = _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-05-02T00:00:00+00:00",
            "metadata": {
                "mode": "grpc",
                "endpoint": "localhost:50051",
                "auth_token_provided": True,
                "auth_token_scope_checked": True,
                "auth_token_scope_match": True,
                "auth_token_scopes": ["metrics.read"],
            },
        },
        key=key,
        key_id=None,
    )
    log_path.write_text(json.dumps(metadata_entry) + "\n", encoding="utf-8")

    ok_exit = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-auth-token",
            "--require-auth-scope",
            "metrics.read",
        ]
    )
    assert ok_exit == 0

    fail_exit = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-auth-token",
            "--require-auth-scope",
            "metrics.write",
        ]
    )
    assert fail_exit == 2


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


def _tls_metadata_entry(*, key: bytes, fingerprint: str, materials: dict[str, bool], source: str | None = None):
    metadata_payload = {
        "kind": "metadata",
        "timestamp": "2024-03-01T00:00:00+00:00",
        "metadata": {
            "mode": "grpc",
            "use_tls": True,
            "tls_materials": materials,
            "server_sha256": fingerprint,
        },
    }
    if source is not None:
        metadata_payload["metadata"]["server_sha256_source"] = source
    return _signed_entry(metadata_payload, key=key, key_id=None)


def _tls_snapshot_entry(*, key: bytes) -> dict[str, object]:
    return _signed_entry(
        {
            "kind": "snapshot",
            "timestamp": "2024-03-01T00:00:01+00:00",
            "source": "grpc",
            "event": "reduce_motion",
            "severity": "info",
            "screen": {"index": 0, "name": "Ops"},
            "notes": {"event": "reduce_motion", "severity": "info"},
        },
        key=key,
        key_id=None,
    )


def _risk_tls_metadata_entry(
    *,
    key: bytes,
    fingerprint: str | None,
    materials: Mapping[str, bool],
    risk_overrides: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    metadata: dict[str, Any] = {
        "kind": "metadata",
        "timestamp": "2024-03-01T00:00:00+00:00",
        "metadata": {
            "mode": "grpc",
            "use_tls": True,
            "tls_materials": dict(materials),
        },
    }
    if fingerprint:
        metadata["metadata"]["server_sha256"] = fingerprint
    risk_meta: dict[str, Any] = {
        "tls_enabled": True,
        "root_cert_configured": bool(materials.get("root_cert")),
        "client_cert_configured": bool(materials.get("client_cert")),
        "client_key_configured": bool(materials.get("client_key")),
        "client_auth": bool(materials.get("client_auth")),
        "auth_token_scope_required": "risk.read",
        "required_scopes": {"risk.read": ["summary"]},
        "auth_token_scope_checked": True,
        "auth_token_scope_match": True,
        "auth_token_scopes": ["risk.read"],
        "auth_token_token_id": "risk-reader",
    }
    if fingerprint:
        risk_meta["pinned_fingerprints"] = [fingerprint]
    if risk_overrides:
        risk_meta.update(risk_overrides)
    metadata["metadata"]["risk_service"] = risk_meta
    return _signed_entry(metadata, key=key, key_id=None)


def test_verify_tls_requirements_success(tmp_path):
    key = b"tls"
    log_path = tmp_path / "decision.jsonl"
    fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    entries = [
        _tls_metadata_entry(
            key=key,
            fingerprint=fingerprint,
            materials={"root_cert": True, "server_sha256": True},
            source="pinned_fingerprint",
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-tls-material",
            "root_cert",
            "--require-tls-material",
            "server_sha256",
            "--expect-server-sha256",
            fingerprint,
            "--expect-server-sha256-source",
            "pinned_fingerprint",
        ]
    )

    assert exit_code == 0


def test_verify_tls_flag_requires_fingerprint(tmp_path):
    key = b"tls"
    log_path = tmp_path / "decision.jsonl"
    metadata_entry = _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-03-01T00:00:00+00:00",
            "metadata": {
                "mode": "grpc",
                "use_tls": True,
                "tls_materials": {"server_sha256": True},
            },
        },
        key=key,
        key_id=None,
    )
    entries = [metadata_entry, _tls_snapshot_entry(key=key)]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8")])

    assert exit_code == 2


def test_verify_tls_fingerprint_requires_flag(tmp_path):
    key = b"tls"
    log_path = tmp_path / "decision.jsonl"
    fingerprint = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    metadata_entry = _signed_entry(
        {
            "kind": "metadata",
            "timestamp": "2024-03-01T00:00:00+00:00",
            "metadata": {
                "mode": "grpc",
                "use_tls": True,
                "tls_materials": {"server_sha256": False},
                "server_sha256": fingerprint,
            },
        },
        key=key,
        key_id=None,
    )
    entries = [metadata_entry, _tls_snapshot_entry(key=key)]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8")])

    assert exit_code == 2


def test_verify_tls_requirements_missing_material(tmp_path):
    key = b"tls"
    log_path = tmp_path / "decision.jsonl"
    fingerprint = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    entries = [
        _tls_metadata_entry(
            key=key,
            fingerprint=fingerprint,
            materials={"root_cert": False, "server_sha256": True},
            source="cli",
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-tls-material",
            "root_cert",
            "--require-tls-material",
            "server_sha256",
        ]
    )

    assert exit_code == 2


def test_verify_tls_requirements_env_overrides(monkeypatch, tmp_path):
    key = b"tls"
    log_path = tmp_path / "decision.jsonl"
    fingerprint = "aa11bb22cc33dd44ee55ff6677889900aa11bb22cc33dd44ee55ff6677889900"
    fingerprint_env = ":".join(fingerprint[i : i + 2] for i in range(0, len(fingerprint), 2))
    entries = [
        _tls_metadata_entry(
            key=key,
            fingerprint=fingerprint,
            materials={"root_cert": True, "server_sha256": True, "server_name": True},
            source="env",
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n", encoding="utf-8")

    monkeypatch.setenv(
        "BOT_CORE_VERIFY_DECISION_LOG_REQUIRE_TLS_MATERIALS",
        "[\"root_cert\", \"server_sha256\"]",
    )
    monkeypatch.setenv(
        "BOT_CORE_VERIFY_DECISION_LOG_EXPECT_SERVER_SHA256",
        fingerprint_env,
    )
    monkeypatch.setenv(
        "BOT_CORE_VERIFY_DECISION_LOG_EXPECT_SERVER_SHA256_SOURCE",
        "env",
    )

    exit_code = verify_main([str(log_path), "--hmac-key", key.decode("utf-8")])

    assert exit_code == 0


def test_verify_risk_service_tls_requirements(tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk.jsonl"
    fingerprint = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    materials = {
        "root_cert": True,
        "client_cert": True,
        "client_key": True,
        "client_auth": True,
        "server_sha256": True,
    }
    entries = [
        _risk_tls_metadata_entry(
            key=key,
            fingerprint=fingerprint,
            materials=materials,
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-risk-service-tls",
            "--require-risk-service-tls-material",
            "root_cert",
            "--require-risk-service-tls-material",
            "client_cert",
            "--require-risk-service-tls-material",
            "client_key",
            "--require-risk-service-tls-material",
            "client_auth",
            "--expect-risk-service-server-sha256",
            fingerprint,
            "--require-risk-service-scope",
            "risk.read",
            "--require-risk-service-auth-token",
        ]
    )

    assert exit_code == 0


def test_verify_risk_service_tls_missing_material(tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk.jsonl"
    fingerprint = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    materials = {
        "root_cert": False,
        "client_cert": True,
        "client_key": True,
        "client_auth": True,
        "server_sha256": True,
    }
    entries = [
        _risk_tls_metadata_entry(
            key=key,
            fingerprint=fingerprint,
            materials=materials,
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-risk-service-tls",
            "--require-risk-service-tls-material",
            "root_cert",
        ]
    )

    assert exit_code == 2


def test_verify_risk_service_scope_required(tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk.jsonl"
    entries = [
        _risk_tls_metadata_entry(
            key=key,
            fingerprint=None,
            materials={"root_cert": True, "server_sha256": False},
            risk_overrides={
                "auth_token_scope_required": None,
                "required_scopes": {},
                "auth_token_scopes": [],
            },
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-risk-service-scope",
            "risk.read",
        ]
    )

    assert exit_code == 2


def test_verify_risk_service_auth_token_required(tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk.jsonl"
    entries = [
        _risk_tls_metadata_entry(
            key=key,
            fingerprint=None,
            materials={"root_cert": True, "server_sha256": False},
            risk_overrides={"auth_token_scope_checked": False},
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-risk-service-auth-token",
        ]
    )

    assert exit_code == 2


def test_verify_risk_service_token_id_required(tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk.jsonl"
    entries = [
        _risk_tls_metadata_entry(
            key=key,
            fingerprint=None,
            materials={"root_cert": True, "server_sha256": False},
            risk_overrides={"auth_token_token_id": "risk-reader"},
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-risk-service-token-id",
            "risk-reader",
        ]
    )

    assert exit_code == 0


def test_verify_risk_service_token_id_missing(tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk.jsonl"
    entries = [
        _risk_tls_metadata_entry(
            key=key,
            fingerprint=None,
            materials={"root_cert": True, "server_sha256": False},
            risk_overrides={"auth_token_token_id": "other-token"},
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
            "--require-risk-service-token-id",
            "risk-reader",
        ]
    )

    assert exit_code == 2


def test_verify_risk_service_token_id_env(monkeypatch, tmp_path):
    key = b"risk"
    log_path = tmp_path / "risk.jsonl"
    entries = [
        _risk_tls_metadata_entry(
            key=key,
            fingerprint=None,
            materials={"root_cert": True, "server_sha256": False},
            risk_overrides={"auth_token_token_id": "risk-reader"},
        ),
        _tls_snapshot_entry(key=key),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "BOT_CORE_VERIFY_DECISION_LOG_REQUIRE_RISK_SERVICE_TOKEN_ID",
        "risk-reader",
    )

    exit_code = verify_main(
        [
            str(log_path),
            "--hmac-key",
            key.decode("utf-8"),
        ]
    )

    assert exit_code == 0
