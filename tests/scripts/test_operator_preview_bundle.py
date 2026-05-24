from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/operator_preview_bundle.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_operator_preview_bundle_safe_demo_json() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "5",
        "--max-signals",
        "1",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["mode"] == "demo"
    assert len(payload["steps"]) == 7
    assert payload["summary"]["steps_total"] == 7
    assert payload["summary"]["steps_passed"] == 7
    assert payload["summary"]["real_orders_submitted"] is False
    assert payload["summary"]["exchange_io"] == "disabled"
    assert payload["summary"]["api_keys_required"] is False
    assert payload["summary"]["runtime_loop_started"] is False
    assert payload["summary"]["controller_backed_preview"] is True
    assert payload["issues"] == []
    adapter_step = payload["steps"][1]
    assert adapter_step["name"] == "paper_adapter_readiness"
    assert adapter_step["status"] == "ok"
    assert adapter_step["payload"]["adapter_readiness"]["enabled"] is True
    readiness_step = payload["steps"][2]
    assert readiness_step["name"] == "sandbox_testnet_readiness"
    assert readiness_step["status"] == "ok"
    assert readiness_step["payload"]["status"] == "ok"
    readiness = readiness_step["payload"]["sandbox_testnet_readiness"]
    assert readiness["static_only"] is True
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["secrets_read"] is False
    assert readiness["api_keys_required"] is False
    assert readiness["runtime_loop_started"] is False
    assert readiness["live_mode_allowed"] is False
    credential_step = payload["steps"][3]
    assert credential_step["name"] == "credential_reference_readiness"
    assert credential_step["status"] == "ok"
    assert credential_step["payload"]["status"] == "ok"
    credential_readiness = credential_step["payload"]["credential_reference_readiness"]
    assert credential_readiness["static_only"] is True
    assert credential_readiness["secrets_read"] is False
    assert credential_readiness["keychain_read"] is False
    assert credential_readiness["env_values_read"] is False
    assert credential_readiness["credential_values_read"] is False
    assert credential_readiness["api_keys_required"] is False
    assert credential_readiness["exchange_io"] == "disabled"
    assert credential_readiness["order_submission"] == "disabled"
    assert credential_readiness["runtime_loop_started"] is False
    assert credential_readiness["live_mode_allowed"] is False
    assert payload["safety_contract_version"] == "operator_preview_bundle.v1"


def test_operator_preview_bundle_live_mode_blocked() -> None:
    result = _run("--mode", "live", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "operator_preview_bundle_forbids_live_mode"
    assert payload["steps"] == []


def test_operator_preview_bundle_unsafe_config_stops_at_precheck(tmp_path: Path) -> None:
    unsafe_config = tmp_path / "unsafe_demo_paper.yml"
    config_payload = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    config_payload.setdefault("execution", {}).setdefault("live", {})["enabled"] = True
    unsafe_config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    result = _run("--mode", "demo", "--config", str(unsafe_config), "--json")
    assert result.returncode in {1, 2}

    payload = json.loads(result.stdout)
    assert payload["status"] in {"blocked", "error"}
    assert payload["failed_step"] == "demo_paper_precheck"
    assert len(payload["steps"]) == 1
    assert payload["steps"][0]["name"] == "demo_paper_precheck"
    assert payload["summary"]["steps_passed"] == 0
    assert any("step_failed:demo_paper_precheck" in issue for issue in payload["issues"])


def test_operator_preview_bundle_child_payload_sanity() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--max-signals", "1", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    names = [step["name"] for step in payload["steps"]]
    assert names == [
        "demo_paper_precheck",
        "paper_adapter_readiness",
        "sandbox_testnet_readiness",
        "credential_reference_readiness",
        "preview_plan",
        "mock_runtime_preview",
        "controller_mock_preview",
    ]
    controller_payload = payload["steps"][-1]["payload"]
    assert controller_payload["real_orders_submitted"] is False
    assert controller_payload["runtime_loop_started"] is False
    assert controller_payload["api_keys_required"] is False


def test_operator_preview_bundle_controller_blocked_stops_chain() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "5",
        "--max-signals",
        "999",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)

    assert payload["failed_step"] == "controller_mock_preview"
    assert len(payload["steps"]) == 7
    assert payload["summary"]["steps_passed"] == 6
    assert any("step_failed:controller_mock_preview" in issue for issue in payload["issues"])
    assert [step["name"] for step in payload["steps"][:6]] == [
        "demo_paper_precheck",
        "paper_adapter_readiness",
        "sandbox_testnet_readiness",
        "credential_reference_readiness",
        "preview_plan",
        "mock_runtime_preview",
    ]
    assert all(step["exit_code"] == 0 for step in payload["steps"][:6])
    assert payload["steps"][6]["name"] == "controller_mock_preview"
    assert payload["steps"][6]["exit_code"] == 2


def test_operator_preview_bundle_mock_duration_3600_allowed() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "3600",
        "--max-signals",
        "1",
        "--json",
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)

    assert payload["status"] == "ok"
    assert [step["name"] for step in payload["steps"][:6]] == [
        "demo_paper_precheck",
        "paper_adapter_readiness",
        "sandbox_testnet_readiness",
        "credential_reference_readiness",
        "preview_plan",
        "mock_runtime_preview",
    ]
    assert all(step["exit_code"] == 0 for step in payload["steps"][:6])
    assert payload["steps"][6]["name"] == "controller_mock_preview"
    assert payload["steps"][6]["exit_code"] == 0
    assert payload["summary"]["steps_passed"] == 7
    child_payload = payload["steps"][5]["payload"]
    assert child_payload["status"] == "ok"
    assert child_payload["bounded_duration_seconds"] == 3600


def test_operator_preview_bundle_inline_secret_blocked_at_credential_step(tmp_path: Path) -> None:
    inline_config = tmp_path / "inline_secret_demo_paper.yml"
    config_payload = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    config_payload.setdefault("credentials", {})["api_key"] = "REAL_VALUE_SHOULD_NOT_BE_INLINE"
    inline_config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    result = _run("--mode", "demo", "--config", str(inline_config), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)

    assert payload["status"] == "blocked"
    assert payload["failed_step"] == "credential_reference_readiness"
    assert len(payload["steps"]) == 4
    assert [step["name"] for step in payload["steps"][:3]] == [
        "demo_paper_precheck",
        "paper_adapter_readiness",
        "sandbox_testnet_readiness",
    ]
    assert all(step["exit_code"] == 0 for step in payload["steps"][:3])
    assert payload["steps"][3]["name"] == "credential_reference_readiness"
    assert payload["steps"][3]["exit_code"] == 2
    assert payload["summary"]["steps_passed"] == 3
    assert any("step_failed:credential_reference_readiness" in issue for issue in payload["issues"])


def test_operator_preview_bundle_no_api_keys_required(monkeypatch) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr


def test_operator_preview_bundle_output_cp1252_safe() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")
