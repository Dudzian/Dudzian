from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol, Sequence

import pytest
import yaml

from bot_core.exchanges.base import (
    AccountSnapshot,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_RUNTIME_SCRIPT = REPO_ROOT / "scripts/mock_runtime_preview.py"
CONTROLLER_MOCK_SCRIPT = REPO_ROOT / "scripts/controller_mock_preview.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"
PREVIEW_SCRIPTS = (MOCK_RUNTIME_SCRIPT, CONTROLLER_MOCK_SCRIPT)

FORBIDDEN_PREVIEW_SOURCE_TOKENS = (
    "ccxt",
    "create_order",
    "fetch_balance",
    "load_markets",
    "os.environ",
    "getenv",
    "keyring",
    "get_secret",
    "subprocess",
    "shell=True",
    "stream_private_data",
    "stream_public_data",
)

LIVE_IO_METHODS = (
    "configure_network",
    "fetch_account_snapshot",
    "fetch_symbols",
    "fetch_ohlcv",
    "place_order",
    "cancel_order",
    "stream_public_data",
    "stream_private_data",
)


class CanaryExchangeAdapter(ExchangeAdapter):
    """Future-DI canary: fails immediately if a test injects it into live I/O."""

    name = "canary-live-exchange"
    calls: list[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    @classmethod
    def _fail(cls, method_name: str) -> None:
        cls.calls.append(method_name)
        raise AssertionError(f"preview touched live exchange I/O: {method_name}")

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:
        del ip_allowlist
        self._fail("configure_network")

    def fetch_account_snapshot(self) -> AccountSnapshot:
        self._fail("fetch_account_snapshot")

    def fetch_symbols(self) -> Sequence[str]:
        self._fail("fetch_symbols")

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        del symbol, interval, start, end, limit
        self._fail("fetch_ohlcv")

    def place_order(self, request: OrderRequest) -> OrderResult:
        del request
        self._fail("place_order")

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        del order_id, symbol
        self._fail("cancel_order")

    def stream_public_data(self, *, channels: Sequence[str]) -> Protocol:
        del channels
        self._fail("stream_public_data")

    def stream_private_data(self, *, channels: Sequence[str]) -> Protocol:
        del channels
        self._fail("stream_private_data")


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _payload_from(script: Path, *args: str) -> tuple[int, dict[str, object]]:
    result = _run(script, *args)
    assert result.stderr == ""
    return result.returncode, json.loads(result.stdout)


def _mutated_config(tmp_path: Path, dotted_path: str, value: object) -> Path:
    payload = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    current = payload
    segments = dotted_path.split(".")
    for segment in segments[:-1]:
        current = current[segment]
    current[segments[-1]] = value
    path = tmp_path / f"unsafe_{script_safe_name(dotted_path)}.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def script_safe_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


def _assert_safe_preview_payload(payload: dict[str, object]) -> None:
    assert payload["status"] == "ok"
    assert payload.get("exchange_io_disabled") is True or payload["exchange_io"] == "disabled"
    assert payload.get("real_orders_submitted", False) is False
    assert payload["api_keys_required"] is False
    assert payload.get("runtime_loop_started", False) is False
    assert payload["live_mode_allowed"] is False
    assert payload["order_execution"] in {"disabled", "mocked_or_disabled", "mocked"}
    assert payload["issues"] == []

    invariants = payload.get("safety_invariants")
    if isinstance(invariants, dict):
        assert invariants["exchange_io_disabled"] is True
        assert invariants["real_orders_submitted"] is False
        assert invariants["api_keys_required"] is False
        assert invariants["runtime_loop_started"] is False


def test_preview_scripts_forbid_live_route_and_secret_source_tokens() -> None:
    for script in PREVIEW_SCRIPTS:
        source = script.read_text(encoding="utf-8")
        for token in FORBIDDEN_PREVIEW_SOURCE_TOKENS:
            assert token not in source, (
                f"{token!r} must stay out of {script.relative_to(REPO_ROOT)}"
            )


@pytest.mark.parametrize("script", PREVIEW_SCRIPTS)
def test_live_mode_is_process_wide_blocked_before_preview_starts(script: Path) -> None:
    returncode, payload = _payload_from(
        script,
        "--mode",
        "live",
        "--config",
        str(SAFE_CONFIG),
        "--json",
    )
    assert returncode == 2
    assert payload["status"] == "blocked"
    assert payload["live_mode_allowed"] is False
    assert payload["api_keys_required"] is False
    assert payload.get("runtime_loop_started", False) is False
    assert payload["issues"] == ["live_mode_not_allowed"]


@pytest.mark.parametrize("script", PREVIEW_SCRIPTS)
def test_unsafe_live_config_flags_block_preview_process(script: Path, tmp_path: Path) -> None:
    unsafe_cases = (
        ("execution.live.enabled", True),
        ("trading.enable_live_mode", True),
        ("execution.default_mode", "live"),
        ("trading.enable_paper_mode", False),
        ("execution.force_paper_when_offline", False),
    )
    for dotted_path, value in unsafe_cases:
        cfg = _mutated_config(tmp_path, dotted_path, value)
        args = ["--mode", "demo", "--config", str(cfg), "--json"]
        if script == MOCK_RUNTIME_SCRIPT:
            args.extend(["--duration-seconds", "1"])
        else:
            args.extend(["--max-signals", "1"])
        returncode, payload = _payload_from(script, *args)
        assert returncode == 2
        assert payload["status"] == "blocked"
        assert f"unsafe_config:{dotted_path}" in payload["issues"]
        assert payload["live_mode_allowed"] is False
        assert payload["api_keys_required"] is False
        assert payload.get("runtime_loop_started", False) is False


@pytest.mark.parametrize("script", PREVIEW_SCRIPTS)
def test_safe_preview_payload_contract_does_not_require_keys_or_live_loop(
    script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)

    args = ["--mode", "demo", "--config", str(SAFE_CONFIG), "--json"]
    if script == MOCK_RUNTIME_SCRIPT:
        args.extend(["--duration-seconds", "1"])
    else:
        args.extend(["--max-signals", "1"])

    returncode, payload = _payload_from(script, *args)
    assert returncode == 0
    _assert_safe_preview_payload(payload)


def test_source_and_payload_preview_hard_gate_does_not_reference_live_exchange_io() -> None:
    # This is not canary injection: preview scripts run in subprocesses and do not
    # receive this pytest-process canary. The proof here is source + payload only.
    CanaryExchangeAdapter.reset()

    for script in PREVIEW_SCRIPTS:
        args = ["--mode", "demo", "--config", str(SAFE_CONFIG), "--json"]
        if script == MOCK_RUNTIME_SCRIPT:
            args.extend(["--duration-seconds", "1"])
        else:
            args.extend(["--max-signals", "1"])
        returncode, payload = _payload_from(script, *args)
        assert returncode == 0
        _assert_safe_preview_payload(payload)

    assert CanaryExchangeAdapter.calls == []


def test_armed_canary_contract_is_available_for_future_di() -> None:
    for method_name in LIVE_IO_METHODS:
        CanaryExchangeAdapter.reset()
        canary = CanaryExchangeAdapter(ExchangeCredentials(key_id="preview-canary"))
        with pytest.raises(AssertionError, match=method_name):
            if method_name == "configure_network":
                canary.configure_network()
            elif method_name == "fetch_account_snapshot":
                canary.fetch_account_snapshot()
            elif method_name == "fetch_symbols":
                canary.fetch_symbols()
            elif method_name == "fetch_ohlcv":
                canary.fetch_ohlcv("BTC/USDT", "1m")
            elif method_name == "place_order":
                canary.place_order(
                    OrderRequest(symbol="BTC/USDT", side="BUY", quantity=1, order_type="market")
                )
            elif method_name == "cancel_order":
                canary.cancel_order("order-1")
            elif method_name == "stream_public_data":
                canary.stream_public_data(channels=["ticker"])
            elif method_name == "stream_private_data":
                canary.stream_private_data(channels=["orders"])
        assert CanaryExchangeAdapter.calls == [method_name]
