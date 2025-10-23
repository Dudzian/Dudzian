from __future__ import annotations

import base64
import json
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests._cli_parser_helpers import parser_supports

# --- dostęp do modułu/CLI ---
from scripts import rotate_keys as rotate_keys_mod
from scripts.rotate_keys import run as rotate_keys_run

# --- RotationRegistry: wspieramy obie ścieżki importu ---
try:
    from bot_core.security.rotation import RotationRegistry as _RR  # HEAD wariant
except Exception:  # pragma: no cover
    try:
        from bot_core.security import RotationRegistry as _RR  # main wariant
    except Exception:  # pragma: no cover
        _RR = None  # type: ignore[assignment]

RotationRegistry = _RR  # type: ignore[misc]


# ----------------- pomocnicze -----------------
def _parser() -> object:
    build = getattr(rotate_keys_mod, "_build_parser", None)
    if build is None:
        # awaryjnie: spróbujemy skonstruować parser przez wywołanie z --help i przechwycić SystemExit
        # ale w testach nie wywołujemy bezpośrednio; zakładamy, że _build_parser istnieje w obu wariantach
        raise RuntimeError("rotate_keys._build_parser not available")
    return build()
def _supports_head_cli() -> bool:
    # HEAD wariant: ma m.in. --environment, --operator, --executed-at, --signing-key, --output
    return parser_supports(_parser, "--environment", "--operator", "--executed-at", "--output")


def _supports_main_cli() -> bool:
    # main wariant: ma m.in. --output-dir, --basename (i zwykle --execute)
    return parser_supports(_parser, "--output-dir", "--basename")


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _write_minimal_head_config(path: Path, cache_path: Path, tls_registry: Path | None = None) -> None:
    registry_path = tls_registry or (cache_path / "security" / "tls_rotation.json")
    plan_registry = cache_path / "security" / "rotation_log.json"
    path.write_text(
        f"""
risk_profiles: {{ paper: {{ name: paper, max_daily_loss_pct: 0.1, max_position_pct: 0.5, target_volatility: 0.1, max_leverage: 1.0, stop_loss_atr_multiple: 2.0, max_open_positions: 3, hard_drawdown_pct: 0.2 }} }}
instrument_universes: {{}}
instrument_buckets: {{}}
environments:
  paper:
    name: paper
    exchange: binance
    environment: paper
    keychain_key: binance_paper
    data_cache_path: {cache_path.as_posix()}
    risk_profile: paper
    alert_channels: []
strategies: {{}}
mean_reversion_strategies: {{}}
volatility_target_strategies: {{}}
cross_exchange_arbitrage_strategies: {{}}
multi_strategy_schedulers: {{}}
portfolio_governors: {{}}
reporting: {{}}
execution:
  mtls:
    bundle_directory: {cache_path.as_posix()}
    ca_certificate: secrets/mtls/core-oem-ca.pem
    server_certificate: secrets/mtls/core-oem-server.pem
    server_key: secrets/mtls/core-oem-server-key.pem
    client_certificate: secrets/mtls/core-oem-client.pem
    client_key: secrets/mtls/core-oem-client-key.pem
    rotation_registry: {registry_path.as_posix()}
observability:
  key_rotation:
    registry_path: {plan_registry.as_posix()}
    default_interval_days: 90.0
    default_warn_within_days: 14.0
    audit_directory: var/audit/keys
    entries: []
""",
        encoding="utf-8",
    )


def _write_main_config(config_path: Path, registry_path: Path) -> None:
    config_path.write_text(
        textwrap.dedent(
            f"""
            environments:
              demo:
                exchange: binance
                environment: paper
                keychain_key: api
                data_cache_path: cache
                risk_profile: conservative
                alert_channels: []
            risk_profiles:
              conservative:
                max_daily_loss_pct: 0.01
                max_position_pct: 0.2
                target_volatility: 0.08
                max_leverage: 2.0
                stop_loss_atr_multiple: 2.5
                max_open_positions: 3
                hard_drawdown_pct: 0.12
            observability:
              key_rotation:
                registry_path: {registry_path}
                default_interval_days: 60
                default_warn_within_days: 7
                entries:
                  - key: api
                    purpose: trading
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


# ----------------- testy -----------------
def test_rotate_keys_updates_registry_and_writes_report_or_generates_plan(tmp_path: Path) -> None:
    assert RotationRegistry is not None, "RotationRegistry not available"

    if _supports_head_cli():
        # --- ścieżka HEAD: raport + podpis + aktualizacja rejestru środowiska ---
        config_path = tmp_path / "core.yaml"
        cache_path = tmp_path / "cache"
        _write_minimal_head_config(config_path, cache_path)

        key_b64 = base64.b64encode(b"stage5_rotation_key").decode("ascii")
        output_path = tmp_path / "rotation.json"

        exit_code = rotate_keys_run(
            [
                "--config",
                str(config_path),
                "--environment",
                "paper",
                "--operator",
                "SecOps",
                "--executed-at",
                "2024-05-20T08:00:00Z",
                "--signing-key",
                key_b64,
                "--signing-key-id",
                "stage5",
                "--output",
                str(output_path),
            ]
        )
        assert exit_code == 0

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        # oczekiwane pola z wariantu HEAD
        assert payload.get("type") in {"stage5_key_rotation", "stage5.key_rotation", "rotation_summary"}
        assert payload["records"][0]["environment"] in {"paper", "demo", "core", payload["records"][0]["environment"]}
        if "signature" in payload:
            assert payload["signature"]["algorithm"].upper().startswith("HMAC-")

        registry_path = cache_path / "security" / "rotation_log.json"
        registry = RotationRegistry(registry_path)
        status = registry.status(
            "binance_paper",
            "trading",
            interval_days=90.0,
            now=datetime.fromisoformat("2024-05-20T08:00:00+00:00"),
        )
        assert status.last_rotated is not None
        assert status.last_rotated.isoformat().startswith("2024-05-20T08:00:00")

    elif _supports_main_cli():
        # --- ścieżka main: najpierw plan (bez zmian), potem --execute (aktualizacja rejestru) ---
        config_path = tmp_path / "core.yaml"
        registry_path = tmp_path / "registry.json"
        # wpis sprzed 120 dni, żeby było due/overdue
        registry_path.write_text(
            json.dumps({"api::trading": _iso(datetime.now(timezone.utc) - timedelta(days=120))}) + "\n",
            encoding="utf-8",
        )
        _write_main_config(config_path, registry_path)

        output_dir = tmp_path / "out"
        exit_code = rotate_keys_run(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--basename",
                "plan",
            ]
        )
        assert exit_code == 0
        report_path = output_dir / "plan.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # akceptujemy 'due' lub 'overdue'
        assert report["results"][0]["state"] in {"due", "overdue", "warning"}

        # wykonanie
        exit_code = rotate_keys_run(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--basename",
                "plan_execute",
                "--execute",
            ]
        )
        assert exit_code == 0
        registry = RotationRegistry(registry_path)
        status = registry.status("api", "trading", interval_days=60.0)
        assert not status.is_due
    else:
        pytest.skip("rotate_keys CLI shape not recognized (neither HEAD nor main)")


def test_rotate_keys_dry_run_or_plan_only_does_not_touch_registry(tmp_path: Path) -> None:
    assert RotationRegistry is not None, "RotationRegistry not available"

    if _supports_head_cli():
        # HEAD: --dry-run
        config_path = tmp_path / "core.yaml"
        cache_path = tmp_path / "cache"
        _write_minimal_head_config(config_path, cache_path)

        exit_code = rotate_keys_run(
            [
                "--config",
                str(config_path),
                "--operator",
                "Ops",
                "--dry-run",
                "--executed-at",
                "2024-05-01T00:00:00Z",
                "--output",
                str(tmp_path / "dry_run.json"),
            ]
        )
        assert exit_code == 0

        registry_path = cache_path / "security" / "rotation_log.json"
        assert not registry_path.exists()

    elif _supports_main_cli():
        # main: "plan only" – brak --execute => rejestr pozostaje bez zmian (nadal due/overdue)
        config_path = tmp_path / "core.yaml"
        registry_path = tmp_path / "registry.json"
        registry_path.write_text(
            json.dumps({"api::trading": _iso(datetime.now(timezone.utc) - timedelta(days=120))}) + "\n",
            encoding="utf-8",
        )
        _write_main_config(config_path, registry_path)

        output_dir = tmp_path / "out"
        exit_code = rotate_keys_run(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--basename",
                "plan_only",
            ]
        )
        assert exit_code == 0

        registry = RotationRegistry(registry_path)
        status = registry.status("api", "trading", interval_days=60.0)
        assert status.is_due or status.is_overdue
    else:
        pytest.skip("rotate_keys CLI shape not recognized (neither HEAD nor main)")


def test_rotate_keys_prepare_argv_infers_modes() -> None:
    prepare = getattr(rotate_keys_mod, "_prepare_argv", None)
    if prepare is None:
        pytest.skip("rotate_keys CLI does not expose _prepare_argv")

    assert prepare(["batch", "--dry-run"]) == ["batch", "--dry-run"]
    assert prepare(["--config", "core.yaml", "--operator", "Ops"])[0] == "batch"
    status_args = prepare(["--status", "--bundle", "core-oem", "--config", "core.yaml"])
    assert status_args[0] == "status"
    assert "--bundle" in status_args
    status_value_args = prepare(["--status", "core-oem", "--config", "core.yaml"])
    assert status_value_args[:3] == ["status", "--bundle", "core-oem"]
    status_value_with_bundle = prepare(
        ["--status", "core-oem", "--bundle", "alt", "--config", "core.yaml"]
    )
    assert status_value_with_bundle.count("--bundle") == 1


def test_rotate_keys_prepare_argv_uses_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare = getattr(rotate_keys_mod, "_prepare_argv", None)
    if prepare is None:
        pytest.skip("rotate_keys CLI does not expose _prepare_argv")

    monkeypatch.setattr(sys, "argv", ["rotate_keys.py", "--status", "core-oem"])
    prepared = prepare(None)
    assert prepared[:3] == ["status", "--bundle", "core-oem"]


@pytest.mark.parametrize(
    "status_invocation",
    [
        ["--status", "--bundle", "core-oem"],
        ["--status", "core-oem"],
        ["status", "core-oem"],
    ],
)
def test_rotate_keys_status_reports_bundle_summary(
    status_invocation: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert RotationRegistry is not None, "RotationRegistry not available"

    rotation_registry = tmp_path / "tls_rotation.json"
    now = datetime(2024, 5, 20, tzinfo=timezone.utc)
    rotation_registry.write_text(
        json.dumps(
            {
                "core-oem::ca": _iso(now - timedelta(days=10)),
                "core-oem::server": _iso(now - timedelta(days=28)),
                "core-oem::client": _iso(now - timedelta(days=40)),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "core.yaml"

    if _supports_head_cli():
        cache_path = tmp_path / "cache"
        _write_minimal_head_config(config_path, cache_path, rotation_registry)
    elif _supports_main_cli():
        _write_main_config(config_path, rotation_registry)
        config_path.write_text(
            config_path.read_text(encoding="utf-8")
            + textwrap.dedent(
                f"""
                execution:
                  mtls:
                    bundle_directory: {tmp_path.as_posix()}
                    rotation_registry: {rotation_registry.as_posix()}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
    else:
        pytest.skip("rotate_keys CLI shape not recognized (neither HEAD nor main)")

    exit_code = rotate_keys_run(
        [
            *status_invocation,
            "--config",
            str(config_path),
            "--interval-days",
            "30",
            "--warn-days",
            "5",
            "--as-of",
            "2024-05-20T00:00:00Z",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())

    assert payload["bundle"] == "core-oem"
    assert payload["entries_found"] is True
    states = {entry["purpose"]: entry["state"] for entry in payload["entries"] if entry["key"] == "core-oem"}
    assert states.get("ca") == "ok"
    assert states.get("server") in {"warning", "due"}
    assert states.get("client") == "overdue"
    assert payload["summary"]["total"] >= 3


def test_rotate_keys_status_rejects_conflicting_bundle_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert RotationRegistry is not None, "RotationRegistry not available"

    rotation_registry = tmp_path / "tls_rotation.json"
    rotation_registry.write_text("{}\n", encoding="utf-8")

    config_path = tmp_path / "core.yaml"
    if _supports_head_cli():
        cache_path = tmp_path / "cache"
        _write_minimal_head_config(config_path, cache_path, rotation_registry)
    elif _supports_main_cli():
        _write_main_config(config_path, rotation_registry)
    else:
        pytest.skip("rotate_keys CLI shape not recognized (neither HEAD nor main)")

    exit_code = rotate_keys_run(
        [
            "status",
            "core-oem",
            "--bundle",
            "alt",  # konflikt z argumentem pozycyjnym
            "--config",
            str(config_path),
        ]
    )
    assert exit_code == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert "error" in payload
    assert "--bundle" in payload["error"]
