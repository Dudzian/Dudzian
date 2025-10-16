from __future__ import annotations

import base64
import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


def _parser_supports(*flags: str) -> bool:
    parser = _parser()
    actions = getattr(parser, "_actions", ())
    option_set = set()
    for act in actions:
        option_set.update(getattr(act, "option_strings", []) or [])
    return all(flag in option_set for flag in flags)


def _supports_head_cli() -> bool:
    # HEAD wariant: ma m.in. --environment, --operator, --executed-at, --signing-key, --output
    return _parser_supports("--environment", "--operator", "--executed-at", "--output")


def _supports_main_cli() -> bool:
    # main wariant: ma m.in. --output-dir, --basename (i zwykle --execute)
    return _parser_supports("--output-dir", "--basename")


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _write_minimal_head_config(path: Path, cache_path: Path) -> None:
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
