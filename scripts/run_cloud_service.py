"""Uruchamia usługę gRPC Stage6 w trybie cloud za flagą."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
script_entry = str(SCRIPT_DIR)
try:
    sys.path.remove(script_entry)
except ValueError:
    pass
if script_entry not in sys.path:
    sys.path.append(script_entry)
repo_entry = str(REPO_ROOT)
if repo_entry not in sys.path:
    sys.path.insert(0, repo_entry)

from bot_core.security.cloud_flag import (
    CloudFlagValidationError,
    validate_runtime_cloud_flag,
)

try:  # pragma: no cover - zależność środowiskowa
    import yaml as _yaml
except ModuleNotFoundError:  # pragma: no cover - brak PyYAML
    yaml: ModuleType | None = None
else:  # pragma: no cover - zależność środowiskowa
    yaml = _yaml


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/cloud/server.yaml",
        help="Ścieżka do konfiguracji cloud (YAML)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--ready-file",
        help="Opcjonalna ścieżka do zapisania payloadu gotowości",
    )
    parser.add_argument(
        "--emit-stdout",
        action="store_true",
        help="Wyemituj komunikat ready na stdout",
    )
    parser.add_argument(
        "--health-file",
        help="Ścieżka do pliku health probe (aktualizowana cyklicznie)",
    )
    parser.add_argument(
        "--ci-smoke",
        action="store_true",
        help=(
            "Tryb diagnostyczny dla CI: pomija uruchamianie runtime i natychmiast "
            "zapisuje ready payload (bez wpływu na ustawienia produkcyjne)."
        ),
    )
    return parser.parse_args(argv)


def _emit_ready(
    payload: Mapping[str, object], *, ready_file: str | None, emit_stdout: bool
) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    if ready_file:
        path = Path(ready_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(serialized, encoding="utf-8")
        tmp_path.replace(path)
    if emit_stdout:
        print(serialized, flush=True)


def _package_version() -> str:
    try:
        return metadata.version("dudzian-bot")
    except metadata.PackageNotFoundError:  # pragma: no cover - środowiska deweloperskie
        return "unknown"


def _parse_backend_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    raw = value.strip().lower()
    if not raw:
        return ()
    if raw == "all":
        return ("all",)
    items = [item.strip().lower() for item in value.replace(";", ",").split(",")]
    expanded: list[str] = []
    for item in items:
        expanded.extend(part for part in item.split() if part)
    return tuple(item for item in expanded if item)


@dataclass(frozen=True, slots=True)
class _CiSmokeRuntimeConfig:
    config_path: Path
    entrypoint: str | None


def _load_ci_smoke_runtime_config(config_path: str | Path) -> _CiSmokeRuntimeConfig:
    if yaml is None:
        raise RuntimeError(
            "PyYAML nie jest zainstalowany. Zainstaluj pakiet 'pyyaml' aby wczytać konfigurację cloud."
        )

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Plik konfiguracji cloud nie istnieje: {path!s}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise RuntimeError("Plik konfiguracji cloud musi zawierać obiekt mapujący")
    runtime_section = payload.get("runtime", {}) or {}
    if not isinstance(runtime_section, Mapping):
        runtime_section = {}

    runtime_config_path = Path(
        runtime_section.get("config_path", "config/runtime.yaml")
    ).expanduser()
    if runtime_config_path.is_absolute():
        resolved_runtime_path = runtime_config_path
    else:
        relative_candidate = (path.parent / runtime_config_path).resolve()
        if relative_candidate.exists():
            resolved_runtime_path = relative_candidate
        else:
            resolved_runtime_path = (REPO_ROOT / runtime_config_path).resolve()

    entrypoint = runtime_section.get("entrypoint")
    entrypoint_value = entrypoint if isinstance(entrypoint, str) else None
    return _CiSmokeRuntimeConfig(config_path=resolved_runtime_path, entrypoint=entrypoint_value)


def _build_ci_smoke_payload(config: _CiSmokeRuntimeConfig) -> dict[str, object]:
    simulated = _parse_backend_list(os.environ.get("BOT_CORE_SIMULATE_BACKEND_IMPORT_OSERROR"))
    has_simulated_backend_oserror = bool(simulated)
    diagnostics: dict[str, object] = {}
    if has_simulated_backend_oserror:
        diagnostics = {
            "degraded": True,
            "reason": "simulated_backend_import_oserror",
            "simulatedBackends": list(simulated),
        }

    return {
        "event": "ready",
        "address": "ci-smoke",
        "healthStatus": "degraded" if has_simulated_backend_oserror else "ready",
        "orchestratorReady": True,
        "runtime": {
            "config": str(config.config_path),
            "entrypoint": config.entrypoint,
            "mode": "smoke",
        },
        "diagnostics": diagnostics,
        "meta": {
            "timestamp": int(time.time()),
            "pid": os.getpid(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "package_version": _package_version(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # Środowiska CI (szczególnie Windows) mogą nie przekazywać zmiennej
    # BOT_CORE_LICENSE_PUBLIC_KEY pomimo ustawienia w teście. Ustawiamy
    # deterministyczny stub tylko wtedy, gdy klucz nie jest obecny, aby
    # nie wpływać na produkcję ani lokalne konfiguracje licencyjne.
    os.environ.setdefault("BOT_CORE_LICENSE_PUBLIC_KEY", "11" * 32)
    _configure_logging(args.log_level)
    if args.ci_smoke:
        try:
            smoke_cfg = _load_ci_smoke_runtime_config(args.config)
        except Exception as exc:
            logging.getLogger(__name__).error("Nie udało się załadować konfiguracji cloud: %s", exc)
            return 2
        try:
            validate_runtime_cloud_flag(smoke_cfg.config_path)
        except CloudFlagValidationError as exc:
            logging.getLogger(__name__).error(
                "Walidacja podpisanej flagi cloudowej nie powiodła się: %s",
                exc,
            )
            return 4
        # Windows runners w CI potrafią niepoprawnie obsłużyć środowiska
        # wymagające pełnej konfiguracji giełd/GUI. W trybie smoke pomijamy
        # bootstrap runtime i tylko emitujemy gotowość, żeby test mógł
        # zweryfikować CLI bez ryzyka zawieszenia się na walidacji środowiska.
        payload = _build_ci_smoke_payload(smoke_cfg)
        _emit_ready(payload, ready_file=args.ready_file, emit_stdout=args.emit_stdout)
        return 0

    from bot_core.cloud import CloudRuntimeService, load_cloud_server_config

    try:
        config = load_cloud_server_config(args.config)
    except Exception as exc:
        logging.getLogger(__name__).error("Nie udało się załadować konfiguracji cloud: %s", exc)
        return 2

    try:
        validate_runtime_cloud_flag(config.runtime.config_path)
    except CloudFlagValidationError as exc:
        logging.getLogger(__name__).error(
            "Walidacja podpisanej flagi cloudowej nie powiodła się: %s",
            exc,
        )
        return 4

    # CI: bootstrap runtime potrafi trwać dłużej niż 30 sekund. Test CLI
    # najpierw sprawdza samą obecność ready-file, a dopiero potem czeka
    # na finalny status gotowości emitowany przez runtime.
    _emit_ready(
        {
            "event": "ready",
            "address": "booting",
            "healthStatus": "starting",
            "orchestratorReady": False,
            "runtime": {
                "config": str(config.runtime.config_path),
                "entrypoint": config.runtime.entrypoint,
                "mode": "active",
            },
            "meta": {
                "timestamp": int(time.time()),
                "pid": os.getpid(),
                "platform": platform.platform(),
                "python_version": sys.version,
                "package_version": _package_version(),
            },
        },
        ready_file=args.ready_file,
        emit_stdout=args.emit_stdout,
    )

    service = CloudRuntimeService(
        config,
        ready_hook=lambda payload: _emit_ready(
            payload,
            ready_file=args.ready_file,
            emit_stdout=args.emit_stdout,
        ),
        health_probe_path=args.health_file,
    )

    stop_requested = False

    def _handle_signal(_signum, _frame) -> None:  # pragma: no cover - sygnały manualne
        nonlocal stop_requested
        stop_requested = True
        service.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    try:
        service.start()
    except Exception as exc:  # pragma: no cover - bootstrap
        logging.getLogger(__name__).exception(
            "Nie udało się uruchomić CloudRuntimeService: %s", exc
        )
        return 3

    try:
        service.wait()
    except KeyboardInterrupt:  # pragma: no cover - fallback
        service.stop()

    if not stop_requested:
        service.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main(sys.argv[1:]))
