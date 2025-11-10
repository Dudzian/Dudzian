#!/usr/bin/env python3
"""Eksportuje listę adapterów giełdowych do raportu benchmarkowego."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - konfiguracja ścieżki wykonywana raz
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - środowiska testowe mogą nie mieć packaging
    import packaging.version  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback minimalny
    import types

    packaging_module = types.ModuleType("packaging")
    version_module = types.ModuleType("version")

    class Version(str):  # type: ignore
        def __new__(cls, value: str) -> "Version":
            return str.__new__(cls, value)

    class InvalidVersion(Exception):
        """Minimalna implementacja na potrzeby loadera konfiguracji."""

    version_module.Version = Version
    version_module.InvalidVersion = InvalidVersion
    packaging_module.version = version_module
    sys.modules.setdefault("packaging", packaging_module)
    sys.modules.setdefault("packaging.version", version_module)

from bot_core.config.loader import load_core_config


def _stream_base_url(stream_settings: Mapping[str, Any], *, environment: str) -> str | None:
    candidates = (
        stream_settings.get(f"{environment}_base_url"),
        stream_settings.get("base_url"),
        stream_settings.get("live_base_url"),
        stream_settings.get("testnet_base_url"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def build_rows(config) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    rows: list[dict[str, Any]] = []

    for exchange, profiles in config.exchange_accounts.items():
        for profile_name, account in profiles.items():
            env_name = account.environment
            env_cfg = config.environments.get(env_name)
            if env_cfg is None:
                continue

            mode_key = env_cfg.exchange.split("_")[-1]
            adapter_entry = None
            adapter_class = "ccxt"
            supports_testnet = False
            for key, entry in config.exchange_adapters.get(exchange, {}).items():
                key_name = getattr(key, "value", str(key))
                if key_name == mode_key:
                    adapter_entry = entry
                    adapter_class = entry.class_path
                    supports_testnet = bool(entry.supports_testnet)
                    break

            default_settings = adapter_entry.default_settings if adapter_entry else {}
            stream_settings = default_settings.get("stream", {}) if isinstance(default_settings, Mapping) else {}
            retry_policy = default_settings.get("retry_policy", {}) if isinstance(default_settings, Mapping) else {}

            live_readiness = getattr(env_cfg, "live_readiness", None)
            readiness_signed = bool(getattr(live_readiness, "signed", False))
            readiness_signed_by: list[str] = []
            signed_by_attr = getattr(live_readiness, "signed_by", None)
            if isinstance(signed_by_attr, (list, tuple)):
                readiness_signed_by = [str(entry) for entry in signed_by_attr]
            if isinstance(live_readiness, Mapping):  # pragma: no cover - kompatybilność starych schematów
                readiness_signed = bool(live_readiness.get("signed", False))
                readiness_signed_by = [str(entry) for entry in live_readiness.get("signed_by", ())]

            rows.append(
                {
                    "exchange": exchange,
                    "profile": profile_name,
                    "mode": mode_key,
                    "environment": env_cfg.environment.value,
                    "adapter": adapter_class,
                    "supports_testnet": supports_testnet,
                    "stream_base_url": _stream_base_url(
                        stream_settings,
                        environment=env_cfg.environment.value,
                    ),
                    "retry_max_attempts": retry_policy.get("max_attempts"),
                    "retry_max_delay": retry_policy.get("max_delay"),
                    "live_readiness_signed": readiness_signed,
                    "live_readiness_signed_by": ",".join(readiness_signed_by),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku config/core.yaml")
    parser.add_argument("--output", help="Plik CSV do zapisania raportu")
    args = parser.parse_args(argv)

    config = load_core_config(Path(args.config))
    rows = build_rows(config)

    fieldnames = [
        "exchange",
        "profile",
        "mode",
        "environment",
        "adapter",
        "supports_testnet",
        "stream_base_url",
        "retry_max_attempts",
        "retry_max_delay",
        "live_readiness_signed",
        "live_readiness_signed_by",
    ]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="", encoding="utf-8")
        close_handle = True
    else:
        handle = sys.stdout
        close_handle = False

    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close_handle:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

