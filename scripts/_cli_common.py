"""Wspólne helpery CLI dla skryptów audytowych."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from bot_core.security import SecretManager, create_default_secret_storage


def env_flag(prefix: str, name: str, default: bool) -> bool:
    value = os.environ.get(f"{prefix}{name}")
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def env_value(prefix: str, name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"{prefix}{name}")
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def env_list(prefix: str, name: str) -> list[str]:
    raw = env_value(prefix, name)
    if not raw:
        return []
    separators = [";", ","]
    for sep in separators:
        if sep in raw:
            return [entry.strip() for entry in raw.split(sep) if entry.strip()]
    return [raw]


def normalize_scopes(
    values: Sequence[str] | None,
    *,
    default: Sequence[str] | None = None,
) -> tuple[str, ...]:
    scopes: list[str] = []
    if not values:
        return tuple(default or ())
    for entry in values:
        normalized = str(entry).strip().lower()
        if normalized:
            scopes.append(normalized)
    if scopes:
        return tuple(scopes)
    return tuple(default or ())


def should_print(prefix: str, *, json_output: str | None, cli_flag: bool, default_when_unspecified: bool) -> bool:
    if cli_flag:
        return True
    return env_flag(prefix, "PRINT", default_when_unspecified)


def timestamp_slug(moment: datetime | None = None) -> str:
    reference = moment or datetime.now(timezone.utc)
    return reference.strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_decision_log_path(
    governor: str,
    *,
    moment: datetime | None = None,
    directory: Path | None = None,
) -> Path:
    base = directory or Path("var/audit/decision_log")
    return base / f"portfolio_decision_{governor}_{timestamp_slug(moment)}.jsonl"


def create_secret_manager(
    *,
    namespace: str,
    headless_passphrase: str | None = None,
    headless_path: str | None = None,
) -> SecretManager:
    storage = create_default_secret_storage(
        namespace=namespace,
        headless_passphrase=headless_passphrase,
        headless_path=headless_path,
    )
    return SecretManager(storage, namespace=namespace)


__all__ = [
    "env_flag",
    "env_value",
    "env_list",
    "normalize_scopes",
    "should_print",
    "timestamp_slug",
    "now_iso",
    "default_decision_log_path",
    "create_secret_manager",
]
