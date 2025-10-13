"""Narzędzia do pobierania i podpisywania odcisku sprzętowego hosta."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from bot_core.security.rotation import RotationRegistry, RotationStatus
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes


_DEFAULT_PURPOSE = "hardware-fingerprint"
_HEX_RE = re.compile(r"[^0-9a-f]")


def _ensure_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    return collapsed


def _component_entry(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    normalized = _normalize_text(raw)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {"raw": raw, "normalized": normalized, "digest": digest}


def _read_first_existing(paths: list[Path]) -> str | None:
    for candidate in paths:
        try:
            data = candidate.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError:
            continue
        content = data.strip()
        if content:
            return content
    return None


def _probe_cpu_info() -> str | None:
    entries: list[str] = []
    uname = platform.uname()
    entries.extend(filter(None, {uname.system, uname.machine, uname.processor}))

    processor = platform.processor()
    if processor and processor not in entries:
        entries.append(processor)

    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    _, _, value = line.partition(":")
                    entries.append(value.strip())
                    break
    except FileNotFoundError:
        pass
    except OSError:
        pass

    if not entries:
        return None
    return " | ".join(dict.fromkeys(entries))


def _probe_tpm_info() -> str | None:
    system = platform.system().lower()
    if system == "windows":
        try:
            completed = subprocess.run(
                ["wmic", "tpm", "get", "ManufacturerId", ",", "ManufacturerVersion"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        output = (completed.stdout or "") + (completed.stderr or "")
        tokens = [token.strip() for token in output.splitlines() if token.strip()]
        if tokens:
            return " | ".join(tokens)
        return None

    if system == "darwin":
        return None

    # Linux i pozostałe systemy POSIX
    base = Path("/sys/class/tpm")
    if not base.exists():
        return None

    candidates: list[Path] = []
    for entry in sorted(base.iterdir()):
        if not entry.name.startswith("tpm"):
            continue
        candidates.extend(
            [
                entry / "device" / "description",
                entry / "device" / "manufacturer",
                entry / "device" / "modalias",
            ]
        )

    if not candidates:
        return None

    content = _read_first_existing(candidates)
    if content:
        return content
    return None


def _format_mac_address(value: int) -> str:
    hex_value = f"{value:012x}"
    parts = [hex_value[i : i + 2] for i in range(0, len(hex_value), 2)]
    return ":".join(parts)


def _probe_mac_address() -> str | None:
    node = uuid.getnode()
    if node is None:
        return None
    return _format_mac_address(node)


def _probe_dongle() -> str | None:
    raw = os.environ.get("DUDZIAN_DONGLE_ID")
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


@dataclass(slots=True)
class FingerprintRecord:
    """Wynik podpisanego fingerprintu."""

    payload: Mapping[str, Any]
    signature: Mapping[str, str]
    key_id: str

    def as_dict(self) -> dict[str, Any]:
        return {"payload": dict(self.payload), "signature": dict(self.signature)}


class RotatingHmacKeyProvider:
    """Zarządza kluczami HMAC w oparciu o ``RotationRegistry``."""

    def __init__(
        self,
        keys: Mapping[str, bytes | str],
        registry: RotationRegistry,
        *,
        purpose: str = _DEFAULT_PURPOSE,
        interval_days: float = 90.0,
    ) -> None:
        if not keys:
            raise ValueError("Wymagany jest co najmniej jeden klucz HMAC.")
        normalized: dict[str, bytes] = {}
        for key_id, raw_value in keys.items():
            if not key_id:
                raise ValueError("Identyfikator klucza HMAC nie może być pusty.")
            if isinstance(raw_value, str):
                value = raw_value.encode("utf-8")
            else:
                value = bytes(raw_value)
            normalized[str(key_id)] = value

        self._keys = normalized
        self._registry = registry
        self._purpose = purpose
        self._interval = interval_days

    @property
    def purpose(self) -> str:
        return self._purpose

    def key_ids(self) -> tuple[str, ...]:
        return tuple(self._keys)

    def status_for(self, key_id: str, *, now: datetime | None = None) -> RotationStatus:
        return self._registry.status(
            key_id,
            self._purpose,
            interval_days=self._interval,
            now=now,
        )

    def _priority_tuple(self, status: RotationStatus) -> tuple[int, float]:
        if not status.is_due and not status.is_overdue:
            priority = 0
        elif status.is_due and not status.is_overdue:
            priority = 1
        else:
            priority = 2
        if status.last_rotated is None:
            timestamp = float("inf")
        else:
            timestamp = -status.last_rotated.timestamp()
        return priority, timestamp

    def select_active_key(self, *, now: datetime | None = None) -> tuple[str, bytes, RotationStatus]:
        statuses: dict[str, RotationStatus] = {}
        for key_id in self._keys:
            statuses[key_id] = self.status_for(key_id, now=now)

        sorted_ids = sorted(
            statuses.items(),
            key=lambda item: (self._priority_tuple(item[1]), item[0]),
        )

        active_id, status = sorted_ids[0]
        return active_id, self._keys[active_id], status

    def sign(self, payload: Mapping[str, Any], *, now: datetime | None = None) -> tuple[str, dict[str, str]]:
        key_id, key, _status = self.select_active_key(now=now)
        signature = build_hmac_signature(payload, key=key, key_id=key_id)
        return key_id, signature


class HardwareFingerprintService:
    """Buduje deterministyczny fingerprint sprzętowy podpisany HMAC."""

    def __init__(
        self,
        key_provider: RotatingHmacKeyProvider,
        *,
        cpu_probe: Callable[[], str | None] | None = None,
        tpm_probe: Callable[[], str | None] | None = None,
        mac_probe: Callable[[], str | None] | None = None,
        dongle_probe: Callable[[], str | None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._provider = key_provider
        self._cpu_probe = cpu_probe or _probe_cpu_info
        self._tpm_probe = tpm_probe or _probe_tpm_info
        self._mac_probe = mac_probe or _probe_mac_address
        self._dongle_probe = dongle_probe or _probe_dongle
        self._clock = clock or _ensure_utc_now

    def _collect_components(self, dongle_serial: str | None) -> dict[str, Any]:
        cpu_component = _component_entry(self._cpu_probe())
        tpm_component = _component_entry(self._tpm_probe())
        mac_raw = self._mac_probe()
        if mac_raw:
            mac_clean = _HEX_RE.sub("", mac_raw.lower())
            mac_component = _component_entry(mac_clean)
            if mac_component:
                mac_component["raw"] = mac_raw.strip()
        else:
            mac_component = None

        if dongle_serial is None:
            dongle_serial = self._dongle_probe()
        dongle_component = _component_entry(dongle_serial)

        return {
            "cpu": cpu_component,
            "tpm": tpm_component,
            "mac": mac_component,
            "dongle": dongle_component,
        }

    def build(self, *, dongle_serial: str | None = None) -> FingerprintRecord:
        components = self._collect_components(dongle_serial)

        digest_material: MutableMapping[str, str] = {}
        for key, entry in components.items():
            if entry is None:
                continue
            digest_material[key] = entry["digest"]

        payload_base = {
            "version": 1,
            "components": digest_material,
        }
        fingerprint_digest = hashlib.sha256(canonical_json_bytes(payload_base)).hexdigest()

        current_time = self._clock().replace(microsecond=0)
        collected_at = current_time.isoformat().replace("+00:00", "Z")

        payload = {
            "version": 1,
            "collected_at": collected_at,
            "components": components,
            "component_digests": digest_material,
            "fingerprint": {
                "algorithm": "sha256",
                "value": fingerprint_digest,
            },
        }

        key_id, signature = self._provider.sign(payload, now=current_time)
        return FingerprintRecord(payload=payload, signature=signature, key_id=key_id)


def decode_secret(value: str) -> bytes:
    text = value.strip()
    if text.startswith("hex:"):
        return bytes.fromhex(text[4:])
    if text.startswith("base64:"):
        import base64

        return base64.b64decode(text[7:])
    return text.encode("utf-8")


def build_key_provider(
    keys: Mapping[str, str | bytes],
    rotation_log: str | Path,
    *,
    purpose: str = _DEFAULT_PURPOSE,
    interval_days: float = 90.0,
) -> RotatingHmacKeyProvider:
    registry = RotationRegistry(rotation_log)
    return RotatingHmacKeyProvider(keys, registry, purpose=purpose, interval_days=interval_days)


def _parse_key_argument(text: str) -> tuple[str, bytes]:
    if "=" not in text:
        raise ValueError("Parametr klucza musi mieć format key_id=wartość")
    key_id, value = text.split("=", 1)
    return key_id.strip(), decode_secret(value)


def _load_keys_from_args(args: list[str]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for entry in args:
        key_id, secret = _parse_key_argument(entry)
        result[key_id] = secret
    return result


def cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generuje odcisk sprzętowy hosta i podpisuje go HMAC.")
    parser.add_argument("--dongle", help="Wymuszony identyfikator klucza sprzętowego USB.")
    parser.add_argument(
        "--rotation-log",
        default="var/licenses/fingerprint_rotation.json",
        help="Ścieżka do rejestru rotacji kluczy HMAC.",
    )
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="Klucz HMAC w formacie key_id=sekret (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--output",
        choices=["json"],
        default="json",
        help="Format wyjścia (domyślnie JSON).",
    )
    parser.add_argument(
        "--purpose",
        default=_DEFAULT_PURPOSE,
        help="Cel wpisów w rejestrze rotacji kluczy.",
    )
    parser.add_argument(
        "--interval-days",
        type=float,
        default=90.0,
        help="Oczekiwany interwał rotacji kluczy HMAC.",
    )

    parsed = parser.parse_args(argv)

    keys = _load_keys_from_args(parsed.keys or [])
    if not keys:
        parser.error("Wymagany jest co najmniej jeden klucz --key do podpisania fingerprintu.")

    provider = build_key_provider(
        keys,
        parsed.rotation_log,
        purpose=parsed.purpose,
        interval_days=parsed.interval_days,
    )
    service = HardwareFingerprintService(provider)
    record = service.build(dongle_serial=parsed.dongle)

    if parsed.output == "json":
        json.dump(record.as_dict(), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(cli())


__all__ = [
    "FingerprintRecord",
    "HardwareFingerprintService",
    "RotatingHmacKeyProvider",
    "build_key_provider",
    "decode_secret",
]

