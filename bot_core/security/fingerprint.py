
"""Generowanie/pozyskiwanie i podpisywanie fingerprintów sprzętowych (OEM + host)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

from bot_core.security.rotation import RotationRegistry, RotationStatus
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes

# ---------------------------------------------------------------------------
# Wspólne stałe/typy
# ---------------------------------------------------------------------------

# OEM (starsze API) – format skrótu i algorytm podpisu
FINGERPRINT_HASH = "SHA384"
SIGNATURE_ALGORITHM = "HMAC-SHA384"

# Nowsze API – domyślny "cel" w rejestrze rotacji
_DEFAULT_PURPOSE = "hardware-fingerprint"
# Starsze API – zachowanie zgodności w helperach OEM
_OEM_DEFAULT_PURPOSE = "oem-fingerprint-signing"

_HEX_RE = re.compile(r"[^0-9a-f]")


class FingerprintError(RuntimeError):
    """Wyjątek zgłaszany przy problemach z generowaniem/podpisywaniem fingerprintu."""


# ---------------------------------------------------------------------------
# Starsze API – OEM Device Fingerprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FingerprintDocument:
    """Dokument OEM fingerprintu z podpisem HMAC."""

    payload: Mapping[str, object]
    signature: Mapping[str, object]

    def to_json(self) -> str:
        return json.dumps(
            {"payload": self.payload, "signature": self.signature},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )


def _now_iso(timestamp: Optional[datetime] = None) -> str:
    value = timestamp or datetime.now(timezone.utc)
    value = value.astimezone(timezone.utc).replace(microsecond=0)
    return value.isoformat().replace("+00:00", "Z")


def _read_first_line(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline().strip()
    except FileNotFoundError:
        return None
    return line or None


def _run_command(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError):
        return None
    output = completed.stdout.strip()
    return output or None


def _sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = value.strip()
    return candidate or None


def _probe_cpu_identifier(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_CPU_ID"))
    if override:
        return override

    serial = _read_first_line(Path("/sys/devices/virtual/dmi/id/product_uuid"))
    if serial:
        return serial

    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            if key.strip().lower() in {"serial", "hardware", "processor"}:
                candidate = _sanitize(raw)
                if candidate:
                    return candidate
    except FileNotFoundError:
        pass

    mac = _sanitize(f"{uuid.getnode():012x}")
    if mac:
        return mac

    uname = platform.uname()
    fallback = "-".join(filter(None, [uname.system, uname.node, uname.machine]))
    return fallback or None


def _probe_tpm_identifier(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_TPM_ID"))
    if override:
        return override

    candidate = _read_first_line(Path("/sys/class/tpm/tpm0/device/unique_id"))
    if candidate:
        return candidate

    candidate = _read_first_line(Path("/sys/class/tpm/tpm0/unique_id"))
    if candidate:
        return candidate

    candidate = _run_command(["tpm2_getcap", "-c", "properties-fixed"])
    if candidate:
        return hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    return None


def _probe_dongle_identifier(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_DONGLE_ID"))
    if override:
        return override

    default_path = Path(env.get("OEM_DONGLE_PATH", "var/oem/dongle_id.txt"))
    return _read_first_line(default_path)


def _collect_mac_addresses(env: Mapping[str, str]) -> list[str]:
    override = _sanitize(env.get("OEM_MAC_ADDRESSES"))
    if override:
        addresses = {
            candidate.replace(":", "").replace("-", "").lower()
            for candidate in (_sanitize(entry) for entry in override.split(","))
            if candidate
        }
        return sorted(addresses)

    addresses: set[str] = set()
    try:
        node = uuid.getnode()
    except ValueError:
        node = None
    if node is not None:
        addresses.add(f"{node:012x}")

    sys_class = Path("/sys/class/net")
    if sys_class.exists():
        for interface in sys_class.iterdir():
            address_path = interface / "address"
            candidate = _read_first_line(address_path)
            if candidate and candidate != "00:00:00:00:00:00":
                addresses.add(candidate.replace(":", "").lower())

    if not addresses:
        try:
            hostname = socket.gethostname().encode("utf-8")
        except OSError:
            hostname = b""
        if hostname:
            addresses.add(hashlib.sha256(hostname).hexdigest()[:12])

    return sorted(addresses)


def _collect_hostname(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_HOSTNAME"))
    if override:
        return override
    return _sanitize(platform.node())


def _collect_os_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def _normalize_factors(factors: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in sorted(factors.items()):
        if value is None:
            continue
        if isinstance(value, Mapping):
            normalized[key] = _normalize_factors(value)
        elif isinstance(value, (list, tuple, set)):
            normalized[key] = [str(item) for item in sorted(value)]
        else:
            normalized[key] = str(value)
    return normalized


def _ensure_key_strength(key: bytes) -> None:
    if len(key) < 32:
        raise FingerprintError("Klucz podpisu fingerprintu musi mieć co najmniej 32 bajty")


def _sign_payload(payload: Mapping[str, object], key: bytes, key_id: Optional[str]) -> dict[str, str]:
    digest = hmac.new(key, canonical_json_bytes(payload), hashlib.sha384).digest()
    signature = {
        "algorithm": SIGNATURE_ALGORITHM,
        "value": base64.b64encode(digest).decode("ascii"),
    }
    if key_id:
        signature["key_id"] = str(key_id)
    return signature


class DeviceFingerprintGenerator:
    """Generator fingerprintu bazujący na metadanych sprzętowych (OEM API)."""

    def __init__(
        self,
        *,
        env: Optional[Mapping[str, str]] = None,
        extra_probes: Optional[Iterable[Callable[[MutableMapping[str, object]], None]]] = None,
    ) -> None:
        self._env = env or os.environ
        self._extra_probes = tuple(extra_probes or ())

    def collect_factors(self) -> dict[str, object]:
        factors: MutableMapping[str, object] = {}

        cpu_id = _probe_cpu_identifier(self._env)
        if cpu_id:
            factors["cpu_id"] = cpu_id

        mac_addresses = _collect_mac_addresses(self._env)
        if mac_addresses:
            factors["mac_addresses"] = mac_addresses

        hostname = _collect_hostname(self._env)
        if hostname:
            factors["hostname"] = hostname

        os_info = _collect_os_info()
        factors["os"] = os_info

        tpm = _probe_tpm_identifier(self._env)
        if tpm:
            factors["tpm"] = tpm

        dongle = _probe_dongle_identifier(self._env)
        if dongle:
            factors["dongle"] = dongle

        salt = _sanitize(self._env.get("OEM_FINGERPRINT_SALT"))
        if salt:
            factors["salt"] = salt

        for probe in self._extra_probes:
            probe(factors)

        return dict(factors)

    def generate_fingerprint(self, *, factors: Optional[Mapping[str, object]] = None) -> str:
        factors = factors or self.collect_factors()
        canonical = canonical_json_bytes(_normalize_factors(factors))
        digest = hashlib.sha384(canonical).digest()
        token = base64.b32encode(digest).decode("ascii").rstrip("=").upper()
        grouped = [token[i : i + 8] for i in range(0, len(token), 8)]
        return "-".join(grouped)

    def build_document(
        self,
        *,
        signing_key: bytes,
        key_id: Optional[str] = None,
        factors: Optional[Mapping[str, object]] = None,
        registry: Optional[RotationRegistry] = None,
        purpose: str = _OEM_DEFAULT_PURPOSE,
        rotation_interval_days: float = 90.0,
        mark_rotation: bool = False,
        created_at: Optional[datetime] = None,
    ) -> FingerprintDocument:
        _ensure_key_strength(signing_key)
        collected = _normalize_factors(factors or self.collect_factors())
        fingerprint = self.generate_fingerprint(factors=collected)
        timestamp = datetime.now(timezone.utc) if created_at is None else created_at.astimezone(timezone.utc)

        if registry and key_id:
            status = registry.status(key_id, purpose, interval_days=rotation_interval_days, now=timestamp)
            if status.last_rotated is not None and status.is_overdue:
                raise FingerprintError(
                    f"Klucz '{key_id}' dla celu '{purpose}' jest przeterminowany (ostatnia rotacja {status.last_rotated})."
                )

        payload = {
            "fingerprint": fingerprint,
            "algorithm": FINGERPRINT_HASH,
            "created_at": _now_iso(timestamp),
            "factors": collected,
        }
        signature = _sign_payload(payload, signing_key, key_id)

        if registry and key_id and mark_rotation:
            registry.mark_rotated(key_id, purpose, timestamp=timestamp)

        return FingerprintDocument(payload=payload, signature=signature)


def verify_document(document: Mapping[str, object], *, key: bytes) -> bool:
    payload = document.get("payload")
    signature = document.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
        raise FingerprintError("Nieprawidłowa struktura dokumentu fingerprintu")
    key_id_val = signature.get("key_id")
    key_id_str = key_id_val if isinstance(key_id_val, str) else None
    expected = _sign_payload(payload, key, key_id_str)
    return expected["value"] == signature.get("value")


def get_local_fingerprint() -> str:
    generator = DeviceFingerprintGenerator()
    return generator.generate_fingerprint()


def build_fingerprint_document(
    *,
    signing_key: bytes,
    key_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    registry: Optional[RotationRegistry] = None,
    purpose: str = _OEM_DEFAULT_PURPOSE,
    rotation_interval_days: float = 90.0,
    mark_rotation: bool = False,
    created_at: Optional[datetime] = None,
) -> FingerprintDocument:
    """Funkcja pomocnicza budująca podpisany dokument fingerprintu OEM."""
    generator = DeviceFingerprintGenerator(env=env)
    return generator.build_document(
        signing_key=signing_key,
        key_id=key_id,
        registry=registry,
        purpose=purpose,
        rotation_interval_days=rotation_interval_days,
        mark_rotation=mark_rotation,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Nowsze API – Hardware Fingerprint Service (z rotacją kluczy)
# ---------------------------------------------------------------------------

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


def _read_first_existing(paths: Sequence[Path]) -> str | None:
    for candidate in paths:
        try:
            data = candidate.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
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
    except (FileNotFoundError, OSError):
        pass

    if not entries:
        return None
    # usuń duplikaty z zachowaniem kolejności
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
    """Wynik podpisanego fingerprintu (nowe API)."""

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
        statuses: dict[str, RotationStatus] = {kid: self.status_for(kid, now=now) for kid in self._keys}
        sorted_ids = sorted(statuses.items(), key=lambda item: (self._priority_tuple(item[1]), item[0]))
        active_id, status = sorted_ids[0]
        return active_id, self._keys[active_id], status

    def sign(self, payload: Mapping[str, Any], *, now: datetime | None = None) -> tuple[str, dict[str, str]]:
        key_id, key, _status = self.select_active_key(now=now)
        signature = build_hmac_signature(payload, key=key, key_id=key_id, algorithm=SIGNATURE_ALGORITHM)
        return key_id, signature


class HardwareFingerprintService:
    """Buduje deterministyczny fingerprint sprzętowy podpisany HMAC (nowe API)."""

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


# ---------------------------------------------------------------------------
# Helpery/CLI (nowe API)
# ---------------------------------------------------------------------------

def decode_secret(value: str) -> bytes:
    text = value.strip()
    if text.startswith("hex:"):
        return bytes.fromhex(text[4:])
    if text.startswith("base64:"):
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
    # OEM API
    "DeviceFingerprintGenerator",
    "FingerprintDocument",
    "FingerprintError",
    "FINGERPRINT_HASH",
    "SIGNATURE_ALGORITHM",
    "build_fingerprint_document",
    "get_local_fingerprint",
    "verify_document",
    # Nowe API
    "FingerprintRecord",
    "HardwareFingerprintService",
    "RotatingHmacKeyProvider",
    "build_key_provider",
    "decode_secret",
]
