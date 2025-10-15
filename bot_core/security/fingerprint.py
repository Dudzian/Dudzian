"""Generowanie i podpisywanie fingerprintów urządzeń OEM."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import socket
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Optional

from bot_core.security.rotation import RotationRegistry
from bot_core.security.signing import canonical_json_bytes

FINGERPRINT_HASH = "SHA384"
SIGNATURE_ALGORITHM = "HMAC-SHA384"
_DEFAULT_PURPOSE = "oem-fingerprint-signing"


class FingerprintError(RuntimeError):
    """Wyjątek zgłaszany przy problemach z generowaniem fingerprintu."""


@dataclass(frozen=True)
class FingerprintDocument:
    """Zwracany dokument fingerprintu z podpisem HMAC."""

    payload: Mapping[str, object]
    signature: Mapping[str, object]

    def to_json(self) -> str:
        return json.dumps({"payload": self.payload, "signature": self.signature}, indent=2, ensure_ascii=False, sort_keys=True)


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
            for candidate in (
                _sanitize(entry) for entry in override.split(",")
            )
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


class DeviceFingerprintGenerator:
    """Generator fingerprintu bazujący na metadanych sprzętowych."""

    def __init__(
        self,
        *,
        env: Optional[Mapping[str, str]] = None,
        extra_probes: Optional[Iterable[Callable[[MutableMapping[str, object]], None]]] = None,
    ) -> None:
        self._env = env or os.environ
        self._extra_probes = tuple(extra_probes or ())

    # ------------------------------------------------------------------
    # Główne API
    # ------------------------------------------------------------------
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
        purpose: str = _DEFAULT_PURPOSE,
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


def verify_document(document: Mapping[str, object], *, key: bytes) -> bool:
    payload = document.get("payload")
    signature = document.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
        raise FingerprintError("Nieprawidłowa struktura dokumentu fingerprintu")
    expected = _sign_payload(payload, key, signature.get("key_id"))
    return expected["value"] == signature.get("value")


def get_local_fingerprint() -> str:
    generator = DeviceFingerprintGenerator()
    return generator.generate_fingerprint()


__all__ = [
    "DeviceFingerprintGenerator",
    "FingerprintDocument",
    "FingerprintError",
    "FINGERPRINT_HASH",
    "SIGNATURE_ALGORITHM",
    "build_fingerprint_document",
    "get_local_fingerprint",
    "verify_document",
]


def build_fingerprint_document(
    *,
    signing_key: bytes,
    key_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    registry: Optional[RotationRegistry] = None,
    purpose: str = _DEFAULT_PURPOSE,
    rotation_interval_days: float = 90.0,
    mark_rotation: bool = False,
    created_at: Optional[datetime] = None,
) -> FingerprintDocument:
    """Funkcja pomocnicza budująca podpisany dokument fingerprintu."""

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
