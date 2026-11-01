"""Podpisana, ograniczona do wyzwania CI ścieżka fizycznych dowodów Linux.

Podpis uwierzytelnia zatwierdzonego producenta, nie zastępuje zaufania do
administratora hosta. Polityka i wyzwanie muszą pochodzić spoza paczki dowodów.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

PHASES = (
    "baseline",
    "clean_install",
    "systemd_lifecycle",
    "pre_reboot",
    "post_reboot",
    "update",
    "failed_update",
)
LINUX_ITEMS = (
    "LINUX_SYSTEMD_INTEGRATION",
    "LINUX_CLEAN_INSTALL",
    "LINUX_REBOOT_LIFECYCLE_UPDATE",
)
MATERIAL_KEYS = {
    "source_revision",
    "collector_sha256",
    "artifact_sha256",
    "update_artifact_sha256",
    "failed_update_artifact_sha256",
    "configuration_sha256",
    "installation_contract_sha256",
}
HOST_FIELDS = {
    "host_id",
    "machine_id_sha256",
    "dmi_product_uuid_sha256",
    "boot_id",
    "os_id",
    "os_version",
    "kernel",
    "architecture",
    "virtualization",
    "python_version",
    "python_executable_sha256",
}
PAYLOAD_FIELDS = {
    "schema_version",
    "challenge_id",
    "sequence",
    "phase",
    "previous_sha256",
    "observed_at_utc",
    "host",
    "materials",
    "observations",
}
CHALLENGE_FIELDS = {
    "schema_version",
    "purpose",
    "repository",
    "source_revision",
    "ci_run_id",
    "ci_run_attempt",
    "host_id",
    "policy_sha256",
    "materials",
    "nonce",
    "created_at_utc",
    "expires_at_utc",
    "challenge_id",
}
DEFAULT_POLICY = Path(__file__).with_name("linux_physical_trust_policy.json")
MAX_JSON_BYTES = 256 * 1024


class LinuxEvidenceError(ValueError):
    """Dowody nie uprawniają do kwalifikacji."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise LinuxEvidenceError("Powtórzone pole JSON")
        value[name] = item
    return value


def _constant(value: str) -> Any:
    raise LinuxEvidenceError("Nieskończona lub nieokreślona liczba JSON")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise LinuxEvidenceError("Przekroczony limit dokumentu")
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise LinuxEvidenceError("Nie można odczytać poprawnego dokumentu JSON") from exc
    if not isinstance(value, dict):
        raise LinuxEvidenceError("Wymagany obiekt JSON")
    return value


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise LinuxEvidenceError("Niekanoniczna treść dowodu") from exc


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _hex(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _time(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LinuxEvidenceError("Wymagana data UTC zakończona Z")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
        if result.tzinfo is None or result.utcoffset() != timedelta(0):
            raise ValueError
    except ValueError as exc:
        raise LinuxEvidenceError("Nieprawidłowa data dowodu") from exc
    return result


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _materials(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != MATERIAL_KEYS
        or any(
            not _hex(item, 40 if name == "source_revision" else 64) for name, item in value.items()
        )
    ):
        raise LinuxEvidenceError("Niepełne przypięcie kodu, artefaktów lub konfiguracji")
    if (
        len(
            {
                value[name]
                for name in (
                    "artifact_sha256",
                    "update_artifact_sha256",
                    "failed_update_artifact_sha256",
                )
            }
        )
        != 3
    ):
        raise LinuxEvidenceError("Wymagane trzy różne artefakty instalacji i aktualizacji")


def _decode(value: object, length: int) -> bytes:
    try:
        if not isinstance(value, str):
            raise ValueError
        result = base64.b64decode(value, validate=True)
        if len(result) != length or base64.b64encode(result).decode("ascii") != value:
            raise ValueError
        return result
    except (ValueError, binascii.Error) as exc:
        raise LinuxEvidenceError("Nieprawidłowy klucz publiczny lub podpis") from exc


def load_policy(path: Path, expected_sha256: str) -> dict[str, Any]:
    policy = read_json(path)
    if not _hex(expected_sha256) or digest(policy) != expected_sha256:
        raise LinuxEvidenceError("Polityka różni się od niezależnie zatwierdzonego skrótu")
    return validate_policy(policy)


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Sprawdza strukturę; zaufanie do tego dokumentu nadaje oddzielny pin CI."""
    if (
        set(policy)
        not in (
            {"schema_version", "purpose", "max_age_seconds", "hosts"},
            {"schema_version", "purpose", "max_age_seconds", "hosts", "release_authorities"},
        )
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or policy["purpose"] != "LIVE_PHYSICAL_LINUX"
        or type(policy["max_age_seconds"]) is not int
        or not 1 <= policy["max_age_seconds"] <= 86400
        or not isinstance(policy["hosts"], list)
        or len(policy["hosts"]) > 32
    ):
        raise LinuxEvidenceError("Nieprawidłowa polityka zaufania")
    seen: set[str] = set()
    seen_keys: set[str] = set()
    fields = {
        "host_id",
        "public_key",
        "machine_id_sha256",
        "dmi_product_uuid_sha256",
        "python_executable_sha256",
        "materials",
        "installation",
    }
    for host in policy["hosts"]:
        if (
            not isinstance(host, dict)
            or set(host) != fields
            or any(
                not _hex(host[name])
                for name in fields - {"public_key", "materials", "installation"}
            )
            or not isinstance(host["public_key"], str)
            or host["host_id"] in seen
            or host["public_key"] in seen_keys
            or not isinstance(host["installation"], dict)
        ):
            raise LinuxEvidenceError("Niejednoznaczny lub nieprawidłowy zatwierdzony host")
        _decode(host["public_key"], 32)
        _materials(host["materials"])
        if digest(host["installation"]) != host["materials"]["installation_contract_sha256"]:
            raise LinuxEvidenceError("Kontrakt instalacji różni się od zatwierdzonego")
        from deployment.linux_physical_observations import validate_installation

        try:
            installation = validate_installation(host["installation"])
        except (ValueError, TypeError, KeyError) as exc:
            raise LinuxEvidenceError("Niepoprawny zatwierdzony kontrakt instalacji") from exc
        for name, phase in (
            ("artifact_sha256", "baseline"),
            ("update_artifact_sha256", "update"),
            ("failed_update_artifact_sha256", "failed_update"),
        ):
            if host["materials"][name] != installation["packages"][phase]["sha256"]:
                raise LinuxEvidenceError("Niespójne przypięcie artefaktu i instalacji")
        configuration = {item["path"]: item["sha256"] for item in installation["configuration"]}
        if digest(configuration) != host["materials"]["configuration_sha256"]:
            raise LinuxEvidenceError("Niespójne przypięcie konfiguracji")
        seen.add(host["host_id"])
        seen_keys.add(host["public_key"])
    from deployment.linux_release_identity import validate_release_authorities

    try:
        validate_release_authorities(policy.get("release_authorities", []), producer_keys=seen_keys)
    except ValueError as exc:
        raise LinuxEvidenceError("Nieprawidłowy niezależny rejestr autorytetów wydania") from exc
    return policy


def _approved_host(policy: dict[str, Any], host_id: str) -> dict[str, Any]:
    matches = [host for host in policy["hosts"] if host["host_id"] == host_id]
    if len(matches) != 1:
        raise LinuxEvidenceError("Host nie został zatwierdzony; domyślna lista zaufania jest pusta")
    return cast(dict[str, Any], matches[0])


def load_challenge(path: Path) -> dict[str, Any]:
    value = read_json(path)
    validate_challenge(value)
    return value


def validate_challenge(
    value: dict[str, Any],
    *,
    now: datetime | None = None,
) -> None:
    if (
        set(value) != CHALLENGE_FIELDS
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["purpose"] != "LIVE_PHYSICAL_LINUX"
        or value["repository"] != "Dudzian/Dudzian"
        or not _hex(value["source_revision"], 40)
        or any(
            not _hex(value[name]) for name in ("host_id", "policy_sha256", "nonce", "challenge_id")
        )
        or any(
            not isinstance(value[name], str) or not re.fullmatch(r"[1-9][0-9]*", value[name])
            for name in ("ci_run_id", "ci_run_attempt")
        )
        or digest({k: v for k, v in value.items() if k != "challenge_id"}) != value["challenge_id"]
    ):
        raise LinuxEvidenceError("Nieprawidłowe wyzwanie CI")
    _materials(value["materials"])
    if value["materials"]["source_revision"] != value["source_revision"]:
        raise LinuxEvidenceError("Rewizja materiałów różni się od kwalifikowanej")
    created, expires = _time(value["created_at_utc"]), _time(value["expires_at_utc"])
    current = now or datetime.now(timezone.utc)
    if not created <= current <= expires or not timedelta(0) < expires - created <= timedelta(
        days=1
    ):
        raise LinuxEvidenceError("Wyzwanie przestarzałe, przyszłe lub zbyt długie")


def issue_challenge(
    policy: dict[str, Any],
    *,
    source_revision: str,
    ci_run_id: str,
    ci_run_attempt: str,
    host_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_policy(policy)
    host = _approved_host(policy, host_id)
    created = now or datetime.now(timezone.utc)
    value = {
        "schema_version": 1,
        "purpose": "LIVE_PHYSICAL_LINUX",
        "repository": "Dudzian/Dudzian",
        "source_revision": source_revision,
        "ci_run_id": ci_run_id,
        "ci_run_attempt": ci_run_attempt,
        "host_id": host_id,
        "policy_sha256": digest(policy),
        "materials": host["materials"],
        "nonce": secrets.token_hex(32),
        "created_at_utc": _stamp(created),
        "expires_at_utc": _stamp(created + timedelta(seconds=policy["max_age_seconds"])),
    }
    value["challenge_id"] = digest(value)
    validate_challenge(value, now=created)
    return value


def sign_record(payload: dict[str, Any], key_bytes: bytes) -> dict[str, Any]:
    if len(key_bytes) != 32:
        raise LinuxEvidenceError("Wymagany dedykowany surowy klucz Ed25519 o długości 32 bajtów")
    signature = Ed25519PrivateKey.from_private_bytes(key_bytes).sign(canonical_bytes(payload))
    return {"payload": payload, "signature": base64.b64encode(signature).decode("ascii")}


def validate_record_chain(
    bundle: dict[str, Any],
    challenge: dict[str, Any],
    host: dict[str, Any],
    *,
    complete: bool = True,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    if (
        set(bundle) != {"schema_version", "challenge_id", "records"}
        or type(bundle["schema_version"]) is not int
        or bundle["schema_version"] != 1
        or bundle["challenge_id"] != challenge["challenge_id"]
        or not isinstance(bundle["records"], list)
        or len(bundle["records"]) > len(PHASES)
        or (complete and len(bundle["records"]) != len(PHASES))
    ):
        raise LinuxEvidenceError("Brak kompletnego łańcucha fizycznych etapów")
    key = Ed25519PublicKey.from_public_bytes(_decode(host["public_key"], 32))
    previous = "0" * 64
    previous_time = _time(challenge["created_at_utc"])
    current = now or datetime.now(timezone.utc)
    payloads = []
    first_identity = None
    for sequence, record in enumerate(bundle["records"]):
        if not isinstance(record, dict) or set(record) != {"payload", "signature"}:
            raise LinuxEvidenceError("Nieprawidłowa koperta podpisu")
        payload = record["payload"]
        if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
            raise LinuxEvidenceError("Nieprawidłowe pola obserwacji")
        try:
            key.verify(_decode(record["signature"], 64), canonical_bytes(payload))
        except InvalidSignature as exc:
            raise LinuxEvidenceError("Podmieniony dowód lub niezatwierdzony producent") from exc
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or type(payload["sequence"]) is not int
            or payload["sequence"] != sequence
            or payload["phase"] != PHASES[sequence]
            or payload["previous_sha256"] != previous
            or payload["challenge_id"] != challenge["challenge_id"]
            or payload["materials"] != challenge["materials"]
        ):
            raise LinuxEvidenceError("Zmieniona kolejność, rewizja lub materiały obserwacji")
        observed = _time(payload["observed_at_utc"])
        if not previous_time <= observed <= min(current, _time(challenge["expires_at_utc"])):
            raise LinuxEvidenceError("Niespójny czas obserwacji")
        identity = payload["host"]
        if (
            not isinstance(identity, dict)
            or set(identity) != HOST_FIELDS
            or any(not isinstance(v, str) or not v for v in identity.values())
            or any(
                identity[name] != host[name]
                for name in (
                    "host_id",
                    "machine_id_sha256",
                    "dmi_product_uuid_sha256",
                    "python_executable_sha256",
                )
            )
            or identity["os_id"] != "ubuntu"
            or identity["os_version"] != "24.04"
            or identity["architecture"] != "x86_64"
            or identity["virtualization"] != "none"
            or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", identity["boot_id"])
            or not isinstance(payload["observations"], dict)
        ):
            raise LinuxEvidenceError("Niezatwierdzone środowisko lub niespójna tożsamość hosta")
        stable = {
            name: value for name, value in identity.items() if name not in {"boot_id", "kernel"}
        }
        if first_identity is None:
            first_identity = stable
        elif stable != first_identity:
            raise LinuxEvidenceError("Utrata ciągłości tożsamości hosta lub interpretera")
        previous_time, previous = observed, digest(record)
        payloads.append(payload)
    if payloads:
        old = [item["host"]["boot_id"] for item in payloads[:4]]
        new = [item["host"]["boot_id"] for item in payloads[4:]]
        if len(set(old)) != 1 or (new and (len(set(new)) != 1 or old[0] == new[0])):
            raise LinuxEvidenceError("Brak jednoznacznego restartu i ciągłości etapów")
    return payloads


def verify_bundle(
    bundle: dict[str, Any],
    challenge: dict[str, Any],
    policy: dict[str, Any],
    *,
    source_revision: str,
    ci_run_id: str,
    ci_run_attempt: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    validate_policy(policy)
    validate_challenge(challenge, now=now)
    if (
        challenge["source_revision"] != source_revision
        or challenge["ci_run_id"] != ci_run_id
        or challenge["ci_run_attempt"] != ci_run_attempt
        or challenge["policy_sha256"] != digest(policy)
        or _time(challenge["expires_at_utc"]) - _time(challenge["created_at_utc"])
        > timedelta(seconds=policy["max_age_seconds"])
    ):
        raise LinuxEvidenceError("Dowody pochodzą z innej rewizji, próby CI lub polityki")
    host = _approved_host(policy, challenge["host_id"])
    if challenge["materials"] != host["materials"]:
        raise LinuxEvidenceError("Artefakty nie zostały zatwierdzone przez niezależną politykę")
    payloads = validate_record_chain(bundle, challenge, host, now=now)
    from deployment.linux_physical_observations import validate_observations

    try:
        validate_observations(payloads, host["installation"])
        from deployment.linux_release_identity import verify_release_identity
        from deployment.linux_physical_ipc import require_ipc_qualification
        from deployment.linux_physical_postgresql import require_service_qualification

        # Podpis hosta nie nadaje dowolnemu pakietowi tożsamości CryptoHunter.
        verify_release_identity(
            host["installation"],
            policy,
            source_revision=source_revision,
            materials=challenge["materials"],
        )
        for payload in payloads[1:]:
            for name in ("snapshot", "before", "after", "started", "restarted", "crashed"):
                snapshot = payload["observations"].get(name)
                if snapshot is None:
                    continue
                require_ipc_qualification(snapshot["ipc_qualification"])
                require_service_qualification(snapshot["postgres"]["service_qualification"])
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        raise LinuxEvidenceError("Surowe obserwacje nie potwierdzają wymaganych testów") from exc
    return {
        "schema_version": 1,
        "platform": "LINUX",
        "source_revision": source_revision,
        "ci_provider": "https://github.com",
        "ci_run_id": ci_run_id,
        "runner_os": "Linux",
        "runner_arch": "x86_64",
        "generated_at_utc": payloads[-1]["observed_at_utc"],
        "linux_physical": {
            "challenge_id": challenge["challenge_id"],
            "bundle_sha256": digest(bundle),
            "policy_sha256": digest(policy),
            "host_id": host["host_id"],
            "ci_run_attempt": ci_run_attempt,
        },
        "results": [
            {
                "item": item,
                "status": "PASS",
                "evidence_class": "LIVE_LINUX_INTEGRATION",
                "test_or_probe": "stage9-linux-physical-v1",
                "details": "Podpisane obserwacje fizycznego hosta; niezależna walidacja semantyczna",
            }
            for item in LINUX_ITEMS
        ],
    }


def verify_functional_diagnostic(
    record: dict[str, Any],
    challenge: dict[str, Any],
    policy: dict[str, Any],
    *,
    source_revision: str,
    ci_run_id: str,
    ci_run_attempt: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Zweryfikuj częściową diagnostykę w osobnej domenie, bez wyniku PASS."""
    validate_policy(policy)
    validate_challenge(challenge, now=now)
    if (
        challenge["source_revision"] != source_revision
        or challenge["ci_run_id"] != ci_run_id
        or challenge["ci_run_attempt"] != ci_run_attempt
        or challenge["policy_sha256"] != digest(policy)
        or _time(challenge["expires_at_utc"]) - _time(challenge["created_at_utc"])
        > timedelta(seconds=policy["max_age_seconds"])
    ):
        raise LinuxEvidenceError("Diagnostyka pochodzi z innego wyzwania lub polityki")
    host = _approved_host(policy, challenge["host_id"])
    if (
        challenge["materials"] != host["materials"]
        or type(record) is not dict
        or set(record) != {"payload", "signature"}
    ):
        raise LinuxEvidenceError("Nieprawidłowa koperta częściowej diagnostyki")
    payload = record["payload"]
    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "kind",
        "challenge_id",
        "observed_at_utc",
        "host",
        "materials",
        "observations",
    }:
        raise LinuxEvidenceError("Nieprawidłowe pola częściowej diagnostyki")
    try:
        key = Ed25519PublicKey.from_public_bytes(_decode(host["public_key"], 32))
        key.verify(_decode(record["signature"], 64), canonical_bytes(payload))
    except InvalidSignature as exc:
        raise LinuxEvidenceError("Podmieniona częściowa diagnostyka") from exc
    identity = payload["host"]
    current = now or datetime.now(timezone.utc)
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["kind"] != "STAGE9_PARTIAL_VERIFIER_DIAGNOSTIC"
        or payload["challenge_id"] != challenge["challenge_id"]
        or payload["materials"] != challenge["materials"]
        or not _time(challenge["created_at_utc"])
        <= _time(payload["observed_at_utc"])
        <= min(current, _time(challenge["expires_at_utc"]))
        or type(identity) is not dict
        or set(identity) != HOST_FIELDS
        or any(type(value) is not str or not value for value in identity.values())
        or any(
            identity[name] != host[name]
            for name in (
                "host_id",
                "machine_id_sha256",
                "dmi_product_uuid_sha256",
                "python_executable_sha256",
            )
        )
        or identity["os_id"] != "ubuntu"
        or identity["os_version"] != "24.04"
        or identity["architecture"] != "x86_64"
        or identity["virtualization"] != "none"
        or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", identity["boot_id"]) is None
    ):
        raise LinuxEvidenceError("Diagnostyka nie jest związana z zatwierdzonym hostem i wyzwaniem")
    from deployment.linux_physical_observations import validate_functional_observations

    try:
        validate_functional_observations(
            payload["observations"], host["installation"], challenge_id=challenge["challenge_id"]
        )
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        raise LinuxEvidenceError(
            "Surowe obserwacje nie potwierdzają częściowej diagnostyki"
        ) from exc
    return {
        "schema_version": 1,
        "status": "PARTIAL_BLOCKED",
        "challenge_id": challenge["challenge_id"],
        "record_sha256": digest(record),
        "production_qualification": "BLOCKED",
    }


def consume_challenge(path: Path, challenge_id: str, bundle_sha256: str) -> None:
    """Atomowe zużycie; ledger musi leżeć w chronionej przestrzeni bramki."""
    if not _hex(challenge_id) or not _hex(bundle_sha256):
        raise LinuxEvidenceError("Nieprawidłowy identyfikator zużycia")
    try:
        with sqlite3.connect(path, timeout=10) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS consumed (challenge_id TEXT PRIMARY KEY, bundle_sha256 TEXT NOT NULL)"
            )
            db.execute("INSERT INTO consumed VALUES (?, ?)", (challenge_id, bundle_sha256))
    except sqlite3.Error as exc:
        raise LinuxEvidenceError("Dowód już wykorzystany albo rejestr zużycia niedostępny") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Niezależna walidacja fizycznych dowodów Linux")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("issue-challenge", "verify", "verify-diagnostic"):
        command = sub.add_parser(name)
        command.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
        command.add_argument("--expected-policy-sha256", required=True)
        command.add_argument("--source-revision", required=True)
        command.add_argument("--ci-run-id", required=True)
        command.add_argument("--ci-run-attempt", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "issue-challenge":
            command.add_argument("--host-id", required=True)
        else:
            command.add_argument("--bundle", type=Path, required=True)
            command.add_argument("--challenge", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy, args.expected_policy_sha256)
        identity = {
            "source_revision": args.source_revision,
            "ci_run_id": args.ci_run_id,
            "ci_run_attempt": args.ci_run_attempt,
        }
        if args.command == "issue-challenge":
            result = issue_challenge(policy, host_id=args.host_id, **identity)
        else:
            verifier = (
                verify_functional_diagnostic
                if args.command == "verify-diagnostic"
                else verify_bundle
            )
            result = verifier(
                read_json(args.bundle), load_challenge(args.challenge), policy, **identity
            )
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    except (LinuxEvidenceError, OSError, TypeError, KeyError) as exc:
        print(f"ODMOWA: {exc}")
        return 1
    print(
        "Zapisano wyzwanie CI"
        if args.command == "issue-challenge"
        else "Zweryfikowano dowody; decyzja bramki jest odrębna"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
