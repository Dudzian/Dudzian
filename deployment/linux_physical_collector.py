"""Producent podpisanych faktów; domyślnie wyłącznie diagnostyka odczytowa.

Nigdy nie uruchamia restartu hosta. Każda faza zmieniająca instalację wymaga
osobnej zgody obejmującej dokładny identyfikator challenge i nazwę fazy.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deployment.linux_physical_evidence import (
    LinuxEvidenceError,
    PHASES,
    canonical_bytes,
    digest,
    load_challenge,
    read_json,
    sign_record,
    validate_challenge,
    validate_record_chain,
)
from deployment.linux_physical_native import (
    NativeDriver,
    command,
    file_observation,
    host_identity,
    read_regular,
)
from deployment.linux_physical_observations import (
    ObservationError,
    require,
    sha256,
    validate_installation,
    validate_observations,
    validate_observation_prefix,
)

MUTATING_PHASES = frozenset({"clean_install", "systemd_lifecycle", "update", "failed_update"})
CLOSURE = (
    "deployment/linux_physical_collector.py",
    "deployment/linux_physical_native.py",
    "deployment/linux_physical_observations.py",
    "deployment/linux_physical_evidence.py",
    "deployment/linux_physical_ipc.py",
    "deployment/linux_physical_postgresql.py",
    "deployment/linux_release_identity.py",
    "bot_core/production_signer_ipc.py",
)


def collect_materials(installation: dict[str, Any]) -> dict[str, str]:
    source = Path(installation["source_root"])
    revision = command(["/usr/bin/git", "-C", str(source), "rev-parse", "HEAD"])
    status = command(
        ["/usr/bin/git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"]
    )
    require(
        revision.returncode == status.returncode == 0 and not status.stdout.strip(),
        "Źródła muszą mieć czyste drzewo dokładnej rewizji",
    )
    # Wykonywany moduł i wszystkie lokalne zależności muszą pochodzić z tego
    # samego, chronionego checkoutu, którego rewizja jest kwalifikowana.
    require(
        Path(__file__).resolve() == (source / CLOSURE[0]).resolve(),
        "Wykonywany producent pochodzi z innego checkoutu",
    )
    closure = {name: sha256(read_regular(source / name, protected=True)) for name in CLOSURE}
    artifacts = {}
    for kind, item in installation["packages"].items():
        actual = sha256(read_regular(Path(item["path"]), protected=True))
        require(actual == item["sha256"], "Artefakt różni się od kontraktu")
        artifacts[kind] = actual
    configuration = {
        item["path"]: file_observation(Path(item["path"]))["sha256"]
        for item in installation["configuration"]
    }
    require(
        configuration == {item["path"]: item["sha256"] for item in installation["configuration"]},
        "Konfiguracja różni się od kontraktu",
    )
    return {
        "source_revision": revision.stdout.strip(),
        "collector_sha256": digest(closure),
        "artifact_sha256": artifacts["baseline"],
        "update_artifact_sha256": artifacts["update"],
        "failed_update_artifact_sha256": artifacts["failed_update"],
        "configuration_sha256": digest(configuration),
        "installation_contract_sha256": digest(installation),
    }


def _atomic_bundle(path: Path, bundle: dict[str, Any]) -> None:
    require(path.is_absolute(), "Wymagana bezwzględna ścieżka paczki dowodowej")
    temporary = path.with_name(path.name + ".pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(bundle) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def collect_phase(
    *,
    challenge: dict[str, Any],
    installation: dict[str, Any],
    bundle: dict[str, Any],
    key_bytes: bytes,
    phase: str,
    approval: str | None,
    driver: NativeDriver | None = None,
) -> dict[str, Any]:
    """Sprawdź zgodę i ciągłość przed przekazaniem sterowania natywnemu producentowi."""
    validate_challenge(challenge)
    validate_installation(installation)
    require(phase in PHASES, "Nieznana faza")
    require(
        phase not in MUTATING_PHASES or approval == challenge["challenge_id"] + ":" + phase,
        "Brak osobnej świadomej zgody na wskazaną fazę",
    )
    require(
        type(key_bytes) is bytes and len(key_bytes) == 32,
        "Nieprawidłowy dedykowany klucz producenta",
    )
    public = (
        Ed25519PrivateKey.from_private_bytes(key_bytes)
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    identity = host_identity()
    require(identity["host_id"] == challenge["host_id"], "Challenge dotyczy innego hosta")
    approved = {**identity, "public_key": base64.b64encode(public).decode("ascii")}
    existing = validate_record_chain(bundle, challenge, approved, complete=False)
    validate_observation_prefix(existing, installation)
    require(
        len(existing) < len(PHASES) and PHASES[len(existing)] == phase,
        "Faza została pominięta albo już wykonana",
    )
    if existing:
        stable = lambda item: {k: v for k, v in item.items() if k not in {"boot_id", "kernel"}}
        require(stable(identity) == stable(existing[0]["host"]), "Utracono ciągłość hosta")
        prior_boot = existing[-1]["host"]["boot_id"]
        require(
            (identity["boot_id"] != prior_boot)
            if phase == "post_reboot"
            else (identity["boot_id"] == prior_boot),
            "Niewłaściwy rozruch dla fazy",
        )
    materials = collect_materials(installation)
    require(materials == challenge["materials"], "Materiały nie należą do zatwierdzonego challenge")
    # Wstrzyknięty driver występuje wyłącznie w hermetycznych testach jednostkowych;
    # CLI zawsze tworzy natywny backend i nie obsługuje importów ani komend od użytkownika.
    if driver is None and phase in MUTATING_PHASES:
        # Sam podpisany manifest i konto root nie uprawniają do uruchomienia
        # kwalifikacji pozorującej brakujące testy rzeczywistych usług.
        from deployment.linux_physical_ipc import (
            blocked_ipc_qualification,
            require_ipc_qualification,
        )

        require_ipc_qualification(blocked_ipc_qualification())
    native = driver if driver is not None else NativeDriver(installation, challenge["challenge_id"])
    observations = native.collect(phase, bundle["records"])
    validate_challenge(challenge)
    identity_after = host_identity()
    require(identity_after == identity, "Host zmienił się podczas fazy")
    require(collect_materials(installation) == materials, "Materiały zmieniły się podczas fazy")
    payload = {
        "schema_version": 1,
        "challenge_id": challenge["challenge_id"],
        "sequence": len(existing),
        "phase": phase,
        "previous_sha256": digest(bundle["records"][-1]) if existing else "0" * 64,
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "host": identity,
        "materials": materials,
        "observations": observations,
    }
    result = {**bundle, "records": [*bundle["records"], sign_record(payload, key_bytes)]}
    validated = validate_record_chain(result, challenge, approved, complete=phase == PHASES[-1])
    validate_observation_prefix(validated, installation)
    if phase == PHASES[-1]:
        validate_observations(validated, installation)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Podpisany producent fizycznych obserwacji Linux")
    parser.add_argument("--diagnose", action="store_true", help="Tylko odczyt tożsamości hosta")
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--approve", help="Osobna zgoda: <challenge_id>:<phase>")
    args = parser.parse_args(argv)
    try:
        if args.diagnose or args.phase is None:
            print(json.dumps({"tryb": "TYLKO_ODCZYT", "host": host_identity()}, sort_keys=True))
            return 0
        require(
            all((args.challenge, args.manifest, args.bundle, args.private_key)),
            "Wymagane challenge, manifest, paczka i dedykowany klucz producenta",
        )
        installation = validate_installation(read_json(args.manifest))
        challenge = load_challenge(args.challenge)
        if args.bundle.exists():
            bundle = read_json(args.bundle)
        else:
            bundle = {"schema_version": 1, "challenge_id": challenge["challenge_id"], "records": []}
        key_info = args.private_key.lstat()
        require(
            key_info.st_uid == 0 and key_info.st_mode & 0o077 == 0,
            "Dedykowany klucz producenta wymaga wyłącznej ochrony administratora",
        )
        key = read_regular(args.private_key, protected=True, maximum=32)
        result = collect_phase(
            challenge=challenge,
            installation=installation,
            bundle=bundle,
            key_bytes=key,
            phase=args.phase,
            approval=args.approve,
        )
        _atomic_bundle(args.bundle, result)
    except (LinuxEvidenceError, ObservationError, OSError, ValueError, KeyError, TypeError) as exc:
        # Nie ujawniamy stderr narzędzi, DSN, treści plików ani danych klucza.
        print(
            "ODMOWA: nie można bezpiecznie zebrać fazy; sprawdź zgodę, kontrakt i chronione materiały"
        )
        return 1
    print("Zapisano podpisaną obserwację; kwalifikacja pozostaje odrębną decyzją CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
