"""Smoke test DecisionOrchestratora Etapu 5."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config.loader import load_core_config
from bot_core.decision import DecisionCandidate, DecisionOrchestrator
from bot_core.security.signing import build_hmac_signature

DEFAULT_CONFIG = REPO_ROOT / "config/core.yaml"


def _load_json(path: Path) -> Any:
    data = path.read_text(encoding="utf-8")
    return json.loads(data)


def _ensure_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} nie może być symlinkiem: {path}")
    if not path.is_file():
        raise ValueError(f"{label} nie istnieje: {path}")
    if os.name != "nt":
        mode = path.stat().st_mode
        if mode & 0o077:
            raise ValueError(f"{label} musi mieć uprawnienia maks. 600: {path}")
    return path


def _load_signing_key(args: argparse.Namespace) -> bytes | None:
    inline = args.signing_key
    env_name = args.signing_key_env
    path_value = args.signing_key_file

    provided = [value for value in (inline, env_name, path_value) if value]
    if len(provided) > 1:
        raise ValueError(
            "Podaj klucz podpisu jako --signing-key, --signing-key-env lub --signing-key-file"
        )
    if inline:
        key = inline.encode("utf-8")
    elif env_name:
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {env_name} nie zawiera klucza podpisu")
        key = value.encode("utf-8")
    elif path_value:
        candidate = _ensure_regular_file(Path(path_value).expanduser(), label="Plik klucza")
        key = candidate.read_bytes()
    else:
        return None
    if len(key) < 32:
        raise ValueError("Klucz podpisu musi mieć co najmniej 32 bajty")
    return key


def _build_signature_payload(
    *,
    output_path: Path,
    generated_at: str,
    accepted: int,
    rejected: int,
    stress_failures: int,
) -> Mapping[str, object]:
    return {
        "schema": "stage5.decision_engine.smoke",
        "schema_version": "1.0",
        "generated_at": generated_at,
        "result": output_path.name,
        "accepted": accepted,
        "rejected": rejected,
        "stress_failures": stress_failures,
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--risk-snapshot", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--tco-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--signing-key")
    parser.add_argument("--signing-key-env")
    parser.add_argument("--signing-key-file")
    parser.add_argument("--signing-key-id")
    return parser.parse_args(argv)


def _load_risk_snapshots(path: Path) -> Mapping[str, Mapping[str, object]]:
    raw = _load_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError("Plik snapshotu ryzyka musi zawierać mapowanie profil->stan")
    snapshots: dict[str, Mapping[str, object]] = {}
    for profile, payload in raw.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"Snapshot profilu {profile!r} nie jest mapą")
        snapshots[str(profile)] = dict(payload)
    return snapshots


def _load_candidates(path: Path) -> Sequence[DecisionCandidate]:
    raw = _load_json(path)
    if isinstance(raw, Mapping):
        raw_list = raw.get("candidates")
    else:
        raw_list = raw
    if not isinstance(raw_list, Sequence):
        raise ValueError("Plik kandydatów musi zawierać listę")
    candidates: list[DecisionCandidate] = []
    for entry in raw_list:
        if not isinstance(entry, Mapping):
            raise ValueError("Każdy kandydat musi być obiektem JSON")
        candidates.append(DecisionCandidate.from_mapping(entry))
    return candidates


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    config_path = args.config.expanduser()
    config = load_core_config(str(config_path))
    if not hasattr(config, "decision_engine") or config.decision_engine is None:
        raise RuntimeError("Konfiguracja nie zawiera sekcji decision_engine")

    orchestrator = DecisionOrchestrator(config.decision_engine)

    if args.tco_report:
        orchestrator.update_costs_from_report(_load_json(args.tco_report.expanduser()))

    risk_snapshots = _load_risk_snapshots(args.risk_snapshot.expanduser())
    candidates = _load_candidates(args.candidates.expanduser())

    evaluations = orchestrator.evaluate_candidates(candidates, risk_snapshots)
    accepted = [evaluation for evaluation in evaluations if evaluation.accepted]
    stress_failures = sum(1 for evaluation in evaluations if evaluation.stress_failures)

    generated_at = datetime.now(tz=timezone.utc).isoformat()
    summary = {
        "generated_at": generated_at,
        "config": config_path.as_posix(),
        "risk_snapshot": args.risk_snapshot.as_posix(),
        "tco_report": args.tco_report.as_posix() if args.tco_report else None,
        "evaluations": [evaluation.to_mapping() for evaluation in evaluations],
        "accepted": len(accepted),
        "rejected": len(evaluations) - len(accepted),
        "stress_failures": stress_failures,
    }

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    key = _load_signing_key(args)
    if key is not None:
        payload = _build_signature_payload(
            output_path=output_path,
            generated_at=generated_at,
            accepted=len(accepted),
            rejected=len(evaluations) - len(accepted),
            stress_failures=stress_failures,
        )
        signature = build_hmac_signature(
            payload,
            key=key,
            algorithm="HMAC-SHA256",
            key_id=args.signing_key_id,
        )
        signature_path = output_path.with_suffix(output_path.suffix + ".sig")
        signature_path.write_text(
            json.dumps({"payload": payload, "signature": signature}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    if accepted or args.allow_empty:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
