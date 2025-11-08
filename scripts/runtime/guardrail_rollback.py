"""Utility to toggle hedge guardrail state for runtime recovery scenarios."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.risk.repository import FileRiskRepository

DEFAULT_REPOSITORY = Path("var") / "risk_state"


def _load_profiles(directory: Path, explicit: Sequence[str] | None) -> Iterable[str]:
    if explicit:
        for name in explicit:
            yield name
        return
    if not directory.exists():
        return
    for candidate in sorted(directory.glob("*.json")):
        yield candidate.stem


def _apply_mode(
    state: MutableMapping[str, object],
    *,
    mode: str,
    reason: str | None,
    clear_force: bool,
    now: datetime,
) -> bool:
    changed = False
    if mode == "disable":
        if state.get("hedge_mode"):
            state["hedge_mode"] = False
            changed = True
        if state.get("hedge_reason"):
            state["hedge_reason"] = None
            changed = True
        if state.get("hedge_activated_at"):
            state["hedge_activated_at"] = None
            changed = True
        if state.get("hedge_cooldown_until"):
            state["hedge_cooldown_until"] = None
            changed = True
        if clear_force and state.get("force_liquidation"):
            state["force_liquidation"] = False
            changed = True
        return changed

    # mode == "enable"
    activation_reason = reason or "manual-rollback"
    timestamp = now.astimezone(timezone.utc).isoformat()
    if not state.get("hedge_mode"):
        state["hedge_mode"] = True
        changed = True
    if state.get("hedge_reason") != activation_reason:
        state["hedge_reason"] = activation_reason
        changed = True
    if state.get("hedge_activated_at") != timestamp:
        state["hedge_activated_at"] = timestamp
        changed = True
    if state.get("hedge_cooldown_until"):
        state["hedge_cooldown_until"] = None
        changed = True
    if clear_force and state.get("force_liquidation"):
        state["force_liquidation"] = False
        changed = True
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rollback hedge guardrail state for runtime profiles")
    parser.add_argument(
        "--repository",
        type=Path,
        default=DEFAULT_REPOSITORY,
        help="Ścieżka do katalogu z zapisanym stanem profili ryzyka (domyślnie var/risk_state)",
    )
    parser.add_argument(
        "--profile",
        action="append",
        help="Nazwa profilu do aktualizacji (można wskazać wielokrotnie). Bez parametru operacja obejmuje wszystkie profile.",
    )
    parser.add_argument(
        "--mode",
        choices=("disable", "enable"),
        default="disable",
        help="Tryb operacji guardraila (disable = rollback do trybu normalnego, enable = wymuszenie trybu hedge).",
    )
    parser.add_argument(
        "--reason",
        help="Powód zapisany w stanie guardraila (dla trybu enable).",
    )
    parser.add_argument(
        "--clear-force-liquidation",
        action="store_true",
        help="Resetuje również flagę force_liquidation podczas rollbacku.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Symuluje operację bez zapisu zmian.",
    )

    args = parser.parse_args(argv)

    repository = FileRiskRepository(args.repository)
    processed = 0
    updated = 0
    now = datetime.now(timezone.utc)

    for profile in _load_profiles(args.repository, args.profile):
        processed += 1
        payload = repository.load(profile)
        if not isinstance(payload, Mapping):
            print(f"[warn] pominięto profil {profile} – brak stanu")
            continue
        state: MutableMapping[str, object] = dict(payload)
        if _apply_mode(
            state,
            mode=args.mode,
            reason=args.reason,
            clear_force=args.clear_force_liquidation,
            now=now,
        ):
            updated += 1
            if args.dry_run:
                print(f"[dry-run] {profile}: {json.dumps(state, ensure_ascii=False)}")
            else:
                repository.store(profile, state)
                print(f"[ok] zaktualizowano profil {profile}")
        else:
            print(f"[noop] profil {profile} już w docelowym stanie")

    if processed == 0:
        print("[warn] nie znaleziono żadnych profili do aktualizacji")
    else:
        print(f"[info] przetworzono {processed} profili, zaktualizowano {updated}")
    return 0 if processed > 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
