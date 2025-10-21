"""Uruchamia scheduler multi-strategy zgodnie z konfiguracją core."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from bot_core.exchanges.base import ExchangeAdapterFactory
from bot_core.runtime.bootstrap import parse_adapter_factory_cli_specs
from bot_core.runtime.pipeline import (
    MultiStrategyRuntime,
    build_multi_strategy_runtime,
    resolve_capital_policy_spec,
)
from bot_core.security import SecretManager, create_default_secret_storage
from bot_core.security.guards import LicenseCapabilityError
from bot_core.security.license import LicenseValidationError


LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - zależne od opcjonalnego PyYAML
    import yaml
except Exception:  # pragma: no cover - brak PyYAML
    yaml = None  # type: ignore[assignment]

_DURATION_PATTERN = re.compile(r"^(?P<value>[-+]?[0-9]*\.?[0-9]+)\s*(?P<unit>[smhdSMHD]?)$")


def _to_serializable(value: Any) -> Any:
    """Konwertuje wartość do formatu zgodnego z JSON."""

    if isinstance(value, Mapping):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _parse_duration_argument(value: str) -> float:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("Wartość czasu trwania nie może być pusta")
    match = _DURATION_PATTERN.match(text)
    if match:
        amount = float(match.group("value"))
        unit = match.group("unit").lower()
        if unit in ("", "s"):
            return amount
        if unit == "m":
            return amount * 60.0
        if unit == "h":
            return amount * 3600.0
        if unit == "d":
            return amount * 86400.0
        return amount
    try:
        return float(text)
    except ValueError as exc:  # pragma: no cover - błąd parsowania
        raise argparse.ArgumentTypeError(
            f"Niepoprawny format czasu trwania: {value!r}"
        ) from exc


def _parse_datetime_argument(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("Wartość czasu nie może być pusta")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - błąd parsowania
        raise argparse.ArgumentTypeError(
            f"Niepoprawny format czasu (ISO 8601 wymagany): {value!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_signal_limit_target(value: str) -> tuple[str, str]:
    text = value.strip()
    if not text:
        raise ValueError("Specyfikacja limitu sygnałów nie może być pusta")
    separator = ":" if ":" in text else ("/" if "/" in text else None)
    if separator is None:
        raise ValueError(
            "Oczekiwano formatu STRATEGIA:PROFIL lub STRATEGIA/PROFIL"
        )
    strategy, profile = text.split(separator, 1)
    strategy = strategy.strip()
    profile = profile.strip()
    if not strategy or not profile:
        raise ValueError(
            "Specyfikacja limitu musi zawierać nazwę strategii i profilu"
        )
    return strategy, profile


def _parse_signal_limit_assignment(value: str) -> tuple[str, str, int]:
    text = value.strip()
    if "=" not in text:
        raise ValueError(
            "Oczekiwano formatu STRATEGIA:PROFIL=LIMIT przy ustawianiu limitu"
        )
    target, limit_text = text.split("=", 1)
    strategy, profile = _parse_signal_limit_target(target)
    limit_str = limit_text.strip()
    if not limit_str:
        raise ValueError("Limit sygnałów nie może być pusty")
    try:
        limit = int(float(limit_str))
    except ValueError as exc:
        raise ValueError(
            f"Niepoprawna wartość limitu sygnałów: {limit_str!r}"
        ) from exc
    return strategy, profile, limit


def _print_suspension_snapshot(snapshot: Mapping[str, Mapping[str, object]]) -> None:
    schedules = snapshot.get("schedules", {}) if isinstance(snapshot, Mapping) else {}
    tags = snapshot.get("tags", {}) if isinstance(snapshot, Mapping) else {}
    if not schedules and not tags:
        print("Brak aktywnych zawieszeń.")
        return
    if schedules:
        print("Zawieszenia harmonogramów:")
        for name in sorted(schedules):
            entry = schedules.get(name, {})
            if isinstance(entry, Mapping):
                reason = entry.get("reason", "brak powodu")
                until = entry.get("until") or "bez terminu"
            else:
                reason = "brak powodu"
                until = "bez terminu"
            print(f"  - {name}: powód={reason}, do={until}")
    if tags:
        print("Zawieszenia tagów:")
        for tag in sorted(tags):
            entry = tags.get(tag, {})
            if isinstance(entry, Mapping):
                reason = entry.get("reason", "brak powodu")
                until = entry.get("until") or "bez terminu"
            else:
                reason = "brak powodu"
                until = "bez terminu"
            print(f"  - {tag}: powód={reason}, do={until}")


def _export_payload(
    path_text: str,
    payload: Mapping[str, object],
    *,
    description: str,
) -> None:
    """Zapisuje dane zarządzania schedulerem do pliku JSON."""

    target = Path(path_text).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = _to_serializable(payload)
        target.write_text(
            json.dumps(serialized, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - rzadkie błędy IO
        LOGGER.error("Nie udało się zapisać %s do %s: %s", description, target, exc)
        raise
    else:
        print(f"Zapisano {description} do {target}")


def _print_capital_state(state: Mapping[str, object]) -> None:
    if not state:
        print("Brak danych alokacji kapitału.")
        return

    normalized = {
        key: _to_serializable(value)
        for key, value in state.items()
    }
    print("Stan alokacji kapitału:")
    print(json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False))


def _print_capital_diagnostics(diagnostics: Mapping[str, object]) -> None:
    if not diagnostics:
        print("Brak diagnostyki polityki kapitału.")
        return

    normalized = {
        key: _to_serializable(value)
        for key, value in diagnostics.items()
    }
    print("Diagnostyka polityki kapitału:")
    print(json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False))


def _print_signal_limits(snapshot: Mapping[str, Mapping[str, object]]) -> None:
    if not snapshot:
        print("Brak nadpisanych limitów sygnałów.")
        return

    print("Nadpisane limity sygnałów:")
    for strategy in sorted(snapshot):
        entry = snapshot.get(strategy) or {}
        if not isinstance(entry, Mapping):
            continue
        print(f"- {strategy}:")
        for profile in sorted(entry):
            limit_entry = entry.get(profile)
            extras: list[str] = []
            value: object | None = None
            if isinstance(limit_entry, Mapping):
                raw_limit = limit_entry.get("limit")
                try:
                    value = int(raw_limit) if raw_limit is not None else None
                except (TypeError, ValueError):
                    value = raw_limit
                reason = limit_entry.get("reason")
                if reason:
                    extras.append(f"powód={reason}")
                expires = limit_entry.get("expires_at")
                if expires:
                    extras.append(f"do={expires}")
                remaining = limit_entry.get("remaining_seconds")
                if isinstance(remaining, (int, float)) and remaining >= 0:
                    extras.append(f"pozostało={remaining:.0f}s")
                active = limit_entry.get("active")
                if isinstance(active, bool):
                    extras.append("aktywne" if active else "wygasłe")
            else:
                limit = limit_entry
                try:
                    value = int(limit) if limit is not None else None
                except (TypeError, ValueError):  # pragma: no cover - defensywne
                    value = limit
            suffix = f" ({', '.join(extras)})" if extras else ""
            print(f"    {profile}: {value}{suffix}")


def _print_schedule_descriptions(descriptions: Mapping[str, Mapping[str, object]]) -> None:
    if not descriptions:
        print("Brak zarejestrowanych harmonogramów.")
        return

    print("Zarejestrowane harmonogramy:")
    for name in sorted(descriptions):
        entry = descriptions.get(name) or {}
        strategy = entry.get("strategy_name", "unknown")
        profile = entry.get("risk_profile", "-")
        cadence = float(entry.get("cadence_seconds", 0.0) or 0.0)
        max_drift = float(entry.get("max_drift_seconds", 0.0) or 0.0)
        warmup = int(entry.get("warmup_bars", 0) or 0)
        tags = entry.get("tags")
        tags_text = ", ".join(tags) if isinstance(tags, Sequence) else "-"
        primary_tag = entry.get("primary_tag") or "-"
        base_limit = entry.get("base_max_signals")
        active_limit = entry.get("active_max_signals")
        allocator_weight = float(entry.get("allocator_weight", 0.0) or 0.0)
        portfolio_weight = float(entry.get("portfolio_weight", 0.0) or 0.0)
        print(
            f"- {name}: strategia={strategy}, profil={profile}, cadence={cadence:.2f}s, "
            f"max_drift={max_drift:.2f}s, warmup={warmup}"
        )
        print(
            f"    sygnały: bazowy={base_limit}, aktywny={active_limit}, "
            f"waga_alokatora={allocator_weight:.4f}, waga_portfela={portfolio_weight:.4f}"
        )
        print(f"    tagi: {tags_text} (primary={primary_tag})")
        limit_override = entry.get("signal_limit_override")
        if limit_override is not None:
            print(f"    override limitu sygnałów: {limit_override}")
        last_run = entry.get("last_run")
        if last_run:
            print(f"    ostatnie uruchomienie: {last_run}")
        suspension = entry.get("active_suspension")
        if isinstance(suspension, Mapping):
            reason = suspension.get("reason", "unknown")
            origin = suspension.get("origin", "schedule")
            until = suspension.get("until", "bez terminu")
            print(
                "    zawieszenie: "
                f"powód={reason}, źródło={origin}, do={until}"
            )


def _load_policy_spec(path_text: str) -> object:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise ValueError(f"Nie znaleziono pliku polityki kapitału: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - błędy IO są rzadkie
        raise ValueError(f"Nie udało się odczytać pliku polityki kapitału: {path}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if yaml is None:
            raise ValueError(
                "Nie udało się sparsować polityki kapitału jako JSON, a PyYAML nie jest dostępny."
            )
        loaded = yaml.safe_load(raw)
        if loaded is None:
            raise ValueError(f"Plik polityki kapitału {path} jest pusty")
        return loaded


def _perform_management_actions(scheduler: object, args: argparse.Namespace) -> bool:
    performed = False
    reason = getattr(args, "suspension_reason", None)
    until = getattr(args, "suspension_until", None)
    duration = getattr(args, "suspension_duration", None)

    for name in getattr(args, "resume_schedules", []) or []:
        resume = getattr(scheduler, "resume_schedule", None)
        if callable(resume):
            success = bool(resume(name))
            status = "Wznowiono" if success else "Nie znaleziono"
            print(f"{status} harmonogram: {name}")
            performed = True

    for tag in getattr(args, "resume_tags", []) or []:
        resume_tag = getattr(scheduler, "resume_tag", None)
        if callable(resume_tag):
            success = bool(resume_tag(tag))
            status = "Wznowiono" if success else "Nie znaleziono"
            print(f"{status} tag: {tag}")
            performed = True

    set_specs = getattr(args, "set_signal_limits", None) or []
    clear_specs = getattr(args, "clear_signal_limits", None) or []
    if set_specs or clear_specs:
        configure_limit = getattr(scheduler, "configure_signal_limit", None)
        if not callable(configure_limit):
            print("Brak obsługi konfiguracji limitów sygnałów w schedulerze.")
        else:
            limit_reason = getattr(args, "signal_limit_reason", None)
            limit_until = getattr(args, "signal_limit_until", None)
            limit_duration = getattr(args, "signal_limit_duration", None)
            for spec in set_specs:
                strategy, profile, limit = _parse_signal_limit_assignment(spec)
                configure_limit(
                    strategy,
                    profile,
                    limit,
                    reason=limit_reason,
                    until=limit_until,
                    duration_seconds=limit_duration,
                )
                extras: list[str] = []
                if limit_reason:
                    extras.append(f"powód={limit_reason}")
                if limit_until:
                    extras.append(f"do={limit_until.isoformat()}")
                elif limit_duration:
                    duration_text = (
                        f"{int(limit_duration)}s"
                        if float(limit_duration).is_integer()
                        else f"{limit_duration:.2f}s"
                    )
                    extras.append(f"czas={duration_text}")
                suffix = f" ({', '.join(extras)})" if extras else ""
                print(
                    "Ustawiono limit sygnałów "
                    f"{strategy}/{profile} = {int(limit)}{suffix}"
                )
            for spec in clear_specs:
                strategy, profile = _parse_signal_limit_target(spec)
                configure_limit(strategy, profile, None)
                print(
                    "Usunięto nadpisanie limitu sygnałów dla "
                    f"{strategy}/{profile}"
                )
        performed = True

    for name in getattr(args, "suspend_schedules", []) or []:
        suspend = getattr(scheduler, "suspend_schedule", None)
        if callable(suspend):
            suspend(
                name,
                reason=reason,
                until=until,
                duration_seconds=duration,
            )
            print(f"Zawieszono harmonogram: {name}")
            performed = True

    for tag in getattr(args, "suspend_tags", []) or []:
        suspend_tag = getattr(scheduler, "suspend_tag", None)
        if callable(suspend_tag):
            suspend_tag(
                tag,
                reason=reason,
                until=until,
                duration_seconds=duration,
            )
            print(f"Zawieszono tag: {tag}")
            performed = True

    suspensions_path = getattr(args, "export_suspensions", None)
    if getattr(args, "list_suspensions", False) or suspensions_path:
        snapshot_fn = getattr(scheduler, "suspension_snapshot", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn()
            if getattr(args, "list_suspensions", False):
                _print_suspension_snapshot(snapshot)
            if suspensions_path:
                _export_payload(
                    suspensions_path,
                    cast(Mapping[str, object], snapshot),
                    description="zawieszenia",
                )
        else:  # pragma: no cover - defensywne
            print("Brak obsługi migawki zawieszeń w schedulerze.")
        performed = True

    signal_limits_path = getattr(args, "export_signal_limits", None)
    if getattr(args, "list_signal_limits", False) or signal_limits_path:
        snapshot_fn = getattr(scheduler, "signal_limit_snapshot", None)
        if callable(snapshot_fn):
            limits = cast(
                Mapping[str, Mapping[str, Mapping[str, object]]],
                snapshot_fn(),
            )
            if getattr(args, "list_signal_limits", False):
                _print_signal_limits(limits)
            if signal_limits_path:
                _export_payload(
                    signal_limits_path,
                    cast(Mapping[str, object], limits),
                    description="limity sygnałów",
                )
        else:  # pragma: no cover - defensywne
            print("Brak obsługi migawki limitów sygnałów w schedulerze.")
        performed = True

    schedules_path = getattr(args, "export_schedules", None)
    if getattr(args, "list_schedules", False) or schedules_path:
        describe_fn = getattr(scheduler, "describe_schedules", None)
        if callable(describe_fn):
            descriptions = cast(
                Mapping[str, Mapping[str, object]],
                describe_fn(),
            )
            if getattr(args, "list_schedules", False):
                _print_schedule_descriptions(descriptions)
            if schedules_path:
                _export_payload(
                    schedules_path,
                    cast(Mapping[str, object], descriptions),
                    description="opis harmonogramów",
                )
        else:  # pragma: no cover - defensywne
            print("Brak obsługi opisu harmonogramów w schedulerze.")
        performed = True

    capital_state_path = getattr(args, "export_capital_state", None)
    if getattr(args, "show_capital_state", False) or capital_state_path:
        state_fn = getattr(scheduler, "capital_allocation_state", None)
        if callable(state_fn):
            state = cast(Mapping[str, object], state_fn())
            if getattr(args, "show_capital_state", False):
                _print_capital_state(state)
            if capital_state_path:
                _export_payload(
                    capital_state_path,
                    state,
                    description="stan alokacji kapitału",
                )
        else:  # pragma: no cover - defensywne
            print("Brak obsługi stanu alokacji kapitału w schedulerze.")
        performed = True

    diagnostics_path = getattr(args, "export_capital_diagnostics", None)
    if getattr(args, "show_capital_diagnostics", False) or diagnostics_path:
        diag_fn = getattr(scheduler, "capital_policy_diagnostics", None)
        if callable(diag_fn):
            diagnostics = cast(Mapping[str, object], diag_fn())
            if getattr(args, "show_capital_diagnostics", False):
                _print_capital_diagnostics(diagnostics)
            if diagnostics_path:
                _export_payload(
                    diagnostics_path,
                    diagnostics,
                    description="diagnostykę polityki kapitału",
                )
        else:  # pragma: no cover - defensywne
            print("Brak obsługi diagnostyki polityki kapitału w schedulerze.")
        performed = True

    if getattr(args, "rebalance_capital", False):
        rebalance = getattr(scheduler, "rebalance_capital", None)
        if callable(rebalance):
            asyncio.run(rebalance(ignore_cooldown=True))
            print("Przeliczono alokację kapitału.")
        else:  # pragma: no cover - defensywne
            print("Brak obsługi ręcznego przeliczenia kapitału w schedulerze.")
        performed = True

    policy_path = getattr(args, "set_capital_policy", None)
    if policy_path:
        policy_spec = _load_policy_spec(policy_path)
        policy, interval = resolve_capital_policy_spec(policy_spec)
        replace = getattr(scheduler, "replace_capital_policy", None)
        if callable(replace):
            rebalance = not getattr(args, "skip_policy_rebalance", False)
            asyncio.run(replace(policy, rebalance=rebalance))
            policy_name = getattr(policy, "name", policy.__class__.__name__)
            action = "Zastosowano" if rebalance else "Załadowano"
            print(f"{action} politykę kapitału: {policy_name}")
        else:
            print("Brak obsługi zmiany polityki kapitału w schedulerze.")
        if getattr(args, "apply_policy_interval", False) and interval is not None:
            setter = getattr(scheduler, "set_allocation_rebalance_seconds", None)
            if callable(setter):
                setter(interval)
                print(
                    "Ustawiono interwał przeliczeń alokacji na "
                    f"{float(interval):.2f} s."
                )
            else:
                print("Brak obsługi zmiany interwału alokacji w schedulerze.")
        performed = True

    interval_override = getattr(args, "set_allocation_interval", None)
    if interval_override not in (None, ""):
        setter = getattr(scheduler, "set_allocation_rebalance_seconds", None)
        if callable(setter):
            try:
                seconds = float(interval_override)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Niepoprawna wartość interwału alokacji: {interval_override}"
                ) from exc
            setter(seconds)
            print(f"Ustawiono interwał przeliczeń alokacji na {seconds:.2f} s.")
        else:
            print("Brak obsługi zmiany interwału alokacji w schedulerze.")
        performed = True

    return performed


def _build_scheduler(
    *,
    config_path: Path,
    environment: str,
    scheduler_name: str | None,
    adapter_factories: Mapping[str, object] | None,
) -> object:
    try:
        runtime = build_multi_strategy_runtime(
            environment_name=environment,
            scheduler_name=scheduler_name,
            config_path=config_path,
            secret_manager=SecretManager(create_default_secret_storage()),
            adapter_factories=cast(Mapping[str, ExchangeAdapterFactory] | None, adapter_factories),
            telemetry_emitter=lambda name, payload: print(
                f"[telemetry] schedule={name} signals={payload.get('signals', 0)} "
                f"latency_ms={payload.get('latency_ms', 0.0):.2f}"
            ),
        )
    except RuntimeError as exc:
        cause = exc.__cause__
        if isinstance(cause, LicenseCapabilityError):
            raise cause
        raise
    scheduler = runtime.scheduler
    setattr(scheduler, "_runtime", runtime)
    return scheduler


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-strategy scheduler")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument("--environment", required=True, help="Nazwa środowiska (np. binance_paper)")
    parser.add_argument("--scheduler", default=None, help="Nazwa schedulera z sekcji multi_strategy_schedulers")
    parser.add_argument(
        "--adapter-factory",
        action="append",
        dest="adapter_factories",
        metavar="NAME=SPEC",
        help=(
            "Override fabryk adapterów przekazywane do bootstrapu runtime. "
            "Użyj '!remove', aby usunąć wpis – opcję można podawać wielokrotnie."
        ),
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Wykonaj pojedynczy cykl harmonogramu i zakończ (tryb smoke/audit)",
    )
    parser.add_argument(
        "--suspend-schedule",
        dest="suspend_schedules",
        action="append",
        metavar="NAME",
        help="Zawieś wskazany harmonogram (argument można powtarzać)",
    )
    parser.add_argument(
        "--suspend-tag",
        dest="suspend_tags",
        action="append",
        metavar="TAG",
        help="Zawieś wszystkie harmonogramy posiadające dany tag",
    )
    parser.add_argument(
        "--resume-schedule",
        dest="resume_schedules",
        action="append",
        metavar="NAME",
        help="Wznów wskazany harmonogram (argument można powtarzać)",
    )
    parser.add_argument(
        "--resume-tag",
        dest="resume_tags",
        action="append",
        metavar="TAG",
        help="Wznów wszystkie harmonogramy posiadające dany tag",
    )
    parser.add_argument(
        "--suspension-reason",
        dest="suspension_reason",
        help="Powód używany przy zawieszaniu harmonogramów/tagów",
    )
    parser.add_argument(
        "--suspension-until",
        dest="suspension_until",
        type=_parse_datetime_argument,
        help="Czas wygaśnięcia zawieszenia w formacie ISO 8601",
    )
    parser.add_argument(
        "--suspension-duration",
        dest="suspension_duration",
        type=_parse_duration_argument,
        help="Czas trwania zawieszenia (np. 15m, 2h, 3600)",
    )
    parser.add_argument(
        "--list-suspensions",
        dest="list_suspensions",
        action="store_true",
        help="Wyświetl bieżącą listę zawieszeń i zakończ",
    )
    parser.add_argument(
        "--export-suspensions",
        dest="export_suspensions",
        metavar="PATH",
        help="Zapisz bieżące zawieszenia do pliku JSON",
    )
    parser.add_argument(
        "--set-signal-limit",
        dest="set_signal_limits",
        action="append",
        metavar="STRATEGIA:PROFIL=LIMIT",
        help="Ustaw nadpisanie limitu sygnałów dla strategii i profilu",
    )
    parser.add_argument(
        "--clear-signal-limit",
        dest="clear_signal_limits",
        action="append",
        metavar="STRATEGIA:PROFIL",
        help="Usuń nadpisanie limitu sygnałów dla strategii i profilu",
    )
    parser.add_argument(
        "--signal-limit-reason",
        dest="signal_limit_reason",
        help="Powód zapisywany przy nadpisaniu limitu sygnałów",
    )
    parser.add_argument(
        "--signal-limit-until",
        dest="signal_limit_until",
        type=_parse_datetime_argument,
        help="Czas wygaśnięcia nadpisania limitu sygnałów (ISO 8601)",
    )
    parser.add_argument(
        "--signal-limit-duration",
        dest="signal_limit_duration",
        type=_parse_duration_argument,
        help="Czas obowiązywania nadpisania limitu sygnałów (np. 30m, 2h)",
    )
    parser.add_argument(
        "--list-signal-limits",
        dest="list_signal_limits",
        action="store_true",
        help="Wyświetl aktualne nadpisania limitów sygnałów i zakończ",
    )
    parser.add_argument(
        "--export-signal-limits",
        dest="export_signal_limits",
        metavar="PATH",
        help="Zapisz nadpisania limitów sygnałów do pliku JSON",
    )
    parser.add_argument(
        "--list-schedules",
        dest="list_schedules",
        action="store_true",
        help="Wyświetl konfigurację zarejestrowanych harmonogramów i zakończ",
    )
    parser.add_argument(
        "--show-capital-state",
        dest="show_capital_state",
        action="store_true",
        help="Wyświetl stan alokacji kapitału i zakończ",
    )
    parser.add_argument(
        "--show-capital-diagnostics",
        dest="show_capital_diagnostics",
        action="store_true",
        help="Wyświetl diagnostykę polityki kapitału i zakończ",
    )
    parser.add_argument(
        "--export-capital-state",
        dest="export_capital_state",
        metavar="PATH",
        help="Zapisz stan alokacji kapitału do pliku JSON",
    )
    parser.add_argument(
        "--export-capital-diagnostics",
        dest="export_capital_diagnostics",
        metavar="PATH",
        help="Zapisz diagnostykę polityki kapitału do pliku JSON",
    )
    parser.add_argument(
        "--export-schedules",
        dest="export_schedules",
        metavar="PATH",
        help="Zapisz opis harmonogramów do pliku JSON",
    )
    parser.add_argument(
        "--rebalance-capital",
        dest="rebalance_capital",
        action="store_true",
        help=(
            "Wymuś natychmiastowe przeliczenie alokacji kapitału i zakończ, chyba że "
            "użyto --run-after-management"
        ),
    )
    parser.add_argument(
        "--set-capital-policy",
        dest="set_capital_policy",
        metavar="PATH",
        help="Załaduj nową politykę kapitału z pliku (JSON/YAML)",
    )
    parser.add_argument(
        "--skip-policy-rebalance",
        dest="skip_policy_rebalance",
        action="store_true",
        help="Załaduj politykę kapitału bez natychmiastowego przeliczenia",
    )
    parser.add_argument(
        "--apply-policy-interval",
        dest="apply_policy_interval",
        action="store_true",
        help="Zastosuj zalecany interwał przeliczeń zwrócony przez politykę",
    )
    parser.add_argument(
        "--set-allocation-interval",
        dest="set_allocation_interval",
        metavar="SECONDS",
        help="Ustaw interwał przeliczeń alokacji (sekundy)",
    )
    parser.add_argument(
        "--run-after-management",
        dest="run_after_management",
        action="store_true",
        help="Uruchom scheduler po wykonaniu operacji zarządzania zawieszeniami",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    cli_adapter_specs = parse_adapter_factory_cli_specs(
        cast(Sequence[str] | None, getattr(args, "adapter_factories", None))
    )
    adapter_factories: Mapping[str, object] | None = cli_adapter_specs if cli_adapter_specs else None

    try:
        scheduler = _build_scheduler(
            config_path=config_path,
            environment=args.environment,
            scheduler_name=args.scheduler,
            adapter_factories=adapter_factories,
        )
    except LicenseCapabilityError as exc:
        LOGGER.error(
            "Uruchomienie scheduler-a multi-strategy zablokowane przez licencję: %s",
            exc,
        )
        return 2
    except LicenseValidationError as exc:
        LOGGER.error("Walidacja licencji nie powiodła się: %s", exc)
        return 2

    runtime: MultiStrategyRuntime | None = getattr(scheduler, "_runtime", None)

    try:
        management_performed = _perform_management_actions(scheduler, args)
        if management_performed and not getattr(args, "run_after_management", False):
            return 0
        if args.run_once:
            asyncio.run(scheduler.run_once())
        else:
            asyncio.run(scheduler.run_forever())
    except ValueError as exc:
        LOGGER.error("Operacja zarządzania schedulerem nie powiodła się: %s", exc)
        return 1
    except KeyboardInterrupt:
        stop = getattr(scheduler, "stop", None)
        if callable(stop):
            stop()
    finally:
        if runtime is not None:
            runtime.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
