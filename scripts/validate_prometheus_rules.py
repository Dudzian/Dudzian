"""Walidacja reguł alertów Prometheusa dla scenariuszy Stage 4/5/6."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml


@dataclass(slots=True)
class _ValidationContext:
    file: Path
    group: str
    alert: str


class PrometheusRuleValidationError(RuntimeError):
    """Wyjątek zgłaszany w przypadku błędnej struktury reguł."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules",
        action="append",
        default=[
            "deploy/prometheus/rules/multi_strategy_rules.yml",
            "deploy/prometheus/stage5_alerts.yaml",
            "deploy/prometheus/stage6_alerts.yaml",
        ],
        help="Ścieżka do pliku YAML z regułami (domyślnie pliki Stage4/Stage5/Stage6)",
    )
    parser.add_argument(
        "--require-label",
        action="append",
        default=["severity", "team"],
        help="Etykieta wymagana w każdej regule alertu",
    )
    parser.add_argument(
        "--require-annotation",
        action="append",
        default=["summary", "description"],
        help="Adnotacja wymagana w każdej regule alertu",
    )
    parser.add_argument(
        "--allowed-severity",
        action="append",
        default=["critical", "warning", "info"],
        help="Dopuszczalne wartości etykiety severity",
    )
    parser.add_argument(
        "--metric-prefix",
        action="append",
        default=[
            "bot_core_multi_strategy",
            "bot_core_stage6",
            "bot_core_decision",
            "bot_core_trade_cost",
            "bot_core_fill_rate",
            "bot_core_key_rotation",
        ],
        help="Prefiks metryk wymagany w wyrażeniu alertu",
    )
    return parser.parse_args(argv)


def _read_rules(path: Path) -> dict:
    if not path.exists():
        raise PrometheusRuleValidationError(f"Plik z regułami '{path}' nie istnieje")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover
        raise PrometheusRuleValidationError(
            f"Nie udało się sparsować pliku YAML '{path}'"
        ) from exc
    if not isinstance(loaded, dict) or "groups" not in loaded:
        raise PrometheusRuleValidationError(
            f"Plik '{path}' nie zawiera sekcji 'groups'"
        )
    if not isinstance(loaded["groups"], list) or not loaded["groups"]:
        raise PrometheusRuleValidationError(
            f"Plik '{path}' musi zawierać co najmniej jedną grupę reguł"
        )
    return loaded


def _normalise_prefixes(prefixes: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalised: list[str] = []
    for prefix in prefixes:
        if not prefix:
            continue
        lowered = prefix.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalised.append(prefix)
    return normalised


def _validate_rule(
    rule: dict,
    ctx: _ValidationContext,
    *,
    required_labels: Sequence[str],
    required_annotations: Sequence[str],
    allowed_severity: Sequence[str],
    metric_prefixes: Sequence[str],
) -> list[str]:
    errors: list[str] = []

    def _error(message: str) -> None:
        errors.append(f"{ctx.file}: grupa='{ctx.group}' alert='{ctx.alert}': {message}")

    # Polityka: wymagamy 'expr' i 'for'
    for field in ("expr", "for"):
        value = rule.get(field)
        if not isinstance(value, str) or not value.strip():
            _error(f"pole '{field}' musi być niepustym łańcuchem znaków")

    labels = rule.get("labels")
    if not isinstance(labels, dict):
        _error("sekcja 'labels' musi być słownikiem")
        labels = {}
    annotations = rule.get("annotations")
    if not isinstance(annotations, dict):
        _error("sekcja 'annotations' musi być słownikiem")
        annotations = {}

    for label in required_labels:
        value = labels.get(label)
        if not isinstance(value, str) or not value.strip():
            _error(f"brak wymaganej etykiety '{label}'")
            continue
        if label == "severity" and value.lower() not in {s.lower() for s in allowed_severity}:
            _error(f"etykieta severity='{value}' poza dozwolonym zakresem {allowed_severity}")

    for annotation in required_annotations:
        value = annotations.get(annotation)
        if not isinstance(value, str) or len(value.strip()) < 10:
            _error(f"adnotacja '{annotation}' musi zawierać co najmniej 10 znaków treści")

    expr = rule.get("expr")
    if isinstance(expr, str) and metric_prefixes:
        lowered = expr.lower()
        if not any(prefix.lower() in lowered for prefix in metric_prefixes):
            _error("wyrażenie 'expr' nie zawiera wymaganego prefiksu metryk")

    return errors


def _validate_file(
    path: Path,
    *,
    required_labels: Sequence[str],
    required_annotations: Sequence[str],
    allowed_severity: Sequence[str],
    metric_prefixes: Sequence[str],
) -> list[str]:
    document = _read_rules(path)
    groups = document["groups"]
    errors: list[str] = []

    for group in groups:
        group_name = group.get("name", "<bez_nazwy>") if isinstance(group, dict) else "<niepoprawna_grupa>"
        if not isinstance(group, dict):
            errors.append(f"{path}: grupa nie jest słownikiem")
            continue
        rules = group.get("rules")
        if not isinstance(rules, list) or not rules:
            errors.append(f"{path}: grupa '{group_name}' musi zawierać niepustą listę reguł")
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                errors.append(f"{path}: grupa '{group_name}' zawiera regułę o niepoprawnym typie")
                continue
            alert_name = rule.get("alert")
            if not isinstance(alert_name, str) or not alert_name.strip():
                errors.append(f"{path}: grupa '{group_name}' zawiera regułę bez pola 'alert'")
                alert_name = "<bez_nazwy>"
            ctx = _ValidationContext(file=path, group=group_name, alert=alert_name)
            errors.extend(
                _validate_rule(
                    rule,
                    ctx,
                    required_labels=required_labels,
                    required_annotations=required_annotations,
                    allowed_severity=allowed_severity,
                    metric_prefixes=metric_prefixes,
                )
            )

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    rule_paths = [Path(path) for path in args.rules]
    metric_prefixes = _normalise_prefixes(args.metric_prefix)

    all_errors: list[str] = []
    for rule_path in rule_paths:
        try:
            errors = _validate_file(
                rule_path,
                required_labels=args.require_label,
                required_annotations=args.require_annotation,
                allowed_severity=args.allowed_severity,
                metric_prefixes=metric_prefixes,
            )
        except PrometheusRuleValidationError as exc:
            print(f"[alerts] {exc}", file=sys.stderr)
            return 2
        all_errors.extend(errors)

    if all_errors:
        for message in all_errors:
            print(f"[alerts] {message}")
        return 1

    for rule_path in rule_paths:
        print(f"[alerts] Walidacja reguł zakończona sukcesem dla {rule_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    sys.exit(main())
