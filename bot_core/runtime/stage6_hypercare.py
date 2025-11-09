"""Agregacja cykli hypercare Stage6 i podpisany raport zbiorczy."""

from __future__ import annotations

import json
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping

from bot_core.observability.hypercare import (
    ObservabilityCycleConfig,
    ObservabilityCycleResult,
    ObservabilityHypercareCycle,
)
from bot_core.portfolio.governor import PortfolioGovernor
from bot_core.resilience.hypercare import (
    ResilienceCycleConfig,
    ResilienceCycleResult,
    ResilienceHypercareCycle,
)
from bot_core.runtime.telemetry_risk_profiles import (
    get_observability_composite_expectations,
)
from bot_core.security.signing import build_hmac_signature, verify_hmac_signature


if TYPE_CHECKING:
    from bot_core.portfolio.hypercare import (
        PortfolioCycleConfig,
        PortfolioCycleResult,
        PortfolioHypercareCycle,
    )


ObservabilityFactory = Callable[[ObservabilityCycleConfig], ObservabilityHypercareCycle]
ResilienceFactory = Callable[[ResilienceCycleConfig], ResilienceHypercareCycle]
PortfolioFactory = Callable[[PortfolioGovernor, Any], Any]


def _now_utc_iso() -> str:
    timestamp = datetime.now(timezone.utc)
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _path_or_none(path: Path | None) -> str | None:
    return path.as_posix() if path is not None else None


def _status_from_flags(issues: list[str], warnings: list[str]) -> str:
    if issues:
        return "fail"
    if warnings:
        return "warn"
    return "ok"


@dataclass(slots=True)
class Stage6HypercareConfig:
    """Konfiguracja raportu zbiorczego hypercare Stage6."""

    output_path: Path
    signature_path: Path | None = None
    signing_key: bytes | None = None
    signing_key_id: str | None = None
    metadata: Mapping[str, Any] | None = None
    observability: ObservabilityCycleConfig | None = None
    resilience: ResilienceCycleConfig | None = None
    portfolio: PortfolioCycleConfig | None = None


@dataclass(slots=True)
class Stage6HypercareResult:
    """Wynik agregacji hypercare Stage6."""

    payload: Mapping[str, Any]
    output_path: Path
    signature_path: Path | None
    observability: ObservabilityCycleResult | None
    resilience: ResilienceCycleResult | None
    portfolio: PortfolioCycleResult | None


@dataclass(slots=True)
class Stage6HypercareVerificationResult:
    """Wynik weryfikacji podpisanego raportu Stage6."""

    summary_path: Path
    signature_path: Path | None
    summary: Mapping[str, Any]
    signature_valid: bool
    overall_status: str
    issues: list[str]
    warnings: list[str]
    component_statuses: Mapping[str, str]


class Stage6HypercareCycle:
    """Wykonuje kolejne cykle Stage6 i buduje raport zbiorczy."""

    def __init__(
        self,
        config: Stage6HypercareConfig,
        *,
        portfolio_governor: PortfolioGovernor | None = None,
        observability_factory: ObservabilityFactory | None = None,
        resilience_factory: ResilienceFactory | None = None,
        portfolio_factory: PortfolioFactory | None = None,
    ) -> None:
        self._config = config
        self._portfolio_governor = portfolio_governor
        self._observability_factory = (
            observability_factory if observability_factory else ObservabilityHypercareCycle
        )
        self._resilience_factory = resilience_factory if resilience_factory else ResilienceHypercareCycle
        self._portfolio_factory = portfolio_factory

    def _get_portfolio_factory(self) -> PortfolioFactory:
        if self._portfolio_factory is not None:
            return self._portfolio_factory
        from bot_core.portfolio.hypercare import PortfolioHypercareCycle  # lokalny import by uniknąć cyklu

        return PortfolioHypercareCycle

    def run(self) -> Stage6HypercareResult:
        issues: list[str] = []
        warnings: list[str] = []
        components: MutableMapping[str, Mapping[str, Any]] = {}

        observability_result: ObservabilityCycleResult | None = None
        if self._config.observability:
            observability_summary = self._run_observability()
            observability_result = observability_summary.pop("__result__", None)
            issues.extend(observability_summary.pop("__issues__", ()))
            warnings.extend(observability_summary.pop("__warnings__", ()))
            components["observability"] = observability_summary
        else:
            components["observability"] = {"status": "skipped"}

        resilience_result: ResilienceCycleResult | None = None
        if self._config.resilience:
            resilience_summary = self._run_resilience()
            resilience_result = resilience_summary.pop("__result__", None)
            issues.extend(resilience_summary.pop("__issues__", ()))
            warnings.extend(resilience_summary.pop("__warnings__", ()))
            components["resilience"] = resilience_summary
        else:
            components["resilience"] = {"status": "skipped"}

        portfolio_result: PortfolioCycleResult | None = None
        if self._config.portfolio:
            portfolio_summary = self._run_portfolio()
            portfolio_result = portfolio_summary.pop("__result__", None)
            issues.extend(portfolio_summary.pop("__issues__", ()))
            warnings.extend(portfolio_summary.pop("__warnings__", ()))
            components["portfolio"] = portfolio_summary
        else:
            components["portfolio"] = {"status": "skipped"}

        overall_status = _status_from_flags(issues, warnings)

        payload: MutableMapping[str, Any] = {
            "type": "stage6_hypercare_summary",
            "generated_at": _now_utc_iso(),
            "overall_status": overall_status,
            "issues": list(issues),
            "warnings": list(warnings),
            "components": dict(components),
        }
        if self._config.metadata:
            payload["metadata"] = dict(self._config.metadata)

        output_path = self._config.output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        signature_path: Path | None = None
        if self._config.signing_key:
            target_path = (
                self._config.signature_path.expanduser()
                if self._config.signature_path
                else output_path.with_suffix(output_path.suffix + ".sig")
            )
            signature = build_hmac_signature(
                payload,
                key=self._config.signing_key,
                key_id=self._config.signing_key_id,
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as handle:
                json.dump(signature, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            signature_path = target_path

        return Stage6HypercareResult(
            payload=dict(payload),
            output_path=output_path,
            signature_path=signature_path,
            observability=observability_result,
            resilience=resilience_result,
            portfolio=portfolio_result,
        )

    # ------------------------------------------------------------------
    # Podsumowania poszczególnych komponentów
    # ------------------------------------------------------------------
    def _run_observability(self) -> MutableMapping[str, Any]:
        summary: MutableMapping[str, Any] = {}
        issues: list[str] = []
        warnings: list[str] = []

        try:
            cycle = self._observability_factory(self._config.observability)  # type: ignore[arg-type]
            result = cycle.run()
        except Exception as exc:  # pragma: no cover - defensywne
            issues.append(f"Observability cycle failed: {exc}")
            summary.update({"status": "fail", "error": str(exc)})
            summary["__issues__"] = tuple(issues)
            summary["__warnings__"] = tuple(warnings)
            return summary

        summary["status"] = "ok"
        summary["artifacts"] = {
            "slo_report": _path_or_none(result.slo_report_path),
            "slo_signature": _path_or_none(result.slo_signature_path),
            "slo_csv": _path_or_none(result.slo_csv_path),
            "alert_overrides": _path_or_none(result.overrides_path),
            "alert_signature": _path_or_none(result.overrides_signature_path),
            "dashboard_annotations": _path_or_none(result.dashboard_annotations_path),
            "dashboard_signature": _path_or_none(result.dashboard_signature_path),
            "bundle": _path_or_none(result.bundle_path),
            "bundle_manifest": _path_or_none(result.bundle_manifest_path),
            "bundle_signature": _path_or_none(result.bundle_signature_path),
        }

        slo_payload: Mapping[str, Any] | None = None
        composite_results: dict[str, Mapping[str, Any]] = {}
        slo_signature_valid: bool | None = None
        if result.slo_report_path:
            try:
                payload_raw = json.loads(result.slo_report_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - raport do operatora
                issues.append(f"Nie udało się odczytać raportu SLO: {exc}")
            else:
                if isinstance(payload_raw, Mapping):
                    slo_payload = payload_raw
                else:
                    issues.append("Raport SLO powinien być obiektem JSON")
        else:
            warnings.append("Brak wygenerowanego raportu SLO")

        if slo_payload is not None:
            slo_section: MutableMapping[str, Any] = {
                "generated_at": slo_payload.get("generated_at"),
                "summary": dict(slo_payload.get("summary", {})),
            }
            results_section = slo_payload.get("results")
            if isinstance(results_section, Mapping):
                slo_section["results"] = {
                    str(name): dict(entry)
                    for name, entry in results_section.items()
                    if isinstance(entry, Mapping)
                }
            summary["slo"] = slo_section

            composites_section = slo_payload.get("composites")
            if isinstance(composites_section, Mapping):
                raw_results = composites_section.get("results")
                if isinstance(raw_results, Mapping):
                    for name, entry in raw_results.items():
                        if isinstance(entry, Mapping):
                            composite_results[str(name)] = dict(entry)
                if composite_results:
                    status_counts: MutableMapping[str, int] = {}
                    for entry in composite_results.values():
                        status_value = str(entry.get("status", "unknown")).lower()
                        status_counts[status_value] = status_counts.get(status_value, 0) + 1
                    slo2_section: MutableMapping[str, Any] = {
                        "results": composite_results,
                        "status_counts": dict(status_counts),
                        "evaluated": len(composite_results),
                    }
                    breached = [
                        name
                        for name, entry in composite_results.items()
                        if str(entry.get("status", "ok")).lower() not in {"ok", "success"}
                    ]
                    if breached:
                        slo2_section["breached"] = breached
                        warnings.append("SLO2 naruszone: " + ", ".join(breached))
                    summary["slo2"] = slo2_section
                else:
                    warnings.append("Raport SLO nie zawiera wyników SLO2")

            signing_key = None
            if self._config.observability:
                signing_key = self._config.observability.signing_key
            if result.slo_signature_path and signing_key:
                try:
                    signature_doc = json.loads(
                        result.slo_signature_path.read_text(encoding="utf-8")
                    )
                    slo_signature_valid = bool(
                        verify_hmac_signature(slo_payload, signature_doc, key=signing_key)
                    )
                except Exception as exc:  # noqa: BLE001 - weryfikacja podpisu
                    issues.append(f"Nie udało się zweryfikować podpisu raportu SLO: {exc}")
                    slo_signature_valid = None
                else:
                    if slo_signature_valid is False:
                        warnings.append("Podpis raportu SLO nie został zweryfikowany")
                summary.setdefault("slo", {})["signature_valid"] = slo_signature_valid
            elif signing_key:
                warnings.append(
                    "Brak podpisu raportu SLO mimo skonfigurowanego klucza HMAC"
                )

            profile_name: str | None = None
            if isinstance(self._config.metadata, Mapping):
                metadata_profile = self._config.metadata.get("observability_profile")
                profile_name = str(metadata_profile) if metadata_profile else None
                if not profile_name and self._config.metadata.get("risk_profile"):
                    profile_name = str(self._config.metadata.get("risk_profile"))
            if profile_name:
                try:
                    expectations = get_observability_composite_expectations(profile_name)
                except Exception as exc:  # noqa: BLE001 - raportujemy w ostrzeżeniach
                    warnings.append(
                        f"Nie udało się pobrać oczekiwań SLO2 dla profilu {profile_name}: {exc}"
                    )
                else:
                    summary.setdefault("slo2", {})["expected"] = expectations
                    expected = set(expectations.get("composites", ()))
                    missing_expected = sorted(expected - set(composite_results))
                    if missing_expected:
                        warnings.append(
                            "Brak wyników SLO2 dla profilu "
                            f"{profile_name}: " + ", ".join(missing_expected)
                        )

        if self._config.observability and self._config.observability.overrides and not result.overrides_path:
            warnings.append("Brak wygenerowanych override'ów alertów")
        if self._config.observability and self._config.observability.bundle:
            if not result.bundle_path:
                warnings.append("Paczka obserwowalności nie została utworzona")
            elif result.bundle_verification:
                summary["bundle_verification"] = dict(result.bundle_verification)
                signature_verified = result.bundle_verification.get("signature_verified")
                if signature_verified is False:
                    warnings.append("Podpis paczki obserwowalności nie został zweryfikowany")

        if warnings:
            summary["status"] = "warn"

        summary["__warnings__"] = tuple(warnings)
        summary["__issues__"] = tuple(issues)
        summary["__result__"] = result
        return summary

    def _run_resilience(self) -> MutableMapping[str, Any]:
        summary: MutableMapping[str, Any] = {}
        issues: list[str] = []
        warnings: list[str] = []

        try:
            cycle = self._resilience_factory(self._config.resilience)  # type: ignore[arg-type]
            result = cycle.run()
        except Exception as exc:  # pragma: no cover - defensywne
            issues.append(f"Resilience cycle failed: {exc}")
            summary.update({"status": "fail", "error": str(exc)})
            summary["__issues__"] = tuple(issues)
            summary["__warnings__"] = tuple(warnings)
            return summary

        summary["status"] = "ok"
        summary["artifacts"] = {
            "bundle": _path_or_none(result.bundle_artifacts.bundle_path),
            "bundle_manifest": _path_or_none(result.bundle_artifacts.manifest_path),
            "bundle_signature": _path_or_none(result.bundle_artifacts.signature_path),
            "audit_json": _path_or_none(result.audit_summary_path),
            "audit_csv": _path_or_none(result.audit_csv_path),
            "audit_signature": _path_or_none(result.audit_signature_path),
            "failover_json": _path_or_none(result.failover_summary_path),
            "failover_csv": _path_or_none(result.failover_csv_path),
            "failover_signature": _path_or_none(result.failover_signature_path),
            "self_healing_report": _path_or_none(result.self_healing_report_path),
            "self_healing_signature": _path_or_none(result.self_healing_signature_path),
        }

        failover_status = result.failover_summary.status
        summary["failover_status"] = failover_status
        if failover_status == "critical":
            issues.append("Failover drill zakończył się statusem critical")
        elif failover_status != "ok":
            warnings.append(f"Failover drill zakończył się statusem {failover_status}")

        if result.verification:
            summary["bundle_verification"] = dict(result.verification)
            if result.verification.get("signature_verified") is False:
                warnings.append("Podpis paczki odpornościowej nie został zweryfikowany")

        if issues:
            summary["status"] = "fail"
        elif warnings:
            summary["status"] = "warn"

        summary["__warnings__"] = tuple(warnings)
        summary["__issues__"] = tuple(issues)
        summary["__result__"] = result
        return summary

    def _run_portfolio(self) -> MutableMapping[str, Any]:
        summary: MutableMapping[str, Any] = {}
        issues: list[str] = []
        warnings: list[str] = []

        if self._portfolio_governor is None:
            issues.append("Brak instancji PortfolioGovernora dla cyklu portfelowego")
            summary.update({"status": "fail", "error": "portfolio governor missing"})
            summary["__warnings__"] = tuple(warnings)
            summary["__issues__"] = tuple(issues)
            return summary

        try:
            factory = self._get_portfolio_factory()
            cycle = factory(self._portfolio_governor, self._config.portfolio)  # type: ignore[arg-type]
            result = cycle.run()
        except Exception as exc:  # pragma: no cover - defensywne
            issues.append(f"Portfolio cycle failed: {exc}")
            summary.update({"status": "fail", "error": str(exc)})
            summary["__warnings__"] = tuple(warnings)
            summary["__issues__"] = tuple(issues)
            return summary

        decision = result.decision
        summary["status"] = "ok"
        summary["decision"] = {
            "portfolio_id": decision.portfolio_id,
            "portfolio_value": decision.portfolio_value,
            "rebalance_required": decision.rebalance_required,
            "adjustments": len(decision.adjustments),
            "advisories": len(decision.advisories),
        }
        summary["artifacts"] = {
            "summary": _path_or_none(result.summary_path),
            "signature": _path_or_none(result.signature_path),
            "csv": _path_or_none(result.csv_path),
        }

        if decision.rebalance_required:
            warnings.append("PortfolioGovernor wymaga przeprowadzenia rebalance")
        if result.stress_overrides:
            summary["stress_overrides"] = [override.to_dict() for override in result.stress_overrides]
        if result.slo_statuses:
            summary["slo_statuses"] = {
                name: status.value for name, status in result.slo_statuses.items()
            }

        if warnings:
            summary["status"] = "warn"

        summary["__warnings__"] = tuple(warnings)
        summary["__issues__"] = tuple(issues)
        summary["__result__"] = result
        return summary


def _coerce_strings(sequence: Any, *, field_name: str) -> list[str]:
    if sequence is None:
        return []
    if isinstance(sequence, (list, tuple)):
        return [str(item) for item in sequence]
    raise ValueError(f"Pole '{field_name}' powinno być listą lub krotką")


def _collect_component_statuses(components: Any) -> tuple[dict[str, str], list[str]]:
    statuses: dict[str, str] = {}
    issues: list[str] = []
    if components is None:
        issues.append("Brak sekcji 'components' w raporcie Stage6")
        return statuses, issues
    if not isinstance(components, Mapping):
        issues.append("Sekcja 'components' powinna być obiektem JSON")
        return statuses, issues
    for name, payload in components.items():
        if not isinstance(payload, Mapping):
            issues.append(f"Komponent '{name}' powinien być obiektem z polem 'status'")
            continue
        status = payload.get("status", "unknown")
        status_str = str(status)
        if status_str not in {"ok", "warn", "fail", "skipped"}:
            issues.append(
                f"Komponent '{name}' posiada nieobsługiwany status '{status_str}'"
            )
        elif status_str == "fail":
            issues.append(f"Komponent '{name}' zakończył się statusem 'fail'")
        statuses[str(name)] = status_str
    return statuses, issues


def verify_stage6_hypercare_summary(
    summary_path: Path,
    *,
    signature_path: Path | None = None,
    signing_key: bytes | None = None,
    require_signature: bool = False,
) -> Stage6HypercareVerificationResult:
    """Weryfikuje raport zbiorczy Stage6 oraz opcjonalny podpis HMAC."""

    summary_path = summary_path.expanduser()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono raportu Stage6: {summary_path}")

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_data, Mapping):  # pragma: no cover - defensywne
        raise ValueError("Raport Stage6 powinien być obiektem JSON")

    issues: list[str] = []
    warnings: list[str] = []

    summary_type = summary_data.get("type")
    if summary_type != "stage6_hypercare_summary":
        issues.append("Niepoprawny typ raportu Stage6 – oczekiwano 'stage6_hypercare_summary'")

    try:
        summary_issues = _coerce_strings(summary_data.get("issues"), field_name="issues")
    except ValueError as exc:
        issues.append(str(exc))
        summary_issues = []

    try:
        summary_warnings = _coerce_strings(
            summary_data.get("warnings"), field_name="warnings"
        )
    except ValueError as exc:
        issues.append(str(exc))
        summary_warnings = []

    issues.extend(summary_issues)
    warnings.extend(summary_warnings)

    components = summary_data.get("components")
    component_statuses, component_issues = _collect_component_statuses(components)
    issues.extend(component_issues)

    overall_status = str(summary_data.get("overall_status", "unknown"))
    if overall_status not in {"ok", "warn", "fail", "skipped", "unknown"}:
        issues.append(f"Nieobsługiwany status raportu Stage6: '{overall_status}'")

    if issues and overall_status == "ok":
        warnings.append("Status 'ok' raportu nie odzwierciedla zgłoszonych problemów")
    elif not issues:
        if any(status == "fail" for status in component_statuses.values()):
            if overall_status != "fail":
                warnings.append("Status raportu powinien być 'fail' przy błędach komponentów")
        elif any(status == "warn" for status in component_statuses.values()):
            if overall_status == "ok":
                warnings.append("Status raportu może wymagać 'warn' przy ostrzeżeniach komponentów")

    signature_doc: Mapping[str, Any] | None = None
    signature_used_path: Path | None = None
    signature_candidate = signature_path.expanduser() if signature_path else None
    if signature_candidate is None:
        candidate = summary_path.with_suffix(summary_path.suffix + ".sig")
        if candidate.exists():
            signature_candidate = candidate

    if signature_candidate:
        if signature_candidate.exists():
            signature_doc = json.loads(signature_candidate.read_text(encoding="utf-8"))
            if not isinstance(signature_doc, Mapping):  # pragma: no cover - defensywne
                issues.append("Podpis HMAC powinien być obiektem JSON")
                signature_doc = None
            else:
                signature_used_path = signature_candidate
        else:
            issues.append(f"Oczekiwany plik podpisu Stage6 nie istnieje: {signature_candidate}")

    signature_valid = False
    if signature_doc is not None:
        if signing_key:
            signature_valid = verify_hmac_signature(
                summary_data, signature_doc, key=signing_key
            )
            if not signature_valid:
                issues.append("Nieprawidłowy podpis HMAC dla raportu Stage6")
        else:
            warnings.append(
                "Dostarczono podpis hypercare Stage6, lecz nie przekazano klucza HMAC do weryfikacji"
            )
    else:
        if require_signature:
            issues.append("Wymagany podpis HMAC raportu Stage6 nie został dostarczony")
        elif signing_key is not None:
            warnings.append(
                "Przekazano klucz HMAC, lecz nie znaleziono podpisu raportu Stage6"
            )

    return Stage6HypercareVerificationResult(
        summary_path=summary_path,
        signature_path=signature_used_path,
        summary=dict(summary_data),
        signature_valid=signature_valid,
        overall_status=overall_status,
        issues=issues,
        warnings=warnings,
        component_statuses=component_statuses,
    )


__all__ = [
    "Stage6HypercareConfig",
    "Stage6HypercareCycle",
    "Stage6HypercareResult",
    "Stage6HypercareVerificationResult",
    "verify_stage6_hypercare_summary",
]

