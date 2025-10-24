"""Uruchamia agregację hypercare Stage6 na podstawie pliku konfiguracyjnego."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta, timezone, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML opcjonalny
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config import load_core_config  # noqa: E402  # pylint: disable=wrong-import-position
from bot_core.observability import (  # noqa: E402
    BundleConfig as ObservabilityBundleConfig,
    DashboardSyncConfig,
    ObservabilityCycleConfig,
    ObservabilityHypercareCycle,
    OverridesOutputConfig,
    SLOOutputConfig,
)
from bot_core.observability.bundle import AssetSource  # noqa: E402
from bot_core.portfolio import (  # noqa: E402
    PortfolioCycleConfig,
    PortfolioCycleInputs,
    PortfolioCycleOutputConfig,
    PortfolioDecisionLog,
    PortfolioGovernor,
    resolve_decision_log_config,
)
from bot_core.resilience.hypercare import (  # noqa: E402
    AuditConfig,
    BundleConfig as ResilienceBundleConfig,
    FailoverConfig,
    ResilienceCycleConfig,
    ResilienceHypercareCycle,
    SelfHealingConfig,
)
from bot_core.resilience.policy import load_policy  # noqa: E402
from bot_core.runtime.stage6_hypercare import (  # noqa: E402
    Stage6HypercareConfig,
    Stage6HypercareCycle,
)
from scripts._cli_common import default_decision_log_path, timestamp_slug


def _load_text_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML jest wymagany do wczytania konfiguracji YAML (pip install pyyaml)."
            )
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, Mapping):
        raise ValueError("Konfiguracja hypercare Stage6 musi być mapowaniem klucz-wartość")
    return data


def _expand_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def _as_dict(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not payload:
        return None
    return dict(payload)


def _minutes_to_timedelta(value: Any) -> timedelta | None:
    if value in (None, "", 0):
        return None
    return timedelta(minutes=float(value))


def _default_summary_path() -> Path:
    return Path("var/audit/stage6") / f"stage6_hypercare_summary_{timestamp_slug()}.json"


def _default_portfolio_summary_path(governor: str) -> Path:
    return Path("var/audit/portfolio") / f"portfolio_cycle_{governor}.json"


def _resolve_signing(section: Mapping[str, Any] | None, *, allow_key_id: bool = True) -> tuple[bytes | None, str | None]:
    if not section:
        return None, None
    key_value = section.get("key")
    key_env = section.get("key_env")
    key_path = section.get("key_path")
    provided = [name for name in (key_value, key_env, key_path) if name]
    if len(provided) > 1:
        raise ValueError("Wybierz jedno źródło klucza HMAC (key, key_env lub key_path)")

    key: bytes | None = None
    if key_value:
        key = str(key_value).encode("utf-8")
    elif key_env:
        env_value = os.environ.get(str(key_env))
        if not env_value:
            raise ValueError(f"Zmienna środowiskowa {key_env} nie zawiera klucza HMAC")
        key = env_value.encode("utf-8")
    elif key_path:
        path = Path(str(key_path)).expanduser()
        if not path.is_file():
            raise ValueError(f"Plik z kluczem HMAC nie istnieje: {path}")
        key = path.read_bytes().strip()

    key_id: str | None = None
    if allow_key_id:
        raw_id = section.get("key_id")
        key_id = str(raw_id) if raw_id not in (None, "") else None
    return key, key_id


def _parse_observability(config: Mapping[str, Any] | None) -> ObservabilityCycleConfig | None:
    if not config:
        return None
    definitions = _expand_path(config.get("definitions"))
    metrics = _expand_path(config.get("metrics"))
    if definitions is None or metrics is None:
        raise ValueError("Sekcja observability wymaga pól definitions i metrics")

    if metrics and not metrics.exists():
        expected = metrics
        print(
            f"[stage6.hypercare] Oczekiwano metryk w {expected}; "
            "skopiuj artefakt z runbooka Observability (np. var/metrics/"
            "stage6_measurements.json) lub zaktualizuj ścieżkę w konfiguracji.",
            file=sys.stderr,
        )

    slo_cfg = config.get("slo") or {}
    slo_output = SLOOutputConfig(
        json_path=_expand_path(slo_cfg.get("json")) or Path("var/audit/observability/slo_report.json"),
        csv_path=_expand_path(slo_cfg.get("csv")),
        signature_path=_expand_path(slo_cfg.get("signature")),
        pretty_json=bool(slo_cfg.get("pretty", False)),
    )

    overrides_cfg = config.get("overrides")
    overrides_output: OverridesOutputConfig | None = None
    if overrides_cfg:
        ttl = _minutes_to_timedelta(overrides_cfg.get("ttl_minutes", 120)) or timedelta(minutes=120)
        overrides_output = OverridesOutputConfig(
            json_path=_expand_path(overrides_cfg.get("json"))
            or Path("var/audit/observability/alert_overrides.json"),
            signature_path=_expand_path(overrides_cfg.get("signature")),
            include_warning=bool(overrides_cfg.get("include_warning", True)),
            ttl=ttl,
            requested_by=overrides_cfg.get("requested_by"),
            source=overrides_cfg.get("source", "slo_monitor"),
            tags=tuple(overrides_cfg.get("tags", ())),
            severity_overrides=dict(overrides_cfg.get("severity_overrides", {})),
            existing_path=_expand_path(overrides_cfg.get("existing")),
        )

    dashboard_cfg = config.get("dashboard")
    dashboard_output: DashboardSyncConfig | None = None
    if dashboard_cfg:
        definition = _expand_path(dashboard_cfg.get("definition"))
        output = _expand_path(dashboard_cfg.get("output"))
        if definition is None or output is None:
            raise ValueError("Sekcja dashboard wymaga pól definition oraz output")
        dashboard_output = DashboardSyncConfig(
            dashboard_path=definition,
            output_path=output,
            signature_path=_expand_path(dashboard_cfg.get("signature")),
            panel_id=dashboard_cfg.get("panel_id"),
            pretty=bool(dashboard_cfg.get("pretty", False)),
        )

    bundle_cfg = config.get("bundle")
    bundle_output: ObservabilityBundleConfig | None = None
    if bundle_cfg:
        output_dir = _expand_path(bundle_cfg.get("output_dir"))
        if output_dir is None:
            raise ValueError("Sekcja bundle wymaga pola output_dir")
        sources_raw = bundle_cfg.get("sources")
        sources: Sequence[AssetSource] | None = None
        if sources_raw:
            parsed: list[AssetSource] = []
            for entry in sources_raw:
                if not isinstance(entry, Mapping):
                    raise ValueError("Pozycje bundle.sources muszą być mapowaniem z category/root")
                category = entry.get("category")
                root = _expand_path(entry.get("root"))
                if not category or root is None:
                    raise ValueError("Źródło bundle wymaga pól category i root")
                parsed.append(AssetSource(category=str(category), root=root))
            sources = tuple(parsed)
        bundle_output = ObservabilityBundleConfig(
            output_dir=output_dir,
            bundle_name=bundle_cfg.get("bundle_name", "stage6-observability"),
            sources=sources,
            include=tuple(bundle_cfg.get("include", ())) or None,
            exclude=tuple(bundle_cfg.get("exclude", ())) or None,
            metadata=_as_dict(bundle_cfg.get("metadata")),
            verify=bool(bundle_cfg.get("verify", True)),
        )

    signing_key, signing_key_id = _resolve_signing(config.get("signing"))

    return ObservabilityCycleConfig(
        definitions_path=definitions,
        metrics_path=metrics,
        slo=slo_output,
        overrides=overrides_output,
        dashboard=dashboard_output,
        bundle=bundle_output,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )


def _parse_resilience(config: Mapping[str, Any] | None) -> ResilienceCycleConfig | None:
    if not config:
        return None

    bundle_section = config.get("bundle") or {}
    source = _expand_path(bundle_section.get("source"))
    output_dir = _expand_path(bundle_section.get("output_dir"))
    if source is None or output_dir is None:
        raise ValueError("Resilience.bundle wymaga pól source oraz output_dir")
    bundle_config = ResilienceBundleConfig(
        source=source,
        output_dir=output_dir,
        bundle_name=bundle_section.get("bundle_name", "stage6-resilience"),
        include=tuple(bundle_section.get("include", ())) or None,
        exclude=tuple(bundle_section.get("exclude", ())) or None,
        metadata=_as_dict(bundle_section.get("metadata")),
    )

    audit_section = config.get("audit") or {}
    audit_json = _expand_path(audit_section.get("json"))
    if audit_json is None:
        raise ValueError("Resilience.audit wymaga pola json")
    policy_path = _expand_path(audit_section.get("policy"))
    policy = load_policy(policy_path) if policy_path else None
    audit_config = AuditConfig(
        json_path=audit_json,
        csv_path=_expand_path(audit_section.get("csv")),
        signature_path=_expand_path(audit_section.get("signature")),
        require_signature=bool(audit_section.get("require_signature", False)),
        verify_signature=bool(audit_section.get("verify_signature", True)),
        policy=policy,
    )

    failover_section = config.get("failover") or {}
    plan_path = _expand_path(failover_section.get("plan"))
    failover_json = _expand_path(failover_section.get("json"))
    if plan_path is None or failover_json is None:
        raise ValueError("Resilience.failover wymaga pól plan oraz json")
    failover_config = FailoverConfig(
        plan_path=plan_path,
        json_path=failover_json,
        csv_path=_expand_path(failover_section.get("csv")),
        signature_path=_expand_path(failover_section.get("signature")),
    )

    self_heal_section = config.get("self_healing")
    self_healing_config: SelfHealingConfig | None = None
    if self_heal_section:
        rules = _expand_path(self_heal_section.get("rules"))
        output = _expand_path(self_heal_section.get("output"))
        if rules is None or output is None:
            raise ValueError("Resilience.self_healing wymaga pól rules i output")
        self_healing_config = SelfHealingConfig(
            rules_path=rules,
            output_path=output,
            signature_path=_expand_path(self_heal_section.get("signature")),
            mode=str(self_heal_section.get("mode", "plan")),
        )

    signing_key, signing_key_id = _resolve_signing(config.get("signing"))
    audit_hmac_key, _ = _resolve_signing(config.get("audit_hmac"), allow_key_id=False)

    return ResilienceCycleConfig(
        bundle=bundle_config,
        audit=audit_config,
        failover=failover_config,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
        audit_hmac_key=audit_hmac_key,
        self_healing=self_healing_config,
    )


def _parse_portfolio(config: Mapping[str, Any] | None) -> tuple[PortfolioCycleConfig | None, PortfolioGovernor | None]:
    if not config:
        return None, None

    core_path = _expand_path(config.get("core_config")) or Path("config/core.yaml")
    environment = config.get("environment")
    governor_name = config.get("governor")
    if not environment or not governor_name:
        raise ValueError("Portfolio sekcja wymaga pól environment oraz governor")

    core_config = load_core_config(core_path)
    if environment not in core_config.environments:
        raise ValueError(f"Środowisko {environment} nie istnieje w konfiguracji core.yaml")
    if governor_name not in core_config.portfolio_governors:
        raise ValueError(f"PortfolioGovernor {governor_name} nie istnieje w konfiguracji core.yaml")

    governor_cfg = core_config.portfolio_governors[governor_name]

    inputs_cfg = config.get("inputs") or {}
    allocations = _expand_path(inputs_cfg.get("allocations"))
    market_intel = _expand_path(inputs_cfg.get("market_intel"))
    portfolio_value = inputs_cfg.get("portfolio_value")
    if allocations is None or market_intel is None or portfolio_value is None:
        raise ValueError(
            "Portfolio.inputs wymaga pól allocations, market_intel oraz portfolio_value"
        )

    fallback_dirs = tuple(
        _expand_path(item) for item in inputs_cfg.get("fallback_dirs", ()) if item
    )
    required_symbols = tuple(inputs_cfg.get("market_intel_required", ())) or None

    inputs = PortfolioCycleInputs(
        allocations_path=allocations,
        market_intel_path=market_intel,
        portfolio_value=float(portfolio_value),
        slo_report_path=_expand_path(inputs_cfg.get("slo_report")),
        stress_report_path=_expand_path(inputs_cfg.get("stress_report")),
        fallback_directories=tuple(filter(None, fallback_dirs)),
        market_intel_required_symbols=required_symbols,
        market_intel_max_age=_minutes_to_timedelta(inputs_cfg.get("market_intel_max_age")),
        slo_max_age=_minutes_to_timedelta(inputs_cfg.get("slo_max_age")),
        stress_max_age=_minutes_to_timedelta(inputs_cfg.get("stress_max_age")),
    )

    output_cfg = config.get("output") or {}
    summary_path = _expand_path(output_cfg.get("summary")) or _default_portfolio_summary_path(
        governor_name
    )
    output = PortfolioCycleOutputConfig(
        summary_path=summary_path,
        signature_path=_expand_path(output_cfg.get("signature")),
        csv_path=_expand_path(output_cfg.get("csv")),
        pretty_json=bool(output_cfg.get("pretty", True)),
    )

    metadata = {"environment": environment, "governor": governor_name}
    metadata.update(config.get("metadata", {}))
    log_context = {"environment": environment, "governor": governor_name}
    log_context.update(config.get("log_context", {}))

    signing_key, signing_key_id = _resolve_signing(config.get("signing"))

    cycle_config = PortfolioCycleConfig(
        inputs=inputs,
        output=output,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
        metadata=metadata,
        log_context=log_context,
    )

    decision_section = config.get("decision_log") or {}
    skip_log = bool(decision_section.get("skip", False))
    decision_log_path: Path | None = None
    decision_log_kwargs: Mapping[str, Any] = {}
    decision_log: PortfolioDecisionLog | None = None
    if not skip_log:
        configured_path, decision_log_kwargs = resolve_decision_log_config(core_config)
        decision_log_path = _expand_path(decision_section.get("path"))
        if decision_log_path is None:
            decision_log_path = configured_path or default_decision_log_path(governor_name)
        if decision_log_path:
            decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            decision_log = PortfolioDecisionLog(
                jsonl_path=decision_log_path,
                **decision_log_kwargs,
            )

    governor = PortfolioGovernor(governor_cfg, decision_log=decision_log)
    return cycle_config, governor


def _build_hypercare_config(data: Mapping[str, Any]) -> tuple[Stage6HypercareConfig, PortfolioGovernor | None]:
    summary_cfg = data.get("summary") or {}
    output_path = _expand_path(summary_cfg.get("path")) or _default_summary_path()
    signature_path = _expand_path(summary_cfg.get("signature"))
    metadata = summary_cfg.get("metadata")
    signing_key, signing_key_id = _resolve_signing(summary_cfg.get("signing"))

    observability = _parse_observability(data.get("observability"))
    resilience = _parse_resilience(data.get("resilience"))
    portfolio_config, governor = _parse_portfolio(data.get("portfolio"))

    config = Stage6HypercareConfig(
        output_path=output_path,
        signature_path=signature_path,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
        metadata=_as_dict(metadata),
        observability=observability,
        resilience=resilience,
        portfolio=portfolio_config,
    )
    return config, governor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uruchamia automatyczny cykl hypercare Stage6 na podstawie konfiguracji YAML/JSON.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Plik YAML/JSON z konfiguracją cyklu (sekcje summary/observability/resilience/portfolio)",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config_path = Path(args.config).expanduser()
        config_data = _load_text_config(config_path)
        hypercare_config, governor = _build_hypercare_config(config_data)

        cycle = Stage6HypercareCycle(
            hypercare_config,
            portfolio_governor=governor,
            observability_factory=ObservabilityHypercareCycle,
            resilience_factory=ResilienceHypercareCycle,
        )
        result = cycle.run()

        print(f"Raport hypercare Stage6 zapisany w {result.output_path}")
        if result.signature_path:
            print(
                "Podpis HMAC zapisany w",
                result.signature_path,
                f"(key_id={hypercare_config.signing_key_id or 'brak'})",
            )

        components = result.payload.get("components", {})
        if isinstance(components, Mapping):
            for name, payload in components.items():
                status = payload.get("status") if isinstance(payload, Mapping) else "?"
                print(f" - {name}: {status}")

        if result.payload.get("issues"):
            print("Wykryto problemy:")
            for issue in result.payload["issues"]:
                print(f"  * {issue}")
            return 2
        if result.payload.get("warnings"):
            print("Wygenerowano ostrzeżenia:")
            for warning in result.payload["warnings"]:
                print(f"  - {warning}")
        return 0
    except Exception as exc:  # pragma: no cover - obsługa błędów CLI
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(run())

