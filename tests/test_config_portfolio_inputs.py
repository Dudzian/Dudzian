from __future__ import annotations

from pathlib import Path

from bot_core.config.loader import load_core_config


def test_load_core_config_reads_portfolio_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        instrument_universes: {}
        environments: {}
        reporting: {}
        alerts: {}
        portfolio_governors:
          stage6:
            portfolio_id: stage6
            drift_tolerance:
              absolute: 0.02
              relative: 0.3
            assets:
              - symbol: BTC_USDT
                target_weight: 0.6
        runtime:
          multi_strategy_schedulers:
            stage6:
              telemetry_namespace: runtime.stage6
              decision_log_category: runtime.stage6
              health_check_interval: 90
              portfolio_governor: stage6
              portfolio_inputs:
                slo_report_path: var/audit/observability/slo_report.json
                slo_max_age_minutes: 60
                stress_lab_report_path: var/audit/stage6/stress_lab_report.json
                stress_max_age_minutes: 240
              schedules: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    scheduler = config.multi_strategy_schedulers["stage6"]

    assert scheduler.portfolio_governor == "stage6"
    assert scheduler.portfolio_inputs is not None
    assert scheduler.portfolio_inputs.slo_report_path == "var/audit/observability/slo_report.json"
    assert scheduler.portfolio_inputs.slo_max_age_minutes == 60
    assert (
        scheduler.portfolio_inputs.stress_lab_report_path
        == "var/audit/stage6/stress_lab_report.json"
    )
    assert scheduler.portfolio_inputs.stress_max_age_minutes == 240
