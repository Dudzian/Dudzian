from pathlib import Path

from bot_core.config import load_core_config
import scripts.slo_monitor as slo_monitor
import scripts.run_stage6_observability_cycle as stage6_cycle


def test_repository_slo_definitions_are_loadable() -> None:
    config = load_core_config(Path("config/core.yaml"))
    assert config.observability is not None
    slo = config.observability.slo
    assert slo, "Oczekiwano co najmniej jednej definicji SLO"
    assert set(slo).issuperset({"decision_latency", "trade_cost", "fill_rate"})
    definition = slo["decision_latency"]
    assert definition.metric == "bot_core_decision_latency_ms"
    assert definition.window_minutes == 1440.0


def test_slo_monitor_defaults_to_repository_definitions(tmp_path: Path) -> None:
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text("{}", encoding="utf-8")

    args = slo_monitor._parse_args(["evaluate", "--metrics", str(metrics_file)])

    assert args.definitions == "config/observability/slo.yml"
    assert Path(args.definitions).is_file()


def test_stage6_cycle_defaults_to_repository_definitions(tmp_path: Path) -> None:
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text("{}", encoding="utf-8")

    parser = stage6_cycle.build_parser()
    args = parser.parse_args(
        [
            "--metrics",
            str(metrics_file),
            "--slo-json",
            str(tmp_path / "out.json"),
            "--slo-csv",
            str(tmp_path / "out.csv"),
            "--skip-overrides",
            "--skip-bundle",
        ]
    )

    assert args.definitions == "config/observability/slo.yml"
    assert Path(args.definitions).is_file()
