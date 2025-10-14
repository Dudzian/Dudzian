from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import validate_prometheus_rules


@pytest.fixture()
def rules_file(tmp_path: Path) -> Path:
    path = tmp_path / "rules.yml"
    path.write_text(
        """
groups:
  - name: multi_strategy_runtime
    rules:
      - alert: MultiStrategySchedulerLatencyHigh
        expr: max_over_time(bot_core_multi_strategy_latency_ms[2m]) > 250
        for: 2m
        labels:
          severity: warning
          team: trading-ops
        annotations:
          summary: "Opóźnienie pętli scheduler-a przekracza próg"
          description: "Dłuższy opis zawierający co najmniej dziesięć znaków."
      - alert: MultiStrategySecondaryDelay
        expr: max_over_time(bot_core_multi_strategy_secondary_delay_ms[5m]) > 450
        for: 3m
        labels:
          severity: critical
          team: trading-ops
        annotations:
          summary: "Opóźnienie giełdy secondary"
          description: "Opis przekraczający wymagane minimum znaków."
        """.strip(),
        encoding="utf-8",
    )
    return path


def test_validation_passes_for_valid_rules(
    rules_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = validate_prometheus_rules.main([
        "--rules",
        str(rules_file),
        "--metric-prefix",
        "bot_core_multi_strategy",
    ])
    assert exit_code == 0
    assert "Walidacja reguł" in capsys.readouterr().out


def test_validation_fails_without_required_label(rules_file: Path) -> None:
    data = yaml.safe_load(rules_file.read_text(encoding="utf-8"))
    data["groups"][0]["rules"][0]["labels"].pop("team", None)
    rules_file.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    exit_code = validate_prometheus_rules.main([
        "--rules",
        str(rules_file),
    ])
    assert exit_code == 1


def test_validation_errors_on_missing_metric_prefix(rules_file: Path) -> None:
    content = rules_file.read_text(encoding="utf-8").replace(
        "bot_core_multi_strategy_latency_ms", "custom_metric_latency_ms"
    )
    rules_file.write_text(content, encoding="utf-8")
    exit_code = validate_prometheus_rules.main([
        "--rules",
        str(rules_file),
        "--metric-prefix",
        "bot_core_multi_strategy",
    ])
    assert exit_code == 1


def test_validation_returns_error_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yml"
    exit_code = validate_prometheus_rules.main([
        "--rules",
        str(missing),
    ])
    assert exit_code == 2
