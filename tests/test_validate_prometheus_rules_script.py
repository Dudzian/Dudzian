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
    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(rules_file),
            "--metric-prefix",
            "bot_core_multi_strategy",
        ]
    )
    assert exit_code == 0
    assert "Walidacja reguł" in capsys.readouterr().out


def test_validation_fails_without_required_label(rules_file: Path) -> None:
    data = yaml.safe_load(rules_file.read_text(encoding="utf-8"))
    data["groups"][0]["rules"][0]["labels"].pop("team", None)
    rules_file.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(rules_file),
        ]
    )
    assert exit_code == 1


def test_validation_errors_on_missing_metric_prefix(rules_file: Path) -> None:
    content = rules_file.read_text(encoding="utf-8").replace(
        "bot_core_multi_strategy_latency_ms", "custom_metric_latency_ms"
    )
    rules_file.write_text(content, encoding="utf-8")
    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(rules_file),
            "--metric-prefix",
            "bot_core_multi_strategy",
        ]
    )
    assert exit_code == 1


def test_validation_returns_error_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yml"
    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(missing),
        ]
    )
    assert exit_code == 2


def test_validation_rejects_cloud_worker_rules_without_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "cloud.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerMissingTemplates
        expr: max_over_time(bot_cloud_worker_last_error[5m]) > 0
        for: 5m
        labels:
          severity: critical
          team: hypercare
        annotations:
          summary: "Alert dla workera"
          description: "Ten opis ma wystarczającą długość, ale brakuje szablonów."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
            "--metric-prefix",
            "bot_cloud_worker",
        ]
    )

    assert exit_code == 1


def test_stage6_rules_file_passes_validation() -> None:
    rules_path = Path("deploy/prometheus/stage6_alerts.yaml")
    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(rules_path),
            "--metric-prefix",
            "bot_core_stage6",
            "--metric-prefix",
            "bot_core_multi_strategy",
        ]
    )
    assert exit_code == 0


def test_cloud_worker_rules_require_owner_and_channel(tmp_path: Path) -> None:
    path = tmp_path / "cloud_missing_labels.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerMissingOwner
        expr: max_over_time(bot_cloud_worker_status[5m]) == 0
        for: 5m
        labels:
          severity: warning
          team: hypercare
          service: cloud-orchestrator
        annotations:
          summary: "Worker {{ $labels.worker }} w stanie degraded"
          description: "Opis zawiera {{ $labels.worker }} i ma odpowiednią długość."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
        ]
    )

    assert exit_code == 1


def test_cloud_worker_rules_file_passes_validation() -> None:
    rules_path = Path("deploy/prometheus/rules/cloud_worker_alerts.yml")
    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(rules_path),
        ]
    )
    assert exit_code == 0


def test_cloud_worker_rules_require_correct_channel_and_owner(tmp_path: Path) -> None:
    path = tmp_path / "cloud_wrong_channel.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerWrongChannel
        expr: max_over_time(bot_cloud_worker_status[5m]) == 0
        for: 5m
        labels:
          severity: warning
          team: hypercare
          service: cloud-orchestrator
          channel: CloudAlertService
          owner: platform
        annotations:
          summary: "Worker {{ $labels.worker }} w stanie degraded"
          description: "Opis zawiera {{ $labels.worker }} i ma odpowiednią długość."
      - alert: CloudWorkerErrorWrongChannel
        expr: max_over_time(bot_cloud_worker_last_error[5m]) > 0
        for: 5m
        labels:
          severity: critical
          team: hypercare
          service: cloud-orchestrator
          channel: HyperCare
          owner: cloud-alerts
        annotations:
          summary: "Błąd {{ $labels.error }} na workerze {{ $labels.worker }}"
          description: "Opis zawiera {{ $labels.worker }} i {{ $labels.error }} oraz jest wystarczająco długi."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
        ]
    )

    assert exit_code == 1


def test_cloud_worker_rules_require_hypercare_team_and_service(tmp_path: Path) -> None:
    path = tmp_path / "cloud_wrong_team.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerWrongTeam
        expr: max_over_time(bot_cloud_worker_status[5m]) == 0
        for: 5m
        labels:
          severity: warning
          team: platform
          service: orchestration
          channel: HyperCare
          owner: cloud-alerts
        annotations:
          summary: "Worker {{ $labels.worker }} w stanie degraded"
          description: "Cloud worker {{ $labels.worker }} jest degraded przez 5 minut."
      - alert: CloudWorkerErrorWrongService
        expr: max_over_time(bot_cloud_worker_last_error[10m]) > 0
        for: 10m
        labels:
          severity: critical
          team: hypercare
          service: orchestration
          channel: CloudAlertService
          owner: cloud-alerts
        annotations:
          summary: "Błąd {{ $labels.error }} na workerze {{ $labels.worker }}"
          description: "Błąd {{ $labels.error }} utrzymuje się na {{ $labels.worker }}."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
        ]
    )

    assert exit_code == 1


def test_cloud_worker_status_requires_worker_placeholder_in_summary(tmp_path: Path) -> None:
    path = tmp_path / "cloud_missing_worker_summary.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerMissingWorkerInSummary
        expr: max_over_time(bot_cloud_worker_status[5m]) == 0
        for: 5m
        labels:
          severity: warning
          team: hypercare
          service: cloud-orchestrator
          channel: HyperCare
          owner: cloud-alerts
        annotations:
          summary: "Worker w stanie degraded"
          description: "Worker {{ $labels.worker }} ma status degraded i wymaga interwencji."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
        ]
    )

    assert exit_code == 1


def test_cloud_worker_error_requires_worker_and_error_in_summary(tmp_path: Path) -> None:
    path = tmp_path / "cloud_missing_error_summary.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerErrorMissingSummaryPlaceholders
        expr: max_over_time(bot_cloud_worker_last_error[10m]) > 0
        for: 10m
        labels:
          severity: critical
          team: hypercare
          service: cloud-orchestrator
          channel: CloudAlertService
          owner: cloud-alerts
        annotations:
          summary: "Krytyczny błąd wykryty na workerze"
          description: "Błąd {{ $labels.error }} na workerze {{ $labels.worker }} trwa od 10 minut."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
        ]
    )

    assert exit_code == 1


def test_cloud_worker_status_requires_worker_in_description(tmp_path: Path) -> None:
    path = tmp_path / "cloud_missing_worker_description.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerMissingWorkerInDescription
        expr: max_over_time(bot_cloud_worker_status[5m]) == 0
        for: 5m
        labels:
          severity: warning
          team: hypercare
          service: cloud-orchestrator
          channel: HyperCare
          owner: cloud-alerts
        annotations:
          summary: "Worker {{ $labels.worker }} w stanie degraded"
          description: "Opis nie zawiera wymaganych placeholderów."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
        ]
    )

    assert exit_code == 1


def test_cloud_worker_error_requires_worker_and_error_in_description(tmp_path: Path) -> None:
    path = tmp_path / "cloud_missing_error_description.yml"
    path.write_text(
        """
groups:
  - name: cloud_worker
    rules:
      - alert: CloudWorkerErrorMissingDescriptionPlaceholders
        expr: max_over_time(bot_cloud_worker_last_error[10m]) > 0
        for: 10m
        labels:
          severity: critical
          team: hypercare
          service: cloud-orchestrator
          channel: CloudAlertService
          owner: cloud-alerts
        annotations:
          summary: "Błąd {{ $labels.error }} na workerze {{ $labels.worker }}"
          description: "Opis nie zawiera wymaganych placeholderów, powinien wprost wskazywać błąd i workera."
        """.strip(),
        encoding="utf-8",
    )

    exit_code = validate_prometheus_rules.main(
        [
            "--rules",
            str(path),
        ]
    )

    assert exit_code == 1
