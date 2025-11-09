from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml


class _StubResult:
    def __init__(self, payload: Mapping[str, Any], output_path: Path, signature_path: Path) -> None:
        self.payload = payload
        self.output_path = output_path
        self.signature_path = signature_path
        self.observability = None
        self.resilience = None
        self.portfolio = None


class _FakeCoreConfig:
    def __init__(self) -> None:
        self.environments = {"binance_paper": {}}
        self.portfolio_governors = {
            "stage6_core": {
                "portfolio_id": "stage6_core",
                "assets": [
                    {"symbol": "BTC_USDT", "target_weight": 1.0},
                ],
            }
        }


class _FakePortfolioDecisionLog:
    def __init__(self, jsonl_path: Path, **_: Any) -> None:  # pragma: no cover - simple stub
        self.jsonl_path = jsonl_path


class _FakePortfolioGovernor:
    def __init__(self, cfg: Any, decision_log: Any | None = None) -> None:  # pragma: no cover - stub
        self.cfg = cfg
        self.decision_log = decision_log


@pytest.fixture(name="repo_root")
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _touch(relative_path: str, base: Path, content: str = "{}\n") -> Path:
    path = base / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_stage6_hypercare_cycle_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    config_path = repo_root / "config/stage6/hypercare.yaml"

    monkeypatch.chdir(tmp_path)

    # Prepare key material referenced by the configuration.
    for name in (
        "stage6_summary.key",
        "observability.key",
        "resilience.key",
        "resilience_audit.key",
        "stage6_portfolio.key",
    ):
        _touch(f"secrets/hmac/{name}", tmp_path, content="test-key")

    # Mock artefacts required by the cycle inputs.
    slo_report = _touch("var/audit/observability/slo_report.json", tmp_path)
    stress_report = _touch("var/audit/stage6/stress_lab_report.json", tmp_path)
    resilience_audit = _touch("var/audit/resilience/audit_summary.json", tmp_path)

    _touch("var/audit/observability/metrics.json", tmp_path)
    _touch("config/observability/slo.yml", tmp_path, content="service_level_objectives: []\n")
    _touch(
        "deploy/grafana/provisioning/dashboards/stage6_resilience_operations.json",
        tmp_path,
        content="{\n  \"dashboard\": {}\n}\n",
    )
    _touch("config/stage6/resilience_self_heal.json", tmp_path)
    _touch("data/stage6/resilience/failover_plan.json", tmp_path)
    _touch("var/audit/stage6/market_intel.json", tmp_path)

    # Minimal core configuration so the loader succeeds.
    _touch(
        "config/core.yaml",
        tmp_path,
        content=("environments:\n  binance_paper: {}\nportfolio_governors:\n  stage6_core: {}\n"),
    )

    required_inputs = {
        "slo": slo_report,
        "stress": stress_report,
        "resilience": resilience_audit,
    }

    from bot_core.runtime import stage6_hypercare as stage6_module

    captured: dict[str, Any] = {}

    class _StubCycle:
        def __init__(
            self,
            cfg: stage6_module.Stage6HypercareConfig,
            *,
            portfolio_governor: Any | None = None,
            **_: Any,
        ) -> None:
            assert portfolio_governor is not None, "Portfolio governor should be provided"
            captured["portfolio_governor"] = portfolio_governor
            self._config = cfg
            self._required = required_inputs

        def _assert_config_paths(self) -> None:
            assert self._config.observability is not None
            assert self._config.resilience is not None
            assert self._config.portfolio is not None

            assert self._config.signing_key == b"test-key"
            assert self._config.signing_key_id == "stage6-summary"
            assert (
                self._config.output_path.resolve(strict=False)
                == self._required["resilience"].parent.parent / "stage6" / "hypercare_summary.json"
            )
            assert (
                (self._config.signature_path or self._config.output_path.with_suffix(".sig")).resolve(strict=False)
                == self._required["resilience"].parent.parent / "stage6" / "hypercare_summary.sig"
            )

            observability_cfg = self._config.observability
            assert observability_cfg.slo.json_path.resolve(strict=False) == self._required["slo"].resolve()
            assert observability_cfg.metrics_path.resolve(strict=False) == (
                tmp_path / "var/audit/observability/metrics.json"
            )
            assert observability_cfg.definitions_path.resolve(strict=False) == (
                tmp_path / "config/observability/slo.yml"
            )

            resilience_cfg = self._config.resilience
            assert resilience_cfg.audit.json_path.resolve(strict=False) == self._required["resilience"].resolve()
            assert resilience_cfg.failover.plan_path.resolve(strict=False) == (
                tmp_path / "data/stage6/resilience/failover_plan.json"
            )

            portfolio_cfg = self._config.portfolio
            assert portfolio_cfg.inputs.slo_report_path.resolve(strict=False) == self._required["slo"].resolve()
            assert portfolio_cfg.inputs.stress_report_path.resolve(strict=False) == self._required["stress"].resolve()

        def run(self) -> _StubResult:
            for path in self._required.values():
                assert path.exists(), f"Missing artefact: {path}"

            self._assert_config_paths()

            payload = {
                "overall_status": "ok",
                "issues": [],
                "warnings": [],
                "components": {
                    "observability": {
                        "status": "ok",
                        "slo_report": self._required["slo"].as_posix(),
                    },
                    "resilience": {
                        "status": "ok",
                        "audit": self._required["resilience"].as_posix(),
                    },
                    "portfolio": {
                        "status": "ok",
                        "stress_report": self._required["stress"].as_posix(),
                    },
                },
            }

            output_path = self._config.output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload), encoding="utf-8")

            signature_path = self._config.signature_path or output_path.with_suffix(".sig")
            signature_path.parent.mkdir(parents=True, exist_ok=True)
            signature_payload = {
                "key_id": self._config.signing_key_id or "stage6-summary",
                "signature": "stub",
            }
            signature_path.write_text(json.dumps(signature_payload), encoding="utf-8")

            return _StubResult(payload, output_path, signature_path)

    monkeypatch.setattr(stage6_module, "Stage6HypercareCycle", _StubCycle)

    import bot_core.config as config_module
    import bot_core.portfolio as portfolio_module

    monkeypatch.setattr(config_module, "load_core_config", lambda _path: _FakeCoreConfig())
    monkeypatch.setattr(portfolio_module, "PortfolioGovernor", _FakePortfolioGovernor)
    monkeypatch.setattr(portfolio_module, "PortfolioDecisionLog", _FakePortfolioDecisionLog)
    monkeypatch.setattr(portfolio_module, "resolve_decision_log_config", lambda _cfg: (None, {}))

    argv = [
        "run_stage6_hypercare_cycle.py",
        "--config",
        config_path.as_posix(),
    ]

    
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("scripts.run_stage6_hypercare_cycle", run_name="__main__")

    assert exit_info.value.code == 0

    summary_path = Path("var/audit/stage6/hypercare_summary.json")
    signature_path = Path("var/audit/stage6/hypercare_summary.sig")
    assert summary_path.exists()
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "ok"
    assert report["issues"] == []
    assert report["warnings"] == []
    assert set(report["components"].keys()) == {"observability", "resilience", "portfolio"}
    for name, component in report["components"].items():
        assert component["status"] == "ok", f"Unexpected status for {name}: {component}"

    signature_payload = json.loads(signature_path.read_text(encoding="utf-8"))
    assert signature_payload["key_id"] == "stage6-summary"
    assert signature_payload["signature"] == "stub"

    assert isinstance(captured["portfolio_governor"], _FakePortfolioGovernor)

    allocations_path = Path("var/audit/portfolio/allocations_stage6.yaml")
    assert allocations_path.exists()
    allocations_payload = yaml.safe_load(allocations_path.read_text(encoding="utf-8"))
    assert allocations_payload == {"BTC_USDT": 1.0}

    summary_path.unlink()
    signature_path.unlink()
    allocations_path.unlink()
    assert not summary_path.exists()
    assert not signature_path.exists()
    assert not allocations_path.exists()

