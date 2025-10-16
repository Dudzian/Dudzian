from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_stage6_hypercare_cycle


class _FakeResult:
    def __init__(self, output_path: Path, signature_path: Path, payload: dict[str, object]) -> None:
        self.output_path = output_path
        self.signature_path = signature_path
        self.payload = payload
        self.observability = None
        self.resilience = None
        self.portfolio = None


def test_cli_parses_minimal_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary_path = tmp_path / "summary.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"summary:\n  path: {summary_path.as_posix()}\n",
        encoding="utf-8",
    )

    def fake_cycle_factory(config: run_stage6_hypercare_cycle.Stage6HypercareConfig, **_: object) -> object:
        assert config.output_path == summary_path

        class _FakeCycle:
            def __init__(self, cfg: run_stage6_hypercare_cycle.Stage6HypercareConfig) -> None:
                self._config = cfg

            def run(self) -> _FakeResult:
                payload = {
                    "overall_status": "ok",
                    "components": {
                        "observability": {"status": "skipped"},
                        "resilience": {"status": "skipped"},
                        "portfolio": {"status": "skipped"},
                    },
                    "issues": [],
                    "warnings": [],
                }
                output = self._config.output_path
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(payload), encoding="utf-8")
                signature_path = output.with_suffix(".sig")
                signature_path.write_text("{}", encoding="utf-8")
                return _FakeResult(output, signature_path, payload)

        return _FakeCycle(config)

    monkeypatch.setattr(run_stage6_hypercare_cycle, "Stage6HypercareCycle", fake_cycle_factory)

    exit_code = run_stage6_hypercare_cycle.run(["--config", str(config_path)])
    assert exit_code == 0
    assert summary_path.exists()
