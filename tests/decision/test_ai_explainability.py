from __future__ import annotations

from bot_core.ai import build_explainability_report
from bot_core.ai.manager import AIManager
from bot_core.ai.models import ModelScore


class StubInference:
    def __init__(self) -> None:
        self.last_features: dict[str, float] | None = None

    def score(self, features: dict[str, float]) -> ModelScore:
        self.last_features = dict(features)
        return ModelScore(expected_return_bps=12.5, success_probability=0.72)

    def explain(self, features: dict[str, float]) -> dict[str, float]:
        return {name: float(index + 1) for index, name in enumerate(features)}


def test_build_explainability_report_uses_fallback() -> None:
    inference = StubInference()
    features = {"momentum": 1.2, "volatility": 0.4}
    score = ModelScore(expected_return_bps=15.0, success_probability=0.66)

    report = build_explainability_report(
        inference,
        features,
        model_name="stub",
        score=score,
    )

    assert report is not None
    assert report.method in {"perturbation", "shap"}
    assert [attr.name for attr in report.attributions][:2] == ["volatility", "momentum"]
    assert report.expected_return_bps == 15.0
    assert report.success_probability == 0.66


def test_ai_manager_payload_contains_explainability(tmp_path) -> None:
    manager = AIManager(model_dir=tmp_path)
    inference = StubInference()
    manager._decision_inferences["__default__"] = inference
    manager._decision_default_name = "__default__"

    payload = manager.build_decision_engine_payload(
        strategy="trend",
        action="enter",
        risk_profile="balanced",
        symbol="BTCUSDT",
        notional=10_000.0,
        features={"momentum": 1.2, "volatility": 0.3},
    )

    assert "explainability" in payload
    assert "explainability_json" in payload
    explainability = payload["explainability"]
    assert explainability["model"] == "__default__"
    assert explainability["method"] in {"perturbation", "shap"}
    assert payload["ai"]["explainability"]["top_features"]
    candidate_meta = payload["candidate"]["metadata"]["decision_engine"]
    assert "explainability_json" in candidate_meta
