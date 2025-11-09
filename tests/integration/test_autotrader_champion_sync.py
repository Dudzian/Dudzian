from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from bot_core.ai.inference import ModelRepository
from bot_core.ai.models import ModelArtifact
from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.ai.validation import ModelQualityReport, record_model_quality_report
from bot_core.auto_trader import AutoTrader
from bot_core.auto_trader.decision_scheduler import AutoTraderDecisionScheduler


def _make_artifact(*, metadata: dict[str, object] | None = None) -> ModelArtifact:
    payload = dict(metadata or {})
    metrics = {
        "summary": {"mae": 1.25, "directional_accuracy": 0.6},
        "train": {},
        "validation": {},
        "test": {},
    }
    return ModelArtifact(
        feature_names=("f1", "f2"),
        model_state={"weights": [0.1, 0.2], "bias": 0.0},
        trained_at=datetime.now(timezone.utc),
        metrics=metrics,
        metadata=payload,
        target_scale=1.0,
        training_rows=64,
        validation_rows=0,
        test_rows=0,
        feature_scalers={"f1": (0.0, 1.0), "f2": (0.0, 1.0)},
        decision_journal_entry_id=None,
        backend="builtin",
    )


def _build_report(version: str, directional: float, mae: float, status: str = "improved") -> ModelQualityReport:
    metrics = {
        "summary": {
            "directional_accuracy": directional,
            "mae": mae,
        }
    }
    return ModelQualityReport(
        model_name="decision_engine",
        version=version,
        evaluated_at=datetime.now(timezone.utc),
        metrics=metrics,
        status=status,
    )


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.logs: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, dict(payload)))

    def log(self, message: str, *_, **kwargs: object) -> None:
        self.logs.append((message, dict(kwargs)))


class _Var:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _GUI:
    def __init__(self) -> None:
        self.timeframe_var = _Var("1h")
        self.ai_mgr = None

    def is_demo_mode_active(self) -> bool:
        return True


class _Provider:
    def __init__(self) -> None:
        index = pd.date_range(datetime.now(timezone.utc), periods=120, freq="H")
        self._df = pd.DataFrame(
            {
                "open": pd.Series(range(120), index=index).astype(float),
                "high": pd.Series(range(1, 121), index=index).astype(float),
                "low": pd.Series(range(120), index=index).astype(float),
                "close": pd.Series(range(1, 121), index=index).astype(float),
                "volume": pd.Series([1.0] * 120, index=index),
            }
        )

    def get_historical(self, symbol: str, timeframe: str, limit: int = 256) -> pd.DataFrame:
        del symbol, timeframe, limit
        return self._df.copy(deep=True)


class _AIManagerChampionStub:
    def __init__(self) -> None:
        self.ai_threshold_bps = 5.0
        self.is_degraded = False
        self.loaded: list[tuple[str, Path]] = []
        self.active_models: dict[str, str] = {}

    def run_due_training_jobs(self) -> None:  # pragma: no cover - stub hook
        return None

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame, **_: object) -> MarketRegimeAssessment:
        del market_data
        return MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.8,
            risk_score=0.2,
            metrics={"volatility": 0.1},
            symbol=symbol,
        )

    def get_regime_summary(self, symbol: str) -> None:
        del symbol
        return None

    def predict_series(self, symbol: str, df: pd.DataFrame, **_: object) -> pd.Series:
        del symbol, df
        return pd.Series([0.003], index=[datetime.now(timezone.utc)])

    def predict_probability(self, **_: object) -> float:
        return 0.65

    def score_decision_features(self, features: dict[str, float]) -> SimpleNamespace:
        del features
        return SimpleNamespace(
            expected_return_bps=18.0,
            success_probability=0.6,
            model_name="stub",
        )

    def build_decision_engine_payload(self, **_: object) -> dict[str, object]:
        return {"ai": {"direction": "buy", "prediction_bps": 22.0}}

    def load_decision_artifact(
        self,
        name: str,
        artifact: Path | str,
        *,
        repository_root: Path | None = None,
        set_default: bool = False,
    ) -> SimpleNamespace:
        del repository_root, set_default
        path = Path(artifact)
        self.loaded.append((name, path))
        return SimpleNamespace(is_ready=lambda: True)

    def set_active_model(self, symbol: str, model_type: str | None) -> None:
        if model_type is None:
            self.active_models.pop(symbol, None)
        else:
            self.active_models[symbol] = str(model_type)


def _model_events(emitter: _Emitter) -> list[dict[str, object]]:
    return [payload for event, payload in emitter.events if event == "auto_trader.model_changed"]


def _publish(repository: ModelRepository, version: str) -> None:
    repository.publish(
        _make_artifact(),
        version=version,
        filename=f"model-{version}.json",
        aliases=("latest",),
        activate=True,
    )


def test_autotrader_synchronises_champion_and_fallback(tmp_path: Path) -> None:
    repo_root = tmp_path / "var" / "models"
    model_dir = repo_root / "decision_engine"
    repository = ModelRepository(model_dir)

    _publish(repository, "v1")

    quality_root = repo_root / "quality"
    record_model_quality_report(
        _build_report("v1", directional=0.62, mae=14.0),
        history_root=quality_root,
    )

    emitter = _Emitter()
    gui = _GUI()
    provider = _Provider()
    manager = _AIManagerChampionStub()

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        auto_trade_interval_s=0.0,
        enable_auto_trade=True,
        market_data_provider=provider,
        model_quality_dir=quality_root,
        champion_repository_root=repo_root,
        champion_model_map={"BTCUSDT": "decision_engine"},
    )
    trader.ai_manager = manager
    scheduler = AutoTraderDecisionScheduler(trader, interval_s=0.01)

    trader.run_cycle_once()
    scheduler._drain_model_change_events()

    events = _model_events(emitter)
    assert events[-1]["version"] == "v1"
    assert events[-1]["fallback"] is False
    assert "v1" in manager.loaded[-1][1].name

    _publish(repository, "v2")
    record_model_quality_report(
        _build_report("v2", directional=0.50, mae=18.0, status="ok"),
        history_root=quality_root,
    )

    champion_path = quality_root / "decision_engine" / "champion.json"
    payload = json.loads(champion_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("report"), dict):
        payload["report"]["status"] = "degraded"
        champion_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trader.run_cycle_once()
    scheduler._drain_model_change_events()

    events = _model_events(emitter)
    fallback_event = events[-1]
    assert fallback_event["version"] == "v2"
    assert fallback_event["fallback"] is True
    assert fallback_event.get("fallback_reason") == "champion_degraded"
    assert "v2" in fallback_event.get("challenger_versions", [])
    assert "v2" in manager.loaded[-1][1].name

    _publish(repository, "v3")
    record_model_quality_report(
        _build_report("v3", directional=0.70, mae=12.0, status="improved"),
        history_root=quality_root,
    )

    trader.run_cycle_once()
    scheduler._drain_model_change_events()

    events = _model_events(emitter)
    champion_event = events[-1]
    assert champion_event["version"] == "v3"
    assert champion_event.get("previous_version") == "v2"
    assert champion_event["fallback"] is False
    assert "v3" in manager.loaded[-1][1].name
