import asyncio
import json
import logging
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bot_core.ai import DataQualityCheck, FeatureDataset, FeatureVector, manager as ai_manager_module
from bot_core.ai.monitoring import DataCompletenessWatcher
from bot_core.runtime.journal import InMemoryTradingDecisionJournal
from tests._ai_manager_helpers import make_stub_model, positive_negative_predict


def _make_df(rows: int = 20) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="min")
    base = np.linspace(100.0, 110.0, rows)
    data = {
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.5,
        "volume": np.linspace(1_000.0, 2_000.0, rows),
    }
    return pd.DataFrame(data, index=index)


def test_ai_manager_rank_and_train_multiple_models(tmp_path, monkeypatch):
    calls: list[tuple[str, str, Path | None]] = []
    lock = threading.Lock()

    def _record(action: str, model_type: str, payload: Path | None = None) -> None:
        with lock:
            calls.append((action, model_type, payload))

    stub_model = make_stub_model(
        predict_fn=positive_negative_predict,
        init_hook=lambda model_type, model_dir: _record("init", model_type, model_dir),
        train_hook=lambda model_type: _record("train", model_type, None),
        predict_hook=lambda model_type: _record("predict", model_type, None),
        predict_series_hook=lambda model_type: _record("predict_series", model_type, None),
    )

    monkeypatch.setattr(ai_manager_module, "_AIModels", stub_model)

    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _make_df()

    evaluations = asyncio.run(
        manager.rank_models(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            folds=2,
            epochs=1,
            batch_size=1,
        )
    )
    assert [ev.model_type for ev in evaluations] == ["alpha", "beta"]

    results = asyncio.run(
        manager.train_all_models(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            epochs=1,
            batch_size=1,
        )
    )
    assert set(results) == {"alpha", "beta"}
    assert results["alpha"].hit_rate >= results["beta"].hit_rate

    with lock:
        init_dirs = {entry for action, model_type, entry in calls if action == "init" and model_type == "alpha"}
    assert tmp_path in init_dirs


def test_run_pipeline_sets_active_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=positive_negative_predict),
    )

    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _make_df(40)

    selection = asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            folds=2,
        )
    )

    assert selection.best_model == "alpha"
    assert manager.get_active_model("BTCUSDT") == "alpha"

    predictions = asyncio.run(
        manager.predict_series(
            "BTCUSDT",
            df,
        )
    )

    assert isinstance(predictions, pd.Series)
    assert np.allclose(predictions.values, 1.0)

    history = manager.get_pipeline_history("BTCUSDT")
    assert len(history) == 1
    record = history[0]
    assert record.best_model == "alpha"
    assert record.prediction_count == len(df)
    assert record.prediction_mean == 1.0
    assert manager.last_pipeline_selection("BTCUSDT") == record


def test_schedule_pipeline_runs_and_updates_active_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=positive_negative_predict),
    )

    async def _run() -> tuple[list[str], ai_manager_module.AIManager]:
        manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
        df = _make_df(50)
        results: list[str] = []

        async def on_result(selection):
            results.append(selection.best_model)

        schedule = manager.schedule_pipeline(
            "BTCUSDT",
            lambda: df,
            ["alpha", "beta"],
            interval_seconds=0.05,
            seq_len=3,
            folds=2,
            baseline_provider=lambda: df,
            on_result=on_result,
        )

        await asyncio.sleep(0.12)
        manager.cancel_pipeline_schedule("BTCUSDT")
        await asyncio.sleep(0)  # allow cancellation to propagate

        assert schedule.symbol == "btcusdt"
        return results, manager

    results, manager = asyncio.run(_run())

    assert results, "Pipeline schedule should produce at least one selection"
    assert manager.get_active_model("BTCUSDT") == "alpha"
    history = manager.get_pipeline_history("BTCUSDT")
    assert history, "Historia pipeline'u powinna zawierać wpisy"
    assert all(entry.best_model == "alpha" for entry in history)


def test_pipeline_history_captures_improving_model(tmp_path, monkeypatch):
    class AdaptiveModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.prediction = 1.0 if model_type == "alpha" else -1.0
            self.train_calls = 0

        def train(self, X, y, **_):
            self.train_calls += 1
            if self.model_type == "beta" and self.train_calls >= 2:
                self.prediction = 1.0

        def predict(self, X):
            return np.full((len(X),), self.prediction, dtype=float)

        def predict_series(self, df, feature_cols):
            return pd.Series(np.full(len(df), self.prediction, dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", AdaptiveModel)

    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _make_df(60)

    selection_first = asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=4,
            folds=2,
        )
    )

    selection_second = asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=4,
            folds=2,
        )
    )

    history = manager.get_pipeline_history("BTCUSDT")
    assert len(history) == 2
    assert selection_first.best_model == "alpha"
    assert selection_second.best_model in {"alpha", "beta"}
    first_eval = {ev.model_type: ev.hit_rate for ev in history[0].evaluations}
    second_eval = {ev.model_type: ev.hit_rate for ev in history[1].evaluations}
    assert first_eval["alpha"] >= first_eval.get("beta", 0.0)
    assert second_eval["beta"] >= first_eval.get("beta", 0.0)


def test_ensemble_registry_snapshot_and_diff_roundtrip(tmp_path):
    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)

    base = manager.register_ensemble("combo", ["alpha", "beta"])
    weighted = manager.register_ensemble(
        "weighted",
        ["alpha", "gamma"],
        aggregation="weighted",
        weights=[0.7, 0.3],
    )

    snapshot = manager.snapshot_ensembles()
    assert snapshot.total_ensembles() == 2
    assert snapshot.get("combo") == base
    assert snapshot.get("weighted") == weighted

    as_dict = ai_manager_module.ensemble_registry_snapshot_to_dict(snapshot)
    rebuilt_snapshot = ai_manager_module.ensemble_registry_snapshot_from_dict(as_dict)
    assert rebuilt_snapshot.ensembles == snapshot.ensembles

    json_payload = ai_manager_module.ensemble_registry_snapshot_to_json(snapshot, indent=2)
    json_snapshot = ai_manager_module.ensemble_registry_snapshot_from_json(json_payload)
    assert json_snapshot.ensembles == snapshot.ensembles

    file_path = tmp_path / "ensembles.json"
    ai_manager_module.ensemble_registry_snapshot_to_file(snapshot, file_path)
    file_snapshot = ai_manager_module.ensemble_registry_snapshot_from_file(file_path)
    assert file_snapshot.ensembles == snapshot.ensembles

    manager.unregister_ensemble("combo")
    manager.register_ensemble(
        "weighted",
        ["alpha", "beta"],
        aggregation="weighted",
        weights=[0.6, 0.4],
        override=True,
    )
    after_snapshot = manager.snapshot_ensembles()

    diff = ai_manager_module.diff_ensemble_snapshots(snapshot, after_snapshot)
    assert "combo" in diff.removed_names()
    assert "weighted" in diff.changed_names()
    assert diff.changed["weighted"][0].weights == weighted.weights

    diff_dict = ai_manager_module.ensemble_registry_diff_to_dict(diff)
    rebuilt_diff = ai_manager_module.ensemble_registry_diff_from_dict(diff_dict)
    assert rebuilt_diff.changed["weighted"][1].weights == (0.6, 0.4)

    diff_json = ai_manager_module.ensemble_registry_diff_to_json(diff, indent=2)
    diff_from_json = ai_manager_module.ensemble_registry_diff_from_json(diff_json)
    assert diff_from_json.removed["combo"].components == base.components

    diff_path = tmp_path / "diff.json"
    ai_manager_module.ensemble_registry_diff_to_file(diff, diff_path)
    diff_from_file = ai_manager_module.ensemble_registry_diff_from_file(diff_path)
    assert diff_from_file.added == rebuilt_diff.added

    manager.restore_ensembles(file_snapshot)
    restored = manager.list_ensembles()
    assert set(restored) == {"combo", "weighted"}
    assert restored["weighted"].weights == weighted.weights


def test_pipeline_history_diff_and_serialization(tmp_path):
    evaluation_alpha = ai_manager_module.ModelEvaluation(
        model_type="alpha",
        hit_rate=0.7,
        pnl=12.0,
        sharpe=1.2,
        cv_scores=[0.6, 0.7],
        model_path="alpha.bin",
    )
    evaluation_beta = ai_manager_module.ModelEvaluation(
        model_type="beta",
        hit_rate=0.4,
        pnl=5.0,
        sharpe=0.8,
        cv_scores=[0.4, 0.45],
        model_path="beta.bin",
    )

    record_old = ai_manager_module.PipelineExecutionRecord(
        symbol="btcusdt",
        decided_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        best_model="alpha",
        evaluations=(evaluation_alpha,),
        prediction_count=20,
        prediction_mean=0.2,
    )

    record_new = ai_manager_module.PipelineExecutionRecord(
        symbol="btcusdt",
        decided_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        best_model="beta",
        evaluations=(evaluation_beta,),
        prediction_count=25,
        prediction_mean=-0.1,
    )

    record_added = ai_manager_module.PipelineExecutionRecord(
        symbol="ethusdt",
        decided_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
        best_model="alpha",
        evaluations=(evaluation_alpha,),
        prediction_count=15,
    )

    snapshot_before = ai_manager_module.PipelineHistorySnapshot({"btcusdt": (record_old,)})
    snapshot_after = ai_manager_module.PipelineHistorySnapshot(
        {
            "btcusdt": (record_old, record_new),
            "ethusdt": (record_added,),
        }
    )

    diff = ai_manager_module.diff_pipeline_history_snapshots(snapshot_before, snapshot_after)

    assert not diff.is_empty()
    assert diff.added_symbols() == ("ethusdt",)
    assert diff.removed_symbols() == ()
    assert diff.changed_symbols() == ("btcusdt",)
    assert diff.total_added_records() == 1
    assert diff.total_changed_records() == 2
    assert diff.total_removed_records() == 0

    dict_payload = ai_manager_module.pipeline_history_diff_to_dict(diff)
    restored_from_dict = ai_manager_module.pipeline_history_diff_from_dict(dict_payload)
    assert restored_from_dict == diff

    json_payload = ai_manager_module.pipeline_history_diff_to_json(diff, indent=2)
    restored_from_json = ai_manager_module.pipeline_history_diff_from_json(json_payload)
    assert restored_from_json == diff

    target_path = tmp_path / "diff.json"
    ai_manager_module.pipeline_history_diff_to_file(diff, target_path)
    restored_from_file = ai_manager_module.pipeline_history_diff_from_file(target_path)
    assert restored_from_file == diff


def test_pipeline_history_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=1.0),
    )

    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _make_df(60)

    for _ in range(3):
        asyncio.run(
            manager.run_pipeline(
                "BTCUSDT",
                df,
                ["alpha"],
                seq_len=3,
                folds=2,
            )
        )

    full_snapshot = manager.snapshot_pipeline_history()
    assert full_snapshot.total_records() == 3
    assert full_snapshot.symbols() == ("btcusdt",)

    manager.set_pipeline_history_limit(2)
    assert manager.get_pipeline_history_limit() == 2
    truncated_history = manager.get_pipeline_history("BTCUSDT")
    assert len(truncated_history) == 2

    snapshot = manager.snapshot_pipeline_history()
    assert snapshot.total_records() == 2

    as_dict = ai_manager_module.pipeline_history_snapshot_to_dict(snapshot)
    rebuilt_snapshot = ai_manager_module.pipeline_history_snapshot_from_dict(as_dict)
    assert rebuilt_snapshot.records == snapshot.records

    json_payload = ai_manager_module.pipeline_history_snapshot_to_json(snapshot, indent=None)
    assert json.loads(json_payload)["btcusdt"]
    from_json = ai_manager_module.pipeline_history_snapshot_from_json(json_payload)
    assert from_json.records == snapshot.records

    path = tmp_path / "history.json"
    ai_manager_module.pipeline_history_snapshot_to_file(snapshot, path)
    assert path.exists()
    from_file = ai_manager_module.pipeline_history_snapshot_from_file(path)
    assert from_file.records == snapshot.records

    manager.clear_pipeline_history("BTCUSDT")
    assert manager.get_pipeline_history("BTCUSDT") == []

    manager.restore_pipeline_history(snapshot)
    assert len(manager.get_pipeline_history("BTCUSDT")) == snapshot.total_records()

    manager.clear_pipeline_history("BTCUSDT")
    manager.set_pipeline_history_limit(1)
    manager.restore_pipeline_history(full_snapshot, replace=True)
    restored = manager.get_pipeline_history("BTCUSDT")
    assert len(restored) == 1

    with pytest.raises(TypeError):
        ai_manager_module.pipeline_history_snapshot_from_dict(["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        ai_manager_module.pipeline_history_snapshot_from_json(123)  # type: ignore[arg-type]
    assert manager.active_pipeline_schedules() == {}
    history = manager.get_pipeline_history("BTCUSDT")
    assert history


def test_pipeline_history_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=positive_negative_predict),
    )

    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _make_df(30)

    asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            folds=2,
        )
    )
    asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            folds=2,
        )
    )

    history = manager.get_pipeline_history("BTCUSDT")
    assert len(history) == 2
    assert history[-1].best_model == "alpha"
    limited = manager.get_pipeline_history("BTCUSDT", limit=1)
    assert len(limited) == 1
    assert limited[0] == history[-1]

    snapshot = manager.pipeline_history()
    assert snapshot == {"btcusdt": history}

    last = manager.last_pipeline_selection("BTCUSDT")
    assert last is not None
    assert last.prediction_count == len(df)

    manager.clear_pipeline_history("BTCUSDT")
    assert manager.get_pipeline_history("BTCUSDT") == []
    assert manager.last_pipeline_selection("BTCUSDT") is None


def test_pipeline_history_formatting_and_logging(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=positive_negative_predict),
    )

    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _make_df(36)

    selection = asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            folds=2,
        )
    )

    record = manager.last_pipeline_selection("BTCUSDT")
    assert record is not None
    assert selection.best_model == record.best_model

    record_text = ai_manager_module.format_pipeline_execution_record(record, include_evaluations=True)
    assert f"Najlepszy model: {record.best_model}" in record_text
    assert "Ewaluacje modeli" in record_text

    snapshot = manager.snapshot_pipeline_history()
    snapshot_text = ai_manager_module.format_pipeline_history_snapshot(snapshot)
    assert "Liczba symboli: 1" in snapshot_text

    future_time = record.decided_at + timedelta(minutes=5)
    updated_record = replace(record, decided_at=future_time, best_model="beta")
    next_snapshot = ai_manager_module.PipelineHistorySnapshot({"btcusdt": (record, updated_record)})
    diff = ai_manager_module.diff_pipeline_history_snapshots(snapshot, next_snapshot)
    diff_text = ai_manager_module.format_pipeline_history_diff(diff, include_evaluations=True)
    assert "Zmienione symbole" in diff_text
    assert "beta" in diff_text

    history_logger_name = f"{ai_manager_module.logger.name}.history"
    history_logger = logging.getLogger(history_logger_name)
    previous_handlers = list(ai_manager_module.logger.handlers)
    for handler in previous_handlers:
        ai_manager_module.logger.removeHandler(handler)
    caplog.clear()
    captured_messages: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - prosty kolektor
            captured_messages.append(record.getMessage())

    capture_handler = _CaptureHandler()
    history_logger.addHandler(capture_handler)
    try:
        with caplog.at_level(logging.INFO, logger=history_logger_name):
            ai_manager_module.log_pipeline_execution_record(record, include_evaluations=True)
            ai_manager_module.log_pipeline_history_snapshot(snapshot)
            ai_manager_module.log_pipeline_history_diff(diff, include_evaluations=True)
    finally:
        history_logger.removeHandler(capture_handler)
        for handler in previous_handlers:
            ai_manager_module.logger.addHandler(handler)

    logged_messages = captured_messages or [entry.message for entry in caplog.records]
    logged = "\n".join(logged_messages)
    assert record.symbol in logged
    assert "Liczba symboli" in logged
    assert "Zmiany w historii pipeline'u" in logged


def test_predict_series_with_ensemble(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=positive_negative_predict),
    )

    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)
    df = _make_df(32)

    asyncio.run(
        manager.train_all_models(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            epochs=1,
            batch_size=1,
        )
    )

    definition = manager.register_ensemble(
        "hybrid",
        ["alpha", "beta"],
        aggregation="weighted",
        weights=[0.75, 0.25],
    )
    assert definition.components == ("alpha", "beta")
    assert definition.aggregation == "weighted"

    predictions = asyncio.run(
        manager.predict_series(
            "BTCUSDT",
            df,
            model_types=["hybrid"],
        )
    )

    assert isinstance(predictions, pd.Series)
    assert np.allclose(predictions.values, 0.5)
    assert predictions.index.equals(df.index)

    ensembles = manager.list_ensembles()
    assert "hybrid" in ensembles

    manager.unregister_ensemble("hybrid")
    assert "hybrid" not in manager.list_ensembles()

    with pytest.raises(KeyError):
        manager.unregister_ensemble("hybrid")


def test_format_and_log_ensemble_registry(caplog):
    definition = ai_manager_module.EnsembleDefinition(
        name="hybrid",
        components=("alpha", "beta"),
        aggregation="weighted",
        weights=(0.7, 0.3),
    )

    definition_text = ai_manager_module.format_ensemble_definition(definition)
    assert "Zespół: hybrid" in definition_text
    assert "waga" in definition_text

    before_snapshot = ai_manager_module.EnsembleRegistrySnapshot(
        {
            "hybrid": definition,
            "gamma": ai_manager_module.EnsembleDefinition(
                name="gamma",
                components=("gamma_a",),
                aggregation="mean",
            ),
        }
    )

    snapshot_text = ai_manager_module.format_ensemble_registry_snapshot(before_snapshot)
    assert "Liczba zespołów: 2" in snapshot_text
    assert "hybrid" in snapshot_text

    after_snapshot = ai_manager_module.EnsembleRegistrySnapshot(
        {
            "hybrid": ai_manager_module.EnsembleDefinition(
                name="hybrid",
                components=("alpha", "beta"),
                aggregation="mean",
            ),
            "delta": ai_manager_module.EnsembleDefinition(
                name="delta",
                components=("delta_a", "delta_b"),
                aggregation="max",
            ),
        }
    )

    diff = ai_manager_module.diff_ensemble_snapshots(before_snapshot, after_snapshot)
    diff_text = ai_manager_module.format_ensemble_registry_diff(diff)
    assert "Zmiany w rejestrze zespołów" in diff_text
    assert "Dodane zespoły" in diff_text
    assert "Usunięte zespoły" in diff_text
    assert "Zmienione zespoły" in diff_text

    history_logger_name = f"{ai_manager_module.logger.name}.history"
    history_logger = logging.getLogger(history_logger_name)
    previous_handlers = list(ai_manager_module.logger.handlers)
    for handler in previous_handlers:
        ai_manager_module.logger.removeHandler(handler)
    caplog.clear()
    captured_messages = []

    class _EnsembleCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - prosty kolektor
            captured_messages.append(record.getMessage())

    capture_handler = _EnsembleCapture()
    history_logger.addHandler(capture_handler)
    try:
        with caplog.at_level(logging.INFO, logger=history_logger_name):
            ai_manager_module.log_ensemble_definition(definition)
            ai_manager_module.log_ensemble_registry_snapshot(before_snapshot)
            ai_manager_module.log_ensemble_registry_diff(diff)
    finally:
        history_logger.removeHandler(capture_handler)
        for handler in previous_handlers:
            ai_manager_module.logger.addHandler(handler)

    logged_messages = captured_messages or [entry.message for entry in caplog.records]
    logged = "\n".join(logged_messages)
    assert "Zespół: hybrid" in logged
    assert "Liczba zespołów" in logged
    assert "Zmiany w rejestrze zespołów" in logged


@pytest.mark.parametrize(
    "exception_cls",
    [TypeError, ValueError, AttributeError, ZeroDivisionError, OverflowError],
    ids=[
        "type_error",
        "value_error",
        "attribute_error",
        "zero_division_error",
        "overflow_error",
    ],
)
def test_detect_drift_falls_back_to_python_math(tmp_path, monkeypatch, exception_cls):
    manager = ai_manager_module.AIManager(ai_threshold_bps=5.0, model_dir=tmp_path)

    base = _make_df(16)
    recent = base.copy()
    recent.loc[recent.index[-4]:, "close"] *= 6
    recent.loc[recent.index[-6]:, "volume"] *= 3

    base_obj = base.astype(str)
    recent_obj = recent.astype(str)

    def _boom(self, *args, **kwargs):
        raise exception_cls("forced pct_change failure")

    monkeypatch.setattr(pd.DataFrame, "pct_change", _boom, raising=False)

    report = manager.detect_drift(
        base_obj,
        recent_obj,
        feature_cols=["close", "volume"],
        threshold=0.1,
    )

    assert report.triggered is True
    assert report.volatility_shift > 0.0
    assert report.feature_drift > 0.0


def test_run_pipeline_persists_drift_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=positive_negative_predict),
    )

    journal = InMemoryTradingDecisionJournal()
    manager = ai_manager_module.AIManager(
        ai_threshold_bps=5.0,
        model_dir=tmp_path / "models",
        audit_root=tmp_path / "ai_audit",
        decision_journal=journal,
        decision_journal_context={
            "environment": "paper-trading",
            "portfolio": "ai-pipeline",
            "risk_profile": "ai-monitoring",
        },
    )
    df = _make_df(32)

    selection = asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha", "beta"],
            seq_len=3,
            folds=2,
            baseline=df,
        )
    )

    assert selection.drift_report is not None
    drift_path = manager.get_last_drift_report_path()
    assert drift_path is not None and drift_path.exists()
    payload = json.loads(drift_path.read_text(encoding="utf-8"))
    assert payload["job_name"] == "pipeline:btcusdt"
    assert payload["metrics"]["feature_drift"]["score"] >= 0.0
    assert "psi" in payload["metrics"]["feature_drift"]
    assert "distribution_summary" in payload["metrics"]
    assert "triggered_features" in payload["metrics"]["distribution_summary"]
    assert "features" in payload["metrics"]
    assert payload["metrics"]["triggered"]["value"] in {True, False}
    assert payload["baseline_window"]["rows"] == len(df)
    assert payload["production_window"]["rows"] == len(df)
    drift_payload = manager.load_last_drift_report()
    assert drift_payload is not None
    assert drift_payload["job_name"] == "pipeline:btcusdt"
    listed = manager.list_drift_report_paths(limit=1)
    assert listed and listed[0] == drift_path
    events = tuple(journal.export())
    drift_events = [event for event in events if event["event"] == "ai_drift_report"]
    assert drift_events, "Decision journal should contain drift report entry"
    event = drift_events[-1]
    assert event["environment"] == "paper-trading"
    assert event["portfolio"] == "ai-pipeline"
    assert event["report_type"] == "drift"
    assert event["report_path"].endswith(".json")
    assert event["threshold"]
    assert event["symbol"].lower() == "btcusdt"
    assert event["triggered"] in {"true", "false"}


def test_run_pipeline_records_data_quality_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ai_manager_module,
        "_AIModels",
        make_stub_model(predict_fn=positive_negative_predict),
    )

    audit_root = tmp_path / "audit"
    journal = InMemoryTradingDecisionJournal()
    manager = ai_manager_module.AIManager(
        ai_threshold_bps=5.0,
        model_dir=tmp_path / "models",
        audit_root=audit_root,
        decision_journal=journal,
        decision_journal_context={
            "environment": "paper-trading",
            "portfolio": "ai-pipeline",
            "risk_profile": "ai-monitoring",
            "strategy": "data-quality",
        },
    )

    watcher = DataCompletenessWatcher("1min", warning_gap_ratio=0.0)
    check = DataQualityCheck(
        name="completeness",
        callback=lambda frame: watcher.assess(frame),
        tags=("completeness", "monitoring"),
        source="monitoring:test",  # źródło nadpisuje domyślne
    )
    manager.register_data_quality_check(check)

    df = _make_df(24)
    df = df.drop(df.index[5])

    selection = asyncio.run(
        manager.run_pipeline(
            "BTCUSDT",
            df,
            ["alpha"],
            seq_len=3,
            folds=2,
        )
    )

    assert selection.predictions is not None
    path = manager.get_last_data_quality_report_path()
    assert path is not None and path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_name"] == "pipeline:btcusdt:completeness"
    assert payload["source"] == "monitoring:test"
    assert payload["tags"] == ["completeness", "monitoring"]
    assert payload["issues"], "Raport jakości danych powinien zawierać wykryte problemy"
    assert payload["summary"]["status"] in {"warning", "critical"}
    events = tuple(journal.export())
    dq_events = [event for event in events if event["event"] == "ai_data_quality_report"]
    assert dq_events, "Decision journal should contain data quality audit entry"
    journal_event = dq_events[-1]
    assert journal_event["environment"] == "paper-trading"
    assert journal_event["report_type"] == "data_quality"
    assert journal_event["report_path"].endswith(".json")
    assert journal_event["status"] in {"warning", "critical"}
    assert journal_event["schedule"] == "pipeline:btcusdt:completeness"
    assert journal_event.get("tags") == "completeness,monitoring"


def test_record_data_quality_issues_creates_audit_entry(tmp_path):
    journal = InMemoryTradingDecisionJournal()
    manager = ai_manager_module.AIManager(
        ai_threshold_bps=5.0,
        model_dir=tmp_path / "models",
        audit_root=tmp_path / "ai_audit",
        decision_journal=journal,
        decision_journal_context={
            "environment": "paper-trading",
            "portfolio": "ai-pipeline",
            "risk_profile": "ai-monitoring",
        },
    )
    vector = FeatureVector(
        timestamp=1_700_000_000,
        symbol="BTCUSDT",
        features={"momentum": 1.0, "volume_ratio": 1.2},
        target_bps=0.0,
    )
    dataset = FeatureDataset(vectors=(vector,), metadata={"symbols": ["BTCUSDT"]})

    path = manager.record_data_quality_issues(
        [
            {"code": "missing_rows", "message": "braki w danych", "row": None},
            {"code": "timestamp_gap", "message": "luka czasowa", "row": 42},
        ],
        dataset=dataset,
        job_name="btc-data-quality",
        source="backtest-monitor",
        summary={"total_gaps": 2},
        tags=("ohlcv", "monitoring"),
    )

    assert path is not None and path.exists()
    assert path == manager.get_last_data_quality_report_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_name"] == "btc-data-quality"
    assert payload["source"] == "backtest-monitor"
    assert payload["dataset"]["rows"] == len(dataset.vectors)
    assert payload["summary"]["total_gaps"] == 2
    assert payload["issues"][0]["code"] == "missing_rows"
    dq_payload = manager.load_last_data_quality_report()
    assert dq_payload is not None
    assert dq_payload["issues"][1]["code"] == "timestamp_gap"
    manager._last_data_quality_report_path = None  # simulate restart
    fallback_payload = manager.load_last_data_quality_report()
    assert fallback_payload is not None
    assert fallback_payload["job_name"] == "btc-data-quality"
    listed = manager.list_data_quality_report_paths(limit=1)
    assert listed and listed[0] == path
    events = tuple(journal.export())
    assert events and events[-1]["event"] == "ai_data_quality_report"
    journal_entry = events[-1]
    assert journal_entry["symbol"].lower() == "btcusdt"
    assert journal_entry["report_type"] == "data_quality"
    assert journal_entry["issues_count"] == "2"
    assert journal_entry["schedule"] == "btc-data-quality"
    assert journal_entry.get("tags") == "ohlcv,monitoring"
