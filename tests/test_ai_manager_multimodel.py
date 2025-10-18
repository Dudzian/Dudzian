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

from KryptoLowca import ai_manager as ai_manager_module


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

    class StubModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None
            with lock:
                calls.append(("init", model_type, self.model_dir))

        def train(self, X, y, **_):
            with lock:
                calls.append(("train", self.model_type, None))
            # symulujemy szybki trening
            return None

        def predict(self, X):
            with lock:
                calls.append(("predict", self.model_type, None))
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return np.full((len(X),), sign, dtype=float)

        def predict_series(self, df, feature_cols):
            with lock:
                calls.append(("predict_series", self.model_type, None))
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return pd.Series(np.full(len(df), sign, dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", StubModel)

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
    lock = threading.Lock()

    class StubModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None

        def train(self, X, y, **_):
            return None

        def predict(self, X):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return np.full((len(X),), sign, dtype=float)

        def predict_series(self, df, feature_cols):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return pd.Series(np.full(len(df), sign, dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", StubModel)

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
    class StubModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None

        def train(self, X, y, **_):
            return None

        def predict(self, X):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return np.full((len(X),), sign, dtype=float)

        def predict_series(self, df, feature_cols):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return pd.Series(np.full(len(df), sign, dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", StubModel)

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
    class StubModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None

        def train(self, X, y, **_):
            return None

        def predict(self, X):
            return np.ones(len(X), dtype=float)

        def predict_series(self, df, feature_cols):
            return pd.Series(np.ones(len(df), dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", StubModel)

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
    class StubModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None

        def train(self, X, y, **_):
            return None

        def predict(self, X):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return np.full((len(X),), sign, dtype=float)

        def predict_series(self, df, feature_cols):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return pd.Series(np.full(len(df), sign, dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", StubModel)

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
    class StubModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None

        def train(self, X, y, **_):
            return None

        def predict(self, X):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return np.full((len(X),), sign, dtype=float)

        def predict_series(self, df, feature_cols):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return pd.Series(np.full(len(df), sign, dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", StubModel)

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
    with caplog.at_level(logging.INFO, logger=history_logger_name):
        ai_manager_module.log_pipeline_execution_record(record, include_evaluations=True)
        ai_manager_module.log_pipeline_history_snapshot(snapshot)
        ai_manager_module.log_pipeline_history_diff(diff, include_evaluations=True)

    logged = "\n".join(entry.message for entry in caplog.records)
    assert record.symbol in logged
    assert "Liczba symboli" in logged
    assert "Zmiany w historii pipeline'u" in logged


def test_predict_series_with_ensemble(tmp_path, monkeypatch):
    class StubModel:
        def __init__(self, input_size: int, seq_len: int, model_type: str, *, model_dir: Path | None = None):
            self.input_size = input_size
            self.seq_len = seq_len
            self.model_type = model_type
            self.model_dir = Path(model_dir) if model_dir is not None else None

        def train(self, X, y, **_):
            return None

        def predict(self, X):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return np.full((len(X),), sign, dtype=float)

        def predict_series(self, df, feature_cols):
            sign = 1.0 if self.model_type == "alpha" else -1.0
            return pd.Series(np.full(len(df), sign, dtype=float), index=df.index)

    monkeypatch.setattr(ai_manager_module, "_AIModels", StubModel)

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
    with caplog.at_level(logging.INFO, logger=history_logger_name):
        ai_manager_module.log_ensemble_definition(definition)
        ai_manager_module.log_ensemble_registry_snapshot(before_snapshot)
        ai_manager_module.log_ensemble_registry_diff(diff)

    logged = "\n".join(entry.message for entry in caplog.records)
    assert "Zespół: hybrid" in logged
    assert "Liczba zespołów" in logged
    assert "Zmiany w rejestrze zespołów" in logged
