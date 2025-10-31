from pathlib import Path


from bot_core.ai.feature_engineering import FeatureDataset, FeatureVector


def _vector(ts: float, symbol: str, target: float) -> FeatureVector:
    return FeatureVector(timestamp=ts, symbol=symbol, features={"x": target}, target_bps=target)


def test_feature_dataset_orders_vectors_and_metadata() -> None:
    dataset = FeatureDataset(
        vectors=(
            _vector(1_700_000_060, "ETHUSDT", 4.0),
            _vector(1_700_000_000, "BTCUSDT", -2.0),
            _vector(1_700_000_030, "BTCUSDT", 1.5),
        ),
        metadata={"source": "unit"},
    )

    assert [vector.timestamp for vector in dataset.vectors] == [1_700_000_000, 1_700_000_030, 1_700_000_060]
    assert dataset.metadata["row_count"] == 3
    assert dataset.metadata["start_timestamp"] == 1_700_000_000
    assert dataset.metadata["end_timestamp"] == 1_700_000_060
    assert dataset.feature_names == ("x",)
    stats = dataset.feature_stats["x"]
    assert stats["min"] == -2.0
    assert stats["max"] == 4.0


def test_feature_dataset_subset_updates_metadata_and_scale() -> None:
    dataset = FeatureDataset(
        vectors=(
            _vector(100.0, "BTC", -5.0),
            _vector(120.0, "BTC", 0.0),
            _vector(140.0, "BTC", 5.0),
        ),
        metadata={"source": "unit"},
    )

    subset = dataset.subset([2, 1])

    assert [vector.timestamp for vector in subset.vectors] == [120.0, 140.0]
    assert subset.metadata["row_count"] == 2
    assert subset.metadata["start_timestamp"] == 120.0
    assert subset.metadata["end_timestamp"] == 140.0
    assert "target_scale" in subset.metadata
    assert subset.target_scale > 0.0
    assert subset.feature_stats["x"]["mean"] == 2.5


def test_feature_dataset_learning_arrays() -> None:
    dataset = FeatureDataset(
        vectors=(
            FeatureVector(
                timestamp=1.0,
                symbol="BTCUSDT",
                features={"a": 1.0, "b": 0.5},
                target_bps=2.0,
            ),
            FeatureVector(
                timestamp=2.0,
                symbol="BTCUSDT",
                features={"a": 2.0, "b": 1.0},
                target_bps=-1.0,
            ),
        ),
        metadata={},
    )

    matrix, targets, feature_names = dataset.to_learning_arrays()

    assert feature_names == ["a", "b"]
    assert matrix == [[1.0, 0.5], [2.0, 1.0]]
    assert targets == [2.0, -1.0]
