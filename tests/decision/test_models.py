import pytest

from bot_core.ai.training import SimpleGradientBoostingModel


def test_gradient_boosting_uses_scalers_for_missing_features() -> None:
    model = SimpleGradientBoostingModel(
        learning_rate=0.3,
        n_estimators=5,
        min_samples_leaf=1,
        max_bins=4,
    )
    matrix = [[10.0], [12.0], [14.0], [16.0], [18.0]]
    targets = [-1.0, -0.2, 0.0, 0.2, 1.0]

    model.fit_matrix(matrix, ["x"], targets)

    assert "x" in model.feature_scalers
    mean, stdev = model.feature_scalers["x"]
    assert mean == pytest.approx(14.0)
    assert stdev > 0.0

    baseline = model.predict({"x": mean})
    missing = model.predict({})
    assert missing == pytest.approx(baseline)

    state = model.to_state()
    clone = SimpleGradientBoostingModel()
    clone.load_state(state)

    clone_mean, clone_stdev = clone.feature_scalers["x"]
    assert clone_mean == pytest.approx(mean)
    assert clone_stdev == pytest.approx(stdev)
    assert clone.predict({"x": mean}) == pytest.approx(baseline)
