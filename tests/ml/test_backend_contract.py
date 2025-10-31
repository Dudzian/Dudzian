from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest

from core import ml
from core.ml.backends import ReferenceRegressor


@pytest.fixture(autouse=True)
def _reset_registry():
    # Każdy test powinien zaczynać z czystym cache konfiguracji backendów.
    from bot_core.ai import backends as optional_backends

    optional_backends.clear_backend_caches()
    yield
    optional_backends.clear_backend_caches()


def _sample_dataset() -> tuple[list[Mapping[str, float]], list[float]]:
    samples = [
        {"volume": 1.0, "momentum": 0.2},
        {"volume": 2.0, "momentum": 0.1},
        {"volume": 3.0, "momentum": -0.1},
        {"volume": 4.0, "momentum": -0.2},
    ]
    targets = [0.5, 0.9, 1.3, 1.7]
    return samples, targets


def test_reference_backend_fulfills_basic_contract(tmp_path: Path) -> None:
    name, model = ml.build_backend(preferred=("reference",))
    assert name == "reference"
    assert isinstance(model, ReferenceRegressor)

    samples, targets = _sample_dataset()
    model.fit(samples, targets)

    single_prediction = model.predict(samples[0])
    batch_predictions = model.batch_predict(samples)

    assert isinstance(single_prediction, float)
    assert len(batch_predictions) == len(samples)

    output_path = tmp_path / "reference_model.json"
    model.save(output_path)
    assert output_path.exists()

    clone = ReferenceRegressor()
    clone.load(output_path)
    assert pytest.approx(model.predict(samples[-1]), rel=1e-6) == clone.predict(samples[-1])


def test_reference_backend_handles_degenerate_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    name, model = ml.build_backend(preferred=("reference",))
    assert name == "reference"

    samples = [{"feature": 1.0}, {"feature": 1.0}, {"feature": 1.0}]
    targets = [2.0, 2.0, 2.0]

    model.fit(samples, targets)
    assert model.coefficients == (0.0,)
    assert pytest.approx(model.predict({"feature": 42.0})) == 2.0


def test_reference_backend_raises_on_untrained_usage() -> None:
    _, model = ml.build_backend(preferred=("reference",))
    with pytest.raises(RuntimeError):
        model.predict({"volume": 1.0})


def test_factory_skips_unregistered_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    _, model = ml.build_backend(preferred=("non-existent", "reference"))
    assert isinstance(model, ReferenceRegressor)
