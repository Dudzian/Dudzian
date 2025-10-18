from datetime import datetime, timezone

import pytest

from bot_core.ai.models import ModelArtifact


@pytest.fixture
def base_payload() -> dict[str, object]:
    return {
        "feature_names": ["momentum", "volume"],
        "model_state": {"weights": [0.1, -0.2]},
        "metrics": {"mae": 1.5, "directional_accuracy": 0.6},
        "metadata": {
            "feature_scalers": {
                "momentum": {"mean": 0.0, "stdev": 1.0},
                "volume": {"mean": 0.0, "stdev": 2.0},
            }
        },
        "backend": "linear",
    }


def test_from_dict_missing_trained_at_uses_utc_now(base_payload):
    artifact = ModelArtifact.from_dict(dict(base_payload))

    assert isinstance(artifact.trained_at, datetime)
    assert artifact.trained_at.tzinfo == timezone.utc
    assert artifact.trained_at <= datetime.now(timezone.utc)


def test_from_dict_invalid_timestamp_falls_back_to_now(base_payload):
    payload = dict(base_payload)
    payload["trained_at"] = "not-a-valid-date"

    artifact = ModelArtifact.from_dict(payload)

    assert artifact.trained_at.tzinfo == timezone.utc
    assert artifact.trained_at <= datetime.now(timezone.utc)


def test_from_dict_ignores_non_mapping_sections(base_payload):
    payload = dict(base_payload)
    payload.update(
        {
            "model_state": ["unexpected"],
            "metrics": None,
            "metadata": "raw",
        }
    )

    artifact = ModelArtifact.from_dict(payload)

    assert artifact.model_state == {}
    assert artifact.metrics == {}
    assert artifact.metadata == {}
