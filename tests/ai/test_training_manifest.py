import json
from pathlib import Path

import pandas as pd

from bot_core.ai.pipeline import load_training_manifest, run_training_profile


def _build_dataset(rows: int = 24) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        data.append(
            {
                "f1": float(idx % 5) / 10.0,
                "f2": float(idx) / 5.0,
                "target": float((idx % 3) - 1),
            }
        )
    return pd.DataFrame(data)


def test_run_training_profile_trains_multiple_models(tmp_path: Path) -> None:
    frame = _build_dataset()
    dataset_path = tmp_path / "dataset.csv"
    frame.to_csv(dataset_path, index=False)
    manifest_payload = {
        "profiles": {
            "demo": {
                "datasets": {
                    "ohlcv": {"path": str(dataset_path)},
                },
                "models": [
                    {
                        "name": "alpha",
                        "dataset": "ohlcv",
                        "target": "target",
                        "features": ["f1", "f2"],
                    },
                    {
                        "name": "beta",
                        "dataset": "ohlcv",
                        "target": "target",
                        "features": ["f1"],
                    },
                ],
                "ensembles": [
                    {
                        "name": "combo",
                        "components": ["alpha", "beta"],
                        "aggregation": "mean",
                    }
                ],
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    manifest = load_training_manifest(manifest_path)
    summary = run_training_profile(manifest, "demo", output_dir=tmp_path / "artifacts")
    assert len(summary.models) == 2
    for result in summary.models:
        assert result.artifact_path.exists()
        assert "summary" in result.metrics
        assert result.metadata
    assert summary.profile.ensembles and summary.profile.ensembles[0].name == "combo"


def test_run_training_profile_with_overrides(tmp_path: Path) -> None:
    manifest = load_training_manifest(
        {
            "profiles": {
                "override": {
                    "datasets": {"memory": {}},
                    "models": [
                        {
                            "name": "alpha",
                            "dataset": "memory",
                            "target": "target",
                            "features": ["f1", "f2"],
                        }
                    ],
                }
            }
        }
    )
    frame = _build_dataset(rows=16)
    summary = run_training_profile(
        manifest,
        "override",
        output_dir=tmp_path / "artifacts",
        dataset_overrides={"memory": frame},
    )
    assert summary.models[0].artifact_path.exists()
    assert summary.models[0].metrics["summary"]["mae"] >= 0.0
