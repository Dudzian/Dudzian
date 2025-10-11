import json
from pathlib import Path

import pytest

from bot_core.runtime.telemetry_risk_profiles import (
    get_risk_profile,
    load_risk_profiles_from_file,
    load_risk_profiles_with_metadata,
    risk_profile_metadata,
)


def test_load_risk_profiles_from_directory(tmp_path: Path) -> None:
    directory = tmp_path / "profiles"
    directory.mkdir()

    (directory / "alpha.json").write_text(
        json.dumps({"risk_profiles": {"alpha_dir": {"severity_min": "warning"}}}),
        encoding="utf-8",
    )
    (directory / "beta.yaml").write_text(
        "risk_profiles:\n  beta_dir:\n    severity_min: info\n",
        encoding="utf-8",
    )

    registered = load_risk_profiles_from_file(directory)
    assert set(registered) >= {"alpha_dir", "beta_dir"}

    alpha_meta = risk_profile_metadata("alpha_dir")
    assert alpha_meta["origin"].startswith("dir:")
    assert alpha_meta["origin"].endswith("#alpha.json")

    beta_profile = get_risk_profile("beta_dir")
    assert beta_profile["severity_min"] == "info"

    registered_with_meta, metadata = load_risk_profiles_with_metadata(directory)
    assert set(registered_with_meta) >= {"alpha_dir", "beta_dir"}
    assert metadata["type"] == "directory"
    assert metadata["path"] == str(directory)
    assert {entry["path"] for entry in metadata["files"]} == {
        str(directory / "alpha.json"),
        str(directory / "beta.yaml"),
    }


def test_load_risk_profiles_directory_empty(tmp_path: Path) -> None:
    directory = tmp_path / "empty_profiles"
    directory.mkdir()

    with pytest.raises(ValueError):
        load_risk_profiles_from_file(directory)

    with pytest.raises(ValueError):
        load_risk_profiles_with_metadata(directory)


def test_load_risk_profiles_directory_with_origin(tmp_path: Path) -> None:
    directory = tmp_path / "profiles"
    directory.mkdir()

    (directory / "ops.json").write_text(
        json.dumps({"risk_profiles": {"ops_dir": {"severity_min": "error"}}}),
        encoding="utf-8",
    )

    _, metadata = load_risk_profiles_with_metadata(directory, origin_label="cli:profiles")
    assert metadata["origin"] == "cli:profiles"
    assert metadata["type"] == "directory"
    assert metadata["files"][0]["registered_profiles"] == ["ops_dir"]
