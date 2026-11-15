"""The native Mypy jobs must install their parser and typing closure offline."""

from pathlib import Path
import tomllib

import pytest
import yaml

from scripts.ci.bootstrap_dependency import project_requirements


ROOT = Path(__file__).resolve().parents[2]


def test_dev_wheelhouse_requirements_include_hardened_parser_and_typing() -> None:
    requirements = project_requirements(ROOT / "pyproject.toml", ("dev",))
    assert "defusedxml==0.7.1" in requirements
    assert "types-defusedxml==0.7.0.20240218" in requirements
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["mypy"]
    assert "deployment/stage9_observer_kvm_domain_v5d1.py" in config["files"]
    assert not config.get("ignore_missing_imports", False)
    for override in config.get("overrides", []):
        if override.get("ignore_missing_imports"):
            assert not any("defusedxml" in module for module in override["module"])


@pytest.mark.parametrize(
    ("job_id", "runner", "check_name"),
    [
        ("mypy-core", "windows-latest", "Mypy (core packages)"),
        ("mypy-core-linux", "ubuntu-24.04", "Mypy (core packages, Linux)"),
    ],
)
def test_native_mypy_jobs_use_dev_extra_from_required_wheelhouse(
    job_id: str, runner: str, check_name: str
) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    job = jobs[job_id]
    assert job["runs-on"] == runner
    assert job["name"] == check_name
    if job_id == "mypy-core-linux":
        assert job["permissions"] == {"contents": "read"}
    assert job["if"] == "github.event_name != 'schedule'"
    assert job["needs"] == "prepare-wheelhouse"
    assert not job.get("continue-on-error", False)
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Download wheelhouse"]["with"]["name"] == (
        "wheelhouse-${{ runner.os }}-py${{ env.PYTHON_VERSION }}"
    )
    assert "--require-wheelhouse -- .[dev]" in steps["Install dependencies"]["run"]
    assert '--wheelhouse "$WHEELHOUSE_DIR"' in steps["Install dependencies"]["run"]
    assert steps["Run mypy for core packages"]["run"] == "python -m mypy"
