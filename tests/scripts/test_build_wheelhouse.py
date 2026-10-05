import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import build_wheelhouse
from scripts.ci.bootstrap_dependency import project_requirements
from scripts.ci.validate_locked_resolution import locked_versions, validate_reports


def _write_wheel(path: Path, name: str, version: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )


def test_build_download_cmd_appends_extra_pip_args() -> None:
    args = SimpleNamespace(
        no_binary=None,
        index_url=None,
        extra_index_url=None,
        find_links=None,
        only_binary=":all:",
        requirements=None,
    )

    cmd = build_wheelhouse.build_download_cmd(
        wheelhouse=Path("wheelhouse"),
        args=args,
        packages=["PySide6==6.10.2"],
        python_executable="python",
        extra_pip_args=["--no-cache-dir", "--timeout", "120"],
    )

    assert "--only-binary" in cmd
    assert ":all:" in cmd
    assert cmd[-4:] == ["--no-cache-dir", "--timeout", "120", "PySide6==6.10.2"]


def test_download_retries_and_succeeds_on_third_attempt(monkeypatch, capsys) -> None:
    returncodes = iter([1, 1, 0])
    run_calls: list[list[str]] = []
    sleep_calls: list[int] = []

    def fake_run(cmd, check):
        run_calls.append(cmd)
        assert check is False
        return SimpleNamespace(returncode=next(returncodes))

    monkeypatch.setattr(build_wheelhouse.subprocess, "run", fake_run)
    monkeypatch.setattr(build_wheelhouse.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    build_wheelhouse.download(
        wheelhouse=Path("wheelhouse"),
        cmd=["python", "-m", "pip", "download", "PySide6==6.10.2"],
        attempts=3,
        retry_delay_seconds=7,
    )

    captured = capsys.readouterr()
    assert len(run_calls) == 3
    assert sleep_calls == [7, 7]
    assert "attempt 1/3" in captured.out
    assert "attempt 3/3" in captured.out


def test_download_raises_after_last_failed_attempt(monkeypatch) -> None:
    sleep_calls: list[int] = []

    def fake_run(_cmd, check):
        assert check is False
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(build_wheelhouse.subprocess, "run", fake_run)
    monkeypatch.setattr(build_wheelhouse.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(SystemExit, match="Download failed with exit code 2"):
        build_wheelhouse.download(
            wheelhouse=Path("wheelhouse"),
            cmd=["python", "-m", "pip", "download", "PySide6==6.10.2"],
            attempts=3,
            retry_delay_seconds=5,
        )

    assert sleep_calls == [5, 5]


@pytest.mark.parametrize("attempts", [0, -1])
def test_download_rejects_attempts_less_than_one(attempts: int) -> None:
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        build_wheelhouse.download(
            wheelhouse=Path("wheelhouse"),
            cmd=["python", "-m", "pip", "download", "PySide6==6.10.2"],
            attempts=attempts,
        )


def test_download_command_constrains_every_resolution_to_release_lock() -> None:
    args = SimpleNamespace(
        no_binary=None,
        index_url=None,
        extra_index_url=None,
        find_links=None,
        only_binary=":all:",
        requirements="deploy/packaging/requirements-desktop.lock",
    )
    cmd = build_wheelhouse.build_download_cmd(
        Path("wheelhouse"), args, ["PySide6>=6.7,<6.11"], "python"
    )
    assert cmd[-3:] == ["--constraint", args.requirements, "PySide6>=6.7,<6.11"]


def test_wheelhouse_rejects_duplicate_normalized_package_versions(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "PySide6-6.7.0-cp39-abi3-manylinux_2_28_x86_64.whl", "PySide6", "6.7.0")
    _write_wheel(
        tmp_path / "pyside6-6.10.2-cp39-abi3-manylinux_2_34_x86_64.whl", "pyside6", "6.10.2"
    )
    with pytest.raises(SystemExit, match=r"pyside6=6.10.2,6.7.0"):
        build_wheelhouse.validate_unique_wheel_versions(tmp_path)


def test_wheelhouse_accepts_one_version_per_normalized_package(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "PySide6-6.7.0-cp39-abi3-manylinux_2_28_x86_64.whl", "PySide6", "6.7.0")
    _write_wheel(tmp_path / "requests-2.33.0-py3-none-any.whl", "requests", "2.33.0")
    build_wheelhouse.validate_unique_wheel_versions(tmp_path)


def test_wheelhouse_ignores_vendored_metadata_when_identifying_owner(tmp_path: Path) -> None:
    wheel = tmp_path / "setuptools-84.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "setuptools-84.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: setuptools\nVersion: 84.0.0\n",
        )
        archive.writestr(
            "setuptools/_vendor/foo-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n",
        )
        archive.writestr(
            "setuptools/_vendor/bar-2.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: bar\nVersion: 2.0\n",
        )

    build_wheelhouse.validate_unique_wheel_versions(tmp_path)


def test_bootstrap_paths_execute_without_site_packages(tmp_path: Path) -> None:
    lock = tmp_path / "desktop.lock"
    lock.write_text("Example_Package==1.2.3 ; python_version >= '3.11'\n", encoding="utf-8")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_wheel(wheelhouse / "example.whl", "Example.Package", "1.2.3")
    root = Path(__file__).resolve().parents[2]
    program = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.ci.validate_locked_resolution import locked_versions
from scripts.ci.build_wheelhouse import validate_unique_wheel_versions
assert locked_versions(Path(sys.argv[2])) == {'example-package': '1.2.3'}
validate_unique_wheel_versions(Path(sys.argv[3]))
"""
    subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(root), str(lock), str(wheelhouse)],
        check=True,
    )
    for script in ("validate_locked_resolution.py", "build_wheelhouse.py"):
        subprocess.run(
            [sys.executable, "-I", "-S", str(root / "scripts" / "ci" / script), "--help"],
            check=True,
            capture_output=True,
        )


@pytest.mark.parametrize(
    "entry",
    ["demo>=1", "demo<=1", "demo~=1", "demo!=1", "demo==", "demo==1,==2", "demo"],
)
def test_lock_parser_rejects_non_exact_entries(tmp_path: Path, entry: str) -> None:
    lock = tmp_path / "desktop.lock"
    lock.write_text(entry + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-exact lock entry"):
        locked_versions(lock)


def _write_pip_report(
    path: Path, packages: list[tuple[str, str]], *, requested: bool = False
) -> None:
    path.write_text(
        json.dumps(
            {
                "install": [
                    {"requested": requested, "metadata": {"name": name, "version": version}}
                    for name, version in packages
                ]
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("active", "desktop", "escaped"),
    [
        ([], [("secretstorage", "3.5.0")], "secretstorage==3.5.0"),
        ([], [("dmgbuild", "1.6.7")], "dmgbuild==1.6.7"),
        ([], [("ds-store", "1.3.3")], "ds-store==1.3.3"),
        ([], [("mac-alias", "2.2.3")], "mac-alias==2.2.3"),
        ([("requests", "2.33.0")], [("requests", "2.32.5")], "requests==2.32.5"),
        ([], [("foo", "1")], "foo==1"),
    ],
)
def test_desktop_plan_rejects_dependencies_outside_native_active_lock(
    tmp_path: Path,
    active: list[tuple[str, str]],
    desktop: list[tuple[str, str]],
    escaped: str,
) -> None:
    active_report = tmp_path / "native-lock-plan.json"
    desktop_report = tmp_path / "desktop-plan.json"
    _write_pip_report(active_report, active)
    _write_pip_report(desktop_report, desktop)
    with pytest.raises(SystemExit, match=escaped):
        validate_reports(active_report, desktop_report)


def test_desktop_plan_accepts_matching_native_active_dependency(tmp_path: Path) -> None:
    active_report = tmp_path / "native-lock-plan.json"
    desktop_report = tmp_path / "desktop-plan.json"
    _write_pip_report(active_report, [("foo", "1.2.3")])
    _write_pip_report(desktop_report, [("foo", "1.2.3")])
    validate_reports(active_report, desktop_report)


@pytest.mark.parametrize(
    "dependency",
    ["pyarrow", "pyinstaller"],
    ids=["base-project-dependency", "desktop-direct-dependency"],
)
def test_requested_direct_dependency_is_not_skipped(tmp_path: Path, dependency: str) -> None:
    active_report = tmp_path / "native-lock-plan.json"
    desktop_report = tmp_path / "desktop-plan.json"
    _write_pip_report(active_report, [])
    _write_pip_report(desktop_report, [(dependency, "1")], requested=True)
    with pytest.raises(SystemExit, match=rf"{dependency}==1.*missing"):
        validate_reports(active_report, desktop_report)


def test_static_project_dependencies_fail_closed_when_dynamic(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        "[project]\nname='demo'\ndynamic=['dependencies']\n"
        "[project.optional-dependencies]\ndesktop=[]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="statically declared"):
        project_requirements(project, ("desktop",))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"install": [{}]},
        {"install": [{"metadata": {"version": "1"}}]},
        {"install": [{"metadata": {"name": "foo"}}]},
        {"install": [{"requested": "false", "metadata": {"name": "foo", "version": "1"}}]},
    ],
)
@pytest.mark.parametrize("malformed_plan", ["active", "desktop"])
def test_plan_comparison_rejects_malformed_reports(
    tmp_path: Path, payload: dict, malformed_plan: str
) -> None:
    active_report = tmp_path / "native-lock-plan.json"
    desktop_report = tmp_path / "desktop-plan.json"
    _write_pip_report(active_report, [])
    _write_pip_report(desktop_report, [])
    malformed_report = active_report if malformed_plan == "active" else desktop_report
    malformed_report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_reports(active_report, desktop_report)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("Metadata-Version: 2.1\nVersion: 1\n", "exactly one Name"),
        ("Metadata-Version: 2.1\nName: demo\n", "exactly one Version"),
    ],
)
def test_wheelhouse_rejects_incomplete_metadata(
    tmp_path: Path, metadata: str, message: str
) -> None:
    wheel = tmp_path / "demo.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo-1.dist-info/METADATA", metadata)
    with pytest.raises(ValueError, match=message):
        build_wheelhouse.validate_unique_wheel_versions(tmp_path)


@pytest.mark.parametrize("metadata_count", [0, 2])
def test_wheelhouse_requires_exactly_one_metadata_file(tmp_path: Path, metadata_count: int) -> None:
    wheel = tmp_path / "demo.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for index in range(metadata_count):
            archive.writestr(
                f"demo{index}-1.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: demo\nVersion: 1\n",
            )
        if metadata_count == 0:
            archive.writestr(
                "vendor/foo-1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: foo\nVersion: 1.0\n",
            )
    with pytest.raises(ValueError, match="exactly one top-level .dist-info/METADATA"):
        build_wheelhouse.validate_unique_wheel_versions(tmp_path)


def test_wheelhouse_validates_actual_desktop_target(monkeypatch) -> None:
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        build_wheelhouse.subprocess,
        "run",
        lambda command, check: calls.append((command, check)),
    )
    build_wheelhouse.validate_desktop_resolution("python3.11", "desktop.lock")
    assert calls == [
        (
            [
                "python3.11",
                "scripts/ci/validate_locked_resolution.py",
                "desktop.lock",
                "--target",
                ".[desktop]",
            ],
            True,
        )
    ]
