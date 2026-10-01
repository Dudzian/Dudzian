from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import build_wheelhouse


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
    cmd = build_wheelhouse.build_download_cmd(Path("wheelhouse"), args, [".[desktop]"], "python")
    assert cmd[-3:] == ["--constraint", args.requirements, ".[desktop]"]


def test_wheelhouse_rejects_duplicate_normalized_package_versions(tmp_path: Path) -> None:
    (tmp_path / "PySide6-6.7.0-cp39-abi3-manylinux_2_28_x86_64.whl").touch()
    (tmp_path / "pyside6-6.10.2-cp39-abi3-manylinux_2_34_x86_64.whl").touch()
    with pytest.raises(SystemExit, match=r"pyside6=6.10.2,6.7.0"):
        build_wheelhouse.validate_unique_wheel_versions(tmp_path)


def test_wheelhouse_accepts_one_version_per_normalized_package(tmp_path: Path) -> None:
    (tmp_path / "PySide6-6.7.0-cp39-abi3-manylinux_2_28_x86_64.whl").touch()
    (tmp_path / "requests-2.33.0-py3-none-any.whl").touch()
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
