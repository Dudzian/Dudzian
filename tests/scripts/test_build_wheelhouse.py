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
