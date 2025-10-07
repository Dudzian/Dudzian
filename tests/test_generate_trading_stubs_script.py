from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from scripts import generate_trading_stubs


@pytest.fixture(autouse=True)
def _cleanup_grpc_tools(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    original = sys.modules.pop("grpc_tools", None)

    def restore() -> None:
        if original is not None:
            sys.modules["grpc_tools"] = original
        else:
            sys.modules.pop("grpc_tools", None)

    yield restore
    restore()


def test_generates_python_and_cpp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        calls.append(cmd)

    def fake_which(name: str) -> str | None:
        return str(tmp_path / name)

    sys.modules["grpc_tools"] = SimpleNamespace()
    monkeypatch.setattr(generate_trading_stubs.subprocess, "run", fake_run)
    monkeypatch.setattr(generate_trading_stubs.shutil, "which", fake_which)

    result = generate_trading_stubs.main(
        [
            f"--proto-path={Path('proto').as_posix()}",
            "--proto-file=trading.proto",
            f"--out-python={tmp_path/'py'}",
            f"--out-cpp={tmp_path/'cpp'}",
        ]
    )

    assert result == 0
    assert len(calls) == 2
    python_cmd, cpp_cmd = calls
    assert "grpc_tools.protoc" in " ".join(python_cmd)
    assert "--cpp_out" in " ".join(cpp_cmd)


def test_missing_grpc_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import builtins

    monkeypatch.setattr(generate_trading_stubs.shutil, "which", lambda name: str(tmp_path / name))
    monkeypatch.delitem(sys.modules, "grpc_tools", raising=False)

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[override]
        if name == "grpc_tools":
            raise ImportError("grpc_tools not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc:
        generate_trading_stubs.main(["--proto-path", str(Path("proto"))])

    assert exc.value.code == 2


def test_skip_python_allows_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(generate_trading_stubs.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.setattr(generate_trading_stubs.shutil, "which", lambda name: str(tmp_path / name))

    result = generate_trading_stubs.main(["--skip-python", "--proto-path", str(Path("proto"))])

    assert result == 0
    assert len(calls) == 1  # tylko generowanie C++


def test_dry_run_prints_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sys.modules["grpc_tools"] = SimpleNamespace()
    monkeypatch.setattr(generate_trading_stubs.shutil, "which", lambda name: str(tmp_path / name))

    result = generate_trading_stubs.main(["--dry-run", "--proto-path", str(Path("proto"))])

    captured = capsys.readouterr()
    assert result == 0
    assert "DRY-RUN" in captured.out
