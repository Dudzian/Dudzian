import logging
from pathlib import Path

import pytest

from scripts.packaging import build_app_bundles as bundles


def _recording_run(commands):
    def _fake_run(command, *, cwd=None):
        commands.append((tuple(command), Path(cwd) if cwd else None))

    return _fake_run


def test_build_pyinstaller_invokes_pyinstaller(monkeypatch, tmp_path):
    entry = tmp_path / "launcher.py"
    entry.write_text("print('hi')", encoding="utf-8")

    recorded = []

    monkeypatch.setattr(bundles, "_run", _recording_run(recorded))

    dist_dir = tmp_path / "dist"
    work_dir = tmp_path / "build"

    executable = bundles._build_pyinstaller(entry, dist_dir, work_dir, "linux")

    expected_command = (
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        str(entry),
    )

    assert recorded[0][0] == expected_command
    assert executable == dist_dir / "launcher" / "launcher"
    assert dist_dir.exists()
    assert work_dir.exists()


def test_package_briefcase_copies_artifacts(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(bundles, "_run", _recording_run(recorded))

    project_root = tmp_path
    monkeypatch.chdir(project_root)

    dist_root = project_root / "dist"
    dist_root.mkdir()
    (dist_root / "FinanceApp-1.0").mkdir()
    (dist_root / "FinanceApp-1.0.zip").write_bytes(b"zip")

    output_dir = project_root / "bundles"

    result = bundles._package_briefcase("FinanceApp", "linux", output_dir)

    # Ensure every Briefcase phase was scheduled.
    expected_commands = {
        ("briefcase", "create", "linux", "FinanceApp"),
        ("briefcase", "build", "linux", "FinanceApp"),
        ("briefcase", "package", "linux", "FinanceApp"),
    }
    assert {cmd for cmd, _ in recorded} == expected_commands

    copied = sorted(path.name for path in output_dir.iterdir())
    assert copied == ["FinanceApp-1.0", "FinanceApp-1.0.zip"]
    assert result == output_dir


def test_main_dispatches_selected_tasks(monkeypatch, tmp_path):
    entry = tmp_path / "main.py"
    entry.write_text("print('hello')", encoding="utf-8")

    pyinstaller_calls: list[Path] = []
    briefcase_calls: list[tuple[str, str]] = []

    def fake_build(entrypoint, dist_dir, work_dir, platform_id):
        pyinstaller_calls.append(entrypoint)
        executable = dist_dir / entrypoint.stem / entrypoint.stem
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
        return executable

    def fake_package(app, platform_id, output_dir):
        briefcase_calls.append((app, platform_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    monkeypatch.setattr(bundles, "_build_pyinstaller", fake_build)
    monkeypatch.setattr(bundles, "_package_briefcase", fake_package)

    args = [
        "--pyinstaller-entry",
        str(entry),
        "--briefcase-app",
        "FinanceApp",
        "--briefcase-output",
        str(tmp_path / "out"),
        "--platform",
        "Windows",
    ]

    assert bundles.main(args) == 0
    assert pyinstaller_calls == [entry.resolve()]
    assert briefcase_calls == [("FinanceApp", "windows")]


def test_main_requires_existing_entrypoint(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(SystemExit):
        bundles.main(["--pyinstaller-entry", str(missing)])


def test_main_warns_when_no_tasks(caplog):
    caplog.set_level(logging.WARNING)
    assert bundles.main([]) == 0
    assert any("pomijam" in message for message in caplog.messages)
