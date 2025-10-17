from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from pathbootstrap import (
    clear_cache,
    chdir_repo_root,
    ensure_repo_root_on_sys_path,
    get_repo_root,
    main,
    repo_on_sys_path,
)


def test_ensure_repo_root_on_sys_path_inserts_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cleaned_path = [entry for entry in sys.path if entry != str(repo_root)]
    monkeypatch.setattr(sys, "path", cleaned_path, raising=False)

    inserted = ensure_repo_root_on_sys_path(repo_root)

    assert inserted == str(repo_root)
    assert sys.path[0] == str(repo_root)


def test_ensure_repo_root_on_sys_path_appends_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    monkeypatch.setattr(sys, "path", ["/tmp/alpha"], raising=False)

    inserted = ensure_repo_root_on_sys_path(repo_root, position="append")

    assert inserted == repo_str
    assert sys.path[-1] == repo_str
    assert sys.path[0] == "/tmp/alpha"


def test_ensure_repo_root_on_sys_path_adds_additional_paths_prepend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    monkeypatch.setattr(sys, "path", ["/tmp/base"], raising=False)

    ensure_repo_root_on_sys_path(repo_root, additional_paths=("tests",))

    assert sys.path[0] == repo_str
    assert sys.path[1] == extra
    assert sys.path[2] == "/tmp/base"


def test_ensure_repo_root_on_sys_path_adds_additional_paths_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    monkeypatch.setattr(sys, "path", ["/tmp/base"], raising=False)

    ensure_repo_root_on_sys_path(
        repo_root, position="append", additional_paths=("tests",)
    )

    assert sys.path == ["/tmp/base", repo_str, extra]


def test_ensure_repo_root_on_sys_path_honours_env_additional_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    monkeypatch.setattr(sys, "path", ["/tmp/base"], raising=False)
    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "tests")

    ensure_repo_root_on_sys_path(repo_root)

    assert sys.path[0] == repo_str
    assert sys.path[1] == extra
    assert sys.path[2] == "/tmp/base"


def test_ensure_repo_root_on_sys_path_can_ignore_env_additional_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    monkeypatch.setattr(sys, "path", ["/tmp/base"], raising=False)
    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "tests")

    ensure_repo_root_on_sys_path(repo_root, use_env_additional_paths=False)

    assert sys.path[0] == repo_str
    assert extra not in sys.path


def test_ensure_repo_root_on_sys_path_ignores_duplicate_additional_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    monkeypatch.setattr(sys, "path", [extra], raising=False)

    ensure_repo_root_on_sys_path(repo_root, additional_paths=("tests",))

    assert sys.path[0] == repo_str
    assert sys.path.count(extra) == 1


def test_ensure_repo_root_on_sys_path_merges_env_and_argument_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    tests_extra = str((repo_root / "tests").resolve())
    docs_extra = str((repo_root / "docs").resolve())
    monkeypatch.setattr(sys, "path", ["/tmp/base"], raising=False)
    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", os.pathsep.join(["tests"]))

    ensure_repo_root_on_sys_path(repo_root, additional_paths=("docs",))

    assert sys.path[:3] == [repo_str, tests_extra, docs_extra]
def test_get_repo_root_does_not_touch_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "path", list(original_path), raising=False)

    discovered = get_repo_root(repo_root)

    assert discovered == repo_root
    assert sys.path == original_path


def test_ensure_repo_root_on_sys_path_requires_sentinel(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_repo_root_on_sys_path(tmp_path, sentinels=("pyproject.toml",))


def test_ensure_repo_root_on_sys_path_rejects_empty_sentinels() -> None:
    with pytest.raises(ValueError):
        ensure_repo_root_on_sys_path(sentinels=())


def test_cache_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()

    calls: list[int] = []

    def fake_discover(start: Path, sentinels: tuple[str, ...]) -> Path:  # type: ignore[override]
        calls.append(1)
        return Path(__file__).resolve().parents[1]

    monkeypatch.setattr("pathbootstrap._discover_repo_root", fake_discover)

    repo_root = Path(__file__).resolve().parents[1]
    ensure_repo_root_on_sys_path(repo_root)
    ensure_repo_root_on_sys_path(repo_root)

    assert len(calls) == 1


def test_clear_cache_forces_rediscovery(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()

    calls: list[int] = []

    def fake_discover(start: Path, sentinels: tuple[str, ...]) -> Path:  # type: ignore[override]
        calls.append(1)
        return Path(__file__).resolve().parents[1]

    monkeypatch.setattr("pathbootstrap._discover_repo_root", fake_discover)

    repo_root = Path(__file__).resolve().parents[1]
    ensure_repo_root_on_sys_path(repo_root)
    clear_cache()
    ensure_repo_root_on_sys_path(repo_root)

    assert len(calls) == 2


def test_repo_on_sys_path_temporarily_inserts(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    original_path = [entry for entry in sys.path if entry != repo_str]
    monkeypatch.setattr(sys, "path", list(original_path), raising=False)

    with repo_on_sys_path(repo_root) as discovered:
        assert discovered == repo_root
        assert sys.path[0] == repo_str

    assert sys.path == original_path


def test_repo_on_sys_path_keeps_existing_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    original_path = [repo_str, "/tmp/other"]
    monkeypatch.setattr(sys, "path", list(original_path), raising=False)

    with repo_on_sys_path(repo_root) as discovered:
        assert discovered == repo_root
        assert sys.path[0] == repo_str

    assert sys.path == original_path


def test_repo_on_sys_path_appends_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    monkeypatch.setattr(sys, "path", ["/tmp/beta"], raising=False)

    with repo_on_sys_path(repo_root, position="append") as discovered:
        assert discovered == repo_root
        assert sys.path[-1] == repo_str
        assert sys.path[0] == "/tmp/beta"

    assert sys.path == ["/tmp/beta"]


def test_repo_on_sys_path_handles_additional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    original_path = ["/tmp/base"]
    monkeypatch.setattr(sys, "path", list(original_path), raising=False)

    with repo_on_sys_path(repo_root, additional_paths=("tests",)) as discovered:
        assert discovered == repo_root
        assert sys.path[0] == repo_str
        assert sys.path[1] == extra

    assert sys.path == original_path


def test_repo_on_sys_path_includes_env_additional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    original_path = ["/tmp/base"]
    monkeypatch.setattr(sys, "path", list(original_path), raising=False)
    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "tests")

    with repo_on_sys_path(repo_root) as discovered:
        assert discovered == repo_root
        assert sys.path[0] == repo_str
        assert sys.path[1] == extra

    assert sys.path == original_path


def test_repo_on_sys_path_can_ignore_env_additional_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    original_path = ["/tmp/base"]
    monkeypatch.setattr(sys, "path", list(original_path), raising=False)
    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "tests")

    with repo_on_sys_path(repo_root, use_env_additional_paths=False) as discovered:
        assert discovered == repo_root
        assert sys.path[0] == repo_str
        assert extra not in sys.path

    assert sys.path == original_path


def test_chdir_repo_root_temporarily_switches_directory() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original_cwd = Path.cwd()

    try:
        with chdir_repo_root(repo_root) as discovered:
            assert discovered == repo_root
            assert Path.cwd() == repo_root
    finally:
        os.chdir(original_cwd)

    assert Path.cwd() == original_cwd


def test_chdir_repo_root_is_noop_when_already_in_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original_cwd = Path.cwd()

    try:
        os.chdir(repo_root)
        with chdir_repo_root(repo_root) as discovered:
            assert discovered == repo_root
            assert Path.cwd() == repo_root
    finally:
        os.chdir(original_cwd)

    assert Path.cwd() == original_cwd


def test_get_repo_root_prefers_env_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "alt_repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("", encoding="utf-8")
    nested = repo_root / "pkg"
    nested.mkdir()

    clear_cache()
    try:
        monkeypatch.setenv("PATHBOOTSTRAP_ROOT_HINT", str(nested))

        discovered = get_repo_root()

        assert discovered == repo_root
    finally:
        clear_cache()


def test_env_sentinels_override_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "alt_repo"
    repo_root.mkdir()
    (repo_root / "custom.sentinel").write_text("", encoding="utf-8")
    nested = repo_root / "pkg"
    nested.mkdir()

    clear_cache()
    try:
        monkeypatch.setenv("PATHBOOTSTRAP_ROOT_HINT", str(nested))
        monkeypatch.setenv("PATHBOOTSTRAP_SENTINELS", "custom.sentinel")

        discovered = get_repo_root(sentinels=("pyproject.toml",))

        assert discovered == repo_root
    finally:
        clear_cache()


def test_env_sentinels_reject_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()
    try:
        monkeypatch.setenv("PATHBOOTSTRAP_SENTINELS", "  :  ")

        with pytest.raises(ValueError):
            get_repo_root(Path(__file__).resolve().parent)
    finally:
        clear_cache()


def test_main_prints_repo_root(capsys: pytest.CaptureFixture[str]) -> None:
    clear_cache()
    try:
        exit_code = main([])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == str(repo_root)


def test_main_with_ensure_adds_repo_root(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    monkeypatch.setattr(sys, "path", ["/tmp/other"], raising=False)

    clear_cache()
    try:
        exit_code = main(["--ensure"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == repo_str
    assert sys.path[0] == repo_str


def test_main_with_position_append_places_repo_at_end(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    monkeypatch.setattr(sys, "path", ["/tmp/gamma"], raising=False)

    clear_cache()
    try:
        exit_code = main(["--ensure", "--position", "append"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == repo_str
    assert sys.path[-1] == repo_str
    assert sys.path[0] == "/tmp/gamma"


def test_main_with_add_path_includes_additional_entries(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    monkeypatch.setattr(sys, "path", ["/tmp/gamma"], raising=False)

    clear_cache()
    try:
        exit_code = main(["--ensure", "--add-path", "tests"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == repo_str
    assert sys.path[0] == repo_str
    assert sys.path[1] == extra


def test_main_with_set_env_prints_assignment(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    clear_cache()
    try:
        exit_code = main(["--set-env", "REPO_ROOT"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"REPO_ROOT={repo_root}"


def test_main_with_set_env_and_export_prints_export_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    clear_cache()
    try:
        exit_code = main(["--export", "--set-env", "REPO_ROOT"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"export REPO_ROOT={repo_root}"


def test_main_requires_set_env_for_export(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--export"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--set-env" in captured.err


def test_main_with_chdir_prints_repo_root(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original_cwd = Path.cwd()

    clear_cache()
    try:
        exit_code = main(["--chdir"])
        captured = capsys.readouterr()
        assert Path.cwd() == original_cwd
    finally:
        clear_cache()
        os.chdir(original_cwd)

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == str(repo_root)


def test_main_accepts_custom_sentinels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "alt_repo"
    repo_root.mkdir()
    sentinel = repo_root / "custom.sentinel"
    sentinel.write_text("", encoding="utf-8")
    nested = repo_root / "pkg"
    nested.mkdir()

    clear_cache()
    try:
        exit_code = main(
            [
                "--root-hint",
                str(nested),
                "--sentinel",
                "custom.sentinel",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == str(repo_root)


def test_main_runs_command_with_repo_on_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "target = sys.argv[1]\n"
        "import pathlib\n"
        "from pathlib import Path\n"
        "target_path = Path(target).resolve()\n"
        "sys.exit(0 if str(target_path) in sys.path else 5)\n"
    )

    clear_cache()
    try:
        exit_code = main([
            "--",
            sys.executable,
            "-c",
            code,
            str(repo_root),
        ])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_runs_command_propagating_additional_paths_to_pythonpath(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    code = (
        "import os\n"
        "import sys\n"
        "repo = sys.argv[1]\n"
        "extra = sys.argv[2]\n"
        "parts = [part for part in os.environ.get('PYTHONPATH', '').split(os.pathsep) if part]\n"
        "sys.exit(0 if parts[:2] == [repo, extra] else 17)\n"
    )

    clear_cache()
    try:
        exit_code = main(
            [
                "--add-path",
                "tests",
                "--",
                sys.executable,
                "-c",
                code,
                repo_str,
                extra,
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_runs_command_using_env_additional_paths(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    code = (
        "import os\n"
        "import sys\n"
        "repo = sys.argv[1]\n"
        "extra = sys.argv[2]\n"
        "parts = [part for part in os.environ.get('PYTHONPATH', '').split(os.pathsep) if part]\n"
        "sys.exit(0 if parts[:2] == [repo, extra] else 23)\n"
    )

    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "tests")
    clear_cache()
    try:
        exit_code = main(
            [
                "--",
                sys.executable,
                "-c",
                code,
                repo_str,
                extra,
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_runs_command_ignoring_env_additional_paths(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    extra = str((repo_root / "tests").resolve())
    code = (
        "import os\n"
        "import sys\n"
        "repo = sys.argv[1]\n"
        "parts = [part for part in os.environ.get('PYTHONPATH', '').split(os.pathsep) if part]\n"
        "sys.exit(0 if parts == [repo] else 29)\n"
    )

    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "tests")
    clear_cache()
    try:
        exit_code = main(
            [
                "--no-env-add-paths",
                "--",
                sys.executable,
                "-c",
                code,
                repo_str,
                extra,
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_runs_command_with_append_pythonpath(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    monkeypatch.setenv("PYTHONPATH", "/tmp/delta")
    code = (
        "import os\n"
        "import sys\n"
        "target = sys.argv[1]\n"
        "pythonpath = os.environ.get('PYTHONPATH', '')\n"
        "parts = [part for part in pythonpath.split(os.pathsep) if part]\n"
        "sys.exit(0 if parts and parts[-1] == target else 11)\n"
    )

    clear_cache()
    try:
        exit_code = main(
            [
                "--position",
                "append",
                "--",
                sys.executable,
                "-c",
                code,
                repo_str,
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()
        monkeypatch.delenv("PYTHONPATH", raising=False)

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_runs_command_with_custom_pythonpath_var(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    monkeypatch.setenv("ALT_PYTHONPATH", "/tmp/epsilon")
    code = (
        "import os\n"
        "import sys\n"
        "target = sys.argv[1]\n"
        "pythonpath = os.environ.get('ALT_PYTHONPATH', '')\n"
        "parts = [part for part in pythonpath.split(os.pathsep) if part]\n"
        "sys.exit(0 if parts and parts[0] == target else 13)\n"
    )

    clear_cache()
    try:
        exit_code = main(
            [
                "--pythonpath-var",
                "ALT_PYTHONPATH",
                "--ensure",
                "--",
                sys.executable,
                "-c",
                code,
                repo_str,
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()
        monkeypatch.delenv("ALT_PYTHONPATH", raising=False)

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_passes_set_env_to_command(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import os\n"
        "import sys\n"
        "target = sys.argv[1]\n"
        "value = os.environ.get('REPO_ROOT')\n"
        "sys.exit(0 if value == target else 7)\n"
    )

    clear_cache()
    try:
        exit_code = main([
            "--set-env",
            "REPO_ROOT",
            "--",
            sys.executable,
            "-c",
            code,
            str(repo_root),
        ])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_with_chdir_runs_command_in_repo(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    original_cwd = Path.cwd()
    code = (
        "import pathlib\n"
        "import sys\n"
        "target = pathlib.Path(sys.argv[1]).resolve()\n"
        "sys.exit(0 if pathlib.Path.cwd() == target else 9)\n"
    )

    clear_cache()
    try:
        exit_code = main(
            [
                "--chdir",
                "--",
                sys.executable,
                "-c",
                code,
                str(repo_root),
            ]
        )
        captured = capsys.readouterr()
        assert Path.cwd() == original_cwd
    finally:
        clear_cache()
        os.chdir(original_cwd)

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == ""


def test_main_requires_command_after_separator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "należy podać polecenie" in captured.err


def test_main_rejects_empty_pythonpath_var(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--pythonpath-var", "", "--"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--pythonpath-var" in captured.err
