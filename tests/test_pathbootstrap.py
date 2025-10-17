from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pathbootstrap
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


def test_format_env_assignment_plain() -> None:
    assert pathbootstrap._format_env_assignment("VAR", "/tmp/path", "plain") == "VAR=/tmp/path"


def test_format_env_assignment_posix() -> None:
    assert (
        pathbootstrap._format_env_assignment("VAR", "/tmp/path", "posix")
        == "export VAR=/tmp/path"
    )


def test_format_env_assignment_powershell() -> None:
    assert (
        pathbootstrap._format_env_assignment("VAR", "C:/Program Files/App", "powershell")
        == "$Env:VAR = 'C:/Program Files/App'"
    )


def test_format_env_assignment_powershell_escapes_single_quotes() -> None:
    assert (
        pathbootstrap._format_env_assignment("VAR", "C:/O'Brian", "powershell")
        == "$Env:VAR = 'C:/O''Brian'"
    )


def test_format_env_assignment_cmd() -> None:
    assert (
        pathbootstrap._format_env_assignment("VAR", "C:/Program Files/App", "cmd")
        == "set VAR=C:/Program Files/App"
    )


def test_format_env_assignment_unknown() -> None:
    with pytest.raises(ValueError):
        pathbootstrap._format_env_assignment("VAR", "VALUE", "unknown")
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


def test_main_prints_json_when_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--format", "json"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "repo_root": str(repo_root),
        "additional_paths": [],
    }


def test_main_prints_json_with_additional_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--format", "json", "--add-path", "tests"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    expected_path = str((repo_root / "tests").resolve())
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["repo_root"] == str(repo_root)
    assert payload["additional_paths"] == [expected_path]


def test_main_prints_json_with_additional_paths_from_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    path_file = tmp_path / "paths.txt"
    path_file.write_text("# komentarz\n tests \n docs\n", encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(["--format", "json", "--add-path-file", str(path_file)])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(captured.out)
    expected_tests = str((repo_root / "tests").resolve())
    expected_docs = str((repo_root / "docs").resolve())

    assert exit_code == 0
    assert captured.err == ""
    assert payload["additional_paths"] == [expected_tests, expected_docs]


def test_main_combines_additional_paths_file_env_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "data")
    path_file = tmp_path / "paths.txt"
    path_file.write_text("docs\n", encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(
            [
                "--format",
                "json",
                "--add-path-file",
                str(path_file),
                "--add-path",
                "tests",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(captured.out)
    expected_data = str((repo_root / "data").resolve())
    expected_docs = str((repo_root / "docs").resolve())
    expected_tests = str((repo_root / "tests").resolve())

    assert exit_code == 0
    assert captured.err == ""
    assert payload["additional_paths"] == [expected_data, expected_docs, expected_tests]


def test_main_prints_pythonpath_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--print-pythonpath", "--add-path", "tests"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    tests_path = str((repo_root / "tests").resolve())
    expected_pythonpath = os.pathsep.join([str(repo_root), tests_path])

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == expected_pythonpath


def test_main_prints_pythonpath_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(
            ["--print-pythonpath", "--format", "json", "--add-path", "tests"]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    tests_path = str((repo_root / "tests").resolve())
    expected_pythonpath = os.pathsep.join([str(repo_root), tests_path])
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "repo_root": str(repo_root),
        "additional_paths": [tests_path],
        "pythonpath": expected_pythonpath,
        "pythonpath_entries": [str(repo_root), tests_path],
    }


def test_main_print_pythonpath_rejects_set_env(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-pythonpath", "--set-env", "ROOT"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-pythonpath" in captured.err


def test_main_print_pythonpath_rejects_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-pythonpath", "--", "python", "-c", "pass"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-pythonpath" in captured.err


def test_main_prints_sys_path_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    monkeypatch.setattr(pathbootstrap.sys, "path", ["/tmp/base"], raising=False)

    clear_cache()
    try:
        exit_code = main(["--print-sys-path", "--add-path", "tests"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    tests_path = str((repo_root / "tests").resolve())

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == [
        str(repo_root),
        tests_path,
        "/tmp/base",
    ]


def test_main_prints_sys_path_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    monkeypatch.setattr(pathbootstrap.sys, "path", ["/tmp/base"], raising=False)

    clear_cache()
    try:
        exit_code = main(["--print-sys-path", "--format", "json", "--add-path", "tests"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    tests_path = str((repo_root / "tests").resolve())
    expected_entries = [str(repo_root), tests_path, "/tmp/base"]
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "repo_root": str(repo_root),
        "additional_paths": [tests_path],
        "sys_path": "\n".join(expected_entries),
        "sys_path_entries": expected_entries,
    }


def test_main_print_sys_path_rejects_set_env(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-sys-path", "--set-env", "ROOT"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-sys-path" in captured.err


def test_main_print_sys_path_rejects_command(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-sys-path", "--", "python", "-c", "pass"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-sys-path" in captured.err


def test_main_print_sys_path_rejects_print_pythonpath(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-sys-path", "--print-pythonpath"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-sys-path" in captured.err


def test_main_print_config_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    monkeypatch.delenv("PATHBOOTSTRAP_SENTINELS", raising=False)
    config_file = tmp_path / "paths.txt"
    config_file.write_text("extras/lib\n", encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-config",
                "--add-path",
                "tests",
                "--add-path-file",
                str(config_file),
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    normalized_from_file = str((repo_root / "extras/lib").resolve())
    normalized_tests = str((repo_root / "tests").resolve())

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == [
        f"repo_root: {repo_root}",
        "sentinels: pyproject.toml",
        "sentinels_cli: (brak)",
        "sentinels_file: (brak)",
        "sentinels_env: (brak)",
        "position: prepend",
        "include_env_additional_paths: True",
        "set_env_format: plain",
        "additional_paths_normalized:",
        f"  - {normalized_from_file}",
        f"  - {normalized_tests}",
        "additional_paths_cli:",
        "  - tests",
        "additional_paths_files:",
        "  - extras/lib",
        "additional_paths_env:",
        "  (brak)",
        "pythonpath_var: PYTHONPATH",
        "ensure: False",
        "chdir: False",
    ]


def test_main_print_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_paths = os.pathsep.join(["env_extra", str(tmp_path / "absolute")])
    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", env_paths)
    monkeypatch.setenv("PATHBOOTSTRAP_SENTINELS", os.pathsep.join(["pyproject.toml", "env_only.txt"]))
    sentinel_file = tmp_path / "sentinels.txt"
    sentinel_file.write_text("pyproject.toml\nother.txt\n", encoding="utf-8")
    path_file = tmp_path / "extra_paths.txt"
    path_file.write_text("from_file\n", encoding="utf-8")
    (tmp_path / "absolute").mkdir(exist_ok=True)

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-config",
                "--format",
                "json",
                "--sentinel",
                "custom.cfg",
                "--sentinel-file",
                str(sentinel_file),
                "--add-path",
                "tests",
                "--add-path-file",
                str(path_file),
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(captured.out)
    normalized_env = str((repo_root / "env_extra").resolve())
    normalized_abs = str((tmp_path / "absolute").resolve())
    normalized_file = str((repo_root / "from_file").resolve())
    normalized_tests = str((repo_root / "tests").resolve())

    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "repo_root": str(repo_root),
        "sentinels": ["pyproject.toml", "env_only.txt"],
        "sentinels_cli": ["custom.cfg"],
        "sentinels_file": ["pyproject.toml", "other.txt"],
        "sentinels_env": ["pyproject.toml", "env_only.txt"],
        "position": "prepend",
        "include_env_additional_paths": True,
        "set_env_format": "plain",
        "additional_paths": {
            "normalized": [
                normalized_env,
                normalized_abs,
                normalized_file,
                normalized_tests,
            ],
            "cli": ["tests"],
            "files": ["from_file"],
            "env": ["env_extra", str(tmp_path / "absolute")],
        },
        "pythonpath_var": "PYTHONPATH",
        "ensure": False,
        "chdir": False,
    }


def test_main_print_config_rejects_set_env(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--set-env", "ROOT"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-config" in captured.err


def test_main_print_config_rejects_command(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--", "python", "-c", "pass"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-config" in captured.err


def test_main_print_config_rejects_print_pythonpath(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--print-pythonpath"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-config" in captured.err


def test_main_print_config_rejects_print_sys_path(
    capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--print-sys-path"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-config" in captured.err


def test_main_writes_output_to_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "repo.txt"

    clear_cache()
    try:
        exit_code = main(["--output", str(destination)])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]

    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert destination.read_text(encoding="utf-8") == f"{repo_root}\n"


def test_main_writes_json_output_to_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    destination = tmp_path / "result.json"

    clear_cache()
    try:
        exit_code = main(["--format", "json", "--output", str(destination)])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert payload == {
        "repo_root": str(repo_root),
        "additional_paths": [],
    }


def test_main_rejects_json_format_with_set_env(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--format", "json", "--set-env", "ROOT"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--format=json" in captured.err


def test_main_verbose_prints_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    clear_cache()
    try:
        exit_code = main(["--verbose"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    assert exit_code == 0
    assert captured.out.strip() == str(repo_root)
    assert "[pathbootstrap] katalog repozytorium:" in captured.err
    assert "PATHBOOTSTRAP_ADD_PATHS" in captured.err
    assert "dodatkowe ścieżki z plików" in captured.err


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


def test_main_with_set_env_format_posix(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    clear_cache()
    try:
        exit_code = main(["--set-env-format", "posix", "--set-env", "REPO_ROOT"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"export REPO_ROOT={repo_root}"


def test_main_with_set_env_format_powershell(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    clear_cache()
    try:
        exit_code = main(["--set-env-format", "powershell", "--set-env", "REPO_ROOT"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"$Env:REPO_ROOT = '{repo_root}'"


def test_main_with_set_env_format_cmd(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    clear_cache()
    try:
        exit_code = main(["--set-env-format", "cmd", "--set-env", "REPO_ROOT"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"set REPO_ROOT={repo_root}"


def test_main_requires_set_env_for_export(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--export"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--set-env" in captured.err


def test_main_requires_set_env_for_set_env_format(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--set-env-format", "powershell"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--set-env-format" in captured.err


def test_main_export_rejects_incompatible_set_env_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--export", "--set-env-format", "powershell", "--set-env", "ROOT"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--set-env-format" in captured.err


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


def test_main_accepts_sentinels_from_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "alt_repo"
    repo_root.mkdir()
    sentinel = repo_root / "custom.sentinel"
    sentinel.write_text("", encoding="utf-8")
    nested = repo_root / "pkg"
    nested.mkdir()
    sentinel_file = tmp_path / "sentinels.txt"
    sentinel_file.write_text("# komentarz\n custom.sentinel \n\n", encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(
            [
                "--root-hint",
                str(nested),
                "--sentinel-file",
                str(sentinel_file),
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == str(repo_root)


def test_main_combines_sentinel_file_with_cli_sentinel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "combo_repo"
    repo_root.mkdir()
    file_marker = repo_root / "file.sentinel"
    file_marker.write_text("", encoding="utf-8")
    cli_marker = repo_root / "cli.sentinel"
    cli_marker.write_text("", encoding="utf-8")
    nested = repo_root / "pkg"
    nested.mkdir()
    sentinel_file = tmp_path / "sentinels.txt"
    sentinel_file.write_text("file.sentinel\n", encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(
            [
                "--root-hint",
                str(nested),
                "--sentinel-file",
                str(sentinel_file),
                "--sentinel",
                "cli.sentinel",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == str(repo_root)


def test_main_rejects_missing_sentinel_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.txt"

    with pytest.raises(SystemExit) as excinfo:
        main(["--sentinel-file", str(missing)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "nie istnieje" in captured.err
    assert captured.out == ""


def test_main_rejects_empty_sentinel_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "sentinels.txt"
    empty.write_text("# tylko komentarze\n\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--sentinel-file", str(empty)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "nie zawiera żadnych nazw" in captured.err
    assert captured.out == ""


def test_main_rejects_missing_additional_path_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.txt"

    with pytest.raises(SystemExit) as excinfo:
        main(["--add-path-file", str(missing)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "plik ścieżek" in captured.err
    assert captured.out == ""


def test_main_rejects_empty_additional_path_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "paths.txt"
    empty.write_text("# brak danych\n\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--add-path-file", str(empty)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "nie zawiera żadnych ścieżek" in captured.err
    assert captured.out == ""


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


def test_main_rejects_output_with_command(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    destination = tmp_path / "output.txt"

    with pytest.raises(SystemExit) as excinfo:
        main([
            "--output",
            str(destination),
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(0)",
        ])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--output" in captured.err
    assert captured.out == ""


def test_main_verbose_reports_command_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clear_cache()
    try:
        exit_code = main(
            [
                "--verbose",
                "--",
                sys.executable,
                "-c",
                "import sys; sys.exit(0)",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.out == ""
    assert "[pathbootstrap] uruchamianie polecenia:" in captured.err
    assert "aktualizacja" in captured.err


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
