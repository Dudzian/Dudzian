from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pathbootstrap
import pytest

from pathbootstrap import (
    clear_cache,
    chdir_repo_root,
    ensure_repo_root_on_sys_path,
    get_repo_info,
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


def test_ensure_repo_root_on_sys_path_expands_env_and_user_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    home_dir = tmp_path / "home"
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    home_lib = home_dir / "lib"
    env_pkg = env_dir / "pkg"
    home_lib.mkdir(parents=True)
    env_pkg.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("CUSTOM_LIB_DIR", str(env_dir))
    monkeypatch.setenv(
        "PATHBOOTSTRAP_ADD_PATHS",
        os.pathsep.join(["~/lib", "${CUSTOM_LIB_DIR}/pkg"]),
    )
    monkeypatch.setattr(sys, "path", ["/tmp/base"], raising=False)

    ensure_repo_root_on_sys_path(repo_root)

    expected_home = str(home_lib.resolve())
    expected_env = str(env_pkg.resolve())

    assert sys.path[:3] == [repo_str, expected_home, expected_env]


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


def test_get_repo_root_respects_max_depth(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "pkg" / "module"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        get_repo_root(
            nested,
            sentinels=("pyproject.toml",),
            allow_git=False,
            max_depth=1,
        )

    discovered = get_repo_root(
        nested,
        sentinels=("pyproject.toml",),
        allow_git=False,
        max_depth=2,
    )

    assert discovered == repo_root


def _skip_if_git_missing() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable not available")


def _init_git_repository(path: Path) -> None:
    subprocess.run(
        ["git", "init"],
        check=True,
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_get_repo_root_allows_git_fallback_when_enabled(
    tmp_path: Path,
) -> None:
    _skip_if_git_missing()
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()
    _init_git_repository(repo_root)
    nested = repo_root / "pkg"
    nested.mkdir()

    clear_cache()
    try:
        with pytest.raises(FileNotFoundError):
            get_repo_root(nested, sentinels=("missing.marker",), allow_git=False)

        clear_cache()
        discovered = get_repo_root(nested, sentinels=("missing.marker",), allow_git=True)
    finally:
        clear_cache()

    assert discovered == repo_root.resolve()


def test_get_repo_root_git_fallback_respects_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_if_git_missing()
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()
    _init_git_repository(repo_root)
    nested = repo_root / "pkg"
    nested.mkdir()

    clear_cache()
    try:
        monkeypatch.setenv("PATHBOOTSTRAP_ALLOW_GIT", "1")
        discovered = get_repo_root(nested, sentinels=("missing.marker",))
        assert discovered == repo_root.resolve()

        clear_cache()
        monkeypatch.setenv("PATHBOOTSTRAP_ALLOW_GIT", "0")
        with pytest.raises(FileNotFoundError):
            get_repo_root(nested, sentinels=("missing.marker",))
    finally:
        clear_cache()
        monkeypatch.delenv("PATHBOOTSTRAP_ALLOW_GIT", raising=False)


def test_get_repo_info_reports_sentinel_details() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    info = get_repo_info(repo_root, sentinels=("pyproject.toml",))

    assert info.root == repo_root
    assert info.method == "sentinel"
    assert info.sentinel == "pyproject.toml"
    assert info.depth == 0
    assert info.start == repo_root


def test_get_repo_info_uses_git_when_requested(tmp_path: Path) -> None:
    _skip_if_git_missing()
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()
    _init_git_repository(repo_root)
    nested = repo_root / "pkg"
    nested.mkdir()

    clear_cache()
    try:
        info = get_repo_info(
            nested,
            sentinels=("missing.marker",),
            allow_git=True,
        )
    finally:
        clear_cache()

    assert info.root == repo_root.resolve()
    assert info.method == "git"
    assert info.sentinel is None
    assert info.depth is None
    assert info.start == nested.resolve()


def test_ensure_repo_root_on_sys_path_requires_sentinel(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_repo_root_on_sys_path(tmp_path, sentinels=("pyproject.toml",))


def test_ensure_repo_root_on_sys_path_rejects_empty_sentinels() -> None:
    with pytest.raises(ValueError):
        ensure_repo_root_on_sys_path(sentinels=())


def test_cache_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()

    calls: list[int] = []

    def fake_discover(
        start: Path, sentinels: tuple[str, ...], allow_git: bool, *, max_depth: Optional[int]
    ) -> pathbootstrap.RepoDiscovery:  # type: ignore[override]
        calls.append(1)
        root = Path(__file__).resolve().parents[1]
        return pathbootstrap.RepoDiscovery(root, "sentinel", "pyproject.toml", 0, start)

    monkeypatch.setattr("pathbootstrap._discover_repo_root", fake_discover)

    repo_root = Path(__file__).resolve().parents[1]
    ensure_repo_root_on_sys_path(repo_root)
    ensure_repo_root_on_sys_path(repo_root)

    assert len(calls) == 1


def test_clear_cache_forces_rediscovery(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_cache()

    calls: list[int] = []

    def fake_discover(
        start: Path, sentinels: tuple[str, ...], allow_git: bool, *, max_depth: Optional[int]
    ) -> pathbootstrap.RepoDiscovery:  # type: ignore[override]
        calls.append(1)
        root = Path(__file__).resolve().parents[1]
        return pathbootstrap.RepoDiscovery(root, "sentinel", "pyproject.toml", 0, start)

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


def test_main_prints_repo_root_windows_style(capsys: pytest.CaptureFixture[str]) -> None:
    clear_cache()
    try:
        exit_code = main(["--path-style", "windows"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    expected = str(repo_root).replace("/", "\\")

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == expected


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
    repo_root = Path(__file__).resolve().parents[1]

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


def test_main_print_pythonpath_windows_style_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-pythonpath",
                "--add-path",
                "tests",
                "--format",
                "json",
                "--path-style",
                "windows",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    expected_repo = str(repo_root).replace("/", "\\")
    expected_tests = str((repo_root / "tests").resolve()).replace("/", "\\")
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "repo_root": expected_repo,
        "additional_paths": [expected_tests],
        "pythonpath": ";".join([expected_repo, expected_tests]),
        "pythonpath_entries": [expected_repo, expected_tests],
    }


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


def test_main_print_discovery_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_SENTINELS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--print-discovery"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    lines = captured.out.strip().splitlines()
    data = dict(line.split(": ", 1) for line in lines)

    assert exit_code == 0
    assert captured.err == ""
    assert data["discovery_method"] == "sentinel"
    assert data["discovery_sentinel"] == "pyproject.toml"
    assert data["discovery_depth"] == "0"
    assert data["discovery_root"] == str(repo_root)
    assert data["discovery_root_raw"] == str(repo_root)
    assert data["discovery_start"] == str(repo_root)
    assert data["discovery_start_display"] == str(repo_root)
    assert data["discovery_allow_git_effective"] == "False"
    assert data["discovery_allow_git_cli"] == "(brak)"
    assert data["discovery_allow_git_config"] == "(brak)"
    assert data["discovery_allow_git_env"] == "(brak)"
    assert data["discovery_allow_git_env_raw"] == "(brak)"
    assert data["discovery_max_depth_effective"] == "(brak)"
    assert data["discovery_max_depth_cli"] == "(brak)"
    assert data["discovery_max_depth_config"] == "(brak)"
    assert data["discovery_max_depth_env"] == "(brak)"
    assert data["discovery_max_depth_env_raw"] == "(brak)"
    assert data["discovery_sentinels_effective"] == "pyproject.toml"
    assert data["discovery_sentinels_candidates"] == "pyproject.toml"


def test_main_print_discovery_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_SENTINELS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--print-discovery", "--format", "json"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["method"] == "sentinel"
    assert payload["sentinel"] == "pyproject.toml"
    assert payload["depth"] == 0
    assert payload["root"] == str(repo_root)
    assert payload["root_raw"] == str(repo_root)
    assert payload["start"] == str(repo_root)
    assert payload["start_display"] == str(repo_root)
    assert payload["sentinels"] == ["pyproject.toml"]
    assert payload["sentinel_candidates"] == ["pyproject.toml"]
    assert payload["allow_git"] == {
        "effective": False,
        "cli": None,
        "config": None,
        "env": None,
        "env_raw": None,
    }
    assert payload["max_depth"] == {
        "effective": None,
        "cli": None,
        "config": None,
        "env": None,
        "env_raw": None,
    }


def test_main_print_discovery_rejects_set_env(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-discovery", "--set-env", "ROOT"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--print-discovery" in captured.err


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


def test_main_print_pythonpath_rejects_module(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--print-pythonpath", "--module", "pathbootstrap"])

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


def test_main_prints_sys_path_expands_cli_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    home_dir = tmp_path / "home"
    env_dir = tmp_path / "env"
    home_target = home_dir / "lib"
    env_target = env_dir / "pkg"
    home_target.mkdir(parents=True)
    env_target.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("EXTRA_LIB_DIR", str(env_dir))
    monkeypatch.setattr(pathbootstrap.sys, "path", ["/tmp/base"], raising=False)

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-sys-path",
                "--format",
                "json",
                "--add-path",
                "~/lib",
                "--add-path",
                "${EXTRA_LIB_DIR}/pkg",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    expected_home = str(home_target.resolve())
    expected_env = str(env_target.resolve())
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["repo_root"] == str(repo_root)
    assert payload["additional_paths"] == [expected_home, expected_env]
    assert payload["sys_path_entries"][1:3] == [expected_home, expected_env]


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
        "allow_git_effective: False",
        "allow_git_cli: (brak)",
        "allow_git_env: (brak)",
        "allow_git_env_raw: (brak)",
        "max_depth_effective: (brak)",
        "max_depth_cli: (brak)",
        "max_depth_config: (brak)",
        "max_depth_env: (brak)",
        "max_depth_env_raw: (brak)",
        "position: prepend",
        "include_env_additional_paths: True",
        "set_env_format: plain",
        "path_style: auto",
        "additional_paths_normalized:",
        f"  - {normalized_from_file}",
        f"  - {normalized_tests}",
        "additional_paths_config:",
        "  (brak)",
        "additional_paths_cli:",
        "  - tests",
        "additional_paths_config_files:",
        "  (brak)",
        "additional_paths_cli_files:",
        "  - extras/lib",
        "additional_paths_env:",
        "  (brak)",
        "pythonpath_var: PYTHONPATH",
        "ensure: False",
        "chdir: False",
        "shell_enabled: False",
        "discovery_method: sentinel",
        "discovery_sentinel: pyproject.toml",
        "discovery_depth: 0",
        f"discovery_start: {repo_root}",
        f"discovery_start_display: {repo_root}",
        "config_files:",
        "  (brak)",
        "config_inline_sources:",
        "  (brak)",
        "config_inline_definitions: (brak)",
        "config_root_hint: (brak)",
        "config_sentinels:",
        "  (brak)",
        "config_additional_paths:",
        "  (brak)",
        "config_additional_path_files:",
        "  (brak)",
        "config_allow_git: (brak)",
        "config_max_depth: (brak)",
        "config_use_env_additional_paths: (brak)",
        "config_position: (brak)",
        "config_path_style: (brak)",
        "config_pythonpath_var: (brak)",
        "config_profiles_defined:",
        "  (brak)",
        "config_default_profiles:",
        "  (brak)",
        "profiles_env:",
        "  (brak)",
        "profiles_env_removed:",
        "  (brak)",
        "profiles_cli:",
        "  (brak)",
        "profiles_cli_removed:",
        "  (brak)",
        "profiles_selected:",
        "  (brak)",
        "config_profiles_definitions: (brak)",
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
        "allow_git": {
            "effective": False,
            "cli": None,
            "env": None,
            "env_raw": None,
        },
        "max_depth": {
            "effective": None,
            "cli": None,
            "config": None,
            "env": None,
            "env_raw": None,
        },
        "position": "prepend",
        "include_env_additional_paths": True,
        "set_env_format": "plain",
        "shell": {"enabled": False, "path": None, "args": []},
        "path_style": "auto",
        "discovery": {
            "method": "sentinel",
            "sentinel": "pyproject.toml",
            "depth": 0,
            "start": str(repo_root),
            "start_display": str(repo_root),
        },
        "additional_paths": {
            "normalized": [
                normalized_env,
                normalized_abs,
                normalized_file,
                normalized_tests,
            ],
            "cli": ["tests"],
            "cli_files": ["from_file"],
            "config": [],
            "config_files": [],
            "env": ["env_extra", str(tmp_path / "absolute")],
        },
        "pythonpath_var": "PYTHONPATH",
        "ensure": False,
        "chdir": False,
        "config": {
            "files": [],
            "includes": [],
            "inline": {"sources": [], "entries": []},
            "values": {
                "root_hint": None,
                "sentinels": [],
                "additional_paths": [],
                "additional_path_files": [],
                "allow_git": None,
                "max_depth": None,
                "use_env_additional_paths": None,
                "position": None,
                "path_style": None,
                "pythonpath_var": None,
            },
            "profiles": {
                "defined": [],
                "default": [],
                "env": [],
                "env_removed": [],
                "cli": [],
                "cli_removed": [],
                "selected": [],
                "definitions": {},
            },
        },
    }


def test_main_config_inline_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_CONFIG_INLINE", raising=False)
    inline_payload = {
        "sentinels": ["pyproject.toml", "inline.cfg"],
        "default_profiles": ["demo"],
        "profiles": {
            "demo": {
                "path_style": "posix",
            }
        },
    }
    inline_str = json.dumps(inline_payload)

    clear_cache()
    try:
        exit_code = main([
            "--print-config",
            "--format",
            "json",
            "--config-inline",
            inline_str,
        ])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)
    inline_section = payload["config"]["inline"]

    assert exit_code == 0
    assert captured.err == ""
    assert inline_section == {
        "sources": ["CLI[0]"],
        "entries": [
            {
                "source": "CLI[0]",
                "values": {
                    "sentinels": ["pyproject.toml", "inline.cfg"],
                    "default_profiles": ["demo"],
                    "profiles": {
                        "demo": {
                            "extends": [],
                            "values": {"path_style": "posix"},
                        }
                    },
                },
            }
        ],
    }
    assert payload["config"]["values"]["sentinels"] == [
        "pyproject.toml",
        "inline.cfg",
    ]
    assert payload["config"]["profiles"]["defined"] == ["demo"]
    assert payload["config"]["profiles"]["default"] == ["demo"]
    assert payload["config"]["profiles"]["definitions"]["demo"]["values"] == {
        "path_style": "posix"
    }


def test_main_config_inline_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_payload = {
        "allow_git": True,
        "profiles": {
            "env": {
                "position": "append",
            }
        },
    }
    monkeypatch.setenv("PATHBOOTSTRAP_CONFIG_INLINE", json.dumps(env_payload))

    clear_cache()
    try:
        exit_code = main(["--print-config", "--format", "json"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)
    inline_section = payload["config"]["inline"]

    assert exit_code == 0
    assert captured.err == ""
    assert inline_section == {
        "sources": ["ENV:PATHBOOTSTRAP_CONFIG_INLINE"],
        "entries": [
            {
                "source": "ENV:PATHBOOTSTRAP_CONFIG_INLINE",
                "values": {
                    "allow_git": True,
                    "profiles": {
                        "env": {
                            "extends": [],
                            "values": {"position": "append"},
                        }
                    },
                },
            }
        ],
    }
    assert payload["allow_git"]["effective"] is True
    assert payload["config"]["values"]["allow_git"] is True
    assert payload["max_depth"]["effective"] is None
    assert payload["config"]["values"]["max_depth"] is None
    assert payload["config"]["profiles"]["defined"] == ["env"]
    assert payload["config"]["profiles"]["definitions"]["env"]["values"] == {
        "position": "append"
    }


def test_main_list_config_files_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_config = tmp_path / "env.toml"
    env_config.write_text("allow_git = true\n", encoding="utf-8")

    included = tmp_path / "included.toml"
    included.write_text('additional_paths = ["from_include"]\n', encoding="utf-8")

    root_config = tmp_path / "root.json"
    root_payload = {
        "includes": [included.name],
        "additional_paths": ["from_root"],
    }
    root_config.write_text(json.dumps(root_payload), encoding="utf-8")

    inline_snippet = json.dumps({"sentinels": ["inline.sentinel"], "allow_git": False})

    monkeypatch.setenv("PATHBOOTSTRAP_CONFIG", str(env_config))

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-config-files",
                "--config-file",
                str(root_config),
                "--config-inline",
                inline_snippet,
                "--path-style",
                "posix",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()
        monkeypatch.delenv("PATHBOOTSTRAP_CONFIG", raising=False)

    assert exit_code == 0
    assert captured.err == ""

    env_config_str = str(env_config)
    root_config_str = str(root_config)
    included_str = str(included.resolve())

    expected_lines = [
        "config_files_env:",
        f"  - {env_config_str}",
        "config_files_env_display:",
        f"  - {env_config_str}",
        "config_files_cli:",
        f"  - {root_config_str}",
        "config_files_cli_display:",
        f"  - {root_config_str}",
        "config_files_roots:",
        f"  - {env_config_str}",
        f"  - {root_config_str}",
        "config_files_roots_display:",
        f"  - {env_config_str}",
        f"  - {root_config_str}",
        "config_files_loaded:",
        f"  - {env_config_str}",
        f"  - {included_str}",
        f"  - {root_config_str}",
        "config_files_loaded_display:",
        f"  - {env_config_str}",
        f"  - {included_str}",
        f"  - {root_config_str}",
        "config_inline_sources:",
        "  - CLI[0]",
        "config_inline_definition_CLI[0]_keys: allow_git, sentinels",
        "config_include_edges:",
        f"  - {root_config_str} -> {included_str}",
        "config_include_edges_display:",
        f"  - {root_config_str} -> {included_str}",
        f"config_files_env_var: {pathbootstrap.ENV_CONFIG_FILES}",
    ]

    assert captured.out.strip().splitlines() == expected_lines


def test_main_list_config_files_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_config = tmp_path / "env.toml"
    env_config.write_text("sentinels = [\"env.sentinel\"]\n", encoding="utf-8")

    included = tmp_path / "included.toml"
    included.write_text('additional_paths = ["from_include"]\n', encoding="utf-8")

    root_config = tmp_path / "root.json"
    root_payload = {
        "includes": [included.name],
        "additional_paths": ["from_root"],
    }
    root_config.write_text(json.dumps(root_payload), encoding="utf-8")

    inline_snippet = json.dumps({"profiles": {"inline": {"pythonpath_var": "INLINE_VAR"}}})

    monkeypatch.setenv("PATHBOOTSTRAP_CONFIG", str(env_config))

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-config-files",
                "--format",
                "json",
                "--config-file",
                str(root_config),
                "--config-inline",
                inline_snippet,
                "--path-style",
                "posix",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()
        monkeypatch.delenv("PATHBOOTSTRAP_CONFIG", raising=False)

    assert exit_code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)

    env_config_str = str(env_config)
    root_config_str = str(root_config)
    included_str = str(included.resolve())

    assert payload == {
        "env": [env_config_str],
        "env_display": [env_config_str],
        "env_var": pathbootstrap.ENV_CONFIG_FILES,
        "cli": [root_config_str],
        "cli_display": [root_config_str],
        "roots": [env_config_str, root_config_str],
        "roots_display": [env_config_str, root_config_str],
        "loaded": [env_config_str, included_str, root_config_str],
        "loaded_display": [env_config_str, included_str, root_config_str],
        "inline_sources": ["CLI[0]"],
        "inline_definitions": [
            {
                "source": "CLI[0]",
                "definition": {"profiles": {"inline": {"extends": [], "values": {"pythonpath_var": "INLINE_VAR"}}}},
            }
        ],
        "include_edges": [
            {
                "parent": root_config_str,
                "child": included_str,
                "parent_display": root_config_str,
                "child_display": included_str,
            }
        ],
        "include_edges_display": [f"{root_config_str} -> {included_str}"],
    }


def test_main_list_config_files_conflicts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--list-config-files", "--print-config"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--list-config-files" in captured.err


def test_main_list_root_hints_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"root_hint": "config-root"}), encoding="utf-8")

    cli_root = tmp_path / "cli-root"
    env_root = tmp_path / "env-root"
    monkeypatch.setenv("PATHBOOTSTRAP_ROOT_HINT", str(env_root))

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-root-hints",
                "--config-file",
                str(config_file),
                "--root-hint",
                str(cli_root),
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()
        monkeypatch.delenv("PATHBOOTSTRAP_ROOT_HINT", raising=False)

    assert exit_code == 0
    assert captured.err == ""

    config_root_resolved = str((config_file.parent / "config-root").resolve())
    cli_root_resolved = str(cli_root.resolve())
    env_root_resolved = str(env_root.resolve())
    default_root_resolved = str(Path(pathbootstrap.__file__).resolve().parent)

    assert captured.out.strip().splitlines() == [
        "root_hint_effective_source: cli",
        f"root_hint_effective_raw: {cli_root_resolved}",
        f"root_hint_effective_path: {cli_root_resolved}",
        f"root_hint_effective_display: {cli_root_resolved}",
        f"root_hint_cli: {cli_root_resolved}",
        f"root_hint_cli_display: {cli_root_resolved}",
        f"root_hint_config: {config_root_resolved}",
        f"root_hint_config_display: {config_root_resolved}",
        f"root_hint_env_var: {pathbootstrap.ENV_ROOT_HINT}",
        f"root_hint_env: {env_root_resolved}",
        f"root_hint_env_display: {env_root_resolved}",
        f"root_hint_default: {default_root_resolved}",
        f"root_hint_default_display: {default_root_resolved}",
    ]


def test_main_list_root_hints_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"root_hint": "config-root"}), encoding="utf-8")

    env_root = tmp_path / "env-root"
    monkeypatch.setenv("PATHBOOTSTRAP_ROOT_HINT", str(env_root))

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-root-hints",
                "--format",
                "json",
                "--config-file",
                str(config_file),
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()
        monkeypatch.delenv("PATHBOOTSTRAP_ROOT_HINT", raising=False)

    assert exit_code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)

    config_root_resolved = str((config_file.parent / "config-root").resolve())
    env_root_resolved = str(env_root.resolve())
    default_root_resolved = str(Path(pathbootstrap.__file__).resolve().parent)

    assert payload == {
        "effective_source": "config",
        "effective_raw": config_root_resolved,
        "effective_path": config_root_resolved,
        "effective_display": config_root_resolved,
        "cli_raw": None,
        "cli_display": None,
        "config_raw": config_root_resolved,
        "config_display": config_root_resolved,
        "env_var": pathbootstrap.ENV_ROOT_HINT,
        "env_raw": str(env_root),
        "env_display": env_root_resolved,
        "default_raw": default_root_resolved,
        "default_display": default_root_resolved,
    }


def test_main_list_root_hints_conflicts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--list-root-hints", "--list-sentinels"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--list-root-hints" in captured.err


def test_main_list_sentinels_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "sentinels.json"
    config_file.write_text(json.dumps({"sentinels": ["config.sentinel"]}), encoding="utf-8")
    sentinel_file = tmp_path / "sentinels.txt"
    sentinel_file.write_text("file.sentinel\n# comment\n\n", encoding="utf-8")
    monkeypatch.setenv("PATHBOOTSTRAP_SENTINELS", os.pathsep.join(["env.one", "env.two"]))

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-sentinels",
                "--config-file",
                str(config_file),
                "--sentinel-file",
                str(sentinel_file),
                "--sentinel",
                "cli.alpha",
                "--sentinel",
                "cli.beta",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == [
        "sentinels_effective:",
        "  - env.one",
        "  - env.two",
        "sentinels_candidates:",
        "  - config.sentinel",
        "  - file.sentinel",
        "  - cli.alpha",
        "  - cli.beta",
        "sentinels_default:",
        "  - pyproject.toml",
        "sentinels_config:",
        "  - config.sentinel",
        "sentinels_env:",
        "  - env.one",
        "  - env.two",
        "sentinels_cli:",
        "  - cli.alpha",
        "  - cli.beta",
        "sentinels_file:",
        "  - file.sentinel",
        "sentinel_file_cli:",
        f"  - {sentinel_file}",
    ]


def test_main_list_sentinels_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "sentinels.json"
    config_file.write_text(json.dumps({"sentinels": ["config.sentinel"]}), encoding="utf-8")
    monkeypatch.delenv("PATHBOOTSTRAP_SENTINELS", raising=False)

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-sentinels",
                "--format",
                "json",
                "--config-file",
                str(config_file),
                "--sentinel",
                "cli.alpha",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)

    assert payload == {
        "effective": ["config.sentinel", "cli.alpha"],
        "candidates": ["config.sentinel", "cli.alpha"],
        "default": list(pathbootstrap.DEFAULT_SENTINELS),
        "config": ["config.sentinel"],
        "env": [],
        "cli": ["cli.alpha"],
        "file": [],
        "sentinel_file": None,
    }


def test_main_list_sentinels_conflicts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--list-sentinels", "--print-config"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--list-sentinels" in captured.err


def test_main_list_add_paths_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_paths_file = tmp_path / "config_paths.txt"
    config_paths_file.write_text("config.from_file\n", encoding="utf-8")
    config_file = tmp_path / "paths.json"
    config_payload = {
        "additional_paths": ["config.direct"],
        "additional_path_files": [str(config_paths_file)],
    }
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")

    cli_paths_file = tmp_path / "cli_paths.txt"
    cli_paths_file.write_text("cli.from_file\n", encoding="utf-8")

    monkeypatch.setenv(
        "PATHBOOTSTRAP_ADD_PATHS", os.pathsep.join(["env.one", "env.two"])
    )

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-add-paths",
                "--config-file",
                str(config_file),
                "--add-path-file",
                str(cli_paths_file),
                "--add-path",
                "cli.direct",
                "--path-style",
                "posix",
            ]
        )
        captured = capsys.readouterr()
        repo_root = pathbootstrap.get_repo_root()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""

    repo_root = Path(repo_root)
    expected_effective = [
        (repo_root / "env.one").resolve(strict=False).as_posix(),
        (repo_root / "env.two").resolve(strict=False).as_posix(),
        (repo_root / "config.from_file").resolve(strict=False).as_posix(),
        (repo_root / "config.direct").resolve(strict=False).as_posix(),
        (repo_root / "cli.from_file").resolve(strict=False).as_posix(),
        (repo_root / "cli.direct").resolve(strict=False).as_posix(),
    ]

    assert captured.out.strip().splitlines() == [
        "additional_paths_effective:",
        *[f"  - {entry}" for entry in expected_effective],
        "additional_paths_config:",
        "  - config.direct",
        "additional_paths_config_file_entries:",
        "  - config.from_file",
        "additional_paths_config_files:",
        f"  - {config_paths_file.resolve().as_posix()}",
        "additional_paths_env_included: true",
        "additional_paths_env:",
        "  - env.one",
        "  - env.two",
        "additional_paths_env_var: PATHBOOTSTRAP_ADD_PATHS",
        "additional_paths_cli:",
        "  - cli.direct",
        "additional_paths_cli_file_entries:",
        "  - cli.from_file",
        "additional_paths_cli_files:",
        f"  - {cli_paths_file.resolve().as_posix()}",
    ]


def test_main_list_add_paths_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_paths_file = tmp_path / "config_paths.txt"
    config_paths_file.write_text("config.from_file\n", encoding="utf-8")
    config_file = tmp_path / "paths.json"
    config_payload = {
        "additional_paths": ["config.direct"],
        "additional_path_files": [str(config_paths_file)],
    }
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")

    cli_paths_file = tmp_path / "cli_paths.txt"
    cli_paths_file.write_text("cli.from_file\n", encoding="utf-8")

    monkeypatch.setenv("PATHBOOTSTRAP_ADD_PATHS", "env.ignored")

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-add-paths",
                "--format",
                "json",
                "--no-env-add-paths",
                "--config-file",
                str(config_file),
                "--add-path-file",
                str(cli_paths_file),
                "--add-path",
                "cli.direct",
            ]
        )
        captured = capsys.readouterr()
        repo_root = pathbootstrap.get_repo_root()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""

    repo_root = Path(repo_root)
    payload = json.loads(captured.out)

    expected_effective = [
        str((repo_root / "config.from_file").resolve(strict=False)),
        str((repo_root / "config.direct").resolve(strict=False)),
        str((repo_root / "cli.from_file").resolve(strict=False)),
        str((repo_root / "cli.direct").resolve(strict=False)),
    ]

    expected_file_path_config = str(config_paths_file.resolve())
    expected_file_path_cli = str(cli_paths_file.resolve())

    assert payload == {
        "effective": expected_effective,
        "effective_display": expected_effective,
        "config": ["config.direct"],
        "config_display": ["config.direct"],
        "config_file_entries": ["config.from_file"],
        "config_file_entries_display": ["config.from_file"],
        "config_files": [expected_file_path_config],
        "config_files_display": [expected_file_path_config],
        "env_included": False,
        "env": [],
        "env_display": [],
        "env_var": "PATHBOOTSTRAP_ADD_PATHS",
        "cli": ["cli.direct"],
        "cli_display": ["cli.direct"],
        "cli_file_entries": ["cli.from_file"],
        "cli_file_entries_display": ["cli.from_file"],
        "cli_files": [expected_file_path_cli],
        "cli_files_display": [expected_file_path_cli],
    }


def test_main_list_add_paths_conflicts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--list-add-paths", "--print-config"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--list-add-paths" in captured.err

def test_main_list_profiles_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PATHBOOTSTRAP_PROFILES", "child")
    config_file = tmp_path / "profiles.json"
    config_payload = {
        "profiles": {
            "base": {"allow_git": True, "position": "append"},
            "child": {"extends": ["base"], "pythonpath_var": "CHILD_VAR"},
            "grandchild": {"extends": ["child"], "path_style": "posix"},
        },
        "default_profiles": ["base"],
    }
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(
            [
                "--list-profiles",
                "--config-file",
                str(config_file),
                "--profile",
                "grandchild",
                "--profile",
                "!base",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == [
        "profiles_defined:",
        "  - base (extends: (brak); defined_keys: allow_git, position; resolved_keys: allow_git, position)",
        "  - child (extends: base; defined_keys: pythonpath_var; resolved_keys: allow_git, position, pythonpath_var)",
        "  - grandchild (extends: child; defined_keys: path_style; resolved_keys: allow_git, path_style, position, pythonpath_var)",
        "default_profiles:",
        "  - base",
        "env_profiles_added:",
        "  - child",
        "env_profiles_removed: (brak)",
        "cli_profiles_added:",
        "  - grandchild",
        "cli_profiles_removed:",
        "  - base",
        "selected_profiles:",
        "  - child",
        "  - grandchild",
    ]


def test_main_list_profiles_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_PROFILES", raising=False)
    config_file = tmp_path / "profiles.json"
    config_payload = {
        "profiles": {
            "alpha": {"allow_git": True},
            "beta": {"extends": ["alpha"], "position": "append"},
        },
        "default_profiles": ["alpha"],
    }
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(
            ["--list-profiles", "--format", "json", "--config-file", str(config_file)]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""

    payload = json.loads(captured.out)

    assert payload == {
        "profiles": [
            {
                "name": "alpha",
                "extends": [],
                "defined_keys": ["allow_git"],
                "resolved_keys": ["allow_git"],
            },
            {
                "name": "beta",
                "extends": ["alpha"],
                "defined_keys": ["position"],
                "resolved_keys": ["allow_git", "position"],
            },
        ],
        "default": ["alpha"],
        "selected": ["alpha"],
        "env_added": [],
        "env_removed": [],
        "cli_added": [],
        "cli_removed": [],
    }


def test_main_list_profiles_conflicts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--list-profiles", "--print-config"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--list-profiles" in captured.err


def test_main_config_file_applies_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    monkeypatch.delenv("PATHBOOTSTRAP_SENTINELS", raising=False)
    monkeypatch.delenv("PATHBOOTSTRAP_CONFIG", raising=False)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path_file = config_dir / "paths.txt"
    path_file.write_text("cfg_lib\n", encoding="utf-8")
    config_payload = {
        "additional_paths": ["extras/src"],
        "additional_path_files": ["paths.txt"],
        "position": "append",
        "pythonpath_var": "MYVAR",
        "path_style": "posix",
        "use_env_additional_paths": False,
        "allow_git": True,
    }
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")

    clear_cache()
    try:
        exit_code = main(
            ["--print-config", "--format", "json", "--config-file", str(config_file)]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads(captured.out)
    normalized_cfg_lib = str((repo_root / "cfg_lib").resolve())
    normalized_extra = str((repo_root / "extras/src").resolve())

    assert exit_code == 0
    assert captured.err == ""
    assert payload["position"] == "append"
    assert payload["pythonpath_var"] == "MYVAR"
    assert payload["include_env_additional_paths"] is False
    assert payload["allow_git"]["effective"] is True
    assert payload["max_depth"]["effective"] is None
    assert payload["path_style"] == "posix"
    assert payload["additional_paths"]["config"] == ["extras/src"]
    assert payload["additional_paths"]["config_files"] == [str(path_file.resolve())]
    assert payload["additional_paths"]["normalized"] == [normalized_cfg_lib, normalized_extra]
    assert payload["config"]["values"]["max_depth"] is None
    assert payload["config"]["files"] == [str(config_file)]
    assert payload["config"]["includes"] == []
    assert payload["config"]["values"]["pythonpath_var"] == "MYVAR"
    assert payload["config"]["values"]["position"] == "append"
    assert payload["config"]["values"]["path_style"] == "posix"
    assert payload["config"]["values"]["use_env_additional_paths"] is False
    assert payload["config"]["values"]["allow_git"] is True


def test_main_config_file_accepts_tilde_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    config_path = home_dir / "bootstrap.toml"
    config_path.write_text('additional_paths = ["tests"]\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    clear_cache()
    try:
        exit_code = main(["--format", "json", "--config-file", "~/bootstrap.toml"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]
    expected = str((repo_root / "tests").resolve())
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["repo_root"] == str(repo_root)
    assert payload["additional_paths"] == [expected]


def test_main_config_env_and_cli_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_config = tmp_path / "env.json"
    env_config.write_text(
        json.dumps({"use_env_additional_paths": False, "pythonpath_var": "ENVVAR"}),
        encoding="utf-8",
    )
    cli_config = tmp_path / "cli.json"
    cli_config.write_text(
        json.dumps({"use_env_additional_paths": True, "pythonpath_var": "CLIVAR"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PATHBOOTSTRAP_CONFIG", str(env_config))

    clear_cache()
    try:
        exit_code = main(
            ["--print-config", "--format", "json", "--config-file", str(cli_config)]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["include_env_additional_paths"] is True
    assert payload["pythonpath_var"] == "CLIVAR"
    assert payload["config"]["files"] == [str(env_config), str(cli_config)]
    assert payload["config"]["includes"] == []
    assert payload["config"]["values"]["pythonpath_var"] == "CLIVAR"
    assert payload["config"]["values"]["use_env_additional_paths"] is True


def test_main_config_supports_includes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_CONFIG", raising=False)
    base = tmp_path / "base.toml"
    base.write_text(
        "additional_paths = [\"base/lib\"]\nuse_env_additional_paths = false\n",
        encoding="utf-8",
    )
    main_file = tmp_path / "main.toml"
    main_file.write_text(
        "includes = [\"base.toml\"]\n"
        "additional_paths = [\"main/lib\"]\n"
        "use_env_additional_paths = true\n",
        encoding="utf-8",
    )

    clear_cache()
    try:
        exit_code = main(
            ["--print-config", "--format", "json", "--config-file", str(main_file)]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)
    base_resolved = str(base.resolve())
    main_resolved = str(main_file.resolve())

    assert exit_code == 0
    assert payload["config"]["files"] == [base_resolved, main_resolved]
    assert payload["config"]["includes"] == [
        {"parent": main_resolved, "child": base_resolved}
    ]
    assert payload["config"]["values"]["additional_paths"] == ["main/lib"]
    assert payload["config"]["values"]["use_env_additional_paths"] is True


def test_main_max_depth_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"max_depth": 3}), encoding="utf-8")
    monkeypatch.setenv("PATHBOOTSTRAP_MAX_DEPTH", "2")

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-config",
                "--format",
                "json",
                "--config-file",
                str(config_file),
                "--max-depth",
                "4",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()
        monkeypatch.delenv("PATHBOOTSTRAP_MAX_DEPTH", raising=False)

    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["max_depth"] == {
        "effective": 4,
        "cli": 4,
        "config": 3,
        "env": 2,
        "env_raw": "2",
    }
    assert payload["config"]["values"]["max_depth"] == 3


def test_main_config_include_cycle_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_a = tmp_path / "a.json"
    config_b = tmp_path / "b.json"
    config_a.write_text(json.dumps({"includes": ["b.json"]}), encoding="utf-8")
    config_b.write_text(json.dumps({"includes": ["a.json"]}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(config_a)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "cykliczne dołączanie konfiguracji" in captured.err


def test_main_config_profiles_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_CONFIG", raising=False)
    config_file = tmp_path / "profiles.json"
    config_payload = {
        "additional_paths": ["base/lib"],
        "profiles": {
            "ci": {"allow_git": True},
            "qa": {"pythonpath_var": "QA_PATH"},
            "ops": {"position": "append"},
        },
        "default_profiles": ["ci"],
    }
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("PATHBOOTSTRAP_PROFILES", "qa")

    clear_cache()
    try:
        exit_code = main(
            ["--print-config", "--format", "json", "--config-file", str(config_file), "--profile", "ops"]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)
    repo_root = Path(__file__).resolve().parents[1]

    assert exit_code == 0
    assert payload["config"]["profiles"]["defined"] == ["ci", "qa", "ops"]
    assert payload["config"]["profiles"]["default"] == ["ci"]
    assert payload["config"]["profiles"]["env"] == ["qa"]
    assert payload["config"]["profiles"]["env_removed"] == []
    assert payload["config"]["profiles"]["cli"] == ["ops"]
    assert payload["config"]["profiles"]["cli_removed"] == []
    assert payload["config"]["profiles"]["selected"] == ["ci", "qa", "ops"]
    assert payload["config"]["values"]["allow_git"] is True
    assert payload["config"]["values"]["pythonpath_var"] == "QA_PATH"
    assert payload["config"]["values"]["position"] == "append"
    assert payload["config"]["values"]["additional_paths"] == ["base/lib"]
    assert payload["max_depth"]["effective"] is None
    assert payload["config"]["values"].get("max_depth") is None
    definitions = payload["config"]["profiles"]["definitions"]
    assert definitions["ci"] == {
        "extends": [],
        "values": {"allow_git": True},
        "resolved": {"allow_git": True},
    }
    assert definitions["qa"] == {
        "extends": [],
        "values": {"pythonpath_var": "QA_PATH"},
        "resolved": {"pythonpath_var": "QA_PATH"},
    }
    assert definitions["ops"] == {
        "extends": [],
        "values": {"position": "append"},
        "resolved": {"position": "append"},
    }
    assert payload["discovery"] == {
        "method": "sentinel",
        "sentinel": "pyproject.toml",
        "depth": 0,
        "start": str(repo_root),
        "start_display": str(repo_root),
    }


def test_main_config_profiles_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_CONFIG", raising=False)
    config_file = tmp_path / "profiles.json"
    config_payload = {
        "profiles": {
            "ci": {"allow_git": True},
            "qa": {"pythonpath_var": "QA_PATH"},
            "ops": {"position": "append"},
        },
        "default_profiles": ["ci"],
    }
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("PATHBOOTSTRAP_PROFILES", os.pathsep.join(["!ci", "qa"]))

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-config",
                "--format",
                "json",
                "--config-file",
                str(config_file),
                "--profile",
                "!qa",
                "--profile",
                "ops",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["config"]["profiles"]["default"] == ["ci"]
    assert payload["config"]["profiles"]["env"] == ["qa"]
    assert payload["config"]["profiles"]["env_removed"] == ["ci"]
    assert payload["config"]["profiles"]["cli"] == ["ops"]
    assert payload["config"]["profiles"]["cli_removed"] == ["qa"]
    assert payload["config"]["profiles"]["selected"] == ["ops"]


def test_main_config_profiles_unknown_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "empty.json"
    config_file.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(config_file), "--profile", "missing"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "opcja --profile odwołuje się do nieistniejących profili" in captured.err


def test_main_config_profiles_env_unknown_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "empty.json"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PATHBOOTSTRAP_PROFILES", "missing")

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(config_file)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert (
        "zmienna środowiskowa PATHBOOTSTRAP_PROFILES odwołuje się do nieistniejących profili"
        in captured.err
    )


def test_main_config_profiles_include_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_CONFIG", raising=False)
    included = tmp_path / "base.json"
    included.write_text(json.dumps({"profiles": {"alpha": {"allow_git": True}}}), encoding="utf-8")
    main_config = tmp_path / "main.json"
    main_config.write_text(
        json.dumps(
            {
                "includes": ["base.json"],
                "profiles": {"beta": {"position": "append"}},
                "default_profiles": ["alpha"],
            }
        ),
        encoding="utf-8",
    )

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-config",
                "--format",
                "json",
                "--config-file",
                str(main_config),
                "--profile",
                "beta",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["config"]["profiles"]["defined"] == ["alpha", "beta"]
    assert payload["config"]["profiles"]["default"] == ["alpha"]
    assert payload["config"]["profiles"]["env_removed"] == []
    assert payload["config"]["profiles"]["cli"] == ["beta"]
    assert payload["config"]["profiles"]["cli_removed"] == []
    assert payload["config"]["profiles"]["selected"] == ["alpha", "beta"]
    definitions = payload["config"]["profiles"]["definitions"]
    assert definitions["alpha"] == {
        "extends": [],
        "values": {"allow_git": True},
        "resolved": {"allow_git": True},
    }
    assert definitions["beta"] == {
        "extends": [],
        "values": {"position": "append"},
        "resolved": {"position": "append"},
    }
    assert payload["config"]["values"]["allow_git"] is True
    assert payload["config"]["values"]["position"] == "append"


def test_main_config_profiles_extends_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_PROFILES", raising=False)
    config_file = tmp_path / "profiles_extends.json"
    config_payload = {
        "profiles": {
            "base": {"allow_git": True, "position": "append"},
            "child": {"extends": ["base"], "pythonpath_var": "CHILD_VAR"},
            "grandchild": {"extends": ["child"], "path_style": "posix"},
        },
        "default_profiles": ["base"],
    }
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setenv("PATHBOOTSTRAP_PROFILES", "child")

    clear_cache()
    try:
        exit_code = main(
            [
                "--print-config",
                "--format",
                "json",
                "--config-file",
                str(config_file),
                "--profile",
                "grandchild",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["config"]["profiles"]["env"] == ["child"]
    assert payload["config"]["profiles"]["env_removed"] == []
    assert payload["config"]["profiles"]["cli"] == ["grandchild"]
    assert payload["config"]["profiles"]["cli_removed"] == []
    assert payload["config"]["profiles"]["selected"] == [
        "base",
        "child",
        "grandchild",
    ]
    assert payload["config"]["values"]["allow_git"] is True
    assert payload["config"]["values"]["position"] == "append"
    assert payload["config"]["values"]["pythonpath_var"] == "CHILD_VAR"
    assert payload["config"]["values"]["path_style"] == "posix"

    definitions = payload["config"]["profiles"]["definitions"]
    assert definitions["base"] == {
        "extends": [],
        "values": {"allow_git": True, "position": "append"},
        "resolved": {"allow_git": True, "position": "append"},
    }
    assert definitions["child"] == {
        "extends": ["base"],
        "values": {"pythonpath_var": "CHILD_VAR"},
        "resolved": {
            "allow_git": True,
            "position": "append",
            "pythonpath_var": "CHILD_VAR",
        },
    }
    assert definitions["grandchild"] == {
        "extends": ["child"],
        "values": {"path_style": "posix"},
        "resolved": {
            "allow_git": True,
            "position": "append",
            "pythonpath_var": "CHILD_VAR",
            "path_style": "posix",
        },
    }


def test_main_config_profiles_extends_cycle_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "cycle.json"
    config_file.write_text(
        json.dumps(
            {
                "profiles": {
                    "first": {"extends": ["second"]},
                    "second": {"extends": ["first"]},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(config_file)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "cykliczne dziedziczenie profili" in captured.err


def test_main_config_profiles_extends_missing_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "missing_base.json"
    config_file.write_text(
        json.dumps({"profiles": {"child": {"extends": ["missing"]}}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(config_file)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "odwołują się do nieistniejących baz" in captured.err


def test_main_config_profiles_missing_default_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "invalid.json"
    config_file.write_text(json.dumps({"default_profiles": ["missing"]}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(config_file)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "konfiguracja odwołuje się do nieistniejących profili" in captured.err


def test_main_config_missing_file_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(missing)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "plik konfiguracji" in captured.err


def test_main_config_invalid_key_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = tmp_path / "bad.json"
    config_file.write_text(json.dumps({"unknown": 1}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--print-config", "--config-file", str(config_file)])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "nieobsługiwane opcje" in captured.err


def test_main_print_config_windows_style_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--print-config", "--format", "json", "--path-style", "windows"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    payload = json.loads(captured.out)
    repo_root = Path(__file__).resolve().parents[1]
    expected_repo = str(repo_root).replace("/", "\\")

    assert exit_code == 0
    assert captured.err == ""
    assert payload["repo_root"] == expected_repo
    assert payload["additional_paths"]["normalized"] == []
    assert payload["path_style"] == "windows"
    assert payload["allow_git"]["effective"] is False
    assert payload["max_depth"]["effective"] is None


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


def test_main_writes_output_to_file_with_tilde(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    destination = home_dir / "repo.txt"
    monkeypatch.setenv("HOME", str(home_dir))

    clear_cache()
    try:
        exit_code = main(["--output", "~/repo.txt"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]

    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert destination.read_text(encoding="utf-8") == f"{repo_root}\n"


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


def test_main_with_set_env_windows_style(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_repo = str(repo_root).replace("/", "\\")

    clear_cache()
    try:
        exit_code = main(["--set-env", "REPO_ROOT", "--path-style", "windows"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"REPO_ROOT={expected_repo}"


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


def test_main_supports_allow_git_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _skip_if_git_missing()
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()
    _init_git_repository(repo_root)
    nested = repo_root / "pkg"
    nested.mkdir()

    clear_cache()
    try:
        exit_code = main(
            [
                "--root-hint",
                str(nested),
                "--sentinel",
                "missing.marker",
                "--allow-git",
            ]
        )
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == str(repo_root.resolve())


def test_main_no_allow_git_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _skip_if_git_missing()
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()
    _init_git_repository(repo_root)
    nested = repo_root / "pkg"
    nested.mkdir()

    monkeypatch.setenv("PATHBOOTSTRAP_ALLOW_GIT", "1")

    clear_cache()
    try:
        with pytest.raises(FileNotFoundError) as excinfo:
            main(
                [
                    "--root-hint",
                    str(nested),
                    "--sentinel",
                    "missing.marker",
                    "--no-allow-git",
                ]
            )
    finally:
        clear_cache()
        monkeypatch.delenv("PATHBOOTSTRAP_ALLOW_GIT", raising=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing.marker" in str(excinfo.value)


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


def test_main_module_runs_python_module(capfd: pytest.CaptureFixture[str]) -> None:
    clear_cache()
    try:
        exit_code = main(["--module", "pathbootstrap"])
    finally:
        clear_cache()

    captured = capfd.readouterr()
    repo_root = Path(__file__).resolve().parents[1]
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == str(repo_root)


def test_main_requires_shell_flag_for_shell_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--shell-path", "/bin/zsh"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--shell-path" in captured.err


def test_main_shell_rejects_module(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--shell", "--module", "pathbootstrap"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--shell" in captured.err


def test_main_shell_runs_default_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        recorded["command"] = command
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("SHELL", "/bin/customsh")
    monkeypatch.setattr(pathbootstrap.subprocess, "run", fake_run)

    clear_cache()
    try:
        exit_code = main(["--shell"])
    finally:
        clear_cache()
        monkeypatch.delenv("SHELL", raising=False)

    assert exit_code == 0
    assert recorded["command"] == ["/bin/customsh"]
    pythonpath_value = recorded["env"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath_value[0] == str(repo_root)


def test_main_shell_runs_command_via_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        recorded["command"] = command
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("SHELL", "/bin/othersh")
    monkeypatch.setattr(pathbootstrap.subprocess, "run", fake_run)

    clear_cache()
    try:
        exit_code = main(["--shell", "--", "echo", "hello"])
    finally:
        clear_cache()
        monkeypatch.delenv("SHELL", raising=False)

    assert exit_code == 0
    assert recorded["command"][0] == "/bin/othersh"
    assert recorded["command"][1:] == ["-c", "echo hello"]


def test_main_shell_uses_custom_shell_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        recorded["command"] = command
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pathbootstrap.subprocess, "run", fake_run)

    clear_cache()
    try:
        exit_code = main(["--shell", "--shell-path", "/bin/customshell"])
    finally:
        clear_cache()

    assert exit_code == 0
    assert recorded["command"] == ["/bin/customshell"]


def test_main_module_uses_custom_python_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        recorded["command"] = command
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("PYTHONPATH", "/tmp/base")
    monkeypatch.setattr(pathbootstrap.subprocess, "run", fake_run)

    clear_cache()
    try:
        exit_code = main(
            [
                "--module",
                "pathbootstrap",
                "--python-executable",
                "/custom/python",
                "--",
                "--print-config",
            ]
        )
    finally:
        clear_cache()

    assert exit_code == 0
    assert recorded["command"] == [
        "/custom/python",
        "-m",
        "pathbootstrap",
        "--print-config",
    ]
    env_value = recorded["env"]["PYTHONPATH"].split(os.pathsep)
    assert env_value[0] == str(repo_root)
    assert env_value[-1] == "/tmp/base"


def test_main_print_config_json_with_indent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clear_cache()
    try:
        exit_code = main(["--print-config", "--format", "json", "--json-indent", "2"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    repo_root = Path(__file__).resolve().parents[1]

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.startswith("{\n  ")
    payload = json.loads(captured.out)
    assert payload["repo_root"] == str(repo_root)


def test_main_json_indent_requires_json_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--json-indent", "2"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "--format=json" in captured.err


def test_main_json_indent_rejects_negative(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--format", "json", "--json-indent", "-1"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert "nieujemna" in captured.err


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
