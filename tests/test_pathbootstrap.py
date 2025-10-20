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


def test_main_with_ensure_adds_repo_root(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    cleaned_path = [entry for entry in sys.path if entry != repo_str]
    monkeypatch.setattr(sys, "path", list(cleaned_path), raising=False)

    clear_cache()
    try:
        exit_code = main(["--ensure"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == repo_str


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


def test_main_with_position_append_places_repo_at_end(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    cleaned_path = [entry for entry in sys.path if entry != repo_str]
    monkeypatch.setattr(sys, "path", [*cleaned_path, "/tmp/gamma"], raising=False)

    clear_cache()
    try:
        exit_code = main(["--ensure", "--position", "append"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == repo_str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_output(
    capsys: pytest.CaptureFixture[str],
) -> tuple[str, str]:
    captured = capsys.readouterr()
    return captured.out, captured.err


def test_main_prints_json_with_additional_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--format", "json", "--add-path", "tests"])
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    repo_root = _repo_root()
    expected = str((repo_root / "tests").resolve())
    payload = json.loads(out)

    assert exit_code == 0
    assert err == ""
    assert payload == {
        "repo_root": str(repo_root),
        "additional_paths": [expected],
    }


def test_main_with_add_path_includes_additional_entries(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _repo_root()
    repo_str = str(repo_root)
    cleaned_path = [entry for entry in sys.path if entry != repo_str]
    monkeypatch.setattr(sys, "path", [*cleaned_path, "/tmp/gamma"], raising=False)

    clear_cache()
    try:
        exit_code = main(["--ensure", "--add-path", "tests"])
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    assert exit_code == 0
    assert err == ""
    assert out.strip() == repo_str


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
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    repo_root = _repo_root()
    expected_repo = str(repo_root).replace("/", "\\")
    expected_tests = str((repo_root / "tests").resolve()).replace("/", "\\")
    payload = json.loads(out)

    assert exit_code == 0
    assert err == ""
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
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    repo_root = _repo_root()
    payload = json.loads(out)
    expected_tests = str((repo_root / "tests").resolve())
    expected_docs = str((repo_root / "docs").resolve())

    assert exit_code == 0
    assert err == ""
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
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    repo_root = _repo_root()
    expected_data = str((repo_root / "data").resolve())
    expected_docs = str((repo_root / "docs").resolve())
    expected_tests = str((repo_root / "tests").resolve())
    payload = json.loads(out)

    assert exit_code == 0
    assert err == ""
    assert payload["additional_paths"] == [
        expected_data,
        expected_docs,
        expected_tests,
    ]


def test_main_with_set_env_prints_assignment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = _repo_root()

    clear_cache()
    try:
        exit_code = main(["--set-env", "REPO_ROOT"])
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    assert exit_code == 0
    assert err == ""
    assert out.strip() == f"REPO_ROOT={repo_root}"


def test_main_with_set_env_prints_assignment(capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    clear_cache()
    try:
        exit_code = main(["--set-env", "REPO_ROOT"])
        captured = capsys.readouterr()
    finally:
        clear_cache()

    expected_repo = str(repo_root)

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"REPO_ROOT={expected_repo}"


def test_main_prints_pythonpath_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PATHBOOTSTRAP_ADD_PATHS", raising=False)

    clear_cache()
    try:
        exit_code = main(["--print-pythonpath", "--add-path", "tests"])
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    repo_root = _repo_root()
    expected_tests = str((repo_root / "tests").resolve())

    assert exit_code == 0
    assert err == ""
    assert out.strip() == ":".join([str(repo_root), expected_tests])


def test_main_with_set_env_prints_assignment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = _repo_root()

    clear_cache()
    try:
        exit_code = main(
            [
                "--export",
                "--set-env",
                "REPO_ROOT",
                "--set-env-format",
                "posix",
            ]
        )
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    assert exit_code == 0
    assert err == ""
    assert out.strip() == f"export REPO_ROOT={repo_root}"


def test_main_prints_pythonpath_value(
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

    repo_root = _repo_root()
    expected_tests = str((repo_root / "tests").resolve())

    assert exit_code == 0
    assert err == ""
    assert out.strip() == ":".join([str(repo_root), expected_tests])


def test_main_with_set_env_and_export_prints_export_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = _repo_root()

    clear_cache()
    try:
        exit_code = main(
            [
                "--export",
                "--set-env",
                "REPO_ROOT",
                "--set-env-format",
                "posix",
            ]
        )
        out, err = _read_output(capsys)
    finally:
        clear_cache()

    expected_repo = str(repo_root)

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip() == f"export REPO_ROOT={expected_repo}"


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
    expected_repo = str(repo_root)
    expected_tests = str((repo_root / "tests").resolve())
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert err == ""
    assert payload == {
        "repo_root": expected_repo,
        "additional_paths": [expected_tests],
        "pythonpath": os.pathsep.join([expected_repo, expected_tests]),
        "pythonpath_entries": [expected_repo, expected_tests],
    }
