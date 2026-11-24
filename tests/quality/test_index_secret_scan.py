from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.quality import betterleaks_hook

MARKER = b"synthetic-secret-for-index-regression"


def _git(repository: Path, *args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repository, input=data, capture_output=True, check=True
    ).stdout


@pytest.fixture
def repository(tmp_path: Path, monkeypatch) -> Path:
    for name in (
        "BETTERLEAKS_CONFIG",
        "GITLEAKS_CONFIG",
        "BETTERLEAKS_CONFIG_TOML",
        "GITLEAKS_CONFIG_TOML",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "repo with spaces"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "fixture.txt").write_bytes(b"fixture")
    _git(path, "add", "fixture.txt")
    _git(path, "commit", "-qm", "fixture")
    monkeypatch.chdir(path)
    return path


def _scanner(monkeypatch) -> list[tuple[str, bytes]]:
    captured: list[tuple[str, bytes]] = []
    real_run = subprocess.run
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")

    def run(command: list[str], **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        root = Path(kwargs["cwd"])
        targets = command[command.index("--") + 1 :]
        contents = [(target, (root / target).read_bytes()) for target in targets]
        captured.extend(contents)
        return subprocess.CompletedProcess(
            command, int(any(MARKER in data for _, data in contents))
        )

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", run)
    return captured


def test_index_bytes_are_scanned_when_working_copy_differs(repository: Path, monkeypatch) -> None:
    candidate = repository / "source with ünicode.txt"
    candidate.write_bytes(MARKER)
    _git(repository, "add", candidate.name)
    candidate.write_bytes(b"clean working copy")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 1
    assert captured == [(candidate.name, MARKER)]


def test_missing_working_copy_does_not_hide_index_blob(repository: Path, monkeypatch) -> None:
    candidate = repository / "removed-from-working-tree.txt"
    candidate.write_bytes(MARKER)
    _git(repository, "add", candidate.name)
    candidate.unlink()
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 1
    assert captured == [(candidate.name, MARKER)]


@pytest.mark.parametrize("mode", ["100644", "120000"])
def test_index_type_change_is_scanned(repository: Path, monkeypatch, mode: str) -> None:
    candidate = repository / "changed-type.txt"
    previous_mode = "120000" if mode == "100644" else "100644"
    previous = _git(repository, "hash-object", "-w", "--stdin", data=b"benign-link-target").strip()
    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        previous_mode,
        previous.decode(),
        candidate.name,
    )
    _git(repository, "commit", "-qm", "previous type")
    staged = _git(repository, "hash-object", "-w", "--stdin", data=MARKER).strip()
    _git(repository, "update-index", "--cacheinfo", mode, staged.decode(), candidate.name)
    candidate.write_bytes(b"clean working copy")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 1
    assert captured == [(candidate.name, MARKER)]


def test_unmerged_index_refuses_scan_success(repository: Path, monkeypatch) -> None:
    blob = _git(repository, "hash-object", "-w", "--stdin", data=b"fixture").strip()
    _git(
        repository,
        "update-index",
        "--index-info",
        data=b"".join(
            b"100644 " + blob + b" " + str(stage).encode() + b"\tconflict.txt\n"
            for stage in (1, 2, 3)
        ),
    )
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 2
    assert captured == []


@pytest.mark.parametrize("payload", [b"binary\0fixture", b"non-UTF-8\xfffixture"])
def test_binary_or_encoded_blob_cannot_silently_pass(
    repository: Path, monkeypatch, payload: bytes
) -> None:
    candidate = repository / "binary.dat"
    candidate.write_bytes(payload)
    _git(repository, "add", candidate.name)
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 2
    assert captured == []


@pytest.mark.parametrize("limit", ["_MAX_BLOB_BYTES", "_MAX_SNAPSHOT_BYTES"])
def test_oversized_scan_refuses_instead_of_omitting_files(
    repository: Path, monkeypatch, limit: str
) -> None:
    candidate = repository / "large.txt"
    candidate.write_bytes(b"x" * 20)
    _git(repository, "add", candidate.name)
    captured = _scanner(monkeypatch)
    monkeypatch.setattr(betterleaks_hook, limit, 10)

    assert betterleaks_hook.main([]) == 2
    assert captured == []


def test_rename_scans_destination_once(repository: Path, monkeypatch) -> None:
    original = repository / "before.txt"
    original.write_bytes(MARKER)
    _git(repository, "add", original.name)
    _git(repository, "commit", "-qm", "previous file")
    _git(repository, "mv", original.name, "after.txt")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 1
    assert captured == [("after.txt", MARKER)]


def test_deleted_then_readded_file_is_scanned(repository: Path, monkeypatch) -> None:
    _git(repository, "rm", "fixture.txt")
    (repository / "fixture.txt").write_bytes(MARKER)
    _git(repository, "add", "fixture.txt")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 1
    assert captured == [("fixture.txt", MARKER)]


def test_deletion_only_does_not_start_scanner(repository: Path, monkeypatch) -> None:
    _git(repository, "rm", "fixture.txt")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 0
    assert captured == []


def test_missing_index_object_refuses_success(repository: Path, monkeypatch) -> None:
    _git(repository, "update-index", "--add", "--cacheinfo", "100644", "1" * 40, "missing.txt")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 2
    assert captured == []


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "/outside.txt",
        "C:drive.txt",
        "nested/../escape.txt",
        "a\\b.txt",
        ".git/config",
    ],
)
def test_unsafe_index_path_refuses_before_materialization(
    repository: Path, monkeypatch, path: str
) -> None:
    real_run = subprocess.run
    captured = _scanner(monkeypatch)

    def run(command, **kwargs):
        if command[0:3] == ["git", "diff", "--cached"]:
            return subprocess.CompletedProcess(
                command,
                0,
                b":000000 100644 " + b"0" * 40 + b" " + b"1" * 40 + b" A\0" + path.encode() + b"\0",
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", run)

    assert betterleaks_hook.main([]) == 2
    assert captured == []


@pytest.mark.parametrize(
    "paths",
    [
        ("Alias.txt", "alias.txt"),
        ("Folder/first.txt", "folder/second.txt"),
        ("FILE", "file/nested.txt"),
        ("trailing.txt.",),
        ("trailing.txt ",),
        ("nested./file.txt",),
        ("NUL.txt",),
        ("nested/COM1.txt",),
        ("nested/LPT¹.log",),
        ("nested/CON .txt",),
        ('invalid"name.txt',),
        ("invalid\x01name.txt",),
    ],
)
def test_real_index_windows_aliases_refuse_before_blob_read(
    repository: Path, monkeypatch, paths: tuple[str, ...]
) -> None:
    _git(repository, "config", "core.protectNTFS", "false")
    _git(repository, "config", "core.ignorecase", "false")
    blob = _git(repository, "hash-object", "-w", "--stdin", data=MARKER).strip().decode()
    for path in paths:
        _git(repository, "update-index", "--add", "--cacheinfo", "100644", blob, path)
    real_run = subprocess.run
    captured = _scanner(monkeypatch)

    def run(command, **kwargs):
        assert command[:2] != ["git", "cat-file"]
        return real_run(command, **kwargs)

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", run)

    assert betterleaks_hook.main([]) == 2
    assert captured == []


def test_snapshot_never_overwrites_an_existing_file(repository: Path, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    destination = snapshot / "existing.txt"
    destination.write_bytes(b"original")
    blob = _git(repository, "hash-object", "-w", "--stdin", data=MARKER).strip().decode()

    with pytest.raises(FileExistsError):
        betterleaks_hook._snapshot_entries(
            repository, snapshot, [(destination.name, "100644", blob)]
        )

    assert destination.read_bytes() == b"original"


@pytest.mark.parametrize(
    "selection",
    ["default", ".betterleaks.toml", ".gitleaks.toml", "BETTERLEAKS_CONFIG", "GITLEAKS_CONFIG"],
)
def test_staged_scan_preserves_repository_config_ignore_and_baseline(
    repository: Path, monkeypatch, selection: str
) -> None:
    (repository / "target.txt").write_bytes(b"fixture")
    _git(repository, "add", "target.txt")
    baseline = repository / ".betterleaks-baseline.json"
    baseline.write_text("[]", encoding="utf-8")
    # Both discovery names exist to exercise Betterleaks' documented precedence.
    config = repository / (selection if selection.startswith(".") else "custom.toml")
    if selection != "default":
        config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
        if selection == ".betterleaks.toml":
            (repository / ".gitleaks.toml").write_text("invalid", encoding="utf-8")
        elif selection.endswith("CONFIG"):
            monkeypatch.setenv(selection, config.name)
    (repository / ".betterleaksignore").write_text("synthetic-fingerprint\n", encoding="utf-8")
    real_run = subprocess.run
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")

    def run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        assert Path(kwargs["cwd"]) != repository
        assert Path(command[command.index("--baseline-path") + 1]) == baseline
        assert Path(command[command.index("--gitleaks-ignore-path") + 1]) == repository
        selected = Path(command[command.index("--config") + 1])
        if selection == "default":
            assert selected.parent != Path(kwargs["cwd"])
            assert selected.read_text() == "[extend]\nuseDefault = true\n"
        else:
            assert selected == config
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, stdout=b"sensitive stdout", stderr=b"")

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", run)

    assert betterleaks_hook.main([]) == 0


def test_inline_environment_config_keeps_precedence(repository: Path, monkeypatch) -> None:
    (repository / ".betterleaks.toml").write_text("invalid", encoding="utf-8")
    monkeypatch.setenv("GITLEAKS_CONFIG_TOML", "[extend]\nuseDefault = true\n")

    assert betterleaks_hook._configuration_arguments(repository, repository / "unused.toml") == [
        "--gitleaks-ignore-path",
        str(repository),
    ]
    assert not (repository / "unused.toml").exists()


@pytest.mark.parametrize("output", [b"broken", b"bad\0path\0", b"\0", b"metadata\0"])
def test_incomplete_index_report_refuses_success(
    repository: Path, monkeypatch, output: bytes
) -> None:
    monkeypatch.setattr(
        betterleaks_hook.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, output),
    )

    assert betterleaks_hook.main([]) == 2


def test_index_change_during_scanner_refuses_success(repository: Path, monkeypatch) -> None:
    candidate = repository / "changing.txt"
    candidate.write_bytes(b"before")
    _git(repository, "add", candidate.name)
    real_run = subprocess.run
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")

    def run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        candidate.write_bytes(MARKER)
        real_run(["git", "add", candidate.name], cwd=repository, check=True, capture_output=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", run)

    assert betterleaks_hook.main([]) == 2


@pytest.mark.parametrize("path", ["missing.txt", "../outside.txt"])
def test_explicit_unreadable_or_escaping_path_refuses(
    repository: Path, monkeypatch, path: str
) -> None:
    (repository.parent / "outside.txt").write_bytes(b"fixture")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([path]) == 2
    assert captured == []


@pytest.mark.parametrize(
    "diagnostics",
    [
        b"DBG skipping binary file",
        b"WRN skipping archive: exceeds max archive depth",
        b"WRN partial scan completed",
        b"WRN could not read compressed file",
        b"18:12 ERR failure",
    ],
)
def test_scanner_incomplete_diagnostics_refuse_zero_exit(
    repository: Path, monkeypatch, diagnostics: bytes
) -> None:
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")
    monkeypatch.setattr(
        betterleaks_hook.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stderr=diagnostics),
    )

    assert (
        betterleaks_hook._run_scan(repository, ["fixture.txt"], repository / "baseline.json") == 2
    )


@pytest.mark.parametrize(
    "configuration", ["diff.ignoreSubmodules", "submodule.module.ignore", "gitmodules"]
)
def test_local_configuration_cannot_hide_staged_gitlink(
    repository: Path, monkeypatch, configuration: str
) -> None:
    if configuration == "gitmodules":
        (repository / ".gitmodules").write_text(
            '[submodule "module"]\n'
            "\tpath = module\n"
            "\turl = https://example.invalid/synthetic-module\n"
            "\tignore = all\n",
            encoding="utf-8",
        )
        _git(repository, "add", ".gitmodules")
        _git(repository, "commit", "-qm", "synthetic submodule configuration")
    else:
        _git(repository, "config", configuration, "all")
    commit = _git(repository, "rev-parse", "HEAD").strip().decode("ascii")
    _git(repository, "update-index", "--add", "--cacheinfo", "160000", commit, "module")
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 2
    assert captured == []


def test_type_change_to_gitlink_refuses_without_modifying_index(
    repository: Path, monkeypatch
) -> None:
    commit = _git(repository, "rev-parse", "HEAD").strip().decode("ascii")
    _git(repository, "update-index", "--cacheinfo", "160000", commit, "fixture.txt")
    index = repository / _git(repository, "rev-parse", "--git-path", "index").decode().strip()
    before = index.read_bytes()
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 2
    assert captured == []
    assert index.read_bytes() == before


def test_staged_scan_does_not_modify_git_index(repository: Path, monkeypatch) -> None:
    candidate = repository / "candidate.txt"
    candidate.write_bytes(MARKER)
    _git(repository, "add", candidate.name)
    candidate.write_bytes(b"different synthetic working copy")
    index = repository / _git(repository, "rev-parse", "--git-path", "index").decode().strip()
    before = index.read_bytes()
    captured = _scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 1
    assert captured == [(candidate.name, MARKER)]
    assert index.read_bytes() == before
    assert candidate.read_bytes() == b"different synthetic working copy"
