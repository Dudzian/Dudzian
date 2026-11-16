"""Nieuprzywilejowane próby V5d3 na odrębnych checkoutach i kopiach danych."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any

import pytest

from scripts import stage9_v5d3_release as release

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/stage9_v5d3_prepare_release.sh"
VERIFY = ROOT / "scripts/stage9_v5d3_verify_bundle.sh"
BASE = "17e03a9cb42a84302401415f96ac90e756926915"
pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform != "linux", reason="Wymagany zaufany Linux, Bash i Git"),
]


def _git(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repository),
            *args,
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
        },
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _commit(repository: Path) -> None:
    _git(repository, "add", "-A")
    _git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "Syntetyczne dane próby pakietu",
    )


@pytest.fixture(scope="session")
def source_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = tmp_path_factory.mktemp("v5d3-source") / "checkout"
    subprocess.run(
        ["/usr/bin/git", "clone", "--shared", "--no-checkout", str(ROOT), str(source)],
        check=True,
        capture_output=True,
    )
    _git(source, "sparse-checkout", "init", "--no-cone")
    _git(
        source,
        "sparse-checkout",
        "set",
        "--no-cone",
        "/deployment/",
        "/scripts/stage9*",
        "/docs/architecture/stage9*",
        "/pyproject.toml",
    )
    _git(source, "checkout", "--detach", BASE)
    # Nowe narzędzia są kopiowane jako jawny commit potomny historycznej bazy.
    for directory in ("scripts", "docs/architecture"):
        for item in (ROOT / directory).glob("stage9*v5d3*"):
            target = source / directory / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    _commit(source)
    assert not _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    return source


def _prepare(
    source: Path, parent: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/usr/bin/bash",
            str(PREPARE),
            "--repository",
            str(source),
            "--output-parent",
            str(parent),
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _bundle_path(parent: Path) -> Path:
    bundles = list(parent.glob("stage9-v5d3-*/bundle"))
    assert len(bundles) == 1
    return bundles[0]


@pytest.fixture(scope="session")
def prepared_bundle(source_checkout: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    parent = tmp_path_factory.mktemp("v5d3-prepared")
    result = _prepare(source_checkout, parent)
    assert result.returncode == 0, result.stdout + result.stderr
    bundle = _bundle_path(parent)
    assert (bundle / "COMPLETE.json").is_file()
    return bundle


@pytest.fixture
def bundle(prepared_bundle: Path, tmp_path: Path) -> Path:
    result = tmp_path / "bundle"
    shutil.copytree(prepared_bundle, result)
    return result


def _verify(bundle: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    result = subprocess.run(
        ["/usr/bin/bash", str(VERIFY), str(bundle)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    assert isinstance(report, dict)
    return result, report


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _reseal(bundle: Path) -> None:
    """Lokalne odtworzenie sum nie może nadać zmienionym danym aprobaty."""
    manifest_path = bundle / "MANIFEST.json"
    manifest = _json(manifest_path)
    for item in manifest["files"]:
        target = bundle / item["path"]
        if target.is_file() and not target.is_symlink():
            item["size"] = target.stat().st_size
            item["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)
    paths = [item["path"] for item in manifest["files"]] + ["MANIFEST.json"]
    (bundle / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}\n"
            for name in sorted(paths)
            if (bundle / name).is_file() and not (bundle / name).is_symlink()
        )
    )


def _source(bundle: Path) -> Path:
    matches = list(bundle.rglob("stage9_observer_collector_identity_v5d1.py"))
    assert len(matches) == 1
    return matches[0]


def test_prepare_complete_bundle_remains_unapproved(prepared_bundle: Path) -> None:
    result, report = _verify(prepared_bundle)
    assert result.returncode == 0, report
    assert (prepared_bundle / "SUMMARY.tsv").is_file()
    summary = (prepared_bundle / "SUMMARY.json").read_text()
    for status in (
        "NOT_AUTHORIZED",
        "NOT_CLAIMED",
        "UNVERIFIED",
        "BLOCKED",
        "NO-GO",
        "IN_PROGRESS",
    ):
        assert status in summary
    assert "UNVERIFIED" in json.dumps(report)


def test_changed_source_sha256_is_rejected(bundle: Path) -> None:
    with _source(bundle).open("ab") as stream:
        stream.write(b"\n# Zmienione bajty po przygotowaniu.\n")
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_self_resealed_source_is_not_external_approval(bundle: Path, tmp_path: Path) -> None:
    sentinel = tmp_path / "nie_wykonano_badanego_kodu"
    with _source(bundle).open("a") as stream:
        stream.write(f"\nopen({str(sentinel)!r}, 'w').write('wykonano')\n")
    _reseal(bundle)
    result, report = _verify(bundle)
    assert result.returncode == 0, report
    assert "UNVERIFIED" in json.dumps(report)
    assert not sentinel.exists()


def test_missing_required_source_still_rejected_after_resealing(bundle: Path) -> None:
    missing = _source(bundle)
    name = missing.relative_to(bundle).as_posix()
    missing.unlink()
    manifest = _json(bundle / "MANIFEST.json")
    manifest["files"] = [item for item in manifest["files"] if item["path"] != name]
    _write_json(bundle / "MANIFEST.json", manifest)
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_missing_from_package_alias_refused_by_independent_verifier(bundle: Path) -> None:
    with _source(bundle).open("a") as stream:
        stream.write("\nfrom deployment import absent_v5d3_probe\n")
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_incomplete_project_dependencies_rejected_after_resealing(bundle: Path) -> None:
    path = bundle / "DEPENDENCIES.json"
    dependencies = _json(path)
    dependencies["project_dependencies"] = [
        value for value in dependencies["project_dependencies"] if "cryptography" not in value
    ]
    _write_json(path, dependencies)
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_dynamic_import_cannot_be_removed_from_inventory(bundle: Path) -> None:
    path = bundle / "IMPORTS.json"
    inventory = _json(path)
    assert inventory["dynamic_imports"]
    inventory["dynamic_imports"].pop()
    _write_json(path, inventory)
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


@pytest.mark.parametrize("relative", ["MANIFEST.json", "SHA256SUMS", "DEPENDENCIES.json"])
def test_symlink_is_rejected_before_reading_target(bundle: Path, relative: str) -> None:
    target = bundle / relative
    target.unlink()
    target.symlink_to("/dev/null")
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_hardlink_is_rejected(bundle: Path, tmp_path: Path) -> None:
    target = _source(bundle)
    os.link(target, tmp_path / "dodatkowy_link")
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_symlink_ancestor_is_rejected(bundle: Path, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(bundle.parent, target_is_directory=True)
    result, _ = _verify(alias / bundle.name)
    assert result.returncode == 2


@pytest.mark.parametrize("relative", ["../escape", "/tmp/escape", "a/../b", "a//b", "a\\b"])
def test_unsafe_manifest_paths_are_rejected(bundle: Path, relative: str) -> None:
    manifest = _json(bundle / "MANIFEST.json")
    manifest["files"][0]["path"] = relative
    _write_json(bundle / "MANIFEST.json", manifest)
    result, _ = _verify(bundle)
    assert result.returncode == 2


@pytest.mark.parametrize("value", ["{}", "[]", "{", '{"profile":"a","profile":"b"}'])
def test_malformed_manifest_refused(bundle: Path, value: str) -> None:
    (bundle / "MANIFEST.json").write_text(value)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_duplicate_inventory_entry_refused(bundle: Path) -> None:
    manifest = _json(bundle / "MANIFEST.json")
    manifest["files"].append(manifest["files"][0])
    _write_json(bundle / "MANIFEST.json", manifest)
    result, _ = _verify(bundle)
    assert result.returncode == 2


@pytest.mark.parametrize("name", ["private-key.pem", ".env", "startup.pth", "target.pyc"])
def test_forbidden_payload_refused_even_with_local_manifest(bundle: Path, name: str) -> None:
    target = bundle / name
    target.write_text("SYNTHETIC_SENTINEL_WITH_NO_KEY_MATERIAL\n")
    manifest = _json(bundle / "MANIFEST.json")
    manifest["files"].append(
        {
            "path": name,
            "size": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "mode": f"{stat.S_IMODE(target.stat().st_mode):04o}",
        }
    )
    _write_json(bundle / "MANIFEST.json", manifest)
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_fake_authorization_refused_even_after_resealing(bundle: Path) -> None:
    summary = bundle / "SUMMARY.json"
    original = summary.read_text()
    assert "NOT_AUTHORIZED" in original
    summary.write_text(original.replace("NOT_AUTHORIZED", "AUTHORIZED"))
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_tsv_cannot_claim_different_authorization(bundle: Path) -> None:
    summary = bundle / "SUMMARY.tsv"
    original = summary.read_text()
    assert "NOT_AUTHORIZED" in original
    summary.write_text(original.replace("NOT_AUTHORIZED", "AUTHORIZED"))
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


@pytest.mark.parametrize(
    ("name", "field", "value"),
    [
        ("RELEASE.json", "release_commit", "0" * 40),
        ("NATIVE.json", "status", "PASS"),
        ("INTERPRETER.json", "startup_files", "PASS"),
        ("MISSING.json", "execution_authorization", "AUTHORIZED"),
    ],
)
def test_configuration_cannot_promote_readiness_after_resealing(
    bundle: Path, name: str, field: str, value: str
) -> None:
    path = bundle / name
    document = _json(path)
    document[field] = value
    _write_json(path, document)
    _reseal(bundle)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_duplicate_sha256_line_refused(bundle: Path) -> None:
    sums = bundle / "SHA256SUMS"
    with sums.open("a") as stream:
        stream.write(sums.read_text().splitlines()[0] + "\n")
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_missing_completion_marker_refused(bundle: Path) -> None:
    (bundle / "COMPLETE.json").unlink()
    result, _ = _verify(bundle)
    assert result.returncode == 2


@pytest.mark.parametrize("kind", ["file", "directory", "fifo"])
def test_unexpected_object_refused(bundle: Path, kind: str) -> None:
    target = bundle / "niespodziewany_obiekt"
    if kind == "file":
        target.write_text("niezadeklarowane dane\n")
    elif kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_read_only_verifier_preserves_existing_bytes_and_modes(bundle: Path) -> None:
    before = {
        p.relative_to(bundle): (p.read_bytes(), stat.S_IMODE(p.stat().st_mode))
        for p in bundle.rglob("*")
        if p.is_file()
    }
    result, report = _verify(bundle)
    assert result.returncode == 0, report
    after = {
        p.relative_to(bundle): (p.read_bytes(), stat.S_IMODE(p.stat().st_mode))
        for p in bundle.rglob("*")
        if p.is_file()
    }
    assert after == before


@pytest.mark.parametrize("interrupted", [True, False])
def test_failed_or_interrupted_final_verification_revokes_completion_preserving_receipts(
    bundle: Path, prepared_bundle: Path, interrupted: bool
) -> None:
    sealed_names = ("COMPLETE.json", "MANIFEST.json", "SHA256SUMS", "SUMMARY.json", "SUMMARY.tsv")
    retained = {name: (bundle / name).read_bytes() for name in sealed_names}
    state = _json(prepared_bundle.parent / "state.json")
    release.invalidate(bundle.parent, state, interrupted=interrupted)
    archive = list(bundle.parent.glob("rejected-seal-*"))
    assert len(archive) == 1
    for name, data in retained.items():
        assert (archive[0] / name).read_bytes() == data
    assert not (bundle / "COMPLETE.json").exists()
    assert not (bundle / "MANIFEST.json").exists()
    summary = _json(bundle / "SUMMARY.json")
    assert any(row["area"] == "package" and row["status"] == "FAIL" for row in summary["areas"])
    assert any(
        row["area"] == "final_verification" and row["status"] == "FAIL" for row in summary["areas"]
    )
    result, _ = _verify(bundle)
    assert result.returncode == 2


def test_root_invocation_is_refused_without_creating_attempt(
    source_checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release.os, "geteuid", lambda: 0)
    with pytest.raises(release.Refusal):
        release.initialize(str(source_checkout), str(tmp_path), "")
    assert not list(tmp_path.iterdir())


def test_repeated_preparation_preserves_previous_attempt(
    source_checkout: Path, tmp_path: Path
) -> None:
    first = _prepare(source_checkout, tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    previous = _bundle_path(tmp_path)
    retained = (previous / "SHA256SUMS").read_bytes()
    second = _prepare(source_checkout, tmp_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert len(list(tmp_path.glob("stage9-v5d3-*/bundle"))) == 2
    assert (previous / "SHA256SUMS").read_bytes() == retained


def test_resume_incomplete_attempt_keeps_previous_evidence(
    source_checkout: Path, tmp_path: Path
) -> None:
    first = _prepare(source_checkout, tmp_path)
    assert first.returncode == 0, first.stdout + first.stderr
    previous = _bundle_path(tmp_path)
    (previous / "COMPLETE.json").unlink()
    retained = (previous / "SHA256SUMS").read_bytes()
    second = _prepare(source_checkout, tmp_path, "--resume", str(previous))
    assert second.returncode == 0, second.stdout + second.stderr
    assert len(list(tmp_path.glob("stage9-v5d3-*/bundle"))) == 2
    assert not (previous / "COMPLETE.json").exists()
    assert (previous / "SHA256SUMS").read_bytes() == retained


def test_resume_complete_bundle_rechecks_it_without_overwriting(
    source_checkout: Path, prepared_bundle: Path, tmp_path: Path
) -> None:
    retained = (prepared_bundle / "SHA256SUMS").read_bytes()
    result = _prepare(source_checkout, tmp_path, "--resume", str(prepared_bundle))
    assert result.returncode == 0, result.stdout + result.stderr
    assert _bundle_path(tmp_path) != prepared_bundle
    assert (prepared_bundle / "SHA256SUMS").read_bytes() == retained


def test_resume_corrupted_complete_bundle_refuses_new_completion(
    source_checkout: Path, bundle: Path, tmp_path: Path
) -> None:
    source = _source(bundle)
    with source.open("a") as stream:
        stream.write("\n# Zmieniony poprzedni pakiet.\n")
    retained = source.read_bytes()
    parent = tmp_path / "nowe-wyniki"
    parent.mkdir()
    result = _prepare(source_checkout, parent, "--resume", str(bundle))
    assert result.returncode != 0
    assert source.read_bytes() == retained
    assert not list(parent.glob("stage9-v5d3-*/bundle/COMPLETE.json"))


def test_output_inside_repository_is_refused(source_checkout: Path) -> None:
    before = _git(source_checkout, "status", "--porcelain=v1", "--untracked-files=all")
    result = _prepare(source_checkout, source_checkout)
    assert result.returncode != 0
    assert _git(source_checkout, "status", "--porcelain=v1", "--untracked-files=all") == before


def test_symlink_output_parent_refused(source_checkout: Path, tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    result = _prepare(source_checkout, alias)
    assert result.returncode != 0
    assert not list(parent.iterdir())


def test_dirty_repository_is_recorded_and_refused(source_checkout: Path, tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(source_checkout, source)
    untracked = source / "operator-retained-evidence"
    untracked.write_text("nie usuwać\n")
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source, parent)
    assert result.returncode != 0
    assert untracked.read_text() == "nie usuwać\n"
    assert not list(parent.glob("stage9-v5d3-*/bundle/COMPLETE.json"))


def test_missing_tracked_source_is_refused(source_checkout: Path, tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(source_checkout, source)
    (source / "deployment/stage9_observer_kvm_domain_v5d1.py").unlink()
    _commit(source)
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source, parent)
    assert result.returncode != 0
    assert not list(parent.glob("stage9-v5d3-*/bundle/COMPLETE.json"))


def test_missing_from_package_import_is_refused(source_checkout: Path, tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(source_checkout, source)
    (source / "deployment/stage9_observer_apt_v4a.py").unlink()
    _commit(source)
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source, parent)
    assert result.returncode != 0
    assert not list(parent.glob("stage9-v5d3-*/bundle/COMPLETE.json"))


@pytest.mark.parametrize("kind", ["private-key-marker", "token-marker"])
def test_secret_marker_refused_before_source_is_written(
    source_checkout: Path, tmp_path: Path, kind: str
) -> None:
    source = tmp_path / "source"
    shutil.copytree(source_checkout, source)
    # Wyłącznie syntetyczne znaczniki: nie tworzymy żadnego klucza ani tokenu konta.
    needle = (
        ("-----BEGIN " + "PRIVATE KEY-----")
        if kind == "private-key-marker"
        else "gh" + "p_" + "X" * 36
    )
    target = source / "deployment/stage9_observer_collector_identity_v5d1.py"
    with target.open("a") as stream:
        stream.write(f"\n# {needle}\n")
    _commit(source)
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source, parent)
    assert result.returncode != 0
    for path in parent.rglob("*"):
        if path.is_file():
            assert needle.encode() not in path.read_bytes()
    assert not list(parent.glob("stage9-v5d3-*/bundle/COMPLETE.json"))


def test_git_replace_cannot_change_bytes_bound_to_release_commit(
    source_checkout: Path, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    shutil.copytree(source_checkout, source)
    original_revision = _git(source, "rev-parse", "HEAD")
    target = source / "deployment/stage9_observer_collector_identity_v5d1.py"
    original_bytes = target.read_bytes()
    with target.open("a") as stream:
        stream.write("\n# Niezaakceptowana treść zastępującego obiektu Git.\n")
    _commit(source)
    alternate_revision = _git(source, "rev-parse", "HEAD")
    _git(source, "reset", "--hard", original_revision)
    _git(source, "replace", original_revision, alternate_revision)
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source, parent)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _source(_bundle_path(parent)).read_bytes() == original_bytes


def test_missing_dependency_declaration_prevents_completion(
    source_checkout: Path, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    shutil.copytree(source_checkout, source)
    declaration = source / "pyproject.toml"
    original = declaration.read_text()
    assert '"defusedxml==0.7.1",' in original
    declaration.write_text(original.replace('"defusedxml==0.7.1",', ""))
    _commit(source)
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source, parent)
    assert result.returncode != 0
    assert not list(parent.glob("stage9-v5d3-*/bundle/COMPLETE.json"))
    summary = _json(_bundle_path(parent) / "SUMMARY.json")
    assert any(row["area"] == "C" and row["status"] == "FAIL" for row in summary["areas"])


def test_preparation_never_executes_repository_python(
    source_checkout: Path, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    shutil.copytree(source_checkout, source)
    marker = tmp_path / "wykonano-kod-checkoutu"
    target = source / "deployment/__init__.py"
    target.write_text(f"open({str(marker)!r}, 'w').write('wykonano')\n")
    _commit(source)
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source, parent)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_inherited_python_startup_and_git_config_are_not_executed(
    source_checkout: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "wykonano-niejawne-startup"
    startup = tmp_path / "sitecustomize.py"
    startup.write_text(f"open({str(marker)!r}, 'w').write('wykonano')\n")
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path),
            "PYTHONSTARTUP": str(startup),
            "GIT_TRACE": str(marker),
            "GIT_TRACE2_EVENT": str(marker),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": f"printf niejawne > {marker}",
        }
    )
    parent = tmp_path / "results"
    parent.mkdir()
    result = _prepare(source_checkout, parent, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_all_preparation_stages_use_only_read_only_git_without_keys_or_vm(
    source_checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run = subprocess.run
    real_open = Path.open
    observed: list[list[str]] = []
    forbidden = (
        "/etc/cryptohunter",
        "/opt/cryptohunter",
        "/home/dudzian",
        "/root/.ssh",
    )

    def read_only_command(arguments: list[str], **kwargs: Any) -> Any:
        # Zwykła próba bez resume może wywoływać wyłącznie odczytowy Git.
        assert arguments[0] == "/usr/bin/git"
        operation = arguments[arguments.index("-C") + 2]
        assert operation in {"rev-parse", "status", "merge-base", "ls-tree", "show"}
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert "GIT_TRACE" not in kwargs["env"]
        assert all(not str(value).startswith(forbidden) for value in arguments)
        observed.append(arguments)
        return real_run(arguments, **kwargs)

    def no_trust_or_private_material(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert not str(path).startswith(forbidden)
        assert path.name not in {"id_ed25519", "id_rsa", "private.key"}
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(release.subprocess, "run", read_only_command)
    monkeypatch.setattr(Path, "open", no_trust_or_private_material)
    attempt = release.initialize(str(source_checkout), str(tmp_path), "")
    state = _json(attempt / "state.json")
    for letter in "ABCDEFGH":
        release.stage(attempt, letter, state)
    release.finalize(attempt, state)
    release.seal(attempt, state)
    assert observed
    assert all(state["areas"][letter]["status"] == "PASS" for letter in "ABCDEFGHI")
    assert (attempt / "bundle/COMPLETE.json").is_file()


def test_interrupted_preparation_retains_attempt_state(
    source_checkout: Path, tmp_path: Path
) -> None:
    process = subprocess.Popen(
        [
            "/usr/bin/bash",
            str(PREPARE),
            "--repository",
            str(source_checkout),
            "--output-parent",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        states = list(tmp_path.glob("stage9-v5d3-*/state.json"))
        if states:
            os.killpg(process.pid, signal.SIGTERM)
            break
        time.sleep(0.005)
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode != 0, stdout + stderr
    assert list(tmp_path.glob("stage9-v5d3-*/state.json"))
    assert not list(tmp_path.glob("stage9-v5d3-*/bundle/COMPLETE.json"))
