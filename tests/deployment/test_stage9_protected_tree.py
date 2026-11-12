"""Data-only Git exports and adversarial inventory checks; no VM installation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_stage9_protected_tree as tree

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux protected-tree procedure")


def git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(source), *arguments],
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


@pytest.fixture
def exported(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    (source / "deployment").mkdir()
    (source / "deployment/stage9_vm_native.py").write_text(
        "raise RuntimeError('This data fixture must never execute')\n"
    )
    (source / "deployment/launch.sh").write_text("exit 77\n")
    (source / "deployment/launch.sh").chmod(0o755)
    (source / "extra.txt").write_text("not selected\n")
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    revision = git(source, "rev-parse", "HEAD")
    candidate = tmp_path / "candidate"
    inventory = tmp_path / "inventory.json"
    paths = ("deployment/stage9_vm_native.py", "deployment/launch.sh")
    before = {
        str(p.relative_to(source)): (p.stat().st_mode, p.read_bytes())
        for p in source.rglob("*")
        if p.is_file()
    }
    result = tree.export(source, revision, candidate, tmp_path / "protected", inventory, paths)
    after = {
        str(p.relative_to(source)): (p.stat().st_mode, p.read_bytes())
        for p in source.rglob("*")
        if p.is_file()
    }
    assert before == after
    document = tree._document(inventory.read_bytes(), result["inventory_sha256"])
    return source, revision, candidate, inventory, paths, result, document


def test_export_is_exact_git_data_and_leaves_checkout_permissions(exported):
    source, revision, candidate, inventory, paths, result, document = exported
    assert result["status"] == "UNTRUSTED_CANDIDATE"
    assert document["source_revision"] == revision
    assert {item["path"] for item in document["objects"]} == {".", "deployment", *paths}
    assert not (candidate / "extra.txt").exists()
    assert not any(path.suffix in {".sig", ".pem", ".pyc"} for path in candidate.rglob("*"))
    assert tree.verify_tree(candidate, document, protected=False) == {
        path: git(source, "show", revision + ":" + path).encode() + b"\n" for path in paths
    }
    assert {item["mode"] for item in document["objects"] if item["kind"] == "file"} == {
        0o444,
        0o555,
    }


def test_export_reads_commit_instead_of_changed_checkout(exported, tmp_path):
    source, revision, _candidate, _inventory, paths, _result, _document = exported
    (source / paths[0]).write_text("uncommitted replacement must not be copied")
    output = tmp_path / "second"
    tree.export(
        source, revision, output, tmp_path / "protected-second", tmp_path / "second.json", paths
    )
    assert (output / paths[0]).read_bytes() == git(
        source, "show", revision + ":" + paths[0]
    ).encode() + b"\n"


@pytest.mark.parametrize(
    "relative",
    [
        "unexpected.py",
        "deployment/stage9_vm_native.so",
        "deployment/package",
        "deployment/__pycache__",
    ],
)
def test_extra_files_and_empty_directories_are_rejected(exported, relative):
    candidate, document = exported[2], exported[6]
    path = candidate / relative
    if relative.endswith(("package", "__pycache__")):
        path.mkdir()
    else:
        path.write_text("unexecuted adversarial data")
    with pytest.raises(tree.TreeError, match="UNAPPROVED_TREE_OBJECT"):
        tree.verify_tree(candidate, document, protected=False)


def test_missing_file_and_changed_bytes_are_rejected(exported):
    candidate, document = exported[2], exported[6]
    path = candidate / exported[4][0]
    path.chmod(0o600)
    path.write_text("changed")
    with pytest.raises(tree.TreeError, match="TREE_INTEGRITY_REFUSED"):
        tree.verify_tree(candidate, document, protected=False)
    path.unlink()
    with pytest.raises(tree.TreeError, match="TREE_INVENTORY_INCOMPLETE"):
        tree.verify_tree(candidate, document, protected=False)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory"])
def test_selected_executable_replaced_by_nonregular_object_is_rejected(exported, kind):
    candidate, document = exported[2], exported[6]
    path = candidate / exported[4][0]
    path.unlink()
    if kind == "symlink":
        path.symlink_to(candidate / exported[4][1])
    elif kind == "hardlink":
        os.link(candidate / exported[4][1], path)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    with pytest.raises(tree.TreeError):
        tree.verify_tree(candidate, document, protected=False)


def test_transport_candidate_cannot_pass_protected_verification(exported):
    candidate, document = exported[2], exported[6]
    document["tree_root"] = str(candidate)
    with pytest.raises(tree.TreeError, match="PROTECTED_PARENT_REQUIRED|TREE_NOT_PROTECTED"):
        tree.verify_tree(candidate, document, protected=True)


@pytest.mark.parametrize(
    "change", ["pin", "field", "mode", "duplicate", "missing-directory", "traversal"]
)
def test_inventory_refuses_unreviewed_or_malformed_contract(exported, change):
    document = exported[6]
    if change == "field":
        document["unexpected"] = True
    elif change == "mode":
        document["objects"][-1]["mode"] = 0o777
    elif change == "duplicate":
        document["objects"].append(document["objects"][-1].copy())
    elif change == "missing-directory":
        document["objects"] = [item for item in document["objects"] if item["path"] != "deployment"]
    elif change == "traversal":
        document["objects"][-1]["path"] = "../outside"
    raw = json.dumps(document).encode()
    digest = "0" * 64 if change == "pin" else hashlib.sha256(raw).hexdigest()
    with pytest.raises(tree.TreeError):
        tree._document(raw, digest)


def test_install_requires_native_root_before_opening_input(tmp_path, monkeypatch):
    monkeypatch.setattr(tree.os, "geteuid", lambda: 1001)
    with pytest.raises(tree.TreeError, match="NATIVE_ROOT_INSTALL_REQUIRED"):
        tree.install(tmp_path / "absent", tmp_path / "absent.json", "0" * 64)


@pytest.mark.parametrize("selected", [("deployment",), ("../escape",), ("unknown.py",)])
def test_export_refuses_directories_traversal_and_unknown_files(exported, tmp_path, selected):
    with pytest.raises(tree.TreeError):
        tree.export(
            exported[0],
            exported[1],
            tmp_path / "refused",
            tmp_path / "new-root",
            tmp_path / "refused.json",
            selected,
        )
    assert not (tmp_path / "refused").exists()


def test_cli_refusal_has_real_nonzero_exit(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(Path(tree.__file__)),
            "install",
            "--candidate",
            str(tmp_path / "absent"),
            "--inventory",
            str(tmp_path / "absent.json"),
            "--inventory-sha256",
            "0" * 64,
        ],
        capture_output=True,
        check=False,
    )
    assert process.returncode == 1
    assert json.loads(process.stdout) == {"status": "FAIL", "reason": "PROTECTED_TREE_REFUSED"}


@pytest.mark.parametrize("kind", ["commit", "tree", "blob"])
def test_git_object_content_must_match_its_address(exported, tmp_path, monkeypatch, kind):
    original = tree._git

    def altered_object(source, *arguments):
        value = original(source, *arguments)
        if arguments[:2] == ("cat-file", kind):
            return value + b"tampered"
        return value

    monkeypatch.setattr(tree, "_git", altered_object)
    with pytest.raises(tree.TreeError, match="GIT_OBJECT_INTEGRITY_REFUSED"):
        tree.export(
            exported[0],
            exported[1],
            tmp_path / "refused",
            tmp_path / "new-root",
            tmp_path / "refused.json",
            exported[4],
        )
    assert not (tmp_path / "refused").exists()


def test_git_symlink_blob_is_not_exported(exported, tmp_path):
    source = exported[0]
    (source / "deployment/unapproved.py").symlink_to("stage9_vm_native.py")
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "link",
    )
    with pytest.raises(tree.TreeError, match="REGULAR_GIT_BLOB_REQUIRED"):
        tree.export(
            source,
            git(source, "rev-parse", "HEAD"),
            tmp_path / "refused",
            tmp_path / "new-root",
            tmp_path / "refused.json",
            ("deployment/unapproved.py",),
        )
    assert not (tmp_path / "refused").exists()


def test_installed_inventory_metadata_is_checked_before_parsing(exported, monkeypatch):
    inventory = exported[3]
    inventory.chmod(0o600)
    monkeypatch.setattr(tree, "INVENTORY", inventory)
    # Model only the already-checked ancestor boundary; leaf metadata remains real.
    monkeypatch.setattr(tree, "_protected_ancestors", lambda _path: None)

    def must_not_parse(*_arguments):
        raise AssertionError("Unprotected inventory must never be interpreted")

    monkeypatch.setattr(tree, "_document", must_not_parse)
    with pytest.raises(tree.TreeError, match="INVENTORY_NOT_PROTECTED"):
        tree.verify_installed(exported[5]["inventory_sha256"])


def test_duplicate_json_fields_are_rejected_before_schema():
    raw = b'{"schema":1,"schema":1}'
    with pytest.raises(tree.TreeError, match="DUPLICATE_JSON_FIELD"):
        tree._document(raw, hashlib.sha256(raw).hexdigest())
