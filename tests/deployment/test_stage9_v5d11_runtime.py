from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import stat
import sys
import tempfile

import pytest

from deployment.stage9_observer_collector_identity_v5d1 import _read_protected
from scripts import stage9_v5d10_contract as contract
from scripts import stage9_v5d11_runtime as runtime

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux protected runtime")

UID = 49_991
GID = 49_992
NATIVE = "cryptography/native.so"
CODE = "scripts/observer.py"
WHEEL = f"cryptography-50.0.2-cp311-abi3-manylinux_2_34_x86_64.whl:{NATIVE}"
CONFIG = b"include-system-site-packages = false\n"


def inputs(root):
    payloads = {
        "venv/bin/python3.12": (0o755, b"synthetic interpreter"),
        "venv/pyvenv.cfg": (0o600, CONFIG),
        f"packages/{CODE}": (0o644, b"synthetic controller"),
        f"packages/{NATIVE}": (0o600, b"synthetic native extension"),
        **{f"packages/{name}": (0o600, name.encode()) for name in contract.BINDINGS},
    }
    files = {
        path: {"mode": mode, "sha256": contract.digest(data)}
        for path, (mode, data) in payloads.items()
    }
    folders = {
        ".",
        "packages",
        "packages/cryptography",
        "packages/scripts",
        "venv",
        "venv/bin",
    }
    tree = {"files": files, "directories": dict.fromkeys(folders, 0o700)}
    host = {**contract.HOST, "interpreter_sha256": files["venv/bin/python3.12"]["sha256"]}
    base = {
        "profile": "stage9-v5d4-base-runtime-inventory-v1",
        "roots": ["/usr/lib/python3.12"],
        "files": {"/usr/lib/python3.12/os.py": contract.digest(b"base")},
        "native": [],
        "symlinks": {},
        "excluded": [],
        "scope": "STATIC_BYTES_SUPERSET_NOT_DYNAMIC_COMPLETENESS",
        "installation_approved": False,
    }
    tools = {contract.QEMU_IMG: contract.digest(b"qemu")}
    common = dict.fromkeys(contract.APPROVAL_DIGESTS, contract.digest(b"pin"))
    common["file_plan_sha256"] = contract.plan_sha256(tree)
    approval = {
        "profile": contract.APPROVAL,
        **common,
        **{key: [] for key in contract.OMISSION_FIELDS},
        "host": host,
        "code": {CODE: files[f"packages/{CODE}"]["sha256"]},
        "wheel_native": {WHEEL: files[f"packages/{NATIVE}"]["sha256"]},
        "bindings": {
            name: {
                "path": f"/usr/lib/python3/dist-packages/{name}",
                "sha256": files[f"packages/{name}"]["sha256"],
            }
            for name in contract.BINDINGS
        },
        "base_runtime": base,
        "host_tools": tools,
    }
    receipt = {
        "profile": contract.RECEIPT,
        **common,
        **tree,
        "root": str(root),
        "entrypoint": contract.ENTRYPOINT,
        "approval_sha256": contract.digest(contract.canonical(approval)),
        "host": host,
        "base_runtime": base,
        "external_files": dict(tools),
        "authorization": "NOT_AUTHORIZED",
        "native_load": "NOT_EXECUTED",
    }
    return receipt, approval, payloads


def _derive(receipt, approval, uid=UID, gid=GID):
    rec_raw, app_raw = contract.canonical(receipt), contract.canonical(approval)
    return runtime.derive(
        rec_raw,
        contract.digest(rec_raw),
        app_raw,
        contract.digest(app_raw),
        uid,
        gid,
        vars(contract),
    )


def _pin(document):
    return contract.digest(contract.canonical(document))


def _write_tree(root, receipt, payloads):
    for name, mode in sorted(receipt["directories"].items(), key=lambda row: row[0].count("/")):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(mode)
    for name, (mode, content) in payloads.items():
        path = root / name
        path.write_bytes(content)
        path.chmod(mode)
    receiving = root / contract.RECEIPT_NAME
    receiving.write_bytes(contract.canonical(receipt))
    receiving.chmod(0o600)


def test_receipt_runtime_projection_preserves_source_and_separate_pins(tmp_path):
    receipt, approval, _ = inputs(tmp_path / "materialized")
    original = deepcopy((receipt, approval))
    projected = _derive(receipt, approval)
    assert _derive(receipt, approval) == projected
    assert (receipt, approval) == original
    assert projected["profile"] == runtime.PROFILE
    assert projected["materialization_receipt_sha256"] == _pin(receipt)
    assert projected["materialization_approval_sha256"] == _pin(approval)
    assert projected["materialization_file_plan_sha256"] == receipt["file_plan_sha256"]
    assert projected["source_code_pins_sha256"] == _pin(approval["code"])
    assert projected["runtime_file_plan_sha256"] == _pin(
        {"directories": projected["directories"], "files": projected["files"]}
    )
    assert projected["authorization"] == "NOT_AUTHORIZED"
    assert projected["owner_uid"] == 0 and projected["uid"] == UID and projected["gid"] == GID
    assert set(projected["directories"].values()) == {runtime.DIRECTORY_MODE}
    assert projected["files"][contract.RECEIPT_NAME] == {
        "mode": 0o440,
        "sha256": _pin(receipt),
    }
    for name, row in receipt["files"].items():
        assert projected["files"][name]["sha256"] == row["sha256"]
        assert projected["files"][name]["mode"] == (0o550 if row["mode"] == 0o755 else 0o440)
    assert _pin(projected) != _pin(_derive(receipt, approval, gid=GID + 1))
    with pytest.raises(contract.Refusal, match="PROFILE"):
        contract.validate_receipt(projected)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("approval_binding", "RUNTIME_APPROVAL_BINDING_REJECTED"),
        ("receipt_binding", "RUNTIME_APPROVAL_BINDING_REJECTED"),
        ("wheel_path", "RUNTIME_CODE_BINDING_REJECTED"),
        ("duplicate_wheel", "RUNTIME_CODE_BINDING_REJECTED"),
        ("code_hash", "RUNTIME_CODE_BINDING_REJECTED"),
        ("executable", "RUNTIME_INTERPRETER_BINDING_REJECTED"),
    ],
)
def test_rejects_rehashed_inconsistent_approval(tmp_path, change, reason):
    receipt, approval, _ = inputs(tmp_path / "materialized")
    if change == "approval_binding":
        receipt["abi_request_sha256"] = "0" * 64
    elif change == "receipt_binding":
        receipt["external_files"][contract.QEMU_IMG] = "0" * 64
    elif change == "wheel_path":
        approval["wheel_native"] = {f"dep.whl:missing/{NATIVE}": "0" * 64}
    elif change == "duplicate_wheel":
        approval["wheel_native"][f"other.whl:{NATIVE}"] = approval["wheel_native"][WHEEL]
    elif change == "code_hash":
        approval["code"][CODE] = "0" * 64
    else:
        receipt["files"]["venv/bin/python3.12"]["mode"] = 0o600
        plan = {name: receipt[name] for name in ("files", "directories")}
        approval["file_plan_sha256"] = receipt["file_plan_sha256"] = contract.plan_sha256(plan)
    receipt["approval_sha256"] = _pin(approval)
    with pytest.raises(runtime.Refusal, match=reason):
        _derive(receipt, approval)


def test_exact_source_pins_account_and_modes(tmp_path):
    receipt, approval, _ = inputs(tmp_path / "materialized")
    raw_receipt, raw_approval = contract.canonical(receipt), contract.canonical(approval)
    with pytest.raises(runtime.Refusal, match="RUNTIME_SOURCE_PIN_MISMATCH"):
        runtime.derive(
            raw_receipt + b"\n",
            _pin(receipt),
            raw_approval,
            _pin(approval),
            UID,
            GID,
            vars(contract),
        )
    with pytest.raises(runtime.Refusal, match="RUNTIME_SOURCE_PIN_MISMATCH"):
        runtime.derive(
            raw_receipt,
            _pin(receipt),
            raw_approval + b"\n",
            _pin(approval),
            UID,
            GID,
            vars(contract),
        )
    with pytest.raises(runtime.Refusal, match="RUNTIME_ACCOUNT_REJECTED"):
        _derive(receipt, approval, uid=0)
    with pytest.raises(runtime.Refusal, match="RUNTIME_ACCOUNT_REJECTED"):
        _derive(receipt, approval, gid=True)
    with pytest.raises(runtime.Refusal, match="RUNTIME_SOURCE_MODE_REJECTED"):
        runtime.file_mode(0o777)


def _stage(root, stage):
    receipt, approval, payloads = inputs(root)
    _write_tree(root, receipt, payloads)
    relative = root.relative_to("/")
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.chmod(0o755)
    stage.parent.chmod(0o700)
    for ancestor in target.parents:
        if ancestor == stage:
            break
        ancestor.chmod(0o755)
    _write_tree(target, receipt, payloads)
    return receipt, approval, target


@pytest.fixture
def projected_tree(tmp_path, monkeypatch):
    source = tmp_path / "materialized"
    view = tmp_path / "staging" / "root"
    receipt, approval, target = _stage(source, view)
    document = _derive(receipt, approval)
    owner_by_inode = {}
    real_stat = Path.lstat
    real_fstat = os.fstat
    stage_contents = [target, *target.rglob("*")]
    for path in [view, view.parent, *target.parents, *stage_contents]:
        if path.is_relative_to(view.parent):
            observed = real_stat(path)
            owner_by_inode[(observed.st_dev, observed.st_ino)] = (
                (UID, GID) if path in stage_contents else (0, 0)
            )

    def owner(info):
        ids = owner_by_inode.get((info.st_dev, info.st_ino))
        if ids is None:
            return info
        values = list(info)
        values[4:6] = ids
        extra = {
            "st_atime_ns": info.st_atime_ns,
            "st_mtime_ns": info.st_mtime_ns,
            "st_ctime_ns": info.st_ctime_ns,
        }
        return os.stat_result(values, extra)

    def fchown(fd, uid, gid):
        info = real_fstat(fd)
        owner_by_inode[(info.st_dev, info.st_ino)] = (uid, gid)

    monkeypatch.setattr(Path, "lstat", lambda path: owner(real_stat(path)))
    monkeypatch.setattr(os, "fstat", lambda fd: owner(real_fstat(fd)))
    monkeypatch.setattr(os, "fchown", fchown)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    return source, view, target, document, owner_by_inode


def test_apply_tree_checks_source_is_separate_and_preserves_actual_modes(projected_tree):
    source, view, target, document, _ = projected_tree
    with pytest.raises(runtime.Refusal, match="RUNTIME_DAC_MISMATCH"):
        runtime.verify_tree(view, document, _pin(document), vars(contract))
    runtime.apply_tree(view, document, _pin(document), vars(contract))
    runtime.verify_tree(view, document, _pin(document), vars(contract))
    received = source / contract.RECEIPT_NAME
    installed = target / contract.RECEIPT_NAME
    assert installed.read_bytes() == received.read_bytes()
    assert installed.stat().st_ino != received.stat().st_ino
    assert stat.S_IMODE(installed.stat().st_mode) == 0o440
    assert stat.S_IMODE(received.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.stat().st_mode) == 0o550
    assert stat.S_IMODE(source.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("new", "RUNTIME_(EXTRA_PATHS|UNEXPECTED_PATH)"),
        ("missing", "RUNTIME_PATHS_MISSING"),
        ("symlink", "RUNTIME_FILE_REJECTED"),
        ("hardlink", "RUNTIME_FILE_REJECTED"),
        ("write", "RUNTIME_DAC_MISMATCH"),
        ("identity", "RUNTIME_DAC_MISMATCH"),
        ("owner", "RUNTIME_DAC_MISMATCH"),
        ("wrong_contents", "RUNTIME_FILE_PIN_MISMATCH"),
        ("group_write_ancestor", "RUNTIME_ANCESTOR_NOT_PROTECTED"),
        ("stale_pin", "RUNTIME_RECEIPT_PIN_MISMATCH"),
        ("fabricated_plan", "RUNTIME_PROJECTION_MISMATCH"),
    ],
)
def test_verifier_refuses_unapproved_dac_paths_and_identity(projected_tree, change, reason):
    source, view, target, document, owner = projected_tree
    runtime.apply_tree(view, document, _pin(document), vars(contract))
    controlled = target / f"packages/{CODE}"
    if change in {"new", "missing", "symlink", "hardlink"}:
        controlled.parent.chmod(0o750)
        if change == "new":
            controlled.with_name("added.py").write_bytes(b"unplanned")
        else:
            controlled.unlink()
        if change == "symlink":
            controlled.symlink_to(source / f"packages/{CODE}")
        elif change == "hardlink":
            controlled.hardlink_to(source / f"packages/{CODE}")
        controlled.parent.chmod(0o550)
    elif change == "write":
        controlled.chmod(0o460)
    elif change in {"identity", "owner"}:
        info = controlled.lstat()
        owner[(info.st_dev, info.st_ino)] = (0, GID + 1) if change == "identity" else (UID, GID)
    elif change == "wrong_contents":
        controlled.chmod(0o640)
        controlled.write_bytes(b"other data")
        controlled.chmod(0o440)
    elif change == "group_write_ancestor":
        target.parent.chmod(0o775)
    elif change == "fabricated_plan":
        document["files"][f"packages/{CODE}"]["mode"] = 0o444
    pin = "0" * 64 if change == "stale_pin" else _pin(document)
    with pytest.raises(runtime.Refusal, match=reason):
        runtime.verify_tree(view, document, pin, vars(contract))


def test_staged_inode_already_materialized_refused_without_changes(projected_tree, monkeypatch):
    source, view, target, document, _ = projected_tree
    saved_lstat = Path.lstat

    def aliased(path):
        if path == source / "packages":
            return saved_lstat(target / "venv")
        return saved_lstat(path)

    monkeypatch.setattr(Path, "lstat", aliased)
    with pytest.raises(runtime.Refusal, match="RUNTIME_MATERIALIZATION_OVERLAP"):
        runtime.apply_tree(view, document, _pin(document), vars(contract))
    assert stat.S_IMODE((source / contract.RECEIPT_NAME).stat().st_mode) == 0o600
    assert stat.S_IMODE((target / contract.RECEIPT_NAME).stat().st_mode) == 0o600


def test_runtime_candidate_command(tmp_path, capsys):
    source = tmp_path / "materialized"
    receipt, approval, payloads = inputs(source)
    _write_tree(source, receipt, payloads)
    approval_path = tmp_path / "approval.json"
    approval_path.write_bytes(contract.canonical(approval))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        contract.canonical(
            {
                "root": str(source),
                "approval": str(approval_path),
                "receipt_sha256": _pin(receipt),
                "approval_sha256": _pin(approval),
                "uid": UID,
                "gid": GID,
            }
        )
    )
    assert runtime.main(["review", str(manifest)]) == 0
    assert contract.loads(capsys.readouterr().out.encode()) == _derive(receipt, approval)
    assert runtime.main(["review", "relative.json"]) == 2
    assert capsys.readouterr().err == "REFUSED RUNTIME_SOURCE_PATH_REJECTED\n"


def test_read_and_execute_without_write_reaches_kernel(projected_tree, monkeypatch):
    _, view, target, document, _ = projected_tree
    runtime.apply_tree(view, document, _pin(document), vars(contract))
    controlled = target / f"packages/{CODE}"
    monkeypatch.undo()
    if os.geteuid() == 0:
        pytest.skip("root bypasses kernel DAC")
    assert controlled.read_bytes() == b"synthetic controller"
    assert os.access(target / "venv/bin/python3.12", os.R_OK | os.X_OK)
    assert not os.access(controlled, os.W_OK)
    with pytest.raises(PermissionError):
        controlled.open("ab").close()
    with pytest.raises(PermissionError):
        controlled.with_name("new.py").touch()


def _root_dac_available():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return False
    caps = int(
        next(
            row.split()[1]
            for row in Path("/proc/self/status").read_text().splitlines()
            if row.startswith("CapEff:")
        ),
        16,
    )
    return sys.platform == "linux" and caps & 0x400C0 == 0x400C0


@pytest.mark.skipif(not _root_dac_available(), reason="requires root SETUID, SETGID and SYS_CHROOT")
def test_v5d1_reads_runtime_as_dedicated_group_and_denies_nonmember():
    with tempfile.TemporaryDirectory(prefix="stage9-dac-") as scratch:
        directory = Path(scratch)
        source, view = directory / "source", directory / "staging" / "root"
        receipt, approval, target = _stage(source, view)
        document = _derive(receipt, approval)
        for path in [target, *target.rglob("*")]:
            os.chown(path, UID, GID)
        runtime.apply_tree(view, document, _pin(document), vars(contract))
        path = Path(receipt["root"]) / f"packages/{CODE}"
        for group, allowed in [(GID, True), (GID + 1, False)]:
            child = os.fork()
            if child == 0:
                status = 1
                try:
                    os.chroot(view)
                    os.chdir("/")
                    os.setgroups([])
                    os.setgid(group)
                    os.setuid(UID)
                    try:
                        content, info = _read_protected(path, 4096)
                    except PermissionError:
                        status = 1 if allowed else 0
                    else:
                        status = int(
                            not allowed
                            or content != b"synthetic controller"
                            or info.st_uid != 0
                            or os.access(path, os.W_OK)
                            or os.access(path.parent, os.W_OK)
                        )
                finally:
                    os._exit(status)
            assert os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]) == 0
