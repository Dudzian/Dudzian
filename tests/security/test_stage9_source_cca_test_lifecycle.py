"""Unprivileged lifecycle simulations: no native service or account is changed."""

from contextlib import nullcontext
import copy
import os
from pathlib import Path
import signal
import stat
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_source_cca_test_runtime as runtime
from deployment.stage9_source_cca_test_package import HarnessRefusal, plan

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-only systemd, pidfd and protected filesystem lifecycle"
)


def unused():
    return {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "FragmentPath": "",
        "DropInPaths": "",
        "Transient": "no",
        "InvocationID": "",
        "MainPID": "0",
        "ControlGroup": "",
    }


def scoped_observation(*, pids=(), uncertain=False):
    return {
        "scope": "VISIBLE_UID_GID_CGROUP",
        "state": "UNCERTAIN"
        if uncertain
        else "RESIDUAL_PROCESS_FOUND"
        if pids
        else "SCOPED_ABSENCE_OBSERVED",
        "matched_pids": list(pids),
        "uncertain_pids": [1] if uncertain else [],
        "uncertainty": ["PROCESS_OBSERVATION_UNAVAILABLE"] if uncertain else [],
        "complete": not uncertain,
        "full_confirmation": False,
        "executable_scan_complete": False,
        "excluded_processes": 1,
    }


@pytest.fixture
def local(monkeypatch, tmp_path):
    resources = plan("012345abcdef")
    resources["root"] = str(tmp_path / "owned")
    system = tmp_path / "system"
    system.mkdir()
    for role in resources["roles"]:
        role["executable"] = str(Path(resources["root"]) / role["role"])
        role["unit_path"] = str(system / role["unit"])
    files = {
        "manifest.json": b"manifest",
        "TEST_ONLY-pin.json": b"pin",
        "resources.json": b"resources",
    }
    for role in resources["roles"]:
        files[role["role"]] = b"harmless ELF fixture"
        files[role["role"] + ".service"] = b"owned test unit"
    package = {"plan": resources, "commit": "1" * 40, "files": files, "package_sha256": "a" * 64}
    users, groups, commands = {}, {}, []

    def nss(database, key=None):
        values = users if database == "passwd" else groups
        if key is None:
            return list(values.values())
        return [value for name, value in values.items() if name == key or value[2] == key]

    def command(name, *arguments, **kwargs):
        commands.append((name, arguments))
        if name == "groupadd":
            principal = arguments[-1]
            groups[principal] = [principal, "x", arguments[arguments.index("--gid") + 1], ""]
        elif name == "useradd":
            principal = arguments[-1]
            users[principal] = [
                principal,
                "x",
                arguments[arguments.index("--uid") + 1],
                arguments[arguments.index("--gid") + 1],
                "CH S9 TEST ONLY " + resources["run_id"],
                "/nonexistent",
                "/usr/sbin/nologin",
            ]
        elif name == "userdel":
            del users[arguments[0]]
        elif name == "groupdel":
            del groups[arguments[0]]
        return 0, ""

    def protected(path, *, directory):
        info = path.lstat()
        valid_kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if (
            not valid_kind
            or (info.st_mode & 0o022 and path != Path("/tmp"))
            or (not directory and info.st_nlink != 1)
        ):
            raise HarnessRefusal("UNPROTECTED_HOST_PATH")
        return info

    monkeypatch.setattr(runtime, "_nss", nss)
    monkeypatch.setattr(runtime, "_command", command)
    monkeypatch.setattr(runtime, "_protected", protected)
    monkeypatch.setattr(runtime, "_defer_interrupts", nullcontext)

    def administrative_preflight(_resources, *, administrative):
        assert administrative is True
        return {"ready": True}

    def administrative_inspection(_resources, *, administrative):
        assert administrative is True
        clean = not users and not groups and not Path(resources["root"]).exists()
        return {"clean": clean, "full_confirmation": clean, "checks": {}, "blockers": []}

    monkeypatch.setattr(runtime, "preflight", administrative_preflight)
    monkeypatch.setattr(runtime, "inspect_unit", lambda _: unused())
    monkeypatch.setattr(runtime, "_processes", lambda _: [])
    monkeypatch.setattr(
        runtime,
        "inspect_cleanup",
        administrative_inspection,
    )
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    return SimpleNamespace(
        run=runtime.PhysicalRun(package),
        package=package,
        users=users,
        groups=groups,
        commands=commands,
        nss=nss,
        protected=protected,
    )


def test_installs_only_planned_resources_and_cleans_every_receipt(local):
    with local.run as run:
        assert len(local.users) == len(local.groups) == 3
        assert len(run.receipts) == 12
        assert run.manifest_path.read_bytes() == b"manifest"
        assert all(
            Path(role["executable"]).read_bytes() == b"harmless ELF fixture"
            for role in run.resources
        )
        journal = run.root / "registry.json"
        assert b'"TEST_ONLY": true' in journal.read_bytes()
        assert not run.pending
    assert run.cleanup_result["clean"] is True
    assert not run.root.exists()
    assert run.cleanup() == run.cleanup_result
    assert not local.users and not local.groups


def test_preflight_collision_prevents_any_mutation(local, monkeypatch):
    monkeypatch.setattr(runtime, "preflight", lambda _, *, administrative: {"ready": False})
    with pytest.raises(HarnessRefusal, match="PREFLIGHT_BLOCKED"):
        local.run.__enter__()
    assert not local.commands and not local.run.root.exists()


def test_root_requirement_precedes_preflight_and_mutation(local, monkeypatch):
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 1000)
    with pytest.raises(HarnessRefusal, match="ROOT_EXECUTION_REQUIRED"):
        local.run.__enter__()
    assert not local.commands


def test_existing_root_is_not_adopted_or_deleted(local):
    local.run.root.mkdir()
    sentinel = local.run.root / "foreign"
    sentinel.write_text("retain")
    with pytest.raises(FileExistsError):
        local.run.__enter__()
    assert sentinel.read_text() == "retain"
    assert local.run.cleanup_result["clean"] is False


def test_unknown_descendant_blocks_cleanup_and_retains_journal(local):
    run = local.run.__enter__()
    unknown = run.root / "unknown"
    unknown.write_text("foreign")
    result = run.cleanup()
    assert result["clean"] is False
    assert unknown.read_text() == "foreign"
    assert (run.root / "registry.json").is_file()
    assert result == run.cleanup()


def test_symlink_replacement_is_retained_without_touching_target(local, tmp_path):
    run = local.run.__enter__()
    executable = Path(run.resources[0]["executable"])
    executable.unlink()
    target = tmp_path / "foreign"
    target.write_text("untouched")
    executable.symlink_to(target)
    assert run.cleanup()["clean"] is False
    assert executable.is_symlink() and target.read_text() == "untouched"


def test_unsafe_hardlink_blocks_mutation(local, tmp_path):
    with local.run as run:
        manifest = run.manifest_path
        hardlink = tmp_path / "foreign-link"
        hardlink.hardlink_to(manifest)
        try:
            with pytest.raises(HarnessRefusal, match="UNPROTECTED_HOST_PATH"):
                run.write_owned(manifest, b"changed")
            assert manifest.read_bytes() == b"manifest"
        finally:
            hardlink.unlink()
    assert run.cleanup_result["clean"] is True


def test_owned_write_rejects_unrecorded_or_changed_inode(local):
    with local.run as run:
        with pytest.raises(HarnessRefusal, match="UNOWNED_RESOURCE"):
            run.write_owned(run.root / "unowned", b"bad")
        manifest = run.manifest_path
        original = manifest.read_bytes()
        run.write_owned(manifest, b"temporary")
        assert manifest.read_bytes() == b"temporary"
        run.write_owned(manifest, original)


def test_replacement_retains_original_inode_and_restores_after_exception(local):
    with local.run as run:
        path = Path(run.resources[0]["executable"])
        inode = path.stat().st_ino
        with pytest.raises(RuntimeError, match="probe"):
            with run.replace_owned(path, replacement=b"different executable"):
                assert path.read_bytes() == b"different executable"
                assert path.stat().st_ino != inode
                raise RuntimeError("probe")
        assert path.read_bytes() == b"harmless ELF fixture" and path.stat().st_ino == inode
        assert not run.pending


def test_symlink_probe_restores_original_without_following_target(local, monkeypatch):
    original_lstat = Path.lstat

    def root_owned_symlink(path):
        info = original_lstat(path)
        if stat.S_ISLNK(info.st_mode):
            fields = list(info)
            fields[4] = fields[5] = 0
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", root_owned_symlink)
    with local.run as run:
        path = Path(run.resources[0]["executable"])
        inode = path.stat().st_ino
        with run.replace_owned(path, symlink_target=run.resources[1]["executable"]):
            assert path.is_symlink()
        assert not path.is_symlink() and path.stat().st_ino == inode


def test_dropin_probe_is_scoped_removed_and_reloaded(local):
    with local.run as run:
        role = run.resources[0]
        dropin = Path(role["unit_path"] + ".d")
        with run.test_dropin(role, b"[Service]\nEnvironment=TEST_ONLY=yes\n"):
            assert (dropin / "test-only.conf").is_file()
        assert not dropin.exists()
        assert not run.pending
    reloads = [
        command for command in local.commands if command == ("systemctl", ("daemon-reload",))
    ]
    assert len(reloads) >= 4


def test_unknown_dropin_directory_is_never_adopted(local):
    run = local.run.__enter__()
    directory = Path(run.resources[0]["unit_path"] + ".d")
    directory.mkdir()
    sentinel = directory / "foreign.conf"
    sentinel.write_text("retain")
    with pytest.raises(FileExistsError):
        with run.test_dropin(run.resources[0], b"unused"):
            pytest.fail("foreign dropin must block")
    assert run.cleanup()["clean"] is False
    assert sentinel.read_text() == "retain"


def test_account_replacement_is_retained_and_blocks_cleanup(local):
    run = local.run.__enter__()
    principal = run.resources[0]["principal"]
    local.users[principal] = [principal, "x", "64000", "64000", "foreign", "/foreign", "/bin/bash"]
    assert run.cleanup()["clean"] is False
    assert local.users[principal][4] == "foreign"
    assert principal in local.groups


def test_unapproved_supplementary_membership_blocks_account_removal(local):
    run = local.run.__enter__()
    principal = run.resources[0]["principal"]
    local.groups["foreign"] = ["foreign", "x", "17", principal]
    assert run.cleanup()["clean"] is False
    assert principal in local.users


def test_pending_creation_after_command_failure_cannot_authorize_adoption(local, monkeypatch):
    original = runtime._command

    def fail(name, *args, **kwargs):
        value = original(name, *args, **kwargs)
        if name == "groupadd":
            raise RuntimeError("native data must not appear")
        return value

    monkeypatch.setattr(runtime, "_command", fail)
    with pytest.raises(RuntimeError):
        local.run.__enter__()
    assert local.run.cleanup_result["clean"] is False
    assert local.groups  # Side effect before receipt: retained, never adopted.
    assert "UNRESOLVED_CREATION_OR_MUTATION" in local.run.cleanup_result["blockers"]
    assert "native data" not in str(local.run.cleanup_result)


def owned_snapshot(run, role, pid="1234"):
    return {
        "Id": role["unit"],
        "FragmentPath": role["unit_path"],
        "DropInPaths": "",
        "Transient": "no",
        "Description": "Stage9 SourceCCA TEST ONLY " + run.plan["run_id"] + " " + role["role"],
        "InvocationID": "b" * 32,
        "MainPID": pid,
        "ControlGroup": role["control_group"],
        "ActiveState": "active",
        "SubState": "running",
        "Type": "exec",
    }


def test_stop_signals_only_pinned_descriptor_with_original_lifetime(local, monkeypatch):
    run = local.run.__enter__()
    role = run.resources[0]
    name = role["role"]
    snapshot = owned_snapshot(run, role)
    run.pins[name] = snapshot
    run.handles[name] = 77
    run.lifetimes[name] = {"pid": 1234, "start_time": 98, "control_group": role["control_group"]}
    ended = {**snapshot, "MainPID": "0", "ActiveState": "inactive"}
    observations = iter((snapshot, ended))
    monkeypatch.setattr(runtime, "inspect_unit", lambda _: next(observations))
    monkeypatch.setattr(runtime, "_process_lifetime", lambda _: dict(run.lifetimes[name]))
    checks = iter([False, True])
    monkeypatch.setattr(runtime, "_process_finished", lambda *args: next(checks))
    signals = []
    monkeypatch.setattr(runtime.signal, "pidfd_send_signal", lambda *args: signals.append(args))
    run.stop(role)
    assert signals == [(77, signal.SIGTERM, None, 0)]
    run.handles.clear()
    run.cleanup()


@pytest.mark.parametrize("change", ["invocation", "pid", "starttime", "cgroup"])
def test_stop_never_signals_after_process_or_unit_replacement(local, monkeypatch, change):
    run = local.run.__enter__()
    role = run.resources[0]
    name = role["role"]
    snapshot = owned_snapshot(run, role)
    run.pins[name] = snapshot
    run.handles[name] = 77
    lifetime = {"pid": 1234, "start_time": 98, "control_group": role["control_group"]}
    run.lifetimes[name] = lifetime
    observed = copy.deepcopy(snapshot)
    current = dict(lifetime)
    if change == "invocation":
        observed["InvocationID"] = "c" * 32
    elif change == "pid":
        observed["MainPID"] = "1235"
    elif change == "starttime":
        current["start_time"] = 99
    else:
        current["control_group"] = "/foreign"
    monkeypatch.setattr(runtime, "inspect_unit", lambda _: observed)
    monkeypatch.setattr(runtime, "_process_lifetime", lambda _: current)
    monkeypatch.setattr(runtime, "_process_finished", lambda *args: False)
    monkeypatch.setattr(
        runtime.signal,
        "pidfd_send_signal",
        lambda *args: pytest.fail("must not signal replacement"),
    )
    with pytest.raises(HarnessRefusal, match="PROCESS_IDENTITY_CHANGED"):
        run.stop(role)
    run.handles.clear()
    run.cleanup()


def test_inspector_requires_nss_units_paths_and_process_absence(monkeypatch):
    resources = plan("012345abcdef")
    monkeypatch.setattr(
        runtime, "_collisions", lambda _: {"users_absent": True, "units_absent": True}
    )
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    monkeypatch.setattr(runtime, "rootless_process_observation", lambda _: scoped_observation())
    monkeypatch.setattr(runtime, "_processes", lambda _: [])
    administrative = runtime.inspect_cleanup(resources, administrative=True)
    assert administrative["clean"] is True
    assert administrative["full_confirmation"] is True
    rootless = runtime.inspect_cleanup(resources)
    assert rootless["clean"] is False
    assert rootless["scope_clear"] is True
    assert rootless["full_confirmation"] is False
    monkeypatch.setattr(runtime, "_processes", lambda _: [1234])
    assert runtime.inspect_cleanup(resources, administrative=True)["clean"] is False
    monkeypatch.setattr(
        runtime, "_processes", lambda _: (_ for _ in ()).throw(PermissionError("native"))
    )
    result = runtime.inspect_cleanup(resources, administrative=True)
    assert result["clean"] is False
    assert "native" not in str(result)


def test_collision_checks_include_numeric_ids_groups_dropins_and_cgroups(monkeypatch):
    resources = plan("012345abcdef")
    lookups, paths, units = [], [], []
    monkeypatch.setattr(runtime, "_nss", lambda db, key: lookups.append((db, key)) or [])
    monkeypatch.setattr(runtime, "_protected", lambda *args, **kwargs: True)
    monkeypatch.setattr(runtime, "_absent", lambda path: paths.append(str(path)) or True)
    monkeypatch.setattr(runtime, "inspect_unit", lambda name: units.append(name) or unused())
    result = runtime._collisions(resources)
    assert all(result.values())
    assert len(lookups) == 12 and len(units) == 3
    assert ("passwd", "62000") in lookups and ("group", "62002") in lookups
    assert any(path.endswith(".service.d") for path in paths)
    assert any(path.startswith("/sys/fs/cgroup/system.slice/") for path in paths)


def test_changed_contents_in_same_inode_are_never_deleted(local):
    run = local.run.__enter__()
    executable = Path(run.resources[0]["executable"])
    inode = executable.stat().st_ino
    executable.write_bytes(b"unknown replacement contents")
    assert executable.stat().st_ino == inode
    result = run.cleanup()
    assert result["clean"] is False
    assert executable.read_bytes() == b"unknown replacement contents"
    assert (run.root / "registry.json").exists()


def test_cgroup_observation_unavailable_never_establishes_absence(monkeypatch):
    monkeypatch.setattr(
        runtime, "_protected", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError())
    )
    monkeypatch.setattr(runtime, "rootless_process_observation", lambda _: scoped_observation())
    result = runtime.inspect_cleanup(plan("012345abcdef"))
    assert result["clean"] is False
    assert "resource_observation" in result["blockers"]


def test_capture_closes_unproven_descriptor_on_process_identity_race(local, monkeypatch):
    run = local.run.__enter__()
    role = run.resources[0]
    run.started.add(role["role"])
    snapshot = owned_snapshot(run, role)
    monkeypatch.setattr(runtime, "inspect_unit", lambda _: snapshot)
    monkeypatch.setattr(runtime.os, "pidfd_open", lambda *args: 77)
    monkeypatch.setattr(runtime, "_process_finished", lambda *args: False)
    lifetimes = iter(
        [
            {"pid": 1234, "start_time": 1, "control_group": role["control_group"]},
            {"pid": 1234, "start_time": 2, "control_group": role["control_group"]},
        ]
    )
    monkeypatch.setattr(runtime, "_process_lifetime", lambda _: next(lifetimes))
    closed = []
    real_close = runtime.os.close
    monkeypatch.setattr(
        runtime.os,
        "close",
        lambda descriptor: closed.append(descriptor)
        if descriptor == 77
        else real_close(descriptor),
    )
    with pytest.raises(HarnessRefusal, match="PROCESS_IDENTITY_CHANGED"):
        run._capture(role)
    assert closed == [77] and not run.handles and not run.pins
    run.started.clear()
    run.cleanup()


def test_capture_pins_live_descriptor_lifetime_and_exact_unit_invocation(local, monkeypatch):
    run = local.run.__enter__()
    role = run.resources[0]
    name = role["role"]
    run.started.add(name)
    snapshot = owned_snapshot(run, role)
    lifetime = {"pid": 1234, "start_time": 98, "control_group": role["control_group"]}
    monkeypatch.setattr(runtime, "inspect_unit", lambda _: snapshot)
    monkeypatch.setattr(runtime.os, "pidfd_open", lambda *args: 77)
    monkeypatch.setattr(runtime, "_process_finished", lambda *args: False)
    monkeypatch.setattr(runtime, "_process_lifetime", lambda _: dict(lifetime))
    assert run._capture(role) is True
    assert (
        run.handles[name] == 77 and run.lifetimes[name] == lifetime and run.pins[name] == snapshot
    )
    run.handles.clear()
    run.started.clear()
    run.cleanup()


def test_cleanup_after_failed_preflight_performs_only_independent_inspection(local, monkeypatch):
    monkeypatch.setattr(runtime, "preflight", lambda _, *, administrative: {"ready": False})
    with pytest.raises(HarnessRefusal, match="PREFLIGHT_BLOCKED"):
        local.run.__enter__()
    result = local.run.cleanup()
    assert result["clean"] is True
    assert not local.commands
    assert result == local.run.cleanup()


def test_useradd_disables_untracked_spool_and_subordinate_id_allocations(local):
    with local.run:
        useradds = [args for command, args in local.commands if command == "useradd"]
        assert len(useradds) == 3
        assert all("--system" in args for args in useradds)
        assert all(
            "CREATE_MAIL_SPOOL=no" in args
            and "SUB_UID_COUNT=0" in args
            and "SUB_GID_COUNT=0" in args
            for args in useradds
        )


def test_subordinate_id_leftovers_are_independently_blocked(monkeypatch):
    original = Path.read_text
    resources = plan("012345abcdef")
    principal = resources["roles"][0]["principal"]
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, **kwargs: principal + ":100000:65536\n"
        if str(path) == "/etc/subuid"
        else original(path, **kwargs),
    )
    assert runtime._subordinate_absent(resources) is False


def test_rootless_preflight_only_observes_host_and_reports_missing_facts(monkeypatch):
    resources = plan("012345abcdef")
    commands = []
    real_text = Path.read_text
    texts = {
        "/proc/1/comm": "systemd\n",
        "/sys/class/dmi/id/sys_vendor": "Dell Inc.\n",
        "/sys/class/dmi/id/product_name": "Latitude 5490\n",
    }
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        runtime.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"}
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, **kwargs: texts[str(path)]
        if str(path) in texts
        else real_text(path, **kwargs),
    )
    monkeypatch.setattr(runtime, "_program", lambda name: Path("/usr/bin") / name)
    monkeypatch.setattr(runtime, "_protected", lambda *args, **kwargs: True)
    monkeypatch.setattr(runtime, "_collisions", lambda _: {"names_unused": True})
    monkeypatch.setattr(runtime, "_nss", lambda *args: [])
    monkeypatch.setattr(runtime, "_absent", lambda _: True)
    monkeypatch.setattr(Path, "glob", lambda *args: iter([]))
    monkeypatch.setattr(
        runtime, "_command", lambda name, *args: commands.append((name, args)) or (0, "")
    )
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda argv, **kwargs: commands.append(tuple(argv)) or SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(
        runtime, "rootless_process_observation", lambda _: scoped_observation(uncertain=True)
    )
    monkeypatch.setattr(
        runtime,
        "_processes",
        lambda _: pytest.fail("rootless mode must not run the full exe sweep"),
    )
    result = runtime.preflight(resources)
    assert result["ready"] is False and result["read_only"] is True
    assert result["checks"]["dell_latitude_5490"] is True
    assert "no_test_processes" in result["blockers"]
    assert "unreadable exe" not in str(result)
    assert all(
        "useradd" not in str(command)
        and "start" not in str(command)
        and "daemon-reload" not in str(command)
        for command in commands
    )


def test_live_and_completed_run_instances_cannot_be_reused(local):
    with local.run as run:
        with pytest.raises(HarnessRefusal, match="RUN_INSTANCE_REUSE_FORBIDDEN"):
            run.__enter__()
    commands = list(local.commands)
    with pytest.raises(HarnessRefusal, match="RUN_INSTANCE_REUSE_FORBIDDEN"):
        run.__enter__()
    assert local.commands == commands
    assert not run.root.exists()
