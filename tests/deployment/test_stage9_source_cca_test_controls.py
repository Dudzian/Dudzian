"""Simulated controls exercise the real gate without privileged host mutation."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from deployment import stage9_source_cca_test_controls as controls
from deployment import stage9_trusted_installation_identity_gate as gate
from deployment.stage9_source_cca_test_package import HarnessRefusal
from deployment.stage9_trusted_installation_contract import InstallationRefusal, parse_manifest
from deployment.stage9_trusted_installation_files import FileRefusal, PinnedFile
from deployment.stage9_trusted_installation_runtime import RuntimeRefusal

_ROLES = ("root_source", "history_source", "cca")


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class SimulatedRun:
    def __init__(self):
        self.manifest_path = "/run/test/manifest.json"
        self.pin_path = "/run/test/pin.json"
        self.resources = []
        self.files = {}
        self.modes = {}
        self.symlinks = set()
        self.dropins = set()
        self.stopped = set()
        self.writes = []
        self.fail_write = None
        roles = []
        for index, name in enumerate(_ROLES):
            unit = f"test-{name.replace('_', '-')}.service"
            executable = f"/run/test/{name}"
            fragment = f"/run/systemd/system/{unit}"
            self.files[executable] = f"simulated ELF {name}".encode()
            self.files[fragment] = f"simulated unit {name}".encode()
            self.modes[executable] = 0o755
            self.modes[fragment] = 0o644
            resource = {
                "role": name,
                "unit": unit,
                "uid": 62000 + index,
                "gid": 62000 + index,
                "executable": executable,
                "unit_path": fragment,
                "control_group": "/system.slice/" + unit,
            }
            self.resources.append(resource)

            def expectation(path):
                return {
                    "path": path,
                    "sha256": hashlib.sha256(self.files[path]).hexdigest(),
                    "uid": 0,
                    "gid": 0,
                    "mode": self.modes[path],
                }

            roles.append(
                {
                    "role": name,
                    "installation_identity": "TEST_ONLY-" + name,
                    "unit": unit,
                    "principal": "test_" + name,
                    "uid": resource["uid"],
                    "gid": resource["gid"],
                    "supplementary_groups": [],
                    "executable": expectation(executable),
                    "unit_file": expectation(fragment),
                    "control_group": resource["control_group"],
                }
            )
        manifest = {
            "schema": "CryptoHunter.SourceCCAInstallationV1",
            "version": 1,
            "installation_id": "TEST_ONLY-SYNTHETIC-control",
            "roles": roles,
        }
        self.files[self.manifest_path] = _json(manifest)
        self.files[self.pin_path] = _json(
            {
                "schema": "CryptoHunter.SourceCCAInstallationApprovalPinV1",
                "version": 1,
                "installation_id": manifest["installation_id"],
                "approval_id": "TEST_ONLY-NOT-INDEPENDENT",
                "manifest_sha256": hashlib.sha256(self.files[self.manifest_path]).hexdigest(),
            }
        )
        self.modes[self.manifest_path] = self.modes[self.pin_path] = 0o600

    def assert_owned_file(self, path):
        if path not in self.files:
            raise HarnessRefusal("CONTROL_RUNTIME_INVALID")

    def write_owned(self, path, data, mode=None):
        self.assert_owned_file(path)
        self.writes.append(path)
        if self.fail_write == path:
            raise HarnessRefusal("CONTROL_RESTORE_FAILED")
        self.files[path] = data
        if mode is not None:
            self.modes[path] = mode

    def chmod_owned(self, path, mode):
        self.assert_owned_file(path)
        self.modes[path] = mode

    @contextmanager
    def replace_owned(self, path, replacement=None, symlink_target=None):
        original = self.files[path]
        try:
            if symlink_target is not None:
                self.symlinks.add(path)
            else:
                self.files[path] = replacement
            yield
        finally:
            self.symlinks.discard(path)
            self.files[path] = original

    @contextmanager
    def test_dropin(self, role, data):
        self.dropins.add(role["role"])
        try:
            yield
        finally:
            self.dropins.remove(role["role"])

    def stop(self, role):
        self.stopped.add(role["role"])

    def start_role(self, role):
        self.stopped.remove(role["role"])


@pytest.fixture
def observation(monkeypatch):
    run = SimulatedRun()

    class File:
        def __init__(self, path, **kwargs):
            if path in run.symlinks or run.modes[path] & 0o022:
                raise FileRefusal("FILE_METADATA_UNSAFE")
            if kwargs.get("mode", run.modes[path]) != run.modes[path]:
                raise FileRefusal("FILE_METADATA_UNSAFE")
            self.path = path
            self.fd = 17
            self.data = run.files[path]
            self.sha256 = hashlib.sha256(self.data).hexdigest()
            self.identity = (1, int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big"))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def recheck(self):
            if self.data != run.files[self.path]:
                raise FileRefusal("FILE_CHANGED_DURING_CHECK")

    def service(role, _descriptor):
        resource = next(item for item in run.resources if item["role"] == role.role)
        if role.role in run.dropins:
            raise RuntimeRefusal("UNEXPECTED_SYSTEMD_DROPINS")
        if role.role in run.stopped or role.control_group != resource["control_group"]:
            raise RuntimeRefusal("SYSTEMD_IDENTITY_MISMATCH")
        if (role.uid, role.gid) != (resource["uid"], resource["gid"]):
            raise RuntimeRefusal("NSS_IDENTITY_MISMATCH")
        index = _ROLES.index(role.role)
        return {
            "pid": 2000 + index,
            "start_time_ticks": 3000 + index,
            "uid": role.uid,
            "gid": role.gid,
            "groups": [role.gid],
            "namespaces": {"mnt": 4000 + index, "ipc": 5000 + index, "net": 6000 + index},
        }

    monkeypatch.setattr(gate, "PinnedFile", File)
    monkeypatch.setattr(controls, "PinnedFile", File)
    monkeypatch.setattr(gate, "observe_service", service)
    monkeypatch.setattr(controls, "observe_service", service)
    return run


def test_thirteen_controls_use_real_gate_and_restore_every_simulated_resource(observation):
    run = observation
    originals = copy.deepcopy((run.files, run.modes))
    reports = controls.run_controls(run)
    assert len(reports) == 13
    assert len({item["name"] for item in reports}) == 13
    assert (run.files, run.modes) == originals
    assert not run.stopped
    assert not run.dropins
    assert not run.symlinks
    for report in reports:
        assert report["result"] == "REFUSED"
        assert report["positive_before"] == report["positive_after"] == "MATCHED"
        assert report["gate"]["observed_failures"]
        assert report["gate"]["status"] == "BLOCKED"
        assert report["gate"]["qualification"] == "NOT_QUALIFIED"
        assert report["gate"]["production_emission"] == "BLOCKED"
        assert report["negative_report_sha256"] == controls.report_sha256(report["gate"])
        assert len(report["positive_before_sha256"]) == 64
        assert len(report["positive_after_sha256"]) == 64
        if report["evidence"] == "OBSERVATION_FAULT_TEST":
            assert report["gate"]["snapshot"] == "SYNTHETIC_TEST_INPUT"
        else:
            assert report["gate"]["snapshot"] == "UNTRUSTED_LIVE_OBSERVATION"
    controls.require_matched(
        gate.inspect_installation(run.manifest_path, approval_pin_path=run.pin_path)
    )


def test_runtime_path_objects_are_converted_at_gate_boundary(observation):
    # This fixture describes a Linux installation even on a Windows test host.
    # Native WindowsPath would stringify with backslashes and violate that
    # contract; conversion at the boundary must not normalize unsafe paths.
    observation.manifest_path = PurePosixPath(observation.manifest_path)
    observation.pin_path = PurePosixPath(observation.pin_path)
    reports = controls.run_controls(observation)
    assert len(reports) == 13


@pytest.mark.parametrize(
    "path",
    [
        PureWindowsPath("/run/test/root_source"),
        PureWindowsPath("C:/run/test/root_source"),
        PureWindowsPath("//host/share/root_source"),
    ],
    ids=["windows-rooted", "windows-drive", "windows-unc"],
)
def test_windows_path_spellings_remain_invalid_linux_installation_paths(observation, path):
    document = json.loads(observation.files[observation.manifest_path])
    document["roles"][0]["executable"]["path"] = str(path)
    with pytest.raises(InstallationRefusal, match="^MANIFEST_PATH_INVALID$"):
        parse_manifest(_json(document))
    # Exercise the real file argument validator, before any host filesystem
    # access. The synthetic file backend must not weaken this boundary.
    with pytest.raises(FileRefusal, match="^FILE_PATH_UNSAFE$"):
        PinnedFile(str(path))


@pytest.mark.parametrize(
    "stimulus", ["pid_starttime", "shared_namespace", "incomplete_observation"]
)
def test_injected_observations_require_live_observer_success_first(
    observation, monkeypatch, stimulus
):
    calls = []

    def unavailable(*_args):
        calls.append(True)
        raise RuntimeRefusal("PROCESS_UNAVAILABLE")

    monkeypatch.setattr(controls, "observe_service", unavailable)
    result = controls._injected_report(observation, stimulus)
    assert calls
    assert result["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert result["observed_failures"] == ["PROCESS_UNAVAILABLE"]
    assert result["installation_checks"] != "MATCHED"


def test_failed_positive_never_starts_mutation(observation):
    observation.stopped.add("root_source")
    with pytest.raises(HarnessRefusal, match="CONTROL_POSITIVE_FAILED"):
        controls.run_controls(observation)
    assert observation.writes == []
    assert not observation.dropins


def test_wrong_negative_refusal_blocks_result_and_still_reads_restored_baseline(observation):
    with pytest.raises(HarnessRefusal, match="CONTROL_NEGATIVE_FAILED"):
        controls._evaluate(
            observation,
            "permissions",
            "PHYSICAL_FILE_FAULT_TEST",
            frozenset({"UNEXPECTED_SYSTEMD_DROPINS"}),
            lambda: controls._permission_fault(observation, observation.resources[0]),
        )
    assert observation.modes[observation.resources[0]["executable"]] == 0o755
    controls.require_matched(
        gate.inspect_installation(observation.manifest_path, approval_pin_path=observation.pin_path)
    )


def test_matching_negative_cannot_count_as_refusal(observation):
    @contextmanager
    def no_fault():
        yield

    with pytest.raises(HarnessRefusal, match="CONTROL_NEGATIVE_FAILED"):
        controls._evaluate(
            observation, "no_fault", "PHYSICAL_FILE_FAULT_TEST", frozenset({"IGNORED"}), no_fault
        )


def test_fault_exception_restores_exact_manifest_and_pin(observation):
    before = copy.deepcopy(observation.files)
    with pytest.raises(RuntimeError, match="controlled failure"):
        with controls._expectation_fault(
            observation, lambda document: document.update(installation_id="TEST_ONLY-changed")
        ):
            raise RuntimeError("controlled failure")
    assert observation.files == before


def test_restore_pin_attempted_even_when_manifest_restore_fails(observation):
    before = observation.files[observation.pin_path]
    with pytest.raises(HarnessRefusal, match="CONTROL_RESTORE_FAILED"):
        with controls._expectation_fault(
            observation, lambda document: document.update(installation_id="TEST_ONLY-changed")
        ):
            observation.fail_write = observation.manifest_path
    assert observation.files[observation.pin_path] == before
    assert observation.writes[-1] == observation.pin_path


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "READY"),
        ("qualification", "QUALIFIED"),
        ("production_emission", "ALLOWED"),
        ("snapshot", "SYNTHETIC_TEST_INPUT"),
        ("role_checks", {}),
        ("observed_failures", ["PROCESS_UNAVAILABLE"]),
        ("blockers", []),
    ],
)
def test_positive_contract_cannot_be_weakened(observation, field, value):
    report = gate.inspect_installation(
        observation.manifest_path, approval_pin_path=observation.pin_path
    )
    report[field] = value
    with pytest.raises(HarnessRefusal, match="CONTROL_POSITIVE_FAILED"):
        controls.require_matched(report)


def test_runtime_role_inventory_must_be_complete(observation):
    observation.resources.pop()
    with pytest.raises(HarnessRefusal, match="CONTROL_RUNTIME_INVALID"):
        controls.run_controls(observation)


def test_report_digest_is_order_independent_and_rejects_nonfinite_numbers():
    assert controls.report_sha256({"a": 1, "b": 2}) == controls.report_sha256({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        controls.report_sha256({"invalid": float("nan")})
