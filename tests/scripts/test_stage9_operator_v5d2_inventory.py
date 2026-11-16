"""Hermetic inventory regressions; synthetic metadata never qualifies a host.

The command boundary models read-only host observations. No test accesses the
operator's trust data, protected interpreter, actual libvirt or existing VMs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stage9_operator_v5d2_inventory.sh"
LIBVIRT_OPT_IN = "--probe-libvirt-after-local-auth-review"
INSTALLATION = "/opt/cryptohunter/observer-v5d1"
POLICY = "/etc/cryptohunter/observer-v5d1"
ANCHOR = f"{POLICY}/operator-anchors.json"
APPROVAL = f"{POLICY}/approved-policy.json"
REQUIREMENTS = f"{POLICY}/domain-requirements.json"
INTERPRETER = f"{INSTALLATION}/venv/bin/python"
BASH = shutil.which("bash")
REAL_ENV = shutil.which("env")
REAL_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(
    os.name != "posix" or BASH is None or REAL_ENV is None,
    reason="The fake executable harness requires POSIX shebangs, Bash and env",
)

# The fake tools intentionally write only the test's command log. Any inventory
# command outside this allowlist fails and is retained for the assertions below.
_TOOL = r"""
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
scenario = json.loads(Path(__SCENARIO_PATH__).read_text())
observed_environment = {
    key: value for key, value in os.environ.items()
    if key.startswith(("GIT_", "LIBVIRT_")) or key in {"VIRSH_HISTFILE", "HOME"}
}
with Path(__COMMAND_LOG_PATH__).open("a") as log:
    log.write(json.dumps([name, args, observed_environment]) + "\n")
if name == "git":
    for key in ("GIT_TRACE", "GIT_TRACE2_EVENT", "GIT_TRACE2_PERF"):
        value = os.environ.get(key)
        if value and value not in {"0", "1"}:
            Path(value).write_text("synthetic inherited Git logging side effect")
if name == "virsh":
    value = os.environ.get("LIBVIRT_LOG_OUTPUTS", "")
    if value.startswith("1:file:"):
        Path(value.removeprefix("1:file:")).write_text("synthetic inherited libvirt log")
    if os.environ.get("LIBVIRT_AUTH_FILE") != "/dev/null":
        sys.exit(99)
if name == "uname":
    if "-s" in args:
        print(scenario.get("system", "Linux"))
    elif "-m" in args:
        print(scenario.get("machine", "x86_64"))
    elif "-r" in args:
        print("synthetic-kernel")
    else:
        print("Linux synthetic-host synthetic-kernel x86_64")
elif name == "stat":
    if "-L" in args or "--dereference" in args:
        sys.exit(99)
    path = args[-1]
    value = scenario.get("paths", {}).get(path)
    if isinstance(value, str):
        print("stat: " + value, file=sys.stderr)
        sys.exit(1)
    directories = {
        "/", "/opt", "/opt/cryptohunter", "/opt/cryptohunter/observer-v5d1",
        "/opt/cryptohunter/observer-v5d1/venv",
        "/opt/cryptohunter/observer-v5d1/venv/bin",
        "/opt/cryptohunter/observer-v5d1/deployment",
        "/opt/cryptohunter/observer-v5d1/scripts",
        "/etc", "/etc/cryptohunter", "/etc/cryptohunter/observer-v5d1",
        "/usr", "/usr/bin", "/dev", "/sys", "/sys/fs", "/sys/fs/cgroup",
    }
    mode = 0o40755 if path in directories else 0o100644
    if path == "/dev/kvm":
        mode = 0o20660
    fields = {"mode": mode, "uid": 0, "gid": 0, "links": 1, "size": 20}
    fields.update(value or {})
    print(f'{fields["mode"]:x} {fields["uid"]} {fields["gid"]} '
          f'{fields["links"]} {fields["size"]}')
elif name == "git":
    if "rev-parse" in args:
        print(scenario.get("head", "1111111111111111111111111111111111111111"))
    elif "status" in args:
        print(scenario.get("git_status", ""), end="")
    else:
        sys.exit(99)
elif name == "timeout":
    if scenario.get("timeout"):
        sys.exit(124)
    if args[:3] != ["--signal=TERM", "--kill-after=2s", "10s"]:
        sys.exit(99)
    index = 3
    os.execv(str(Path(sys.argv[0]).parent / args[index]), args[index:])
elif name == "virsh":
    if args != ["--readonly", "--connect", "qemu:///system", "uri"]:
        sys.exit(99)
    if scenario.get("libvirt_unavailable"):
        print("synthetic libvirt rejection", file=sys.stderr)
        sys.exit(1)
    print(scenario.get("libvirt_uri", "qemu:///system"))
elif name == "dpkg-query":
    print("installed")
else:
    # QEMU and any interpreter are presence-only observations, never executed.
    sys.exit(99)
"""


class InventoryHost:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        self.scenario = root / "scenario.json"
        self.log = root / "commands.jsonl"
        self.repo = root / "repository"
        self.repo.mkdir()
        self.sentinel = self.repo / "retained-evidence"
        self.sentinel.write_text("existing evidence must remain unchanged\n")
        self.scenario.write_text("{}")
        self.log.write_text("")
        (self.bin / "env").symlink_to(str(REAL_ENV))
        for name in (
            "uname",
            "stat",
            "git",
            "timeout",
            "virsh",
            "qemu-img",
            "qemu-system-x86_64",
            "dpkg-query",
            "python",
            "python3",
            "sudo",
            "chmod",
            "chown",
            "mkdir",
            "rm",
            "touch",
            "tee",
            "cat",
        ):
            executable = self.bin / name
            wrapper = _TOOL.replace("__SCENARIO_PATH__", repr(str(self.scenario)))
            wrapper = wrapper.replace("__COMMAND_LOG_PATH__", repr(str(self.log)))
            executable.write_text(f"#!{sys.executable}\n{wrapper}")
            executable.chmod(0o755)

    def run(
        self,
        extra_environment: dict[str, str] | None = None,
        probe_libvirt: bool = False,
        arguments: list[str] | None = None,
        **scenario: Any,
    ) -> subprocess.CompletedProcess[str]:
        self.scenario.write_text(json.dumps(scenario))
        self.log.write_text("")
        env = {
            "PATH": str(self.bin),
            "LC_ALL": "C",
        }
        env.update(extra_environment or {})
        if arguments is None:
            arguments = ["--repository", str(self.repo)]
            if probe_libvirt:
                arguments.append(LIBVIRT_OPT_IN)
        return subprocess.run(
            [str(BASH), str(SCRIPT), *arguments],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def calls(self) -> list[tuple[str, list[str]]]:
        return [
            (name, args) for name, args, _ in map(json.loads, self.log.read_text().splitlines())
        ]

    def environments(self, tool: str) -> list[dict[str, str]]:
        return [
            environment
            for name, _, environment in map(json.loads, self.log.read_text().splitlines())
            if name == tool
        ]


@pytest.fixture
def host(tmp_path: Path) -> InventoryHost:
    return InventoryHost(tmp_path)


def _rows(result: subprocess.CompletedProcess[str]) -> dict[str, tuple[str, str]]:
    assert result.returncode == 0, result.stderr
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3 and fields[0] != "ITEM":
            rows[fields[0]] = (fields[1], fields[2])
    return rows


def _path_status(result: subprocess.CompletedProcess[str], path: str) -> str:
    rows = _rows(result)
    matching = [status for key, (status, _) in rows.items() if key == path]
    assert len(matching) == 1, result.stdout
    return matching[0]


def _assert_blocked_authorization(result: subprocess.CompletedProcess[str]) -> None:
    for invariant in (
        "Stage 9 = IN_PROGRESS",
        "Observer 3A = BLOCKED",
        "Stage 3B = NOT AUTHORIZED",
        "Production = NO-GO",
        "HOST_PREFLIGHT_VERIFIED = NOT_CLAIMED",
        "EXECUTION_AUTHORIZATION = NOT_AUTHORIZED",
    ):
        assert invariant in result.stdout
    assert "HOST_PREFLIGHT_VERIFIED = VERIFIED" not in result.stdout
    assert _rows(result)["repository_release_approval"] == (
        "UNVERIFIED",
        "INDEPENDENT_SOURCE_REVIEW_REQUIRED",
    )


def test_all_present_is_only_inventory_and_retains_independent_approval_gap(
    host: InventoryHost,
) -> None:
    result = host.run()
    assert _path_status(result, ANCHOR) == "PRESENT"
    assert _path_status(result, APPROVAL) == "PRESENT"
    assert _path_status(result, INTERPRETER) == "PRESENT"
    assert (
        _path_status(result, f"{INSTALLATION}/deployment/stage9_observer_kvm_preflight_v5d1.py")
        == "PRESENT"
    )
    _assert_blocked_authorization(result)
    assert any(
        "independent" in key.lower() and status in {"UNVERIFIED", "BLOCKED"}
        for key, (status, _) in _rows(result).items()
    ), result.stdout
    assert _rows(result)["repository_release_approval"] == (
        "UNVERIFIED",
        "INDEPENDENT_SOURCE_REVIEW_REQUIRED",
    )


def test_default_inventory_never_connects_to_libvirt_even_when_dependencies_exist(
    host: InventoryHost,
) -> None:
    result = host.run()
    assert _rows(result)["dependency:virsh"][0] == "PRESENT"
    assert _rows(result)["libvirt_readonly"] == (
        "NOT_TESTED",
        "LOCAL_AUTH_REVIEW_AND_EXPLICIT_OPT_IN_REQUIRED",
    )
    assert not any(name == "virsh" for name, _ in host.calls())
    assert not any(name == "timeout" and "virsh" in args for name, args in host.calls())
    assert not host.environments("virsh")
    _assert_blocked_authorization(result)


@pytest.mark.parametrize("arguments", [[], [LIBVIRT_OPT_IN]])
def test_optional_repository_argument_preserves_default_and_opted_in_modes(
    host: InventoryHost,
    arguments: list[str],
) -> None:
    result = host.run(arguments=arguments)
    expected = "PRESENT" if LIBVIRT_OPT_IN in arguments else "NOT_TESTED"
    assert _rows(result)["libvirt_readonly"][0] == expected
    assert any(name == "virsh" for name, _ in host.calls()) == (LIBVIRT_OPT_IN in arguments)
    _assert_blocked_authorization(result)


@pytest.mark.parametrize("flag_first", [False, True])
def test_explicit_libvirt_opt_in_accepts_both_argument_orders_without_approval(
    host: InventoryHost,
    flag_first: bool,
) -> None:
    arguments = ["--repository", str(host.repo)]
    if flag_first:
        arguments.insert(0, LIBVIRT_OPT_IN)
    else:
        arguments.append(LIBVIRT_OPT_IN)
    result = host.run(arguments=arguments)
    assert _rows(result)["libvirt_readonly"][0] == "PRESENT"
    assert [(name, args) for name, args in host.calls() if name == "virsh"] == [
        ("virsh", ["--readonly", "--connect", "qemu:///system", "uri"])
    ]
    assert _rows(result)["independent_approval"][0] == "UNVERIFIED"
    assert _rows(result)["repository_release_approval"][0] == "UNVERIFIED"
    _assert_blocked_authorization(result)


@pytest.mark.parametrize(
    "path", [INSTALLATION, INTERPRETER, POLICY, ANCHOR, APPROVAL, REQUIREMENTS]
)
def test_absent_installation_policy_and_each_trust_artifact_are_reported_missing(
    host: InventoryHost,
    path: str,
) -> None:
    result = host.run(paths={path: "No such file or directory"})
    assert _path_status(result, path) == "MISSING"
    _assert_blocked_authorization(result)
    if path == POLICY:
        observed = [args[-1] for name, args in host.calls() if name == "stat"]
        assert ANCHOR not in observed


@pytest.mark.parametrize("path", [POLICY, ANCHOR, INSTALLATION])
def test_permission_denied_is_inaccessible_and_never_absence(
    host: InventoryHost,
    path: str,
) -> None:
    result = host.run(paths={path: "Permission denied"})
    assert _path_status(result, path) == "INACCESSIBLE"
    assert "Permission denied" not in result.stdout
    _assert_blocked_authorization(result)


def test_unknown_metadata_failure_is_unverified_and_never_absence(host: InventoryHost) -> None:
    result = host.run(paths={ANCHOR: "Input/output error"})
    assert _path_status(result, ANCHOR) == "UNVERIFIED"
    _assert_blocked_authorization(result)


@pytest.mark.parametrize(
    ("path", "metadata"),
    [
        (POLICY, {"mode": stat.S_IFDIR | 0o777}),
        (ANCHOR, {"mode": stat.S_IFREG | 0o666}),
        ("/etc/cryptohunter", {"uid": 1000}),
        (APPROVAL, {"uid": 1000}),
        (ANCHOR, {"links": 2}),
        (REQUIREMENTS, {"mode": stat.S_IFIFO | 0o600}),
        (APPROVAL, {"mode": stat.S_IFDIR | 0o755}),
    ],
)
def test_unsafe_permissions_ownership_hardlinks_and_wrong_types_fail_closed(
    host: InventoryHost,
    path: str,
    metadata: dict[str, int],
) -> None:
    result = host.run(paths={path: metadata})
    rows = _rows(result)
    # Ancestor refusal may be attached to the requested leaf; it must prevent
    # every dependent policy result from becoming a protected-file observation.
    affected = path if path in rows else POLICY
    assert _path_status(result, affected) == "BLOCKED"
    _assert_blocked_authorization(result)


@pytest.mark.parametrize("path", ["/etc/cryptohunter", POLICY, ANCHOR, INTERPRETER])
def test_symlinked_trust_ancestor_leaf_or_interpreter_is_blocked_without_following(
    host: InventoryHost,
    path: str,
) -> None:
    result = host.run(paths={path: {"mode": stat.S_IFLNK | 0o777}})
    rows = _rows(result)
    affected = path if path in rows else POLICY
    assert _path_status(result, affected) == "BLOCKED"
    assert all(
        "-L" not in args and "--dereference" not in args
        for name, args in host.calls()
        if name == "stat"
    )
    _assert_blocked_authorization(result)


@pytest.mark.parametrize(
    "dependency", ["env", "virsh", "qemu-img", "qemu-system-x86_64", "timeout"]
)
def test_missing_host_dependency_is_explicit_and_never_installed(
    host: InventoryHost,
    dependency: str,
) -> None:
    (host.bin / dependency).unlink()
    result = host.run()
    rows = _rows(result)
    assert any(dependency in key and status == "MISSING" for key, (status, _) in rows.items()), (
        result.stdout
    )
    _assert_blocked_authorization(result)
    assert all(name not in {"sudo", "mkdir", "chmod", "chown"} for name, _ in host.calls())


@pytest.mark.parametrize(
    "scenario",
    [{"libvirt_unavailable": True}, {"timeout": True}, {"libvirt_uri": "qemu:///session"}],
)
def test_unavailable_timed_out_or_alternate_libvirt_cannot_be_accepted(
    host: InventoryHost,
    scenario: dict[str, Any],
) -> None:
    result = host.run(probe_libvirt=True, **scenario)
    rows = _rows(result)
    assert any(
        "libvirt" in key.lower() and status in {"BLOCKED", "INACCESSIBLE"}
        for key, (status, _) in rows.items()
    ), result.stdout
    for name, args in host.calls():
        if name == "virsh":
            assert args == ["--readonly", "--connect", "qemu:///system", "uri"]
    _assert_blocked_authorization(result)


@pytest.mark.parametrize("dependency", ["env", "virsh", "timeout"])
def test_opted_in_probe_with_missing_dependency_does_not_connect_or_broaden_access(
    host: InventoryHost,
    dependency: str,
) -> None:
    (host.bin / dependency).unlink()
    result = host.run(probe_libvirt=True)
    assert _rows(result)[f"dependency:{dependency}"][0] == "MISSING"
    assert _rows(result)["libvirt_readonly"][0] == "NOT_TESTED"
    assert not any(name == "virsh" for name, _ in host.calls())
    _assert_blocked_authorization(result)


@pytest.mark.parametrize("probe_libvirt", [False, True])
@pytest.mark.parametrize("scenario", [{"system": "Darwin"}, {"machine": "aarch64"}])
def test_unsupported_host_stops_native_probes_without_relaxing_qualification(
    host: InventoryHost,
    scenario: dict[str, str],
    probe_libvirt: bool,
) -> None:
    result = host.run(probe_libvirt=probe_libvirt, **scenario)
    assert "BLOCKED" in result.stdout
    assert not any(name == "virsh" for name, _ in host.calls())
    _assert_blocked_authorization(result)


@pytest.mark.parametrize(
    "head", ["213226f6222f8d1ece05987afe038c82ed61923d", "a" * 40, "b" * 40, "0" * 40]
)
def test_each_observed_current_revision_is_present_without_approving_a_release(
    host: InventoryHost,
    head: str,
) -> None:
    result = host.run(head=head)
    rows = _rows(result)
    assert rows["repository_revision"] == (
        "PRESENT",
        f"{head} OBSERVED_REVISION_NOT_RELEASE_APPROVAL",
    )
    assert rows["repository_release_approval"] == (
        "UNVERIFIED",
        "INDEPENDENT_SOURCE_REVIEW_REQUIRED",
    )
    _assert_blocked_authorization(result)


@pytest.mark.parametrize("head", ["malformed revision", ""])
def test_unobservable_revision_keeps_independent_release_approval_unverified(
    host: InventoryHost,
    head: str,
) -> None:
    result = host.run(head=head)
    assert _rows(result)["repository_revision"][0] == "INACCESSIBLE"
    assert _rows(result)["repository_release_approval"][0] == "UNVERIFIED"
    _assert_blocked_authorization(result)


def test_local_repository_changes_remain_unverified_with_observed_revision(
    host: InventoryHost,
) -> None:
    result = host.run(git_status=" M existing-file\n")
    assert _rows(result)["repository_revision"][0] == "PRESENT"
    assert _rows(result)["repository_worktree"][0] == "UNVERIFIED"
    assert _rows(result)["repository_release_approval"][0] == "UNVERIFIED"
    _assert_blocked_authorization(result)


def _snapshot(root: Path) -> dict[str, tuple[Any, ...]]:
    snapshot = {}
    for path in [root, *sorted(root.rglob("*"))]:
        if path in {root / "scenario.json", root / "commands.jsonl"}:
            continue
        metadata = path.lstat()
        payload = path.read_bytes() if not path.is_symlink() and path.is_file() else None
        snapshot[str(path.relative_to(root))] = (
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_nlink,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            os.readlink(path) if path.is_symlink() else None,
            payload,
        )
    return snapshot


@pytest.mark.parametrize("probe_libvirt", [False, True])
def test_inventory_uses_only_read_queries_and_does_not_mutate_retained_state(
    host: InventoryHost,
    probe_libvirt: bool,
) -> None:
    # Exclude only the scenario/log that the test harness writes. Access times
    # are intentionally omitted: inspecting existing bytes may update atime.
    (host.repo / "symlink").symlink_to(host.sentinel.name)
    before = _snapshot(host.root)
    result = host.run(probe_libvirt=probe_libvirt)
    assert _snapshot(host.root) == before
    calls = host.calls()
    assert calls
    assert {name for name, _ in calls} <= {
        "uname",
        "stat",
        "git",
        "timeout",
        "virsh",
        "dpkg-query",
    }
    assert not result.stderr, result.stderr
    for name, args in calls:
        if name == "git":
            assert "--no-optional-locks" in args
            assert "core.fsmonitor=false" in args
            assert "core.untrackedCache=false" in args
            assert "rev-parse" in args or "status" in args
        if name == "stat":
            assert "--" in args
            assert args[-1].startswith(("/opt", "/etc", "/usr", "/dev", "/sys", "/"))
            assert "private" not in args[-1] and not args[-1].endswith((".key", ".pem"))
    _assert_blocked_authorization(result)


def test_inherited_logging_and_authentication_settings_are_removed_before_tools(
    host: InventoryHost,
) -> None:
    git_trace = host.root / "unexpected-git-trace"
    git_events = host.root / "unexpected-git-events"
    libvirt_log = host.root / "unexpected-libvirt-log"
    auth = host.root / "operator-auth-sentinel"
    auth.write_text("Synthetic sentinel: no tool may read this file.\n")
    before = auth.stat()
    result = host.run(
        probe_libvirt=True,
        extra_environment={
            "GIT_TRACE": str(git_trace),
            "GIT_TRACE2_EVENT": str(git_events),
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "trace2.eventTarget",
            "GIT_CONFIG_VALUE_0": str(git_events),
            "LIBVIRT_LOG_OUTPUTS": f"1:file:{libvirt_log}",
            "LIBVIRT_AUTH_FILE": str(auth),
            "VIRSH_HISTFILE": str(host.root / "unexpected-virsh-history"),
            "HOME": str(host.root / "unreviewed-home"),
        },
    )
    assert not git_trace.exists()
    assert not git_events.exists()
    assert not libvirt_log.exists()
    assert auth.stat() == before
    git_environments = host.environments("git")
    assert git_environments
    for environment in git_environments:
        assert "GIT_TRACE" not in environment
        assert environment["GIT_TRACE2_EVENT"] == "0"
        assert environment["GIT_OPTIONAL_LOCKS"] == "0"
        assert environment["GIT_TERMINAL_PROMPT"] == "0"
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert "GIT_CONFIG_COUNT" not in environment
        assert "HOME" not in environment
    libvirt_environments = host.environments("virsh")
    assert libvirt_environments
    for environment in libvirt_environments:
        assert environment["LIBVIRT_LOG_OUTPUTS"] == "4:stderr"
        assert environment["LIBVIRT_AUTH_FILE"] == "/dev/null"
        assert environment["VIRSH_HISTFILE"] == "/dev/null"
        assert "HOME" not in environment
    _assert_blocked_authorization(result)


@pytest.mark.skipif(REAL_GIT is None, reason="Native Git is required for its logging regression")
def test_native_git_cannot_create_inherited_trace_files(host: InventoryHost) -> None:
    (host.bin / "git").unlink()
    (host.bin / "git").symlink_to(str(REAL_GIT))
    trace_paths = {
        name: host.root / name.lower()
        for name in ("GIT_TRACE", "GIT_TRACE_SETUP", "GIT_TRACE2_EVENT", "GIT_TRACE2_PERF")
    }
    configured_targets = {
        key: host.root / key.replace(".", "-")
        for key in ("trace2.normalTarget", "trace2.eventTarget", "trace2.perfTarget")
    }
    setup_environment = {
        "PATH": os.defpath,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TRACE2": "0",
        "GIT_TRACE2_EVENT": "0",
        "GIT_TRACE2_PERF": "0",
    }
    subprocess.run(
        [str(REAL_GIT), "init", "--quiet", str(host.repo)],
        env=setup_environment,
        capture_output=True,
        check=True,
    )
    for key, path in configured_targets.items():
        subprocess.run(
            [str(REAL_GIT), "-C", str(host.repo), "config", key, str(path)],
            env=setup_environment,
            capture_output=True,
            check=True,
        )
    # The disposable repository deliberately has no HEAD. Native Git's error
    # path must not activate inherited or repository-configured tracing.
    result = host.run(extra_environment={name: str(path) for name, path in trace_paths.items()})
    assert all(not path.exists() for path in trace_paths.values())
    assert all(not path.exists() for path in configured_targets.values())
    assert _rows(result)["repository_revision"][0] == "INACCESSIBLE"
    _assert_blocked_authorization(result)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--policy-directory", "/unapproved/path"],
        ["--repository"],
        ["--repository", ""],
        ["--repository", LIBVIRT_OPT_IN],
        ["--repository", ".", "--repository", "."],
        [LIBVIRT_OPT_IN, LIBVIRT_OPT_IN],
        [LIBVIRT_OPT_IN, "--unknown-option"],
        [LIBVIRT_OPT_IN, "--repository"],
        [LIBVIRT_OPT_IN, "true"],
    ],
)
def test_invalid_or_duplicate_arguments_fail_before_any_host_query(
    host: InventoryHost,
    arguments: list[str],
) -> None:
    result = host.run(arguments=arguments)
    assert result.returncode == 64
    assert not result.stdout
    assert not host.calls()
