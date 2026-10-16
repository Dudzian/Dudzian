from __future__ import annotations

import stat
from copy import deepcopy
from pathlib import PureWindowsPath
from types import SimpleNamespace

import pytest

from deployment import windows_runner_acl as acl
from deployment.windows_runner_acl import (
    TRUSTED_INSTALLER_SID,
    WINDOWS_ACL_TREE_SCRIPT,
    validate_acl_tree,
)

ROOT = r"C:\runners\stage10"
SERVICE_SID = "S-1-5-21-111-222-333-1001"
MANDATORY = ("bin", "_work", "_update", ".runner", r"bin\RunnerService.exe")


def safe_tree() -> dict:
    root = PureWindowsPath(ROOT)
    paths = [*reversed(root.parents), root]
    paths.extend(root / path for path in MANDATORY)
    paths.extend((root / "_work" / "checkout", root / "_work" / "checkout" / "config.json"))
    entries = []
    for index, path in enumerate(paths):
        directory = path.name not in {".runner", "RunnerService.exe", "config.json"}
        entries.append(
            {
                "path": str(path),
                "final_path": str(path),
                "kind": "directory" if directory else "file",
                "attributes": 16 if directory else 32,
                "file_id": f"01234567:{index:016x}",
                "link_count": 1,
                "length": 0 if directory else 123,
                "last_write_ticks": 123,
                "change_ticks": 123,
                "owner_sid": "S-1-5-32-544",
                "inheritance_disabled": False,
                "dacl_present": True,
                "dacl_canonical": True,
                "aces": [
                    {
                        "sid": "S-1-5-32-544",
                        "rights": 2032127,
                        "access_type": "Allow",
                        "inherited": False,
                        "inheritance_flags": 3 if directory else 0,
                        "propagation_flags": 0,
                    },
                    {
                        "sid": "S-1-5-32-545",
                        "rights": 131209,
                        "access_type": "Allow",
                        "inherited": True,
                        "inheritance_flags": 0,
                        "propagation_flags": 0,
                    },
                ],
            }
        )
    return {
        "schema_version": 1,
        "root": ROOT,
        "complete": True,
        "errors": [],
        "before": entries,
        "after": deepcopy(entries),
    }


def errors(tree: object) -> list[str]:
    return validate_acl_tree(
        tree, runner_root=ROOT, service_sid=SERVICE_SID, mandatory_paths=MANDATORY
    )


def mutate(tree: dict, key: str, value: object, *, index: int = -1) -> None:
    for phase in ("before", "after"):
        tree[phase][index][key] = deepcopy(value)


def test_complete_safe_tree_with_nested_parents_and_files_passes_without_mutation() -> None:
    tree = safe_tree()
    original = deepcopy(tree)
    assert errors(tree) == []
    assert tree == original


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("rights", [2, 4, 16, 64, 256, 65536, 262144, 524288, 2032127])
@pytest.mark.parametrize("index", [0, 1, 3, -1])
def test_parent_or_descendant_group_mutation_cannot_hide_behind_safe_root(
    inherited, rights, index
) -> None:
    tree = safe_tree()
    for phase in ("before", "after"):
        ace = tree[phase][index]["aces"][1]
        ace.update(sid="S-1-5-21-111-222-333-513", rights=rights, inherited=inherited)
        tree[phase][index]["inheritance_disabled"] = not inherited
    assert any("group mutation" in error for error in errors(tree))


def test_compensating_deny_does_not_excuse_untrusted_allow() -> None:
    tree = safe_tree()
    for phase in ("before", "after"):
        ace = tree[phase][-1]["aces"][1]
        ace["rights"] = 2
        tree[phase][-1]["aces"].insert(0, dict(ace, access_type="Deny"))
    assert errors(tree)


def test_safe_disabled_inheritance_and_specific_service_grants_are_supported() -> None:
    tree = safe_tree()
    for phase in ("before", "after"):
        tree[phase][-1]["inheritance_disabled"] = True
        ace = tree[phase][-1]["aces"][1]
        ace.update(sid=SERVICE_SID, rights=2032127, inherited=False)
        tree[phase][-1]["owner_sid"] = SERVICE_SID
        tree[phase][0]["owner_sid"] = TRUSTED_INSTALLER_SID
    assert errors(tree) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("owner_sid", "S-1-1-0"),
        ("owner_sid", "S-1-5-21-111-222-333-513"),
        ("owner_sid", []),
        ("inheritance_disabled", None),
        ("dacl_present", False),
        ("dacl_canonical", False),
        ("aces", []),
        ("aces", None),
        ("aces", [None]),
        ("attributes", 0x410),
        ("attributes", -1),
        ("attributes", True),
        ("file_id", "unknown"),
        ("link_count", 2),
        ("kind", []),
        ("length", -1),
        ("change_ticks", None),
        ("last_write_ticks", True),
        ("final_path", r"C:\outside\config.json"),
        ("path", r"C:\outside\config.json"),
        ("path", r"C:\runners\stage10\..\config.json"),
        ("path", r"C:\runners\stage10\config.json:stream"),
        ("path", "C:\\runners\\stage10\\config.json."),
        ("path", "C:\\runners\\stage10\\config.json "),
        ("path", "C:\\runners\\stage10\\folder\\\\config.json"),
    ],
)
def test_untrusted_incomplete_aliased_or_reparse_entries_fail_closed(key, value) -> None:
    tree = safe_tree()
    mutate(tree, key, value)
    assert errors(tree)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sid", None),
        ("sid", "unresolved-group"),
        ("rights", -2147483648),
        ("rights", 0x10000000),
        ("rights", True),
        ("rights", None),
        ("access_type", "Audit"),
        ("access_type", []),
        ("inherited", None),
        ("inheritance_flags", True),
        ("inheritance_flags", 4),
        ("propagation_flags", 4),
    ],
)
def test_unsupported_or_malformed_aces_fail_closed(key, value) -> None:
    tree = safe_tree()
    for phase in ("before", "after"):
        tree[phase][-1]["aces"][0][key] = value
    assert errors(tree)


@pytest.mark.parametrize(
    "key", ["path", "file_id", "length", "last_write_ticks", "change_ticks", "aces"]
)
def test_change_between_inventories_fails_closed(key) -> None:
    tree = safe_tree()
    tree["after"][-1][key] = "changed"
    assert any("changed during" in error for error in errors(tree))


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, -1])
def test_missing_parent_required_child_or_disconnected_descendant_fails(index) -> None:
    tree = safe_tree()
    for phase in ("before", "after"):
        del tree[phase][index]
    # A non-mandatory leaf can be absent from both complete inventories. Its
    # absence is only unsafe when inventory provenance is forged; live callers
    # collect these lists independently rather than trusting snapshot claims.
    if index == -1:
        assert errors(tree) == []
    else:
        assert errors(tree)


def test_duplicate_paths_and_file_id_aliases_fail_closed() -> None:
    tree = safe_tree()
    for phase in ("before", "after"):
        tree[phase].append(deepcopy(tree[phase][-1]))
    assert errors(tree)
    tree = safe_tree()
    mutate(tree, "file_id", tree["before"][0]["file_id"])
    assert errors(tree)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"root": r"C:\other"},
        {"complete": False},
        {"errors": ["read denied"]},
        {"before": None},
        {"before": []},
        {"after": None},
        {"after": []},
    ],
)
def test_read_failures_and_forged_or_incomplete_envelopes_fail_closed(change) -> None:
    tree = safe_tree()
    tree.update(change)
    assert errors(tree)


@pytest.mark.parametrize("tree", [None, [], {}, True])
def test_missing_or_malformed_observation_fails_closed(tree) -> None:
    assert errors(tree)


def test_product_specific_approved_service_sid_can_be_reused_without_broad_groups() -> None:
    tree = safe_tree()
    product_sid = "S-1-5-80-111-222-333-444-555"
    for phase in ("before", "after"):
        tree[phase][-1]["aces"][1].update(sid=product_sid, rights=2)
    assert errors(tree)
    assert not validate_acl_tree(
        tree,
        runner_root=ROOT,
        service_sid=SERVICE_SID,
        trusted_write_sids=frozenset({product_sid}),
    )
    assert validate_acl_tree(
        tree,
        runner_root=ROOT,
        service_sid=SERVICE_SID,
        trusted_write_sids=frozenset({"S-1-1-0"}),
    )


def test_live_collector_is_read_only_bounded_pins_objects_and_rejects_unknown_aces() -> None:
    assert "Get-Acl -LiteralPath" in WINDOWS_ACL_TREE_SCRIPT
    assert "CreateFile(path, 0x20080, 3" in WINDOWS_ACL_TREE_SCRIPT
    assert "0x02200000" in WINDOWS_ACL_TREE_SCRIPT
    assert "GetFinalPathNameByHandle" in WINDOWS_ACL_TREE_SCRIPT
    assert "GetFileInformationByHandleEx" in WINDOWS_ACL_TREE_SCRIPT
    assert "Read-ProtectedTree $canonical 0" in WINDOWS_ACL_TREE_SCRIPT
    assert WINDOWS_ACL_TREE_SCRIPT.count("Read-ProtectedTree $canonical 0") == 2
    assert "Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop" in WINDOWS_ACL_TREE_SCRIPT
    assert "50000" in WINDOWS_ACL_TREE_SCRIPT and "256" in WINDOWS_ACL_TREE_SCRIPT
    assert "$ace.IsCallback" in WINDOWS_ACL_TREE_SCRIPT
    assert "$handle.Dispose()" in WINDOWS_ACL_TREE_SCRIPT
    for forbidden in ("Set-Acl", "icacls", "takeown", "Get-Content", "ReadAllBytes"):
        assert forbidden not in WINDOWS_ACL_TREE_SCRIPT


def _mock_native_paths(monkeypatch, system_directory=r"C:\Windows\System32") -> dict:
    monkeypatch.setattr(acl, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(acl, "_native_system_directory", lambda: system_directory)
    executable = PureWindowsPath(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    metadata = {
        path: SimpleNamespace(
            st_mode=stat.S_IFREG if path == executable else stat.S_IFDIR,
            st_file_attributes=0,
        )
        for path in (executable, *executable.parents)
    }

    class ObservedPath:
        def __init__(self, path):
            self.path = PureWindowsPath(path)

        def lstat(self):
            if self.path not in metadata:
                raise FileNotFoundError("native path unavailable")
            return metadata[self.path]

    monkeypatch.setattr(acl, "Path", ObservedPath)
    return metadata


def test_native_powershell_uses_only_api_system_directory_and_checks_all_components(
    monkeypatch,
) -> None:
    metadata = _mock_native_paths(monkeypatch)
    monkeypatch.setenv("PATH", r"C:\attacker")
    monkeypatch.setenv("SystemRoot", r"C:\attacker")
    monkeypatch.setenv("WINDIR", r"C:\attacker")
    executable = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    assert acl.native_powershell_path() == executable
    for path in tuple(metadata):
        removed = metadata.pop(path)
        with pytest.raises(FileNotFoundError):
            acl.native_powershell_path()
        metadata[path] = removed


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("kind", ["symlink", "reparse", "wrong_type"])
def test_native_powershell_rejects_redirected_or_invalid_executable_and_parents(
    monkeypatch, index, kind
) -> None:
    metadata = _mock_native_paths(monkeypatch)
    path = tuple(metadata)[index]
    if kind == "symlink":
        metadata[path].st_mode = stat.S_IFLNK
    elif kind == "reparse":
        metadata[path].st_file_attributes = 0x400
    else:
        metadata[path].st_mode = stat.S_IFDIR if index == 0 else stat.S_IFREG
    with pytest.raises(OSError, match="path|type"):
        acl.native_powershell_path()


@pytest.mark.parametrize(
    "directory",
    [
        r"C:\attacker",
        r"C:\Windows\SysWOW64",
        r"C:\Windows\System32\..\attacker",
        r"\\server\Windows\System32",
        "relative",
        None,
    ],
)
def test_native_powershell_rejects_untrusted_or_inconsistent_api_directory(
    monkeypatch, directory
) -> None:
    _mock_native_paths(monkeypatch, directory)
    with pytest.raises(OSError, match="directory is invalid"):
        acl.native_powershell_path()


def test_native_powershell_rejects_non_windows_and_wow64_processes(monkeypatch) -> None:
    monkeypatch.setattr(acl, "os", SimpleNamespace(name="posix"))
    with pytest.raises(OSError, match="64-bit Windows"):
        acl.native_powershell_path()
    monkeypatch.setattr(acl, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(acl.ctypes, "sizeof", lambda _kind: 4)
    with pytest.raises(OSError, match="64-bit Windows"):
        acl.native_powershell_path()


@pytest.mark.parametrize("returned_length", [0, 32768, 999])
def test_native_system_directory_api_incomplete_reads_fail_closed(
    monkeypatch, returned_length
) -> None:
    def query(buffer, _length):
        buffer.value = r"C:\Windows\System32"
        return returned_length

    def library(name, **kwargs):
        assert name == "kernel32.dll"
        assert kwargs == {"use_last_error": True, "winmode": 0x00000800}
        return SimpleNamespace(GetSystemDirectoryW=query)

    monkeypatch.setattr(acl.ctypes, "WinDLL", library, raising=False)
    with pytest.raises(OSError, match="directory is unavailable"):
        acl._native_system_directory()


def test_native_system_directory_uses_winapi_unicode_output(monkeypatch) -> None:
    directory = r"C:\Windows\System32"

    def query(buffer, capacity):
        assert capacity == 32768
        buffer.value = directory
        return len(directory)

    monkeypatch.setattr(
        acl.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: SimpleNamespace(GetSystemDirectoryW=query),
        raising=False,
    )
    assert acl._native_system_directory() == directory


def test_native_system_directory_requires_native_loader(monkeypatch) -> None:
    monkeypatch.delattr(acl.ctypes, "WinDLL", raising=False)
    with pytest.raises(OSError, match="DLL loader is unavailable"):
        acl._native_system_directory()


def test_native_powershell_propagates_native_api_failure_without_fallback(monkeypatch) -> None:
    _mock_native_paths(monkeypatch)

    def unavailable():
        raise OSError("native API unavailable")

    monkeypatch.setattr(acl, "_native_system_directory", unavailable)
    with pytest.raises(OSError, match="native API unavailable"):
        acl.native_powershell_path()
