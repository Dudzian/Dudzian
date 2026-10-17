"""Fail-closed, read-only Windows protected-tree ACL observations.

The validator evaluates every Allow ACE conservatively, including group SIDs;
an untrusted write grant is never excused by a compensating Deny. Disabled
inheritance is supported only when the complete observed DACL is safe. This
module does not grant access, repair ACLs, resolve junctions or read file data.
"""

from __future__ import annotations

import ctypes
import os
import re
import stat
from pathlib import Path, PureWindowsPath
from typing import Any

SYSTEM_SID = "S-1-5-18"
ADMINISTRATORS_SID = "S-1-5-32-544"
TRUSTED_INSTALLER_SID = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
_MUTATION_RIGHTS = 2 | 4 | 16 | 64 | 256 | 65536 | 262144 | 524288
_KNOWN_RIGHTS = 0x1F01FF
_SID = re.compile(r"S-1-(?:[0-9]+-)*[0-9]+")
_FILE_ID = re.compile(r"[0-9a-f]{8}:[0-9a-f]{16}")


def _path(value: Any) -> PureWindowsPath | None:
    if not isinstance(value, str) or re.match(r"^[A-Za-z]:\\", value) is None:
        return None
    parts = value[3:].split("\\") if value[3:] else []
    if any(
        not part
        or part in {".", ".."}
        or part[-1:] in {".", " "}
        or any(ord(char) < 32 or char in '/:*?"<>|' for char in part)
        for part in parts
    ):
        return None
    parsed = PureWindowsPath(value)
    return parsed if parsed.is_absolute() else None


def _sid(value: Any) -> bool:
    return isinstance(value, str) and _SID.fullmatch(value) is not None


def _native_system_directory() -> str:
    """Resolve the OS directory through a native API, never process environment."""
    # kernel32 is an OS KnownDLL; LOAD_LIBRARY_SEARCH_SYSTEM32 also prevents
    # resolution through an attacker-controlled working directory or PATH.
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise OSError("Native Windows DLL loader is unavailable.")
    kernel32 = loader("kernel32.dll", use_last_error=True, winmode=0x00000800)
    query = kernel32.GetSystemDirectoryW
    query.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    query.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = query(buffer, len(buffer))
    if length == 0 or length >= len(buffer) or len(buffer.value) != length:
        raise OSError("Native Windows system directory is unavailable.")
    return buffer.value


def native_powershell_path() -> str:
    """Return an existing native OS PowerShell path, rejecting redirected paths.

    The executable is independent of CWD, PATH, SystemRoot and WINDIR. This
    read-only preflight requires a 64-bit Windows process, matching the native
    runner boundary; it never falls back to an ambient executable.
    """
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError("Native 64-bit Windows PowerShell is unavailable.")
    directory = _path(_native_system_directory())
    if directory is None or directory.name.casefold() != "system32":
        raise OSError("Native Windows system directory is invalid.")
    executable = directory / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    for candidate in (executable, *executable.parents):
        metadata = Path(str(candidate)).lstat()
        if stat.S_ISLNK(metadata.st_mode) or (getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise OSError("Native PowerShell path contains a symlink or reparse point.")
        expected_type = stat.S_ISREG if candidate == executable else stat.S_ISDIR
        if not expected_type(metadata.st_mode):
            raise OSError("Native PowerShell executable or parent has an invalid type.")
    return str(executable)


def validate_acl_tree(
    observation: Any,
    *,
    runner_root: str,
    service_sid: str,
    trusted_write_sids: frozenset[str] | None = None,
    mandatory_paths: tuple[str, ...] = (),
) -> list[str]:
    """Return public errors, or [] for a complete consistent safe observation.

    This validates facts, not their provenance. Callers must independently
    collect live observations before issuing a live PASS. ``mandatory_paths``
    contains relative paths that must exist; every other actual descendant is
    inspected as well. Product callers may supply additional approved service
    SIDs, while still checking their own more specific DACL contract.
    """
    root = _path(runner_root)
    if root is None or root.parent == root or not _sid(service_sid):
        return ["Protected root or approved service SID is invalid."]
    trusted = {SYSTEM_SID, ADMINISTRATORS_SID, service_sid}
    if trusted_write_sids is not None:
        if not isinstance(trusted_write_sids, frozenset) or not all(
            _sid(sid) for sid in trusted_write_sids
        ):
            return ["Approved ACL principals are invalid."]
        # Product service accounts are specific identities, never well-known
        # broad groups such as Everyone, Users, Authenticated Users or OWNER.
        if any(
            sid not in {SYSTEM_SID, ADMINISTRATORS_SID}
            and not sid.startswith(("S-1-5-21-", "S-1-5-80-"))
            for sid in trusted_write_sids
        ):
            return ["Approved ACL principals include an unsupported group."]
        trusted.update(trusted_write_sids)
    if (
        not isinstance(observation, dict)
        or type(observation.get("schema_version")) is not int
        or observation["schema_version"] != 1
        or _path(observation.get("root")) != root
        or observation.get("complete") is not True
        or observation.get("errors") != []
    ):
        return ["Complete independently collected ACL tree evidence is unavailable."]
    before, after = observation.get("before"), observation.get("after")
    if not isinstance(before, list) or not before or not isinstance(after, list):
        return ["ACL tree inventories are missing."]

    def inventory(records: list[Any]) -> tuple[dict[PureWindowsPath, Any], list[str]]:
        result: dict[PureWindowsPath, Any] = {}
        identifiers: set[str] = set()
        errors: list[str] = []
        ancestors = set(root.parents)
        for entry in records:
            if not isinstance(entry, dict):
                errors.append("ACL inventory contains an invalid entry.")
                continue
            path = _path(entry.get("path"))
            if (
                path is None
                or path in result
                or not (path in ancestors or path == root or root in path.parents)
                or _path(entry.get("final_path")) != path
            ):
                errors.append("ACL inventory paths are inconsistent or escape the protected tree.")
                continue
            result[path] = entry
            attributes = entry.get("attributes")
            kind = entry.get("kind")
            file_id = entry.get("file_id")
            if (
                not isinstance(kind, str)
                or kind not in {"directory", "file"}
                or type(attributes) is not int
                or not 0 <= attributes <= 0xFFFFFFFF
                or attributes & 0x400
                or bool(attributes & 0x10) != (kind == "directory")
                or not isinstance(file_id, str)
                or _FILE_ID.fullmatch(file_id) is None
                or file_id in identifiers
                or type(entry.get("link_count")) is not int
                or entry["link_count"] != 1
                or any(
                    type(entry.get(key)) is not int or entry[key] < 0
                    for key in ("length", "last_write_ticks", "change_ticks")
                )
            ):
                errors.append("ACL inventory has a reparse point, alias or invalid file identity.")
            if isinstance(file_id, str):
                identifiers.add(file_id)
            if (
                not isinstance(entry.get("owner_sid"), str)
                or entry["owner_sid"] not in trusted | {TRUSTED_INSTALLER_SID}
                or type(entry.get("inheritance_disabled")) is not bool
                or entry.get("dacl_present") is not True
                or entry.get("dacl_canonical") is not True
            ):
                errors.append("Protected owner, inheritance or complete canonical DACL is unsafe.")
            aces = entry.get("aces")
            if not isinstance(aces, list) or not aces:
                errors.append("Protected DACL cannot be read completely.")
                continue
            for ace in aces:
                if not isinstance(ace, dict):
                    errors.append("Protected DACL contains an unsupported ACE.")
                    continue
                rights = ace.get("rights")
                if (
                    not _sid(ace.get("sid"))
                    or type(rights) is not int
                    or rights < 0
                    or rights & ~_KNOWN_RIGHTS
                    or not isinstance(ace.get("access_type"), str)
                    or ace["access_type"] not in {"Allow", "Deny"}
                    or type(ace.get("inherited")) is not bool
                    or type(ace.get("inheritance_flags")) is not int
                    or ace["inheritance_flags"] not in range(4)
                    or type(ace.get("propagation_flags")) is not int
                    or ace["propagation_flags"] not in range(4)
                ):
                    errors.append("Protected DACL contains an unsupported ACE.")
                elif (
                    ace["access_type"] == "Allow"
                    and ace["sid"] not in trusted
                    and rights & _MUTATION_RIGHTS
                ):
                    errors.append("Untrusted explicit, inherited or group mutation grant exists.")
        required = {root, *root.parents}
        for relative in mandatory_paths:
            relative_path = PureWindowsPath(relative)
            if relative_path.is_absolute() or relative_path.drive or ".." in relative_path.parts:
                errors.append("Mandatory protected path is invalid.")
                continue
            required.add(root / relative_path)
        if not required.issubset(result):
            errors.append("Protected root, parents or mandatory descendants are missing.")
        for path, entry in result.items():
            if path in ancestors | {root} and entry.get("kind") != "directory":
                errors.append("Protected root or parent is not a directory.")
            if path.parent != path and (
                path.parent not in result or result[path.parent].get("kind") != "directory"
            ):
                errors.append("Protected tree inventory is disconnected.")
        return result, errors

    first, errors = inventory(before)
    second, last_errors = inventory(after)
    errors.extend(last_errors)
    if first != second:
        errors.append("Protected tree changed during ACL inspection.")
    return list(dict.fromkeys(errors))


# Used only as a function prelude to a live, bounded PowerShell process. Every
# component is opened without FILE_SHARE_DELETE and with OPEN_REPARSE_POINT.
# FILE_READ_DATA/LIST_DIRECTORY engages sharing checks; metadata-only access
# does not prevent deletion or renaming.
# Handles stay open across both complete inventories; final paths and file IDs
# bind path-based Get-Acl reads to pinned objects. No file contents are read.
WINDOWS_ACL_TREE_SCRIPT = r"""
if (-not ('CryptoHunterReadOnlyAcl' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;
public static class CryptoHunterReadOnlyAcl {
  [StructLayout(LayoutKind.Sequential)] public struct Info {
    public uint Attributes; public System.Runtime.InteropServices.ComTypes.FILETIME Creation;
    public System.Runtime.InteropServices.ComTypes.FILETIME Access;
    public System.Runtime.InteropServices.ComTypes.FILETIME Write;
    public uint Volume, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
  }
  [StructLayout(LayoutKind.Sequential)] public struct Basic {
    public long Creation, Access, Write, Change; public uint Attributes;
  }
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern SafeFileHandle CreateFile(string path, uint access, uint share,
    IntPtr security, uint creation, uint flags, IntPtr template);
  [DllImport("kernel32.dll", SetLastError=true)]
  static extern bool GetFileInformationByHandle(SafeFileHandle handle, out Info info);
  [DllImport("kernel32.dll", SetLastError=true)]
  static extern bool GetFileInformationByHandleEx(SafeFileHandle handle, int kind,
    out Basic info, uint size);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  static extern uint GetFinalPathNameByHandle(SafeFileHandle handle, StringBuilder path,
    uint length, uint flags);
  public static SafeFileHandle Open(string path) {
    var handle=CreateFile(path, 0x20081, 3, IntPtr.Zero, 3, 0x02200000, IntPtr.Zero);
    if(handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
    return handle;
  }
  public static Info Read(SafeFileHandle handle) {
    Info info; if(!GetFileInformationByHandle(handle, out info))
      throw new Win32Exception(Marshal.GetLastWin32Error()); return info;
  }
  public static Basic ReadBasic(SafeFileHandle handle) {
    Basic info; if(!GetFileInformationByHandleEx(handle, 0, out info,
      (uint)Marshal.SizeOf(typeof(Basic))))
      throw new Win32Exception(Marshal.GetLastWin32Error()); return info;
  }
  public static string FinalPath(SafeFileHandle handle) {
    var path=new StringBuilder(32768);
    uint length=GetFinalPathNameByHandle(handle,path,(uint)path.Capacity,0);
    if(length==0 || length>=path.Capacity) throw new Win32Exception();
    string result=path.ToString();
    if(!result.StartsWith(@"\\?\") || result.StartsWith(@"\\?\UNC\"))
      throw new InvalidOperationException("local path required");
    return result.Substring(4);
  }
}
'@ -ErrorAction Stop
}
function Collect-RunnerAclTree {
  param([Parameter(Mandatory=$true)][string]$Root)
  $handles = [Collections.Generic.Dictionary[string,IDisposable]]::new(
    [StringComparer]::OrdinalIgnoreCase)
  try {
    # The native executable does not make ambient PSModulePath trustworthy.
    Import-Module -Name ([IO.Path]::Combine($PSHOME,'Modules','Microsoft.PowerShell.Security',
      'Microsoft.PowerShell.Security.psd1')) -ErrorAction Stop
    $canonical = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    if ($canonical -notmatch '^[A-Za-z]:\\' -or $canonical.Length -lt 4) {
      throw 'protected local root required'
    }
    function Read-ProtectedEntry([string]$Path) {
      if (-not $handles.ContainsKey($Path)) {
        if ($handles.Count -ge 50000) { throw 'bounded inventory limit exceeded' }
        $handles.Add($Path, [CryptoHunterReadOnlyAcl]::Open($Path))
      }
      $handle = $handles[$Path]
      $info = [CryptoHunterReadOnlyAcl]::Read($handle)
      $basic = [CryptoHunterReadOnlyAcl]::ReadBasic($handle)
      $final = [CryptoHunterReadOnlyAcl]::FinalPath($handle)
      if (($info.Attributes -band 0x400) -or
          -not [StringComparer]::OrdinalIgnoreCase.Equals($Path,$final)) {
        throw 'reparse point or path mismatch'
      }
      $acl = Microsoft.PowerShell.Security\Get-Acl -LiteralPath $Path -ErrorAction Stop
      $raw = [Security.AccessControl.RawSecurityDescriptor]::new(
        $acl.GetSecurityDescriptorBinaryForm(),0)
      if ($null -eq $raw.DiscretionaryAcl) { throw 'null DACL' }
      $aces = @(foreach ($ace in $raw.DiscretionaryAcl) {
        if ($ace -isnot [Security.AccessControl.CommonAce] -or $ace.IsCallback -or
            $ace.AceQualifier.ToString() -notin @('AccessAllowed','AccessDenied') -or
            (([int]$ace.AceFlags) -band 224)) { throw 'unsupported ACE' }
        @{ sid=$ace.SecurityIdentifier.Value; rights=[long]$ace.AccessMask;
           access_type=$(if($ace.AceQualifier.ToString() -eq 'AccessAllowed'){'Allow'}else{'Deny'});
           inherited=([bool](([int]$ace.AceFlags) -band 16));
           inheritance_flags=(([int]$ace.AceFlags) -band 3);
           propagation_flags=((([int]$ace.AceFlags) -band 12) -shr 2) }
      })
      @{ path=$Path; final_path=$final;
         kind=$(if($info.Attributes -band 16){'directory'}else{'file'});
         attributes=[long]$info.Attributes;
         file_id=('{0:x8}:{1:x8}{2:x8}' -f $info.Volume,$info.IndexHigh,$info.IndexLow);
         link_count=[long]$info.Links;
         length=([long]$info.SizeHigh * 4294967296 + [long]$info.SizeLow);
         last_write_ticks=$basic.Write; change_ticks=$basic.Change;
         owner_sid=$raw.Owner.Value; inheritance_disabled=$acl.AreAccessRulesProtected;
         dacl_present=([bool](([int]$raw.ControlFlags) -band 4));
         dacl_canonical=$acl.AreAccessRulesCanonical; aces=$aces }
    }
    function Read-ProtectedTree([string]$Path, [int]$Depth) {
      if ($Depth -gt 256) { throw 'bounded depth limit exceeded' }
      $entry = Read-ProtectedEntry $Path
      $entry
      if ($entry.kind -eq 'directory') {
        foreach ($child in @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop |
            Sort-Object FullName)) {
          Read-ProtectedTree $child.FullName ($Depth + 1)
        }
      }
    }
    $parents = [Collections.Generic.List[string]]::new()
    $parent = [IO.Directory]::GetParent($canonical)
    while ($null -ne $parent) { $parents.Insert(0,$parent.FullName); $parent=$parent.Parent }
    $before = @()
    foreach ($path in $parents) { $before += Read-ProtectedEntry $path }
    $before += @(Read-ProtectedTree $canonical 0)
    $after = @()
    foreach ($path in $parents) { $after += Read-ProtectedEntry $path }
    $after += @(Read-ProtectedTree $canonical 0)
    @{ schema_version=1; root=$canonical; complete=$true; errors=@(); before=$before; after=$after }
  } catch {
    @{ schema_version=1; root=$Root; complete=$false;
       errors=@('Protected ACL tree collection failed.'); before=@(); after=@() }
  } finally {
    foreach ($handle in $handles.Values) { $handle.Dispose() }
  }
}
"""
