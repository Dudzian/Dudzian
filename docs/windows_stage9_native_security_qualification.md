# S9-S10-PRE-01B.2B — native Windows security qualification

Base: `main` after PR #3142, `1ac5be4bf7aa549a40e8a7dcfbf5bb202f76b4a6`.
This substep qualifies file custody APIs on temporary runner data. It grants no
permission to install, enroll, start services, access TPM/NV, reboot or run Stage 10.
Stage 9 remains `IN_PROGRESS`; Stage 10 remains `BLOCKED` / `NOT_RUN`.

## Evidence boundaries

Before this change, the following boundaries in
`deployment/windows_stage9_install_native.py` had portable emulated evidence only:

| Boundary | New real Windows evidence | Limit |
| --- | --- | --- |
| System32 DLL loading, AMD64 guard and ctypes ABI | `_Native` construction; independent raw identity buffers | Native Windows x64 only |
| `CreateFileW`, access, bilateral share modes, `CloseHandle` | Actual failed write/truncate/delete/rename/replace; identical operations succeed after release; existing writer blocks acquisition | Ordinary file sharing, not a hostile administrator or kernel adversary |
| `GetFileInformationByHandle`, `GetFileInformationByHandleEx`, `GetFinalPathNameByHandleW` | Independent calls verify volume/file ID, link count, size, attributes, final path, write/change times for files and directories | Local runner filesystem |
| No-follow opens; aliases | Actual hardlinks, file/directory symlinks and junction mount-point reparse tags rejected | Unavailable privileges/filesystem features produce explicit skips; other reparse tags are not synthesized |
| Handle-backed seek/read/hash/ZIP and write/truncate/flush | Actual handle bytes and archive member reads, real read/write operations | Inert test bytes only |
| Directory pins and inventory | OS denies directory removal/rename; added children and metadata changes fail revalidation | Inventory composition harness scans only its four temporary trust domains |
| Protected SDDL conversion, allocation/free, `CREATE_NEW`, directory creation | Real protected directory/file creation and read-only collection of their actual DACLs | Elevated token required; no existing ACL is changed |
| Shared ACL collector handle lifetime | Actual production C# `Open`, deletion/rename denial, identical mutations after `Dispose` | Read-data/list-directory permission now required; no contents are read |
| Materialization and source integrity | Bad ZIP/digest rejected, sanitized failure report, unchanged source hashes, extraction handle released | Failure precedes OS installer binding; complete successful materialization is not asserted |
| Handle lifetime | Repeated success/exception cycles, process handle counts, invalid-handle checks, successful mutations after release | No process/installer handle lifetime qualification |

The adapter previously obtained `CreateFileW`'s last error after `LocalFree`.
It now captures failure before descriptor cleanup, preserving the actual denial
code. A portable regression simulates cleanup changing error 32 to error 5.
The fail-closed decision and security policy are unchanged.

The first real Windows run exposed a security defect hidden by the emulator:
`FILE_READ_ATTRIBUTES` alone did not engage sharing denial, so an open log could
be deleted or renamed. The log now requests the minimal additional
`FILE_READ_DATA` right (`0x81` combined), while retaining read/write sharing and
omitting delete sharing. It still admits writers that share reads. Compatibility
with the actual Windows Installer logger remains unqualified without MSI.

`MoveFileExW` replacement of an immutable pinned destination was genuinely denied
with `ERROR_ACCESS_DENIED` (5), while direct writes/deletes/renames report
`ERROR_SHARING_VIOLATION` (32). Replacement tests accept those two documented
denial results only; operation success still fails, and the identical operation
must succeed after release. Native tests exercise log custody separately from
immutable source pins.

The shared ACL collector had the same metadata-only sharing assumption. Its
access mask now includes `FILE_READ_DATA` / `FILE_LIST_DIRECTORY` (`0x20081`),
with unchanged read/write sharing and no delete sharing. Native tests execute
the actual C# helper, rather than an equivalent mock. Missing additional read
permission remains a fail-closed collection failure; no ACL is granted or repaired.

The next Windows run confirmed the log fix (26 native passes) and diagnosed the
remaining collection failure: native Windows PowerShell could not load the
ambient `Microsoft.PowerShell.Security` module when launched from `pwsh` CI.
The collector now imports the module by its absolute native `$PSHOME` manifest
and invokes its qualified `Get-Acl`. The native DACL case prepends an owned,
same-named hostile module that throws on load; the real collector must use the
system module and still produce the expected owned DACL records. PowerShell
compiler `TEMP`/`TMP` paths are confined to each test's sandbox.

## Emulator interpretation

`tests/deployment/test_windows_stage9_install_native.py::Win32` is a portable
control-flow stand-in. It stores paths, reopens those paths for reads/identity,
does not enforce access/share modes or DACLs, follows symlinks when observing
identity and synthesizes directory link counts and stable shared-ancestor times.
Its replacement negatives prove that validation rejects changed observations;
they do not establish native object retention or sharing denial. This deliberate
scope is now explicit in its docstring. Native tests load real DLLs independently
and do not import or substitute this emulator.

`test_windows_stage9_install_native_integration_windows.py::SandboxInputs`
only restricts inventory traversal to owned temporary domains. It uses actual
production `_walk`, `_pin`, `verify_pins`, materialization, hash and report code.
It does not call `_establish` or pretend that shared runner ancestry passed ACL
policy. The DACL test calls the actual collector and examines owned entries;
it does not reinterpret collection as whole-path policy acceptance.

## CI and reproduction

The independent `windows-native-security-qualification` job in the existing
`CI` workflow uses `windows-latest`, Python 3.11 x64 and a ten-minute limit. It
does not depend on the application build or long test matrix. It executes:

```powershell
python -m pytest --confcutdir=tests/deployment -vv -rA `
  tests/deployment/test_windows_stage9_install_native_windows.py `
  tests/deployment/test_windows_stage9_install_native_integration_windows.py `
  --basetemp (Join-Path $env:RUNNER_TEMP "stage9-native-security") `
  --junitxml=test-results/windows-native-security/native.xml
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `--confcutdir` isolate this suite from
unrelated application fixtures. The test code neither invokes executor entry
points nor launches MSI. A separate step runs the emulated adapter and executor
regressions, with separate `emulated.xml` and log. Environment revision/run/
attempt/Windows/Python metadata, native log, JUnit and explicit skip reasons are
retained in `windows-native-security-<run_id>-<attempt>` for seven days.

On Linux, native skips are a collection check only and never Windows evidence.
Any privileged/unsupported case skipped on Windows remains an explicit gap.

## Remaining qualification

The public `locked_installation_inputs` pins and validates ancestry up to the
drive root. Hosted runner temporary ancestry may have owners/write permissions
outside the protected-host policy, and shared ancestor metadata can change
during concurrent runner activity or PowerShell compiler temporary-file use.
These are reasons for fail-closed rejection, not grounds to relax the validators
or modify runner ACLs. Whole protected-host acceptance still needs an approved
host with independently protected ancestry.

`windows_stage9_install_executor.py` additionally has native interactive-console
and operator/session boundaries. Their existing mocks are not replaced by this
file-custody qualification. `GetSystemDirectoryW`/OS `msiexec.exe` binding,
complete successful materialization under protected host ACLs, console consent,
process handoff, Installer-service behavior, durable power-loss recovery,
post-installation qualification and physical TPM/Stage 10 remain outside the
evidence here. No MSI is executed to close these gaps.

GO requires linked successful real Windows evidence for the stated substep and
review of every skip. A green emulator/Linux run alone is NO-GO. Even a successful
native file-custody run is not production Stage 9/10 PASS or installation approval.
