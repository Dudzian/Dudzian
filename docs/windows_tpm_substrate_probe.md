# Windows Stage-9 TPM substrate acceptance probe

This is a disposable **acceptance-only** probe. It is not a production root
provider, does not change the MSI, and does not authorize the existing
fail-closed external provisioning handoff. A CI run or a virtual TPM is not
live substrate evidence.

## Physical Windows 11 run

Use a normal physical Windows 11 computer whose TPM 2.0 is ready and remains
under standard Windows auto-provisioning. Do not clear the TPM, change BIOS,
disable auto-provisioning, or change TPM registry configuration.

1. Check out the exact revision to qualify and install the repository's Python
   3.11 development dependencies (including `pywin32`).
2. Open **Windows Terminal as Administrator**. Elevation is needed only for the
   random disposable SCM service; it does not grant platform-hierarchy access.
3. From the repository root run:

   ```powershell
   py -3.11 -m deployment.windows_tpm_substrate_probe --output dist/windows/tpm-substrate-evidence.json
   $LASTEXITCODE
   Get-Content dist/windows/tpm-substrate-evidence.json
   ```

The probe uses `Tbsi_Context_Create`, `Tbsip_Submit_Command`, and
`Tbsip_Context_Close` directly. It enumerates existing NV handles before
randomly choosing an unused handle exclusively from the canonical TCG Owner
NV range `0x01800000..0x01BFFFFF`. It does not allocate from `0x01C00000` or
any platform, endorsement, OEM, or other reserved range. It requests an
8-byte `TPM_NT_COUNTER` using `TPM_RH_OWNER`; it never uses `TPM_RH_PLATFORM`
or `TPMA_NV_PLATFORMCREATE`. Failure of empty owner authorization is recorded
as a blocked substrate result and does not trigger ownership takeover.

A newly defined TPM NV counter has `TPMA_NV_WRITTEN` clear and cannot be read
as an initialized counter. Therefore the probe performs an initial
`NV_Increment`, then its first `NV_Read` to observe `g`, and only then performs
the qualified monotonic transition `NV_Increment` plus `NV_Read` to prove
`g -> g+1`. It never assumes that the TPM-selected initial value is `1`.

The selected handle is recorded before creation. Cleanup may undefine only
that exact handle and only after this process recorded successful creation.
The service name is random and is deleted in a `finally` path. Always inspect
the `cleanup` object; any cleanup failure is an operator action item. The JSON
contains command sizes and return codes, but no auth values, nonces, private
keys, or product secrets.

Before a physical execution all seven claims remain:

```text
CAN_CREATE_REQUIRED_NV_INDEX = NOT_RUN
CAN_READ_REQUIRED_NV_INDEX = NOT_RUN
CAN_INCREMENT_REQUIRED_NV_INDEX = NOT_RUN
CAN_OPEN_POLICY_SESSION = NOT_RUN
CAN_SATISFY_SELECTED_POLICY = NOT_RUN
CAN_SURVIVE_SERVICE_RESTART = NOT_RUN
CAN_DETECT_DISK_STATE_BEHIND_COUNTER = NOT_RUN
```

A nonzero exit is expected whenever the complete required substrate has not
been demonstrated. The current implementation executes a real policy session,
`PolicyNV`, and `PolicyCommandCode`. It does **not** execute `PolicyAuthorize`;
the machine-readable reason is
`POLICY_AUTHORIZE_COMMAND_PATH_NOT_IMPLEMENTED`. It also does not execute a
fake `PolicyOR` with duplicate digests: one real candidate branch is recorded
and `PolicyOR` remains `NOT_RUN` until two genuine branches exist. These are
implementation limits, not TPM feasibility findings. Do not infer production
feasibility from the static tests or from partial PASS results.

Before opening TBS, the probe qualifies machine-readable CIM values. The OS
must have `ProductType == 1` (Windows client, never Windows Server) and a
numeric build number of at least `22000`. A localized `Caption` is retained as
evidence but is not the qualification authority. A rejected OS reports
`PHYSICAL_WINDOWS_11_REQUIRED` before any formal output can become `PASS`.
