# Stage-9 two-branch physical TPM probe

This is a disposable, TEST-only successor qualification artifact. It does not
replace or weaken the already live-proven acceptance probe, and none of its
keys are Product Release Root, PDSA, PSA, or recovery production material.

On an elevated physical Windows 11 client with its normally auto-provisioned
TPM 2.0, use the canonical project interpreter, Python 3.12:

```powershell
py -3.12 -m deployment.windows_stage9_two_branch_probe `
  --output dist/windows/stage9-two-branch-evidence.json
```

The probe creates one random owner-range counter with exactly
`POLICYWRITE | COUNTER | AUTHREAD | NO_DA`; it has neither `OWNERWRITE` nor
`PLATFORMCREATE`. Empty `authValue` is used only for read and `PolicyNV`.
Writes use the final two-branch policy.

The bootstrap/recovery path executes `PolicyCommandCode`, `PolicyCpHash` over
the pre-write Name, `VerifySignature(K_RECOVERY_TEST)`, `PolicyAuthorize`, the
canonical `PolicyOR`, and `NV_Increment`. The normal path then executes
`PolicyNV` with the post-write Name, `PolicyCommandCode`,
`VerifySignature(K_PSA_TEST)`, `PolicyAuthorize`, the same ordered `PolicyOR`,
and `NV_Increment`.

Negative checks require rejection of a recovery-key ticket presented as the
normal authority and rejection of an increment after reversed branch ordering.
Cleanup flushes every session/key and undefines only the index created by the
run. Every semantic cleanup resource reports `PASS`, `FAIL`, or `NOT_CREATED`;
any `FAIL` forces a nonzero process exit and `failure.reason=CLEANUP_FAILED`.
Inspect all evidence and cleanup fields. Until a physical run succeeds,
this file and static tests prove executable construction only, not live TPM
qualification.

## Wynik fizyczny

Fizyczny Windows 11 / TPM 2.0 zakończył probe kodem `0`. Branch bootstrap/recovery
i normalny oraz negatywne cross-authority i reversed-OR mają `PASS`; wszystkie
sesje, oba klucze zewnętrzne, NV oraz kontekst TBS mają cleanup `PASS`.
Cross-authority zwrócił raw `0x00000084` (`TPM_RC_VALUE`). Reversed `PolicyOR`
był poprawną operacją, ale wytworzył inny digest, więc późniejszy
`NV_Increment` został oczekiwanie odrzucony jako `TPM_RC_POLICY_FAIL`.
Probe-specific refs w tym dowodzie pozostają niezmienione i nie są refs
canonical `ReleasePolicyV1`.
