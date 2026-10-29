# PR #3224 — archive correction and Windows SCM review

Review date **2026-10-29 UTC**. Audited PR HEAD:
`1932f5af941ca5a436888d8691c6d71af75d9f49`; base:
`cad53867231a1a6e49ea770c07043b417af84449`. This document records a
documentation correction and failure investigation. Production changes belong
to a separate repair PR. No Stage 3B, physical qualification, timeout increase,
test removal, `continue-on-error`, automatic PASS or merge is authorized here.

## Archive-count correction and evidence boundary

**`OPERATOR_REPORTED_AND_CHAT_REVIEWED` / `BUILD_SPECIFIC_UNVERIFIED`:** the
original uploaded request's **13/13** statement was imprecise. The independent
original-archive review in the operator conversation reports **14 manifest
entries, 14/14 verified, 0 mismatches and 15 regular files total**, including
the manifest itself. Archive name and SHA-256 remain
`stage9-elf-triage-rNlpy9.tar.gz` and
`0a26027f9bf94149fe2b36d3fbba1fdb6e5393e2016c9bd9956463b154005b0f`.
Manifest-file SHA-256:
`f52859c04f703bfe6ba36190229e7b18ac1543df5c443d589a549835177c3fdb`.
Local Dell readelf is **2.42**, separate from the cloud's **2.44**.

Codex still has no original Dell archive and did not replay this check. The
three new reports now explicitly record the supplied correction. The
[historical request](../architecture/evidence/stage9_observer_3a_cpython_request_v1.txt)
is unchanged: **5366 bytes**, SHA-256
`5f0ef91f0078ea530558b8fdc873008bf3f73d1375b7d40359904c02db9939d8`.
Historical manifests are unchanged. The new packet manifest is regenerated
for the corrected reports and this review's evidence only.

## Complete hosted-log comparison

Both complete job logs, job/run metadata and actual Git checkout objects were
read. [Machine-readable evidence](../architecture/evidence/stage9_observer_3a_windows_scm_review_v1.json)
retains log identities, timestamps, runner/package comparison and source blob
identities. Local hashes identify the exact connector-decoded UTF-8 text,
including BOM/CRLF; they do not claim identity of the HTTP archive transport.

| Evidence | Failed SCM job, PR #3224 | Previous successful SCM job, PR #3223 |
| --- | --- | --- |
| Run / job | [40866365166 / 123422984959](https://github.com/Dudzian/Dudzian/actions/runs/40866365166/job/123422984959) | [40838999279 / 123330819643](https://github.com/Dudzian/Dudzian/actions/runs/40838999279/job/123330819643) |
| PR HEAD | `1932f5af941ca5a436888d8691c6d71af75d9f49` | `097e43db60f91b86fddc7a0cbc7db1844d3ffd05` |
| Actual synthetic-merge checkout | `87f19583b0828417904aecba9b04a3415a21ce91` | `ab71a924a2139f3218d4264473d27f50221b7378` |
| Full decoded log | 61184 bytes; SHA-256 `ec0e5cfedee1c97c7676e18432444eaa81ec54bec47cf8093aa0995bf5848d84` | 65859 bytes; SHA-256 `95ce9a2210621d1b9fb84b1371b7977700a053fb12d136b224967d09608ab0e6` |
| Core tests | **790 passed in 96.69 s**, phase PASS at 11:40:21.844 UTC | **790 passed in 74.74 s**, phase PASS at 08:03:22.471 UTC |
| SCM phase | START 11:40:21.844, FAIL 11:41:12.033 UTC; **50.188 s** | START 08:03:22.471, PASS 08:03:56.907 UTC; **34.438 s** |
| Process tree | `ProcessTreeError: grandchild creation deadline expired`; `WINDOWS_PROCESS_TREE=FAIL` | `WINDOWS_PROCESS_TREE=PASS`, `WINDOWS_NO_ORPHAN_CHILDREN=PASS` |

The historical success is the same Windows integration **job**, not a claim
that its entire Platform workflow passed. Both runs retained
`WINDOWS_PRODUCTION_READY=NOT_READY`. A Core unit-test PASS does not prove the
separate live service/child handshake succeeded.

Direct comparison of both actual checkout trees finds **20 documentation /
evidence additions only**, with no executable source, workflow or dependency
configuration change. Eleven critical source paths have identical Git blob
IDs and SHA-256, including the SCM probe, parent/child/qualifier, service,
acceptance runner, logging, ACL code, platform workflow and pyproject.

| Environment dimension | Observed comparison and limit |
| --- | --- |
| Runner/image | Same runner **2.338.0**, provisioner **20261021.618** and commit, image **windows-2025-vs2026 / 20261021.278.1**, Windows Server 2025 Datacenter **10.0.26100**, Git **2.56.0.windows.2**. |
| Interpreter/packages | Both explicitly log CPython **3.11.9 x64**; all **89 logged pip-installed name/version tokens match**, including **pywin32 312**. This is not a full pip freeze, wheel-digest comparison or proof that all preinstalled state is identical. |
| PowerShell | Both show the Core/pwsh parent and native Desktop WindowsPowerShell ACL path; both qualify native Get-Acl authority. Numeric invoked PSVersion is not printed. The image README's **7.6.7** is an image declaration, not runtime-version evidence. |
| Host difference | Successful worker is in **centralus**, failed worker in **westus**, with different worker/runner IDs. No CPU, disk, load, antivirus event, filesystem sharing or helper traceback capture establishes a causal host difference. Longer Core duration alone does not prove host overload. |

## Source, deadlines and cleanup

Directly reviewed at the audited code revision:
[SCM probe](../../deployment/windows_scm_probe.ps1),
[parent Job Object owner](../../deployment/windows_process_tree.py),
[child publisher](../../deployment/windows_process_tree_child.py),
[service](../../deployment/windows_test_service.py),
[acceptance runner](../../deployment/windows_acceptance.py),
[existing process-tree tests](../../tests/deployment/test_windows_stage5_process_tree.py).
Codebase Memory MCP/CLI remain unavailable; conclusions use actual source and
Git object comparisons, not an index-coverage claim.

1. `ChildJob.start` creates a named query-only Job Object, verifies
   KILL_ON_JOB_CLOSE, starts the detached child, assigns it and queries the
   exact child membership **before** releasing the gate. Only then does the
   child start a grandchild; parent queries the exact two-PID set before
   publishing `process-tree.json`. The error identifies the wait for the
   grandchild PID, not a successful tree or proof no grandchild ever existed.
2. Child gate wait is **20 s** with **50 ms** polls. Parent membership and
   grandchild waits each have their own **20 s** / **50 ms** bounds. These
   are separate sequential intervals, not one total 20-second service budget.
   The PowerShell probe separately waits for service state (**60 × 500 ms**),
   health marker (**30 × 500 ms**) and tree marker (**40 × 500 ms**).
   A whole SCM-phase duration of 50.188 s therefore does not show that one
   particular 20-second deadline was ignored or needs enlargement.
   `windows_acceptance.py::run_scm` additionally caps the whole PowerShell
   subprocess at **180 s**, with a separate **60 s** CleanupOnly attempt if
   that outer timeout fires. The observed 50.188-second failure is not that
   outer timeout. Orphan polling uses **60 × 250 ms** plus qualifier work;
   neither elapsed Core time nor that polling interval proves host overload.
3. The old parent repeatedly opens `process-tree.gate` using `read_text`.
   The child writes a PID to `process-tree.tmp` and calls `os.replace` on
   that same gate **once**. On Windows a concurrent reader without delete
   sharing can prevent replacement. An injected transient Windows sharing
   error reproduces the defect: the child publication aborts on its first
   attempt, the gate stays `assigned`, and the parent later reports the exact
   grandchild deadline error. No unchanged timeout can repair a dead publisher.
4. Original parent wait ignores child exit. A bounded Windows-only publication
   retry inside the child's **existing** deadline, plus prompt parent refusal
   if the child exits before readiness, is implemented with regressions in
   separate [repair PR #3225](https://github.com/Dudzian/Dudzian/pull/3225),
   HEAD `4b873a0d80640e3408c153103044c6c462abff25`.
   Other errors and permanent denial must still fail;
   this is not a conversion of timeout into PASS.
5. Startup rollback closes the proven-membership Job Object as primary
   descendant termination, waits up to **5 s** for the child and has bounded
   emergency termination; transient files are removed in `finally`. The
   service also closes the job and removes gate/tree/health markers in its
   own `finally`. The PowerShell probe performs ownership-bound service,
   transient, logging, DACL, ACL and ownership-record cleanup and retains
   cleanup failures alongside the original failure.

In the failed log, diagnostics actually show **SCM Stopped / PID 0**, no health
marker, no tree JSON and no gate. These observations are consistent with
rollback but do not establish the absence of every child/grandchild or named
job. The full failure log contains no separate `[CLEANUP]` error, but no
published complete SCM cleanup/PID inventory either. Generic runner
`Cleaning up orphan processes` is not that proof. The Windows evidence upload
step was skipped on failure; its absence prevents inspection of an equivalent
final evidence JSON for this job. No complete cleanup PASS is inferred.

## Classification and merge recommendation

| Classification | Evidence-based conclusion |
| --- | --- |
| Regression introduced by PR #3224 | **Not supported**: actual checkout difference is documentation-only and relevant code blobs are identical. This does not prove all external inputs are identical. |
| Synchronization implementation defect | **Reproduced independently of the hosted failure**: one-shot gate replacement does not tolerate a transient Windows sharing denial, and the parent hides child exit until timeout. Separate repair and positive/negative regressions are justified. |
| GitHub Actions host problem | **Not established**: region/timing differs, but neither image/package drift nor host overload/permissions/antivirus cause is proved. Do not relabel the failure as infrastructure-only. |
| Unsatisfied test condition | **Observed**: startup did not publish a validated child/grandchild tree within the required waits. Core and ACL prechecks passed; the missing helper traceback leaves the lower-level reason unresolved. |
| Root cause of this particular failed execution | **Undetermined**: the reproduced sharing race is compatible with the symptom, but no helper WinError/traceback or captured overlap binds it to this job. |

**Recommendation: HOLD merge of PR #3224 until the applicable checks for its
corrected HEAD are assessed.** Local document integrity and a separately
reproduced defect do not override the historical red SCM result. A future
green run must be evaluated for its exact source/checkout; it does not by
itself prove why the earlier run failed. The separate repair requires its own
tests and live Windows integration result. No production fix is included in
this documentation PR and no automatic merge is performed.

Local correction checks pass: **869** existing contract tests, mandatory
all-files ruff-format, Tier A Ruff, full-tree Betterleaks and final changed-path
scan; regenerated packet manifest **36/36**, historical manifests
**22/22, 4/4, 6/6, 19/19**, all **6 JSON** documents and relative file links.
The repair branch's separate Linux selection passes **175 tests / 3 Windows
skips** and includes an actual reader-handle regression to run on Windows.
These local results do not supply the pending hosted SCM result for either PR.

```text
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
```
