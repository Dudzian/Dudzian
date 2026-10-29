# Windows SCM process-tree PID publication review — 2026-10-29

Source baseline: `cad53867231a1a6e49ea770c07043b417af84449`. This change fixes
the Stage-5 Windows acceptance harness's PID handoff. It does not change the
production Observer, Stage 9 contracts, qualification gates or timeout values.
Codebase Memory tools/executable were unavailable; the conclusions below use
direct source reads and deterministic portable regressions.

## Reproduced implementation defect

[`ChildJob._wait_for_grandchild`](../../deployment/windows_process_tree.py)
polls `process-tree.gate` with `Path.read_text` every 50 ms. The
[child helper](../../deployment/windows_process_tree_child.py) publishes the
grandchild PID by replacing that same destination. On Windows, an overlapping
reader can prevent replacement with a sharing/access error. Previously the
helper made one `os.replace` attempt, so a transient lock could terminate it
after spawning its grandchild. The parent did not check the child's exit and
would eventually report `grandchild creation deadline expired`.

The positive regression invokes the actual helper `main` with a deterministic
first-attempt `PermissionError` carrying WinError 5 or 32; replacement would
succeed on the next attempt. Loading the original baseline helper in memory
produces **two expected failing cases**. The fixed helper publishes the PID
and completes both cases. This is an implementation regression test, not a
claim that an injected error was observed on a physical machine.

The fix retries only those two Windows errors and uses the helper's original
absolute twenty-second gate deadline. It neither creates a fresh publication
timeout nor extends the budget. Other errors propagate immediately; persistent
sharing failures still propagate when that deadline is exhausted. The parent
now rejects a child that exits before publishing a positive PID, including its
exit code. Exact Job Object membership is still checked after publication.

The existing rollback remains responsible for closing the kill-on-close job,
bounded child waiting and transient removal. Regression coverage verifies that
the new early-exit path enters that rollback without falsely claiming a ready
tree or replacing job containment with manual child termination.

## Observed CI failure and attribution limit

The failed [Platform job 123422984959](https://github.com/Dudzian/Dudzian/actions/runs/40866365166/job/123422984959)
passed **790 Core tests**, then failed `SCM_STAGE0_5_AND_STAGE7_SHUTDOWN` at
`PROCESS_TREE_START`. The service log records
`ProcessTreeError: grandchild creation deadline expired` for PID 2784.
The diagnostic observed SCM stopped/PID 0 and absent health/tree/gate markers.
It does not include the helper's traceback or establish that every cleanup
assertion passed. Generic runner orphan cleanup is not that proof.

The preceding [successful Platform job 123330819643](https://github.com/Dudzian/Dudzian/actions/runs/40838999279/job/123330819643)
passed the same phase. Direct comparison of their actual synthetic checkout
trees (`ab71a924a2139f3218d4264473d27f50221b7378` and
`87f19583b0828417904aecba9b04a3415a21ce91`) found only 20 added documentation/
evidence files; executable code, workflow and dependency files were identical.
Their logs report the same runner/image versions, Python 3.11.9 and installed
package versions, including pywin32 312, on different hosted instances.

These facts neither prove a deterministic code-change regression in the docs
PR nor prove a runner fault. The one-shot publication defect is independently
reproduced and repaired here; **its attribution to that specific CI failure
remains unproven**. No arbitrary timeout increase, accepted failure, skipped
gate or manufactured PASS is used as a remedy.

## Validation boundary

The focused portable suite passes **47 tests**, including transient/permanent
sharing, non-Windows/unrelated permission errors, reuse of the original
deadline, child-exit detection and rollback. Targeted Ruff and diff whitespace
checks pass. One additional Windows-only test holds an actual destination
reader through the first replacement attempt, then releases it and verifies
publication recovery; it is skipped on the cloud Linux host. It starts no
service or real child process. The broader process-tree, acceptance, Stage-5
logging and service-recovery selection passes **175 tests**, with **3
Windows-related skips** on Linux. The mandatory exact all-files ruff-format
hook, Tier A and targeted Ruff, configured mypy (**359 source files**) and
full-tree Betterleaks pass. Final staged-path scanning and diff whitespace
checks are required before publication. No scanner scope or baseline changed.
Native Windows CI remains necessary to validate the actual SCM and Job Object
path for the fix's final HEAD. Portable fakes do not qualify
Windows deployment, physical shutdown, Stage 3B or the production Observer.
