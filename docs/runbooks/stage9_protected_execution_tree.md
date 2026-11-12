# Stage 9: separate protected execution tree

Status: procedure prepared; physical installation and VM qualification **BLOCKED**
in this cloud environment. `S3=BLOCKED`, `PRODUCTION=NO_GO`. A successful data-only
export or a green unit test does not change these gates.

The original checkout is an editing workspace. Never recursively `chown` or
`chmod` it, and never execute a transport TAR as an installed harness. The new
`scripts/prepare_stage9_protected_tree.py` has deliberately separate actions:

* `export` reads explicit regular Git blobs from one full commit into a new,
  untrusted candidate. It never reads corresponding working-tree file contents,
  imports candidate modules, or changes the checkout. Directory, symlink,
  submodule, missing-path and traversal selections are refused. Commit, tree and blob IDs
  are checked against Git's object encoding along every selected ancestry path
  before SHA-256 inventory creation.
* `install` requires native Linux root and an independently supplied SHA-256 pin
  for the inventory. It reads candidate files as bounded, non-following data,
  validates the entire candidate inventory, then copies the authenticated bytes
  into a previously absent tree under root-owned protected ancestors. It never
  executes candidate bytes or Git. An existing target or existing installed
  inventory causes refusal; replacement needs a separate reviewed change.

The provenance collector is a separate read-only administrator tool. It must
run from its own independently trusted location; it is not added to this
harness tree, whose executable source set is fixed.

The candidate is transport data: its directories retain owner write permission,
so it cannot pass protected execution verification. The installed tree uses
root ownership, directories `0555`, regular files `0444`, and only files marked
executable by the exact Git tree use `0555`. No symlink, hardlink, special object,
extra file, empty directory, package directory or import cache is accepted.
The inventory records the root `.` and every intermediate directory, not just
Python files. Its schema is exactly:

```json
{"schema":1,"source_revision":"<full 40-hex commit>","tree_root":"<absolute final path>","objects":[{"path":".","kind":"directory","mode":365},{"path":"deployment","kind":"directory","mode":365},{"path":"deployment/stage9_vm_trial.py","kind":"file","mode":292,"sha256":"<64 hex>"}]}
```

Numbers above are decimal representations of `0555` and `0444`. The example is
structural documentation, not a complete harness inventory or approval. Files
and directories are sorted by path. The installed external inventory lives at
`/etc/cryptohunter/stage9-harness-inventory.json`, outside the execution tree.
It binds both the exact commit and final absolute root. A hash printed by the
exporter is an identifier, not independent authorization.

## Administrator procedure (not executed here)

1. Independently review and authenticate the exact remediation commit and the
   exporter source. The audited starting point was
   `fc15b98298c5430286dd2206ec7eeb8f20e2497b`; execution of the fix must use the
   reviewed remediation commit, not this vulnerable starting point. Use an
   already trusted system interpreter and an independently verified protected
   copy of the exporter for the root phase. Do not run a script copied from the
   untrusted candidate as root. Record the tool digest, commit and interpreter
   provenance. This procedure creates no key, signature or approval.
2. The reviewed source set is the following 18 exact paths, obtained by parsing
   `HARNESS_MODULES` and `HARNESS_HELPERS` in the remediation source as AST data.
   Store these lines as `$S9_REVIEWED_PATHS`. Review equality against the final
   reviewed commit's constants and subprocess/`runpy` paths. Additions require a
   new code review of the fixed loader contract; the loader rejects extras even
   when an inventory lists them. There are no package initializers or caches.

   ```text
   bot_core/freshness_service_identity.py
   bot_core/linux_service_ipc.py
   bot_core/linux_service_postgresql.py
   bot_core/production_signer_ipc.py
   deployment/freshness_application.py
   deployment/freshness_runtime.py
   deployment/freshness_runtime_bootstrap.py
   deployment/linux_physical_evidence.py
   deployment/linux_physical_ipc.py
   deployment/linux_physical_native.py
   deployment/linux_physical_observations.py
   deployment/linux_physical_postgresql.py
   deployment/linux_release_identity.py
   deployment/production_local_installer.py
   deployment/production_local_installer_host.py
   deployment/stage9_vm_environment.py
   deployment/stage9_vm_native.py
   deployment/stage9_vm_trial.py
   ```

   These 18 `.py` files have final mode `0444`; root, `deployment` and `bot_core`
   directories have mode `0555`. No glob, directory shorthand or runtime import
   discovery substitutes for this list. Preserve it as review evidence;
   omitting a required file must cause refusal, not a synthesized replacement
   inventory during execution. The performance tool and provenance collector
   execute separately only after independent authentication and review.
3. As an unprivileged builder, use a fresh output directory outside the source
   checkout. The following command uses operator-supplied paths and a full
   reviewed commit. The `--tree-root` value must already identify the intended
   new installation location. The script accepts no TAR input.

   ```bash
   mapfile -t s9_reviewed_paths < "$S9_REVIEWED_PATHS"
   s9_path_arguments=()
   for s9_path in "${s9_reviewed_paths[@]}"; do
       s9_path_arguments+=(--path "$s9_path")
   done
   /usr/bin/python3.12 -I -S -B "$S9_VERIFIED_EXPORTER" export \
       --source "$S9_SOURCE" --revision "$S9_REVIEWED_COMMIT" \
       --candidate "$S9_CANDIDATE" --tree-root "$S9_NEW_TREE" \
       --inventory "$S9_CANDIDATE_INVENTORY" "${s9_path_arguments[@]}" \
       > "$S9_EXPORT_LOG" 2>&1
   s9_export_exit=$?
   printf '%s\n' "$s9_export_exit" > "$S9_EXPORT_EXIT_FILE"
   test "$s9_export_exit" -eq 0 || exit "$s9_export_exit"
   ```

4. Independently compare every inventory byte/hash and executable mode against
   the authenticated Git tree and reviewed file list. Review directory entries
   and complete path equality, including extra objects. Check the intended root.
   Supply the resulting inventory digest through the existing independent
   administrative review channel. Do not trust a digest merely because it came
   in the same TAR or directory as the payload.
5. On the separately authorized native host/VM, confirm the system interpreter,
   loader, libc/OpenSSL and exporter provenance independently. Use existing
   root-owned ancestors with no group/other write permission; create only new
   installation parents if needed. Confirm that the source checkout will remain
   untouched. Review that `/etc/cryptohunter/stage9-harness-inventory.json` does
   not already exist; this tool will not replace it.
6. Run the protected exporter copy as root, supplying the independently reviewed
   digest. Preserve stdout/stderr and the actual exit code:

   ```bash
   /usr/bin/python3.12 -I -S -B "$S9_PROTECTED_EXPORTER" install \
       --candidate "$S9_CANDIDATE" --inventory "$S9_CANDIDATE_INVENTORY" \
       --inventory-sha256 "$S9_INDEPENDENT_INVENTORY_SHA256" \
       > "$S9_INSTALL_LOG" 2>&1
   s9_install_exit=$?
   printf '%s\n' "$s9_install_exit" > "$S9_INSTALL_EXIT_FILE"
   test "$s9_install_exit" -eq 0 || exit "$s9_install_exit"
   ```

7. Run the protected exporter's read-only verification command, preserving its
   log and exit code in the same way:

   ```bash
   /usr/bin/python3.12 -I -S -B "$S9_PROTECTED_EXPORTER" verify \
       --inventory-sha256 "$S9_INDEPENDENT_INVENTORY_SHA256"
   ```

   It checks the fixed inventory's root owner, exact `0444` mode, single link
   and protected ancestors before interpreting the pinned bytes, then checks
   the complete tree. Confirm root ownership, exact modes, no
   symlink/hardlink, exact bytes, directory equality and parent protection.
   Compare the original checkout's inventory and metadata before/after; it
   must be unchanged. The harness must read its fixed protected external
   inventory and reject a mismatched `--revision`, root or source graph before
   running any helper. Capture interpreter command lines, real process exit
   codes and kernel/VM evidence according to the existing manual VM gate.
8. Test missing/changed files, extra empty directories, package substitution,
   `.so` substitution, import-cache files and unauthorized dependency source in
   a separate disposable copy. Never execute the substitute code to prove
   rejection. A refusal must occur before the marker payload can execute.
   Do not modify the installed tree or generate a replacement approval for a
   negative test. Root-only partial outputs from an interrupted/failed install
   require manual investigation; the tool does not delete, reuse or repair them.

The operation relies on the administrator's protection of the Git/authentication
inputs, trusted Python/system code and final ancestors. It does not defend
against a malicious root administrator, hostile kernel, untrusted filesystem
implementation or an unqualified host. Local tests cover unprivileged Git/data
operations and refusal paths, not root metadata enforcement on the target VM.
The host/VM steps remain **BLOCKED** until performed and reviewed separately.
