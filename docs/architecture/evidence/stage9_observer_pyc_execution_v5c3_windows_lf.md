# Byte-pinned fixture checkout audit

Read-only source audit and disposable Git checkout simulation on Linux. This is not native Windows execution. No fixture, C reference, vector or recorded expected digest was modified.

Scope: all 33 tracked files under tests/fixtures were enumerated. Byte-sensitive V5c2 inputs and the analogous V4b detached-signature/keyring fixtures were checked against their integrity tests and manifests; both exact review patches were also included. V5c3 tests construct source/output/runtime bytes synthetically in temporary directories rather than reading an additional frozen fixture.

Simulation: exact origin/main Git blobs were inserted directly into fresh indexes with hash-object/update-index (no text decoding), then git checkout-index --all --force ran with core.autocrlf=true. Compared baseline origin/main .gitattributes with current file-specific C LF rule. System/global Git configuration and system attributes were disabled. Both indexes contain the original fixture blobs.

Selected files: 15. Baseline: only the C reference changed to CRLF; every other selected byte-sensitive file matched its blob. With the exact C rule: all 15 files matched byte-for-byte.

C reference:

- Stored LF SHA-256: 3b3686ea5c9187d5ee36ff4f69db159775b5e9f4462eecb95d1195d280e7c488
- Baseline CRLF checkout SHA-256: 65512132c96579b8df11f8877f62fc1db3a9073fc205cc6dfb89907299c54d06
- Fixed checkout SHA-256: 3b3686ea5c9187d5ee36ff4f69db159775b5e9f4462eecb95d1195d280e7c488

Existing protection:

- source_materials.json and golden_vectors.json: *.json text eol=lf; source_materials matches retained vector/source pin.
- V4b Release, *.asc and *.gpg: explicit -text rules; all 9 manifest file digests verified against original blobs and both checkout results.
- Both review patches: explicit file-specific text eol=lf rules; exact bytes preserved in both checkouts.
- V5c3: no additional committed byte-pinned source/binary fixture to protect.

Finding: no additional .gitattributes rule is justified within this checked fixture scope. Retain only the exact C-reference text eol=lf addition. No mass normalization or expected-hash rewrite is needed.


| Path | Baseline blob identity | Fixed blob identity |
| --- | --- | --- |
| `tests/fixtures/stage9_observer/pyc_source_hash_v5c2/cpython_siphash13_reference.c` | CRLF_CHANGED | MATCH |
| `tests/fixtures/stage9_observer/pyc_source_hash_v5c2/source_materials.json` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/pyc_source_hash_v5c2/golden_vectors.json` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/Release` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/allowed-keyring.gpg` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/allowed.asc` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/wrong-subkey.asc` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/weak-hash.asc` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/expired.asc` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/revoked-keyring.gpg` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/unknown-keyring.gpg` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/unknown.asc` | MATCH | MATCH |
| `tests/fixtures/stage9_observer/apt_v4b/manifest.json` | MATCH | MATCH |
| `tests/fixtures/account_genesis_verified_transport_review.patch` | MATCH | MATCH |
| `tests/fixtures/freshness_verifier_remediation_review.patch` | MATCH | MATCH |
