# Observer 3A V4b — actual offline TEST validation, 2026-11-14

Base: `dcc71020c14ff1c09c49cd8be99f7740e490a818`, current remote `main`.
GitHub confirmed [PR #3269](https://github.com/Dudzian/Dudzian/pull/3269)
merged at `2026-11-14T12:10:11Z`; the merge commit was fetched and its ancestry
checked before creating `feat/stage9-observer-apt-v4b`. The remote main ref was
checked again before delivery and still identified that commit. No merge or
automatic merge was requested or performed.

This report covers the [technical contract](../stage9_observer_apt_v4b.md),
the new application boundary, synthetic TEST fixtures and local validation.
It does **not** mark complete V4b, source authenticity, external trust, host
qualification or production acceptance as completed. V4a and V1–V3 source and
historical evidence remain unchanged. No V5 or Stage 3B implementation, package
installation, maintainer-script execution or production-policy change occurs.

## Implemented cryptographic verification

`verify_detached_v4b` invokes an explicitly selected ELF `gpgv` whose exact bytes
match a SHA-256 pin. The executable, Release, detached signature and public
keyring are copied into kernel-sealed memory descriptors. Arguments use no
shell, default user key home or imported global keys. The environment contains
only `LC_ALL=C` and a fresh private `GNUPGHOME`. The verifier does not acquire
keys or contact a keyserver. Genuine `gpgv` has no key-retrieval facility.

Three separately read output channels share a cumulative bound and deadline.
Only the dedicated machine channel supplies evidence; stdout refuses and human
stderr is bounded and discarded. A zero exit must accompany exactly one
coherent `NEWSIG`/`SIG_ID`/`GOODSIG`/`VALIDSIG` sequence. Unsupported algorithms,
ambiguous signatures, missing/contradictory statuses and process/input failures
refuse. A process group is terminated/reaped during cleanup, including after
failure. These are application controls, not accepted OS confinement.

The public `verify_apt_signature_v4b` passes the same immutable Release bytes
object to the signature adapter and V4a Release parser. It checks the entire
Release → index → DEB chain and V3 control/data inventories. Whole-DEB hashing
and archive parsing use the same sealed copy. Original no-follow input
descriptors/inventory are checked for observed changes. Any local failure,
including final inventory, closing, sealing or deadline failure, discards all
cryptographic evidence, chain results and reported Release digest. Both local
claims then stay `UNVERIFIED`; this is not a statement that an unexamined
signature is mathematically invalid.

## Implemented signer-policy enforcement

Versioned `TrustContextV4b` declares exact primary/signing fingerprint pairs,
an explicit subkey policy, allowed public-key/digest pairs, context/signer
validity and revocation, repository/snapshot labels, exact Release/keyring
digests, local time comparisons and external approval/freshness/checkpoint/
rollback references. Wrong subkeys do not inherit their primary's permission.
Only explicit RSA/EdDSA with SHA-256/SHA-512 pairs are supported.

Local policy checks include signature expiry against the supplied evaluation
time as well as creation age/skew, validity intervals and revocation. These
comparisons enforce caller-supplied rules; they do not authenticate their
origin or clock. References are never resolved online or promoted to approval.
Complete local success returns `cryptographic_signature=VALID` and
`signer_policy=MATCH`, alongside a separate V4a snapshot retaining
`signature_verification=NOT_PERFORMED`, `source_authenticity=UNVERIFIED` and
all historical qualification blocks.

## Unfulfilled external trust requirements

No accepted delivery or authentication mechanism for an authorized external
authority was identified or introduced. The API has no caller-set approved
boolean. A missing authority reference and a syntactically complete reference
both leave `external_authority_acceptance=UNVERIFIED`. A missing trust context
refuses the local operation. TEST public keys and local pins never become
production roots. No real repository's publisher authorization was verified.

## Missing freshness, checkpoints and rollback

`freshness=BLOCKED`, `checkpoint=BLOCKED` and `rollback=BLOCKED` remain fixed.
Caller timestamps and reference strings do not implement authenticated time,
accepted Release Date/Valid-Until semantics, external checkpoint provenance,
durable monotonic state or rollback prevention. Their approved rules and
independently verifiable delivery/state mechanisms remain required. A valid
old signature cannot establish that a snapshot is currently authorized.

## Independent tool and isolation acceptance still required

`tool_acceptance=UNVERIFIED` and `system_isolation=UNVERIFIED` are fixed. An
arbitrary caller-selected ELF can satisfy its own caller-provided pin; that
does not establish it as independently approved genuine `gpgv`. Dedicated
status parsing prevents channel confusion, not a malicious tool from lying.
The kernel, host, dynamic loader/libraries and tool/runtime closure require
separate acceptance. This change does not certify a kernel network sandbox,
syscall confinement, hard RSS limit, interruption of blocked filesystem calls,
or protection against a hostile process controlling the same Python runtime.
Sealed copies bind the bytes compared in this invocation; original stat
monitoring does not establish future immutability of source paths.

## Tests actually run

Environment: unprivileged Linux, CPython 3.11.17, pytest 9.2.0, repository
`.venv`. Only synthetic inert package payloads and TEST signing keys were used.
The existing V2/V3 reproduction-tool regression tests use synthetic archives or
pre-I/O refusals; no actual package was installed or executed. Disposable TEST
key generation used private temporary homes, terminated its temporary agent
and deleted private state. Only TEST public artifacts remain committed.

| Scope | Actual result |
| --- | --- |
| New integration tests, including eight genuine signed complete APT-chain cases | 88 passed |
| Dedicated real-gpgv/frozen TEST fixture and disposable signer tests | 11 passed |
| Adversarial status/process tests using real TEST child processes | 98 passed |
| Existing Observer V1–V4a and synthetic V2/V3 reproduction-tool regressions | 1,580 passed unchanged |
| Final combined invocation of those eleven files | **1,777 passed**, zero skipped/failures, 32.10 seconds |
| Existing workflow/security/coverage inventory contracts | **181 passed**, zero skipped/failures |
| New integration branch-inclusive coverage | 99%: 257 statements, one missed; 34 branches, one partial |
| New adapter branch-inclusive coverage | 97%: 225 statements, five missed; 40 branches, four partial |
| Combined new-module branch-inclusive coverage | 98% |
| Changed executable-line coverage against `origin/main` | 98%: 482 lines, six missed; unchanged 70% gate passed |

Unexecuted module lines cover private unsupported-platform branches and rare
process-cleanup timeout/invariant paths. Win32 type checking is not execution
validation of a Windows verifier; public runtime support remains Linux only.
No test, threshold, timeout, scanner rule or manual hardware gate was disabled.

Negative cases include cryptographically damaged signatures, modified Release,
unknown identities, other subkeys, unsupported/disallowed algorithms, revoked
keys and policy entries, expired signatures and policy intervals, missing
keyring/context, duplicated signatures, forged human-channel statuses,
incomplete and contradictory records, exit zero without evidence, timeouts,
all output channels and aggregate output overflow, process crashes, changed
or removed executable, observed Release/index/DEB mutation, rehashed invalid
inner DEB data, failed sealing and readback, final inventory/cleanup/deadline
failures and failure after an earlier valid package. Real signature tests and
fake-process protocol tests are separate; the latter do not prove cryptography.

Two source-review findings were corrected before final validation: missing
expiry comparison against caller policy time, and refusal labels that could
have implied cryptographic invalidity from an unrelated chain failure. Final
bounded source/test review found no additional confirmed blocking defect in
the inspected new implementation. This is not an exhaustive security proof.

Reproduction (select the eleven files listed by the table's scopes):

```sh
STAGE9_TEST_GPGV=/absolute/local/gpgv \
COVERAGE_FILE=work/validation/final.coverage \
.venv/bin/python -m pytest -q \
  tests/test_stage9_observer_contract.py \
  tests/deployment/test_stage9_observer_offline.py \
  tests/deployment/test_stage9_observer_zstd.py \
  tests/deployment/test_stage9_observer_tar_v3.py \
  tests/deployment/test_stage9_observer_package_v3.py \
  tests/deployment/test_stage9_observer_apt_v4a.py \
  tests/deployment/test_stage9_real_deb_v2.py \
  tests/deployment/test_stage9_real_deb_v3.py \
  tests/deployment/test_stage9_observer_apt_v4b.py \
  tests/deployment/test_stage9_observer_apt_v4b_crypto.py \
  tests/deployment/test_stage9_observer_gpgv_v4b.py \
  --cov=deployment.stage9_observer_apt_v4b \
  --cov=deployment.stage9_observer_gpgv_v4b --cov-branch \
  --cov-report=xml:work/validation/final-coverage.xml \
  --junitxml=work/validation/final-junit.xml
```

The exact local test verifier was `gpgv` 2.4.7, binary SHA-256
`3f29dddc10e4089aeac5b2675313f4e5fe822abe5ab3bf7898d238c32d758304`.
It was separately extracted under `work/tools`, with no system installation;
the [fixture README](../../../tests/fixtures/stage9_observer/apt_v4b/README.md)
records the HTTPS snapshot source and matched package/metadata hashes. This
reproducibility record does not approve the tool or bootstrap APT trust.
The adapter/tests never download tools or keys. Tests explicitly skip if a
required local test tool is unavailable; the reported local run had no skips.

## Quality and security validation

Both modules join configured mypy and existing Ruff/format/pytest/coverage
scopes. Gates and protected hardware/manual workflows remain unchanged.

| Check | Actual result |
| --- | --- |
| Exact `python -m pre_commit run ruff-format --all-files` after staging | Passed twice; no unrelated formatting churn |
| Workflow's full Ruff baseline and new-code strict E/F/B/I/format | Passed |
| Configured mypy Linux and `--platform win32`, no filename narrowing | Passed, 399 files each |
| TOML/YAML parse and diff whitespace | Passed |
| Semgrep CE 1.179.0 configured blocking scope | Five rules, 1,131 Python targets, zero findings |
| Semgrep rule fixtures | 5/5 passed |
| Maintained community rules over the two new modules | 200 applicable rules, zero findings/errors |
| Betterleaks 1.9.0 full working-tree scan | Passed, zero findings; unchanged baseline/config |
| Betterleaks existing staged wrapper after final documentation | Passed, zero findings |
| pip-audit canonical desktop lock | 104 dependencies, zero known vulnerabilities, zero skipped |
| pip-audit exact Observer Zstandard lock | One dependency, zero known vulnerabilities, zero skipped |
| Existing `.venv` `pip check` | Passed |

The initial broad Semgrep invocation inadvertently included the rule fixtures
and reported their expected bad-code examples. The configured source scope
above was then run unchanged and passed. No repository-wide clean advisory
scan is claimed. Codebase Memory was unavailable; direct source inspection
and tests supplied evidence without structural-completeness claims.

The first Betterleaks scan found TEST metadata/fingerprint labels and two
generated cache artifacts. TEST labels/manifest structure were clarified,
with public key/signature bytes unchanged and no scanner suppression. The two
generated `.pyc`/pytest-nodeid cache files were removed; the identical full
working-tree scan then passed. Temporary tooling and logs remain local under
`work/`, outside the commit. Security-tool setup/audits can access registries;
that is separate from the offline verification API.

Validated module hashes:

```text
de3b2a0d70b71f5af7ea74b7f3b1044158fbc44923fe5b566d8215cb525e3741  deployment/stage9_observer_apt_v4b.py
bdb3dd5d759504dea2ab19620f2c50fe414ce4de3eef7dd05422f1126a405c63  deployment/stage9_observer_gpgv_v4b.py
```

The Draft PR records the published commit and exact-HEAD remote workflow
snapshot separately. Local passes do not imply overall remote CI success or
independent trust acceptance.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```
