# Ubuntu DEB Zstandard experiment V2: bounded frames, unsupported TAR inventory

Date: **2026-10-29 UTC**. Base: `c6b0ca36c5b2b8c3506eb32c5fb5b5419c3a3b9a`.
This is an inactive, read-only experiment. It does **not** establish completion
of V2, an approved source, target-image correspondence or production permission.

## Acquisition and separately frozen expectations

The four exact Ubuntu 24.04 `noble-updates` CPython
`3.12.3-1ubuntu0.17`, `amd64` DEBs were downloaded from the HTTPS URLs retained
in the adjacent [inventory JSON](stage9_observer_3a_real_deb_inventory_v2.json).
All full-archive SHA-256 values and byte lengths matched the records already
retained in [archive verification V1](stage9_observer_3a_cpython_archive_verification_v1.json):

| Package | Bytes | SHA-256 |
| --- | ---: | --- |
| python3.12-minimal | 2334634 | d452689b9660845345a4c3e05e4ad82c082d5474e04031b7aa47f1a6d5610a6e |
| libpython3.12-minimal | 838536 | d646ad7112b5adec21ba0e1af015f04ae8ca7f5efdc619622547dd6673c4c14b |
| libpython3.12-stdlib | 2070530 | 45d3f530ba1f9d6e879ad46b92046fabab13fe50a82450e4f65557a3bbad1489 |
| libpython3.12-dev | 5683746 | c123ddab7763e45199e21b4ac8545e705f20a0315b9ada4c236b52fb088c4e16 |

No DEB was installed. No maintainer script, target interpreter, imported target
module or package binary ran. Downloads used the inherited proxy and CA trust.
This session did not repeat V1's signature verification or accept its bootstrap
key/channel. `SOURCE_AUTHENTICITY=UNVERIFIED` remains unconditional here.

Before the production decoder experiment, a separate scratch inspection used
Python 3.11.16 and `zstandard==0.25.0`, bundled libzstd 1.5.7. It checked the
archive size/hash first, walked all raw `ar` records, streamed both TAR bodies,
checked each header checksum and recorded every raw member's name, type, size,
owner, mode, timestamp, link target and format marker. Regular-member SHA-256
values and complete decoded-frame SHA-256/byte counts were frozen separately.
The frozen JSON SHA-256 is
`30f1e21955bca2a54768a0da2f897b1fba0423996d8cfb274408e8a83b0136a3`.

Classification: **SAME_ORIGIN_OBSERVED_COMPARISON_ONLY**. Expectations were
observed from these same pinned upstream artifacts in a prior step, using the
same decoder implementation family. The persisted declaration is useful for
repeatable byte/inventory comparison; it is **not independent proof of
correctness, source authenticity, an approved expected image or decoder trust**.
The measurement script never generates or updates these expectations.

## Actual archive and frame structure

Every archive consists of precisely three ordinary `ar` records:
`debian-binary` (`2.0\n`), `control.tar.zst`, `data.tar.zst`. Raw 60-byte headers,
member offsets/lengths and frame hashes are retained in the JSON. There are no
`ar` long-name/string-table extensions in these four observed archives.

All eight frames have frame checksums, dictionary ID zero and ordinary Zstandard
magic. Control frames declare decoded sizes of 20480, 30720 or 40960 bytes. Every
data frame omits the decoded-size declaration and requests an 8388608-byte
window. The production decoder must therefore support unknown decoded lengths
while enforcing output and window limits throughout the read.

| Package | Control TAR bytes | Control regular/directory | Data TAR bytes | Data regular/directory/symlink |
| --- | ---: | --- | ---: | --- |
| python3.12-minimal | 20480 | 6 / 1 | 8099840 | 7 / 13 / 1 |
| libpython3.12-minimal | 30720 | 6 / 1 | 5355520 | 283 / 24 / 1 |
| libpython3.12-stdlib | 40960 | 3 / 1 | 9318400 | 346 / 42 / 2 |
| libpython3.12-dev | 30720 | 2 / 1 | 29788160 | 235 / 24 / 3 |

The observed typeflags are only `0`, `5`, `2`; no PAX, long-name, sparse, device,
FIFO, hard-link or other typeflag was observed in these exact four archives.
Every TAR header, including control headers, uses GNU magic/version bytes
`75 73 74 61 72 20 20 00` (`ustar` + space + space + NUL). The existing strict
POSIX reader requires `75 73 74 61 72 00 30 30` (`ustar` + NUL + `00`).
Consequently the first control header already refuses `UNSUPPORTED_SOURCE`.
Adding a Zstandard decoder cannot make the complete archive supported.

All observed symlinks, without following or materializing any of them:

| Package | Member (without leading `./`) | Literal target |
| --- | --- | --- |
| python3.12-minimal | usr/share/doc/python3.12-minimal/changelog.Debian.gz | ../libpython3.12-minimal/changelog.Debian.gz |
| libpython3.12-minimal | usr/lib/python3.12/sitecustomize.py | /etc/python3.12/sitecustomize.py |
| libpython3.12-stdlib | usr/lib/python3.12/_sysconfigdata__linux_x86_64-linux-gnu.py | _sysconfigdata__x86_64-linux-gnu.py |
| libpython3.12-stdlib | usr/share/doc/libpython3.12-stdlib | libpython3.12-minimal |
| libpython3.12-dev | usr/lib/x86_64-linux-gnu/libpython3.12.a | ../python3.12/config-3.12-x86_64-linux-gnu/libpython3.12.a |
| libpython3.12-dev | usr/lib/x86_64-linux-gnu/libpython3.12.so | libpython3.12.so.1 |
| libpython3.12-dev | usr/share/doc/libpython3.12-dev | libpython3.12-stdlib |

The frozen inventory is complete as an **observational raw-member inventory**.
The regular-only adapter declaration cannot represent these links. Supplying
all 871 regular data members across four packages cannot establish a supported,
closed package inventory when seven additional symlink members exist.

## Metadata is not an approved installation mapping

All observed TAR member UID/GID values are `0/0`. Directories use `0755`, regular
files use `0644` or `0755`, and symlinks use `0777`. Executable maintainer scripts
are merely hashed observations; execute permission does not permit running them.
These are archive metadata, not observations of an installed Ubuntu filesystem.

The manifest describes target image UID/GID/mode. The offline verifier measures
those declared target records on its explicitly selected staging filesystem.
Neither a matching archive hash nor an inert archive member observation proves
how installation changes ownership, modes, links, directory sharing or files
written by maintainer scripts/triggers. In particular, root-owned package records
do not justify silently changing a non-root staging tree, comparing against a
caller-chosen normalization rule, or accepting symlink metadata as regular-file
metadata. No transformation, installation or approval was added.

`ARCHIVE_TO_TARGET_METADATA_MAPPING=NOT_YET_PROVEN`.

## Minimal separate proposal, requiring review

A follow-up should specify a strictly inert GNU-header subset and a separately
reviewed package inventory model before changing parser acceptance. It should:

1. Define exact accepted GNU marker/version combinations and the semantics of
   bytes 345–511. GNU extension layout must not be mistaken for the POSIX prefix
   field. Reject nonzero unsupported sparse, continuation and extension fields;
   keep checksums, numeric fields, byte bounds and closed type inventories.
2. Represent and compare literal link records without following or extracting
   them. Define whether any such source record may map to a target contract
   record, including absolute `/etc/...` targets, `..` components, dangling and
   cross-package targets, link chains, cycles, path-prefix aliases, duplicate
   names, ordering and replacement. A link target is data, not permission to
   access the host filesystem. Link acceptance must not fill missing directories
   or weaken the closed target image inventory corrected in PR #3226.
3. Specify approved archive-to-target UID/GID/mode mapping independently of
   archive byte matching. Account for package installation transforms,
   maintainer-script and trigger outputs without executing those during offline
   verification. Explicit independent acceptance remains mandatory.
4. Add adversarial negative tests and complete real-package declarations before
   claiming full package support. No automatic format fallback or production
   runtime dependency follows from this proposal.

These are proposed review topics, not an authorization or implemented contract
change. Existing links and GNU formats continue to refuse.

## Production-adapter experiment and reproduction

After the expectation JSON had been frozen, the read-only
[`scripts/verify_stage9_real_deb_v2.py`](../../../scripts/verify_stage9_real_deb_v2.py)
measurement imported the existing adapter's `_Region`, `_ZstdReader` and
`verify_ubuntu_archive`. It checked each entire archive's hash/size again before
reading any frame. All raw `ar` headers matched the separately recorded values.
For each control/data frame it drained the production decoder completely,
checked decoded SHA-256/size against the frozen record, enforced a shared output
budget across all packages and rechecked descriptor/path identity after reads.
No alternate TAR parser or fallback was used by the measurement.

All **8/8** complete frames passed, including their checksums and exact compressed
range termination, decoding **52684800 bytes** in aggregate. This is a decoder
experiment on real DEB members, **not a successful full real-DEB parse**. Calling
the production adapter with every frozen regular data member and empty image
projection refused **4/4** archives with `UNSUPPORTED_SOURCE` on their first GNU
control TAR header. Links would remain unsupported even after that obstacle.
The [retained result JSON](stage9_observer_3a_real_deb_results_v2.json) records:

```text
REAL_DEB_ZSTD_PARSED=PASS                 # control/data compressed frames only
REAL_DEB_MEMBER_INVENTORY=UNSUPPORTED     # complete package parser refused
REAL_DEB_BYTE_CONSISTENCY=INCOMPLETE      # no full package/member MATCH
SOURCE_AUTHENTICITY=UNVERIFIED
ARCHIVE_TO_TARGET_METADATA_MAPPING=NOT_YET_PROVEN
V2_COMPLETE=NO
```

To repeat without network access, first provision the exact four archives under
an explicitly selected directory through a separately reviewed acquisition
process. From the repository root, with the pinned offline verifier environment:

```sh
python -m scripts.verify_stage9_real_deb_v2 --directory /path/to/exact-debs
```

The command has no acquisition/update mode. It pins the expectation JSON digest,
refuses archive hash/size differences, and exits **2** while full package inventory
is unsupported/incomplete. It can select one package with `--package`. A frame
`PASS` never makes the overall command an authorization decision. The downloaded
DEBs and scratch inspector are deliberately not repository fixtures; their
identifiers and the complete separately frozen comparison inventory are retained.

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
