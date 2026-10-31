# V3 raw DEB observation and separately frozen package inventory

Date: **2026-10-31 UTC**. Source baseline:
`4e5044ac1bff8f078cb27659348f0b2e41b3a4ac`.

This is **SAME_ORIGIN_OBSERVED_COMPARISON_ONLY;
NOT_INDEPENDENT_ACCEPTANCE**. It records an acquisition and a raw observation,
not a production-verifier result. The [observation JSON](stage9_observer_3a_real_deb_observation_v3.json)
retains URLs, exact archive and decoded-TAR hashes, counts, tool identity,
declaration persistence time and the raw GNU-tail observation.

## Acquisition and expectation freeze

The four exact Ubuntu 24.04 `amd64` CPython `3.12.3-1ubuntu0.17` DEBs were
downloaded from the HTTPS `archive.ubuntu.com` URLs retained in the historical
[V2 inventory](stage9_observer_3a_real_deb_inventory_v2.json). Downloads were a
separate preparation step using the inherited proxy and CA trust. The archives
remain scratch inputs, outside the repository; no DEB was installed or extracted.
No package program, Python interpreter or maintainer script was executed.

Each complete archive matched the historical full-archive size and SHA-256
**before** its `ar` or TAR records were inspected:

| Package | Bytes | SHA-256 |
| --- | ---: | --- |
| python3.12-minimal | 2334634 | `d452689b9660845345a4c3e05e4ad82c082d5474e04031b7aa47f1a6d5610a6e` |
| libpython3.12-minimal | 838536 | `d646ad7112b5adec21ba0e1af015f04ae8ca7f5efdc619622547dd6673c4c14b` |
| libpython3.12-stdlib | 2070530 | `45d3f530ba1f9d6e879ad46b92046fabab13fe50a82450e4f65557a3bbad1489` |
| libpython3.12-dev | 5683746 | `c123ddab7763e45199e21b4ac8545e705f20a0315b9ada4c236b52fb088c4e16` |

A separate, one-off [raw observation script](stage9_observer_3a_real_deb_inspector_v3.txt)
used host Python 3.11.17 and `zstandard==0.25.0`, C backend, libzstd 1.5.7.
It did not import the production verifier or use its TAR parser. It checked raw
`ar` headers against the retained V2 records, decoded each bounded real sample,
walked every 512-byte TAR header, checked checksums and padding, and compared
every historical record value and every decoded-TAR digest/size against V2.
The same decoder family and same upstream bytes underlie these observations;
this separation does not make expectations, decoding or source trust independent.

The new [V3 inventory](stage9_observer_3a_real_deb_inventory_v3.json) was persisted
before the first production V3 comparison in this work. Its exact frozen bytes
are **922604 bytes**, SHA-256
`aafaae5853b06b70f704a39bc6471cced92207b9ff64055b97a870f60dc10137`.
The historical V2 inventory remains unchanged, SHA-256
`30f1e21955bca2a54768a0da2f897b1fba0423996d8cfb274408e8a83b0136a3`.

The V3 declaration has an explicit `stage9-ubuntu-package-inventory-v3` format,
separate `control` and `data` declarations, the `gnu-inert-v3` profile and literal
`tail_hex` for every raw header's bytes **345–511**. Existing member values are
preserved, including raw names (`./` root), raw device-field bytes, metadata,
regular-payload digests and literal link targets. Basename-only transport
filenames do not change raw TAR member names.

## Observed GNU layout

The GNU implementation's
[`src/tar.h`, pinned source](https://github.com/gnu-mirror-unofficial/tar/blob/5c713540e34f51323b9f63b7459254bc845603ed/src/tar.h#L131)
explicitly explains that `oldgnu_header` conflicts with the POSIX prefix field.
It defines `OLDGNU_MAGIC` as `ustar` followed by two spaces and NUL. Retrieved
source: **13465 bytes**, SHA-256
`999952fcd93c4fd1fd96276f1f619e27f08ea7b052ab14db551182e29db13e97`.
This is a technical layout reference retrieved from an **unofficial mirror**;
the source download does not independently authenticate GNU, the mirror or a
verifier toolchain.

| Byte offsets, end exclusive | GNU meaning |
| --- | --- |
| 345–357 | incremental `atime` |
| 357–369 | incremental `ctime` |
| 369–381 | multivolume offset |
| 381–385 | unused `longnames` |
| 385–386 | unused byte |
| 386–482 | four sparse descriptors, 24 bytes each |
| 482–483 | sparse continuation flag `isextended` |
| 483–495 | sparse real size |
| 495–512 | remainder of the 512-byte header block |

**All 1002 observed headers have exactly 167 NUL bytes at offsets 345–511.**
This is a measurement of all control and data headers, not an assumption based
on the first member. Consequently a narrow GNU profile that rejects every
non-NUL extension byte can cover these exact archives without implementing
incremental, multivolume, sparse or continuation semantics. These GNU bytes must
never be interpreted as a POSIX prefix. Acceptance of other encodings requires
a separately reviewed profile change.

## Complete raw counts

| Package | Control regular / directory | Data regular / directory / symlink |
| --- | --- | --- |
| python3.12-minimal | 6 / 1 | 7 / 13 / 1 |
| libpython3.12-minimal | 6 / 1 | 283 / 24 / 1 |
| libpython3.12-stdlib | 3 / 1 | 346 / 42 / 2 |
| libpython3.12-dev | 2 / 1 | 235 / 24 / 3 |

Totals: data **871 regular, 103 directory, 7 symlink**; control **17 regular,
4 directory**. The total **1002** includes the explicitly recorded root
directories. Eight complete decoded TAR bodies total **52684800 bytes** and
match the old frozen TAR hashes and sizes. Every old metadata and regular hash
value also matched; V3 adds the previously unrecorded raw GNU tails.

The seven literal link records are unchanged from V2. They were not followed,
created or used to access host paths. In particular the package member
`usr/lib/python3.12/sitecustomize.py` targeting
`/etc/python3.12/sitecustomize.py` remains an inert observation.

## Limits of this evidence

This scratch inspector collected bounded real observations and is not a public
verification API, a sandbox or a qualified tool. Its retained implementation is
evidence for how the declaration was frozen; it has no download or acceptance
role in the production V3 verifier. A later production comparison must consume
these frozen declarations and report its own outcomes separately.

Matching historical and new observations does not verify APT provenance,
approve a trust anchor, establish freshness, independently accept expectations,
map package metadata into an Observer image or close runtime imports. No
target-image correspondence or production capability follows.

```text
SOURCE_AUTHENTICITY=UNVERIFIED
ARCHIVE_TO_TARGET_METADATA_MAPPING=NOT_YET_PROVEN
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
V2_COMPLETE=NO
```
