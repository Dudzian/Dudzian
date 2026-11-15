# V5c2 — exact Ubuntu CPython hash source and independent vectors

This record describes local byte measurements and source inspection on
2026-11-15. The supported declaration is Ubuntu Noble `amd64`, source package
`python3.12`, version `3.12.3-1ubuntu0.17`, CPython `3.12.3`, PEP 552 checked
hash, magic `cb0d0d0a`. No investigated interpreter or Python source was run,
imported or compiled. The exact-target interpreter comparison is `NOT_TESTED`.

The source files were retrieved as data over HTTPS from the Ubuntu archive.
Their source-distribution descriptor and archive hashes identify the reviewed
bytes. The descriptor's publisher signature was **not authenticated** against
an independently accepted key. Distribution/build authority acceptance remains
`UNVERIFIED`; these measurements cannot grant production algorithm acceptance.
The V5c2 result therefore reports local agreement separately from missing
independent algorithm evidence.

## Archive identities and patch application

Every artifact below is at
`https://archive.ubuntu.com/ubuntu/pool/main/p/python3.12/<filename>`.

| Identifier | Bytes | Measured SHA-256 |
| --- | ---: | --- |
| `python3.12_3.12.3-1ubuntu0.17.dsc` | 3916 | `941700b999737d869b80c3b5a0adc0996a0fd760b982e18551bd2104da779384` |
| `python3.12_3.12.3.orig.tar.xz` | 20625068 | `56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1` |
| `python3.12_3.12.3-1ubuntu0.17.debian.tar.xz` | 302472 | `17b15629cb10825924cf7b85055255080ec773133ae0f8b8ce5ea2eec90dd200` |

The two downloaded archive sizes and SHA-256 values agree with the exact
descriptor's `Checksums-Sha256`. The descriptor names the exact source-package
version. This is identity consistency within those retrieved materials, not
authentication of the Ubuntu signing authority or a binary-to-source proof.

The upstream and Debian TAR files were read as data, selecting only regular
files with safe relative names. Each non-comment entry in the exact
`debian/patches/series` was applied in its listed order using
`patch --batch --forward -p1 -i <absolute-patch-file>` to a private source copy.
All **82 active patches** returned zero. No Debian package was installed, no
maintainer scripts, `debian/rules`, `configure`, Makefile or build hooks were
executed. The actual patch list, each patch's size/SHA-256, the original and
patched critical-file identities, and packaging identities are retained in
[source_materials.json](../../../tests/fixtures/stage9_observer/pyc_source_hash_v5c2/source_materials.json).
Its SHA-256 is
`3c8059a18aa39de90a20e3db634a8f6113648f15672c6b51a54099f96e008e0a`.

`debian/patches/series` has SHA-256
`f66a47608220d87ea2284fd17056c54316e867569cb4bf490391b54b9218ad12`.
The active `CVE-2026-2297.patch` changes the loader class selection in
`FileLoader.get_data` to include `SourcelessFileLoader`; its selected data
remains raw bytes. No active patch changes the reviewed `Python/pyhash.c`,
`Python/import.c`, `Lib/importlib/util.py`, `Include/pyhash.h` or
`Include/internal/pycore_pyhash.h` identities. The patched bootstrap file is
separately pinned below.

## Exact reviewed path

| Patched file | SHA-256 | Relevant implementation |
| --- | --- | --- |
| `Python/import.c` | `8b0473e3972d22ed745e44fb3c61d34328e4cd9813e22dd8f4f4d818fc4410c7` | `_imp_source_hash_impl` passes the complete `Py_buffer` and length to `_Py_KeyedHash`, then explicitly emits eight little-endian bytes |
| `Python/pyhash.c` | `7f18429290df0726669cbd6d8b5a8b48b843efff648fe8863d48da5db4745628` | `_Py_KeyedHash(key, src, size)` calls `siphash13(key, 0, src, size)` unconditionally |
| `Lib/importlib/util.py` | `ca54e6458dbe521d591e5b8d9bb651ef929bfae946706c98470cdd569041a64f` | `source_hash(source_bytes)` calls `_imp.source_hash(_RAW_MAGIC_NUMBER, source_bytes)` |
| `Lib/importlib/_bootstrap_external.py` | `97c6211750d63067712ad2cd74d50be33337f1fb1cd2c769022ced9b401edd92` | Defines magic and integer key, reads raw source bytes, hashes them and places exactly eight bytes at offsets 8–15 |
| `Include/pyhash.h` | `a6ea755ff42ec955feaf49b1d234a5c2935899309ea59925d1d30f3e62fed67d` | Distinguishes configurable general Python hashing; that setting does not change `_Py_KeyedHash`'s unconditional SipHash-1-3 call |
| `Include/internal/pycore_pyhash.h` | `7c631d06afad90fa9c2ddc8dc04b7c2855ee5aa6e7ece0b22d0a966a702abf73` | Declares `_Py_KeyedHash` |
| `Lib/py_compile.py` | `ba0e2ea55612c24ed88c075bdf37dba1527492bd848b1b7f068bbbb3371ac2d9` | Retains `loader.get_data(file)` bytes and passes the same source bytes to `importlib.util.source_hash` for hash-mode output |

`Include/internal/pycore_magic_number.h` **does not exist** in the retrieved
exact 3.12.3 source archive or the patched tree. Applying a layout from a
different CPython release would give the wrong evidence path. In this release
`Lib/importlib/_bootstrap_external.py` defines:

```python
MAGIC_NUMBER = (3531).to_bytes(2, 'little') + b'\r\n'
_RAW_MAGIC_NUMBER = int.from_bytes(MAGIC_NUMBER, 'little')
```

Consequently the complete 32-bit key is `0x0a0d0dcb` (`168627659`), **not** the
16-bit value `3531`. It is zero-extended to the 64-bit SipHash `k0`; `k1 = 0`.
The keyed path does not use `PYTHONHASHSEED`, `_Py_HashSecret`, the general
Python object-hash dispatch, the empty-bytes special case or a signed hash's
`-1` remapping. In particular the correct empty input here is
`baf18eedc9387a25`, rather than eight zero bytes.

SipHash uses one compression round for every complete eight-byte little-endian
message word and one for the final length/tail word. That tail word places the
source length modulo 256 in the high byte and the remaining zero to seven raw
message bytes in its low bytes. Finalization XORs `0xff` into `v2` and performs
three rounds. All additions and rotations are modulo 2^64; the XOR of the
four final lanes is emitted as eight little-endian bytes.

The exact bytes are the raw file data supplied to the loader. Hashing does not
decode text, normalize LF/CRLF, strip a BOM, expand source encodings, validate
Python syntax or inspect a code object. The V5c2 implementation deliberately
hashes NUL and non-ASCII data too; this establishes hash behavior, not source
syntax validity.

## Independent static golden vectors

The retained
[C reference](../../../tests/fixtures/stage9_observer/pyc_source_hash_v5c2/cpython_siphash13_reference.c)
contains the **verbatim SipHash-1-3 function and macros** from the exact patched
`Python/pyhash.c`, including its MIT license and upstream attribution. The
excerpt (starting at the MIT-license comment and ending before the SipHash-2-4
conditional) has SHA-256
`0a29ca19b91cb0ac14b63c3fba590d395409d718bad9722d06a38b82d23d77ad`.
The complete reference harness has SHA-256
`3b3686ea5c9187d5ee36ff4f69db159775b5e9f4462eecb95d1195d280e7c488`.
Only trusted-host C tooling compiled this hash-only excerpt:

```sh
cc -std=c99 -O2 -Wall -Wextra cpython_siphash13_reference.c -o siphash13-reference
```

The compiler was `cc (Debian 14.2.0-19) 14.2.0`. The resulting executable's
SHA-256 is retained in `source_materials.json`. Its sole interface reads up to
1 MiB from stdin and prints the supplied-key SipHash-1-3 value. It contains no
interpreter, parser, bytecode handling, code execution, imports or source
compilation. This reference is independent of the new Python arithmetic
function, although it deliberately shares the reviewed distribution-source
algorithm definition. It is not a separate authority attestation.

The reference generated the static
[golden_vectors.json](../../../tests/fixtures/stage9_observer/pyc_source_hash_v5c2/golden_vectors.json),
SHA-256 `7573665af98673cc4fae7c5493508baa1c10943132baef30d61aa073bfaa6e0d`.
Ordinary tests read these expected values and never generate them with the
production implementation. Each vector pins raw bytes (or a fixed repeated
byte string), size, source SHA-256, magic and expected eight-byte output.
There are 20 vectors: empty; one byte; lengths 7/8/9 and 15/16/17; 32/64 bytes;
NULs; LF; CRLF; UTF-8; a 360448-byte source; one-byte mutation; changed magic;
a second independent integration source; low-16-bit key; and reversed key
byte order. The LF source
`b'value = 37\n'` gives `0bc3e3a8ab4ff6bb`; the second
`b'value = 73\n'` gives `c58b0fde5804a619`.

Negative tests independently require rejected algorithm identifiers, mutable
inputs, malformed magic, expanded or exhausted source bounds, interruption,
invalid clocks and deadline expiry. The primitive retains only bounded
eight-byte slices and integer state, calls the shared budget callback at most
4096 input bytes apart, and preserves an independent hard 60-second monotonic
ceiling. It permits an explicit four-byte magic solely for diagnostic vectors;
the V5c2 integration separately requires the fixed target-profile magic.

## Evidence still missing

| Requirement | Status | Consequence |
| --- | --- | --- |
| Exact archive/source bytes and all active patch identities | Locally measured and recorded | Reproducible source review, without publisher authentication |
| Independent acceptance of publisher key, source origin and algorithm profile | `UNVERIFIED` | No production `SOURCE_HASH_VERIFIED` status |
| Exact target interpreter SHA-256/build-ID/version/origin reference execution | `NOT_TESTED` | No claim of target-build execution equivalence |
| Target binary-to-reviewed-source/build-configuration binding | `UNVERIFIED` | Source inspection alone cannot attest how a specific binary was built |
| Actual reconstruction and accepted execution evidence | `RECONSTRUCTION_UNVERIFIED` | A correct header hash cannot prove the opaque code-object body |

No signature, trusted time, acceptance authority, interpreter invocation or
reconstruction provenance is manufactured. A future separately authorized
stage must supply an accepted isolation/evidence policy and bind the exact
interpreter and libraries, actual invocation, source, output and independent
expectation. V5c2 does not discharge that requirement.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
PYC_RECONSTRUCTION = UNVERIFIED
```
