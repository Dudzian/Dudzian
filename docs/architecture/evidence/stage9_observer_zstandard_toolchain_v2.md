# Offline verifier Zstandard toolchain evidence V2

Date: **2026-10-29 UTC**. Classification:
`OBSERVED_PYPI_COMPARISON_ONLY_NOT_INDEPENDENT_ACCEPTANCE`.
The [machine-readable record](stage9_observer_zstandard_toolchain_v2.json)
identifies observed artifact bytes. It does not enroll a trust root or approve
the library, Python, operating system, package repository or build process.

## Exact selected tool and artifact

The analysis tool uses **python-zstandard 0.25.0**, backend **`cext`**, reporting
embedded **libzstd 1.5.7**. The adapter checks these identifiers before creating
a Zstandard decoder and refuses other versions/backends. These checks prevent
an implicit dependency change; a local module's self-reported identity is not
independent attestation of its bytes or behavior.

The existing broad optional `compression` requirement (`zstandard>=0.22`)
does not establish this toolchain. The separately pinned offline verifier/test
dependency does not add Zstandard to the future minimal Observer runtime.
The verifier imports no `bot_core`, starts no `zstd` process, fetches no
dependencies and executes no code from an analyzed DEB. Dependency acquisition
and environment preparation take place before offline verification.

For this cloud measurement the selected artifact is:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `zstandard-0.25.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` | 5559635 | `9300d02ea7c6506f00e627e287e0492a5eb0371ec1670ae852fefffa6164b072` |
| `zstandard-0.25.0.tar.gz` (source inspection only) | 711513 | `7713e1179d162cf5c7906da876ec2ccb9c3a9dcbdffef0cc7f70c3667a205f0b` |

Both were downloaded using HTTPS with inherited proxy/CA settings. Full size
and SHA-256 matched the [PyPI 0.25.0 metadata](https://pypi.org/pypi/zstandard/0.25.0/json).
Exact artifact URLs and upload timestamps are retained in the JSON. PyPI's
`has_sig` field is false; no independently accepted publisher signature,
provenance attestation or reproducible build was verified here. Comparison
against metadata obtained through the same service is not independent approval.

## Native binaries and dependencies

The wheel's installed payload matched the independently downloaded wheel at
all **10/10 payload file paths**, excluding its regenerated `RECORD`. Both
native files also matched their SHA-256 entries in the wheel `RECORD`:

| Wheel member | Bytes | SHA-256 |
| --- | ---: | --- |
| `zstandard/backend_c.cpython-311-x86_64-linux-gnu.so` | 11614400 | `532a86d3bc0f420a4f7038bf6039c93b55e6f149d7a314b17dbb27750788964d` |
| `zstandard/_cffi.cpython-311-x86_64-linux-gnu.so` | 12135008 | `0cd90a4eeedf490a3207a8ca0eca69b73b887b464509407ebc058007d9f94e55` |

Only `backend_c` is selected. `readelf -d` (GNU Binutils 2.44) reports direct
`DT_NEEDED` entries **`libpthread.so.0` and `libc.so.6`** for both binaries;
there is no separate dynamic libzstd dependency. The source distribution
contains bundled `zstd/zstd.c`; the selected backend reports libzstd 1.5.7.
The wheel metadata names setuptools 80.9.0 as its generator; this does not
identify or authenticate the compiler or complete build recipe.

The base wheel has **no unconditional Python package dependencies**. Its
optional `cffi` extra is unselected; its presence in metadata does not make it
part of this verifier's selected backend. The C extension also depends on the
CPython C ABI, system dynamic loader, libc/pthread implementation, operating
system and the execution environment. The analysis ran on CPython **3.11.16**
(reporting Clang **22.1.3**) with glibc **2.41**. The JSON records observed
Python/libc/pthread binary hashes. These are local measurements, not a complete
authenticated environment inventory or a reproducible compiler chain.

## Bounded decoder source review

The source distribution was read as inert source material; it was not built.
SHA-256 and size for inspected `c-ext/decompressor.c`,
`c-ext/decompressionreader.c` and bundled `zstd/zstd.c` are retained in JSON.

In `c-ext/decompressor.c`, `ensure_dctx` passes `max_window_size` directly to
`ZSTD_DCtx_setMaxWindowSize`. For this exact C backend **the value is in
bytes**, despite documentation elsewhere describing KiB. The adapter therefore
uses bytes and pins the backend/version on which this interpretation depends.
`memory_size()` returns `ZSTD_sizeof_DCtx`; it measures decoder state, not
Python allocations or process RSS. The reader's explicit `read_size` bounds
compressed input requests, and each explicit positive `read(size)` bounds its
returned output allocation. The verifier does not request `read(-1)`.

Bundled libzstd's `ZSTD_decodingBufferSize_internal` derives the history/ring
buffer from the frame window plus at most two 128 KiB blocks and two wild-copy
allowances. `ZSTD_estimateDStreamSize` also includes a context and one input
block. The adapter checks the frame window and reserves conservative decoder
headroom before native decoding, then checks live native memory usage. Its
native allowance is `window_size + estimate_decompression_context_size() +
524288`; the pinned backend reports a context estimate of **95968 bytes**.
The 512 KiB headroom exceeds three 128 KiB blocks plus 64 bytes in the reviewed
1.5.7 formula. It separately reserves `8 * chunk_bytes` and
`2048 * (max_archive_members + declared_member_count)` after the caller has
subtracted shared retained memory. Zstandard additionally reserves the existing
**1 MiB** fixed Python/control-buffer allowance separately from the native
window, including when read chunks are tightened to 512 bytes.
Decoder/stream references are released
between control and data frames. Shared
TAR output, ratio, member, buffer and deadline limits still apply. This is
cooperative allocation accounting, **not an OS-enforced memory limit** or hard
termination boundary; native allocator/runtime overhead and loaded code remain
part of the unqualified verifier environment.

## Preparation and provenance required before trusted use

The [platform-specific hash lock](../../../deployment/stage9-offline-verifier-cp311-manylinux-x86_64.lock)
identifies only the CPython 3.11 Linux x86_64 manylinux2014 wheel (glibc >= 2.17).
It makes no claim about other Python ABIs, architectures, platforms or locally
built wheels. First acquire/review this exact wheel through a separately
accepted channel. Then prepare the analysis environment from a local wheel
directory, before invoking verification:

```sh
python -m pip install --no-index --find-links /reviewed/wheels \
  --no-deps --only-binary :all: --require-hashes \
  -r deployment/stage9-offline-verifier-cp311-manylinux-x86_64.lock
```

This is an explicit preparation command, not a verifier fallback or automatic
download. Missing libraries and unsupported variants must refuse. The hash
lock does not independently authorize the wheel it identifies.

Trusted use would additionally require independent acceptance of the publisher
and PyPI/bootstrap channel, release/source identity, bundled libzstd source,
build recipe/toolchain and binary provenance, plus CPython, OS loader/libc and
the analysis host. The project must review that complete chain and its update
and revocation policy separately. Source-to-wheel reproducibility and formal
tool acceptance remain `NOT_YET_PROVEN` / `UNVERIFIED`. Local installation,
matching hashes and a clean vulnerability scan cannot supply that approval.
