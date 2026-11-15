"""Independent, bounded PEP 552 hash for a locally reviewed CPython profile.

The exact Ubuntu source and independent C vectors are identified in
``tests/fixtures/stage9_observer/pyc_source_hash_v5c2/source_materials.json``.
This pure arithmetic implementation imports no studied code or host hash API.
Its local profile is not independent authority acceptance of the target build.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from deployment.stage9_observer_contract import ContractRefusal

ALGORITHM_ID = "cpython-3.12.3-ubuntu0.17-siphash13-le-v1"
MAGIC = bytes.fromhex("cb0d0d0a")
# All values are measured identities, never synthetic signatures/acceptance.
ALGORITHM_MATERIALS: tuple[tuple[str, str], ...] = (
    (
        "python3.12_3.12.3-1ubuntu0.17.dsc",
        "941700b999737d869b80c3b5a0adc0996a0fd760b982e18551bd2104da779384",
    ),
    (
        "python3.12_3.12.3.orig.tar.xz",
        "56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1",
    ),
    (
        "python3.12_3.12.3-1ubuntu0.17.debian.tar.xz",
        "17b15629cb10825924cf7b85055255080ec773133ae0f8b8ce5ea2eec90dd200",
    ),
    ("patched/Python/import.c", "8b0473e3972d22ed745e44fb3c61d34328e4cd9813e22dd8f4f4d818fc4410c7"),
    ("patched/Python/pyhash.c", "7f18429290df0726669cbd6d8b5a8b48b843efff648fe8863d48da5db4745628"),
    (
        "patched/Lib/importlib/util.py",
        "ca54e6458dbe521d591e5b8d9bb651ef929bfae946706c98470cdd569041a64f",
    ),
    (
        "patched/Lib/importlib/_bootstrap_external.py",
        "97c6211750d63067712ad2cd74d50be33337f1fb1cd2c769022ced9b401edd92",
    ),
    (
        "patched/Include/pyhash.h",
        "a6ea755ff42ec955feaf49b1d234a5c2935899309ea59925d1d30f3e62fed67d",
    ),
    (
        "patched/Include/internal/pycore_pyhash.h",
        "7c631d06afad90fa9c2ddc8dc04b7c2855ee5aa6e7ece0b22d0a966a702abf73",
    ),
    (
        "patched/Lib/py_compile.py",
        "ba0e2ea55612c24ed88c075bdf37dba1527492bd848b1b7f068bbbb3371ac2d9",
    ),
    ("debian/patches/series", "f66a47608220d87ea2284fd17056c54316e867569cb4bf490391b54b9218ad12"),
    ("debian/rules", "0146c8e8c01756f41387213f5c289df20310e75ebd880d94977f5373e2dcc33c"),
    ("debian/changelog", "0919a82649eb8d3d1110d5498ac1f159929190b2e6ec75605d47f1b8f17ade94"),
    ("source_materials.json", "3c8059a18aa39de90a20e3db634a8f6113648f15672c6b51a54099f96e008e0a"),
    (
        "cpython_siphash13_reference.c",
        "3b3686ea5c9187d5ee36ff4f69db159775b5e9f4462eecb95d1195d280e7c488",
    ),
)
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MASK = (1 << 64) - 1


def _round(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    a = (a + b) & _MASK
    c = (c + d) & _MASK
    b = ((b << 13) | (b >> 51)) & _MASK
    d = ((d << 16) | (d >> 48)) & _MASK
    b ^= a
    d ^= c
    a = ((a << 32) | (a >> 32)) & _MASK
    c = (c + b) & _MASK
    a = (a + d) & _MASK
    b = ((b << 17) | (b >> 47)) & _MASK
    d = ((d << 21) | (d >> 43)) & _MASK
    b ^= c
    d ^= a
    c = ((c << 32) | (c >> 32)) & _MASK
    return a, b, c, d


def independent_source_hash(
    source: bytes,
    algorithm_id: str,
    *,
    magic: bytes = MAGIC,
    check: Callable[[], None] | None = None,
    max_source_bytes: int = _MAX_SOURCE_BYTES,
) -> bytes:
    """Hash exact immutable bytes with SipHash-1-3 and explicit LE output.

    ``k0`` is all four magic bytes interpreted little-endian; ``k1`` is zero.
    The primitive supports explicit four-byte keys for diagnostic vectors;
    the V5c2 integration separately enforces the fixed approved-profile magic.
    No text decoding, BOM removal, newline changes or source compilation occur.

    Data and elapsed limits cannot exceed existing V5c1 defaults. Retained
    scratch memory is constant, with only bounded eight-byte source slices.
    ``check`` joins the caller's shared deadline/cancellation budget. The local
    60-second monotonic deadline remains active even without that callback.
    These checks are cooperative, not an operating-system preemption promise.
    """
    if type(algorithm_id) is not str or algorithm_id != ALGORITHM_ID:
        raise ContractRefusal("UNSUPPORTED_SOURCE")
    if type(source) is not bytes:
        raise ContractRefusal("FORMAT_REJECTED")
    if type(magic) is not bytes or len(magic) != 4:
        raise ContractRefusal("FORMAT_REJECTED")
    if (
        type(max_source_bytes) is not int
        or not 0 <= max_source_bytes <= _MAX_SOURCE_BYTES
        or len(source) > max_source_bytes
    ):
        raise ContractRefusal("LIMIT_EXCEEDED")
    started = time.monotonic()

    def checkpoint() -> None:
        if check is not None:
            check()
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed) or not 0 <= elapsed < 60:
            raise ContractRefusal("LIMIT_EXCEEDED")

    checkpoint()
    key = int.from_bytes(magic, "little")
    a, b, c, d = (
        key ^ 0x736F6D6570736575,
        0x646F72616E646F6D,
        key ^ 0x6C7967656E657261,
        0x7465646279746573,
    )
    end = len(source) - len(source) % 8
    for position in range(0, end, 8):
        if position % 4096 == 0:
            checkpoint()
        word = int.from_bytes(source[position : position + 8], "little")
        d ^= word
        a, b, c, d = _round(a, b, c, d)
        a ^= word
    tail = ((len(source) & 255) << 56) | int.from_bytes(source[end:], "little")
    d ^= tail
    a, b, c, d = _round(a, b, c, d)
    a ^= tail
    c ^= 255
    for _ in range(3):
        a, b, c, d = _round(a, b, c, d)
    checkpoint()
    return (a ^ b ^ c ^ d).to_bytes(8, "little")
