from __future__ import annotations

import copy

import pytest
from bot_core.licensing.canonical import digest, parse_canonical
from bot_core.security.license import _verify_signature
from bot_core.security.signing import build_hmac_signature
from deployment.canonical_json import canonical_json_bytes
from hypothesis import given, strategies as st

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-9_007_199_254_740_991, max_value=9_007_199_254_740_991),
    st.text(max_size=40),
)
json_objects = st.dictionaries(st.text(min_size=1, max_size=20), json_scalars, max_size=8)


@pytest.mark.security
@given(json_objects)
def test_canonical_license_roundtrip_and_digest_are_stable(payload: dict[str, object]) -> None:
    raw = canonical_json_bytes(payload)

    assert parse_canonical(raw) == payload
    assert digest(payload) == digest(dict(reversed(list(payload.items()))))


@pytest.mark.security
@given(json_objects)
def test_non_canonical_whitespace_is_rejected(payload: dict[str, object]) -> None:
    raw = canonical_json_bytes(payload)

    with pytest.raises(ValueError, match="not canonical"):
        parse_canonical(raw + b"\n")


@pytest.mark.security
@given(json_objects, st.text(min_size=1, max_size=30))
def test_signed_payload_modification_never_verifies(
    payload: dict[str, object],
    replacement: str,
) -> None:
    key = b"property-test-key-not-for-production"
    signature = build_hmac_signature(payload, key=key, key_id="property")
    changed = copy.deepcopy(payload)
    changed["tampered"] = replacement
    errors = []
    warnings = []

    verified = _verify_signature(
        changed,
        signature,
        keys={"property": key},
        errors=errors,
        warnings=warnings,
        label="property test",
    )

    assert verified is None
    assert any(message.code == "license.signature.mismatch" for message in errors)
