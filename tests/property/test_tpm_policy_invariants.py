from __future__ import annotations

import hashlib

import pytest
from deployment.windows_tpm_substrate_probe import expected_policy_authorize_digest
from hypothesis import given, strategies as st


@pytest.mark.security
@pytest.mark.tpm
@given(
    key_name=st.binary(min_size=1, max_size=96),
    policy_ref=st.binary(min_size=0, max_size=96),
)
def test_policy_authorize_digest_matches_independent_construction(
    key_name: bytes,
    policy_ref: bytes,
) -> None:
    command_code = (0x0000016A).to_bytes(4, "big")
    first = hashlib.sha256(bytes(32) + command_code + key_name).digest()
    expected = hashlib.sha256(first + policy_ref).digest()

    assert expected_policy_authorize_digest(key_name, policy_ref) == expected
    assert len(expected) == 32


@pytest.mark.security
@pytest.mark.tpm
@given(
    key_name=st.binary(min_size=1, max_size=96),
    policy_ref=st.binary(min_size=0, max_size=96),
    suffix=st.binary(min_size=1, max_size=32),
)
def test_policy_authorize_digest_binds_key_name_and_policy_ref(
    key_name: bytes,
    policy_ref: bytes,
    suffix: bytes,
) -> None:
    original = expected_policy_authorize_digest(key_name, policy_ref)

    assert expected_policy_authorize_digest(key_name + suffix, policy_ref) != original
    assert expected_policy_authorize_digest(key_name, policy_ref + suffix) != original
