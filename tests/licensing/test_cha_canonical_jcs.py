"""RFC8785 UTF-16 ordering in the shared restricted integer/string profile."""

import pytest

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical


def test_jcs_utf16_order_and_exact_utf8():
    value = {"\ufffd": "last", "\U0001f600": "astral", "\ue000": "bmp", "a": "Żółw"}
    raw = '{"a":"Żółw","😀":"astral","\ue000":"bmp","�":"last"}'.encode()
    assert canonical_json_bytes(value) == raw
    assert parse_canonical(raw) == value
    with pytest.raises(ValueError, match="canonical"):
        parse_canonical('{"a":"Żółw","\ue000":"bmp","�":"last","😀":"astral"}'.encode())


def test_interoperable_values_have_exact_canonical_representations():
    value = {"values": [None, True, False, -9_007_199_254_740_991, 9_007_199_254_740_991, 0]}
    raw = b'{"values":[null,true,false,-9007199254740991,9007199254740991,0]}'
    assert canonical_json_bytes(value) == raw
    assert parse_canonical(raw) == value


@pytest.mark.parametrize("value", [{1: "invalid key"}, {"value": object()}, {"value": set()}])
def test_non_json_values_cannot_enter_security_payloads(value):
    with pytest.raises(ValueError):
        canonical_json_bytes(value)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), 1.5, 9_007_199_254_740_992, "\ud800"]
)
def test_restricted_jcs_profile_rejects_noninteroperable_values(value):
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": value})
