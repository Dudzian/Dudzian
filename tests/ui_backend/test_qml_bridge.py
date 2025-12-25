from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from ui.backend.qml_bridge import to_plain_value


def test_to_plain_value_preserves_numeric_types() -> None:
    assert to_plain_value(5) == 5
    assert isinstance(to_plain_value(5), int)

    assert to_plain_value(2.5) == 2.5
    assert isinstance(to_plain_value(2.5), float)

    integer_decimal = to_plain_value(Decimal("1"))
    assert integer_decimal == 1
    assert isinstance(integer_decimal, int)

    fractional_decimal = to_plain_value(Decimal("1.5"))
    assert fractional_decimal == 1.5
    assert isinstance(fractional_decimal, float)


def test_to_plain_value_converts_nested_dataclass_and_mapping_numbers() -> None:
    @dataclass
    class Payload:
        a: int
        b: Decimal
        c: list[Decimal]
        d: float

    payload = Payload(a=3, b=Decimal("2"), c=[Decimal("4.5"), Decimal("5")], d=1.25)
    converted = to_plain_value(payload)

    assert converted == {"a": 3, "b": 2, "c": [4.5, 5], "d": 1.25}
    assert all(isinstance(item, (int, float)) for item in converted["c"])

    mapping = {"x": Decimal("1"), "y": Decimal("2.5"), "z": 1, "w": 2.0}
    mapped = to_plain_value(mapping)
    assert mapped == {"x": 1, "y": 2.5, "z": 1, "w": 2.0}
    assert isinstance(mapped["x"], int)
    assert isinstance(mapped["y"], float)


def test_to_plain_value_handles_fraction_numbers() -> None:
    whole = to_plain_value(Fraction(6, 3))
    assert whole == 2
    assert isinstance(whole, int)

    fractional = to_plain_value(Fraction(3, 2))
    assert fractional == 1.5
    assert isinstance(fractional, float)
