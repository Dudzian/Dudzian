"""Pomocnicze funkcje przygotowujące metadane wypłat wymagających HW wallet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


@dataclass(slots=True)
class PayoutRequest:
    """Reprezentuje żądanie wypłaty z portfela strategii."""

    account_id: str
    asset: str
    amount: float
    destination: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_hardware_wallet_requirement(self) -> "PayoutRequest":
        enriched = require_hardware_wallet_metadata(
            self.metadata,
            account_id=self.account_id,
            operation="withdrawal",
        )
        return PayoutRequest(
            account_id=self.account_id,
            asset=self.asset,
            amount=self.amount,
            destination=self.destination,
            metadata=enriched,
        )


def require_hardware_wallet_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    account_id: str,
    operation: str = "withdrawal",
) -> MutableMapping[str, Any]:
    """Zwraca metadane oznaczające wypłatę wymagającą podpisu z HW wallet."""

    document: MutableMapping[str, Any] = dict(metadata or {})
    document["operation"] = operation
    document["account"] = account_id
    document["requires_hardware_wallet"] = True
    return document


__all__ = ["PayoutRequest", "require_hardware_wallet_metadata"]
