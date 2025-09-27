"""Zarządzanie kluczami API z szyfrowaniem i rotacją."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from KryptoLowca.exchanges.interfaces import ExchangeCredentials

Encryptor = Callable[[str, Dict[str, Any]], Dict[str, Any]]
Decryptor = Callable[[str, Dict[str, Any]], Dict[str, Any]]


@dataclass(slots=True)
class APIKeyRecord:
    exchange: str
    account: str
    version: int
    created_at: datetime
    expires_at: Optional[datetime]
    credentials: ExchangeCredentials
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self, encryptor: Encryptor) -> Dict[str, Any]:
        payload = {
            "exchange": self.exchange,
            "account": self.account,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }
        encrypted = encryptor(
            "exchange",
            {
                "api_key": self.credentials.api_key,
                "api_secret": self.credentials.api_secret,
                "passphrase": self.credentials.passphrase or "",
                "is_read_only": self.credentials.is_read_only,
            },
        )
        payload["data"] = encrypted
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], decryptor: Decryptor) -> "APIKeyRecord":
        decrypted = decryptor("exchange", payload.get("data", {}))
        metadata = payload.get("metadata", {}) or {}
        return cls(
            exchange=payload["exchange"],
            account=payload["account"],
            version=int(payload.get("version", 1)),
            created_at=datetime.fromisoformat(payload["created_at"]),
            expires_at=(
                datetime.fromisoformat(payload["expires_at"])
                if payload.get("expires_at")
                else None
            ),
            credentials=ExchangeCredentials(
                api_key=decrypted.get("api_key", ""),
                api_secret=decrypted.get("api_secret", ""),
                passphrase=decrypted.get("passphrase") or None,
                is_read_only=bool(decrypted.get("is_read_only", False)),
                metadata=metadata,
            ),
            metadata=metadata,
        )


class APIKeyManager:
    """Magazyn kluczy API korzystający z szyfrowania ConfigManagera."""

    def __init__(
        self,
        storage_path: Path,
        *,
        encryptor: Encryptor,
        decryptor: Decryptor,
        default_ttl: timedelta | None = timedelta(days=180),
    ) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._encryptor = encryptor
        self._decryptor = decryptor
        self._default_ttl = default_ttl

    def _load_records(self) -> List[APIKeyRecord]:
        if not self._path.exists():
            return []
        raw = json.loads(self._path.read_text())
        records = []
        for payload in raw.get("records", []):
            try:
                records.append(APIKeyRecord.from_payload(payload, self._decryptor))
            except Exception:
                continue
        return records

    def _write_records(self, records: Iterable[APIKeyRecord]) -> None:
        payload = {"records": [record.to_payload(self._encryptor) for record in records]}
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def save_credentials(
        self,
        exchange: str,
        account: str,
        credentials: ExchangeCredentials,
        *,
        compliance_ack: bool = False,
        expires_at: Optional[datetime] = None,
        rotate: bool = False,
    ) -> APIKeyRecord:
        env = str(credentials.metadata.get("environment", "demo")).lower()
        if env not in {"demo", "test", "paper"} and not compliance_ack:
            raise ValueError("Zapis kluczy live wymaga potwierdzenia compliance")
        records = self._load_records()
        existing = [
            record for record in records if record.exchange == exchange and record.account == account
        ]
        next_version = max([r.version for r in existing], default=0) + 1
        expires = expires_at or (
            datetime.now(timezone.utc) + self._default_ttl if self._default_ttl else None
        )
        record = APIKeyRecord(
            exchange=exchange,
            account=account,
            version=next_version,
            created_at=datetime.now(timezone.utc),
            expires_at=expires,
            credentials=credentials,
            metadata=dict(credentials.metadata),
        )
        remaining = [r for r in records if r.exchange != exchange or r.account != account]
        if rotate:
            remaining.extend(existing)
        remaining.append(record)
        self._write_records(remaining)
        return record

    def rotate_credentials(
        self,
        exchange: str,
        account: str,
        credentials: ExchangeCredentials,
        *,
        compliance_ack: bool = False,
        expires_at: Optional[datetime] = None,
    ) -> APIKeyRecord:
        return self.save_credentials(
            exchange,
            account,
            credentials,
            compliance_ack=compliance_ack,
            expires_at=expires_at,
            rotate=True,
        )

    def load_credentials(self, exchange: str, account: str) -> ExchangeCredentials:
        records = sorted(
            (
                record
                for record in self._load_records()
                if record.exchange == exchange and record.account == account
            ),
            key=lambda r: (r.version, r.created_at),
        )
        if not records:
            raise KeyError(f"Brak kluczy dla konta {exchange}:{account}")
        return records[-1].credentials

    def list_accounts(self) -> List[Dict[str, Any]]:
        summary: Dict[tuple[str, str], APIKeyRecord] = {}
        for record in self._load_records():
            key = (record.exchange, record.account)
            if key not in summary or summary[key].version < record.version:
                summary[key] = record
        return [
            {
                "exchange": exchange,
                "account": account,
                "version": record.version,
                "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                "metadata": record.metadata,
            }
            for (exchange, account), record in sorted(summary.items())
        ]

    def purge_expired(self) -> int:
        now = datetime.now(timezone.utc)
        before = self._load_records()
        records = [record for record in before if not record.expires_at or record.expires_at > now]
        removed = len(before) - len(records)
        self._write_records(records)
        return removed


__all__ = ["APIKeyManager", "APIKeyRecord"]
