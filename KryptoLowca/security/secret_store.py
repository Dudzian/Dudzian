"""Uniwersalny interfejs do zarządzania sekretami (ENV/File/Vault/AWS)."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - opcjonalny backend
    import hvac  # type: ignore
except Exception:  # pragma: no cover
    hvac = None  # type: ignore

try:  # pragma: no cover - opcjonalny backend
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore


class SecretBackend(str, Enum):
    ENV = "env"
    FILE = "file"
    VAULT = "vault"
    AWS = "aws"


class SecretManager:
    """Zapewnia spójny sposób pobierania/rotacji sekretów."""

    def __init__(
        self,
        *,
        backend: SecretBackend = SecretBackend.ENV,
        prefix: str = "KRYPT_LOWCA_",
        file_path: Optional[str | Path] = None,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        vault_mount: str = "secret",
        aws_secret_id: Optional[str] = None,
        aws_region: Optional[str] = None,
    ) -> None:
        self.backend = SecretBackend(backend)
        self.prefix = prefix
        self.file_path = Path(file_path) if file_path else Path(".secrets.json")
        self.vault_url = vault_url
        self.vault_token = vault_token
        self.vault_mount = vault_mount
        self.aws_secret_id = aws_secret_id
        self.aws_region = aws_region
        self._lock = threading.RLock()
        self._metadata_path = self.file_path.with_suffix(".meta.json")

        if self.backend == SecretBackend.FILE:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == SecretBackend.VAULT and hvac is None:
            raise RuntimeError("hvac (HashiCorp Vault SDK) is required for vault backend")
        if self.backend == SecretBackend.AWS and boto3 is None:
            raise RuntimeError("boto3 is required for AWS backend")

    # ------------------------ Public API ------------------------
    def get_secret(self, key: str) -> Optional[str]:
        if self.backend == SecretBackend.ENV:
            return os.getenv(f"{self.prefix}{key}")
        if self.backend == SecretBackend.FILE:
            payload = self._load_file()
            return payload.get(key)
        if self.backend == SecretBackend.VAULT:
            return self._vault_read(key)
        if self.backend == SecretBackend.AWS:
            return self._aws_read().get(key)
        raise RuntimeError(f"Unsupported backend {self.backend}")

    def set_secret(self, key: str, value: str) -> None:
        with self._lock:
            if self.backend == SecretBackend.ENV:
                os.environ[f"{self.prefix}{key}"] = value
            elif self.backend == SecretBackend.FILE:
                payload = self._load_file()
                payload[key] = value
                self.file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            elif self.backend == SecretBackend.VAULT:
                self._vault_write(key, value)
            elif self.backend == SecretBackend.AWS:
                payload = self._aws_read()
                payload[key] = value
                self._aws_write(payload)
            else:  # pragma: no cover - zabezpieczenie
                raise RuntimeError(f"Unsupported backend {self.backend}")
            self._record_rotation(key)

    def rotate_secret(self, key: str, new_value: str) -> None:
        self.set_secret(key, new_value)

    def last_rotation(self, key: str) -> Optional[datetime]:
        metadata = self._load_metadata()
        ts = metadata.get(key)
        if ts:
            return datetime.fromisoformat(ts)
        return None

    # ------------------------ Helpers ------------------------
    def _load_file(self) -> Dict[str, str]:
        if not self.file_path.exists():
            return {}
        try:
            return json.loads(self.file_path.read_text())
        except Exception:
            return {}

    def _load_metadata(self) -> Dict[str, str]:
        if not self._metadata_path.exists():
            return {}
        try:
            return json.loads(self._metadata_path.read_text())
        except Exception:
            return {}

    def _record_rotation(self, key: str) -> None:
        metadata = self._load_metadata()
        metadata[key] = datetime.now(timezone.utc).isoformat()
        self._metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    # ---- Vault backend ----
    def _vault_client(self):  # pragma: no cover - wymaga środowiska Vault
        assert hvac is not None
        client = hvac.Client(url=self.vault_url, token=self.vault_token)
        if not client.is_authenticated():
            raise RuntimeError("Vault authentication failed")
        return client

    def _vault_read(self, key: str) -> Optional[str]:  # pragma: no cover
        client = self._vault_client()
        secret_path = f"{self.vault_mount}/data/{key}"
        resp = client.secrets.kv.v2.read_secret_version(path=key, mount_point=self.vault_mount)
        data = resp.get("data", {}).get("data", {})
        return data.get("value")

    def _vault_write(self, key: str, value: str) -> None:  # pragma: no cover
        client = self._vault_client()
        client.secrets.kv.v2.create_or_update_secret(
            path=key, mount_point=self.vault_mount, secret={"value": value}
        )

    # ---- AWS backend ----
    def _aws_client(self):  # pragma: no cover - wymaga AWS
        assert boto3 is not None
        session = boto3.session.Session(region_name=self.aws_region)
        return session.client("secretsmanager")

    def _aws_read(self) -> Dict[str, str]:  # pragma: no cover
        if not self.aws_secret_id:
            raise RuntimeError("aws_secret_id is required")
        client = self._aws_client()
        res = client.get_secret_value(SecretId=self.aws_secret_id)
        payload = res.get("SecretString")
        return json.loads(payload) if payload else {}

    def _aws_write(self, payload: Dict[str, str]) -> None:  # pragma: no cover
        if not self.aws_secret_id:
            raise RuntimeError("aws_secret_id is required")
        client = self._aws_client()
        client.put_secret_value(SecretId=self.aws_secret_id, SecretString=json.dumps(payload))


__all__ = ["SecretManager", "SecretBackend"]
