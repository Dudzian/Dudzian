from __future__ import annotations

import hashlib


def derive_fixture_hmac_key(doc_relative: str, key_id: str) -> bytes:
    """Derive a deterministic HMAC key for test fixtures only.

    Uwaga: to nie jest sekret kryptograficzny. Klucz jest celowo
    deterministyczny i oparty o publiczne dane wejściowe, aby zapewnić
    powtarzalność fixture'ów testowych.
    """

    key_material = hashlib.sha256(f"{doc_relative}:{key_id}".encode("utf-8")).hexdigest()
    return key_material.encode("ascii")
