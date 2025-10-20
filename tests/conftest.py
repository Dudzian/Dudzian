"""Globalna konfiguracja testów."""
from __future__ import annotations

import sys
from types import ModuleType


if "nacl" not in sys.modules:
    nacl_module = ModuleType("nacl")
    nacl_exceptions = ModuleType("nacl.exceptions")
    nacl_exceptions.BadSignatureError = Exception

    class _VerifyKey:
        def verify(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _SigningKey:
        def __init__(self) -> None:
            self.verify_key = self

        @classmethod
        def generate(cls) -> "_SigningKey":
            return cls()

        def sign(self, payload: bytes) -> "_SignatureWrapper":
            return _SignatureWrapper(b"stub-signature" + payload[:1])

        def encode(self) -> bytes:
            return b"stub-signing-key"

    class _SignatureWrapper:
        def __init__(self, signature: bytes) -> None:
            self.signature = signature

    nacl_signing = ModuleType("nacl.signing")
    nacl_signing.VerifyKey = lambda *_args, **_kwargs: _VerifyKey()
    nacl_signing.SigningKey = _SigningKey
    nacl_module.exceptions = nacl_exceptions
    nacl_module.signing = nacl_signing
    sys.modules["nacl"] = nacl_module
    sys.modules["nacl.exceptions"] = nacl_exceptions
    sys.modules["nacl.signing"] = nacl_signing

# Import modułu zapewniającego, że katalog repozytorium znajduje się na sys.path.
# Dzięki temu wszystkie testy mogą importować kod projektu niezależnie od miejsca uruchomienia.
import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import
