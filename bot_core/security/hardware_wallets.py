"""Obsługa podpisów transakcji przy użyciu portfeli sprzętowych.

Moduł udostępnia klasy ``LedgerSigner`` oraz ``TrezorSigner`` zgodne z
interfejsem :class:`bot_core.security.signing.TransactionSigner`.  W
środowiskach deweloperskich biblioteki HID/USB zwykle nie są dostępne –
dlatego obie implementacje mają wbudowany tryb symulatora uruchamiany gdy
brakuje zależności lub ustawiono zmienną środowiskową
``BOT_CORE_HW_SIMULATOR=1``.  Symulator wykorzystuje klucze generowane w
pamięci i zapewnia podpisy ECDSA (Ledger) oraz EdDSA (Trezor), co pozwala na
uruchamianie testów integracyjnych bez fizycznego urządzenia.
"""

from __future__ import annotations

import base64
import os
import contextlib
import hashlib
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature

from bot_core.security.signing import JsonPayload, TransactionSigner, canonical_json_bytes


__all__ = ["HardwareWalletError", "LedgerSigner", "TrezorSigner"]


_LOGGER = logging.getLogger(__name__)


class HardwareWalletError(RuntimeError):
    """Błąd komunikacji lub konfiguracji portfela sprzętowego."""


def _use_simulator(preferred: bool | None = None) -> bool:
    if preferred is not None:
        return bool(preferred)
    return os.environ.get("BOT_CORE_HW_SIMULATOR", "0").strip() not in {"", "0", "false"}


@dataclass(slots=True)
class _SimulatedLedgerBackend:
    """Symulator podpisów Ledger (ECDSA/secp256k1)."""

    private_key: ec.EllipticCurvePrivateKey

    @classmethod
    def create(cls, seed: bytes | None = None) -> "_SimulatedLedgerBackend":
        if seed is not None:
            entropy = hashlib.sha256(seed).digest()
            int_seed = int.from_bytes(entropy, "big")
            max_value = 1 << ec.SECP256K1().key_size
            private_value = (int_seed % (max_value - 1)) + 1
            key = ec.derive_private_key(private_value, ec.SECP256K1())
        else:
            key = ec.generate_private_key(ec.SECP256K1())
        return cls(private_key=key)

    def sign(self, message: bytes) -> bytes:
        signature = self.private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(signature)
        return encode_dss_signature(r, s)

    def describe(self) -> Mapping[str, str]:
        public_numbers = self.private_key.public_key().public_numbers()
        return {
            "curve": "secp256k1",
            "public_x": hex(public_numbers.x),
            "public_y": hex(public_numbers.y),
        }


@dataclass(slots=True)
class _SimulatedTrezorBackend:
    """Symulator podpisów Trezor (Ed25519)."""

    private_key: ed25519.Ed25519PrivateKey

    @classmethod
    def create(cls, seed: bytes | None = None) -> "_SimulatedTrezorBackend":
        if seed is not None:
            entropy = seed.ljust(32, b"\x00")[:32]
            key = ed25519.Ed25519PrivateKey.from_private_bytes(entropy)
        else:
            key = ed25519.Ed25519PrivateKey.generate()
        return cls(private_key=key)

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)

    def describe(self) -> Mapping[str, str]:
        public_bytes = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return {
            "curve": "ed25519",
            "public_key": base64.b16encode(public_bytes).decode("ascii"),
        }


def _parse_bip32_path(path: str) -> Sequence[int]:
    """Konwertuje ścieżkę BIP-32 do sekwencji indeksów (z bitem harden)."""

    parts = [segment for segment in path.strip().split("/") if segment and segment != "m"]
    indexes: list[int] = []
    for segment in parts:
        hardened = segment.endswith("'")
        raw_segment = segment[:-1] if hardened else segment
        if not raw_segment:
            raise ValueError(f"Niepoprawna sekcja ścieżki BIP-32: {segment!r}")
        try:
            index = int(raw_segment, 10)
        except ValueError as exc:  # pragma: no cover - walidacja konfiguracji
            raise ValueError(f"Sekcja ścieżki BIP-32 nie jest liczbą całkowitą: {segment!r}") from exc
        if index < 0:
            raise ValueError("Indeks ścieżki BIP-32 nie może być ujemny")
        if index >= 0x80000000:
            raise ValueError("Indeks ścieżki BIP-32 przekracza maksymalną wartość 0x80000000")
        if hardened:
            index |= 0x80000000
        indexes.append(index)
    return tuple(indexes)


def _ledger_path_bytes(path: str) -> bytes:
    components = _parse_bip32_path(path)
    if len(components) > 10:  # pragma: no cover - ograniczenie bezpieczeństwa
        raise ValueError("Ścieżka BIP-32 zawiera zbyt wiele segmentów dla komendy Ledger")
    blob = bytearray([len(components)])
    for component in components:
        blob.extend(component.to_bytes(4, "big"))
    return bytes(blob)


class _HardwareWalletSigner(TransactionSigner):
    """Wspólna logika dla portfeli sprzętowych."""

    def __init__(self, *, key_id: str | None = None) -> None:
        self._key_id = key_id

    @property
    def key_id(self) -> str | None:  # pragma: no cover - property trivialny
        return self._key_id

    @property
    def requires_hardware(self) -> bool:
        return True

    def close(self) -> None:  # pragma: no cover - bez efektów ubocznych w symulatorze
        """Zamyka uchwyty do urządzenia (w symulatorze brak efektu)."""

    def __enter__(self) -> "_HardwareWalletSigner":  # pragma: no cover - sugar kontekstowy
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - sugar kontekstowy
        self.close()

    def describe(self) -> Mapping[str, Any]:
        description: dict[str, Any] = {
            "algorithm": getattr(self, "algorithm", "unknown"),
            "requires_hardware": True,
        }
        if self.key_id is not None:
            description["key_id"] = self.key_id
        return MappingProxyType(description)

    def verify(self, payload: JsonPayload, signature: Mapping[str, Any]) -> bool:
        raise HardwareWalletError(
            "Konkretny podpisujący portfela sprzętowego musi dostarczyć implementację verify()"
        )


class LedgerSigner(_HardwareWalletSigner):
    """Podpisy transakcji przy użyciu portfela Ledger (ECDSA).

    W trybie symulatora podpis jest realizowany w oparciu o bibliotekę
    ``cryptography``.  Próba użycia fizycznego urządzenia bez wymaganych
    bibliotek skutkuje zgłoszeniem :class:`HardwareWalletError`.
    """

    algorithm = "LEDGER-ECDSA"

    def __init__(
        self,
        *,
        derivation_path: str = "m/44'/60'/0'/0/0",
        use_simulator: bool | None = None,
        key_id: str | None = None,
        seed: bytes | None = None,
    ) -> None:
        super().__init__(key_id=key_id)
        self._derivation_path = derivation_path
        self._simulator_enabled = _use_simulator(use_simulator)
        if self._simulator_enabled:
            self._backend = _SimulatedLedgerBackend.create(seed=seed)
            self._transport: Any | None = None
        else:  # pragma: no cover - zależne od środowiska zewnętrznego
            try:
                from ledgerblue.comm import CommException, Dongle, getDongle  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise HardwareWalletError(
                    "Biblioteka ledgerblue nie jest dostępna – ustaw BOT_CORE_HW_SIMULATOR=1, "
                    "aby korzystać z symulatora."
                ) from exc
            try:
                self._transport = getDongle(True)
            except Exception as exc:  # noqa: BLE001
                raise HardwareWalletError("Nie udało się zainicjalizować połączenia z Ledgerem.") from exc
            self._backend = None
            self._comm_exception = CommException  # type: ignore[attr-defined]
            self._ledger_dongle: Dongle = self._transport  # type: ignore[assignment]
        self._device_metadata: Mapping[str, str] | None = None

    def _exchange(self, apdu: bytes) -> bytes:
        assert self._transport is not None
        try:
            return bytes(self._transport.exchange(apdu))  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001 pragma: no cover - zależne od środowiska
            if hasattr(self, "_comm_exception") and isinstance(exc, self._comm_exception):
                raise HardwareWalletError("Urządzenie Ledger odrzuciło komendę APDU") from exc
            raise HardwareWalletError("Nie udało się skomunikować z urządzeniem Ledger") from exc

    def _sign_with_transport(self, message: bytes) -> bytes:
        if self._transport is None:
            raise HardwareWalletError("Brak aktywnego połączenia HID z Ledgerem")

        path_blob = _ledger_path_bytes(self._derivation_path)
        if len(message) > 0xFFFF:
            raise HardwareWalletError("Ładunek do podpisu jest zbyt duży dla komendy Ledger")
        payload = bytearray()
        payload.extend(path_blob)
        payload.extend(len(message).to_bytes(2, "big"))

        offset = 0
        response = b""
        chunk_size = 230  # Ledger zaleca 255-LC nagłówka
        first = True
        while offset < len(message):
            chunk = message[offset : offset + chunk_size]
            offset += len(chunk)
            if first:
                payload_chunk = bytes(payload) + chunk
                p1 = 0x00
                first = False
            else:
                payload_chunk = chunk
                p1 = 0x80
            apdu = bytes([0xE0, 0x08, p1, 0x00, len(payload_chunk)]) + payload_chunk
            response = self._exchange(apdu)
            payload.clear()
        if not response:
            raise HardwareWalletError("Urządzenie Ledger nie zwróciło podpisu")

        # Ledger zwraca format V + R(32) + S(32); konwertujemy na DER
        if len(response) == 65:
            r = int.from_bytes(response[1:33], "big")
            s = int.from_bytes(response[33:65], "big")
            return encode_dss_signature(r, s)
        return response

    def _ensure_metadata(self) -> Mapping[str, str]:
        if self._simulator_enabled:
            assert isinstance(self._backend, _SimulatedLedgerBackend)
            return self._backend.describe()
        if self._device_metadata is not None:
            return self._device_metadata
        if self._transport is None:  # pragma: no cover - tylko hardware
            return {"path": self._derivation_path}
        try:
            path_blob = _ledger_path_bytes(self._derivation_path)
        except ValueError:
            return {"path": self._derivation_path}
        apdu = bytes([0xE0, 0x02, 0x01, 0x00, len(path_blob)]) + path_blob
        try:
            response = self._exchange(apdu)
        except HardwareWalletError:
            response = b""
        description: dict[str, str] = {"path": self._derivation_path}
        if response:
            try:
                key_len = response[0]
                pubkey = response[1 : 1 + key_len]
                description["public_key"] = pubkey.hex()
            except Exception:  # pragma: no cover - walidacja odpowiedzi
                pass
        self._device_metadata = description
        return description

    def sign(self, payload: JsonPayload) -> Mapping[str, str]:
        message = canonical_json_bytes(payload)
        if self._simulator_enabled:
            assert isinstance(self._backend, _SimulatedLedgerBackend)
            raw_signature = self._backend.sign(message)
        else:  # pragma: no cover - tylko w środowiskach z urządzeniem
            raw_signature = self._sign_with_transport(message)
        signature_b64 = base64.b64encode(raw_signature).decode("ascii")
        description = self._ensure_metadata()
        signed = {
            "algorithm": self.algorithm,
            "value": signature_b64,
            "derivation_path": self._derivation_path,
        }
        signed.update({f"device_{key}": value for key, value in description.items()})
        if self.key_id:
            signed["key_id"] = self.key_id
        return signed

    def _public_key_from_signature(self, signature: Mapping[str, Any]) -> ec.EllipticCurvePublicKey | None:
        x_value = signature.get("device_public_x")
        y_value = signature.get("device_public_y")
        if isinstance(x_value, str) and isinstance(y_value, str):
            try:
                x = int(x_value.strip().lower().removeprefix("0x"), 16)
                y = int(y_value.strip().lower().removeprefix("0x"), 16)
            except ValueError:
                return None
            numbers = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256K1())
            try:
                return numbers.public_key()
            except ValueError:
                return None

        key_hex = signature.get("device_public_key")
        if isinstance(key_hex, str):
            text = key_hex.strip().lower()
            if text.startswith("0x"):
                text = text[2:]
            if len(text) % 2:
                text = "0" + text
            try:
                key_bytes = bytes.fromhex(text)
            except ValueError:
                return None
            if not key_bytes:
                return None
            try:
                return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), key_bytes)
            except ValueError:
                return None
        return None

    def verify(self, payload: JsonPayload, signature: Mapping[str, Any]) -> bool:
        if signature.get("algorithm") not in {self.algorithm, None}:
            # nie odrzucamy brakującego algorytmu, ale walidujemy niezgodność
            return False

        value = signature.get("value")
        if not isinstance(value, str):
            return False
        try:
            raw = base64.b64decode(value)
        except ValueError:
            return False

        message = canonical_json_bytes(payload)
        public_key = self._public_key_from_signature(signature)
        if public_key is None:
            if self._simulator_enabled:
                assert isinstance(self._backend, _SimulatedLedgerBackend)
                public_key = self._backend.private_key.public_key()
            else:
                return False

        try:
            public_key.verify(raw, message, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            return False
        except ValueError:
            return False
        return True

    def describe(self) -> Mapping[str, Any]:
        description = dict(super().describe())
        description["derivation_path"] = self._derivation_path
        description["simulator"] = self._simulator_enabled
        for key, value in self._ensure_metadata().items():
            description[f"device_{key}"] = value
        return description

    def close(self) -> None:  # pragma: no cover - zależy od biblioteki ledgerblue
        transport = getattr(self, "_transport", None)
        if transport is None:
            return
        self._transport = None
        close_method = getattr(transport, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception as exc:  # noqa: BLE001
                _LOGGER.debug("Zamykanie połączenia z Ledgerem nie powiodło się: %s", exc, exc_info=_LOGGER.isEnabledFor(logging.DEBUG))

    def __del__(self):  # pragma: no cover - obrona przed wyciekami deskryptorów
        with contextlib.suppress(Exception):
            self.close()


class TrezorSigner(_HardwareWalletSigner):
    """Podpisy transakcji przy użyciu portfela Trezor (Ed25519)."""

    algorithm = "TREZOR-EDDSA"

    def __init__(
        self,
        *,
        use_simulator: bool | None = None,
        key_id: str | None = None,
        seed: bytes | None = None,
        derivation_path: str = "m/44'/60'/0'/0/0",
    ) -> None:
        super().__init__(key_id=key_id)
        self._derivation_path = derivation_path
        self._simulator_enabled = _use_simulator(use_simulator)
        if self._simulator_enabled:
            self._backend = _SimulatedTrezorBackend.create(seed=seed)
            self._transport: Any | None = None
        else:  # pragma: no cover - wymaga zewnętrznych bibliotek/urządzeń
            try:
                from trezorlib import solana, tools  # type: ignore
                from trezorlib.client import TrezorClient  # type: ignore
                from trezorlib.transport import get_transport  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise HardwareWalletError(
                    "Biblioteka python-trezor nie jest dostępna – ustaw BOT_CORE_HW_SIMULATOR=1, "
                    "aby korzystać z symulatora."
                ) from exc
            try:
                transport = get_transport()  # type: ignore[call-arg]
                self._transport = TrezorClient(transport=transport)
            except Exception as exc:  # noqa: BLE001
                raise HardwareWalletError("Nie udało się połączyć z urządzeniem Trezor.") from exc
            self._backend = None
            self._trezor_tools = tools  # type: ignore[attr-defined]
            self._trezor_solana = solana  # type: ignore[attr-defined]
            self._address_n: Iterable[int] = self._trezor_tools.parse_path(self._derivation_path)
        self._device_metadata: Mapping[str, str] | None = None

    def _sign_with_transport(self, message: bytes) -> bytes:
        if self._transport is None:
            raise HardwareWalletError("Brak aktywnego klienta Trezor")
        try:
            signature = self._trezor_solana.sign_tx(  # type: ignore[attr-defined]
                self._transport,
                list(self._address_n),
                message,
                additional_info=None,
            )
        except Exception as exc:  # noqa: BLE001 pragma: no cover - zależne od środowiska
            raise HardwareWalletError("Urządzenie Trezor odrzuciło żądanie podpisu") from exc
        if not signature:
            raise HardwareWalletError("Trezor nie zwrócił podpisu")
        return bytes(signature)

    def _ensure_metadata(self) -> Mapping[str, str]:
        if self._simulator_enabled:
            assert isinstance(self._backend, _SimulatedTrezorBackend)
            return self._backend.describe()
        if self._device_metadata is not None:
            return self._device_metadata
        if self._transport is None:  # pragma: no cover - tylko hardware
            return {"path": self._derivation_path}
        try:
            public_key = self._trezor_solana.get_public_key(  # type: ignore[attr-defined]
                self._transport,
                list(self._address_n),
                show_display=False,
            )
        except Exception:  # pragma: no cover - zależne od środowiska
            public_key = None
        metadata = {"path": self._derivation_path}
        if public_key:
            metadata["public_key"] = base64.b16encode(public_key).decode("ascii")
        self._device_metadata = metadata
        return metadata

    def sign(self, payload: JsonPayload) -> Mapping[str, str]:
        message = canonical_json_bytes(payload)
        if self._simulator_enabled:
            assert isinstance(self._backend, _SimulatedTrezorBackend)
            raw_signature = self._backend.sign(message)
        else:  # pragma: no cover - tylko w środowiskach z urządzeniem
            raw_signature = self._sign_with_transport(message)
        signature_b64 = base64.b64encode(raw_signature).decode("ascii")
        metadata = self._ensure_metadata()
        signed = {
            "algorithm": self.algorithm,
            "value": signature_b64,
            "derivation_path": self._derivation_path,
        }
        signed.update({f"device_{key}": value for key, value in metadata.items()})
        if self.key_id:
            signed["key_id"] = self.key_id
        return signed

    def _public_key_from_signature(self, signature: Mapping[str, Any]) -> ed25519.Ed25519PublicKey | None:
        key_hex = signature.get("device_public_key")
        if isinstance(key_hex, str):
            text = key_hex.strip()
            if text.lower().startswith("0x"):
                text = text[2:]
            if len(text) % 2:
                text = "0" + text
            try:
                key_bytes = bytes.fromhex(text)
            except ValueError:
                return None
            if len(key_bytes) == 32:
                try:
                    return ed25519.Ed25519PublicKey.from_public_bytes(key_bytes)
                except ValueError:
                    return None
        return None

    def verify(self, payload: JsonPayload, signature: Mapping[str, Any]) -> bool:
        if signature.get("algorithm") not in {self.algorithm, None}:
            return False

        value = signature.get("value")
        if not isinstance(value, str):
            return False
        try:
            raw = base64.b64decode(value)
        except ValueError:
            return False

        message = canonical_json_bytes(payload)
        public_key = self._public_key_from_signature(signature)
        if public_key is None:
            if self._simulator_enabled:
                assert isinstance(self._backend, _SimulatedTrezorBackend)
                public_key = self._backend.private_key.public_key()
            else:
                return False

        try:
            public_key.verify(raw, message)
        except InvalidSignature:
            return False
        except ValueError:
            return False
        return True

    def describe(self) -> Mapping[str, Any]:
        description = dict(super().describe())
        description["derivation_path"] = self._derivation_path
        description["simulator"] = self._simulator_enabled
        for key, value in self._ensure_metadata().items():
            description[f"device_{key}"] = value
        return description

    def close(self) -> None:  # pragma: no cover - zależy od klienta trezorlib
        transport = getattr(self, "_transport", None)
        if transport is None:
            return
        self._transport = None
        close_method = getattr(transport, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception as exc:  # noqa: BLE001
                _LOGGER.debug(
                    "Zamykanie klienta Trezor nie powiodło się: %s", exc, exc_info=_LOGGER.isEnabledFor(logging.DEBUG)
                )

    def __del__(self):  # pragma: no cover - obrona przed wyciekami deskryptorów
        with contextlib.suppress(Exception):
            self.close()

