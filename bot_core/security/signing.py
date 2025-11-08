"""Pomocnicze funkcje podpisywania ładunków JSON (HMAC oraz portfele)."""
from __future__ import annotations

import abc
import contextlib
import base64
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, TypeAlias

try:
    from typing import Protocol
except ImportError:  # pragma: no cover - Python <3.8 fallback
    Protocol = abc.ABC  # type: ignore[misc, assignment]


_LOGGER = logging.getLogger(__name__)

_CANONICAL_SEPARATORS = (",", ":")

# ``Mapping`` obejmuje dokumenty JSON, ``Sequence`` pozwala podpisywać listy kroków.
JsonPayload: TypeAlias = Mapping[str, Any] | Sequence[Any]


def canonical_json_bytes(payload: JsonPayload) -> bytes:
    """Zwraca kanoniczną reprezentację JSON (UTF-8, sort_keys, brak spacji)."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=_CANONICAL_SEPARATORS,
    ).encode("utf-8")


def build_hmac_signature(
    payload: JsonPayload,
    *,
    key: bytes,
    algorithm: str = "HMAC-SHA256",
    key_id: str | None = None,
) -> dict[str, str]:
    """Buduje podpis HMAC dla ładunku JSON."""

    digest = hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).digest()
    signature = {
        "algorithm": algorithm,
        "value": base64.b64encode(digest).decode("ascii"),
    }
    if key_id:
        signature["key_id"] = str(key_id)
    return signature


def verify_hmac_signature(
    payload: JsonPayload,
    signature: Mapping[str, Any] | None,
    *,
    key: bytes | None,
    algorithm: str = "HMAC-SHA256",
) -> bool:
    """Weryfikuje podpis HMAC.

    Zwraca ``True`` gdy podpis jest poprawny. Jeśli brakuje klucza albo podpisu,
    funkcja zwraca ``False``.
    """

    if not key or not signature:
        return False

    if signature.get("algorithm") != algorithm:
        return False

    expected = build_hmac_signature(payload, key=key, algorithm=algorithm, key_id=signature.get("key_id"))
    actual_value = signature.get("value")
    expected_value = expected.get("value")
    if not isinstance(actual_value, str) or not isinstance(expected_value, str):
        return False
    return hmac.compare_digest(actual_value, expected_value)


class HmacSignedReportMixin:
    """Wspólna implementacja podpisywania raportów HMAC.

    Klasy raportów muszą implementować metodę ``to_mapping`` zwracającą
    reprezentację zgodną z JSON.  Mixin zapewnia jednolite metody
    ``build_signature`` oraz ``write_signature`` wykorzystywane zarówno przez
    moduł resilience, jak i stress-lab.
    """

    def to_mapping(self) -> Mapping[str, Any]:  # pragma: no cover - dokumentuje kontrakt
        raise NotImplementedError

    def build_signature(
        self,
        *,
        key: bytes,
        algorithm: str = "HMAC-SHA256",
        key_id: str | None = None,
    ) -> Mapping[str, str]:
        return build_hmac_signature(self.to_mapping(), key=key, algorithm=algorithm, key_id=key_id)

    def write_signature(
        self,
        path: Path,
        *,
        key: bytes,
        algorithm: str = "HMAC-SHA256",
        key_id: str | None = None,
    ) -> Path:
        signature = self.build_signature(key=key, algorithm=algorithm, key_id=key_id)
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(signature, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path


def validate_hmac_signature(
    payload: JsonPayload,
    signature_doc: Mapping[str, Any],
    *,
    key: bytes,
    algorithm: str = "HMAC-SHA256",
) -> list[str]:
    """Sprawdza poprawność podpisu HMAC i zwraca listę błędów."""

    signature = signature_doc.get("signature")
    if not isinstance(signature, Mapping):
        return ["Dokument podpisu nie zawiera sekcji 'signature'"]
    algorithm_name = signature.get("algorithm")
    if algorithm_name != algorithm:
        return [f"Nieobsługiwany algorytm podpisu: {algorithm_name!r}"]
    expected = build_hmac_signature(
        payload,
        key=key,
        algorithm=algorithm,
        key_id=signature.get("key_id"),
    )
    if dict(expected) != dict(signature):
        return ["Podpis HMAC nie zgadza się z manifestem"]
    return []


class TransactionSigner(Protocol):
    """Abstrakcyjny interfejs podpisujący transakcje lub operacje."""

    algorithm: str
    key_id: str | None

    @property
    def requires_hardware(self) -> bool:  # pragma: no cover - domyślna implementacja
        return False

    def sign(self, payload: JsonPayload) -> Mapping[str, Any]:
        """Zwraca dokument podpisu dla przekazanego ładunku."""

    def close(self) -> None:  # pragma: no cover - metoda opcjonalna
        """Zamyka zasoby podpisującego (domyślnie brak działania)."""

    def describe(self) -> Mapping[str, Any]:  # pragma: no cover - metoda opcjonalna
        """Zwraca metadane podpisującego wykorzystywane do audytu/logowania."""

    def verify(self, payload: JsonPayload, signature: Mapping[str, Any]) -> bool:
        """Weryfikuje podpis i zwraca ``True`` gdy jest poprawny."""


class HmacTransactionSigner(TransactionSigner):
    """Implementacja ``TransactionSigner`` bazująca na HMAC."""

    def __init__(
        self,
        *,
        key: bytes,
        algorithm: str = "HMAC-SHA256",
        key_id: str | None = None,
    ) -> None:
        self._key = bytes(key)
        self.algorithm = algorithm
        self.key_id = key_id

    def sign(self, payload: JsonPayload) -> Mapping[str, str]:
        return build_hmac_signature(payload, key=self._key, algorithm=self.algorithm, key_id=self.key_id)

    def close(self) -> None:  # pragma: no cover - brak zasobów zewnętrznych
        return None

    def describe(self) -> Mapping[str, Any]:  # pragma: no cover - bez efektów ubocznych
        description: dict[str, Any] = {
            "algorithm": self.algorithm,
            "requires_hardware": False,
        }
        if self.key_id is not None:
            description["key_id"] = self.key_id
        return MappingProxyType(description)

    def verify(self, payload: JsonPayload, signature: Mapping[str, Any]) -> bool:
        return verify_hmac_signature(payload, signature, key=self._key, algorithm=self.algorithm)


class TransactionSignerSelector:
    """Wybiera podpisującego per konto (z domyślnym fallbackiem)."""

    def __init__(
        self,
        *,
        default: TransactionSigner | None = None,
        overrides: Mapping[str, TransactionSigner] | None = None,
    ) -> None:
        self._default = default
        self._overrides = {str(key): value for key, value in (overrides or {}).items()}
        self._key_index: dict[str, tuple[TransactionSigner, ...]] = {}
        self._key_index_dirty = True

    def register(self, account_id: str, signer: TransactionSigner) -> None:
        self._overrides[str(account_id)] = signer
        self._key_index_dirty = True

    def resolve(self, account_id: str | None) -> TransactionSigner | None:
        if account_id:
            signer = self._overrides.get(str(account_id))
            if signer is not None:
                return signer
        return self._default

    def _rebuild_key_index(self) -> None:
        mapping: dict[str, list[TransactionSigner]] = {}
        seen_per_key: dict[str, set[int]] = {}
        for _account_id, signer in self.iter_signers():
            key_id = getattr(signer, "key_id", None)
            if key_id is None:
                continue
            text = str(key_id).strip()
            if not text:
                continue
            bucket = mapping.setdefault(text, [])
            seen = seen_per_key.setdefault(text, set())
            identifier = id(signer)
            if identifier in seen:
                continue
            seen.add(identifier)
            bucket.append(signer)

        self._key_index = {key: tuple(values) for key, values in mapping.items()}
        self._key_index_dirty = False

    def resolve_by_key_id(self, key_id: str | None) -> tuple[TransactionSigner, ...]:
        """Zwraca podpisujących posiadających wskazany ``key_id``.

        Kolejność kandydatów odpowiada kolejności zwracanej przez
        :meth:`iter_signers`.  Duplikaty (ten sam obiekt podpisującego)
        są zwracane tylko raz.  Dla nieznanego identyfikatora zwracana jest
        pusta krotka.
        """

        if not key_id:
            return ()

        if self._key_index_dirty:
            self._rebuild_key_index()

        return self._key_index.get(key_id.strip(), ())

    def as_mapping(self) -> Mapping[str, TransactionSigner]:
        result: MutableMapping[str, TransactionSigner] = dict(self._overrides)
        if self._default is not None:
            result.setdefault("__default__", self._default)
        return result

    def iter_signers(self) -> Iterable[tuple[str | None, TransactionSigner]]:
        """Iteruje po podpisujących (zwracając też domyślnego pod kluczem ``None``)."""

        if self._default is not None:
            yield None, self._default
        for account_id, signer in self._overrides.items():
            yield account_id, signer

    def close(self) -> None:
        """Zamyka wszystkich podpisujących, ignorując błędy."""

        visited: set[int] = set()
        for _account, signer in self.iter_signers():
            identifier = id(signer)
            if identifier in visited:
                continue
            visited.add(identifier)
            close_method = getattr(signer, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.debug(
                        "Nie udało się zamknąć podpisującego %s: %s", signer, exc, exc_info=_LOGGER.isEnabledFor(logging.DEBUG)
                    )

    def describe_signers(self) -> Mapping[str | None, Mapping[str, Any]]:
        """Zwraca metadane wszystkich podpisujących (łącznie z domyślnym)."""

        descriptions: dict[str | None, Mapping[str, Any]] = {}
        cache: dict[int, Mapping[str, Any]] = {}

        for account_id, signer in self.iter_signers():
            identifier = id(signer)
            info = cache.get(identifier)
            if info is None:
                describe_method = getattr(signer, "describe", None)
                raw: Mapping[str, Any] | None
                if callable(describe_method):
                    try:
                        candidate = describe_method()
                    except Exception as exc:  # noqa: BLE001
                        _LOGGER.debug(
                            "Nie udało się pobrać opisu podpisującego %s: %s",
                            signer,
                            exc,
                            exc_info=_LOGGER.isEnabledFor(logging.DEBUG),
                        )
                        candidate = None
                    raw = candidate if isinstance(candidate, Mapping) else None
                else:
                    raw = None

                info_dict: dict[str, Any] = dict(raw or {})
                if "algorithm" not in info_dict:
                    info_dict["algorithm"] = getattr(signer, "algorithm", "unknown")
                if "requires_hardware" not in info_dict:
                    info_dict["requires_hardware"] = bool(getattr(signer, "requires_hardware", False))
                if "key_id" not in info_dict and getattr(signer, "key_id", None) is not None:
                    info_dict["key_id"] = signer.key_id  # type: ignore[attr-defined]

                info = MappingProxyType(info_dict)
                cache[identifier] = info

            descriptions[account_id] = info

        return MappingProxyType(descriptions)

    def describe_key_index(self) -> Mapping[str, Mapping[str, Any]]:
        """Zwraca informacje audytowe o podpisujących pogrupowanych po ``key_id``.

        Każdy wpis zawiera listę kont korzystających z danego identyfikatora
        klucza (wliczając konto domyślne ``None``), liczbę unikalnych
        podpisujących, jak również zestaw używanych algorytmów i flagę,
        czy wszystkie instancje wymagają sprzętowego podpisu.
        """

        if self._key_index_dirty:
            self._rebuild_key_index()

        assignments: dict[int, list[str | None]] = {}
        for account_id, signer in self.iter_signers():
            bucket = assignments.setdefault(id(signer), [])
            bucket.append(account_id)

        summaries: dict[str, Mapping[str, Any]] = {}
        for key_id, signers in self._key_index.items():
            accounts: list[str | None] = []
            algorithms: list[str] = []
            hardware_requirements: list[bool] = []

            for signer in signers:
                accounts.extend(assignments.get(id(signer), []))

                algorithm_name = str(getattr(signer, "algorithm", "unknown"))
                if algorithm_name not in algorithms:
                    algorithms.append(algorithm_name)

                hardware_requirements.append(bool(getattr(signer, "requires_hardware", False)))

            unique_accounts = tuple(dict.fromkeys(accounts))
            hardware_modes = tuple(dict.fromkeys(hardware_requirements))
            summary: dict[str, Any] = {
                "accounts": unique_accounts,
                "account_count": len(unique_accounts),
                "signer_count": len(signers),
                "algorithms": tuple(algorithms),
                "requires_hardware": bool(signers) and all(hardware_requirements),
                "hardware_modes": hardware_modes,
                "mixed_hardware": len(hardware_modes) > 1,
            }

            summaries[key_id] = MappingProxyType(summary)

        return MappingProxyType(summaries)

    def describe_hardware_requirements(self) -> Mapping[str, Any]:
        """Agreguje informacje o wymaganiach sprzętowych per konto."""

        def _dedupe(values: Iterable[str | None]) -> tuple[str | None, ...]:
            return tuple(dict.fromkeys(values))

        accounts: list[str | None] = []
        hardware_accounts: list[str | None] = []
        software_accounts: list[str | None] = []
        missing_key_id_accounts: list[str | None] = []

        for account_id, signer in self.iter_signers():
            accounts.append(account_id)

            requires_hardware = bool(getattr(signer, "requires_hardware", False))
            if requires_hardware:
                hardware_accounts.append(account_id)
            else:
                software_accounts.append(account_id)

            key_id = getattr(signer, "key_id", None)
            if key_id is None or not str(key_id).strip():
                missing_key_id_accounts.append(account_id)

        deduped_accounts = _dedupe(accounts)
        hardware_unique = _dedupe(hardware_accounts)
        software_unique = _dedupe(software_accounts)
        missing_key_id_unique = _dedupe(missing_key_id_accounts)

        summary: dict[str, Any] = {
            "accounts": deduped_accounts,
            "total_accounts": len(deduped_accounts),
            "hardware_accounts": hardware_unique,
            "hardware_account_count": len(hardware_unique),
            "software_accounts": software_unique,
            "software_account_count": len(software_unique),
            "missing_key_id_accounts": missing_key_id_unique,
            "missing_key_id_count": len(missing_key_id_unique),
            "all_require_hardware": not software_unique,
        }

        return MappingProxyType(summary)

    def describe_audit_bundle(self) -> Mapping[str, Any]:
        """Zwraca skonsolidowany raport audytowy konfiguracji podpisów.

        W raporcie znajdują się wcześniej udostępniane dane:

        * ``signers`` – wynik :meth:`describe_signers`.
        * ``key_index`` – wynik :meth:`describe_key_index`.
        * ``hardware_requirements`` – wynik :meth:`describe_hardware_requirements`.
        * ``issues`` – lista potencjalnych problemów wykrytych na podstawie
          powyższych sekcji (np. brak ``key_id`` albo konta korzystające ze
          ``software``).
        """

        signers = self.describe_signers()
        key_index = self.describe_key_index()
        hardware_summary = self.describe_hardware_requirements()

        issues: list[Mapping[str, Any]] = []

        missing_key_accounts = tuple(hardware_summary.get("missing_key_id_accounts", ()))
        if missing_key_accounts:
            issues.append(
                MappingProxyType(
                    {
                        "type": "missing_key_id",
                        "severity": "warning",
                        "accounts": missing_key_accounts,
                        "count": int(hardware_summary.get("missing_key_id_count", len(missing_key_accounts))),
                    }
                )
            )

        software_accounts = tuple(hardware_summary.get("software_accounts", ()))
        if software_accounts:
            issues.append(
                MappingProxyType(
                    {
                        "type": "software_signer",
                        "severity": "warning",
                        "accounts": software_accounts,
                        "count": int(hardware_summary.get("software_account_count", len(software_accounts))),
                    }
                )
            )

        for key_id, summary in key_index.items():
            algorithms = summary.get("algorithms") if isinstance(summary, Mapping) else None
            if isinstance(algorithms, (list, tuple)) and len(algorithms) > 1:
                issues.append(
                    MappingProxyType(
                        {
                            "type": "key_id_algorithm_conflict",
                            "severity": "critical",
                            "key_id": key_id,
                            "algorithms": tuple(algorithms),
                            "accounts": tuple(summary.get("accounts", ())) if isinstance(summary, Mapping) else (),
                            "signer_count": int(summary.get("signer_count", 0)) if isinstance(summary, Mapping) else 0,
                        }
                    )
                )

            hardware_modes = summary.get("hardware_modes") if isinstance(summary, Mapping) else None
            mixed_hardware = summary.get("mixed_hardware") if isinstance(summary, Mapping) else None
            if isinstance(hardware_modes, (list, tuple)) and len(hardware_modes) > 1 and any(
                isinstance(mode, bool) for mode in hardware_modes
            ):
                issues.append(
                    MappingProxyType(
                        {
                            "type": "key_id_hardware_mismatch",
                            "severity": "warning",
                            "key_id": key_id,
                            "hardware_modes": tuple(bool(mode) for mode in hardware_modes),
                            "mixed_hardware": bool(mixed_hardware),
                            "accounts": tuple(summary.get("accounts", ())) if isinstance(summary, Mapping) else (),
                            "signer_count": int(summary.get("signer_count", 0)) if isinstance(summary, Mapping) else 0,
                        }
                    )
                )

        bundle: dict[str, Any] = {
            "signers": signers,
            "key_index": key_index,
            "hardware_requirements": hardware_summary,
            "issues": tuple(issues),
        }

        return MappingProxyType(bundle)

    def verify(
        self,
        account_id: str | None,
        payload: JsonPayload,
        signature: Mapping[str, Any],
    ) -> bool:
        """Weryfikuje podpis dla wskazanego konta (lub domyślnego)."""

        def _attempt(candidate: TransactionSigner) -> bool:
            verify_method = getattr(candidate, "verify", None)
            if not callable(verify_method):
                return False
            try:
                return bool(verify_method(payload, signature))
            except Exception as exc:  # noqa: BLE001
                _LOGGER.debug(
                    "Nie udało się zweryfikować podpisu z użyciem %s: %s",
                    candidate,
                    exc,
                    exc_info=_LOGGER.isEnabledFor(logging.DEBUG),
                )
                return False

        candidates: list[TransactionSigner] = []
        visited: set[int] = set()

        def _register(candidate: TransactionSigner | None) -> None:
            if candidate is None:
                return
            identifier = id(candidate)
            if identifier in visited:
                return
            visited.add(identifier)
            candidates.append(candidate)

        _register(self.resolve(account_id))

        key_id_value = signature.get("key_id")
        if isinstance(key_id_value, str):
            for candidate in self.resolve_by_key_id(key_id_value):
                _register(candidate)

        for candidate in candidates:
            if _attempt(candidate):
                return True
        return False

    def __enter__(self) -> "TransactionSignerSelector":  # pragma: no cover - sugar kontekstowy
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - sugar kontekstowy
        self.close()


def _resolve_hmac_key(config: Mapping[str, Any]) -> bytes:
    value = config.get("key_value")
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value).strip()
        if raw:
            return raw
    if isinstance(value, str) and value.strip():
        return value.encode("utf-8")

    env_name = config.get("key_env")
    if isinstance(env_name, str) and env_name.strip():
        env_value = os.environ.get(env_name.strip())
        if env_value:
            raw = env_value.strip().encode("utf-8")
            if raw:
                return raw

    path_value = config.get("key_path")
    if isinstance(path_value, (str, os.PathLike)):
        path = Path(path_value).expanduser()
        content = path.read_bytes().strip()
        if content:
            return content

    raise ValueError("Konfiguracja HMAC wymaga podania klucza (value/env/path)")


def _decode_seed(config: Mapping[str, Any]) -> bytes | None:
    seed = config.get("seed")
    if isinstance(seed, (bytes, bytearray)):
        return bytes(seed)
    if isinstance(seed, str):
        text = seed.strip()
        if not text:
            return None
        return text.encode("utf-8")
    seed_hex = config.get("seed_hex")
    if isinstance(seed_hex, str) and seed_hex.strip():
        try:
            return bytes.fromhex(seed_hex.strip())
        except ValueError:
            raise ValueError("seed_hex musi być zakodowany w hex")
    return None


def build_transaction_signer_from_config(config: Mapping[str, Any]) -> TransactionSigner:
    """Buduje ``TransactionSigner`` na podstawie mapy konfiguracyjnej."""

    if not isinstance(config, Mapping):
        raise ValueError("Konfiguracja podpisu musi być mapą")

    signer_type = str(config.get("type") or "hmac").strip().lower()
    key_id = config.get("key_id")
    key_id_text = str(key_id) if key_id is not None else None

    if signer_type in {"hmac", "mac"}:
        algorithm = str(config.get("algorithm") or "HMAC-SHA256")
        key = _resolve_hmac_key(config)
        return HmacTransactionSigner(key=key, algorithm=algorithm, key_id=key_id_text)

    if signer_type in {"ledger", "ledger_nano", "ledger-x"}:
        from bot_core.security.hardware_wallets import LedgerSigner

        return LedgerSigner(
            derivation_path=str(config.get("derivation_path") or "m/44'/60'/0'/0/0"),
            use_simulator=config.get("simulate"),
            key_id=key_id_text,
            seed=_decode_seed(config),
        )

    if signer_type in {"trezor", "trezor_t", "trezor_one"}:
        from bot_core.security.hardware_wallets import TrezorSigner

        return TrezorSigner(
            derivation_path=str(config.get("derivation_path") or "m/44'/60'/0'/0/0"),
            use_simulator=config.get("simulate"),
            key_id=key_id_text,
            seed=_decode_seed(config),
        )

    raise ValueError(f"Nieznany typ podpisującego: {signer_type}")


def build_transaction_signer_selector(config: Mapping[str, Any] | None) -> TransactionSignerSelector | None:
    """Tworzy selektor podpisów na podstawie konfiguracji runtime."""

    if not isinstance(config, Mapping) or not config:
        return None

    created_signers: list[TransactionSigner] = []

    def _register(signer: TransactionSigner) -> TransactionSigner:
        created_signers.append(signer)
        return signer

    try:
        default_signer: TransactionSigner | None = None
        overrides: dict[str, TransactionSigner] = {}

        default_cfg = config.get("default")
        if isinstance(default_cfg, Mapping):
            default_signer = _register(build_transaction_signer_from_config(default_cfg))

        accounts_cfg = config.get("accounts")
        if isinstance(accounts_cfg, Mapping):
            for account_id, account_cfg in accounts_cfg.items():
                if isinstance(account_cfg, Mapping):
                    overrides[str(account_id)] = _register(
                        build_transaction_signer_from_config(account_cfg)
                    )

        for key, value in config.items():
            if key in {"default", "accounts"}:
                continue
            if isinstance(value, Mapping):
                overrides[str(key)] = _register(build_transaction_signer_from_config(value))

        if not overrides and default_signer is None:
            return None

        created_signers.clear()
        return TransactionSignerSelector(default=default_signer, overrides=overrides or None)
    except Exception:
        visited: set[int] = set()
        for signer in created_signers:
            identifier = id(signer)
            if identifier in visited:
                continue
            visited.add(identifier)
            close_method = getattr(signer, "close", None)
            if callable(close_method):
                with contextlib.suppress(Exception):
                    close_method()
        raise


__all__ = [
    "canonical_json_bytes",
    "build_hmac_signature",
    "verify_hmac_signature",
    "validate_hmac_signature",
    "HmacSignedReportMixin",
    "TransactionSigner",
    "HmacTransactionSigner",
    "TransactionSignerSelector",
    "build_transaction_signer_from_config",
    "build_transaction_signer_selector",
]

