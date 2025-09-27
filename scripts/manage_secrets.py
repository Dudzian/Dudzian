"""Narzędzie CLI do zarządzania sekretami w natywnych keychainach."""
from __future__ import annotations

import argparse
import getpass
import logging
from pathlib import Path
from typing import Sequence

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage

_LOGGER = logging.getLogger("scripts.manage_secrets")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zarządza kluczami API i innymi sekretami w magazynie systemowym.",
    )
    parser.add_argument(
        "--namespace",
        default="dudzian.trading",
        help="Prefiks identyfikatorów w magazynie sekretów (domyślnie: dudzian.trading)",
    )
    parser.add_argument(
        "--headless-passphrase",
        help="Hasło do magazynu plikowego w trybie Linux headless",
    )
    parser.add_argument(
        "--headless-secret-path",
        help="Ścieżka do zaszyfrowanego magazynu w trybie headless",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania komunikatów narzędzia",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    store_exchange = subparsers.add_parser(
        "store-exchange",
        help="Zapisuje poświadczenia giełdy w magazynie sekretów",
    )
    _add_common_exchange_arguments(store_exchange)
    store_exchange.add_argument(
        "--secret",
        help="Sekret API (jeśli pominięty, narzędzie poprosi o bezpieczne wprowadzenie)",
    )
    store_exchange.add_argument(
        "--passphrase",
        help="Opcjonalne hasło/phrase wymagane przez niektóre giełdy (np. Coinbase, Kraken)",
    )
    store_exchange.add_argument(
        "--permission",
        action="append",
        default=[],
        help="Lista uprawnień (można podać wiele razy, np. --permission read --permission trade)",
    )

    show_exchange = subparsers.add_parser(
        "show-exchange",
        help="Wyświetla parametry zapisanych poświadczeń (bez pełnego sekretu)",
    )
    _add_common_exchange_arguments(show_exchange, include_key_id=False)
    show_exchange.add_argument(
        "--mask-length",
        type=int,
        default=4,
        help="Liczba widocznych znaków sekretu po zamaskowaniu (domyślnie: 4)",
    )

    delete_exchange = subparsers.add_parser(
        "delete-exchange",
        help="Usuwa zapisane poświadczenia giełdy",
    )
    _add_common_exchange_arguments(delete_exchange, include_key_id=False)

    store_secret = subparsers.add_parser(
        "store-secret",
        help="Zapisuje dowolny sekret tekstowy (np. token bota Telegram)",
    )
    _add_common_secret_arguments(store_secret)
    store_secret.add_argument(
        "--value",
        help="Wartość sekretu (jeśli pominięta, zostanie poproszony bezpieczny input)",
    )

    show_secret = subparsers.add_parser(
        "show-secret",
        help="Wyświetla informacje o sekrecie (domyślnie bez ujawniania pełnej wartości)",
    )
    _add_common_secret_arguments(show_secret)
    show_secret.add_argument(
        "--reveal",
        action="store_true",
        help="Wymusza wypisanie pełnej wartości sekretu (ostrożnie!)",
    )
    show_secret.add_argument(
        "--mask-length",
        type=int,
        default=3,
        help="Liczba widocznych znaków maskowanej wartości (domyślnie: 3)",
    )

    delete_secret = subparsers.add_parser(
        "delete-secret",
        help="Usuwa sekret zapisany w magazynie",
    )
    _add_common_secret_arguments(delete_secret)

    return parser


def _add_common_exchange_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_key_id: bool = True,
) -> None:
    parser.add_argument("--key", required=True, help="Nazwa klucza w magazynie (np. binance_paper)")
    if include_key_id:
        parser.add_argument("--key-id", required=True, help="Identyfikator API key (publiczny klucz)")
    parser.add_argument(
        "--environment",
        default=Environment.LIVE.value,
        choices=[env.value for env in Environment],
        help="Środowisko poświadczeń (live/paper/testnet)",
    )
    parser.add_argument(
        "--purpose",
        default="trading",
        help="Cel sekretu (np. trading, data, readonly)",
    )


def _add_common_secret_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key",
        required=True,
        help="Identyfikator sekretu (np. telegram_bot, email_smtp)",
    )
    parser.add_argument(
        "--purpose",
        default="generic",
        help="Cel przechowywania (np. alerts, reporting)",
    )


def _prompt_secret(prompt: str) -> str:
    value = getpass.getpass(prompt)
    if not value:
        raise ValueError("Podana wartość nie może być pusta.")
    return value


def _mask_value(value: str | None, *, visible: int) -> str:
    if not value:
        return "(brak)"
    if visible <= 0:
        return "*" * min(len(value), 8)
    if len(value) <= visible:
        return value[0] + "*" * (len(value) - 1)
    prefix = value[:visible]
    suffix = value[-1]
    hidden_length = max(len(value) - visible - 1, 0)
    masked_middle = "*" * max(hidden_length, 3)
    return f"{prefix}{masked_middle}{suffix}"


def _store_exchange_credentials(manager: SecretManager, args: argparse.Namespace) -> None:
    secret = args.secret
    if secret is None:
        secret = _prompt_secret("Wprowadź sekret API: ")
    passphrase = args.passphrase or None
    permissions: Sequence[str] = tuple(args.permission) if args.permission else ()
    credentials = ExchangeCredentials(
        key_id=args.key_id,
        secret=secret or None,
        passphrase=passphrase,
        environment=Environment(args.environment),
        permissions=permissions,
    )
    manager.store_exchange_credentials(args.key, credentials, purpose=args.purpose)
    _LOGGER.info(
        "Zapisano poświadczenia '%s' (środowisko: %s, cel: %s, uprawnienia: %s)",
        args.key,
        args.environment,
        args.purpose,
        permissions or "(brak)",
    )


def _show_exchange_credentials(manager: SecretManager, args: argparse.Namespace) -> None:
    credentials = manager.load_exchange_credentials(
        args.key,
        expected_environment=Environment(args.environment),
        purpose=args.purpose,
    )
    masked_secret = _mask_value(credentials.secret, visible=args.mask_length)
    masked_passphrase = _mask_value(credentials.passphrase, visible=min(args.mask_length, 2))
    permissions = ", ".join(credentials.permissions) if credentials.permissions else "(brak)"
    print("--- Poświadczenia giełdowe ---")
    print(f"Klucz magazynu: {args.key}")
    print(f"Środowisko: {credentials.environment.value}")
    print(f"Cel: {args.purpose}")
    print(f"ID API: {credentials.key_id}")
    print(f"Sekret: {masked_secret}")
    print(f"Passphrase: {masked_passphrase}")
    print(f"Uprawnienia: {permissions}")


def _delete_exchange_credentials(manager: SecretManager, args: argparse.Namespace) -> None:
    manager.delete_exchange_credentials(args.key, purpose=args.purpose)
    _LOGGER.info(
        "Usunięto poświadczenia '%s' (cel: %s)",
        args.key,
        args.purpose,
    )


def _store_generic_secret(manager: SecretManager, args: argparse.Namespace) -> None:
    value = args.value
    if value is None:
        value = _prompt_secret("Wprowadź wartość sekretu: ")
    manager.store_secret_value(args.key, value, purpose=args.purpose)
    _LOGGER.info("Zapisano sekret '%s' (cel: %s)", args.key, args.purpose)


def _show_generic_secret(manager: SecretManager, args: argparse.Namespace) -> None:
    value = manager.load_secret_value(args.key, purpose=args.purpose)
    if args.reveal:
        print("--- Sekret ---")
        print(f"Klucz: {args.key}")
        print(f"Cel: {args.purpose}")
        print(f"Wartość: {value}")
    else:
        masked = _mask_value(value, visible=args.mask_length)
        print("--- Sekret ---")
        print(f"Klucz: {args.key}")
        print(f"Cel: {args.purpose}")
        print(f"Długość: {len(value)} znaków")
        print(f"Podgląd: {masked}")
        print("Użyj --reveal, aby wyświetlić pełną wartość (tylko w zaufanym środowisku).")


def _delete_generic_secret(manager: SecretManager, args: argparse.Namespace) -> None:
    manager.delete_secret_value(args.key, purpose=args.purpose)
    _LOGGER.info("Usunięto sekret '%s' (cel: %s)", args.key, args.purpose)


def _create_manager(args: argparse.Namespace) -> SecretManager:
    storage = create_default_secret_storage(
        namespace=args.namespace,
        headless_passphrase=args.headless_passphrase,
        headless_path=Path(args.headless_secret_path) if args.headless_secret_path else None,
    )
    return SecretManager(storage, namespace=args.namespace)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    try:
        manager = _create_manager(args)
    except SecretStorageError as exc:
        _LOGGER.error("Nie udało się zainicjalizować magazynu sekretów: %s", exc)
        return 1

    try:
        if args.command == "store-exchange":
            _store_exchange_credentials(manager, args)
        elif args.command == "show-exchange":
            _show_exchange_credentials(manager, args)
        elif args.command == "delete-exchange":
            _delete_exchange_credentials(manager, args)
        elif args.command == "store-secret":
            _store_generic_secret(manager, args)
        elif args.command == "show-secret":
            _show_generic_secret(manager, args)
        elif args.command == "delete-secret":
            _delete_generic_secret(manager, args)
        else:  # pragma: no cover - zabezpieczenie przed nowymi komendami
            parser.error(f"Nieobsługiwane polecenie: {args.command}")
            return 2
    except SecretStorageError as exc:
        _LOGGER.error("Operacja na magazynie nie powiodła się: %s", exc)
        return 1
    except ValueError as exc:
        _LOGGER.error("Nieprawidłowe dane wejściowe: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
