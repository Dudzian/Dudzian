"""Integracyjne testy środowisk demo/testnet dla adapterów Binance i Kraken."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
from cryptography.fernet import Fernet, InvalidToken

from KryptoLowca.exchanges import BinanceTestnetAdapter, KrakenDemoAdapter
from KryptoLowca.exchanges.interfaces import ExchangeCredentials, OrderRequest
from KryptoLowca.security.api_key_manager import APIKeyManager


class MissingCredentials(RuntimeError):
    """Wyjątek oznaczający brak dostępnych poświadczeń."""


@dataclass(slots=True)
class ExchangeAuthConfig:
    env_prefix: str
    manager_exchange: str
    manager_account: str


EXCHANGES = {
    "binance-testnet": ExchangeAuthConfig(
        env_prefix="BINANCE_TESTNET",
        manager_exchange=os.getenv("BINANCE_TESTNET_MANAGER_EXCHANGE", "binance-testnet"),
        manager_account=os.getenv("BINANCE_TESTNET_MANAGER_ACCOUNT", "default"),
    ),
    "kraken-demo": ExchangeAuthConfig(
        env_prefix="KRAKEN_DEMO",
        manager_exchange=os.getenv("KRAKEN_DEMO_MANAGER_EXCHANGE", "kraken-demo"),
        manager_account=os.getenv("KRAKEN_DEMO_MANAGER_ACCOUNT", "default"),
    ),
}

ENCRYPTED_FIELDS = {
    "exchange": {"api_key", "api_secret"},
}


def _transform(section: str, data: dict, transform_value) -> dict:
    if section not in ENCRYPTED_FIELDS:
        return data
    transformed = dict(data)
    for field in ENCRYPTED_FIELDS[section]:
        value = transformed.get(field)
        if isinstance(value, str) and value:
            try:
                transformed[field] = transform_value(value)
            except InvalidToken:  # pragma: no cover - błędny klucz szyfrujący
                pass
    return transformed


def _load_credentials_from_manager(config: ExchangeAuthConfig) -> Optional[ExchangeCredentials]:
    store_path = os.getenv("KRYPTLOWCA_API_KEYS_PATH")
    if not store_path:
        return None

    store = Path(store_path)
    if not store.exists():
        return None

    encryption_key = os.getenv("KRYPTLOWCA_API_KEYS_FERNET_KEY")
    fernet = Fernet(encryption_key.encode()) if encryption_key else None

    def encryptor(section: str, data):  # pragma: no cover - używane tylko przy zapisie
        if fernet is None:
            return data
        return _transform(section, data, lambda value: fernet.encrypt(value.encode()).decode())

    def decryptor(section: str, data):
        if fernet is None:
            return data

        def _decrypt(value: str) -> str:
            try:
                return fernet.decrypt(value.encode()).decode()
            except InvalidToken:
                return value

        return _transform(section, data, _decrypt)

    manager = APIKeyManager(store, encryptor=encryptor, decryptor=decryptor)
    try:
        return manager.load_credentials(config.manager_exchange, config.manager_account)
    except KeyError:
        return None


def _load_credentials_from_env(config: ExchangeAuthConfig) -> Optional[ExchangeCredentials]:
    api_key = os.getenv(f"{config.env_prefix}_API_KEY")
    api_secret = os.getenv(f"{config.env_prefix}_API_SECRET")
    if not api_key or not api_secret:
        return None
    passphrase = os.getenv(f"{config.env_prefix}_PASSPHRASE")
    return ExchangeCredentials(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase or None,
        metadata={"environment": "demo"},
    )


def load_credentials(exchange_name: str) -> ExchangeCredentials:
    config = EXCHANGES[exchange_name]
    credentials = _load_credentials_from_manager(config)
    if credentials:
        return credentials
    credentials = _load_credentials_from_env(config)
    if credentials:
        return credentials
    raise MissingCredentials(exchange_name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_binance_testnet_order_lifecycle():
    try:
        credentials = load_credentials("binance-testnet")
    except MissingCredentials:
        pytest.skip("Brak poświadczeń Binance Testnet")

    adapter = BinanceTestnetAdapter(demo_mode=True)
    await adapter.connect()
    await adapter.authenticate(credentials)

    order_request = OrderRequest(
        symbol=os.getenv("BINANCE_TESTNET_SYMBOL", "BTCUSDT"),
        side=os.getenv("BINANCE_TESTNET_SIDE", "BUY"),
        quantity=float(os.getenv("BINANCE_TESTNET_QUANTITY", "0.001")),
        order_type="LIMIT",
        price=float(os.getenv("BINANCE_TESTNET_PRICE", "20000")),
        time_in_force="GTC",
        client_order_id=f"test-{uuid.uuid4().hex[:10]}",
    )

    try:
        submitted = await adapter.submit_order(order_request)
    except RuntimeError as exc:
        pytest.skip(f"Zlecenie Binance Testnet odrzucone: {exc}")

    assert submitted.order_id

    fetched = await adapter.fetch_order_status(submitted.order_id, symbol=order_request.symbol)
    assert fetched.order_id == submitted.order_id

    canceled = await adapter.cancel_order(submitted.order_id, symbol=order_request.symbol)
    assert canceled.order_id == submitted.order_id

    await adapter.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kraken_demo_order_lifecycle():
    try:
        credentials = load_credentials("kraken-demo")
    except MissingCredentials:
        pytest.skip("Brak poświadczeń Kraken Demo")

    adapter = KrakenDemoAdapter(demo_mode=True)
    await adapter.connect()
    await adapter.authenticate(credentials)

    order_request = OrderRequest(
        symbol=os.getenv("KRAKEN_DEMO_SYMBOL", "XBTUSDT"),
        side=os.getenv("KRAKEN_DEMO_SIDE", "buy"),
        quantity=float(os.getenv("KRAKEN_DEMO_QUANTITY", "0.001")),
        order_type="LIMIT",
        price=float(os.getenv("KRAKEN_DEMO_PRICE", "20000")),
        client_order_id=f"test-{uuid.uuid4().hex[:10]}",
    )

    try:
        submitted = await adapter.submit_order(order_request)
    except RuntimeError as exc:
        pytest.skip(f"Zlecenie Kraken Demo odrzucone: {exc}")

    assert submitted.order_id

    fetched = await adapter.fetch_order_status(submitted.order_id, symbol=order_request.symbol)
    assert fetched.order_id == submitted.order_id

    canceled = await adapter.cancel_order(submitted.order_id, symbol=order_request.symbol)
    assert canceled.order_id == submitted.order_id

    await adapter.close()
