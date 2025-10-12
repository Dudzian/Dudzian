import hashlib
import os

import pytest

from bot_core.config.models import ServiceTokenConfig
from bot_core.security.tokens import (
    ServiceToken,
    ServiceTokenValidator,
    build_service_token_validator,
    resolve_service_token,
    resolve_service_token_secret,
)


def test_service_token_from_config_env(monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "env-secret")
    config = ServiceTokenConfig(
        token_id="env",
        token_env="METRICS_TOKEN",
        scopes=("metrics.read",),
    )
    token = ServiceToken.from_config(config, env=os.environ)
    assert token.secret == "env-secret"
    assert token.hash_algorithm is None
    assert token.hashed_value is None
    assert token.matches("env-secret", scope="metrics.read")


def test_service_token_hash_validation():
    digest = hashlib.sha256(b"scope-secret").hexdigest()
    config = ServiceTokenConfig(
        token_id="hashed",
        token_hash=f"sha256:{digest}",
        scopes=("metrics.write",),
    )
    token = ServiceToken.from_config(config)
    assert token.secret is None
    assert token.hash_algorithm == "sha256"
    assert token.hashed_value == digest
    assert token.matches("scope-secret", scope="metrics.write")
    assert token.matches("scope-secret", scope="metrics.read") is False


def test_service_token_validator_scopes(monkeypatch):
    monkeypatch.setenv("WRITER_TOKEN", "writer")
    reader_digest = hashlib.sha256(b"reader").hexdigest()
    validator = build_service_token_validator(
        [
            ServiceTokenConfig(
                token_id="writer",
                token_env="WRITER_TOKEN",
                scopes=("metrics.write",),
            ),
            ServiceTokenConfig(
                token_id="reader",
                token_hash=f"sha256:{reader_digest}",
                scopes=("metrics.read",),
            ),
        ],
        default_scope="metrics.read",
        env=os.environ,
    )
    assert isinstance(validator, ServiceTokenValidator)
    assert validator.validate("writer", scope="metrics.write") is not None
    assert validator.validate("writer", scope="metrics.read") is None
    assert validator.validate("reader", scope="metrics.read") is not None
    assert validator.validate("reader", scope="metrics.write") is None
    meta = validator.metadata()
    assert meta["default_scope"] == "metrics.read"
    assert any(entry["token_id"] == "writer" for entry in meta["tokens"])


def test_resolve_service_token_secret(monkeypatch):
    monkeypatch.setenv("STREAM_READER", "stream-secret")
    configs = [
        ServiceTokenConfig(
            token_id="reader",
            token_env="STREAM_READER",
            scopes=("metrics.read",),
        ),
        ServiceTokenConfig(
            token_id="writer",
            token_value="writer-secret",
            scopes=("metrics.write",),
        ),
    ]

    token = resolve_service_token(configs, scope="metrics.read", env=os.environ)
    assert token is not None
    assert token.secret == "stream-secret"
    assert token.token_id == "reader"
    assert sorted(token.scopes) == ["metrics.read"]
    assert (
        resolve_service_token_secret(configs, scope="metrics.read", env=os.environ)
        == "stream-secret"
    )
    assert (
        resolve_service_token_secret(configs, scope="metrics.write", env=os.environ)
        == "writer-secret"
    )


def test_resolve_service_token_secret_skips_hashed():
    digest = hashlib.sha256(b"foo").hexdigest()
    configs = [
        ServiceTokenConfig(
            token_id="hashed",
            token_hash=f"sha256:{digest}",
            scopes=("metrics.read",),
        )
    ]

    assert resolve_service_token(configs, scope="metrics.read") is None
    assert resolve_service_token_secret(configs, scope="metrics.read") is None
