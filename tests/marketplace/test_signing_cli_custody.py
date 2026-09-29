from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from bot_core.marketplace.signing_key_policy import SigningKeyPolicyError
from scripts import (
    build_marketplace_catalog,
    reconcile_exchange_presets,
    sign_marketplace_presets,
    ui_marketplace_bridge,
)


REPO_HMAC = "config/marketplace/keys/dev-hmac.key"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _external_secrets(tmp_path: Path) -> tuple[Path, Path]:
    private_key = tmp_path / "custody" / "production-ed25519.key"
    hmac_key = tmp_path / "custody" / "production-hmac.key"
    private_key.parent.mkdir()
    private_key.write_bytes(b"external ed25519 material")
    hmac_key.write_bytes(b"external hmac material")
    return private_key, hmac_key


def test_build_catalog_production_rejects_repo_hmac(tmp_path: Path) -> None:
    private_key, _ = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        build_marketplace_catalog.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--key-id",
                "production",
                "--signing-key",
                f"catalog:{REPO_HMAC}",
                "--catalog-signature-key",
                "catalog",
            ]
        )


@pytest.mark.parametrize(
    "script,extra_args",
    [
        (
            "scripts/build_marketplace_catalog.py",
            [
                "--key-id",
                "production",
                "--signing-key",
                f"catalog:{REPO_HMAC}",
                "--catalog-signature-key",
                "catalog",
            ],
        ),
        (
            "scripts/sign_marketplace_presets.py",
            [
                "--hmac-key",
                REPO_HMAC,
                "--hmac-key-id",
                "marketplace-prod-hmac-v1",
                "--ed25519-key-id",
                "marketplace-prod-ed25519-v1",
            ],
        ),
    ],
)
def test_production_cli_returns_nonzero_for_repo_hmac(
    tmp_path: Path, script: str, extra_args: list[str]
) -> None:
    private_key, _ = _external_secrets(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            script,
            "--environment",
            "production",
            "--private-key",
            str(private_key),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "repository-contained" in result.stderr


def test_build_catalog_production_rejects_all_keys_if_one_hmac_is_in_repo(
    tmp_path: Path,
) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        build_marketplace_catalog.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--key-id",
                "production",
                "--signing-key",
                f"external:{external_hmac}",
                "--signing-key",
                f"catalog:{REPO_HMAC}",
                "--catalog-signature-key",
                "catalog",
            ]
        )


def test_build_catalog_production_rejects_reserved_dev_key_id(tmp_path: Path) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match="dev-presets"):
        build_marketplace_catalog.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--key-id",
                "dev-presets",
                "--signing-key",
                f"marketplace-prod-hmac-v1:{external_hmac}",
            ]
        )


def test_build_catalog_production_rejects_reserved_signing_key_id(tmp_path: Path) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match="dev-hmac"):
        build_marketplace_catalog.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--key-id",
                "marketplace-prod-ed25519-v1",
                "--signing-key",
                f"dev-hmac:{external_hmac}",
            ]
        )


def test_build_catalog_production_accepts_external_signing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(build_marketplace_catalog, "_load_ed25519_private_key", lambda path: path)
    monkeypatch.setattr(
        build_marketplace_catalog, "build_catalog", lambda **kwargs: calls.append(kwargs)
    )

    assert (
        build_marketplace_catalog.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--key-id",
                "production",
                "--signing-key",
                f"marketplace-prod-hmac-v1:{external_hmac}",
                "--catalog-signature-key",
                "marketplace-prod-hmac-v1",
                "--issuer",
                "marketplace-production",
            ]
        )
        == 0
    )
    assert calls and calls[0]["signing_keys"] == {
        "marketplace-prod-hmac-v1": b"external hmac material"
    }
    assert calls[0]["issuer"] == "marketplace-production"


@pytest.mark.parametrize("issuer", [None, "marketplace-ci"])
def test_build_catalog_production_rejects_missing_or_dev_issuer(
    tmp_path: Path, issuer: str | None
) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)
    args = [
        "--environment",
        "production",
        "--private-key",
        str(private_key),
        "--key-id",
        "marketplace-prod-ed25519-v1",
        "--signing-key",
        f"marketplace-prod-hmac-v1:{external_hmac}",
    ]
    if issuer is not None:
        args.extend(["--issuer", issuer])

    expected = "explicit --issuer" if issuer is None else "marketplace-ci"
    with pytest.raises(SigningKeyPolicyError, match=expected):
        build_marketplace_catalog.main(args)


def test_sign_presets_production_rejects_missing_hmac(tmp_path: Path) -> None:
    private_key, _ = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match="explicit external.*--hmac-key"):
        sign_marketplace_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--hmac-key-id",
                "marketplace-prod-hmac-v1",
                "--ed25519-key-id",
                "marketplace-prod-ed25519-v1",
            ]
        )


def test_sign_presets_production_rejects_missing_explicit_identities(tmp_path: Path) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match="explicit --hmac-key-id"):
        sign_marketplace_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--hmac-key",
                str(external_hmac),
            ]
        )

    with pytest.raises(SigningKeyPolicyError, match="explicit --ed25519-key-id"):
        sign_marketplace_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--hmac-key",
                str(external_hmac),
                "--hmac-key-id",
                "marketplace-prod-hmac-v1",
            ]
        )


@pytest.mark.parametrize(
    "extra_args,reserved_id",
    [
        (
            [
                "--hmac-key-id",
                "marketplace-prod-hmac-v1",
                "--ed25519-key-id",
                "dev-presets-ed25519",
            ],
            "dev-presets-ed25519",
        ),
        (
            [
                "--hmac-key-id",
                "dev-hmac",
                "--ed25519-key-id",
                "marketplace-prod-ed25519-v1",
            ],
            "dev-hmac",
        ),
    ],
)
def test_sign_presets_production_rejects_reserved_dev_identities(
    tmp_path: Path, extra_args: list[str], reserved_id: str
) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match=reserved_id):
        sign_marketplace_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--hmac-key",
                str(external_hmac),
                *extra_args,
            ]
        )


def test_sign_presets_production_rejects_repo_hmac(tmp_path: Path) -> None:
    private_key, _ = _external_secrets(tmp_path)

    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        sign_marketplace_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--hmac-key",
                REPO_HMAC,
                "--hmac-key-id",
                "marketplace-prod-hmac-v1",
                "--ed25519-key-id",
                "marketplace-prod-ed25519-v1",
            ]
        )


def test_sign_presets_production_accepts_external_signing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(sign_marketplace_presets, "_load_ed25519_key", lambda path: path)
    monkeypatch.setattr(
        sign_marketplace_presets, "sign_presets", lambda **kwargs: calls.append(kwargs) or 0
    )

    assert (
        sign_marketplace_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--hmac-key",
                str(external_hmac),
                "--hmac-key-id",
                "marketplace-prod-hmac-v1",
                "--ed25519-key-id",
                "marketplace-prod-ed25519-v1",
                "--issuer",
                "marketplace-production",
            ]
        )
        == 0
    )
    assert calls and calls[0]["hmac_key"] == b"external hmac material"


@pytest.mark.parametrize("issuer", [None, "marketplace-ci"])
def test_sign_presets_production_rejects_missing_or_dev_issuer(
    tmp_path: Path, issuer: str | None
) -> None:
    private_key, external_hmac = _external_secrets(tmp_path)
    args = [
        "--environment",
        "production",
        "--private-key",
        str(private_key),
        "--hmac-key",
        str(external_hmac),
        "--hmac-key-id",
        "marketplace-prod-hmac-v1",
        "--ed25519-key-id",
        "marketplace-prod-ed25519-v1",
    ]
    if issuer is not None:
        args.extend(["--issuer", issuer])

    expected = "explicit --issuer" if issuer is None else "marketplace-ci"
    with pytest.raises(SigningKeyPolicyError, match=expected):
        sign_marketplace_presets.main(args)


def test_review_signing_production_rejects_repo_secret_file() -> None:
    with pytest.raises(SigningKeyPolicyError, match="repository-contained"):
        ui_marketplace_bridge._load_signing_keys([], [REPO_HMAC], environment="production")


def test_review_signing_production_accepts_external_secret_file(tmp_path: Path) -> None:
    key_file = tmp_path / "custody" / "review-keys.json"
    key_file.parent.mkdir()
    key_file.write_text('{"reviews": "external review secret"}', encoding="utf-8")

    assert ui_marketplace_bridge._load_signing_keys(
        [], [str(key_file)], environment="production"
    ) == {"reviews": b"external review secret"}


def test_review_signing_production_rejects_reserved_dev_identity(tmp_path: Path) -> None:
    key_file = tmp_path / "custody" / "review-keys.json"
    key_file.parent.mkdir()
    key_file.write_text('{"dev-hmac": "external review secret"}', encoding="utf-8")

    with pytest.raises(SigningKeyPolicyError, match="dev-hmac"):
        ui_marketplace_bridge._command_submit_review(
            Namespace(
                presets_dir=str(tmp_path / "presets"),
                licenses_path=str(tmp_path / "licenses.json"),
                reviews_dir=str(tmp_path / "reviews"),
                signing_key=[],
                signing_key_file=[str(key_file)],
                environment="production",
                review_key_id="dev-hmac",
            )
        )


def test_reconcile_production_rejects_missing_key_id(tmp_path: Path) -> None:
    private_key, _ = _external_secrets(tmp_path)
    public_key = tmp_path / "custody" / "production.pub"
    public_key.write_bytes(b"public verification material")

    with pytest.raises(SigningKeyPolicyError, match="explicit --key-id"):
        reconcile_exchange_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--public-key",
                str(public_key),
                "--issuer",
                "marketplace-production",
            ]
        )


def test_reconcile_production_rejects_dev_issuer(tmp_path: Path) -> None:
    private_key, _ = _external_secrets(tmp_path)
    public_key = tmp_path / "custody" / "production.pub"
    public_key.write_bytes(b"public verification material")

    with pytest.raises(SigningKeyPolicyError, match="marketplace-ci"):
        reconcile_exchange_presets.main(
            [
                "--environment",
                "production",
                "--private-key",
                str(private_key),
                "--public-key",
                str(public_key),
                "--key-id",
                "marketplace-prod-ed25519-v1",
                "--issuer",
                "marketplace-ci",
            ]
        )
