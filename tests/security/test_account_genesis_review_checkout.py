"""Regresje checkoutu: surowe bajty i odmowy, bez normalizacji walidatora."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core import freshness_runtime_ipc as ipc
from tests.security import account_genesis_transport_review_support as account
from tests.security import freshness_verifier_remediation_support as verifier
from tests.security import test_account_genesis_verified_transport_contract as contract

# Dodatkowa regresja poza zamkniętym wykazem S9-FV-01; fixture pozostają wspólne.
module = contract.module
authority = contract.authority

ROOT = Path(__file__).resolve().parents[2]
PATCHES = (
    "tests/fixtures/account_genesis_verified_transport_review.patch",
    "tests/fixtures/freshness_verifier_remediation_review.patch",
)
PROTECTED = (account.SOURCE, *PATCHES)
SOURCES = (account.SOURCE, verifier.SOURCE, *PATCHES)


def _git(root, *arguments, autocrlf=None, check=True):
    command = ["git"]
    if autocrlf is not None:
        command += ["-c", f"core.autocrlf={autocrlf}", "-c", "core.safecrlf=false"]
    return subprocess.run(
        [*command, *arguments], cwd=root, capture_output=True, text=True, check=check
    )


def _facts(root, label, *, autocrlf=None):
    """Transformacja CRLF→LF służy wyłącznie diagnostyce, nigdy loaderowi."""
    report = {
        "checkout": label,
        "platform": platform.system(),
        "git": _git(root, "--version").stdout.strip(),
        "core.autocrlf": _git(
            root,
            "config",
            "--show-origin",
            "--get",
            "core.autocrlf",
            autocrlf=autocrlf,
            check=False,
        ).stdout.strip()
        or "unset",
        "attributes": _git(root, "check-attr", "text", "eol", "--", *PROTECTED).stdout,
        "files": {},
    }
    for name in PROTECTED:
        raw = (root / name).read_bytes()
        report["files"][name] = {
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "diagnostic_crlf_to_lf_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
            "bytes": len(raw),
            "crlf_count": raw.count(b"\r\n"),
        }
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return report


def _checkout(tmp_path, attributes):
    """Prawdziwy Git; ustawienia -c dotyczą wyłącznie tymczasowej operacji."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "--quiet")
    (root / ".gitattributes").write_bytes(attributes)
    for name in SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    _git(root, "add", "--", ".", autocrlf="false")
    for name in SOURCES:
        (root / name).unlink()
    _git(root, "checkout-index", "--all", "--force", autocrlf="true")
    return root


def test_current_checkout_preserves_raw_source_and_exact_lf_attributes():
    facts = _facts(ROOT, "actual-runner")
    for name in PROTECTED:
        assert f"{name}: text: set" in facts["attributes"]
        assert f"{name}: eol: lf" in facts["attributes"]
        assert facts["files"][name]["crlf_count"] == 0
    assert hashlib.sha256((ROOT / account.SOURCE).read_bytes()).hexdigest() == (
        account.BASELINE_SOURCE_SHA256
    )


def test_previous_autocrlf_checkout_is_rejected_as_raw_bytes(tmp_path, monkeypatch):
    # Dokładnie reguły dawnego checkoutu istotne dla tych czterech plików.
    root = _checkout(
        tmp_path,
        b"* text=auto\nbot_core/freshness_semantic_verifier.py text eol=lf\n",
    )
    facts = _facts(root, "previous-attributes-control", autocrlf="true")
    source = facts["files"][account.SOURCE]
    assert source["raw_sha256"] == (
        "aa809fdafcceaf66f1afadec9732a83b5daff8057fa54e81ddc0bb1758acad83"
    )
    assert source["diagnostic_crlf_to_lf_sha256"] == account.BASELINE_SOURCE_SHA256
    assert all(facts["files"][name]["crlf_count"] > 0 for name in PROTECTED)
    monkeypatch.setattr(account, "ROOT", root)
    monkeypatch.setattr(account, "PATCH", root / "absent.patch")
    with pytest.raises(ValueError, match="zmieniła się dokładna baza"):
        account.load_account_genesis(tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_lf_checkout_applies_both_review_patches_to_exact_bases(tmp_path, monkeypatch):
    root = _checkout(tmp_path, (ROOT / ".gitattributes").read_bytes())
    facts = _facts(root, "fixed-attributes-control", autocrlf="true")
    assert all(facts["files"][name]["crlf_count"] == 0 for name in PROTECTED)
    monkeypatch.setenv("FRESHNESS_REMEDIATION_SOURCE", "candidate")
    for support, patch, loader in (
        (account, PATCHES[0], account.load_account_genesis),
        (verifier, PATCHES[1], verifier.load_verifier),
    ):
        monkeypatch.setattr(support, "ROOT", root)
        monkeypatch.setattr(support, "PATCH", root / patch)
        module = loader(tmp_path / Path(patch).stem)
        assert Path(module.__file__).read_bytes() != (root / support.SOURCE).read_bytes()
        assert (root / support.SOURCE).read_bytes() == (ROOT / support.SOURCE).read_bytes()


@pytest.mark.parametrize("mutation", ["single-byte", "crlf"])
def test_review_loader_refuses_unapproved_raw_bytes_before_patch(tmp_path, monkeypatch, mutation):
    raw = (ROOT / account.SOURCE).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == account.BASELINE_SOURCE_SHA256
    changed = (
        bytes([raw[0] ^ 1]) + raw[1:] if mutation == "single-byte" else raw.replace(b"\n", b"\r\n")
    )
    path = tmp_path / "source" / account.SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes(changed)
    monkeypatch.setattr(account, "ROOT", tmp_path / "source")
    monkeypatch.setattr(account, "PATCH", tmp_path / "absent.patch")
    with pytest.raises(ValueError, match="zmieniła się dokładna baza"):
        account.load_account_genesis(tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_missing_af_unix_refuses_before_identity_transport_or_sql(authority, monkeypatch):
    """Brak transportu nadal odmawia, także w modelu platformy Linux."""
    monkeypatch.setattr(ipc, "sys", SimpleNamespace(platform="linux"))
    constructor = Mock(side_effect=AssertionError("Brak AF_UNIX musi poprzedzać socket"))
    monkeypatch.setattr(ipc, "socket", SimpleNamespace(socket=constructor))
    identity = Mock(side_effect=AssertionError("Brak AF_UNIX musi poprzedzać tożsamość"))
    transport = Mock(side_effect=AssertionError("Brak AF_UNIX musi poprzedzać transport"))
    monkeypatch.setattr(ipc, "_require_runtime_identity", identity)
    monkeypatch.setattr(ipc, "_pinned_verifier_socket", transport)
    sql = contract._forbid_sql(monkeypatch)
    assert authority.authenticate_and_advance(b"p", b"d", b"r").outcome.value == "UNAVAILABLE"
    identity.assert_not_called()
    transport.assert_not_called()
    constructor.assert_not_called()
    sql.assert_not_called()
