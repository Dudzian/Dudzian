"""Regenerate inert TEST-only public fixtures in an isolated temporary key home.

This helper is never part of the Observer runtime. It creates no trusted root,
never imports into a user's key home, and never leaves a private key in this
fixture directory. Byte identities change on regeneration because TEST keys are
freshly random. Only public key material and detached signatures are retained.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _gpg(home: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    executable = shutil.which("gpg")
    if executable is None:
        raise RuntimeError("Test fixture generation requires a local gpg executable")
    result = subprocess.run(
        [
            executable,
            "--no-options",
            "--homedir",
            str(home),
            "--batch",
            "--no-tty",
            "--pinentry-mode",
            "loopback",
            "--passphrase",
            "",
            "--faked-system-time",
            "20260101T000000",
            *arguments,
        ],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GNUPGHOME": str(home)},
    )
    return result.stdout


def _fingerprints(home: Path) -> list[str]:
    listing = _gpg(home, "--with-colons", "--list-keys").decode("ascii")
    return [line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:")]


def regenerate(destination: Path) -> None:
    release = b"Origin: SYNTHETIC TEST ONLY\nSHA256:\n " + b"1" * 64 + b" 1 Packages\n"
    output: dict[str, bytes] = {"Release": release}
    fingerprints: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="stage9-apt-v4b-test-") as temporary:
        root = Path(temporary)
        for role in ("allowed", "unknown"):
            home = root / role
            home.mkdir(mode=0o700)
            _gpg(
                home, "--quick-generate-key", f"SYNTHETIC TEST ONLY {role}", "ed25519", "cert", "0"
            )
            primary = _fingerprints(home)[0]
            _gpg(home, "--quick-add-key", primary, "ed25519", "sign", "0")
            signing = _fingerprints(home)[1]
            fingerprints[role + "_primary"] = primary
            fingerprints[role + "_signing"] = signing
            output[role + "-keyring.gpg"] = _gpg(home, "--export", primary)
            output[role + ".asc"] = _gpg(
                home,
                "--armor",
                "--digest-algo",
                "SHA256",
                "--local-user",
                signing + "!",
                "--detach-sign",
                input_bytes=release,
            )
            if role == "allowed":
                _gpg(home, "--quick-add-key", primary, "ed25519", "sign", "0")
                other = _fingerprints(home)[2]
                fingerprints["disallowed_signing_fingerprint"] = other
                output["wrong-subkey.asc"] = _gpg(
                    home,
                    "--armor",
                    "--digest-algo",
                    "SHA256",
                    "--local-user",
                    other + "!",
                    "--detach-sign",
                    input_bytes=release,
                )
                output["allowed-keyring.gpg"] = _gpg(home, "--export", primary)
                output["weak-hash.asc"] = _gpg(
                    home,
                    "--armor",
                    "--digest-algo",
                    "SHA1",
                    "--local-user",
                    signing + "!",
                    "--detach-sign",
                    input_bytes=release,
                )
                output["expired.asc"] = _gpg(
                    home,
                    "--armor",
                    "--digest-algo",
                    "SHA256",
                    "--default-sig-expire",
                    "1d",
                    "--local-user",
                    signing + "!",
                    "--detach-sign",
                    input_bytes=release,
                )
                certificate = (home / "openpgp-revocs.d" / (primary + ".rev")).read_text()
                revocation = certificate[
                    certificate.index(":-----BEGIN PGP PUBLIC KEY BLOCK-----") + 1 :
                ]
                _gpg(home, "--import", input_bytes=revocation.encode("ascii"))
                output["revoked-keyring.gpg"] = _gpg(home, "--export", primary)
            gpgconf = shutil.which("gpgconf")
            if gpgconf is not None:
                subprocess.run(
                    [gpgconf, "--homedir", str(home), "--kill", "gpg-agent"],
                    check=True,
                    timeout=10,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GNUPGHOME": str(home)},
                )
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in output.items():
        (destination / name).write_bytes(raw)
    manifest = {
        "scope": "SYNTHETIC_TEST_ONLY_NO_PRODUCTION_TRUST",
        "fingerprints": fingerprints,
        "signature_timestamp": 1767225600,
        "public_key_algorithm": 22,
        "hash_algorithm": 8,
        "files": [
            {"name": name, "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in output.items()
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    os.umask(0o077)
    regenerate(Path(__file__).parent)
