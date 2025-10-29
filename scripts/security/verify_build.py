"""Verify integrity and signatures of build artifacts."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key

DEFAULT_DIGEST = "sha256"
SUPPORTED_ALGORITHMS = ("ed25519", "rsa")


@dataclass(frozen=True)
class VerificationResult:
    artifact: Path
    signature_path: Path
    public_key_path: Path
    algorithm: str
    digest_algorithm: str
    digest: str
    verified: bool
    details: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "artifact": str(self.artifact),
            "signature": str(self.signature_path),
            "public_key": str(self.public_key_path),
            "algorithm": self.algorithm,
            "digest_algorithm": self.digest_algorithm,
            "digest": self.digest,
            "verified": self.verified,
            "details": self.details,
        }


def _decode_signature(data: bytes, *, fmt: str) -> bytes:
    if fmt == "raw":
        return data
    text = data.decode("utf-8").strip()
    if fmt == "hex":
        try:
            return bytes.fromhex(text)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Signature file is not valid hexadecimal") from exc
    if fmt == "base64":
        try:
            return base64.b64decode(text, validate=True)
        except binascii.Error as exc:
            raise ValueError("Signature file is not valid base64") from exc
    # auto-detect
    for decoder in (base64.b64decode, lambda t: bytes.fromhex(t)):
        try:
            return decoder(text)
        except (binascii.Error, ValueError):
            continue
    return data


def _load_public_key(path: Path, *, algorithm: str):
    key_data = path.read_bytes()
    key = load_pem_public_key(key_data)
    if algorithm == "ed25519" and not isinstance(key, ed25519.Ed25519PublicKey):
        raise ValueError("Public key is not an Ed25519 key")
    if algorithm == "rsa" and not isinstance(key, rsa.RSAPublicKey):
        raise ValueError("Public key is not an RSA key")
    return key


def _compute_digest(path: Path, *, algorithm: str) -> str:
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:  # pragma: no cover - invalid digest configuration
        raise ValueError(f"Unsupported digest algorithm: {algorithm}") from exc
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_signature(
    *,
    algorithm: str,
    public_key,
    data: bytes,
    signature: bytes,
) -> None:
    if algorithm == "ed25519":
        public_key.verify(signature, data)
        return
    if algorithm == "rsa":
        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return
    raise ValueError(f"Unsupported signature algorithm: {algorithm}")


def _write_report(report_path: Optional[Path], result: VerificationResult) -> None:
    if not report_path:
        return
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Ścieżka do zweryfikowanego pliku binarnego")
    parser.add_argument(
        "--signature",
        required=True,
        type=Path,
        help="Ścieżka do podpisu (domyślnie auto-detekcja formatu)",
    )
    parser.add_argument(
        "--public-key",
        required=True,
        type=Path,
        help="Ścieżka do klucza publicznego PEM",
    )
    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_ALGORITHMS,
        default="ed25519",
        help="Algorytm podpisu (domyślnie ed25519)",
    )
    parser.add_argument(
        "--signature-format",
        choices=("auto", "raw", "base64", "hex"),
        default="auto",
        help="Format pliku podpisu",
    )
    parser.add_argument(
        "--digest",
        default=DEFAULT_DIGEST,
        help="Algorytm skrótu do walidacji integralności (domyślnie sha256)",
    )
    parser.add_argument(
        "--expected-digest",
        help="Oczekiwany skrót heksadecymalny; jeśli podany zostanie zweryfikowany",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Ścieżka do pliku JSON z wynikiem weryfikacji",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    artifact = args.artifact.expanduser().resolve()
    signature_path = args.signature.expanduser().resolve()
    public_key_path = args.public_key.expanduser().resolve()

    if not artifact.is_file():
        raise FileNotFoundError(f"Nie znaleziono artefaktu: {artifact}")
    if not signature_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku podpisu: {signature_path}")
    if not public_key_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono klucza publicznego: {public_key_path}")

    digest = _compute_digest(artifact, algorithm=args.digest)
    details: Optional[str] = None

    if args.expected_digest and digest.lower() != args.expected_digest.lower():
        details = (
            f"Niezgodność skrótu: oczekiwano {args.expected_digest}, otrzymano {digest}"
        )
        result = VerificationResult(
            artifact=artifact,
            signature_path=signature_path,
            public_key_path=public_key_path,
            algorithm=args.algorithm,
            digest_algorithm=args.digest,
            digest=digest,
            verified=False,
            details=details,
        )
        _write_report(args.report, result)
        raise ValueError(details)

    signature_bytes = _decode_signature(
        signature_path.read_bytes(), fmt=args.signature_format
    )
    data = artifact.read_bytes()
    public_key = _load_public_key(public_key_path, algorithm=args.algorithm)

    try:
        _verify_signature(
            algorithm=args.algorithm,
            public_key=public_key,
            data=data,
            signature=signature_bytes,
        )
        verified = True
    except InvalidSignature as exc:
        verified = False
        details = "Podpis nie jest prawidłowy"
        result = VerificationResult(
            artifact=artifact,
            signature_path=signature_path,
            public_key_path=public_key_path,
            algorithm=args.algorithm,
            digest_algorithm=args.digest,
            digest=digest,
            verified=verified,
            details=details,
        )
        _write_report(args.report, result)
        raise ValueError(details) from exc

    result = VerificationResult(
        artifact=artifact,
        signature_path=signature_path,
        public_key_path=public_key_path,
        algorithm=args.algorithm,
        digest_algorithm=args.digest,
        digest=digest,
        verified=verified,
    )
    _write_report(args.report, result)
    print(
        "Zweryfikowano podpis dla {artifact} (algorytm={alg}, digest={digest_alg})".format(
            artifact=artifact.name,
            alg=args.algorithm,
            digest_alg=args.digest,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
