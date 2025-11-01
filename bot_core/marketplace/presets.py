from __future__ import annotations

import base64
import json
import logging
import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.security.signing import canonical_json_bytes, verify_hmac_signature

LOGGER = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"json", "yaml", "yml"}
_DIACRITIC_OVERRIDES = str.maketrans({
    "ł": "l",
    "Ł": "L",
    "đ": "d",
    "Đ": "D",
    "Ø": "O",
    "ø": "o",
})
_TOKEN_SPLITTER = re.compile(r"(\d+|(?<![A-Za-z])[IVXLCDM]+(?![A-Za-z]))", re.IGNORECASE)
_ROMAN_NUMERAL_PATTERN = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
_DEFAULT_SIGNATURE_ALGORITHM = "ed25519"


@dataclass(slots=True)
class PresetSignatureVerification:
    """Informacje o wynikach weryfikacji podpisu presetu."""

    verified: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    algorithm: str | None = None
    key_id: str | None = None


@dataclass(slots=True)
class PresetSignature:
    """Metadane podpisu presetu Marketplace."""

    algorithm: str
    value: str
    key_id: str | None = None
    public_key: str | None = None
    signed_at: str | None = None
    issuer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "algorithm": self.algorithm,
            "value": self.value,
        }
        if self.key_id:
            document["key_id"] = self.key_id
        if self.public_key:
            document["public_key"] = self.public_key
        if self.signed_at:
            document["signed_at"] = self.signed_at
        if self.issuer:
            document["issuer"] = self.issuer
        return document


@dataclass(slots=True)
class PresetDocument:
    """Reprezentuje plik presetu Marketplace wraz z podpisem."""

    payload: Mapping[str, Any]
    signature: PresetSignature | None
    verification: PresetSignatureVerification
    fmt: str
    path: Path | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def metadata(self) -> Mapping[str, Any]:
        raw = self.payload.get("metadata", {})
        if isinstance(raw, Mapping):
            return raw
        return {}

    @property
    def preset_id(self) -> str:
        meta = self.metadata
        value = meta.get("id") if isinstance(meta, Mapping) else None
        return str(value).strip() if value not in (None, "") else ""

    @property
    def version(self) -> str | None:
        meta = self.metadata
        value = meta.get("version") if isinstance(meta, Mapping) else None
        if value in (None, ""):
            return None
        return str(value).strip()

    @property
    def name(self) -> str | None:
        value = self.payload.get("name")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def tags(self) -> Sequence[str]:
        tags = self.metadata.get("tags") if isinstance(self.metadata, Mapping) else None
        if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
            return tuple(str(tag) for tag in tags)
        return ()


def canonical_preset_bytes(payload: Mapping[str, Any]) -> bytes:
    """Zwraca kanoniczną reprezentację JSON dla payloadu presetu."""

    return canonical_json_bytes(payload)


def _normalize_format(filename: str | None) -> str | None:
    if not filename:
        return None
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix in _SUPPORTED_FORMATS:
        return "yaml" if suffix in {"yaml", "yml"} else "json"
    return None


def _roman_to_int(value: str) -> int | None:
    if not _ROMAN_NUMERAL_PATTERN.match(value):
        return None

    mapping = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    i = 0
    upper = value.upper()
    while i < len(upper):
        current = mapping[upper[i]]
        if i + 1 < len(upper):
            nxt = mapping[upper[i + 1]]
            if nxt > current:
                total += nxt - current
                i += 2
                continue
        total += current
        i += 1
    return total


def _tokenize_sort_key(key: str) -> list[str]:
    """Dzieli napis na tokeny tekstowe i numeryczne na potrzeby sortowania."""

    tokens: list[str] = []
    last_index = 0
    for match in _TOKEN_SPLITTER.finditer(key):
        start, end = match.span()
        if start > last_index:
            tokens.extend(_split_unicode_numeric_segments(key[last_index:start]))
        tokens.append(match.group(0))
        last_index = end
    if last_index < len(key):
        tokens.extend(_split_unicode_numeric_segments(key[last_index:]))
    return tokens


def _split_unicode_numeric_segments(value: str) -> list[str]:
    """Rozbija fragment na tekst i znaki numeryczne UNICODE."""

    if not value:
        return []

    segments: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        char = value[index]
        category = unicodedata.category(char)
        if category == "Nl":
            if buffer:
                segments.append("".join(buffer))
                buffer.clear()
            start = index
            index += 1
            while index < length and unicodedata.category(value[index]) == "Nl":
                index += 1
            segments.append(value[start:index])
            continue
        if category == "No":
            if buffer:
                segments.append("".join(buffer))
                buffer.clear()
            segments.append(char)
            index += 1
            continue
        buffer.append(char)
        index += 1

    if buffer:
        segments.append("".join(buffer))
    return segments


def _coerce_unicode_number(token: str) -> tuple[int, object]:
    """Próbuje zinterpretować znak lub sekwencję numeryczną UNICODE."""

    if not token:
        return (0, token)

    categories = {unicodedata.category(char) for char in token}

    if categories <= {"Nl"}:
        ascii_candidate = unicodedata.normalize("NFKC", token)
        if ascii_candidate and _ROMAN_NUMERAL_PATTERN.match(ascii_candidate.upper()):
            roman_value = _roman_to_int(ascii_candidate)
            if roman_value is not None:
                return (1, (roman_value, 1))

    if len(token) != 1:
        return (0, token)

    category = next(iter(categories))
    if category not in {"Nl", "No"}:
        return (0, token)

    try:
        numeric_value = unicodedata.numeric(token)
    except (TypeError, ValueError):
        return (0, token)

    if category == "Nl":
        ascii_candidate = unicodedata.normalize("NFKC", token)
        if ascii_candidate and _ROMAN_NUMERAL_PATTERN.match(ascii_candidate.upper()):
            roman_value = _roman_to_int(ascii_candidate)
            if roman_value is not None:
                return (1, (roman_value, 1))

    if float(numeric_value).is_integer():
        return (1, (int(numeric_value), 0))
    return (0, token)


@lru_cache(maxsize=None)
def _sorting_key(name: str) -> tuple[tuple[int, object], ...]:
    normalized = unicodedata.normalize("NFKD", name).translate(_DIACRITIC_OVERRIDES)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    compatibility = unicodedata.normalize("NFKC", without_marks)
    ascii_equivalent = compatibility.encode("ascii", "ignore").decode("ascii")
    if ascii_equivalent:
        base = ascii_equivalent
    elif compatibility:
        base = compatibility
    else:
        base = unicodedata.normalize("NFKC", name)
    key = base.casefold()

    tokens = _tokenize_sort_key(key)
    result: list[tuple[int, object]] = []
    for token in tokens:
        if token.isdigit():
            number = int(token.lstrip("0") or "0")
            result.append((1, (number, 0)))
            continue

        roman_value = _roman_to_int(token)
        if roman_value is not None:
            result.append((1, (roman_value, 1)))
            continue

        coerced = _coerce_unicode_number(token)
        result.append(coerced)
    return tuple(result)


def _to_text(value: bytes | str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8").strip()
        except UnicodeDecodeError:
            return base64.b64encode(value).decode("ascii")
    return value.strip()


def decode_key_material(value: bytes | str) -> bytes:
    """Dekoduje materiał klucza zakodowany w base64/hex/UTF-8."""

    text = _to_text(value)
    if not text:
        raise ValueError("pusty materiał klucza")

    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        pass

    try:
        return bytes.fromhex(text)
    except ValueError:
        pass

    return text.encode("utf-8")


def _load_ed25519_public_key(payload: bytes) -> ed25519.Ed25519PublicKey:
    try:
        return ed25519.Ed25519PublicKey.from_public_bytes(payload)
    except ValueError:
        try:
            public_key = serialization.load_pem_public_key(payload)
        except ValueError as exc:  # pragma: no cover - defensywne logowanie
            raise ValueError("niepoprawny klucz publiczny ed25519") from exc
        if not isinstance(public_key, ed25519.Ed25519PublicKey):
            raise ValueError("klucz publiczny nie jest typu Ed25519")
        return public_key


def _load_ed25519_private_key(data: bytes) -> ed25519.Ed25519PrivateKey:
    try:
        return ed25519.Ed25519PrivateKey.from_private_bytes(data)
    except ValueError:
        try:
            key = serialization.load_pem_private_key(data, password=None)
        except ValueError as exc:  # pragma: no cover - defensywne logowanie
            raise ValueError("niepoprawny klucz prywatny Ed25519") from exc
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise ValueError("klucz prywatny nie jest typu Ed25519")
        return key


def verify_preset_signature(
    payload: Mapping[str, Any],
    signature_doc: Mapping[str, Any] | None,
    *,
    signing_keys: Mapping[str, bytes | str] | None = None,
) -> tuple[PresetSignatureVerification, PresetSignature | None]:
    """Weryfikuje podpis i zwraca wynik wraz z obiektem podpisu."""

    if not signature_doc:
        return PresetSignatureVerification(False, ("missing-signature",)), None

    algorithm = str(signature_doc.get("algorithm") or "").strip().lower()
    key_id = signature_doc.get("key_id")
    key_id_text = str(key_id).strip() if key_id not in (None, "") else None

    signature_value = signature_doc.get("value") or signature_doc.get("signature")
    if not isinstance(signature_value, str):
        return PresetSignatureVerification(False, ("signature-missing",), algorithm or None, key_id_text), None

    if algorithm in {"", "ed25519"}:
        algorithm = _DEFAULT_SIGNATURE_ALGORITHM
        try:
            signature_bytes = decode_key_material(signature_value)
        except ValueError as exc:
            return PresetSignatureVerification(
                False,
                (f"signature-invalid:{exc}",),
                algorithm,
                key_id_text,
            ), None

        if len(signature_bytes) != 64:
            return PresetSignatureVerification(
                False,
                ("signature-invalid-length",),
                algorithm,
                key_id_text,
            ), None

        public_key_bytes: bytes | None = None
        raw_public_key = signature_doc.get("public_key")
        if raw_public_key not in (None, ""):
            try:
                public_key_bytes = decode_key_material(raw_public_key)
            except ValueError as exc:
                return PresetSignatureVerification(
                    False,
                    (f"public-key-invalid:{exc}",),
                    algorithm,
                    key_id_text,
                ), None
        elif signing_keys and key_id_text:
            candidate = signing_keys.get(key_id_text)
            if candidate is not None:
                try:
                    public_key_bytes = decode_key_material(candidate)
                except ValueError as exc:
                    return PresetSignatureVerification(
                        False,
                        (f"signing-key-invalid:{exc}",),
                        algorithm,
                        key_id_text,
                    ), None

        if public_key_bytes is None:
            return PresetSignatureVerification(
                False,
                ("missing-public-key",),
                algorithm,
                key_id_text,
            ), None

        try:
            public_key = _load_ed25519_public_key(public_key_bytes)
        except ValueError as exc:
            return PresetSignatureVerification(
                False,
                (f"public-key-invalid:{exc}",),
                algorithm,
                key_id_text,
            ), None

        try:
            public_key.verify(signature_bytes, canonical_preset_bytes(payload))
        except InvalidSignature:
            return PresetSignatureVerification(
                False,
                ("signature-mismatch",),
                algorithm,
                key_id_text,
            ), None

        signature = PresetSignature(
            algorithm=algorithm,
            value=base64.b64encode(signature_bytes).decode("ascii"),
            key_id=key_id_text,
            public_key=base64.b64encode(public_key_bytes).decode("ascii"),
            signed_at=str(signature_doc.get("signed_at") or "").strip() or None,
            issuer=str(signature_doc.get("issuer") or "").strip() or None,
        )
        return PresetSignatureVerification(True, (), algorithm, key_id_text), signature

    if algorithm in {"hmac-sha256", "hmac_sha256"}:
        key_bytes: bytes | None = None
        if signing_keys and key_id_text:
            candidate = signing_keys.get(key_id_text)
            if candidate is not None:
                try:
                    key_bytes = decode_key_material(candidate)
                except ValueError:
                    key_bytes = None
        verified = verify_hmac_signature(payload, signature_doc, key=key_bytes, algorithm="HMAC-SHA256")
        signature = PresetSignature(
            algorithm="HMAC-SHA256",
            value=str(signature_value),
            key_id=key_id_text,
        )
        issues: tuple[str, ...] = () if verified else ("signature-mismatch",)
        return PresetSignatureVerification(verified, issues, "HMAC-SHA256", key_id_text), signature

    return PresetSignatureVerification(
        False,
        (f"unsupported-algorithm:{algorithm}",),
        algorithm,
        key_id_text,
    ), None


def _normalize_payload(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    if "preset" in document:
        payload = document.get("preset")
        signature_doc = document.get("signature")
    else:
        payload = document
        signature_doc = None
    if not isinstance(payload, Mapping):
        raise ValueError("Preset musi być obiektem JSON/YAML")
    normalized_payload: MutableMapping[str, Any] = {}
    for key, value in payload.items():
        normalized_payload[str(key)] = value
    return dict(normalized_payload), signature_doc if isinstance(signature_doc, Mapping) else None


def parse_preset_document(
    data: bytes | str,
    *,
    source: Path | None = None,
    signing_keys: Mapping[str, bytes | str] | None = None,
) -> PresetDocument:
    """Parsuje dokument presetu z JSON lub YAML."""

    raw_text = data.decode("utf-8") if isinstance(data, bytes) else data
    document: Mapping[str, Any] | None = None
    fmt: str = "json"
    try:
        document = json.loads(raw_text)
        fmt = "json"
    except json.JSONDecodeError:
        try:
            document = yaml.safe_load(raw_text)
            fmt = "yaml"
        except yaml.YAMLError as exc:
            raise ValueError(f"Nie udało się wczytać presetu {source or '<memory>'}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Dokument presetu musi być obiektem JSON/YAML")

    payload, signature_doc = _normalize_payload(document)
    verification, signature = verify_preset_signature(payload, signature_doc, signing_keys=signing_keys)
    issues: list[str] = list(verification.issues)
    if not payload.get("metadata"):
        issues.append("missing-metadata")

    preset = PresetDocument(
        payload=payload,
        signature=signature,
        verification=verification,
        fmt=fmt,
        path=source,
        issues=tuple(dict.fromkeys(issues)),
    )
    return preset


def serialize_preset_document(document: PresetDocument, *, format: str = "json") -> bytes:
    """Serializuje dokument presetu do formatu JSON lub YAML."""

    payload = {"preset": document.payload}
    if document.signature is not None:
        payload["signature"] = document.signature.as_dict()

    fmt = format.lower().strip()
    if fmt == "json":
        return (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    if fmt in {"yaml", "yml"}:
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8")
    raise ValueError(f"Nieobsługiwany format eksportu: {format}")


def sign_preset_payload(
    payload: Mapping[str, Any],
    *,
    private_key: ed25519.Ed25519PrivateKey,
    key_id: str,
    issuer: str | None = None,
    include_public_key: bool = True,
    signed_at: datetime | None = None,
) -> PresetSignature:
    """Buduje podpis Ed25519 dla payloadu presetu."""

    canonical = canonical_preset_bytes(payload)
    signature_bytes = private_key.sign(canonical)
    public_key_bytes = (
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        if include_public_key
        else None
    )
    timestamp = signed_at or datetime.now(timezone.utc)
    return PresetSignature(
        algorithm=_DEFAULT_SIGNATURE_ALGORITHM,
        value=base64.b64encode(signature_bytes).decode("ascii"),
        key_id=str(key_id).strip() or None,
        public_key=(
            base64.b64encode(public_key_bytes).decode("ascii") if public_key_bytes else None
        ),
        signed_at=timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        issuer=issuer.strip() if isinstance(issuer, str) and issuer.strip() else None,
    )


def _sanitize_filename(preset_id: str) -> str:
    sanitized = [ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in preset_id]
    collapsed = "".join(sanitized).strip("-")
    return collapsed or "preset"


class PresetRepository:
    """Zarządza lokalnym repozytorium presetów Marketplace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path_for(self, preset_id: str) -> Path:
        return self.root / f"{_sanitize_filename(preset_id)}.json"

    def load_all(
        self,
        *,
        signing_keys: Mapping[str, bytes | str] | None = None,
    ) -> tuple[PresetDocument, ...]:
        documents: list[PresetDocument] = []
        if not self.root.exists():
            return tuple()
        files = sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_file() and _normalize_format(path.name)
            ),
            key=lambda path: (_sorting_key(path.name), path.name.casefold(), path.name),
        )
        for path in files:
            try:
                doc = parse_preset_document(path.read_bytes(), source=path, signing_keys=signing_keys)
            except Exception:
                LOGGER.exception("Nie udało się wczytać presetu %s", path)
                continue
            documents.append(doc)
        return tuple(documents)

    def import_payload(
        self,
        payload: bytes | str,
        *,
        filename: str | None = None,
        signing_keys: Mapping[str, bytes | str] | None = None,
        require_signature: bool = True,
    ) -> PresetDocument:
        document = parse_preset_document(payload, source=None, signing_keys=signing_keys)
        if require_signature and (document.signature is None or not document.verification.verified):
            raise ValueError("Preset musi zawierać zweryfikowany podpis.")
        preset_id = document.preset_id
        if not preset_id:
            raise ValueError("Preset nie zawiera identyfikatora metadata.id")

        target_path = self._path_for(preset_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        existing_version: str | None = None
        if target_path.exists():
            existing_doc = parse_preset_document(
                target_path.read_bytes(),
                source=target_path,
                signing_keys=signing_keys,
            )
            existing_version = existing_doc.version
            if document.version and existing_version and document.version == existing_version:
                raise ValueError(
                    f"Preset {preset_id} w wersji {existing_version} jest już zainstalowany."
                )

        fmt = _normalize_format(filename) or document.fmt
        serialized = serialize_preset_document(document, format=fmt)
        target_path.write_bytes(serialized)
        document.path = target_path
        return document

    def export_preset(
        self,
        preset_id: str,
        *,
        format: str = "json",
        signing_keys: Mapping[str, bytes | str] | None = None,
    ) -> tuple[PresetDocument, bytes]:
        path = self._path_for(preset_id)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono presetu {preset_id}")
        document = parse_preset_document(path.read_bytes(), source=path, signing_keys=signing_keys)
        if format.lower().strip() != document.fmt:
            # Wygeneruj tymczasowy dokument z nowym formatem (bez zapisu na dysk)
            document = PresetDocument(
                payload=document.payload,
                signature=document.signature,
                verification=document.verification,
                fmt=format.lower().strip(),
                path=document.path,
                issues=document.issues,
            )
        document_blob = serialize_preset_document(document, format=format)
        # aktualizacja path nie jest konieczna, ale zachowujemy spójność
        exported = PresetDocument(
            payload=document.payload,
            signature=document.signature,
            verification=document.verification,
            fmt=format.lower().strip(),
            path=path,
            issues=document.issues,
        )
        return exported, document_blob

    def remove(self, preset_id: str) -> bool:
        path = self._path_for(preset_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False


def load_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    data = path.read_bytes()
    try:
        material = decode_key_material(data)
    except ValueError:
        material = data
    return _load_ed25519_private_key(material)


__all__ = [
    "PresetDocument",
    "PresetRepository",
    "PresetSignature",
    "PresetSignatureVerification",
    "canonical_preset_bytes",
    "decode_key_material",
    "parse_preset_document",
    "serialize_preset_document",
    "sign_preset_payload",
    "verify_preset_signature",
    "load_private_key",
]
