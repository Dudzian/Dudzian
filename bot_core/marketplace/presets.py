from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

try:  # opcjonalna zależność
    import yaml
except ImportError:  # pragma: no cover - środowiska bez PyYAML
    yaml = None  # type: ignore[assignment]
from bot_core.marketplace.models import (
    PresetDocument,
    PresetSignature,
    PresetSignatureVerification,
    canonical_preset_bytes,
    normalize_preset_document,
)
from bot_core.marketplace.signatures import (
    DEFAULT_SIGNATURE_ALGORITHM,
    decode_key_material,
    sign_preset_payload,
    verify_preset_signature,
)

if TYPE_CHECKING:
    from .signatures import SignatureProvider

LOGGER = logging.getLogger(__name__)


def _require_yaml():
    if yaml is None:
        raise RuntimeError(
            "PyYAML nie jest zainstalowany. Zainstaluj pakiet 'pyyaml' aby wczytać lub zapisać preset YAML."
        )

    return yaml


_SUPPORTED_FORMATS = {"json", "yaml", "yml"}
_DIACRITIC_OVERRIDES = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "đ": "d",
        "Đ": "D",
        "Ø": "O",
        "ø": "o",
    }
)
_TOKEN_SPLITTER = re.compile(r"(\d+|(?<![A-Za-z])[IVXLCDM]+(?![A-Za-z]))", re.IGNORECASE)
_ROMAN_NUMERAL_PATTERN = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
_DEFAULT_SIGNATURE_ALGORITHM = DEFAULT_SIGNATURE_ALGORITHM


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


def parse_preset_document(
    data: bytes | str,
    *,
    source: Path | None = None,
    signing_keys: Mapping[str, bytes | str] | None = None,
    providers: tuple["SignatureProvider", ...] | None = None,
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
            _yaml = _require_yaml()
            document = _yaml.safe_load(raw_text)
            fmt = "yaml"
        except _yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ValueError(
                f"Nie udało się wczytać presetu {source or '<memory>'}: {exc}"
            ) from exc
    if not isinstance(document, Mapping):
        raise ValueError("Dokument presetu musi być obiektem JSON/YAML")

    payload, signature_doc = normalize_preset_document(document)
    verification, signature = verify_preset_signature(
        payload,
        signature_doc,
        signing_keys=signing_keys,
        providers=providers,
    )
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


def serialize_preset_document(
    document: PresetDocument, *, format: str = "json", ensure_ascii: bool = False
) -> bytes:
    """Serializuje dokument presetu do formatu JSON lub YAML.

    Parametr ``ensure_ascii`` pozwala wymusić reprezentację U+XXXX, co
    stabilizuje podpisy/digest przy ponownym generowaniu artefaktów
    Marketplace.
    """

    payload = {"preset": document.payload}
    if document.signature is not None:
        payload["signature"] = document.signature.as_dict()

    fmt = format.lower().strip()
    if fmt == "json":
        return (
            json.dumps(payload, ensure_ascii=ensure_ascii, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    if fmt in {"yaml", "yml"}:
        _yaml = _require_yaml()
        return _yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8")
    raise ValueError(f"Nieobsługiwany format eksportu: {format}")


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
                doc = parse_preset_document(
                    path.read_bytes(), source=path, signing_keys=signing_keys
                )
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


__all__ = [
    "PresetDocument",
    "PresetRepository",
    "PresetSignature",
    "PresetSignatureVerification",
    "canonical_preset_bytes",
    "parse_preset_document",
    "serialize_preset_document",
    "sign_preset_payload",
    "verify_preset_signature",
]
