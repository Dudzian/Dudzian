"""Runtime verification of bundled resource integrity manifests."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


class RuntimeIntegrityError(RuntimeError):
    """Raised when runtime resource integrity verification fails."""


@dataclass(slots=True)
class IntegrityIssue:
    path: str
    expected: str
    actual: str | None


def _resolve_candidate_roots(bundle_root: Path | None) -> Sequence[Path]:
    roots: list[Path] = []
    if bundle_root is not None:
        roots.append(bundle_root.expanduser().resolve())
    env_root = os.environ.get("BOT_CORE_BUNDLE_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    module_root = Path(__file__).resolve()
    for parent in module_root.parents:
        roots.append(parent)
    seen: set[Path] = set()
    result: list[Path] = []
    for candidate in roots:
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _hash_file(path: Path, algorithm: str) -> str:
    import hashlib

    normalized = algorithm.strip().lower()
    try:
        hasher = hashlib.new(normalized)
    except ValueError as exc:  # pragma: no cover - defensive path
        raise RuntimeIntegrityError(f"Nieobsługiwany algorytm skrótu: {algorithm}") from exc
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(131072), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeIntegrityError(f"Nie udało się odczytać manifestu integralności: {exc}") from exc
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeIntegrityError("Manifest integralności ma niepoprawny format JSON.") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeIntegrityError("Manifest integralności musi być obiektem JSON.")
    return manifest


def _manifest_entries(manifest: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    entries = manifest.get("files")
    if not isinstance(entries, Sequence):
        raise RuntimeIntegrityError("Manifest integralności wymaga listy plików w polu 'files'.")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeIntegrityError("Każdy wpis manifestu integralności musi być słownikiem.")
        yield entry


def verify_bundle_integrity(
    bundle_root: Path | None = None,
    *,
    manifest_relpath: str = "resources/integrity_manifest.json",
    strict: bool | None = None,
) -> bool:
    """Verify hashes of bundle resources using integrity manifest.

    Args:
        bundle_root: Optional override bundle root. When ``None`` search heuristics are used.
        manifest_relpath: Relative path of manifest inside bundle root.
        strict: When ``True`` missing manifest or mismatches raise :class:`RuntimeIntegrityError`.

    Returns:
        ``True`` when verification succeeded, ``False`` otherwise.
    """

    if strict is None:
        strict = os.environ.get("BOT_CORE_INTEGRITY_STRICT") in {"1", "true", "TRUE", "yes", "on"}

    for root in _resolve_candidate_roots(bundle_root):
        candidate = root / manifest_relpath
        if not candidate.exists():
            continue
        manifest = _load_manifest(candidate)
        algorithm = str(manifest.get("algorithm") or "sha384")
        issues: list[IntegrityIssue] = []
        for entry in _manifest_entries(manifest):
            path_value = entry.get("path")
            expected = entry.get("sha384") or entry.get("digest")
            if not isinstance(path_value, str) or not path_value:
                raise RuntimeIntegrityError("Każdy wpis manifestu musi zawierać klucz 'path'.")
            if not isinstance(expected, str) or not expected:
                raise RuntimeIntegrityError("Wpis manifestu wymaga skrótu w polu 'sha384'.")
            target = root / path_value
            if not target.exists():
                issues.append(IntegrityIssue(path=path_value, expected=expected, actual=None))
                continue
            actual = _hash_file(target, algorithm)
            if actual.lower() != expected.lower():
                issues.append(IntegrityIssue(path=path_value, expected=expected, actual=actual))

        if issues:
            message = "Wykryto naruszenie integralności zasobów:" + "\n" + "\n".join(
                f"- {issue.path}: expected={issue.expected} actual={issue.actual or 'missing'}"
                for issue in issues
            )
            if strict:
                raise RuntimeIntegrityError(message)
            LOGGER.error(message)
            return False

        generated_at = manifest.get("generated_at")
        if generated_at:
            try:
                datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
            except Exception:  # pragma: no cover - metadane pomocnicze
                LOGGER.debug("Manifest integralności zawiera niepoprawne pole generated_at: %s", generated_at)

        LOGGER.info("Integralność zasobów zweryfikowana dla katalogu: %s", root)
        return True

    if strict:
        raise RuntimeIntegrityError(
            f"Nie znaleziono manifestu integralności ({manifest_relpath}) dla żadnego kandydata root."
        )
    LOGGER.debug(
        "Pominięto weryfikację integralności – manifest %s nie został znaleziony.",
        manifest_relpath,
    )
    return False


__all__ = ["RuntimeIntegrityError", "verify_bundle_integrity"]

