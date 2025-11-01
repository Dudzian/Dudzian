"""Utilities for generating signed marketplace presets per exchange configuration."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.marketplace.presets import (
    PresetDocument,
    canonical_preset_bytes,
    parse_preset_document,
    sign_preset_payload,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExchangeEnvironment:
    name: str
    description: str | None
    payload: Mapping[str, object]


@dataclass(slots=True)
class ExchangePresetSpec:
    exchange_id: str
    file_path: Path
    environments: Sequence[ExchangeEnvironment]

    @property
    def preset_id(self) -> str:
        return f"exchange_{self.exchange_id.lower()}"

    @property
    def display_name(self) -> str:
        return self.exchange_id.replace("_", " ").title()


@dataclass(slots=True)
class ExchangePresetValidationResult:
    """Raport z walidacji istniejących presetów giełdowych."""

    exchange_id: str
    preset_id: str
    spec_path: Path | None
    preset_path: Path
    exists: bool
    verified: bool
    up_to_date: bool
    current_version: str | None
    expected_version: str
    issues: tuple[str, ...]


def _load_exchange_file(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Nie udało się odczytać pliku giełdy {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Plik giełdy {path} zawiera niepoprawny YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError(f"Plik giełdy {path} musi być słownikiem.")
    return data


def _collect_environments(payload: Mapping[str, object]) -> list[ExchangeEnvironment]:
    environments: list[ExchangeEnvironment] = []
    for name, value in payload.items():
        if name.startswith("#"):
            continue
        if not isinstance(value, Mapping):
            LOGGER.debug("Pomijam sekcję %s – oczekiwano mapy konfiguracji.", name)
            continue
        description = value.get("description")
        desc_text = str(description).strip() if isinstance(description, str) else None
        environments.append(
            ExchangeEnvironment(
                name=str(name),
                description=desc_text,
                payload=dict(value),
            )
        )
    return environments


def _normalize_exchange_selection(
    selected_exchanges: Iterable[str] | None,
) -> tuple[set[str] | None, set[str] | None]:
    if selected_exchanges is None:
        return None, None
    normalized: set[str] = set()
    for value in selected_exchanges:
        text = str(value).strip()
        if not text:
            continue
        normalized.add(text.upper())
    if not normalized:
        return set(), set()
    preset_stems = {f"exchange_{value.lower()}" for value in normalized}
    return normalized, preset_stems


def _normalize_version_strategy(value: str | None) -> str:
    if not value:
        return "static"
    normalized = value.replace("-", "_").strip().lower()
    if normalized in {"static", "hash", "spec_hash"}:
        return "spec_hash" if normalized in {"hash", "spec_hash"} else "static"
    raise ValueError(f"Nieobsługiwana strategia wersjonowania presetów: {value}")


def _fingerprint_exchange_spec(spec: ExchangePresetSpec) -> str:
    payload = {
        "exchange_id": spec.exchange_id,
        "environments": [
            {
                "name": environment.name,
                "description": environment.description,
                "payload": environment.payload,
            }
            for environment in spec.environments
        ],
    }
    canonical = canonical_preset_bytes(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    return digest[:16]


def _resolve_preset_version(
    spec: ExchangePresetSpec,
    *,
    base_version: str,
    strategy: str,
) -> str:
    normalized = _normalize_version_strategy(strategy)
    if normalized == "static":
        return base_version
    fingerprint = _fingerprint_exchange_spec(spec)
    base = base_version.split("+", 1)[0]
    return f"{base}+{fingerprint}"


def _expected_version_for_spec(
    spec: ExchangePresetSpec,
    *,
    strategy: str,
    explicit_version: str | None,
    current_version: str | None,
) -> str:
    normalized = _normalize_version_strategy(strategy)
    if normalized == "static":
        if explicit_version:
            return explicit_version
        if current_version:
            return current_version
        return "1.0.0"

    base_version = explicit_version
    if not base_version:
        if current_version:
            base_version = current_version.split("+", 1)[0]
        else:
            base_version = "1.0.0"
    return _resolve_preset_version(
        spec,
        base_version=base_version,
        strategy=normalized,
    )


def load_exchange_specs(
    exchanges_dir: Path,
    *,
    selected_exchanges: Iterable[str] | None = None,
) -> Sequence[ExchangePresetSpec]:
    specs: list[ExchangePresetSpec] = []
    selected_ids, _ = _normalize_exchange_selection(selected_exchanges)
    for path in sorted(exchanges_dir.glob("*.y*ml")):
        payload = _load_exchange_file(path)
        environments = _collect_environments(payload)
        if not environments:
            LOGGER.warning("Plik giełdy %s nie zawiera żadnych środowisk – pomijam.", path)
            continue
        exchange_id = path.stem.upper()
        if selected_ids is not None and exchange_id not in selected_ids:
            continue
        specs.append(
            ExchangePresetSpec(
                exchange_id=exchange_id,
                file_path=path,
                environments=environments,
            )
        )
    return specs


def _build_preset_payload(spec: ExchangePresetSpec, *, version: str) -> Mapping[str, object]:
    default_environment = spec.environments[0]
    environment_payloads: list[Mapping[str, object]] = []
    for environment in spec.environments:
        environment_payloads.append(
            {
                "name": environment.name,
                "description": environment.description,
                "config": environment.payload,
            }
        )

    try:
        relative_source = spec.file_path.relative_to(Path.cwd())
    except ValueError:
        relative_source = spec.file_path

    metadata: MutableMapping[str, object] = {
        "id": spec.preset_id,
        "version": version,
        "summary": f"Autogenerated preset dla giełdy {spec.display_name}",
        "tags": ["exchange", spec.exchange_id.lower()],
        "profile": "grid",
        "required_exchanges": [spec.exchange_id],
        "source_file": relative_source.as_posix(),
        "environments": [env.name for env in spec.environments],
    }

    payload: dict[str, object] = {
        "name": f"{spec.display_name} Connectivity",
        "strategies": [
            {
                "name": f"{spec.exchange_id.lower()}_connectivity",
                "engine": "grid_trading",
                "parameters": {
                    "exchange": spec.exchange_id,
                    "default_environment": default_environment.name,
                    "available_environments": [env.name for env in spec.environments],
                },
                "license_tier": "community",
                "risk_classes": ["infrastructure"],
                "required_data": ["ohlcv"],
                "tags": ["exchange", spec.exchange_id.lower()],
                "metadata": {
                    "description": (
                        f"Automatycznie wygenerowany preset środowisk dla giełdy {spec.display_name}."
                    ),
                    "environments": environment_payloads,
                },
            }
        ],
        "metadata": metadata,
    }

    return payload


def _load_private_key(material: bytes | str | Path) -> ed25519.Ed25519PrivateKey:
    if isinstance(material, ed25519.Ed25519PrivateKey):  # type: ignore[unreachable]
        return material
    if isinstance(material, (bytes, bytearray)):
        data = bytes(material)
    else:
        path = Path(material).expanduser()
        data = path.read_bytes()
    try:
        key = ed25519.Ed25519PrivateKey.from_private_bytes(data)
    except ValueError:
        key = serialization.load_pem_private_key(data, password=None)
        if not isinstance(key, ed25519.Ed25519PrivateKey):
            raise ValueError("Klucz prywatny musi być typu Ed25519")
    return key


def _emit_signed_preset(
    spec: ExchangePresetSpec,
    *,
    version: str,
    private_key: ed25519.Ed25519PrivateKey,
    key_id: str,
    issuer: str | None,
    output_dir: Path,
) -> PresetDocument:
    payload = _build_preset_payload(spec, version=version)
    signature = sign_preset_payload(
        payload,
        private_key=private_key,
        key_id=key_id,
        issuer=issuer,
    )
    document = {
        "preset": payload,
        "signature": signature.as_dict(),
    }
    target_path = output_dir / f"{spec.preset_id}.json"
    serialized = json.dumps(document, ensure_ascii=False, indent=2)
    target_path.write_text(serialized, encoding="utf-8")
    parsed = parse_preset_document(serialized.encode("utf-8"), source=target_path)
    return parsed


def generate_exchange_presets(
    *,
    exchanges_dir: Path,
    output_dir: Path,
    private_key: ed25519.Ed25519PrivateKey | bytes | str | Path,
    key_id: str,
    issuer: str | None = None,
    version: str | None = None,
    version_strategy: str = "static",
    selected_exchanges: Iterable[str] | None = None,
) -> Sequence[PresetDocument]:
    """Generate signed presets for every exchange configuration file.

    When ``version_strategy`` is set to ``"spec-hash"`` each preset receives a
    deterministic build metadata suffix derived from the exchange definition to
    simplify drift detection.
    """

    resolved_key = _load_private_key(private_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = load_exchange_specs(
        exchanges_dir,
        selected_exchanges=selected_exchanges,
    )
    documents: list[PresetDocument] = []
    base_version = version or "1.0.0"

    for spec in specs:
        preset_version = _resolve_preset_version(
            spec,
            base_version=base_version,
            strategy=version_strategy,
        )
        parsed = _emit_signed_preset(
            spec,
            version=preset_version,
            private_key=resolved_key,
            key_id=key_id,
            issuer=issuer,
            output_dir=output_dir,
        )
        documents.append(parsed)
    return documents


def reconcile_exchange_presets(
    *,
    exchanges_dir: Path,
    output_dir: Path,
    private_key: ed25519.Ed25519PrivateKey | bytes | str | Path,
    key_id: str,
    issuer: str | None = None,
    version: str | None = None,
    signing_keys: Mapping[str, bytes | str] | None = None,
    remove_orphans: bool = False,
    selected_exchanges: Iterable[str] | None = None,
    version_strategy: str = "static",
) -> Sequence[ExchangePresetValidationResult]:
    """Naprawia brakujące lub przestarzałe presety, opcjonalnie usuwając osierocone pliki."""

    initial_results = list(
        validate_exchange_presets(
            exchanges_dir=exchanges_dir,
            output_dir=output_dir,
            version=version,
            signing_keys=signing_keys,
            selected_exchanges=selected_exchanges,
            version_strategy=version_strategy,
        )
    )

    to_regenerate = [
        result
        for result in initial_results
        if result.spec_path is not None
        and (not result.exists or not result.verified or not result.up_to_date)
    ]
    orphans = [result for result in initial_results if result.spec_path is None]

    if to_regenerate:
        resolved_key = _load_private_key(private_key)
        specs = {
            spec.exchange_id: spec
            for spec in load_exchange_specs(
                exchanges_dir,
                selected_exchanges=selected_exchanges,
            )
        }
        for result in to_regenerate:
            spec = specs.get(result.exchange_id)
            if spec is None:
                LOGGER.warning(
                    "Nie znaleziono definicji giełdy %s podczas regeneracji presetu.",
                    result.exchange_id,
                )
                continue
            preset_version = result.expected_version or version or "1.0.0"
            _emit_signed_preset(
                spec,
                version=preset_version,
                private_key=resolved_key,
                key_id=key_id,
                issuer=issuer,
                output_dir=output_dir,
            )

    if remove_orphans and orphans:
        for orphan in orphans:
            try:
                orphan.preset_path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - defensywne logowanie
                LOGGER.warning(
                    "Nie udało się usunąć osieroconego presetu %s: %s",
                    orphan.preset_path,
                    exc,
                )

    return validate_exchange_presets(
        exchanges_dir=exchanges_dir,
        output_dir=output_dir,
        version=version,
        signing_keys=signing_keys,
        selected_exchanges=selected_exchanges,
        version_strategy=version_strategy,
    )


def validate_exchange_presets(
    *,
    exchanges_dir: Path,
    output_dir: Path,
    version: str | None = None,
    signing_keys: Mapping[str, bytes | str] | None = None,
    selected_exchanges: Iterable[str] | None = None,
    version_strategy: str = "static",
) -> Sequence[ExchangePresetValidationResult]:
    """Porównuje wygenerowane presety z aktualnymi plikami na dysku."""

    selected_ids, selected_stems = _normalize_exchange_selection(selected_exchanges)
    results: list[ExchangePresetValidationResult] = []
    specs = load_exchange_specs(
        exchanges_dir,
        selected_exchanges=selected_ids,
    )
    expected_paths: set[Path] = set()
    for spec in specs:
        preset_path = output_dir / f"{spec.preset_id}.json"
        expected_paths.add(preset_path)
        issues: list[str] = []
        exists = preset_path.exists()
        verified = False
        up_to_date = False
        current_version: str | None = None

        document = None
        if exists:
            try:
                document = parse_preset_document(
                    preset_path.read_bytes(),
                    source=preset_path,
                    signing_keys=signing_keys,
                )
            except Exception as exc:  # pragma: no cover - logowanie w miejscu wywołania
                issues.append(f"parse-error:{exc}")
            else:
                if document.issues:
                    issues.extend(document.issues)
                verified = document.verification.verified
                if not verified:
                    issues.extend(document.verification.issues or ("signature-invalid",))
                current_version = document.version
        else:
            issues.append("missing-file")

        expected_version = _expected_version_for_spec(
            spec,
            strategy=version_strategy,
            explicit_version=version,
            current_version=current_version,
        )

        if exists and document is not None:
            try:
                expected_payload = _build_preset_payload(spec, version=expected_version)
                actual_canonical = json.dumps(
                    document.payload, sort_keys=True, separators=(",", ":")
                )
                expected_canonical = json.dumps(
                    expected_payload, sort_keys=True, separators=(",", ":")
                )
                if actual_canonical == expected_canonical:
                    up_to_date = True
                else:
                    issues.append("payload-mismatch")
            except Exception as exc:  # pragma: no cover - błędy generowania payloadu
                issues.append(f"payload-error:{exc}")

        deduped_issues = tuple(dict.fromkeys(issues))
        results.append(
            ExchangePresetValidationResult(
                exchange_id=spec.exchange_id,
                preset_id=spec.preset_id,
                spec_path=spec.file_path,
                preset_path=preset_path,
                exists=exists,
                verified=verified,
                up_to_date=up_to_date,
                current_version=current_version,
                expected_version=expected_version,
                issues=deduped_issues,
            )
        )

    observed_paths = {path for path in output_dir.glob("*.json")}
    if selected_stems is not None:
        observed_paths = {
            path
            for path in observed_paths
            if any(path.stem.lower().startswith(prefix) for prefix in selected_stems)
        }
    orphan_paths = sorted(observed_paths - expected_paths)
    for path in orphan_paths:
        issues = ["orphan-file"]
        verified = False
        current_version: str | None = None
        expected_version = version or "1.0.0"
        exchange_id = path.stem.replace("exchange_", "").upper() or path.stem.upper()
        preset_id = path.stem
        document = None
        try:
            document = parse_preset_document(
                path.read_bytes(),
                source=path,
                signing_keys=signing_keys,
            )
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            issues.append(f"parse-error:{exc}")
        else:
            if document.issues:
                issues.extend(document.issues)
            verified = document.verification.verified
            if not verified:
                issues.extend(document.verification.issues or ("signature-invalid",))
            current_version = document.version
            preset_id = document.preset_id or preset_id
            metadata = document.metadata
            required = metadata.get("required_exchanges") if isinstance(metadata, Mapping) else None
            if isinstance(required, Sequence) and required:
                exchange_id = str(required[0]).upper()
            expected_version = version or current_version or expected_version

        deduped_issues = tuple(dict.fromkeys(issues))
        results.append(
            ExchangePresetValidationResult(
                exchange_id=exchange_id,
                preset_id=preset_id or path.stem,
                spec_path=None,
                preset_path=path,
                exists=True,
                verified=verified,
                up_to_date=False,
                current_version=current_version,
                expected_version=expected_version,
                issues=deduped_issues,
            )
        )

    return tuple(results)


__all__ = [
    "ExchangePresetSpec",
    "ExchangePresetValidationResult",
    "generate_exchange_presets",
    "reconcile_exchange_presets",
    "load_exchange_specs",
    "validate_exchange_presets",
]

