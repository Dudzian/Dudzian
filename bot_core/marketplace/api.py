"""High level Marketplace API primitives (dependencies, updates, assignments)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from packaging.version import InvalidVersion, Version

from .presets import PresetDocument


def _normalize_identifier(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    raise ValueError("wymagany identyfikator presetu")


def _optional_str(value: object | None) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return False


def _coerce_version(value: object | None) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _parse_version_constraint(value: str) -> tuple[str, ...]:
    constraints: list[str] = []
    for part in value.split(","):
        text = part.strip()
        if text:
            constraints.append(text)
    return tuple(constraints)


def _compare_versions(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None
    try:
        left_version = Version(left)
        right_version = Version(right)
    except InvalidVersion:
        return None
    if left_version < right_version:
        return -1
    if left_version > right_version:
        return 1
    return 0


def _satisfies_constraint(version: str | None, constraints: Sequence[str]) -> bool | None:
    if not constraints:
        return True
    if version is None:
        return None
    try:
        current = Version(version)
    except InvalidVersion:
        return None

    for constraint in constraints:
        if constraint.startswith(">="):
            try:
                if current < Version(constraint[2:].strip()):
                    return False
            except InvalidVersion:
                return None
        elif constraint.startswith("<="):
            try:
                if current > Version(constraint[2:].strip()):
                    return False
            except InvalidVersion:
                return None
        elif constraint.startswith(">"):
            try:
                if current <= Version(constraint[1:].strip()):
                    return False
            except InvalidVersion:
                return None
        elif constraint.startswith("<"):
            try:
                if current >= Version(constraint[1:].strip()):
                    return False
            except InvalidVersion:
                return None
        elif constraint.startswith("=="):
            try:
                if current != Version(constraint[2:].strip()):
                    return False
            except InvalidVersion:
                return None
        elif constraint.startswith("~="):
            try:
                base = Version(constraint[2:].strip())
            except InvalidVersion:
                return None
            upper = Version(f"{base.major}.{base.minor + 1 if base.release and len(base.release) > 1 else base.minor + 1}.0")
            if current < base or current >= upper:
                return False
        else:
            try:
                if current != Version(constraint.strip()):
                    return False
            except InvalidVersion:
                return None
    return True


@dataclass(slots=True, frozen=True)
class PresetDependency:
    """Definition of a dependency between Marketplace presets."""

    preset_id: str
    constraints: tuple[str, ...] = field(default_factory=tuple)
    optional: bool = False
    capability: str | None = None
    notes: str | None = None

    def to_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "presetId": self.preset_id,
        }
        if self.constraints:
            payload["constraints"] = list(self.constraints)
        if self.optional:
            payload["optional"] = True
        if self.capability:
            payload["capability"] = self.capability
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True, frozen=True)
class PresetUpdateChannel:
    """Single update channel (stable/beta/etc.) exposed by a preset."""

    name: str
    version: str | None
    released_at: str | None = None
    severity: str | None = None
    notes: str | None = None

    def to_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {"name": self.name}
        if self.version:
            payload["version"] = self.version
        if self.released_at:
            payload["released_at"] = self.released_at
        if self.severity:
            payload["severity"] = self.severity
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True, frozen=True)
class PresetUpdateDirective:
    """Describes how the current preset supersedes or replaces other packages."""

    replaces: tuple[str, ...] = field(default_factory=tuple)
    requires_approval: bool = False
    channel: str | None = None
    message: str | None = None

    def to_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {}
        if self.replaces:
            payload["replaces"] = list(self.replaces)
        if self.requires_approval:
            payload["requiresApproval"] = True
        if self.channel:
            payload["channel"] = self.channel
        if self.message:
            payload["message"] = self.message
        return payload


@dataclass(slots=True, frozen=True)
class MarketplacePreset:
    """Normalized preset metadata combining signature, dependencies and updates."""

    document: PresetDocument
    dependencies: tuple[PresetDependency, ...]
    update_channels: tuple[PresetUpdateChannel, ...]
    update_directive: PresetUpdateDirective
    preferred_channel: str | None = None

    @property
    def preset_id(self) -> str:
        return self.document.preset_id

    @property
    def version(self) -> str | None:
        return self.document.version

    @property
    def name(self) -> str | None:
        return self.document.name

    @property
    def path(self) -> Path | None:
        return self.document.path

    @property
    def signature_verified(self) -> bool:
        return self.document.verification.verified

    def to_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "presetId": self.preset_id,
            "version": self.version,
            "name": self.name,
            "signatureVerified": self.signature_verified,
            "dependencies": [dep.to_payload() for dep in self.dependencies],
            "updates": [channel.to_payload() for channel in self.update_channels],
            "updateDirective": self.update_directive.to_payload(),
        }
        if self.preferred_channel:
            payload["preferredChannel"] = self.preferred_channel
        return payload


def _parse_dependency(entry: object) -> PresetDependency | None:
    if isinstance(entry, str):
        return PresetDependency(preset_id=_normalize_identifier(entry))
    if isinstance(entry, Mapping):
        try:
            preset_id = _normalize_identifier(entry.get("preset_id") or entry.get("id") or entry.get("name"))
        except ValueError:
            return None
        constraints_value = entry.get("version") or entry.get("constraint") or entry.get("constraints")
        constraints: tuple[str, ...] = ()
        if isinstance(constraints_value, str):
            constraints = _parse_version_constraint(constraints_value)
        elif isinstance(constraints_value, Sequence):
            collected: list[str] = []
            for value in constraints_value:
                if isinstance(value, str) and value.strip():
                    collected.append(value.strip())
            constraints = tuple(collected)
        optional = _normalize_bool(entry.get("optional"))
        capability = _optional_str(entry.get("capability") or entry.get("requires_capability"))
        notes = _optional_str(entry.get("notes") or entry.get("description"))
        return PresetDependency(
            preset_id=preset_id,
            constraints=constraints,
            optional=optional,
            capability=capability,
            notes=notes,
        )
    return None


def _parse_dependencies(metadata: Mapping[str, object]) -> tuple[PresetDependency, ...]:
    dependencies_field = metadata.get("dependencies")
    if dependencies_field is None:
        return tuple()
    items: Iterable[object]
    if isinstance(dependencies_field, Mapping):
        items = dependencies_field.values()
    elif isinstance(dependencies_field, Sequence) and not isinstance(dependencies_field, (str, bytes)):
        items = dependencies_field
    else:
        return tuple()
    result: list[PresetDependency] = []
    for entry in items:
        dependency = _parse_dependency(entry)
        if dependency is not None:
            result.append(dependency)
    return tuple(result)


def _parse_update_channels(metadata: Mapping[str, object]) -> tuple[PresetUpdateChannel, ...]:
    updates_field = metadata.get("updates")
    if updates_field is None:
        return tuple()
    channels_raw: Iterable[Mapping[str, object]]
    if isinstance(updates_field, Mapping):
        if isinstance(updates_field.get("channels"), Sequence):
            channels_raw = [
                channel
                for channel in updates_field["channels"]  # type: ignore[index]
                if isinstance(channel, Mapping)
            ]
        else:
            channels_raw = [updates_field]
    elif isinstance(updates_field, Sequence):
        channels_raw = [channel for channel in updates_field if isinstance(channel, Mapping)]
    else:
        return tuple()

    channels: list[PresetUpdateChannel] = []
    for channel in channels_raw:
        name = _optional_str(channel.get("name") or channel.get("channel")) or "stable"
        channels.append(
            PresetUpdateChannel(
                name=name,
                version=_coerce_version(channel.get("version")),
                released_at=_optional_str(channel.get("released_at") or channel.get("releasedAt")),
                severity=_optional_str(channel.get("severity")),
                notes=_optional_str(channel.get("notes") or channel.get("summary")),
            )
        )
    return tuple(channels)


def _parse_update_directive(metadata: Mapping[str, object]) -> PresetUpdateDirective:
    updates_field = metadata.get("updates")
    if not isinstance(updates_field, Mapping):
        return PresetUpdateDirective()
    replaces_field = updates_field.get("replaces") or updates_field.get("supersedes")
    if isinstance(replaces_field, str):
        replaces = tuple({part.strip() for part in replaces_field.split(",") if part.strip()})
    elif isinstance(replaces_field, Sequence):
        replaces = tuple(
            {
                str(item).strip()
                for item in replaces_field
                if isinstance(item, (str, bytes)) and str(item).strip()
            }
        )
    else:
        replaces = tuple()
    requires_approval = _normalize_bool(updates_field.get("requires_approval") or updates_field.get("manual"))
    channel = _optional_str(updates_field.get("default_channel") or updates_field.get("channel"))
    message = _optional_str(updates_field.get("message") or updates_field.get("notes"))
    return PresetUpdateDirective(
        replaces=replaces,
        requires_approval=requires_approval,
        channel=channel,
        message=message,
    )


def build_marketplace_preset(document: PresetDocument) -> MarketplacePreset:
    metadata_raw = document.metadata
    if not isinstance(metadata_raw, Mapping):
        metadata: Mapping[str, object] = {}
    else:
        metadata = metadata_raw

    dependencies = _parse_dependencies(metadata)
    channels = _parse_update_channels(metadata)
    directive = _parse_update_directive(metadata)

    preferred_channel = directive.channel
    if preferred_channel is None and channels:
        preferred_channel = channels[0].name

    return MarketplacePreset(
        document=document,
        dependencies=dependencies,
        update_channels=channels,
        update_directive=directive,
        preferred_channel=preferred_channel,
    )


@dataclass(slots=True, frozen=True)
class UpdateStep:
    """Represents a single upgrade decision."""

    preset_id: str
    from_version: str | None
    to_version: str | None
    channel: str | None

    def to_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {"presetId": self.preset_id}
        if self.from_version:
            payload["fromVersion"] = self.from_version
        if self.to_version:
            payload["toVersion"] = self.to_version
        if self.channel:
            payload["channel"] = self.channel
        return payload


@dataclass(slots=True, frozen=True)
class MarketplacePlan:
    """Installation plan with dependency ordering and optional upgrade info."""

    install_order: tuple[str, ...]
    required_dependencies: Mapping[str, tuple[PresetDependency, ...]]
    missing: Mapping[str, tuple[str, ...]]
    issues: tuple[str, ...]
    upgrades: tuple[UpdateStep, ...]

    def to_payload(self) -> Mapping[str, object]:
        return {
            "installOrder": list(self.install_order),
            "requiredDependencies": {
                preset_id: [dep.to_payload() for dep in dependencies]
                for preset_id, dependencies in self.required_dependencies.items()
            },
            "missing": {preset_id: list(entries) for preset_id, entries in self.missing.items()},
            "issues": list(self.issues),
            "upgrades": [step.to_payload() for step in self.upgrades],
        }


class MarketplaceIndex:
    """Collection of marketplace presets with dependency resolution helpers."""

    def __init__(self, presets: Mapping[str, MarketplacePreset]) -> None:
        self._presets = dict(presets)

    @classmethod
    def from_documents(cls, documents: Sequence[PresetDocument]) -> "MarketplaceIndex":
        presets: dict[str, MarketplacePreset] = {}
        for document in documents:
            if not document.preset_id:
                continue
            presets[document.preset_id] = build_marketplace_preset(document)
        return cls(presets)

    def get(self, preset_id: str) -> MarketplacePreset | None:
        return self._presets.get(preset_id)

    def plan_installation(
        self,
        selection: Sequence[str],
        *,
        installed_versions: Mapping[str, str] | None = None,
    ) -> MarketplacePlan:
        installed_versions = {str(key): value for key, value in (installed_versions or {}).items()}
        visited: set[str] = set()
        visiting: set[str] = set()
        order: list[str] = []
        dependency_map: dict[str, tuple[PresetDependency, ...]] = {}
        missing: dict[str, list[str]] = {}
        issues: list[str] = []

        def visit(preset_id: str) -> None:
            normalized = preset_id.strip()
            if not normalized:
                return
            if normalized in visited:
                return
            if normalized in visiting:
                issues.append(f"dependency-cycle:{normalized}")
                return
            visiting.add(normalized)
            preset = self._presets.get(normalized)
            if preset is None:
                issues.append(f"preset-missing:{normalized}")
                visiting.remove(normalized)
                return
            required_deps: list[PresetDependency] = []
            missing_dependencies: list[str] = []
            for dependency in preset.dependencies:
                if dependency.optional:
                    continue
                required_deps.append(dependency)
                visit(dependency.preset_id)
                if dependency.preset_id not in self._presets:
                    missing_dependencies.append(dependency.preset_id)
                else:
                    installed_version = installed_versions.get(dependency.preset_id)
                    constraint_status = _satisfies_constraint(
                        installed_version, dependency.constraints
                    )
                    if constraint_status is False:
                        issues.append(
                            f"version-constraint:{dependency.preset_id}:{','.join(dependency.constraints)}"
                        )
                    elif constraint_status is None and dependency.constraints:
                        issues.append(
                            f"version-constraint-unknown:{dependency.preset_id}:{','.join(dependency.constraints)}"
                        )
            dependency_map[normalized] = tuple(required_deps)
            if missing_dependencies:
                missing[normalized] = missing_dependencies
            visiting.remove(normalized)
            visited.add(normalized)
            order.append(normalized)

        for preset_id in selection:
            visit(str(preset_id))

        seen: set[str] = set()
        install_order: list[str] = []
        for preset_id in order:
            if preset_id not in seen:
                install_order.append(preset_id)
                seen.add(preset_id)

        upgrades: list[UpdateStep] = []
        for preset_id in install_order:
            preset = self._presets.get(preset_id)
            if preset is None:
                continue
            current_version = installed_versions.get(preset_id)
            diff = _compare_versions(current_version, preset.version)
            if diff is None:
                continue
            if diff < 0:
                upgrades.append(
                    UpdateStep(
                        preset_id=preset_id,
                        from_version=current_version,
                        to_version=preset.version,
                        channel=preset.preferred_channel,
                    )
                )

        return MarketplacePlan(
            install_order=tuple(install_order),
            required_dependencies={key: tuple(value) for key, value in dependency_map.items()},
            missing={key: tuple(value) for key, value in missing.items()},
            issues=tuple(dict.fromkeys(issues)),
            upgrades=tuple(upgrades),
        )


__all__ = [
    "MarketplaceIndex",
    "MarketplacePlan",
    "MarketplacePreset",
    "PresetDependency",
    "PresetUpdateChannel",
    "PresetUpdateDirective",
    "UpdateStep",
    "build_marketplace_preset",
]

