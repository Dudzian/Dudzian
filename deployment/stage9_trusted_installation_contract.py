"""Closed installation expectations, never installation authorization.

Only independently reviewed bytes may be pinned by a separate protected loader.
Parsing these bytes neither approves the manifest nor qualifies a live host.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, cast

from deployment.stage9_native_linux_cgroup_membership_observation import _valid_path

SCHEMA = "CryptoHunter.SourceCCAInstallationV1"
VERSION = 1
MAX_MANIFEST_BYTES = 65536
REQUIRED_ROLES = frozenset({"root_source", "history_source", "cca"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PRINCIPAL = re.compile(r"[a-z_][a-z0-9_-]{0,63}\Z")
_UNIT = re.compile(r"[a-z0-9][a-z0-9_-]{0,119}\.service\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_.-]{1,255}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CODES = frozenset(
    {
        "MANIFEST_MISSING",
        "MANIFEST_TOO_LARGE",
        "MANIFEST_FORMAT_INVALID",
        "MANIFEST_SCHEMA_UNSUPPORTED",
        "MANIFEST_FIELDS_INVALID",
        "MANIFEST_ROLES_INVALID",
        "MANIFEST_ROLE_UNKNOWN",
        "MANIFEST_IDENTIFIER_DUPLICATE",
        "MANIFEST_PATH_INVALID",
        "MANIFEST_FILE_POLICY_INVALID",
        "MANIFEST_PRINCIPAL_INVALID",
        "MANIFEST_ROLE_SEPARATION_REQUIRED",
        "MANIFEST_SUPPLEMENTARY_GROUPS_FORBIDDEN",
        "MANIFEST_CGROUP_INVALID",
        "BACKEND_OBSERVATION_FAILED",
    }
)


class InstallationRefusal(ValueError):
    """Only a fixed code may cross the reporting boundary."""

    def __init__(self, code: str):
        self.code = code if type(code) is str and code in _CODES else "BACKEND_OBSERVATION_FAILED"
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class FileExpectation:
    path: str
    sha256: str
    uid: int
    gid: int
    mode: int


@dataclass(frozen=True, slots=True)
class RoleExpectation:
    role: str
    installation_identity: str
    unit: str
    principal: str
    uid: int
    gid: int
    supplementary_groups: tuple[int, ...]
    unit_file: FileExpectation
    executable: FileExpectation
    control_group: str


@dataclass(frozen=True, slots=True)
class Manifest:
    installation_id: str
    roles: tuple[RoleExpectation, ...]
    sha256: str


def manifest_sha256(raw: bytes) -> str:
    """Digest exact supplied bytes; this is not a trust-on-first-use loader."""
    if type(raw) is not bytes:
        raise InstallationRefusal("MANIFEST_FORMAT_INVALID")
    if not raw:
        raise InstallationRefusal("MANIFEST_MISSING")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise InstallationRefusal("MANIFEST_TOO_LARGE")
    return hashlib.sha256(raw).hexdigest()


def _fields(value: Any, expected: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise InstallationRefusal("MANIFEST_FIELDS_INVALID")
    return cast(dict[str, Any], value)


def _integer(value: Any, *, minimum: int = 1, maximum: int = 2**32 - 2) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise InstallationRefusal("MANIFEST_PRINCIPAL_INVALID")
    return value


def _identifier(value: Any) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise InstallationRefusal("MANIFEST_FIELDS_INVALID")
    return value


def _path(value: Any) -> str:
    # Keep the existing observer's canonical-path check and exclude systemd
    # escapes, glob characters, whitespace and alternate component spellings.
    if not _valid_path(value) or any(
        not _COMPONENT.fullmatch(component) for component in value[1:].split("/")
    ):
        raise InstallationRefusal("MANIFEST_PATH_INVALID")
    return cast(str, value)


def _file(value: Any, *, executable: bool) -> FileExpectation:
    fields = _fields(value, {"path", "sha256", "uid", "gid", "mode"})
    path = _path(fields["path"])
    digest = fields["sha256"]
    uid, gid, mode = fields["uid"], fields["gid"], fields["mode"]
    if (
        type(digest) is not str
        or not _DIGEST.fullmatch(digest)
        or type(uid) is not int
        or uid != 0
        or type(gid) is not int
        or gid != 0
        or type(mode) is not int
        or not 0 <= mode <= 0o777
        or bool(mode & 0o022)
        or not mode & 0o400
        or (executable and not mode & 0o100)
        or (not executable and bool(mode & 0o111))
    ):
        raise InstallationRefusal("MANIFEST_FILE_POLICY_INVALID")
    return FileExpectation(path=path, sha256=digest, uid=uid, gid=gid, mode=mode)


def _role(value: Any) -> RoleExpectation:
    fields = _fields(
        value,
        {
            "role",
            "installation_identity",
            "unit",
            "principal",
            "uid",
            "gid",
            "supplementary_groups",
            "unit_file",
            "executable",
            "control_group",
        },
    )
    role = fields["role"]
    if type(role) is not str or role not in REQUIRED_ROLES:
        raise InstallationRefusal("MANIFEST_ROLE_UNKNOWN")
    identity = _identifier(fields["installation_identity"])
    unit, principal = fields["unit"], fields["principal"]
    if type(unit) is not str or not _UNIT.fullmatch(unit):
        raise InstallationRefusal("MANIFEST_FIELDS_INVALID")
    if type(principal) is not str or principal == "root" or not _PRINCIPAL.fullmatch(principal):
        raise InstallationRefusal("MANIFEST_PRINCIPAL_INVALID")
    uid, gid = _integer(fields["uid"]), _integer(fields["gid"])
    # S9-RPI-06/21 preserve isolated principals with no supplementary groups.
    groups = fields["supplementary_groups"]
    if type(groups) is not list or groups:
        raise InstallationRefusal("MANIFEST_SUPPLEMENTARY_GROUPS_FORBIDDEN")
    unit_file = _file(fields["unit_file"], executable=False)
    executable = _file(fields["executable"], executable=True)
    if unit_file.path.rsplit("/", 1)[-1] != unit:
        raise InstallationRefusal("MANIFEST_FIELDS_INVALID")
    control_group = _path(fields["control_group"])
    if not control_group.endswith("/" + unit):
        raise InstallationRefusal("MANIFEST_CGROUP_INVALID")
    return RoleExpectation(
        role=role,
        installation_identity=identity,
        unit=unit,
        principal=principal,
        uid=uid,
        gid=gid,
        supplementary_groups=(),
        unit_file=unit_file,
        executable=executable,
        control_group=control_group,
    )


def _unique(values: list[Any]) -> None:
    if len(set(values)) != len(values):
        raise InstallationRefusal("MANIFEST_IDENTIFIER_DUPLICATE")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InstallationRefusal("MANIFEST_IDENTIFIER_DUPLICATE")
        result[key] = value
    return result


def _constant(_value: str) -> Any:
    raise InstallationRefusal("MANIFEST_FORMAT_INVALID")


def parse_manifest(raw: bytes) -> Manifest:
    """Validate exact bytes, retaining their digest without approving them."""
    if raw is None:
        raise InstallationRefusal("MANIFEST_MISSING")
    digest = manifest_sha256(raw)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except InstallationRefusal:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise InstallationRefusal("MANIFEST_FORMAT_INVALID") from exc
    fields = _fields(value, {"schema", "version", "installation_id", "roles"})
    if (
        type(fields["schema"]) is not str
        or fields["schema"] != SCHEMA
        or type(fields["version"]) is not int
        or fields["version"] != VERSION
    ):
        raise InstallationRefusal("MANIFEST_SCHEMA_UNSUPPORTED")
    installation_id = _identifier(fields["installation_id"])
    candidates = fields["roles"]
    if type(candidates) is not list or len(candidates) != 3:
        raise InstallationRefusal("MANIFEST_ROLES_INVALID")
    roles = tuple(_role(candidate) for candidate in candidates)
    _unique([role.role for role in roles])
    if {role.role for role in roles} != REQUIRED_ROLES:
        raise InstallationRefusal("MANIFEST_ROLES_INVALID")
    for attribute in ("installation_identity", "unit", "principal"):
        _unique([getattr(role, attribute) for role in roles])
    _unique([pin.path for role in roles for pin in (role.unit_file, role.executable)])
    if len({role.uid for role in roles}) != 3 or len({role.gid for role in roles}) != 3:
        raise InstallationRefusal("MANIFEST_ROLE_SEPARATION_REQUIRED")
    for first in roles:
        for second in roles:
            if first is not second and (
                first.control_group == second.control_group
                or first.control_group.startswith(second.control_group + "/")
            ):
                raise InstallationRefusal("MANIFEST_CGROUP_INVALID")
    return Manifest(installation_id=installation_id, roles=roles, sha256=digest)
