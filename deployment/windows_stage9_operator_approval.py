"""Bound operator intent and independent Windows session checks; no execution grant.

An approval document is data, never a bearer credential. The policy must come
from the separately reviewed host inventory, outside the downloaded artifact and
approval. PRE-01B.1 deliberately has no installer executor.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from bot_core.persistence.fingerprints import canonical_json_sha256
from deployment.windows_runner_acl import native_powershell_path
from deployment.windows_stage10_artifact_preflight import REPOSITORY
from deployment.windows_stage10_runner_preflight import _read_json_command

APPROVAL_SCOPE = (
    "transfer-qualified-msi",
    "prepare-clean-install",
    "verify-installed-payload",
)
BINDING_KEYS = (
    "repository",
    "source_revision",
    "ci_run_id",
    "ci_run_attempt",
    "artifact_id",
    "archive_sha256",
    "msi_sha256",
    "manifest_sha256",
    "production_trust",
)
APPROVAL_KEYS = frozenset(BINDING_KEYS) | {
    "schema_version",
    "approval_id",
    "host_identity",
    "operator_sid",
    "issued_at",
    "expires_at",
    "scope",
}
HOST_KEYS = {"hostname", "machine_guid", "hardware_uuid"}


class OperatorApprovalError(ValueError):
    """Approval data or independent session identity is inconsistent."""


class OperatorAuthenticationUnavailable(RuntimeError):
    """An actual interactive elevated Windows session could not be observed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OperatorApprovalError(message)


def _host(value: Any) -> dict[str, str]:
    _require(isinstance(value, dict) and set(value) == HOST_KEYS, "invalid host identity")
    _require(
        all(isinstance(item, str) and item.strip() == item and item for item in value.values()),
        "empty or malformed host identity",
    )
    for key in ("machine_guid", "hardware_uuid"):
        try:
            parsed = UUID(value[key])
        except (ValueError, AttributeError) as exc:
            raise OperatorApprovalError("host requires stable UUID identities") from exc
        _require(parsed.int != 0, "zero host UUID is not an identity")
    return dict(value)


def _sid(value: Any) -> bool:
    return (
        isinstance(value, str) and re.fullmatch(r"S-1-5-21-(?:[0-9]+-){3}[0-9]+", value) is not None
    )


def _utc(value: Any) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), "approval requires UTC timestamps")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OperatorApprovalError("invalid approval timestamp") from exc
    _require(result.utcoffset() == timedelta(0), "approval timestamp must be UTC")
    return result


def validate_operator_approval(
    approval: Any,
    qualification: dict[str, Any],
    *,
    policy: dict[str, Any],
    now: datetime | None = None,
) -> str:
    """Validate binding/expiry only; return a review digest, never authorization.

    Caller-supplied PASS flags and `approved=true` cannot authenticate anybody.
    `qualification` must be freshly obtained through the existing handoff, and
    `policy` independently installed by the host owner, not copied from approval.
    """
    _require(
        isinstance(approval, dict) and set(approval) == APPROVAL_KEYS, "invalid approval schema"
    )
    _require(
        type(approval["schema_version"]) is int and approval["schema_version"] == 1,
        "unsupported approval schema",
    )
    _require(
        isinstance(policy, dict) and set(policy) == {"host_identity", "operator_sids"},
        "invalid independent host policy",
    )
    operators = policy["operator_sids"]
    _require(
        isinstance(operators, list) and bool(operators) and all(_sid(sid) for sid in operators),
        "invalid independent operator allowlist",
    )
    _require(
        _sid(approval["operator_sid"]) and approval["operator_sid"] in operators,
        "operator is outside the independent allowlist",
    )
    _require(
        _host(approval["host_identity"]) == _host(policy["host_identity"]),
        "approval host conflicts with independent inventory",
    )
    _require(isinstance(approval["approval_id"], str), "invalid approval ID")
    try:
        approval_id = UUID(approval["approval_id"])
    except ValueError as exc:
        raise OperatorApprovalError("invalid approval ID") from exc
    _require(
        str(approval_id) == approval["approval_id"] and approval_id.int != 0,
        "approval ID must be a nonzero canonical UUID",
    )
    _require(approval["repository"] == REPOSITORY, "approval repository differs")
    _require(
        isinstance(approval["source_revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}", approval["source_revision"]) is not None,
        "full qualified revision is required",
    )
    for key in ("ci_run_id", "artifact_id"):
        _require(
            isinstance(approval[key], str)
            and re.fullmatch(r"[1-9][0-9]*", approval[key]) is not None,
            "invalid qualification identifier",
        )
    _require(
        type(approval["ci_run_attempt"]) is int and approval["ci_run_attempt"] > 0,
        "invalid run attempt",
    )
    for key in ("archive_sha256", "msi_sha256", "manifest_sha256"):
        _require(
            isinstance(approval[key], str)
            and re.fullmatch(r"[0-9a-f]{64}", approval[key]) is not None,
            "invalid qualification digest",
        )
    _require(
        isinstance(approval["production_trust"], dict) and bool(approval["production_trust"]),
        "missing Production Trust binding",
    )
    _require(
        all(key in qualification and approval[key] == qualification[key] for key in BINDING_KEYS),
        "approval differs from freshly qualified handoff",
    )
    _require(
        approval["scope"] == list(APPROVAL_SCOPE),
        "approval scope is not the reviewed preparation scope",
    )
    issued, expires = _utc(approval["issued_at"]), _utc(approval["expires_at"])
    current = now if now is not None else datetime.now(timezone.utc)
    _require(
        current.tzinfo is not None and current.utcoffset() == timedelta(0),
        "current time must be UTC aware",
    )
    _require(
        issued <= current < expires and timedelta(0) < expires - issued <= timedelta(hours=24),
        "approval expired, future-dated or exceeds 24 hours",
    )
    digest: str = canonical_json_sha256(approval)
    return digest


WINDOWS_OPERATOR_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$session = [Diagnostics.Process]::GetCurrentProcess().SessionId
$machineGuid = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Cryptography' `
  -Name MachineGuid -ErrorAction Stop
$hostIdentity = @{
  hostname = [Environment]::MachineName
  machine_guid = $machineGuid.MachineGuid
  hardware_uuid = (Get-CimInstance Win32_ComputerSystemProduct -ErrorAction Stop).UUID
}
@{
  operator_sid = $identity.User.Value
  authenticated = $identity.IsAuthenticated
  elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  interactive = [Environment]::UserInteractive
  session_id = $session
  impersonation = [string]$identity.ImpersonationLevel
  host_identity = $hostIdentity
} | ConvertTo-Json -Depth 5 -Compress
"""


def _collect_operator_session() -> dict[str, Any]:
    if os.name != "nt":
        raise OperatorAuthenticationUnavailable("native Windows session authentication unavailable")
    encoded = base64.b64encode(WINDOWS_OPERATOR_SCRIPT.encode("utf-16-le")).decode("ascii")
    # The script is fixed; no approval values, credentials or paths enter code.
    try:
        observed = _read_json_command(
            [native_powershell_path(), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise OperatorAuthenticationUnavailable("Windows session observation failed") from exc
    if not isinstance(observed, dict):
        raise OperatorAuthenticationUnavailable("Windows returned no session identity")
    return observed


def authenticate_operator(
    approval: dict[str, Any], qualification: dict[str, Any], *, policy: dict[str, Any]
) -> dict[str, Any]:
    """Independently read integrated OS identity. This is not installation consent.

    No snapshot/session JSON argument exists. A future executor additionally
    needs digest-bound interactive consent and a protected, reviewed policy.
    """
    digest = validate_operator_approval(approval, qualification, policy=policy)
    observed = _collect_operator_session()
    _require(
        observed.get("operator_sid") == approval["operator_sid"],
        "authenticated operator differs from approval",
    )
    _require(
        _host(observed.get("host_identity")) == _host(policy["host_identity"]),
        "authenticated host differs from inventory",
    )
    _require(
        observed.get("authenticated") is True
        and observed.get("elevated") is True
        and observed.get("interactive") is True
        and type(observed.get("session_id")) is int
        and observed["session_id"] > 0
        and observed.get("impersonation") == "None",
        "an authenticated interactive elevated non-impersonated operator session is required",
    )
    # Expiry must still hold after OS collection, not only before a slow read.
    _require(
        validate_operator_approval(approval, qualification, policy=policy) == digest,
        "approval changed during authentication",
    )
    return {
        "approval_sha256": digest,
        "operator_sid": observed["operator_sid"],
        "host_identity": dict(observed["host_identity"]),
        "installation_authorized": False,
    }
