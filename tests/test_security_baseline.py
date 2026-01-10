import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.security.baseline import generate_security_baseline_report
from bot_core.runtime.file_metadata import file_reference_metadata
from tests.test_audit_tls_assets_script import _CERT, _KEY


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _tls_config(
    *,
    enabled: bool,
    cert: Path | None = None,
    key: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        certificate_path=str(cert) if enabled and cert else None,
        private_key_path=str(key) if enabled and key else None,
        client_ca_path=None,
        require_client_auth=False,
        private_key_password_env=None,
        pinned_fingerprints=(),
    )


def _compact_dict(d: object) -> str:
    # Stabilny, krótki stringify do logów pytestowych.
    try:
        if isinstance(d, dict):
            keys = sorted(d.keys())
            items = ", ".join(f"{k}={d.get(k)!r}" for k in keys if k in d)
            return f"{{{items}}}"
    except Exception:
        pass
    return repr(d)


def _dump_file_meta(path: Path, role: str) -> str:
    meta = file_reference_metadata(path, role=role)
    # Wyciągamy tylko pola, które najczęściej tłumaczą warningi baselinu.
    keep = {
        "role": meta.get("role"),
        "path": meta.get("path"),
        "absolute_path": meta.get("absolute_path"),
        "exists": meta.get("exists"),
        "is_file": meta.get("is_file"),
        "is_symlink": meta.get("is_symlink"),
        "mode_octal": meta.get("mode_octal"),
        "permissions": meta.get("permissions"),
        "security_flags": meta.get("security_flags"),
        "security_warnings": meta.get("security_warnings"),
        "parent_directory": meta.get("parent_directory"),
        "parent_mode_octal": meta.get("parent_mode_octal"),
        "parent_security_flags": meta.get("parent_security_flags"),
        "parent_writable": meta.get("parent_writable"),
    }
    return _compact_dict(keep)


def _dump_report(report: object) -> str:
    status = getattr(report, "status", None)
    errors = getattr(report, "errors", None)
    warnings = getattr(report, "warnings", None)

    # Czasem report ma dodatkowe pola (np. details/sections/etc.) — próbujemy je złapać bez ryzyka.
    extra = {}
    for key in ("details", "sections", "metadata", "checks", "items", "context"):
        if hasattr(report, key):
            try:
                extra[key] = getattr(report, key)
            except Exception:
                extra[key] = "<unreadable>"

    return (
        "SECURITY_BASELINE_REPORT\n"
        f"status={status!r}\n"
        f"errors={errors!r}\n"
        f"warnings={warnings!r}\n"
        f"extra={extra!r}\n"
    )


def test_security_baseline_reports_errors_when_tokens_missing() -> None:
    metrics_tls = _tls_config(enabled=False)
    metrics_service = SimpleNamespace(
        enabled=True,
        auth_token="",
        tls=metrics_tls,
        rbac_tokens=(),
    )
    risk_service = SimpleNamespace(
        enabled=False,
        auth_token=None,
        tls=_tls_config(enabled=False),
        rbac_tokens=(),
    )
    scheduler = SimpleNamespace(
        name="core_multi",
        rbac_tokens=(),
    )
    core_config = SimpleNamespace(
        metrics_service=metrics_service,
        risk_service=risk_service,
        multi_strategy_schedulers={"core_multi": scheduler},
    )

    report = generate_security_baseline_report(core_config, env={})

    assert report.status == "error"
    assert report.errors
    assert any("RBAC" in message or "token" in message.lower() for message in report.errors)


def test_security_baseline_reports_ok_for_hardened_config(tmp_path: Path) -> None:
    cert_path = _write(tmp_path / "cert.pem", _CERT)
    key_path = _write(tmp_path / "key.pem", _KEY)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)

    metrics_service = SimpleNamespace(
        enabled=True,
        auth_token=None,
        rbac_tokens=(
            SimpleNamespace(
                token_id="metrics-reader",
                token_value="secret",
                token_env=None,
                token_hash=None,
                scopes=("metrics.read",),
            ),
        ),
        tls=_tls_config(enabled=True, cert=cert_path, key=key_path),
    )
    risk_service = SimpleNamespace(
        enabled=True,
        auth_token=None,
        rbac_tokens=(
            SimpleNamespace(
                token_id="risk-reader",
                token_value="secret",
                token_env=None,
                token_hash=None,
                scopes=("risk.read",),
            ),
        ),
        tls=_tls_config(enabled=True, cert=cert_path, key=key_path),
    )
    scheduler = SimpleNamespace(
        name="core_multi",
        rbac_tokens=(
            SimpleNamespace(
                token_id="scheduler-writer",
                token_value="secret",
                token_env=None,
                token_hash=None,
                scopes=("runtime.schedule.read", "runtime.schedule.write"),
            ),
        ),
    )
    core_config = SimpleNamespace(
        metrics_service=metrics_service,
        risk_service=risk_service,
        multi_strategy_schedulers={"core_multi": scheduler},
    )

    report = generate_security_baseline_report(
        core_config,
        env={},
        scheduler_required_scopes={"*": ("runtime.schedule.read", "runtime.schedule.write")},
    )

    # --- DIAGNOSTYKA: jeżeli fail, pokaż dokładnie co i dlaczego ---
    if getattr(report, "status", None) != "ok" or getattr(report, "warnings", None) or getattr(report, "errors", None):
        diag = []
        diag.append(_dump_report(report))
        diag.append("TLS_FILE_METADATA\n")
        diag.append(f"cert(tls_cert)={_dump_file_meta(cert_path, role='tls_cert')}\n")
        diag.append(f"key(tls_key)={_dump_file_meta(key_path, role='tls_key')}\n")
        # Dodatkowo: jak baseline/runner biega na Windows, często problemem jest parent dir perms.
        # Pokażmy też folder tmp_path jako directory poprzez file_reference_metadata na "fake file" nie ma sensu,
        # więc logujemy same ścieżki:
        diag.append(f"tmp_path={str(tmp_path)!r}\n")
        diag.append(f"cert_path={str(cert_path)!r}\n")
        diag.append(f"key_path={str(key_path)!r}\n")
        pytest.fail("\n".join(diag))

    assert report.status == "ok"
    assert not report.errors
    assert not report.warnings
