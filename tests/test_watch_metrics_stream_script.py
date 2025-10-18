from __future__ import annotations

import argparse
import logging
import base64
import bz2
import gzip
import hashlib
import hmac
import json
import io
import lzma
import sys
import types
import zipfile
import tarfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pytest


try:  # pragma: no cover - zależy od dostępności środowiska gRPC
    from bot_core.runtime.metrics_service import MetricsServer
except Exception:  # pragma: no cover - brak grpcio lub stubów
    MetricsServer = None  # type: ignore

import scripts.watch_metrics_stream as watch_metrics_module
from scripts.watch_metrics_stream import (
    _ENV_PREFIX,
    _load_auth_token,
    _parse_env_bool,
    create_metrics_channel,
    main as watch_metrics_main,
)


def _write_risk_profile_file(tmp_path: Path, name: str = "ops", severity: str = "warning") -> Path:
    profiles_path = tmp_path / "telemetry_profiles.json"
    payload = {"risk_profiles": {name: {"severity_min": severity}}}
    profiles_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return profiles_path


def _write_core_config(
    tmp_path: Path,
    *,
    profiles_path: Path,
    profile_name: str = "ops",
    host: str = "10.20.30.40",
    port: int = 50200,
    metrics_enabled: bool = True,
    metrics_rbac_tokens: Sequence[dict[str, object]] | None = None,
    metrics_tls: Mapping[str, object] | None = None,
    metrics_grpc_metadata: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
    metrics_grpc_metadata_files: Sequence[str] | None = None,
    metrics_grpc_metadata_directories: Sequence[str] | None = None,
    risk_host: str = "127.0.0.1",
    risk_port: int = 60300,
    risk_enabled: bool = True,
    risk_rbac_tokens: Sequence[dict[str, object]] | None = None,
    risk_tls: Mapping[str, object] | None = None,
    offline_environments: Sequence[str] | None = None,
) -> Path:
    config_path = tmp_path / "core.yaml"
    profiles_str = str(profiles_path)
    lines = [
        "risk_profiles:",
        "  conservative:",
        "    max_daily_loss_pct: 0.05",
        "    max_position_pct: 0.10",
        "    target_volatility: 0.2",
        "    max_leverage: 3.0",
        "    stop_loss_atr_multiple: 1.5",
        "    max_open_positions: 5",
        "    hard_drawdown_pct: 0.25",
    ]

    if offline_environments:
        lines.append("environments:")
        for name in offline_environments:
            lines.extend(
                [
                    f"  {name}:",
                    "    exchange: coinbase",
                    "    environment: paper",
                    f"    keychain_key: {name}_key",
                    f"    data_cache_path: {str(tmp_path / ('data_' + name))}",
                    "    risk_profile: conservative",
                    "    alert_channels: []",
                    "    offline_mode: true",
                ]
            )
    else:
        lines.append("environments: {}")

    lines.extend(
        [
            "runtime:",
            "  metrics_service:",
            f"    enabled: {'true' if metrics_enabled else 'false'}",
            f"    host: {host}",
            f"    port: {port}",
            f"    ui_alerts_risk_profile: {profile_name}",
            f"    ui_alerts_risk_profiles_file: {profiles_str}",
        ]
    )

    if metrics_rbac_tokens:
        lines.append("    rbac_tokens:")
        for token in metrics_rbac_tokens:
            token_id = token.get("token_id", "")
            lines.append(f"      - token_id: {token_id}")
            token_env = token.get("token_env")
            if token_env:
                lines.append(f"        token_env: {token_env}")
            token_value = token.get("token_value")
            if token_value:
                lines.append(f"        token_value: {token_value}")
            token_hash = token.get("token_hash")
            if token_hash:
                lines.append(f"        token_hash: {token_hash}")
            scopes = token.get("scopes")
            if scopes:
                lines.append("        scopes:")
                for scope in scopes:
                    lines.append(f"          - {scope}")

    if metrics_tls:
        lines.append("    tls:")
        for key, value in metrics_tls.items():
            if value is None:
                continue
            indent = "      "
            if isinstance(value, (list, tuple)):
                lines.append(f"{indent}{key}:")
                for item in value:
                    lines.append(f"{indent}  - {item}")
            else:
                rendered = value
                if isinstance(value, bool):
                    rendered = "true" if value else "false"
                lines.append(f"{indent}{key}: {rendered}")

    if metrics_grpc_metadata:
        lines.append("    grpc_metadata:")
        if isinstance(metrics_grpc_metadata, Mapping):
            items = metrics_grpc_metadata.items()
        else:
            items = metrics_grpc_metadata
        for entry in items:
            if isinstance(entry, Mapping):
                key = entry.get("key")
                lines.append(f"      - key: {key}")
                if "value" in entry:
                    lines.append(f"        value: {entry['value']}")
                if "value_env" in entry:
                    lines.append(f"        value_env: {entry['value_env']}")
                if "value_file" in entry:
                    lines.append(f"        value_file: {entry['value_file']}")
            else:
                key, value = entry
                lines.append(f"      {key}: {value}")

    if metrics_grpc_metadata_files:
        lines.append("    grpc_metadata_files:")
        for entry in metrics_grpc_metadata_files:
            lines.append(f"      - {entry}")

    if metrics_grpc_metadata_directories:
        lines.append("    grpc_metadata_directories:")
        for entry in metrics_grpc_metadata_directories:
            lines.append(f"      - {entry}")

    lines.extend(
        [
            "  risk_service:",
            f"    enabled: {'true' if risk_enabled else 'false'}",
            f"    host: {risk_host}",
            f"    port: {risk_port}",
            "    publish_interval_seconds: 5.0",
        ]
    )

    if risk_rbac_tokens:
        lines.append("    rbac_tokens:")
        for token in risk_rbac_tokens:
            token_id = token.get("token_id", "")
            lines.append(f"      - token_id: {token_id}")
            token_env = token.get("token_env")
            if token_env:
                lines.append(f"        token_env: {token_env}")
            token_value = token.get("token_value")
            if token_value:
                lines.append(f"        token_value: {token_value}")
            token_hash = token.get("token_hash")
            if token_hash:
                lines.append(f"        token_hash: {token_hash}")
            scopes = token.get("scopes")
            if scopes:
                lines.append("        scopes:")
                for scope in scopes:
                    lines.append(f"          - {scope}")

    if risk_tls:
        lines.append("    tls:")
        for key, value in risk_tls.items():
            if value is None:
                continue
            indent = "      "
            if isinstance(value, (list, tuple)):
                lines.append(f"{indent}{key}:")
                for item in value:
                    lines.append(f"{indent}  - {item}")
            else:
                rendered = value
                if isinstance(value, bool):
                    rendered = "true" if value else "false"
                lines.append(f"{indent}{key}: {rendered}")

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _assert_signed_entry(entry: dict[str, object], key: bytes, *, key_id: str | None) -> None:
    assert "signature" in entry, "brak podpisu w decision logu"
    signature = entry["signature"]
    assert signature["algorithm"] == "HMAC-SHA256"
    if key_id is None:
        assert "key_id" not in signature or signature["key_id"] == key_id
    else:
        assert signature.get("key_id") == key_id
    entry_copy = dict(entry)
    entry_copy.pop("signature", None)
    canonical = json.dumps(
        entry_copy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected = base64.b64encode(hmac.new(key, canonical, hashlib.sha256).digest()).decode("ascii")
    assert signature["value"] == expected


def test_parse_env_bool_recognises_true_false():
    parser = argparse.ArgumentParser()
    assert _parse_env_bool("true", variable="VAR", parser=parser) is True
    assert _parse_env_bool("0", variable="VAR", parser=parser) is False


def test_parse_env_bool_invalid_value():
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        _parse_env_bool("maybe", variable="VAR", parser=parser)


def test_load_auth_token_from_file(tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("secret-token\n")
    assert _load_auth_token(None, str(token_path)) == "secret-token"


def test_load_auth_token_rejects_empty_file(tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("   ")
    with pytest.raises(SystemExit):
        _load_auth_token(None, str(token_path))


def test_load_auth_token_conflicting_sources(tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("secret")
    with pytest.raises(SystemExit):
        _load_auth_token("cli-token", str(token_path))


def test_watch_metrics_stream_print_risk_profiles(capsys):
    exit_code = watch_metrics_main(["--print-risk-profiles"])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert "risk_profiles" in payload
    assert "conservative" in payload["risk_profiles"]
    assert payload["risk_profiles"]["conservative"]["origin"] == "builtin"
    assert payload.get("selected") is None


def test_watch_metrics_stream_print_risk_profiles_env(monkeypatch, capsys):
    monkeypatch.setenv(f"{_ENV_PREFIX}PRINT_RISK_PROFILES", "1")
    exit_code = watch_metrics_main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert "risk_profiles" in payload
    assert payload["risk_profiles"]["conservative"]["origin"] == "builtin"
    assert payload.get("selected") is None


def test_environment_numeric_override_allows_none_keyword(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.limit = 25
    monkeypatch.setenv(f"{_ENV_PREFIX}LIMIT", "NONE")
    watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert args.limit is None


def test_environment_numeric_override_allows_null_with_spaces(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.timeout = 17.5
    monkeypatch.setenv(f"{_ENV_PREFIX}TIMEOUT", "  null \t")
    watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert args.timeout is None


def test_environment_numeric_override_supports_default(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.port = 60000
    monkeypatch.setenv(f"{_ENV_PREFIX}PORT", "default")
    watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert args.port == parser.get_default("port")


def test_environment_simple_override_allows_default(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.host = "10.20.30.40"
    monkeypatch.setenv(f"{_ENV_PREFIX}HOST", "DEFAULT")
    watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert args.host == parser.get_default("host")


def test_environment_simple_override_allows_none(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.event = "reduce_motion"
    monkeypatch.setenv(f"{_ENV_PREFIX}EVENT", " none ")
    watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert args.event is None


def test_environment_list_override_allows_none(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.severity = ["warning"]
    monkeypatch.setenv(f"{_ENV_PREFIX}SEVERITY", "none")
    watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert args.severity is None


def test_environment_list_override_supports_default(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.severity = ["info"]
    monkeypatch.setenv(f"{_ENV_PREFIX}SEVERITY", "DEFAULT")
    watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert args.severity == parser.get_default("severity")


def test_environment_tls_none_does_not_force_tls(monkeypatch):
    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args([])
    args.use_tls = False
    monkeypatch.setenv(f"{_ENV_PREFIX}ROOT_CERT", "NONE")
    tls_env_present, env_use_tls_explicit = watch_metrics_module._apply_environment_overrides(
        args, parser=parser, provided_flags=set()
    )
    assert tls_env_present is False
    assert env_use_tls_explicit is False
    assert args.root_cert is None


def test_watch_metrics_stream_risk_profiles_file_cli(tmp_path, capsys):
    profiles_path = tmp_path / "telemetry_profiles.json"
    profiles_path.write_text(
        json.dumps({"risk_profiles": {"ops": {"severity_min": "error"}}}, ensure_ascii=False)
    )

    exit_code = watch_metrics_main([
        "--print-risk-profiles",
        "--risk-profiles-file",
        str(profiles_path),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert "ops" in payload["risk_profiles"]
    assert payload["risk_profiles"]["ops"]["severity_min"] == "error"
    assert payload["risk_profiles"]["ops"]["origin"].startswith("watcher:")


def test_watch_metrics_stream_risk_profiles_file_env(monkeypatch, tmp_path, capsys):
    profiles_path = tmp_path / "telemetry_profiles.json"
    profiles_path.write_text(
        json.dumps({"risk_profiles": {"lab": {"severity_min": "info"}}}, ensure_ascii=False)
    )
    monkeypatch.setenv(f"{_ENV_PREFIX}RISK_PROFILES_FILE", str(profiles_path))
    exit_code = watch_metrics_main(["--print-risk-profiles"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert "lab" in payload["risk_profiles"]
    assert payload["risk_profiles"]["lab"]["severity_min"] == "info"
    assert payload["risk_profiles"]["lab"]["origin"].startswith("watcher:")


def test_watch_metrics_stream_risk_profiles_directory(tmp_path, capsys):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "ops.json").write_text(
        json.dumps({"risk_profiles": {"ops_dir": {"severity_min": "error"}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (profiles_dir / "lab.yaml").write_text(
        "risk_profiles:\n  lab_dir:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    exit_code = watch_metrics_main(
        ["--print-risk-profiles", "--risk-profiles-file", str(profiles_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["risk_profiles"]["ops_dir"]["origin"].endswith("#ops.json")
    assert payload["risk_profiles"]["lab_dir"]["origin"].endswith("#lab.yaml")


def test_watch_metrics_stream_core_config_prints_profiles(tmp_path, capsys):
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="error")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        profile_name="ops",
        host="192.0.2.10",
        port=50111,
    )

    exit_code = watch_metrics_main(["--print-risk-profiles", "--core-config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["selected"] == "ops"
    selected_profile = payload["selected_profile"]
    assert selected_profile["origin"].startswith("watcher:")
    assert selected_profile["severity_min"] == "error"
    core_meta = payload.get("core_config")
    assert core_meta["path"] == str(config_path)
    assert core_meta["metrics_service"]["risk_profile"] == "ops"
    assert core_meta["metrics_service"]["risk_profiles_file"] == str(profiles_path)
    assert core_meta["metrics_service"]["auth_token_scope_required"] == "metrics.read"


def test_watch_metrics_stream_core_config_applies_tls_defaults(tmp_path, capsys):
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="error")
    root_cert = tmp_path / "metrics_root.pem"
    root_bytes = b"demo-root-cert"
    root_cert.write_bytes(root_bytes)
    client_cert = tmp_path / "metrics_client.pem"
    client_cert.write_text("dummy-cert", encoding="utf-8")
    client_key = tmp_path / "metrics_client.key"
    client_key.write_text("dummy-key", encoding="utf-8")
    fingerprint = hashlib.sha256(root_bytes).hexdigest()

    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        profile_name="ops",
        metrics_tls={
            "enabled": True,
            "client_ca_path": str(root_cert),
            "certificate_path": str(client_cert),
            "private_key_path": str(client_key),
            "require_client_auth": True,
            "pinned_fingerprints": [f"sha256:{fingerprint}"],
        },
    )

    exit_code = watch_metrics_main(["--print-risk-profiles", "--core-config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    metrics_meta = payload["core_config"]["metrics_service"]
    assert metrics_meta["tls_enabled"] is True
    assert metrics_meta["client_auth"] is True
    assert metrics_meta["use_tls"] is True
    assert metrics_meta["root_cert"] == str(root_cert)
    assert metrics_meta["client_cert"] == str(client_cert)
    assert metrics_meta["client_key"] == str(client_key)
    assert metrics_meta["server_sha256"] == fingerprint
    assert metrics_meta["server_sha256_source"] == "pinned_fingerprint"
    assert metrics_meta["pinned_fingerprints"] == [f"sha256:{fingerprint}"]
    assert metrics_meta["pinned_fingerprint_selected"] == fingerprint
    assert metrics_meta["auth_token_scope_required"] == "metrics.read"
    assert metrics_meta["auth_token_scope_checked"] is False


def _start_server(auth_token: str | None = None) -> MetricsServer:
    if MetricsServer is None:
        pytest.skip("Brak MetricsServer (zainstaluj grpcio i wygeneruj stuby)")
    try:
        server = MetricsServer(host="127.0.0.1", port=0, sinks=(), auth_token=auth_token)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    server.start()
    return server


def _append_snapshots(server: MetricsServer, payloads: Iterable[dict[str, object]]) -> None:
    try:
        from bot_core.generated import trading_pb2  # type: ignore
    except Exception:
        pytest.skip("Brak stubów trading_pb2 – uruchom generate_trading_stubs.py")

    for payload in payloads:
        snapshot = trading_pb2.MetricsSnapshot()
        snapshot.notes = json.dumps(payload, ensure_ascii=False)
        server.store.append(snapshot)


@pytest.mark.timeout(5)
def test_watch_metrics_stream_outputs_json(tmp_path, capsys):
    server = _start_server()
    try:
        _append_snapshots(
            server,
            [
                {
                    "event": "reduce_motion",
                    "active": True,
                    "window_count": 2,
                    "severity": "warning",
                },
                {
                    "event": "reduce_motion",
                    "active": False,
                    "window_count": 1,
                    "severity": "info",
                },
            ],
        )

        host, port = server.address.split(":")
        exit_code = watch_metrics_main(
            [
                "--host",
                host,
                "--port",
                port,
                "--limit",
                "1",
                "--format",
                "json",
                "--event",
                "reduce_motion",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "reduce_motion" in captured.out
        assert "window_count" in captured.out
        assert '"severity": "warning"' in captured.out
    finally:
        server.stop(0)


class _DummyGrpc:
    def __init__(self) -> None:
        self.insecure_calls: list[str] = []
        self.secure_calls: list[tuple[str, object, tuple]] = []
        self.ssl_args: list[tuple[bytes | None, bytes | None, bytes | None]] = []

    def insecure_channel(self, address: str) -> str:
        self.insecure_calls.append(address)
        return f"insecure:{address}"

    def secure_channel(self, address: str, credentials: object, options: tuple = ()) -> str:
        self.secure_calls.append((address, credentials, tuple(options)))
        return f"secure:{address}"

    def ssl_channel_credentials(
        self,
        *,
        root_certificates: bytes | None = None,
        private_key: bytes | None = None,
        certificate_chain: bytes | None = None,
    ) -> dict[str, bytes | None]:
        args = (root_certificates, private_key, certificate_chain)
        self.ssl_args.append(args)
        return {"root": root_certificates, "key": private_key, "chain": certificate_chain}


class _DummyMetricsRequest:
    def __init__(self) -> None:
        self.include_ui_metrics = False


class _FakeSnapshot:
    def __init__(self, payload: dict[str, object], *, fps: float | None = None) -> None:
        self.notes = json.dumps(payload, ensure_ascii=False)
        self.fps = fps
        self.generated_at = None

    def HasField(self, field: str) -> bool:  # pragma: no cover - prosty stub
        if field == "generated_at":
            return False
        if field == "fps":
            return self.fps is not None
        return False


class _StubCollector:
    def __init__(self) -> None:
        self.calls: list[
            tuple[object, float | None, list[tuple[str, bytes | str]] | None]
        ] = []
        self.response: Iterable[object] = ()

    def StreamMetrics(
        self,
        request,
        timeout: float | None = None,
        metadata: list[tuple[str, bytes | str]] | None = None,
    ) -> Iterable[object]:
        self.calls.append((request, timeout, metadata))
        return list(self.response)


def _install_dummy_loader(monkeypatch, stub: _StubCollector) -> None:
    dummy_pb2 = types.SimpleNamespace(MetricsRequest=_DummyMetricsRequest)
    dummy_pb2_grpc = types.SimpleNamespace(MetricsServiceStub=lambda channel: stub)
    monkeypatch.setattr(
        watch_metrics_module,
        "_load_grpc_components",
        lambda: (types.SimpleNamespace(), dummy_pb2, dummy_pb2_grpc),
    )


def test_watch_metrics_stream_from_jsonl(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-01-01T00:00:00+00:00",
            "fps": 58.0,
            "notes": json.dumps(
                {
                    "event": "reduce_motion",
                    "active": True,
                    "screen": {"index": 1, "name": "HDR Monitor"},
                },
                ensure_ascii=False,
            ),
        },
        {
            "generated_at": "2024-01-01T00:00:01+00:00",
            "fps": 120.0,
            "notes": json.dumps({"event": "overlay_budget"}, ensure_ascii=False),
        },
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--event",
            "reduce_motion",
            "--screen-index",
            "1",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reduce_motion" in captured.out
    assert "screen" in captured.out


def test_watch_metrics_stream_from_multiple_jsonl_sources(tmp_path, capsys):
    first_records = [
        {
            "generated_at": "2024-01-02T00:00:00+00:00",
            "notes": json.dumps({"event": "reduce_motion", "severity": "info"}, ensure_ascii=False),
        }
    ]
    second_records = [
        {
            "generated_at": "2024-01-03T00:00:00+00:00",
            "notes": json.dumps({"event": "overlay_budget", "severity": "warning"}, ensure_ascii=False),
        }
    ]
    first_path = tmp_path / "metrics_a.jsonl"
    first_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in first_records) + "\n")
    second_path = tmp_path / "metrics_b.jsonl"
    second_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in second_records) + "\n")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(first_path),
            "--from-jsonl",
            str(second_path),
            "--format",
            "json",
            "--summary",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reduce_motion" in captured.out
    assert "overlay_budget" in captured.out
    assert '"summary"' in captured.out


def test_watch_metrics_stream_from_gzip_jsonl(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-03-01T00:00:00+00:00",
            "notes": json.dumps({"event": "overlay_budget", "severity": "warning"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-03-01T00:00:10+00:00",
            "notes": json.dumps({"event": "reduce_motion", "severity": "info"}, ensure_ascii=False),
        },
    ]
    gzip_path = tmp_path / "metrics.jsonl.gz"
    with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(gzip_path),
            "--format",
            "json",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out


def test_watch_metrics_stream_from_lzma_jsonl(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-05-01T00:00:00+00:00",
            "notes": json.dumps({"event": "overlay_budget", "severity": "warning"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-05-01T00:00:05+00:00",
            "notes": json.dumps({"event": "reduce_motion", "severity": "info"}, ensure_ascii=False),
        },
    ]
    lzma_path = tmp_path / "metrics.jsonl.xz"
    with lzma.open(lzma_path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(lzma_path),
            "--format",
            "json",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out


def test_watch_metrics_stream_from_bz2_jsonl(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-06-01T00:00:00+00:00",
            "notes": json.dumps({"event": "overlay_budget", "severity": "warning"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-06-01T00:00:05+00:00",
            "notes": json.dumps({"event": "reduce_motion", "severity": "info"}, ensure_ascii=False),
        },
    ]
    bz2_path = tmp_path / "metrics.jsonl.bz2"
    with bz2.open(bz2_path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(bz2_path),
            "--format",
            "json",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out


def test_watch_metrics_stream_from_zip_jsonl(tmp_path, capsys):
    archive_path = tmp_path / "metrics_bundle.zip"
    records = [
        {
            "generated_at": "2024-07-01T00:00:00+00:00",
            "notes": json.dumps({"event": "overlay_budget", "severity": "warning"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-07-01T00:00:05+00:00",
            "notes": json.dumps({"event": "reduce_motion", "severity": "info"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-07-01T00:00:10+00:00",
            "notes": json.dumps({"event": "hud_toggle", "severity": "info"}, ensure_ascii=False),
        },
    ]
    with zipfile.ZipFile(archive_path, "w") as archive:
        payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        archive.writestr("nested/metrics.jsonl", payload)

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(archive_path),
            "--format",
            "json",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out


def _write_tar_with_payload(path: Path, members: Mapping[str, str], *, mode: str = "w") -> None:
    with tarfile.open(path, mode) as archive:
        for name, payload in members.items():
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_watch_metrics_stream_from_tar_jsonl(tmp_path, capsys):
    tar_path = tmp_path / "metrics_bundle.tar"
    records = [
        {
            "generated_at": "2024-08-01T00:00:00+00:00",
            "notes": json.dumps({"event": "overlay_budget", "severity": "warning"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-08-01T00:00:10+00:00",
            "notes": json.dumps({"event": "hud_toggle", "severity": "info"}, ensure_ascii=False),
        },
    ]
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    _write_tar_with_payload(tar_path, {"nested/metrics.jsonl": payload})

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(tar_path),
            "--format",
            "json",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out


def test_watch_metrics_stream_from_tgz_jsonl(tmp_path, capsys):
    tgz_path = tmp_path / "metrics_bundle.tgz"
    records = [
        {
            "generated_at": "2024-09-01T00:00:00+00:00",
            "notes": json.dumps({"event": "overlay_budget", "severity": "warning"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-09-01T00:00:05+00:00",
            "notes": json.dumps({"event": "reduce_motion", "severity": "info"}, ensure_ascii=False),
        },
    ]
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    _write_tar_with_payload(tgz_path, {"metrics.jsonl": payload}, mode="w:gz")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(tgz_path),
            "--format",
            "json",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out


def test_watch_metrics_stream_from_directory_jsonl(tmp_path, capsys):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    gz_path = segments_dir / "001_part.jsonl.gz"
    gz_records = [
        {
            "generated_at": "2024-10-01T00:00:00+00:00",
            "notes": json.dumps(
                {"event": "reduce_motion", "severity": "info"}, ensure_ascii=False
            ),
        },
        {
            "generated_at": "2024-10-01T00:00:10+00:00",
            "notes": json.dumps(
                {"event": "hud_toggle", "severity": "warning"}, ensure_ascii=False
            ),
        },
    ]
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        for record in gz_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    plain_path = segments_dir / "010_part.jsonl"
    plain_records = [
        {
            "generated_at": "2024-10-01T00:00:20+00:00",
            "notes": json.dumps(
                {"event": "overlay_budget", "severity": "error"}, ensure_ascii=False
            ),
        }
    ]
    plain_payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in plain_records)
    plain_path.write_text(plain_payload + "\n", encoding="utf-8")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(segments_dir),
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0

    parsed_lines = [
        json.loads(line)
        for line in captured.out.splitlines()
        if line.strip()
    ]
    events: list[str] = []
    for entry in parsed_lines:
        raw_notes = entry.get("notes")
        notes_obj = json.loads(raw_notes) if isinstance(raw_notes, str) else raw_notes
        if isinstance(notes_obj, Mapping):
            events.append(str(notes_obj.get("event")))

    assert events == ["reduce_motion", "hud_toggle", "overlay_budget"]


def test_watch_metrics_stream_from_glob_pattern(tmp_path, capsys):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    plain_path = segments_dir / "metrics_2024_part1.jsonl"
    plain_records = [
        {
            "generated_at": "2024-11-01T00:00:05+00:00",
            "notes": json.dumps(
                {"event": "reduce_motion", "severity": "info"}, ensure_ascii=False
            ),
        }
    ]
    plain_payload = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in plain_records)
    plain_path.write_text(plain_payload + "\n", encoding="utf-8")

    gz_path = segments_dir / "metrics_2024_part2.jsonl.gz"
    gz_records = [
        {
            "generated_at": "2024-11-01T00:00:10+00:00",
            "notes": json.dumps(
                {"event": "hud_toggle", "severity": "warning"}, ensure_ascii=False
            ),
        },
        {
            "generated_at": "2024-11-01T00:00:20+00:00",
            "notes": json.dumps(
                {"event": "overlay_budget", "severity": "error"}, ensure_ascii=False
            ),
        },
    ]
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        for record in gz_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    pattern = str(segments_dir / "metrics_2024_part*.jsonl*")
    exit_code = watch_metrics_main(["--from-jsonl", pattern, "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0

    parsed_lines = [
        json.loads(line)
        for line in captured.out.splitlines()
        if line.strip()
    ]
    events = []
    for entry in parsed_lines:
        raw_notes = entry.get("notes")
        notes_obj = json.loads(raw_notes) if isinstance(raw_notes, str) else raw_notes
        if isinstance(notes_obj, Mapping):
            events.append(str(notes_obj.get("event")))

    assert events == ["reduce_motion", "hud_toggle", "overlay_budget"]


def test_watch_metrics_stream_directory_without_jsonl(tmp_path, caplog):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            watch_metrics_main(["--from-jsonl", str(segments_dir)])

    assert excinfo.value.code == 2
    assert "nie zawiera obsługiwanych artefaktów JSONL" in caplog.text


def test_watch_metrics_stream_glob_without_matches(tmp_path, caplog):
    pattern = str(tmp_path / "segments" / "*.jsonl")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            watch_metrics_main(["--from-jsonl", pattern])

    assert excinfo.value.code == 2
    assert "nie pasuje do żadnych artefaktów JSONL" in caplog.text


def test_watch_metrics_stream_from_manifest_text_file(tmp_path, capsys):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    first_segment = bundle_dir / "segment_a.jsonl"
    first_records = [
        {
            "generated_at": "2024-12-01T00:00:00+00:00",
            "notes": json.dumps(
                {"event": "reduce_motion", "severity": "info"}, ensure_ascii=False
            ),
        }
    ]
    first_segment.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in first_records) + "\n",
        encoding="utf-8",
    )

    second_segment = bundle_dir / "segment_b.jsonl.gz"
    second_records = [
        {
            "generated_at": "2024-12-01T00:00:10+00:00",
            "notes": json.dumps(
                {"event": "hud_toggle", "severity": "warning"}, ensure_ascii=False
            ),
        },
        {
            "generated_at": "2024-12-01T00:00:20+00:00",
            "notes": json.dumps(
                {"event": "overlay_budget", "severity": "error"}, ensure_ascii=False
            ),
        },
    ]
    with gzip.open(second_segment, "wt", encoding="utf-8") as handle:
        for record in second_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_path = bundle_dir / "segments.txt"
    manifest_path.write_text(
        "# komentarz\nsegment_a.jsonl\n./segment_b.jsonl.gz\n",
        encoding="utf-8",
    )

    exit_code = watch_metrics_main(
        ["--from-jsonl", f"@manifest:{manifest_path}", "--format", "json"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0

    parsed_lines = [
        json.loads(line)
        for line in captured.out.splitlines()
        if line.strip()
    ]
    events: list[str] = []
    for entry in parsed_lines:
        raw_notes = entry.get("notes")
        notes_obj = json.loads(raw_notes) if isinstance(raw_notes, str) else raw_notes
        if isinstance(notes_obj, Mapping):
            events.append(str(notes_obj.get("event")))

    assert events == ["reduce_motion", "hud_toggle", "overlay_budget"]


def test_watch_metrics_stream_from_manifest_json_list(tmp_path, capsys):
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

    plain_path = segments_dir / "metrics_part1.jsonl"
    plain_payload = [
        {
            "generated_at": "2025-01-01T00:00:05+00:00",
            "notes": json.dumps(
                {"event": "reduce_motion", "severity": "info"}, ensure_ascii=False
            ),
        }
    ]
    plain_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in plain_payload) + "\n",
        encoding="utf-8",
    )

    gz_path = segments_dir / "metrics_part2.jsonl.gz"
    gz_payload = [
        {
            "generated_at": "2025-01-01T00:00:15+00:00",
            "notes": json.dumps(
                {"event": "hud_toggle", "severity": "warning"}, ensure_ascii=False
            ),
        }
    ]
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        for record in gz_payload:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_path = tmp_path / "segments_manifest.json"
    manifest_payload = [
        "segments/metrics_part1.jsonl",
        "segments/metrics_part2.jsonl.gz",
    ]
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")

    exit_code = watch_metrics_main(
        ["--from-jsonl", f"@manifest:{manifest_path}", "--format", "json"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0

    parsed_lines = [
        json.loads(line)
        for line in captured.out.splitlines()
        if line.strip()
    ]
    events = []
    for entry in parsed_lines:
        raw_notes = entry.get("notes")
        notes_obj = json.loads(raw_notes) if isinstance(raw_notes, str) else raw_notes
        if isinstance(notes_obj, Mapping):
            events.append(str(notes_obj.get("event")))

    assert events == ["reduce_motion", "hud_toggle"]


def test_watch_metrics_stream_manifest_without_entries(tmp_path, caplog):
    manifest_path = tmp_path / "empty_segments.txt"
    manifest_path.write_text("# brak wpisów\n\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as excinfo:
            watch_metrics_main(["--from-jsonl", f"@manifest:{manifest_path}"])

    assert excinfo.value.code == 2
    assert "nie zawiera ścieżek JSONL" in caplog.text


def test_watch_metrics_stream_from_jsonl_stdin(monkeypatch, capsys):
    payload = "\n".join(
        json.dumps({"generated_at": "2024-04-01T00:00:00Z", "notes": {"event": "reduce_motion"}}, ensure_ascii=False)
        for _ in range(2)
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            "-",
            "--limit",
            "1",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reduce_motion" in captured.out


def test_watch_metrics_stream_from_jsonl_with_time_filters(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-02-01T00:00:00+00:00",
            "notes": json.dumps({"event": "reduce_motion"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-02-01T00:00:30+00:00",
            "notes": json.dumps({"event": "overlay_budget"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-02-01T00:02:00+00:00",
            "notes": json.dumps({"event": "jank_spike"}, ensure_ascii=False),
        },
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--since",
            "2024-02-01T00:00:10Z",
            "--until",
            "2024-02-01T00:01:00Z",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out
    assert "reduce_motion" not in captured.out
    assert "jank_spike" not in captured.out


def test_watch_metrics_stream_time_filters_from_env(monkeypatch, tmp_path, capsys):
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "generated_at": ts,
                    "notes": json.dumps({"event": label}, ensure_ascii=False),
                },
                ensure_ascii=False,
            )
            for ts, label in (
                ("2024-02-01T00:00:00+00:00", "reduce_motion"),
                ("2024-02-01T00:00:30+00:00", "overlay_budget"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(f"{_ENV_PREFIX}SINCE", "2024-02-01T00:00:10Z")
    monkeypatch.setenv(f"{_ENV_PREFIX}UNTIL", "2024-02-01T00:01:00+00:00")

    exit_code = watch_metrics_main([
        "--from-jsonl",
        str(jsonl_path),
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out
    assert "reduce_motion" not in captured.out


def test_watch_metrics_stream_decision_log_offline(tmp_path, capsys):
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "notes": {
                    "event": "reduce_motion",
                    "active": True,
                    "severity": "warning",
                    "screen": {
                        "index": 1,
                        "name": "Primary",
                        "geometry_px": {"width": 2560, "height": 1440},
                        "refresh_hz": 120,
                    },
                },
                "generated_at": {"seconds": 1_694_000_000, "nanos": 0},
                "fps": 55.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    decision_log = tmp_path / "audit" / "ui_decision.jsonl"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--decision-log",
            str(decision_log),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reduce_motion" in captured.out

    lines = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    metadata_entry = json.loads(lines[0])
    assert metadata_entry["kind"] == "metadata"
    assert metadata_entry["metadata"]["mode"] == "jsonl"
    assert metadata_entry["metadata"]["input_file"] == str(jsonl_path)
    assert metadata_entry["metadata"].get("input_sources") == [str(jsonl_path)]
    snapshot_entry = json.loads(lines[1])
    assert snapshot_entry["kind"] == "snapshot"
    assert snapshot_entry["source"] == "jsonl"
    assert snapshot_entry["event"] == "reduce_motion"
    assert snapshot_entry["severity"] == "warning"
    assert snapshot_entry["screen"]["index"] == 1


def test_watch_metrics_stream_decision_log_multiple_sources(tmp_path, capsys):
    first_path = tmp_path / "metrics_a.jsonl"
    first_path.write_text(
        json.dumps({"generated_at": "2024-02-02T00:00:00Z", "notes": {"event": "reduce_motion"}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    second_path = tmp_path / "metrics_b.jsonl"
    second_path.write_text(
        json.dumps({"generated_at": "2024-02-02T00:01:00Z", "notes": {"event": "overlay_budget"}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    decision_log = tmp_path / "audit" / "combined.jsonl"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(first_path),
            "--from-jsonl",
            str(second_path),
            "--decision-log",
            str(decision_log),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reduce_motion" in captured.out
    assert "overlay_budget" in captured.out

    lines = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    metadata_entry = json.loads(lines[0])
    assert metadata_entry["metadata"]["input_file"] == str(first_path)
    assert metadata_entry["metadata"].get("input_files") == [str(first_path), str(second_path)]
    assert metadata_entry["metadata"].get("input_sources") == [str(first_path), str(second_path)]
    events = [json.loads(entry)["event"] for entry in lines[1:]]
    assert {"reduce_motion", "overlay_budget"}.issubset(set(events))


def test_watch_metrics_stream_decision_log_from_stdin(monkeypatch, tmp_path, capsys):
    payload = "\n".join(
        json.dumps({"generated_at": "2024-05-01T00:00:00Z", "notes": {"event": "overlay_budget"}}, ensure_ascii=False)
        for _ in range(2)
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    decision_log = tmp_path / "stdin_audit.jsonl"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            "-",
            "--decision-log",
            str(decision_log),
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out

    lines = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    metadata_entry = json.loads(lines[0])
    assert metadata_entry["metadata"]["input_file"] == "stdin"
    assert metadata_entry["metadata"].get("input_sources") == ["-"]


def test_watch_metrics_stream_time_filters_in_decision_log(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-02-01T00:00:00+00:00",
            "notes": json.dumps({"event": "reduce_motion"}, ensure_ascii=False),
        },
        {
            "generated_at": "2024-02-01T00:00:05+00:00",
            "notes": json.dumps({"event": "overlay_budget"}, ensure_ascii=False),
        },
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")

    decision_log = tmp_path / "audit" / "filtered.jsonl"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--since",
            "2024-02-01T00:00:02Z",
            "--until",
            "2024-02-01T00:00:06Z",
            "--decision-log",
            str(decision_log),
        ]
    )
    capsys.readouterr()
    assert exit_code == 0

    lines = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    metadata_entry = json.loads(lines[0])
    filters = metadata_entry["metadata"].get("filters", {})
    assert filters.get("since", "").startswith("2024-02-01T00:00:02")
    assert filters.get("until", "").startswith("2024-02-01T00:00:06")
    snapshot_entry = json.loads(lines[1])
    assert snapshot_entry["event"] == "overlay_budget"


def test_watch_metrics_stream_summary_from_jsonl(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-02-01T00:00:00+00:00",
            "fps": 57.5,
            "notes": {
                "event": "reduce_motion",
                "active": True,
                "severity": "warning",
                "screen": {"index": 1, "name": "Monitor A", "refresh_hz": 60},
            },
        },
        {
            "generated_at": "2024-02-01T00:00:01+00:00",
            "fps": 61.2,
            "notes": {
                "event": "overlay_budget",
                "budget_pct": 0.42,
                "severity": "critical",
                "screen": {"index": 2, "name": "Monitor B", "refresh_hz": 144},
            },
        },
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--summary",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    summary_payload = json.loads(lines[-1])
    assert summary_payload["summary"]["total_snapshots"] == 2
    assert summary_payload["summary"]["severity_counts"] == {
        "critical": 1,
        "warning": 1,
    }
    reduce_motion = summary_payload["summary"]["events"]["reduce_motion"]
    assert reduce_motion["count"] == 1
    assert reduce_motion["fps"]["avg"] == pytest.approx(57.5)
    assert reduce_motion["screens"][0]["index"] == 1
    assert reduce_motion["severity"]["counts"] == {"warning": 1}
    overlay_budget = summary_payload["summary"]["events"]["overlay_budget"]
    assert overlay_budget["fps"]["max"] == pytest.approx(61.2)
    assert overlay_budget["severity"]["counts"] == {"critical": 1}


def test_watch_metrics_stream_summary_from_env(monkeypatch, tmp_path, capsys):
    record = {
        "generated_at": "2024-03-05T12:00:00+00:00",
        "fps": 59.9,
        "notes": {
            "event": "jank_alert",
            "severity": "info",
            "screen": {"index": 0, "name": "Laptop"},
        },
    }
    jsonl_path = tmp_path / "env_metrics.jsonl"
    jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setenv("BOT_CORE_WATCH_METRICS_FROM_JSONL", str(jsonl_path))
    monkeypatch.setenv("BOT_CORE_WATCH_METRICS_SUMMARY", "true")

    exit_code = watch_metrics_main(["--format", "json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    summary_payload = json.loads(lines[-1])
    assert summary_payload["summary"]["total_snapshots"] == 1
    assert "jank_alert" in summary_payload["summary"]["events"]
    assert summary_payload["summary"]["severity_counts"] == {"info": 1}


def test_watch_metrics_stream_summary_output_file(tmp_path, capsys):
    record = {
        "generated_at": "2024-04-01T08:00:00+00:00",
        "fps": 60.0,
        "notes": {
            "event": "reduce_motion",
            "severity": "warning",
            "screen": {"index": 2, "name": "Panel"},
        },
    }
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    output_path = tmp_path / "summary.json"
    exit_code = watch_metrics_main(
        ["--from-jsonl", str(jsonl_path), "--summary-output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert lines  # powinien pojawić się co najmniej jeden snapshot
    assert all("summary" not in line for line in lines)

    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload["summary"]["total_snapshots"] == 1
    assert file_payload["summary"]["events"]["reduce_motion"]["count"] == 1
    assert file_payload["summary"]["severity_counts"] == {"warning": 1}


def test_watch_metrics_stream_summary_output_env(monkeypatch, tmp_path, capsys):
    record = {
        "generated_at": "2024-04-02T09:30:00+00:00",
        "fps": 58.5,
        "notes": {
            "event": "overlay_budget",
            "severity": "critical",
            "screen": {"index": 1, "name": "Desk"},
        },
    }
    jsonl_path = tmp_path / "env_metrics.jsonl"
    jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    output_path = tmp_path / "env_summary.json"
    monkeypatch.setenv(f"{_ENV_PREFIX}FROM_JSONL", str(jsonl_path))
    monkeypatch.setenv(f"{_ENV_PREFIX}SUMMARY_OUTPUT", str(output_path))

    exit_code = watch_metrics_main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert lines  # snapshot powinien zostać wypisany
    assert all("summary" not in line for line in lines)

    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload["summary"]["events"]["overlay_budget"]["count"] == 1
    assert file_payload["summary"]["severity_counts"] == {"critical": 1}


def test_watch_metrics_stream_core_config_summary_metadata(tmp_path, capsys):
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="error")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path, profile_name="ops")

    records = [
        {
            "generated_at": "2024-04-01T12:00:00Z",
            "notes": json.dumps(
                {
                    "event": "reduce_motion",
                    "severity": "error",
                    "screen": {"name": "Desk", "index": 0},
                },
                ensure_ascii=False,
            ),
            "fps": 44.0,
        }
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")

    summary_path = tmp_path / "summary.json"
    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--summary",
            "--summary-output",
            str(summary_path),
            "--core-config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = summary_payload["metadata"]
    assert metadata["core_config"]["path"] == str(config_path)
    assert metadata["core_config"]["metrics_service"]["risk_profile"] == "ops"
    assert metadata["core_config"]["metrics_service"]["enabled"] is True
    assert metadata["core_config"]["risk_service"]["auth_token_scope_required"] == "risk.read"
    assert metadata["core_config"]["risk_service"]["required_scopes"] == {
        "risk.read": ["core_config.risk_service"]
    }
    assert metadata["risk_service"]["auth_token_scope_required"] == "risk.read"
    assert metadata["risk_profile"]["source"] == "core_config"
    assert metadata["risk_profile_summary"]["name"] == "ops"
    assert summary_payload["summary"]["events"]["reduce_motion"]["count"] == 1
    assert summary_payload["summary"]["severity_counts"] == {"error": 1}


def test_apply_core_config_defaults_uses_rbac_token(monkeypatch, tmp_path):
    profiles_path = _write_risk_profile_file(tmp_path)
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_rbac_tokens=[
            {
                "token_id": "metrics-reader",
                "token_env": "METRICS_STREAM_TOKEN",
                "scopes": ["metrics.read"],
            }
        ],
    )
    monkeypatch.setenv("METRICS_STREAM_TOKEN", "rbac-secret")

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    provided = {"--core-config"}

    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags=provided,
    )

    assert args.auth_token == "rbac-secret"
    metadata = getattr(args, "_core_config_metadata")
    metrics_meta = metadata["metrics_service"]
    assert metrics_meta["enabled"] is True
    assert metrics_meta["rbac_tokens"] == 1
    assert metrics_meta["auth_token_source"] == "rbac_token"
    assert metrics_meta["auth_token_scope_required"] == "metrics.read"
    assert metrics_meta["auth_token_scope_checked"] is True
    assert metrics_meta["auth_token_scope_match"] is True
    assert metrics_meta["auth_token_scopes"] == ["metrics.read"]
    assert metrics_meta["auth_token_token_id"] == "metrics-reader"
    assert "auth_token_scope_warning" not in metrics_meta
    assert "auth_token_scope_reason" not in metrics_meta

    risk_meta = metadata["risk_service"]
    assert risk_meta["auth_token_scope_required"] == "risk.read"
    assert risk_meta["required_scopes"] == {"risk.read": ["core_config.risk_service"]}


def test_apply_core_config_defaults_records_risk_rbac_token(monkeypatch, tmp_path):
    profiles_path = _write_risk_profile_file(tmp_path)
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        risk_rbac_tokens=[
            {
                "token_id": "risk-reader",
                "token_value": "risk-secret",
                "scopes": ["risk.read"],
            }
        ],
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    provided = {"--core-config"}

    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags=provided,
    )

    metadata = getattr(args, "_core_config_metadata")
    risk_meta = metadata["risk_service"]
    assert risk_meta["rbac_tokens"] == 1
    assert risk_meta["auth_token_scope_required"] == "risk.read"
    assert risk_meta["required_scopes"] == {"risk.read": ["core_config.risk_service"]}
    assert risk_meta["auth_token_scope_checked"] is True
    assert risk_meta["auth_token_scope_match"] is True
    assert risk_meta["auth_token_token_id"] == "risk-reader"
    assert risk_meta["auth_token_scopes"] == ["risk.read"]


def test_apply_core_config_defaults_records_offline_environments(tmp_path):
    profiles_path = _write_risk_profile_file(tmp_path)
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        offline_environments=("paper_offline", "audit_only"),
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    provided = {"--core-config"}

    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags=provided,
    )

    metadata = getattr(args, "_core_config_metadata")
    offline_meta = metadata.get("offline_environments")
    assert offline_meta == ["audit_only", "paper_offline"]
    assert getattr(args, "_offline_environments") == ("audit_only", "paper_offline")
    metrics_meta = metadata["metrics_service"]
    assert metrics_meta["enabled"] is True


def test_watch_metrics_stream_metrics_disabled_requires_artifact(tmp_path, capsys, caplog):
    profiles_path = _write_risk_profile_file(tmp_path)
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_enabled=False,
        offline_environments=("paper_offline",),
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    provided = {"--core-config"}
    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags=provided,
    )

    metadata = getattr(args, "_core_config_metadata")
    metrics_meta = metadata["metrics_service"]
    assert metrics_meta["enabled"] is False
    assert getattr(args, "_metrics_service_enabled") is False

    caplog.set_level(logging.ERROR, logger="bot_core.scripts.watch_metrics_stream")

    exit_code = watch_metrics_main(["--core-config", str(config_path)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == ""
    assert "MetricsService jest wyłączony" in caplog.text
    assert "paper_offline" in caplog.text


def test_watch_metrics_stream_signed_summary(tmp_path, capsys):
    record = {
        "generated_at": "2024-05-05T10:00:00+00:00",
        "fps": 55.0,
        "notes": {
            "event": "reduce_motion",
            "severity": "warning",
            "screen": {"index": 3, "name": "Ops Desk"},
        },
    }
    jsonl_path = tmp_path / "signed.jsonl"
    jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    output_path = tmp_path / "signed_summary.json"
    key = "supersecret"
    key_id = "ops-key"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--summary",
            "--summary-output",
            str(output_path),
            "--decision-log-hmac-key",
            key,
            "--decision-log-key-id",
            key_id,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0

    lines = [line for line in captured.out.splitlines() if line.strip()]
    summary_payload = json.loads(lines[-1])
    assert "signature" in summary_payload
    assert summary_payload["signature"]["algorithm"] == "HMAC-SHA256"
    assert summary_payload["signature"].get("key_id") == key_id

    body = json.dumps(
        {"summary": summary_payload["summary"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_digest = hmac.new(key.encode("utf-8"), body, hashlib.sha256).digest()
    assert summary_payload["signature"]["value"] == base64.b64encode(expected_digest).decode("ascii")

    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload == summary_payload


def test_watch_metrics_stream_summary_signature_metadata(tmp_path, capsys):
    records = [
        {
            "generated_at": "2024-05-06T10:00:00+00:00",
            "fps": 62.0,
            "notes": {
                "event": "overlay_budget",
                "severity": "error",
                "screen": {"index": 1, "name": "Wall"},
            },
        }
    ]
    jsonl_path = tmp_path / "meta.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    decision_log = tmp_path / "audit" / "events.jsonl"
    summary_path = tmp_path / "summary.json"
    key = "ops-secret"
    key_id = "ops-key"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--summary",
            "--summary-output",
            str(summary_path),
            "--decision-log",
            str(decision_log),
            "--decision-log-hmac-key",
            key,
            "--decision-log-key-id",
            key_id,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out

    entries = [line for line in decision_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata_entry = json.loads(entries[0])
    summary_info = metadata_entry["metadata"].get("summary_signature")
    assert summary_info == {"algorithm": "HMAC-SHA256", "key_id": key_id}

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["signature"]["algorithm"] == "HMAC-SHA256"
    assert summary_payload["signature"].get("key_id") == key_id


def test_create_metrics_channel_insecure():
    grpc_module = _DummyGrpc()
    channel = create_metrics_channel(
        grpc_module,
        "127.0.0.1:50061",
        use_tls=False,
        root_cert=None,
        client_cert=None,
        client_key=None,
        server_name=None,
        server_sha256=None,
    )
    assert channel == "insecure:127.0.0.1:50061"
    assert grpc_module.insecure_calls == ["127.0.0.1:50061"]
    assert not grpc_module.secure_calls


def test_create_metrics_channel_tls_with_client_cert(tmp_path):
    grpc_module = _DummyGrpc()
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(b"root")
    cert_path = tmp_path / "client.pem"
    cert_path.write_bytes(b"cert")
    key_path = tmp_path / "client.key"
    key_path.write_bytes(b"key")

    expected_hash = hashlib.sha256(b"root").hexdigest()

    channel = create_metrics_channel(
        grpc_module,
        "metrics.example:50061",
        use_tls=True,
        root_cert=str(root_path),
        client_cert=str(cert_path),
        client_key=str(key_path),
        server_name="metrics.internal",
        server_sha256=expected_hash,
    )

    assert channel == "secure:metrics.example:50061"
    assert grpc_module.secure_calls
    address, credentials, options = grpc_module.secure_calls[0]
    assert address == "metrics.example:50061"
    assert options == (("grpc.ssl_target_name_override", "metrics.internal"),)
    assert grpc_module.ssl_args == [(b"root", b"key", b"cert")]
    assert credentials == {"root": b"root", "key": b"key", "chain": b"cert"}


def test_create_metrics_channel_tls_requires_key_pair(tmp_path):
    grpc_module = _DummyGrpc()
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(b"root")

    with pytest.raises(SystemExit) as exc:
        create_metrics_channel(
            grpc_module,
            "127.0.0.1:50061",
            use_tls=True,
            root_cert=str(root_path),
            client_cert=str(root_path),
            client_key=None,
            server_name=None,
            server_sha256=hashlib.sha256(b"root").hexdigest(),
        )
    assert exc.value.code == 2


def test_create_metrics_channel_tls_pin_mismatch(tmp_path):
    grpc_module = _DummyGrpc()
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(b"root")

    with pytest.raises(SystemExit) as exc:
        create_metrics_channel(
            grpc_module,
            "127.0.0.1:50061",
            use_tls=True,
            root_cert=str(root_path),
            client_cert=None,
            client_key=None,
            server_name=None,
            server_sha256="deadbeef",
        )
    assert exc.value.code == 2


def test_environment_overrides_enable_tls(monkeypatch, tmp_path):
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(b"root")
    monkeypatch.setenv(f"{_ENV_PREFIX}ROOT_CERT", str(root_path))
    monkeypatch.setenv(f"{_ENV_PREFIX}SERVER_SHA256", hashlib.sha256(b"root").hexdigest())
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)

    recorded_kwargs: dict[str, object] = {}

    def _fake_create(grpc_module, address, **kwargs):
        recorded_kwargs.update(kwargs)
        return "channel"

    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", _fake_create)

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert recorded_kwargs["use_tls"] is True
    assert recorded_kwargs["root_cert"] == str(root_path)
    assert stub.calls
    request, _timeout, metadata = stub.calls[0]
    assert request.include_ui_metrics is True
    assert metadata is None


def test_environment_requires_use_tls_when_disabled(monkeypatch, tmp_path):
    root_path = tmp_path / "root.pem"
    root_path.write_text("root")
    monkeypatch.setenv(f"{_ENV_PREFIX}ROOT_CERT", str(root_path))
    monkeypatch.setenv(f"{_ENV_PREFIX}USE_TLS", "0")

    with pytest.raises(SystemExit) as excinfo:
        watch_metrics_main([])

    assert excinfo.value.code == 2


def test_environment_auth_token_file(monkeypatch, tmp_path):
    token_path = tmp_path / "token.txt"
    token_path.write_text("bearer-token\n")
    monkeypatch.setenv(f"{_ENV_PREFIX}AUTH_TOKEN_FILE", str(token_path))
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("authorization", "Bearer bearer-token")]


def test_environment_auth_token(monkeypatch):
    monkeypatch.setenv(f"{_ENV_PREFIX}AUTH_TOKEN", "env-token")
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("authorization", "Bearer env-token")]


def test_watch_metrics_stream_custom_headers_cli(monkeypatch):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([
        "--header",
        "x-trace=abc123",
        "--header",
        "x-user:Ops",
    ])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("x-trace", "abc123"), ("x-user", "Ops")]


def test_environment_headers_override(monkeypatch):
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS", "x-trace=abc; x-team = ops ")
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("x-trace", "abc"), ("x-team", "ops")]


def test_environment_headers_multiline(monkeypatch):
    multiline = "\n".join(
        [
            " x-trace = abc ",
            "x-team=ops;",  # separator średnika na końcu
            " x-role = qa",
            "",  # pusta linia powinna zostać pominięta
        ]
    )
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS", multiline)
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("x-trace", "abc"),
        ("x-team", "ops"),
        ("x-role", "qa"),
    ]


def test_environment_headers_multiline_comments(monkeypatch):
    multiline = "\n".join(
        [
            "x-trace=abc",
            "   # komentarz powinien zostać pominięty",
            "x-team=ops",
        ]
    )
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS", multiline)
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("x-trace", "abc"),
        ("x-team", "ops"),
    ]


def test_environment_headers_file_list(monkeypatch, tmp_path):
    file_one = tmp_path / "headers_1.env"
    file_one.write_text("x-trace=from-file-one\n", encoding="utf-8")
    file_two = tmp_path / "headers_2.env"
    file_two.write_text("x-team=ops\nx-trace=override\n", encoding="utf-8")
    file_three = tmp_path / "headers_3.env"
    file_three.write_text("x-role=qa\n", encoding="utf-8")

    env_value = "\n".join([str(file_one), "# komentarz", str(file_two), str(file_three)])
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS_FILE", env_value)

    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("x-team", "ops"),
        ("x-trace", "override"),
        ("x-role", "qa"),
    ]


def test_environment_headers_file_none(monkeypatch):
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS_FILE", "NONE")
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata is None


def test_headers_file_basic(monkeypatch, tmp_path):
    headers_file = tmp_path / "headers.env"
    headers_file.write_text(
        "\n".join(
            [
                "  # komentarz nagłówków",
                "x-trace=from-file",
                "x-team = ops ; x-role = qa",
            ]
        ),
        encoding="utf-8",
    )

    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main(["--headers-file", str(headers_file)])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("x-trace", "from-file"),
        ("x-team", "ops"),
        ("x-role", "qa"),
    ]


def test_headers_file_cli_override(monkeypatch, tmp_path):
    headers_file = tmp_path / "headers.env"
    headers_file.write_text("authorization=Bearer from-file", encoding="utf-8")

    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main(
        [
            "--headers-file",
            str(headers_file),
            "--header",
            "authorization=Bearer override",
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("authorization", "Bearer override")]


def test_headers_dir_basic(monkeypatch, tmp_path):
    headers_dir = tmp_path / "headers"
    headers_dir.mkdir()
    (headers_dir / "01.env").write_text("x-team=ops\n", encoding="utf-8")
    (headers_dir / "02.env").write_text("x-trace=dir\nx-role=qa\n", encoding="utf-8")

    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main(["--headers-dir", str(headers_dir)])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("x-team", "ops"),
        ("x-trace", "dir"),
        ("x-role", "qa"),
    ]


def test_environment_headers_dirs(monkeypatch, tmp_path):
    dir_one = tmp_path / "d1"
    dir_one.mkdir()
    (dir_one / "a.env").write_text("x-trace=first\n", encoding="utf-8")
    dir_two = tmp_path / "d2"
    dir_two.mkdir()
    (dir_two / "b.env").write_text("x-trace=second\nx-role=ops\n", encoding="utf-8")

    env_value = "\n".join([str(dir_one), str(dir_two)])
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS_DIRS", env_value)

    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("x-trace", "second"),
        ("x-role", "ops"),
    ]


def test_environment_headers_dirs_none(monkeypatch):
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS_DIRS", "NONE")
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata is None


def test_environment_headers_none(monkeypatch):
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS", "NONE")
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    exit_code = watch_metrics_main([])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata is None


def test_core_config_grpc_metadata_applied(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata=[("x-trace", "config"), ("x-role", "ops")],
    )

    exit_code = watch_metrics_main(["--core-config", str(config_path)])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("x-trace", "config"), ("x-role", "ops")]


def test_core_config_grpc_metadata_files_applied(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    headers_file = tmp_path / "metrics_headers.env"
    headers_file.write_text(
        "authorization=Bearer from file\n"
        "x-trace=from-file\n",
        encoding="utf-8",
    )
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata=[("x-role", "inline")],
        metrics_grpc_metadata_files=[str(headers_file)],
    )

    exit_code = watch_metrics_main(["--core-config", str(config_path)])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("authorization", "Bearer from file"),
        ("x-trace", "from-file"),
        ("x-role", "inline"),
    ]


def test_core_config_grpc_metadata_sources_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("BOT_CORE_TRACE", "trace-token")
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata=[{"key": "authorization", "value_env": "BOT_CORE_TRACE"}],
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags={"--core-config"},
    )

    metrics_meta = getattr(args, "_core_config_metadata")["metrics_service"]
    assert metrics_meta["grpc_metadata_keys"] == ["authorization"]
    assert metrics_meta["grpc_metadata_sources"] == {"authorization": "env:BOT_CORE_TRACE"}


def test_core_config_grpc_metadata_files_sources_recorded(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    headers_file = tmp_path / "metrics_headers.env"
    headers_file.write_text(
        "authorization=Bearer from file\n",
        encoding="utf-8",
    )
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={"x-trace": "config"},
        metrics_grpc_metadata_files=[str(headers_file)],
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags={"--core-config"},
    )
    watch_metrics_module._finalize_custom_metadata(args)

    metrics_meta = getattr(args, "_core_config_metadata")["metrics_service"]
    resolved_path = str(headers_file)
    assert metrics_meta["grpc_metadata_files"] == [resolved_path]
    assert metrics_meta["grpc_metadata_keys"] == ["authorization", "x-trace"]
    assert metrics_meta["grpc_metadata_sources"] == {
        "authorization": f"config-file:{resolved_path}",
        "x-trace": "inline",
    }


def test_core_config_grpc_metadata_files_remove(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    headers_file = tmp_path / "metrics_headers.env"
    headers_file.write_text("authorization=NONE\n", encoding="utf-8")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={
            "authorization": "Bearer inline",
            "x-trace": "config",
        },
        metrics_grpc_metadata_files=[str(headers_file)],
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags={"--core-config"},
    )
    watch_metrics_module._finalize_custom_metadata(args)

    metrics_meta = getattr(args, "_core_config_metadata")["metrics_service"]
    resolved_path = str(headers_file)
    assert metrics_meta["grpc_metadata_keys"] == ["x-trace"]
    assert metrics_meta["grpc_metadata_removed"] == ["authorization"]
    assert metrics_meta["grpc_metadata_removed_sources"] == {
        "authorization": f"config-file:{resolved_path}",
    }


def test_core_config_grpc_metadata_directories_applied(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")

    headers_dir = tmp_path / "headers_dir"
    headers_dir.mkdir()
    file_one = headers_dir / "01.env"
    file_one.write_text("authorization=Bearer dir\n", encoding="utf-8")
    file_two = headers_dir / "02.env"
    file_two.write_text("x-trace=from-dir\n", encoding="utf-8")

    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata=[("x-role", "inline")],
        metrics_grpc_metadata_directories=[str(headers_dir)],
    )

    exit_code = watch_metrics_main(["--core-config", str(config_path)])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [
        ("authorization", "Bearer dir"),
        ("x-trace", "from-dir"),
        ("x-role", "inline"),
    ]


def test_core_config_grpc_metadata_directories_sources_recorded(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")

    headers_dir = tmp_path / "headers_dir"
    headers_dir.mkdir()
    headers_file = headers_dir / "auth.env"
    headers_file.write_text("authorization=Bearer dir\n", encoding="utf-8")

    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata_directories=[str(headers_dir)],
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags={"--core-config"},
    )
    watch_metrics_module._finalize_custom_metadata(args)

    metrics_meta = getattr(args, "_core_config_metadata")["metrics_service"]
    resolved_dir = str(headers_dir)
    resolved_file = str(headers_file)
    assert metrics_meta["grpc_metadata_directories"] == [resolved_dir]
    assert metrics_meta["grpc_metadata_directory_files"][resolved_dir] == [resolved_file]
    assert metrics_meta["grpc_metadata_directory_files_total"] == 1
    assert metrics_meta["grpc_metadata_sources"]["authorization"].startswith(
        f"config-dir:{resolved_file}"
    )


def test_core_config_grpc_metadata_directories_remove(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")

    headers_dir = tmp_path / "headers_dir"
    headers_dir.mkdir()
    removal_file = headers_dir / "remove.env"
    removal_file.write_text("authorization=NONE\n", encoding="utf-8")

    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={
            "authorization": "Bearer inline",
            "x-trace": "config",
        },
        metrics_grpc_metadata_directories=[str(headers_dir)],
    )

    parser = watch_metrics_module.build_arg_parser()
    args = parser.parse_args(["--core-config", str(config_path)])
    watch_metrics_module._apply_core_config_defaults(
        args,
        parser=parser,
        provided_flags={"--core-config"},
    )
    watch_metrics_module._finalize_custom_metadata(args)

    metrics_meta = getattr(args, "_core_config_metadata")["metrics_service"]
    resolved_path = str(removal_file)
    assert metrics_meta["grpc_metadata_keys"] == ["x-trace"]
    assert metrics_meta["grpc_metadata_removed"] == ["authorization"]
    assert metrics_meta["grpc_metadata_removed_sources"] == {
        "authorization": f"config-dir:{resolved_path}",
    }


def test_core_config_grpc_metadata_disabled_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS", "NONE")
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={"x-trace": "config"},
    )

    exit_code = watch_metrics_main(["--core-config", str(config_path)])

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata is None


def test_core_config_grpc_metadata_cli_override(monkeypatch, tmp_path):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={"x-trace": "config", "x-role": "config"},
    )

    exit_code = watch_metrics_main(
        ["--core-config", str(config_path), "--header", "x-trace=cli"]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("x-role", "config"), ("x-trace", "cli")]


def test_core_config_grpc_metadata_sources_cli_override_logged(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={"x-trace": "config", "x-role": "config"},
    )
    decision_log = tmp_path / "logs" / "metrics.jsonl"

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            "x-trace=cli",
            "--decision-log",
            str(decision_log),
        ]
    )

    assert exit_code == 0
    entries = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 1
    metadata_entry = json.loads(entries[0])
    assert metadata_entry["kind"] == "metadata"
    custom_meta = metadata_entry["metadata"].get("custom_metadata")
    assert custom_meta is not None
    assert custom_meta["sources"] == {
        "x-role": "inline",
        "x-trace": "cli:--header",
    }
    core_config_meta = metadata_entry["metadata"].get("core_config")
    assert core_config_meta is not None
    metrics_meta = core_config_meta["metrics_service"]
    assert metrics_meta["grpc_metadata_sources"] == {
        "x-role": "inline",
        "x-trace": "cli:--header",
    }


def test_headers_report_only_outputs_summary(capsys):
    exit_code = watch_metrics_main(
        [
            "--headers-report-only",
            "--header",
            "x-trace=cli",
            "--header",
            "auth-token=sekret",
            "--header",
            "x-old=value",
            "--header",
            "x-old=NONE",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["headers_disabled"] is False
    assert report["headers_count"] == 2
    keys = [item["key"] for item in report["headers"]]
    assert keys == ["x-trace", "auth-token"]
    first = report["headers"][0]
    second = report["headers"][1]
    assert first["value_preview"] == "cli"
    assert first.get("source") == "cli:--header"
    assert second["value_preview"] == "<masked>"
    assert second.get("masked") is True
    assert report["removed_count"] == 1
    assert report["removed"] == [{"key": "x-old", "source": "cli:--header"}]


def test_headers_report_env_toggle(monkeypatch, capsys):
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS_REPORT_ONLY", "1")
    exit_code = watch_metrics_main(["--header", "x-env=1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["headers_count"] == 1
    assert report["headers"][0]["key"] == "x-env"
    assert report["headers"][0]["value_preview"] == "1"
    assert report["headers"][0].get("source") == "cli:--header"


def test_headers_report_only_skips_grpc(monkeypatch, capsys):
    called = False

    def _fail_load():
        nonlocal called
        called = True
        raise AssertionError("gRPC components should not load when --headers-report-only is used")

    monkeypatch.setattr(watch_metrics_module, "_load_grpc_components", _fail_load)
    exit_code = watch_metrics_main(["--headers-report-only"])
    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["headers_count"] == 0
    assert called is False


def test_cli_header_file_reference(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)
    token_path = tmp_path / "header_token.txt"
    token_path.write_text("Bearer from file\n", encoding="utf-8")
    decision_log = tmp_path / "logs" / "headers_file.jsonl"
    decision_log.parent.mkdir(parents=True, exist_ok=True)

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            f"authorization=@file:{token_path}",
            "--decision-log",
            str(decision_log),
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("authorization", "Bearer from file")]
    entries = [line for line in decision_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    metadata_entry = json.loads(entries[0])
    assert metadata_entry["kind"] == "metadata"
    custom_meta = metadata_entry["metadata"].get("custom_metadata")
    assert custom_meta is not None
    expected_source = f"cli:--header@file:{token_path}".replace("\\", "/")
    # Porównujemy po normalizacji separatorów, bo Windows używa `\\`.
    normalized_sources = {key: value.replace("\\", "/") for key, value in custom_meta["sources"].items()}
    assert normalized_sources == {"authorization": expected_source}


def test_cli_header_env_reference(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    monkeypatch.setenv("WATCH_HEADER_TOKEN", "env-token")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)
    decision_log = tmp_path / "logs" / "headers_env.jsonl"
    decision_log.parent.mkdir(parents=True, exist_ok=True)

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            "authorization=@env:WATCH_HEADER_TOKEN",
            "--decision-log",
            str(decision_log),
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("authorization", "env-token")]
    entries = [line for line in decision_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    metadata_entry = json.loads(entries[0])
    custom_meta = metadata_entry["metadata"].get("custom_metadata")
    assert custom_meta is not None
    assert custom_meta["sources"] == {"authorization": "cli:--header@env:WATCH_HEADER_TOKEN"}


def test_headers_file_sources_logged(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    headers_file = tmp_path / "headers.env"
    headers_file.write_text("authorization=Bearer file", encoding="utf-8")
    decision_log = tmp_path / "logs" / "headers_file_sources.jsonl"
    decision_log.parent.mkdir(parents=True, exist_ok=True)

    exit_code = watch_metrics_main(
        [
            "--headers-file",
            str(headers_file),
            "--decision-log",
            str(decision_log),
        ]
    )

    assert exit_code == 0
    entries = [line for line in decision_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    metadata_entry = json.loads(entries[0])
    custom_meta = metadata_entry["metadata"].get("custom_metadata")
    assert custom_meta is not None
    normalized_sources = {key: value.replace("\\", "/") for key, value in custom_meta["sources"].items()}
    expected_source = f"file:{headers_file}".replace("\\", "/")
    assert normalized_sources == {"authorization": expected_source}


def test_cli_header_inline_base64_for_binary(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            "x-binary-bin=AQIDBA==",
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("x-binary-bin", b"\x01\x02\x03\x04")]


def test_cli_header_file_base64_binary(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)
    payload = base64.b64encode(b"\xde\xad\xbe\xef").decode("ascii")
    encoded_path = tmp_path / "binary_header.b64"
    encoded_path.write_text(payload, encoding="utf-8")

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            f"trace-bin=@file64:{encoded_path}",
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("trace-bin", b"\xde\xad\xbe\xef")]


def test_cli_header_env_base64_binary(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    monkeypatch.setenv("TRACE_TOKEN_B64", base64.b64encode(b"token-bytes").decode("ascii"))
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            "trace-bin=@env64:TRACE_TOKEN_B64",
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("trace-bin", b"token-bytes")]


def test_cli_header_file64_decodes_utf8_for_text(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)
    encoded_value = base64.b64encode("Bearer secret".encode("utf-8")).decode("ascii")
    encoded_path = tmp_path / "auth_header.b64"
    encoded_path.write_text(encoded_value + "\n", encoding="utf-8")

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            f"authorization=@file64:{encoded_path}",
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("authorization", "Bearer secret")]


def test_cli_header_invalid_base64_raises(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)

    with pytest.raises(SystemExit):
        watch_metrics_main(
            [
                "--core-config",
                str(config_path),
                "--header",
                "trace-bin=invalid-base64",
            ]
        )


def test_cli_header_literal_escape(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path)

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            "x-trace=@@observability",
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("x-trace", "@observability")]
def test_core_config_grpc_metadata_cli_remove(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={"authorization": "Bearer config", "x-trace": "cfg"},
    )
    decision_log = tmp_path / "logs" / "metrics.jsonl"

    exit_code = watch_metrics_main(
        [
            "--core-config",
            str(config_path),
            "--header",
            "authorization=NONE",
            "--decision-log",
            str(decision_log),
        ]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata == [("x-trace", "cfg")]

    entries = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 1
    metadata_entry = json.loads(entries[0])
    metrics_meta = metadata_entry["metadata"]["core_config"]["metrics_service"]
    assert metrics_meta["grpc_metadata_keys"] == ["x-trace"]
    assert metrics_meta["grpc_metadata_sources"] == {"x-trace": "inline"}
    assert metrics_meta["grpc_metadata_removed"] == ["authorization"]
    assert metrics_meta["grpc_metadata_removed_sources"] == {
        "authorization": watch_metrics_module._CLI_HEADER_SOURCE
    }


def test_environment_headers_remove_config_key(monkeypatch, tmp_path):
    stub = _StubCollector()
    stub.response = []
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="warning")
    config_path = _write_core_config(
        tmp_path,
        profiles_path=profiles_path,
        metrics_grpc_metadata={"authorization": "cfg-token"},
    )
    decision_log = tmp_path / "logs" / "metrics.jsonl"
    monkeypatch.setenv(f"{_ENV_PREFIX}HEADERS", "authorization=NONE")

    exit_code = watch_metrics_main(
        ["--core-config", str(config_path), "--decision-log", str(decision_log)]
    )

    assert exit_code == 0
    assert stub.calls
    _request, _timeout, metadata = stub.calls[0]
    assert metadata is None

    entries = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 1
    metadata_entry = json.loads(entries[0])
    metrics_meta = metadata_entry["metadata"]["core_config"]["metrics_service"]
    assert metrics_meta["grpc_metadata_enabled"] is False
    assert metrics_meta["grpc_metadata_removed"] == ["authorization"]
    assert metrics_meta["grpc_metadata_removed_sources"] == {
        "authorization": watch_metrics_module._ENV_HEADER_SOURCE
    }


def test_watch_metrics_stream_header_invalid_format(monkeypatch):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    with pytest.raises(SystemExit):
        watch_metrics_main(["--header", "invalid"])


def test_watch_metrics_stream_header_invalid_key(monkeypatch):
    stub = _StubCollector()
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel")

    with pytest.raises(SystemExit):
        watch_metrics_main(["--header", "X-Trace=abc"])


def test_watch_metrics_stream_requires_use_tls_for_tls_flags(tmp_path):
    tls_file = tmp_path / "root.pem"
    tls_file.write_text("root")

    with pytest.raises(SystemExit) as exc:
        watch_metrics_main([
            "--root-cert",
            str(tls_file),
        ])

    assert exc.value.code == 2


def test_watch_metrics_stream_rejects_negative_screen_index():
    with pytest.raises(SystemExit) as exc:
        watch_metrics_main(["--screen-index", "-1"])

    assert exc.value.code == 2


@pytest.mark.timeout(5)
def test_watch_metrics_stream_table_format(tmp_path, capsys):
    server = _start_server()
    try:
        _append_snapshots(
            server,
            [
                {
                    "event": "overlay_budget",
                    "over": True,
                    "active_overlays": 5,
                    "severity": "warning",
                },
            ],
        )

        host, port = server.address.split(":")
        exit_code = watch_metrics_main(
            [
                "--host",
                host,
                "--port",
                port,
                "--limit",
                "1",
                "--format",
                "table",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "overlay_budget" in captured.out
        assert "severity=warning" in captured.out
    finally:
        server.stop(0)


def test_watch_metrics_stream_filters_screen(monkeypatch, capsys):
    stub = _StubCollector()
    snapshots = [
        _FakeSnapshot(
            {
                "event": "reduce_motion",
                "active": True,
                "severity": "warning",
                "screen": {"index": 0, "name": "Aux Display"},
            }
        ),
        _FakeSnapshot(
            {
                "event": "reduce_motion",
                "active": True,
                "severity": "info",
                "screen": {
                    "index": 1,
                    "name": "Main Display",
                    "geometry_px": {"width": 1920, "height": 1080},
                    "refresh_hz": 60.0,
                },
            }
        ),
    ]
    stub.response = snapshots
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(
        watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel"
    )

    exit_code = watch_metrics_main(["--screen-index", "1", "--limit", "5"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "screen=#1" in captured.out
    assert "screen=#0" not in captured.out

    stub.response = snapshots
    exit_code = watch_metrics_main(["--screen-name", "display", "--limit", "5"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.count("screen=#1") == 1


def test_watch_metrics_stream_filters_severity(monkeypatch, capsys):
    stub = _StubCollector()
    snapshots = [
        _FakeSnapshot({"event": "reduce_motion", "severity": "warning"}),
        _FakeSnapshot({"event": "reduce_motion", "severity": "info"}),
    ]
    stub.response = snapshots
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(
        watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel"
    )

    exit_code = watch_metrics_main(["--severity", "warning", "--limit", "5"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "severity=warning" in captured.out
    assert "severity=info" not in captured.out


def test_watch_metrics_stream_filters_severity_env(monkeypatch, capsys):
    stub = _StubCollector()
    snapshots = [
        _FakeSnapshot({"event": "overlay_budget", "severity": "critical"}),
        _FakeSnapshot({"event": "overlay_budget", "severity": "warning"}),
        _FakeSnapshot({"event": "overlay_budget", "severity": "info"}),
    ]
    stub.response = snapshots
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(
        watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel"
    )
    monkeypatch.setenv(f"{_ENV_PREFIX}SEVERITY", "critical,warning")

    exit_code = watch_metrics_main(["--limit", "5"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "severity=critical" in captured.out
    assert "severity=warning" in captured.out
    assert "severity=info" not in captured.out


def test_watch_metrics_stream_filters_severity_min_cli(monkeypatch, capsys):
    stub = _StubCollector()
    snapshots = [
        _FakeSnapshot({"event": "overlay_budget", "severity": "info"}),
        _FakeSnapshot({"event": "overlay_budget", "severity": "warning"}),
        _FakeSnapshot({"event": "overlay_budget", "severity": "critical"}),
    ]
    stub.response = snapshots
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(
        watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel"
    )

    exit_code = watch_metrics_main(["--severity-min", "warning", "--limit", "5"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "severity=critical" in captured.out
    assert "severity=warning" in captured.out
    assert "severity=info" not in captured.out


def test_watch_metrics_stream_filters_severity_min_env(monkeypatch, capsys):
    stub = _StubCollector()
    snapshots = [
        _FakeSnapshot({"event": "jank", "severity": "notice"}),
        _FakeSnapshot({"event": "jank", "severity": "error"}),
    ]
    stub.response = snapshots
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(
        watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel"
    )
    monkeypatch.setenv(f"{_ENV_PREFIX}SEVERITY_MIN", "error")

    exit_code = watch_metrics_main(["--limit", "5"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "severity=error" in captured.out
    assert "severity=notice" not in captured.out


def test_watch_metrics_stream_rejects_conflicting_severity_filters():
    with pytest.raises(SystemExit) as excinfo:
        watch_metrics_main([
            "--severity",
            "info",
            "--severity-min",
            "warning",
        ])

    assert excinfo.value.code == 2


def test_watch_metrics_stream_risk_profile_defaults(tmp_path):
    records = [
        {
            "generated_at": "2024-03-01T10:00:00Z",
            "notes": json.dumps(
                {
                    "event": "reduce_motion",
                    "severity": "warning",
                    "screen": {"index": 0, "name": "Primary"},
                },
                ensure_ascii=False,
            ),
        },
        {
            "generated_at": "2024-03-01T10:00:01Z",
            "notes": json.dumps(
                {
                    "event": "reduce_motion",
                    "severity": "info",
                    "screen": {"index": 1, "name": "Side"},
                },
                ensure_ascii=False,
            ),
        },
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")

    decision_log_path = tmp_path / "decision.jsonl"
    summary_path = tmp_path / "summary.json"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--decision-log",
            str(decision_log_path),
            "--decision-log-hmac-key",
            "sekret",
            "--risk-profile",
            "conservative",
            "--summary-output",
            str(summary_path),
        ]
    )
    assert exit_code == 0

    decision_entries = [json.loads(line) for line in decision_log_path.read_text().splitlines() if line]
    assert decision_entries[0]["kind"] == "metadata"
    metadata = decision_entries[0]["metadata"]
    assert metadata["filters"]["severity_min"] == "warning"
    assert metadata["risk_profile"]["name"] == "conservative"
    assert metadata["risk_profile"]["severity_min"] == "warning"
    assert metadata["risk_profile"]["origin"] == "builtin"
    assert metadata["risk_profile_summary"]["name"] == "conservative"
    assert metadata["risk_profile_summary"]["severity_min"] == "warning"

    snapshot_events = [entry for entry in decision_entries if entry["kind"] == "snapshot"]
    assert len(snapshot_events) == 1
    assert snapshot_events[0]["severity"] == "warning"

    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["metadata"]["risk_profile"]["name"] == "conservative"
    assert summary_payload["metadata"]["risk_profile"]["origin"] == "builtin"
    assert (
        summary_payload["metadata"]["risk_profile_summary"]["name"] == "conservative"
    )
    assert (
        summary_payload["metadata"]["risk_profile_summary"]["severity_min"] == "warning"
    )
    assert summary_payload["summary"]["events"]["reduce_motion"]["count"] == 1


def test_watch_metrics_stream_risk_profile_from_file(tmp_path):
    records = [
        {
            "generated_at": "2024-03-02T10:00:00Z",
            "notes": json.dumps(
                {
                    "event": "reduce_motion",
                    "severity": "error",
                    "screen": {"index": 0, "name": "Primary"},
                },
                ensure_ascii=False,
            ),
        },
        {
            "generated_at": "2024-03-02T10:00:01Z",
            "notes": json.dumps(
                {
                    "event": "reduce_motion",
                    "severity": "warning",
                    "screen": {"index": 1, "name": "Backup"},
                },
                ensure_ascii=False,
            ),
        },
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")

    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps({"risk_profiles": {"ops": {"severity_min": "error"}}}, ensure_ascii=False)
    )

    decision_log_path = tmp_path / "ops_decision.jsonl"
    summary_path = tmp_path / "ops_summary.json"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--risk-profiles-file",
            str(profiles_path),
            "--risk-profile",
            "ops",
            "--decision-log",
            str(decision_log_path),
            "--summary-output",
            str(summary_path),
        ]
    )

    assert exit_code == 0

    decision_entries = [json.loads(line) for line in decision_log_path.read_text().splitlines() if line]
    metadata = decision_entries[0]["metadata"]
    assert metadata["risk_profile"]["name"] == "ops"
    assert metadata["risk_profile"]["severity_min"] == "error"
    assert metadata["risk_profile"]["origin"].startswith("watcher:")
    assert metadata["risk_profile_summary"]["name"] == "ops"
    assert metadata["risk_profile_summary"]["severity_min"] == "error"

    snapshot_events = [entry for entry in decision_entries if entry["kind"] == "snapshot"]
    assert len(snapshot_events) == 1
    assert snapshot_events[0]["severity"] == "error"

    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["metadata"]["risk_profile"]["name"] == "ops"
    assert summary_payload["metadata"]["risk_profile"]["origin"].startswith("watcher:")
    assert summary_payload["metadata"]["risk_profile_summary"]["name"] == "ops"
    assert summary_payload["metadata"]["risk_profile_summary"]["severity_min"] == "error"
    assert summary_payload["summary"]["events"]["reduce_motion"]["count"] == 1


def test_watch_metrics_stream_core_config_decision_metadata(tmp_path):
    profiles_path = _write_risk_profile_file(tmp_path, name="ops", severity="error")
    config_path = _write_core_config(tmp_path, profiles_path=profiles_path, profile_name="ops")

    records = [
        {
            "generated_at": "2024-04-05T08:00:00Z",
            "notes": json.dumps(
                {
                    "event": "reduce_motion",
                    "severity": "error",
                    "screen": {"name": "Desk", "index": 0},
                },
                ensure_ascii=False,
            ),
            "fps": 40.5,
        }
    ]
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")

    decision_log_path = tmp_path / "decision.jsonl"
    summary_path = tmp_path / "summary.json"

    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--decision-log",
            str(decision_log_path),
            "--decision-log-hmac-key",
            "sekret",
            "--summary-output",
            str(summary_path),
            "--core-config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    decision_entries = [
        json.loads(line)
        for line in decision_log_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    metadata_entry = decision_entries[0]
    assert metadata_entry["kind"] == "metadata"
    meta = metadata_entry["metadata"]
    assert meta["core_config"]["path"] == str(config_path)
    assert meta["core_config"]["metrics_service"]["risk_profiles_file"] == str(profiles_path)
    assert meta["risk_profile"]["source"] == "core_config"


def test_watch_metrics_stream_decision_log_env(monkeypatch, tmp_path, capsys):
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"notes": {"event": "reduce_motion", "severity": "warning"}}),
                json.dumps({"notes": {"event": "reduce_motion", "severity": "info"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    decision_log = tmp_path / "decision.jsonl"
    monkeypatch.setenv(f"{_ENV_PREFIX}FROM_JSONL", str(jsonl_path))
    monkeypatch.setenv(f"{_ENV_PREFIX}DECISION_LOG", str(decision_log))
    monkeypatch.setenv(f"{_ENV_PREFIX}SEVERITY", "warning")

    exit_code = watch_metrics_main(["--format", "json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1

    log_lines = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(log_lines) == 2
    metadata_entry = json.loads(log_lines[0])
    assert metadata_entry["kind"] == "metadata"
    assert metadata_entry["metadata"]["filters"]["severity"] == ["warning"]
    payload = json.loads(log_lines[1])
    assert payload["kind"] == "snapshot"
    assert payload["severity"] == "warning"


def test_watch_metrics_stream_decision_log_metadata_includes_severity_min(
    monkeypatch, tmp_path, capsys
):
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"notes": {"event": "reduce_motion", "severity": "info"}}),
                json.dumps({"notes": {"event": "reduce_motion", "severity": "critical"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    decision_log = tmp_path / "decision.jsonl"
    exit_code = watch_metrics_main(
        [
            "--from-jsonl",
            str(jsonl_path),
            "--decision-log",
            str(decision_log),
            "--severity-min",
            "warning",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "severity=critical" in captured.out
    assert "severity=info" not in captured.out

    entries = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 2
    metadata_entry = json.loads(entries[0])
    assert metadata_entry["kind"] == "metadata"
    assert metadata_entry["metadata"]["filters"]["severity_min"] == "warning"


def test_watch_metrics_stream_decision_log_grpc(monkeypatch, tmp_path, capsys):
    stub = _StubCollector()
    stub.response = [
        _FakeSnapshot({"event": "reduce_motion", "severity": "warning"}, fps=58.0),
        _FakeSnapshot({"event": "reduce_motion", "severity": "info"}, fps=61.0),
    ]
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(
        watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel"
    )

    decision_log = tmp_path / "logs" / "ui_metrics.jsonl"

    exit_code = watch_metrics_main(
        [
            "--decision-log",
            str(decision_log),
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "severity=warning" in captured.out

    entries = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 2
    metadata_entry = json.loads(entries[0])
    assert metadata_entry["kind"] == "metadata"
    assert metadata_entry["metadata"]["mode"] == "grpc"
    assert metadata_entry["metadata"]["filters"]["limit"] == 1
    payload = json.loads(entries[1])
    assert payload["kind"] == "snapshot"
    assert payload["source"] == "grpc"
    assert payload["event"] == "reduce_motion"
    assert payload["fps"] == pytest.approx(58.0)


def test_watch_metrics_stream_decision_log_signatures_cli(monkeypatch, tmp_path, capsys):
    stub = _StubCollector()
    stub.response = [
        _FakeSnapshot({"event": "overlay_budget", "severity": "critical"}, fps=52.0),
    ]
    _install_dummy_loader(monkeypatch, stub)
    monkeypatch.setattr(
        watch_metrics_module, "create_metrics_channel", lambda *args, **kwargs: "channel"
    )

    decision_log = tmp_path / "signed" / "metrics.jsonl"
    key_material = "ops-secret"
    key_bytes = key_material.encode("utf-8")

    exit_code = watch_metrics_main(
        [
            "--decision-log",
            str(decision_log),
            "--decision-log-hmac-key",
            key_material,
            "--decision-log-key-id",
            "ops-key",
            "--limit",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "overlay_budget" in captured.out

    entries = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 2
    metadata_entry = json.loads(entries[0])
    _assert_signed_entry(metadata_entry, key_bytes, key_id="ops-key")
    assert metadata_entry["metadata"]["signing"]["algorithm"] == "HMAC-SHA256"
    assert metadata_entry["metadata"]["signing"]["key_id"] == "ops-key"

    snapshot_entry = json.loads(entries[1])
    _assert_signed_entry(snapshot_entry, key_bytes, key_id="ops-key")
    assert snapshot_entry["event"] == "overlay_budget"


def test_watch_metrics_stream_decision_log_signatures_env_file(monkeypatch, tmp_path, capsys):
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"notes": {"event": "jank", "severity": "warning"}}),
                json.dumps({"notes": {"event": "jank", "severity": "critical"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    key_file = tmp_path / "decision.key"
    key_file.write_text("file-secret\n", encoding="utf-8")

    decision_log = tmp_path / "decision" / "audit.jsonl"

    monkeypatch.setenv(f"{_ENV_PREFIX}FROM_JSONL", str(jsonl_path))
    monkeypatch.setenv(f"{_ENV_PREFIX}DECISION_LOG", str(decision_log))
    monkeypatch.setenv(f"{_ENV_PREFIX}DECISION_LOG_HMAC_KEY_FILE", str(key_file))

    exit_code = watch_metrics_main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "jank" in captured.out

    entries = [ln for ln in decision_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(entries) == 3
    key_bytes = b"file-secret"

    metadata_entry = json.loads(entries[0])
    _assert_signed_entry(metadata_entry, key_bytes, key_id=None)
    assert metadata_entry["metadata"]["signing"]["algorithm"] == "HMAC-SHA256"

    first_snapshot = json.loads(entries[1])
    second_snapshot = json.loads(entries[2])
    _assert_signed_entry(first_snapshot, key_bytes, key_id=None)
    _assert_signed_entry(second_snapshot, key_bytes, key_id=None)


def test_watch_metrics_stream_decision_log_signing_conflict(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        watch_metrics_main(
            [
                "--decision-log",
                str(tmp_path / "log.jsonl"),
                "--decision-log-hmac-key",
                "alpha",
                "--decision-log-hmac-key-file",
                str(tmp_path / "key.txt"),
                "--from-jsonl",
                str(tmp_path / "metrics.jsonl"),
            ]
        )

    assert excinfo.value.code == 2


def test_watch_metrics_stream_decision_log_signing_empty_key(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("   ", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        watch_metrics_main(
            [
                "--decision-log",
                str(tmp_path / "log.jsonl"),
                "--decision-log-hmac-key-file",
                str(key_file),
                "--from-jsonl",
                str(tmp_path / "metrics.jsonl"),
            ]
        )

    assert excinfo.value.code == 2


@pytest.mark.timeout(5)
def test_watch_metrics_stream_with_auth_token(capsys):
    token = "stream-secret"
    server = _start_server(auth_token=token)
    try:
        _append_snapshots(
            server,
            [
                {"event": "reduce_motion", "active": True, "severity": "warning"},
            ],
        )

        host, port = server.address.split(":")
        exit_code = watch_metrics_main(
            [
                "--host",
                host,
                "--port",
                port,
                "--limit",
                "1",
                "--auth-token",
                token,
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "reduce_motion" in captured.out
    finally:
        server.stop(0)
