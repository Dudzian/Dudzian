from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import hmac
import json
import io
import sys
import types
from pathlib import Path
from typing import Iterable

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
        self.calls: list[tuple[object, float | None, list[tuple[str, str]] | None]] = []
        self.response: Iterable[object] = ()

    def StreamMetrics(
        self,
        request,
        timeout: float | None = None,
        metadata: list[tuple[str, str]] | None = None,
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
    snapshot_entry = json.loads(lines[1])
    assert snapshot_entry["kind"] == "snapshot"
    assert snapshot_entry["source"] == "jsonl"
    assert snapshot_entry["event"] == "reduce_motion"
    assert snapshot_entry["severity"] == "warning"
    assert snapshot_entry["screen"]["index"] == 1


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

    snapshot_events = [entry for entry in decision_entries if entry["kind"] == "snapshot"]
    assert len(snapshot_events) == 1
    assert snapshot_events[0]["severity"] == "warning"

    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["metadata"]["risk_profile"]["name"] == "conservative"
    assert summary_payload["summary"]["events"]["reduce_motion"]["count"] == 1


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
