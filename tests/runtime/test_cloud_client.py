import json
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest

from bot_core.cloud.config import CloudAllowedClientConfig, CloudSecurityConfig
from bot_core.cloud.security import CloudAuthInterceptor, CloudAuthServicer, CloudSecurityManager
from bot_core.generated import trading_pb2, trading_pb2_grpc
from bot_core.runtime.cloud_client import (
    CloudClientOptions,
    load_cloud_client_options,
    load_license_identity,
    perform_cloud_handshake,
)


pytestmark = pytest.mark.unstable_windows


class _RuntimeStub(trading_pb2_grpc.RuntimeServiceServicer):
    def ListDecisions(self, request, context):  # type: ignore[override]
        del request, context
        return trading_pb2.ListDecisionsResponse(total=0)


@pytest.fixture()
def _auth_server(tmp_path: Path):
    config = CloudSecurityConfig(
        require_handshake=True,
        session_ttl_seconds=30,
        audit_log_path=tmp_path / "security_admin.log",
        allowed_clients=(
            CloudAllowedClientConfig(
                license_id="LIC-TEST-1",
                fingerprint="HW-TEST-1",
                shared_secret=b"demo-secret",
                note="pytest",
            ),
        ),
    )
    manager = CloudSecurityManager(config)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[CloudAuthInterceptor(manager)],
    )
    trading_pb2_grpc.add_RuntimeServiceServicer_to_server(_RuntimeStub(), server)
    trading_pb2_grpc.add_CloudAuthServiceServicer_to_server(CloudAuthServicer(manager), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield manager, f"127.0.0.1:{port}"
    finally:
        server.stop(0)


def test_load_license_identity_reads_file(tmp_path: Path) -> None:
    status_path = tmp_path / "license_status.json"
    status_path.write_text(
        json.dumps({"license_id": "LIC-TEST", "local_hwid": "HW-123", "bundle_path": "demo.lic"}),
        encoding="utf-8",
    )
    identity = load_license_identity(status_path)
    assert identity is not None
    assert identity.license_id == "LIC-TEST"
    assert identity.fingerprint == "HW-123"
    assert identity.source.endswith("demo.lic")


@pytest.mark.parametrize(
    ("metadata_override", "expected_metadata"),
    [
        (None, (("x-default", "1"),)),
        ((), ()),
        ((("x-override", "abc"),), (("x-override", "abc"),)),
    ],
)
def test_perform_cloud_handshake_respects_metadata_override_contract(
    monkeypatch: pytest.MonkeyPatch,
    metadata_override,
    expected_metadata,
) -> None:
    captured: dict[str, object] = {}

    class _DummyChannel:
        def close(self) -> None:
            captured["closed"] = True

    class _Response:
        authorized = True
        message = ""
        session_token = "tok"

        def HasField(self, name: str) -> bool:  # noqa: N802 - protobuf compat
            return False

    class _Stub:
        def __init__(self, channel) -> None:
            captured["channel"] = channel

        def AuthorizeClient(self, request, *, metadata, timeout):
            captured["metadata"] = tuple(metadata)
            captured["timeout"] = timeout
            captured["request"] = request
            return _Response()

    options = CloudClientOptions(
        config_path=Path("config/cloud/client.yaml"),
        client=SimpleNamespace(address="127.0.0.1:50052"),
        metadata=[("x-default", "1")],
        tls_credentials=None,
        authority_override=None,
    )
    identity = SimpleNamespace(license_id="LIC-1", fingerprint="HW-1")

    monkeypatch.setattr("bot_core.runtime.cloud_client._build_channel", lambda *_: _DummyChannel())
    monkeypatch.setattr("bot_core.runtime.cloud_client.sign_license_payload", lambda *_, **__: {
        "algorithm": "HMAC",
        "value": "sig",
        "key_id": "k1",
    })
    monkeypatch.setattr("bot_core.runtime.cloud_client.trading_pb2_grpc.CloudAuthServiceStub", _Stub)

    result = perform_cloud_handshake(
        options,
        identity,
        metadata=metadata_override,
        license_secret=b"secret",
        secret_path=None,
        timeout=1.5,
    )

    assert result.status == "ok"
    assert captured["metadata"] == expected_metadata
    assert captured.get("closed") is True


@pytest.mark.integration
@pytest.mark.requires_trading_stubs
def test_perform_cloud_handshake_success(tmp_path: Path, _auth_server) -> None:
    manager, address = _auth_server
    client_path = tmp_path / "client.yaml"
    client_path.write_text(
        """
address: {address}
use_tls: false
metadata: {{}}
metadata_env: {{}}
metadata_files: {{}}
fallback_entrypoint: cloud-demo
allow_local_fallback: true
auto_connect: true
""".strip().format(address=address),
        encoding="utf-8",
    )
    options = load_cloud_client_options(client_path)
    status_path = tmp_path / "license_status.json"
    status_path.write_text(
        json.dumps({"license_id": "LIC-TEST-1", "local_hwid": "HW-TEST-1"}),
        encoding="utf-8",
    )
    identity = load_license_identity(status_path)
    assert identity is not None
    result = perform_cloud_handshake(
        options,
        identity,
        metadata=(),
        license_secret=b"demo-secret",
        secret_path=None,
        timeout=2.0,
    )
    assert result.status == "ok"
    assert result.session_token
    assert manager.requires_handshake
