from __future__ import annotations

from concurrent import futures
from pathlib import Path

import grpc
import pytest

from bot_core.cloud.config import CloudAllowedClientConfig, CloudSecurityConfig
from bot_core.cloud.security import CloudAuthInterceptor, CloudAuthServicer, CloudSecurityManager
from bot_core.generated import trading_pb2, trading_pb2_grpc
from bot_core.security.fingerprint import sign_license_payload


class _RuntimeStub(trading_pb2_grpc.RuntimeServiceServicer):
    def ListDecisions(self, request, context):  # type: ignore[override]
        del request, context
        return trading_pb2.ListDecisionsResponse(total=0)


@pytest.fixture()
def _cloud_server(tmp_path: Path):
    config = CloudSecurityConfig(
        require_handshake=True,
        session_ttl_seconds=120,
        audit_log_path=tmp_path / "security_admin.log",
        allowed_clients=(
            CloudAllowedClientConfig(
                license_id="LIC-TRIAL-1",
                fingerprint="HW-DEMO-1",
                shared_secret=b"demo-secret",
                note="fixture",
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


def _build_signature(license_id: str, fingerprint: str) -> trading_pb2.CloudAuthSignature:
    payload = {"license_id": license_id, "fingerprint": fingerprint, "nonce": "pytest"}
    signature = sign_license_payload(payload, fingerprint=fingerprint, secret=b"demo-secret")
    return trading_pb2.CloudAuthSignature(
        algorithm=signature.get("algorithm", ""),
        value=signature.get("value", ""),
        key_id=signature.get("key_id", ""),
    )


@pytest.mark.integration
@pytest.mark.requires_trading_stubs
def test_cloud_hwid_enforcement_blocks_unauthorized_clients(_cloud_server):
    manager, address = _cloud_server
    channel = grpc.insecure_channel(address)
    runtime_stub = trading_pb2_grpc.RuntimeServiceStub(channel)

    with pytest.raises(grpc.RpcError) as unauthorized:
        runtime_stub.ListDecisions(trading_pb2.ListDecisionsRequest(limit=1), timeout=1)
    assert unauthorized.value.code() == grpc.StatusCode.UNAUTHENTICATED

    auth_stub = trading_pb2_grpc.CloudAuthServiceStub(channel)
    signature = _build_signature("LIC-TRIAL-1", "HW-DEMO-1")
    response = auth_stub.AuthorizeClient(
        trading_pb2.CloudAuthRequest(
            fingerprint="HW-DEMO-1",
            license_id="LIC-TRIAL-1",
            nonce="pytest",
            signature=signature,
        ),
        timeout=2,
    )
    assert response.authorized is True
    metadata = (("authorization", f"CloudSession {response.session_token}"),)

    runtime_stub.ListDecisions(trading_pb2.ListDecisionsRequest(limit=1), metadata=metadata, timeout=2)

    manager.refresh_allowed_clients(())
    with pytest.raises(grpc.RpcError) as revoked:
        runtime_stub.ListDecisions(
            trading_pb2.ListDecisionsRequest(limit=1),
            metadata=metadata,
            timeout=1,
        )
    assert revoked.value.code() == grpc.StatusCode.UNAUTHENTICATED

