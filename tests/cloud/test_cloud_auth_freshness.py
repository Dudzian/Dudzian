"""Odmowy cloud authentication dla żądań bez dowodu świeżości serwera."""

from pathlib import Path

import pytest

from bot_core.cloud.config import CloudAllowedClientConfig, CloudSecurityConfig
from bot_core.cloud.security import CloudAuthorizationError, CloudSecurityManager
from bot_core.generated import trading_pb2
from bot_core.security.fingerprint import sign_license_payload


def _manager(tmp_path: Path) -> CloudSecurityManager:
    return CloudSecurityManager(
        CloudSecurityConfig(
            require_handshake=True,
            audit_log_path=tmp_path / "audit.log",
            allowed_clients=(
                CloudAllowedClientConfig("LIC-TEST", "HW-TEST", b"synthetic-test-secret"),
            ),
        )
    )


def _request(nonce: str) -> trading_pb2.CloudAuthRequest:
    payload = {"license_id": "LIC-TEST", "fingerprint": "HW-TEST"}
    if nonce:
        payload["nonce"] = nonce
    signature = sign_license_payload(
        payload, fingerprint="HW-TEST", secret=b"synthetic-test-secret"
    )
    return trading_pb2.CloudAuthRequest(
        license_id="LIC-TEST",
        fingerprint="HW-TEST",
        nonce=nonce,
        signature=trading_pb2.CloudAuthSignature(**signature),
    )


@pytest.mark.parametrize("nonce", ["", "client-generated-without-server-challenge"])
def test_cloud_auth_refuses_unissued_nonce(tmp_path: Path, nonce: str) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(CloudAuthorizationError):
        manager.authorize(_request(nonce))


def _challenge(manager: CloudSecurityManager) -> str:
    return manager.issue_challenge(
        trading_pb2.CloudChallengeRequest(license_id="LIC-TEST", fingerprint="HW-TEST")
    )


def test_cloud_auth_challenge_is_single_use(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    request = _request(_challenge(manager))
    session = manager.authorize(request)
    assert manager.require_session((("authorization", f"CloudSession {session.token}"),)) == session
    with pytest.raises(CloudAuthorizationError):
        manager.authorize(request)


def test_cloud_auth_challenge_has_short_monotonic_ttl(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    clock = [10.0]
    monkeypatch.setattr("bot_core.cloud.security.time", SimpleNamespace(monotonic=lambda: clock[0]))
    manager = _manager(tmp_path)
    request = _request(_challenge(manager))
    clock[0] = 70.0
    with pytest.raises(CloudAuthorizationError):
        manager.authorize(request)
    assert not manager._sessions


def test_cloud_auth_concurrent_replay_grants_exactly_one_session(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    manager = _manager(tmp_path)
    request = _request(_challenge(manager))
    start = Barrier(8)

    def authorize_once() -> bool:
        start.wait(timeout=5)
        try:
            manager.authorize(request)
        except CloudAuthorizationError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: authorize_once(), range(8)))
    assert results.count(True) == 1
    assert len(manager._sessions) == 1


def test_cloud_auth_restart_and_invalidation_refuse_previous_proofs(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    old_request = _request(_challenge(manager))
    session = manager.authorize(_request(_challenge(manager)))
    restarted = _manager(tmp_path)
    metadata = (("authorization", f"CloudSession {session.token}"),)
    for candidate in (restarted, manager):
        if candidate is manager:
            candidate.invalidate_sessions()
        with pytest.raises(CloudAuthorizationError):
            candidate.authorize(old_request)
        with pytest.raises(CloudAuthorizationError):
            candidate.require_session(metadata)


def test_cloud_auth_invalid_signature_consumes_challenge(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    request = _request(_challenge(manager))
    valid_value = request.signature.value
    request.signature.value = "invalid"
    with pytest.raises(CloudAuthorizationError):
        manager.authorize(request)
    request.signature.value = valid_value
    with pytest.raises(CloudAuthorizationError):
        manager.authorize(request)
    assert not manager._sessions


def test_closed_cloud_manager_cannot_issue_new_proofs(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    pending = _request(_challenge(manager))
    session = manager.authorize(_request(_challenge(manager)))
    manager.close()
    manager.refresh_allowed_clients(
        (CloudAllowedClientConfig("LIC-TEST", "HW-TEST", b"synthetic-test-secret"),)
    )
    with pytest.raises(CloudAuthorizationError, match="zatrzymane"):
        _challenge(manager)
    with pytest.raises(CloudAuthorizationError, match="zatrzymane"):
        manager.authorize(pending)
    with pytest.raises(CloudAuthorizationError, match="zatrzymane"):
        manager.require_session((("authorization", f"CloudSession {session.token}"),))
    assert not manager._sessions
    assert not manager._challenges


def test_cloud_auth_challenge_cannot_change_identity(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    first = CloudAllowedClientConfig("LIC-TEST", "HW-TEST", b"synthetic-test-secret")
    second = CloudAllowedClientConfig("LIC-OTHER", "HW-TEST", b"synthetic-test-secret")
    manager.refresh_allowed_clients((first, second))
    nonce = _challenge(manager)
    payload = {"license_id": "LIC-OTHER", "fingerprint": "HW-TEST", "nonce": nonce}
    signature = sign_license_payload(payload, fingerprint="HW-TEST", secret=second.shared_secret)
    request = trading_pb2.CloudAuthRequest(
        license_id="LIC-OTHER",
        fingerprint="HW-TEST",
        nonce=nonce,
        signature=trading_pb2.CloudAuthSignature(**signature),
    )
    with pytest.raises(CloudAuthorizationError):
        manager.authorize(request)
    assert not manager._sessions


def test_cloud_auth_key_rotation_revokes_session_and_pending_challenge(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    pending = _request(_challenge(manager))
    session = manager.authorize(_request(_challenge(manager)))
    manager.refresh_allowed_clients((CloudAllowedClientConfig("LIC-TEST", "HW-TEST", b"rotated"),))
    with pytest.raises(CloudAuthorizationError):
        manager.require_session((("authorization", f"CloudSession {session.token}"),))
    with pytest.raises(CloudAuthorizationError):
        manager.authorize(pending)


def test_cloud_auth_throttles_identity_across_sources(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    challenge_request = trading_pb2.CloudChallengeRequest(
        license_id="LIC-TEST", fingerprint="HW-TEST"
    )
    for i in range(10):
        manager.issue_challenge(challenge_request, peer=f"ipv4:127.0.0.{i + 1}:1234")
    with pytest.raises(CloudAuthorizationError, match="limit prób"):
        manager.issue_challenge(challenge_request, peer="ipv4:127.0.0.100:1234")
    assert len(manager._challenges) == 10


def test_cloud_auth_throttles_unknown_identity_without_allocating_state(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    for i in range(30):
        request = trading_pb2.CloudChallengeRequest(
            license_id=f"UNKNOWN-{i}", fingerprint="HW-TEST"
        )
        with pytest.raises(CloudAuthorizationError):
            manager.issue_challenge(request, peer=f"ipv4:127.0.0.1:{1000 + i}")
    with pytest.raises(CloudAuthorizationError, match="limit prób"):
        manager.issue_challenge(
            trading_pb2.CloudChallengeRequest(license_id="LIC-TEST", fingerprint="HW-TEST"),
            peer="ipv4:127.0.0.1:9999",
        )
    assert not manager._client_attempts
    assert not manager._challenges


def test_cloud_auth_throttles_global_attempts_across_sources(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    request = trading_pb2.CloudChallengeRequest(license_id="UNKNOWN", fingerprint="HW-TEST")
    for i in range(120):
        with pytest.raises(CloudAuthorizationError):
            manager.issue_challenge(request, peer=f"ipv4:127.0.0.{i}:1")
    with pytest.raises(CloudAuthorizationError, match="limit prób"):
        manager.issue_challenge(request, peer="ipv4:127.0.0.200:1")


def test_cloud_auth_attempt_window_recovers_after_timeout(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr("bot_core.cloud.security.time", SimpleNamespace(monotonic=lambda: clock[0]))
    manager = _manager(tmp_path)
    for _ in range(10):
        _challenge(manager)
    with pytest.raises(CloudAuthorizationError):
        _challenge(manager)
    clock[0] = 60.0
    assert _challenge(manager)
    assert len(manager._challenges) == 1


def test_cloud_auth_rechecks_configured_bundle_and_fails_closed(tmp_path: Path) -> None:
    from types import SimpleNamespace

    bundle = tmp_path / "bundle"
    bundle.write_text("synthetic", encoding="utf-8")

    class LicenseService:
        def load_from_file(self, path, *, expected_hwid):
            assert expected_hwid == "HW-TEST"
            path.read_bytes()
            return SimpleNamespace(capabilities=SimpleNamespace(license_id="LIC-TEST"))

    manager = CloudSecurityManager(
        CloudSecurityConfig(
            require_handshake=True,
            audit_log_path=tmp_path / "audit.log",
            allowed_clients=(
                CloudAllowedClientConfig(
                    "LIC-TEST",
                    "HW-TEST",
                    b"synthetic-test-secret",
                    license_bundle_path=bundle,
                ),
            ),
        ),
        license_service=LicenseService(),
    )
    manager.authorize(_request(_challenge(manager)))
    bundle.unlink()
    with pytest.raises(CloudAuthorizationError, match="pakietu licencji"):
        manager.authorize(_request(_challenge(manager)))
    assert len(manager._sessions) == 1


def test_cloud_auth_bundle_without_verifier_is_refused(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.refresh_allowed_clients(
        (
            CloudAllowedClientConfig(
                "LIC-TEST",
                "HW-TEST",
                b"synthetic-test-secret",
                license_bundle_path=tmp_path / "bundle",
            ),
        )
    )
    with pytest.raises(CloudAuthorizationError, match="weryfikatora"):
        manager.authorize(_request(_challenge(manager)))


def test_cloud_auth_session_expiry_uses_monotonic_clock(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr("bot_core.cloud.security.time", SimpleNamespace(monotonic=lambda: clock[0]))
    manager = _manager(tmp_path)
    session = manager.authorize(_request(_challenge(manager)))
    clock[0] = 900.0
    with pytest.raises(CloudAuthorizationError):
        manager.require_session((("authorization", f"CloudSession {session.token}"),))


def test_cloud_auth_pending_capacity_refuses_without_eviction(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    nonce = _challenge(manager)
    challenge = manager._challenges[nonce]
    manager._challenges = {str(i): challenge for i in range(1024)}
    with pytest.raises(CloudAuthorizationError, match="Limit oczekujących"):
        _challenge(manager)
    assert len(manager._challenges) == 1024


def test_cloud_auth_session_capacity_refuses_without_eviction(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session = manager.authorize(_request(_challenge(manager)))
    request = _request(_challenge(manager))
    manager._sessions = {str(i): session for i in range(1024)}
    with pytest.raises(CloudAuthorizationError, match="Limit aktywnych"):
        manager.authorize(request)
    assert len(manager._sessions) == 1024


@pytest.mark.parametrize("shape", ["unary_unary", "unary_stream", "stream_unary", "stream_stream"])
def test_cloud_interceptor_requires_session_for_same_suffix_on_other_service(
    tmp_path: Path, shape: str
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    from bot_core.cloud.security import CloudAuthInterceptor

    manager = _manager(tmp_path)
    response = (
        (lambda _request, _context: iter((b"ok",)))
        if shape.endswith("stream")
        else (lambda _request, _context: b"ok")
    )
    handler = getattr(grpc, shape + "_rpc_method_handler")(response)
    with ThreadPoolExecutor(max_workers=2) as workers:
        server = grpc.server(workers, interceptors=(CloudAuthInterceptor(manager),))
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "other.Service",
                    {"AuthorizeClient": handler},
                ),
            )
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                rpc = getattr(channel, shape)("/other.Service/AuthorizeClient")
                request = iter((b"request",)) if shape.startswith("stream") else b"request"
                with pytest.raises(grpc.RpcError) as denied:
                    result = rpc(request, timeout=2)
                    if shape.endswith("stream"):
                        list(result)
                assert denied.value.code() == grpc.StatusCode.UNAUTHENTICATED
        finally:
            server.stop(0).wait(timeout=3)


def test_cloud_auth_verification_cannot_outlive_challenge(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr("bot_core.cloud.security.time", SimpleNamespace(monotonic=lambda: clock[0]))
    manager = _manager(tmp_path)
    request = _request(_challenge(manager))

    def slow_verification(*args, **kwargs):
        clock[0] = 60.0
        return True

    monkeypatch.setattr(
        "bot_core.cloud.security.verify_license_payload_signature", slow_verification
    )
    with pytest.raises(CloudAuthorizationError, match="wygasł"):
        manager.authorize(request)
    assert not manager._sessions
