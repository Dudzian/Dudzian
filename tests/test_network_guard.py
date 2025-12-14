import socket

import pytest


def test_outbound_connection_is_blocked(monkeypatch):
    monkeypatch.delenv("ALLOW_NETWORK_TESTS", raising=False)
    with pytest.raises(RuntimeError, match="External network access is blocked"):
        socket.create_connection(("1.1.1.1", 443), timeout=0.1)


def test_localhost_connection_is_allowed(monkeypatch):
    monkeypatch.delenv("ALLOW_NETWORK_TESTS", raising=False)

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    client = None
    conn = None
    try:
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        conn, _ = server.accept()
    finally:
        if client:
            client.close()
        if conn:
            conn.close()
        server.close()
