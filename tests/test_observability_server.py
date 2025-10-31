import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


from bot_core.observability import MetricsRegistry, start_http_server


def _find_free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def test_metrics_server_serves_registered_metrics() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("test_counter", "Opis testowej metryki")
    counter.inc(3, labels={"symbol": "BTCUSDT"})

    port = _find_free_port()
    server, thread = start_http_server(port, registry=registry)
    try:
        # czekamy na start serwera
        for _ in range(10):
            try:
                with urlopen(f"http://127.0.0.1:{server.server_address[1]}/metrics") as response:
                    body = response.read().decode("utf-8")
                    assert "test_counter" in body
                    assert 'symbol="BTCUSDT"' in body
                    break
            except (ConnectionError, URLError):
                time.sleep(0.05)
        else:  # pragma: no cover - diagnostyka środowiska CI
            raise AssertionError("Serwer metryk nie wystartował w oczekiwanym czasie")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_metrics_server_returns_404_for_unknown_path() -> None:
    registry = MetricsRegistry()
    port = _find_free_port()
    server, thread = start_http_server(port, registry=registry)
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/invalid") as response:  # pragma: no cover
            raise AssertionError("Żądanie /invalid powinno zakończyć się błędem")
    except HTTPError as exc:
        assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
