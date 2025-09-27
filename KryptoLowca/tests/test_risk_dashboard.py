from __future__ import annotations

from typing import Any, Dict, List

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # type: ignore  # noqa: E402

from KryptoLowca.services.risk_dashboard import create_app


class FakeDB:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    async def fetch_risk_audits(self, *, symbol: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        filtered = [row for row in self._rows if symbol is None or row.get("symbol") == symbol]
        return filtered[:limit]


def test_risk_dashboard_endpoints() -> None:
    rows = [
        {
            "id": 1,
            "timestamp": "2024-01-01T00:00:00",
            "symbol": "BTC/USDT",
            "side": "BUY",
            "state": "warn",
            "reason": "trade_risk_pct",
            "fraction": 0.02,
            "price": 100.0,
            "mode": "demo",
            "schema_version": 1,
            "details": {"limit_events": [{"type": "trade_risk_pct"}]},
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }
    ]
    app = create_app(FakeDB(rows))
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp = client.get("/risk/audits")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "BTC/USDT"

    resp = client.get("/risk/audits/latest", params={"symbol": "BTC/USDT"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "warn"

    resp = client.get("/risk/audits/latest", params={"symbol": "ETH/USDT"})
    assert resp.status_code == 404
