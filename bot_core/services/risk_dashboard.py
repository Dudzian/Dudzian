"""REST API (FastAPI) prezentujące logi audytu ryzyka."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field


class RiskAuditRecord(BaseModel):
    id: int
    timestamp: Optional[datetime] = Field(default=None, description="Znacznik czasu audytu")
    symbol: str
    side: str
    state: str
    reason: Optional[str]
    fraction: float
    price: Optional[float]
    mode: Optional[str]
    schema_version: int
    details: Optional[Dict[str, Any]]
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]


def create_app(db_manager: Any) -> FastAPI:
    """Zbuduj aplikację FastAPI udostępniającą logi audytu ryzyka."""

    app = FastAPI(title="bot_core Risk Dashboard", version="1.0.0")

    async def _get_db() -> Any:
        return db_manager

    @app.get("/health", summary="Sprawdzenie stanu usługi")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/risk/audits", response_model=List[RiskAuditRecord], summary="Lista audytów ryzyka")
    async def list_risk_audits(
        limit: int = Query(100, ge=1, le=500),
        symbol: Optional[str] = Query(None, description="Filtr symbolu, np. BTC/USDT"),
        db: Any = Depends(_get_db),
    ) -> List[RiskAuditRecord]:
        rows = await db.fetch_risk_audits(symbol=symbol, limit=limit)
        return [RiskAuditRecord(**_normalise_row(row)) for row in rows]

    @app.get(
        "/risk/audits/latest",
        response_model=RiskAuditRecord,
        summary="Ostatni audyt dla symbolu lub globalnie",
    )
    async def latest_risk_audit(
        symbol: Optional[str] = Query(None, description="Symbol do odfiltrowania"),
        db: Any = Depends(_get_db),
    ) -> RiskAuditRecord:
        rows = await db.fetch_risk_audits(symbol=symbol, limit=1)
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brak audytów ryzyka")
        return RiskAuditRecord(**_normalise_row(rows[0]))

    return app


def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(row)
    ts = clean.get("timestamp")
    if isinstance(ts, str):
        clean["timestamp"] = datetime.fromisoformat(ts)
    return clean


__all__ = ["create_app", "RiskAuditRecord"]

