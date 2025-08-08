from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter

from app.deps import get_db


router = APIRouter(prefix="/bars", tags=["bars"])


@router.post("")
def upsert_bars(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Upsert normalized OHLCV bars rows to Supabase (table `bars`)."""
    db = get_db()
    count = 0
    for r in rows:
        db.upsert("bars", r)
        count += 1
    return {"count": count}


@router.get("")
def list_bars(symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
    db = get_db()
    return db.select("bars", filters={"symbol": symbol}, limit=limit)


