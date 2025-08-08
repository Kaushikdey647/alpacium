from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from app.deps import get_db


router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("")
def upsert_embedding(row: Dict[str, Any]) -> Dict[str, Any]:
    db = get_db()
    return db.upsert("embeddings", row)


@router.get("")
def list_embeddings(limit: int = 100) -> List[Dict[str, Any]]:
    db = get_db()
    return db.select("embeddings", limit=limit)


@router.delete("")
def delete_embedding(vector_id: int) -> Dict[str, bool]:
    db = get_db()
    ok = db.delete("embeddings", {"vector_id": vector_id})
    return {"deleted": ok}


