from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from app.deps import get_db


router = APIRouter(prefix="/alphas", tags=["alphas"])


@router.post("")
def upsert_alpha(row: Dict[str, Any]) -> Dict[str, Any]:
    db = get_db()
    return db.upsert_alpha(
        name=row["name"], import_path=row["import_path"], version=row.get("version", "dev"), default_params=row.get("default_params", {})
    )


@router.get("")
def list_alphas() -> List[Dict[str, Any]]:
    db = get_db()
    return db.list_alphas()


@router.delete("")
def delete_alpha(name: str) -> Dict[str, bool]:
    db = get_db()
    ok = db.delete_alpha(name)
    return {"deleted": ok}


