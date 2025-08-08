from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os

try:
    from supabase import create_client, Client  # type: ignore
    _SUPABASE_AVAILABLE = True
except Exception:  # pragma: no cover
    _SUPABASE_AVAILABLE = False


@dataclass
class SupabaseConfig:
    url: Optional[str] = None
    key: Optional[str] = None


class SupabaseDB:
    """Thin wrapper around supabase-py for simple CRUD on portfolios and embeddings.

    Tables expected:
      - portfolios(id uuid default uuid_generate_v4() primary key, name text, description text,
                   is_paper boolean, initial_capital numeric, created_at timestamptz default now())
      - embeddings(id bigserial primary key, vector_id bigint, payload jsonb, meta jsonb)
    """

    def __init__(self, config: Optional[SupabaseConfig] = None) -> None:
        if not _SUPABASE_AVAILABLE:
            raise RuntimeError("supabase client not installed. `pip install supabase`.")
        cfg = config or SupabaseConfig(
            url=os.getenv("SUPABASE_URL"),
            key=os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY"),
        )
        if not cfg.url or not cfg.key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY env vars")
        self.client: Client = create_client(cfg.url, cfg.key)

    # --- Portfolios --------------------------------------------------------
    def create_portfolio(self, name: str, description: str | None, is_paper: bool, initial_capital: float) -> Dict[str, Any]:
        data = {
            "name": name,
            "description": description or "",
            "is_paper": is_paper,
            "initial_capital": initial_capital,
        }
        res = self.client.table("portfolios").insert(data).execute()
        rows = getattr(res, "data", None) or []
        return rows[0] if rows else data

    # --- Alpha registry ----------------------------------------------------
    def upsert_alpha(self, name: str, import_path: str, version: str, default_params: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "name": name,
            "import_path": import_path,
            "version": version,
            "default_params": default_params,
        }
        return self.upsert("alpha_registry", row)

    def list_alphas(self) -> List[Dict[str, Any]]:
        return self.select("alpha_registry", limit=1000)

    def delete_alpha(self, name: str) -> bool:
        return self.delete("alpha_registry", {"name": name})

    def get_portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        res = self.client.table("portfolios").select("*").eq("id", portfolio_id).single().execute()
        return getattr(res, "data", None)

    def list_portfolios(self, limit: int = 100) -> List[Dict[str, Any]]:
        res = self.client.table("portfolios").select("*").limit(limit).execute()
        return getattr(res, "data", None) or []

    def update_portfolio(self, portfolio_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        res = self.client.table("portfolios").update(fields).eq("id", portfolio_id).execute()
        rows = getattr(res, "data", None) or []
        return rows[0] if rows else fields

    def delete_portfolio(self, portfolio_id: str) -> bool:
        res = self.client.table("portfolios").delete().eq("id", portfolio_id).execute()
        return True

    # --- Embeddings (optional) --------------------------------------------
    def upsert_embedding(self, vector_id: int, payload: Dict[str, Any], meta: Dict[str, Any] | None = None) -> None:
        row = {"vector_id": vector_id, "payload": payload, "meta": meta or {}}
        self.client.table("embeddings").upsert(row).execute()

    # --- Generic helpers ---------------------------------------------------
    def insert(self, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
        res = self.client.table(table).insert(row).execute()
        rows = getattr(res, "data", None) or []
        return rows[0] if rows else row

    def upsert(self, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
        res = self.client.table(table).upsert(row).execute()
        rows = getattr(res, "data", None) or []
        return rows[0] if rows else row

    def select(self, table: str, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        q = self.client.table(table).select("*")
        if filters:
            for k, v in filters.items():
                q = q.eq(k, v)
        res = q.limit(limit).execute()
        return getattr(res, "data", None) or []

    def get_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        q = self.client.table(table).select("*")
        for k, v in filters.items():
            q = q.eq(k, v)
        res = q.single().execute()
        return getattr(res, "data", None)

    def delete(self, table: str, filters: Dict[str, Any]) -> bool:
        q = self.client.table(table).delete()
        for k, v in filters.items():
            q = q.eq(k, v)
        q.execute()
        return True


