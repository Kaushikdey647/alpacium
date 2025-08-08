from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from app.schemas import BuildIndexRequest, BuildIndexResponse, UpdateIndexRequest, UpdateIndexResponse
from app.deps import get_index
from src.adapters.alpaca import symbols_bars_to_df_map
from src.retrieval.faiss_index import default_indicator_builder


router = APIRouter(prefix="/index", tags=["index"])


@router.post("/build", response_model=BuildIndexResponse)
def build_index(req: BuildIndexRequest) -> BuildIndexResponse:
    sym_map = symbols_bars_to_df_map([s.dict() for s in req.symbols])
    total = get_index().build_from_symbol_dfs(
        sym_map, lookback=req.lookback, indicator_builder=default_indicator_builder, timeframe=req.timeframe
    )
    return BuildIndexResponse(total_candidates=total, symbols_indexed=list(sym_map.keys()))


@router.post("/update", response_model=UpdateIndexResponse)
def update_index(req: UpdateIndexRequest) -> UpdateIndexResponse:
    sym_map = symbols_bars_to_df_map([s.dict() for s in req.symbols])
    total_added = 0
    idx = get_index()
    for sym, df in sym_map.items():
        total_added += idx.update_symbol(sym, df, lookback=req.lookback, timeframe=req.timeframe)
    return UpdateIndexResponse(total_added=total_added, symbols_updated=list(sym_map.keys()))


