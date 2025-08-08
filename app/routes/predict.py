from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import QueryPredictRequest, PredictResponse, CandidateHit
from app.deps import get_index, get_generator
from src.schemas import OHLCVWindow, QueryBasic


router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: QueryPredictRequest) -> PredictResponse:
    sym = req.symbol
    window = OHLCVWindow(recent_dates=req.recent_date_list, adjusted_close_list=req.adjusted_close_list)
    qb = QueryBasic(query_stock=sym, query_date=req.as_of, timeframe=req.timeframe, window=window)
    idx = get_index()
    hits = idx.query(qb, top_k=req.top_k, filter_symbols=req.filter_symbols)
    payloads = [h["payload"] for h in hits]
    gen = get_generator()
    result = gen.predict(qb.to_paper_json(), payloads)
    return PredictResponse(
        movement=result.get("movement", "freeze"),
        probabilities=result.get("probabilities", {}),
        candidates=[CandidateHit(**h) for h in hits],
    )


