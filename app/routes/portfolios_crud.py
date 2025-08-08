from __future__ import annotations

from fastapi import APIRouter

from app.schemas import CreatePortfolioRequest, PortfolioModel
from app.deps import get_db


router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.post("", response_model=PortfolioModel)
def create_portfolio(req: CreatePortfolioRequest) -> PortfolioModel:
    db = get_db()
    row = db.create_portfolio(req.name, req.description, req.is_paper, req.initial_capital)
    return PortfolioModel(**row)


@router.get("", response_model=list[PortfolioModel])
def list_portfolios() -> list[PortfolioModel]:
    db = get_db()
    rows = db.list_portfolios()
    return [PortfolioModel(**r) for r in rows]


@router.delete("/{portfolio_id}")
def delete_portfolio(portfolio_id: str) -> dict:
    db = get_db()
    ok = db.delete_portfolio(portfolio_id)
    return {"deleted": ok}


