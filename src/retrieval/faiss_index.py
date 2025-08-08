from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    _FAISS_AVAILABLE = False

from src.schemas.timeseries import (
    Candidate,
    IndicatorSeries,
    QueryBasic,
    QueryTechnical,
    TimeFrame,
)
from src.retrieval.finseer_client import FinSeerEmbedder


def _to_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    return x


def _hash_id(*parts: str) -> int:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False) & ((1 << 63) - 1)


def default_indicator_builder(symbol_df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for col in ("high", "low", "close", "adjusted_close", "volume"):
        if col in symbol_df.columns:
            out[col] = symbol_df[col].astype(float)
    try:
        import talib  # type: ignore

        close = symbol_df["close"].astype(float).to_numpy()
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        out["RSI"] = pd.Series(rsi, index=symbol_df.index).fillna(method="ffill").fillna(50)
        out["MACD_hist"] = pd.Series(macd_hist, index=symbol_df.index).fillna(0)
    except Exception:
        diff = symbol_df["close"].astype(float).diff().fillna(0)
        up = diff.clip(lower=0).rolling(14).mean()
        down = (-diff.clip(upper=0)).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        rsi = 100 - 100 / (1 + rs)
        out["RSI"] = rsi.fillna(50)
        out["MACD_hist"] = symbol_df["close"].astype(float).diff().fillna(0)
    return out


@dataclass
class VectorMeta:
    symbol: str
    candidate_date: str
    indicator_name: str
    payload: str


class _NumpyIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.V = np.zeros((0, dim), dtype=np.float32)
        self.ids: List[int] = []

    def add_with_ids(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        assert vecs.shape[1] == self.dim
        self.V = np.vstack([self.V, vecs.astype(np.float32)])
        self.ids.extend(ids.tolist())

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.V.size == 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
        sims = self.V @ q.T
        order = np.argsort(-sims, axis=0).ravel()[:k]
        D = sims[order].reshape(1, -1)
        I = np.array([self.ids[i] for i in order], dtype=np.int64).reshape(1, -1)
        return D, I


class FaissCandidateIndex:
    def __init__(self, embedder: FinSeerEmbedder, normalize: bool = True) -> None:
        self.embedder = embedder
        self.normalize = normalize
        self._dim: Optional[int] = None
        self._index = None  # type: ignore
        self.id_to_meta: Dict[int, VectorMeta] = {}
        self.symbol_to_ids: Dict[str, set[int]] = {}
        self.symbol_last_date: Dict[str, str] = {}

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        self._dim = dim
        if _FAISS_AVAILABLE:
            base = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIDMap2(base)
        else:
            self._index = _NumpyIndex(dim)

    def _add_vectors(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self._index.add_with_ids(vectors, ids)

    def _generate_candidates_for_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        lookback: int,
        indicator_builder=default_indicator_builder,
        timeframe: TimeFrame = TimeFrame.day,
        start_date: Optional[date] = None,
    ) -> List[Candidate]:
        df = df.sort_index()
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
            df = df.loc[symbol]
        indicators = indicator_builder(df)
        dates = list(df.index)
        candidates: List[Candidate] = []
        for i in range(lookback - 1, len(dates)):
            end_dt = dates[i]
            end_d = _to_date(end_dt)
            if start_date and end_d <= start_date:
                continue
            window_idx = dates[i - (lookback - 1) : i + 1]
            recent_dates = [_to_date(d) for d in window_idx]
            for name, series in indicators.items():
                values = list(series.loc[window_idx].astype(float).values)
                candidates.append(
                    Candidate(
                        candidate_stock=symbol,
                        candidate_date=end_d,
                        movement=None,
                        timeframe=timeframe,
                        recent_dates=recent_dates,
                        indicator=IndicatorSeries(name=name, values=values),
                    )
                )
        return candidates

    def add_candidates(self, cands: Sequence[Candidate]) -> int:
        if not cands:
            return 0
        texts = [c.to_paper_json() for c in cands]
        vecs = self.embedder.encode(texts)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        self._ensure_index(vecs.shape[1])
        ids: List[int] = []
        metas: List[VectorMeta] = []
        for c, j in zip(cands, texts):
            vid = _hash_id(c.candidate_stock, c.candidate_date.isoformat(), c.indicator.name)
            ids.append(vid)
            metas.append(
                VectorMeta(
                    symbol=c.candidate_stock,
                    candidate_date=c.candidate_date.isoformat(),
                    indicator_name=c.indicator.name,
                    payload=j,
                )
            )
        ids_arr = np.array(ids, dtype=np.int64)
        self._add_vectors(vecs.astype(np.float32), ids_arr)
        for vid, m in zip(ids, metas):
            self.id_to_meta[vid] = m
            self.symbol_to_ids.setdefault(m.symbol, set()).add(vid)
            self.symbol_last_date[m.symbol] = max(
                m.candidate_date, self.symbol_last_date.get(m.symbol, "1900-01-01")
            )
        return len(cands)

    def build_from_symbol_dfs(
        self,
        symbol_to_df: Dict[str, pd.DataFrame],
        lookback: int = 5,
        indicator_builder=default_indicator_builder,
        timeframe: TimeFrame = TimeFrame.day,
    ) -> int:
        total = 0
        for sym, df in symbol_to_df.items():
            cands = self._generate_candidates_for_symbol(sym, df, lookback, indicator_builder, timeframe)
            total += self.add_candidates(cands)
        return total

    def update_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        lookback: int = 5,
        indicator_builder=default_indicator_builder,
        timeframe: TimeFrame = TimeFrame.day,
    ) -> int:
        start_str = self.symbol_last_date.get(symbol)
        start = datetime.fromisoformat(start_str).date() if start_str else None
        cands = self._generate_candidates_for_symbol(symbol, df, lookback, indicator_builder, timeframe, start_date=start)
        return self.add_candidates(cands)

    def query(
        self,
        query: QueryBasic | QueryTechnical,
        top_k: int = 5,
        filter_symbols: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        q_vec = self.embedder.encode_one(query.to_paper_json()).astype(np.float32)
        if self.normalize:
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
        q_vec = q_vec.reshape(1, -1)
        D, I = self._index.search(q_vec, top_k)
        ids = I[0].tolist()
        scores = D[0].tolist()
        results: List[Dict[str, Any]] = []
        allowed = set(filter_symbols) if filter_symbols else None
        for vid, score in zip(ids, scores):
            meta = self.id_to_meta.get(vid)
            if not meta:
                continue
            if allowed and meta.symbol not in allowed:
                continue
            results.append(
                {
                    "id": vid,
                    "score": float(score),
                    "symbol": meta.symbol,
                    "candidate_date": meta.candidate_date,
                    "indicator": meta.indicator_name,
                    "payload": meta.payload,
                }
            )
        return results

    # --- Persistence -------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        meta = {vid: asdict(m) for vid, m in self.id_to_meta.items()}
        with open(os.path.join(directory, "id_to_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(os.path.join(directory, "symbol_last_date.json"), "w", encoding="utf-8") as f:
            json.dump(self.symbol_last_date, f)
        with open(os.path.join(directory, "symbol_to_ids.json"), "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in self.symbol_to_ids.items()}, f)
        if _FAISS_AVAILABLE:
            faiss.write_index(self._index, os.path.join(directory, "faiss.index"))
        else:
            np.save(os.path.join(directory, "vectors.npy"), self._index.V)
            with open(os.path.join(directory, "ids.json"), "w", encoding="utf-8") as f:
                json.dump(self._index.ids, f)

    def load(self, directory: str) -> None:
        with open(os.path.join(directory, "id_to_meta.json"), "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.id_to_meta = {int(k): VectorMeta(**v) for k, v in raw.items()}
        with open(os.path.join(directory, "symbol_last_date.json"), "r", encoding="utf-8") as f:
            self.symbol_last_date = json.load(f)
        with open(os.path.join(directory, "symbol_to_ids.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            self.symbol_to_ids = {k: set(v) for k, v in data.items()}
        any_meta = next(iter(self.id_to_meta.values()), None)
        if any_meta is None:
            return
        dim = self.embedder.encode_one(any_meta.payload).shape[0]
        self._ensure_index(dim)
        if _FAISS_AVAILABLE and os.path.exists(os.path.join(directory, "faiss.index")):
            self._index = faiss.read_index(os.path.join(directory, "faiss.index"))
        elif os.path.exists(os.path.join(directory, "vectors.npy")):
            V = np.load(os.path.join(directory, "vectors.npy"))
            with open(os.path.join(directory, "ids.json"), "r", encoding="utf-8") as f:
                ids = np.array(json.load(f), dtype=np.int64)
            self._index = _NumpyIndex(V.shape[1])
            self._index.add_with_ids(V, ids)


