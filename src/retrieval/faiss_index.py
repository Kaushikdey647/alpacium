from __future__ import annotations

import hashlib
import json
import os
import atexit
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
    try:
        _FAISS_NUM_GPUS = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    except Exception:
        _FAISS_NUM_GPUS = 0
except Exception:  # pragma: no cover
    _FAISS_AVAILABLE = False
    _FAISS_NUM_GPUS = 0

from src.schemas.timeseries import (
    Candidate,
    IndicatorSeries,
    QueryBasic,
    QueryTechnical,
    TimeFrame,
)
from src.retrieval.finseer_client import FinSeerEmbedder


try:
    # tqdm.auto selects the best frontend (notebook/terminal)
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore
        return x

def _to_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    return x


def _hash_id(*parts: str) -> int:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False) & ((1 << 63) - 1)


def default_indicator_builder(symbol_df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    # Include full OHLCV per paper alignment
    for col in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if col in symbol_df.columns:
            out[col] = symbol_df[col].astype(float)
    try:
        import talib  # type: ignore

        close = symbol_df["close"].astype(float).to_numpy()
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        out["RSI"] = pd.Series(rsi, index=symbol_df.index).ffill().fillna(50)
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
    def __init__(
        self,
        embedder: FinSeerEmbedder,
        normalize: bool = True,
        use_gpu: bool = True,
        persist_dir: Optional[str] = None,
        auto_persist: bool = True,
    ) -> None:
        self.embedder = embedder
        self.normalize = normalize
        self.use_gpu = use_gpu
        self._dim: Optional[int] = None
        self._index = None  # type: ignore
        self.id_to_meta: Dict[int, VectorMeta] = {}
        self.symbol_to_ids: Dict[str, set[int]] = {}
        self.symbol_last_date: Dict[str, str] = {}
        self._logger = logging.getLogger(__name__)
        self.persist_dir = persist_dir
        self.auto_persist = auto_persist

        # Auto-load persisted index if configured
        if self.persist_dir:
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                # Only attempt load if files exist
                if os.path.exists(os.path.join(self.persist_dir, "id_to_meta.json")):
                    self.load(self.persist_dir)
                    self._logger.info(
                        "Loaded index from %s (ids=%d symbols=%d)",
                        self.persist_dir,
                        len(self.id_to_meta),
                        len(self.symbol_to_ids),
                    )
            except Exception as e:
                self._logger.warning("Failed to load index from %s: %s", self.persist_dir, e)
            if self.auto_persist:
                atexit.register(self._save_on_exit)

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        self._dim = dim
        if _FAISS_AVAILABLE:
            # Prefer GPU if available and requested
            if self.use_gpu and _FAISS_NUM_GPUS > 0 and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    base = faiss.GpuIndexFlatIP(res, dim)
                    self._index = faiss.IndexIDMap2(base)
                    self._logger.info("Using FAISS GPU (FlatIP) on %d GPU(s)", _FAISS_NUM_GPUS)
                    return
                except Exception as e:
                    self._logger.warning("Falling back to CPU FAISS (GPU init failed: %s)", e)
            base = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIDMap2(base)
        else:
            self._index = _NumpyIndex(dim)

    def _add_vectors(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self._index.add_with_ids(vectors, ids)

    def _save_on_exit(self) -> None:
        try:
            if self.persist_dir:
                self.save(self.persist_dir)
                self._logger.info("Saved index on exit to %s", self.persist_dir)
        except Exception as e:
            self._logger.warning("Failed to save index on exit: %s", e)

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
        self._logger.info(
            "Building candidates: symbol=%s rows=%d indicators=%s lookback=%d",
            symbol, len(df), list(indicators.keys()), lookback,
        )
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
                seg = series.loc[window_idx].astype(float)
                # Ensure 1D values for IndicatorSeries, per paper JSON schema (5 numbers)
                if isinstance(seg, pd.DataFrame):
                    seg = seg.iloc[:, 0]
                values = [float(x) for x in seg.tolist()]
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
        self._logger.info("Built %d candidates for symbol=%s", len(candidates), symbol)
        return candidates

    def add_candidates(self, cands: Sequence[Candidate], show_progress: bool = True) -> int:
        if not cands:
            return 0

        texts = [c.to_paper_json() for c in cands]
        # Derive batch size from embedder if available
        bs = getattr(getattr(self.embedder, "config", None), "batch_size", 32)

        # Prepare ids/metas aligned with texts so we can slice per-batch
        ids_all: List[int] = []
        metas_all: List[VectorMeta] = []
        for c, j in zip(cands, texts):
            vid = _hash_id(c.candidate_stock, c.candidate_date.isoformat(), c.indicator.name)
            ids_all.append(vid)
            metas_all.append(
                VectorMeta(
                    symbol=c.candidate_stock,
                    candidate_date=c.candidate_date.isoformat(),
                    indicator_name=c.indicator.name,
                    payload=j,
                )
            )

        total_added = 0
        rng = range(0, len(texts), bs)
        total_batches = len(texts) // bs + (1 if len(texts) % bs else 0)
        iterator = (
            tqdm(rng, desc="Embedding batches", total=total_batches)
            if show_progress
            else rng
        )

        # Avoid double-normalization when embedder already normalizes
        embedder_normalizes = bool(getattr(getattr(self.embedder, "config", None), "normalize", False))

        for start in iterator:
            end = min(start + bs, len(texts))
            chunk_texts = texts[start:end]
            vecs = self.embedder.encode(chunk_texts)

            if self.normalize and not embedder_normalizes:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms

            if self._index is None:
                self._ensure_index(vecs.shape[1])

            ids_arr = np.array(ids_all[start:end], dtype=np.int64)
            self._add_vectors(vecs.astype(np.float32), ids_arr)

            # Update metadata per-batch
            for vid, m in zip(ids_all[start:end], metas_all[start:end]):
                self.id_to_meta[vid] = m
                self.symbol_to_ids.setdefault(m.symbol, set()).add(vid)
                self.symbol_last_date[m.symbol] = max(
                    m.candidate_date, self.symbol_last_date.get(m.symbol, "1900-01-01")
                )
            total_added += (end - start)

        self._logger.info("Indexed %d vectors (dim=%s)", total_added, self._dim)
        # Optional auto-persist after batch additions
        if self.persist_dir and self.auto_persist:
            try:
                self.save(self.persist_dir)
                self._logger.info("Auto-saved index to %s", self.persist_dir)
            except Exception as e:
                self._logger.warning("Auto-save failed: %s", e)
        return total_added

    def build_from_symbol_dfs(
        self,
        symbol_to_df: Dict[str, pd.DataFrame],
        lookback: int = 5,
        indicator_builder=default_indicator_builder,
        timeframe: TimeFrame = TimeFrame.day,
        show_progress: bool = True,
    ) -> int:
        """Build index from per-symbol DataFrames.

        If show_progress is True, display a tqdm progress bar over symbols.
        """
        total = 0
        items = list(symbol_to_df.items())
        iterator = tqdm(items, desc="Indexing symbols", total=len(items)) if show_progress else items
        for sym, df in iterator:
            cands = self._generate_candidates_for_symbol(sym, df, lookback, indicator_builder, timeframe)
            total += self.add_candidates(cands, show_progress=True)
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
        embedder_normalizes = bool(getattr(getattr(self.embedder, "config", None), "normalize", False))
        if self.normalize and not embedder_normalizes:
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

    # --- Maintenance ------------------------------------------------------------
    def clear(self) -> None:
        """Clear all vectors and metadata from the index."""
        self._index = None
        self._dim = None
        self.id_to_meta.clear()
        self.symbol_to_ids.clear()
        self.symbol_last_date.clear()


