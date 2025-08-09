from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.schemas import QueryBasic, TimeFrame
from src.retrieval.faiss_index import FaissCandidateIndex
from src.llm.stockllm_client import StockLLMGenerator


def stockllm_alpha(
    historical_data: pd.DataFrame,
    index: FaissCandidateIndex,
    generator: StockLLMGenerator,
    lookback: int = 5,
    top_k: int = 5,
    timeframe: TimeFrame = TimeFrame.day,
    filter_symbols: Optional[Iterable[str]] = None,
    confidence_threshold: float = 0.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Generate signals using StockLLM with FinSeer retrieval.

    Returns a DataFrame aligned to `historical_data` MultiIndex with columns:
    - signal in {-1, 0, 1}
    - movement (rise/fall/freeze)
    - prob_rise, prob_fall, prob_freeze, confidence
    """
    df = historical_data.copy()
    symbols = df.index.get_level_values(0).unique()
    out_cols = [
        "movement",
        "prob_rise",
        "prob_fall",
        "prob_freeze",
        "confidence",
        "signal",
    ]
    for c in out_cols:
        df[c] = np.nan

    # Pre-compute total iterations for a single progress bar across all symbols
    total_iters = 0
    for sym in symbols:
        sdf_tmp = df.loc[sym].sort_index()
        total_iters += max(0, len(sdf_tmp.index) - lookback)

    pbar = tqdm(total=total_iters, desc="Generating StockLLM signals", leave=True) if show_progress else None

    for sym in symbols:
        sdf = df.loc[sym].sort_index()
        dates = list(sdf.index)
        # Start once we have at least `lookback` prior bars strictly before `as_of`
        for i in range(lookback, len(dates)):
            as_of = dates[i]
            qb = QueryBasic.from_dataframe(sym, df, as_of=as_of.date(), lookback=lookback, timeframe=timeframe)
            hits = index.query(qb, top_k=top_k, filter_symbols=filter_symbols)
            payloads = [h["payload"] for h in hits]
            result = generator.predict(qb.to_paper_json(), payloads)
            movement = str(result.get("movement", "freeze"))
            probs = result.get("probabilities", {})
            pr = float(probs.get("rise", 0.0))
            pf = float(probs.get("fall", 0.0))
            pz = float(probs.get("freeze", 0.0))
            conf = float(max(pr, pf, pz))
            signal = 0.0
            if conf >= confidence_threshold:
                signal = 1.0 if movement == "rise" else (-1.0 if movement == "fall" else 0.0)
            df.loc[(sym, as_of), [
                "movement", "prob_rise", "prob_fall", "prob_freeze", "confidence", "signal"
            ]] = [movement, pr, pf, pz, conf, signal]

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"symbol": sym, "as_of": str(as_of)})

    # Ensure correct dtypes
    df["signal"] = df["signal"].fillna(0.0).astype(float)
    if pbar is not None:
        pbar.close()
    return df


