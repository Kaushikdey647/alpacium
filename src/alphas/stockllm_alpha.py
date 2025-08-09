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
    batch_size: int = 4,
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

    # Batch over prompts to improve GPU utilization
    for sym in symbols:
        sdf = df.loc[sym].sort_index()
        dates = list(sdf.index)
        # Collect prompts for this symbol in chunks
        q_list: list[str] = []
        cand_list: list[list[str]] = []
        locs: list[pd.Timestamp] = []
        for i in range(lookback, len(dates)):
            as_of = dates[i]
            qb = QueryBasic.from_dataframe(sym, df, as_of=as_of.date(), lookback=lookback, timeframe=timeframe)
            hits = index.query(qb, top_k=top_k, filter_symbols=filter_symbols)
            payloads = [h["payload"] for h in hits]
            q_list.append(qb.to_paper_json())
            cand_list.append(payloads)
            locs.append(as_of)

            # Flush batch
            if len(q_list) >= batch_size:
                results = generator.predict_many(q_list, cand_list)
                for as_of_i, res in zip(locs, results):
                    movement = str(res.get("movement", "freeze"))
                    probs = res.get("probabilities", {})
                    pr = float(probs.get("rise", 0.0))
                    pf = float(probs.get("fall", 0.0))
                    pz = float(probs.get("freeze", 0.0))
                    conf = float(max(pr, pf, pz))
                    signal = 0.0
                    if conf >= confidence_threshold:
                        signal = 1.0 if movement == "rise" else (-1.0 if movement == "fall" else 0.0)
                    df.loc[(sym, as_of_i), [
                        "movement", "prob_rise", "prob_fall", "prob_freeze", "confidence", "signal"
                    ]] = [movement, pr, pf, pz, conf, signal]
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({"symbol": sym, "as_of": str(as_of_i)})

                q_list.clear(); cand_list.clear(); locs.clear()

        # Flush tail
        if q_list:
            results = generator.predict_many(q_list, cand_list)
            for as_of_i, res in zip(locs, results):
                movement = str(res.get("movement", "freeze"))
                probs = res.get("probabilities", {})
                pr = float(probs.get("rise", 0.0))
                pf = float(probs.get("fall", 0.0))
                pz = float(probs.get("freeze", 0.0))
                conf = float(max(pr, pf, pz))
                signal = 0.0
                if conf >= confidence_threshold:
                    signal = 1.0 if movement == "rise" else (-1.0 if movement == "fall" else 0.0)
                df.loc[(sym, as_of_i), [
                    "movement", "prob_rise", "prob_fall", "prob_freeze", "confidence", "signal"
                ]] = [movement, pr, pf, pz, conf, signal]
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"symbol": sym, "as_of": str(as_of_i)})

    # Ensure correct dtypes
    df["signal"] = df["signal"].fillna(0.0).astype(float)
    if pbar is not None:
        pbar.close()
    return df


