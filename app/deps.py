from __future__ import annotations

from typing import Optional

from src.retrieval.finseer_client import FinSeerEmbedder, FinSeerConfig
from src.retrieval.faiss_index import FaissCandidateIndex, default_indicator_builder
from src.llm.stockllm_client import StockLLMGenerator, StockLLMConfig
from src.adapters.supabase_client import SupabaseDB, SupabaseConfig

_embedder: Optional[FinSeerEmbedder] = None
_index: Optional[FaissCandidateIndex] = None
_generator: Optional[StockLLMGenerator] = None
_db: Optional[SupabaseDB] = None


def get_embedder() -> FinSeerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FinSeerEmbedder(FinSeerConfig())
    return _embedder


def get_index() -> FaissCandidateIndex:
    global _index
    if _index is None:
        _index = FaissCandidateIndex(get_embedder())
    return _index


def get_generator() -> StockLLMGenerator:
    global _generator
    if _generator is None:
        _generator = StockLLMGenerator(StockLLMConfig())
    return _generator


def get_db() -> SupabaseDB:
    global _db
    if _db is None:
        _db = SupabaseDB(SupabaseConfig())
    return _db


