from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from haystack.document_stores import FAISSDocumentStore  # type: ignore
from haystack import Document  # type: ignore


class FinSeerHaystackIndex:
    """Haystack FAISS-backed store for FinSeer candidate payloads.

    Stores each candidate as a Document with text=payload and metadata:
    {symbol, candidate_date, indicator}.
    """

    def __init__(self, embedding_dim: int = 768, faiss_index: str = "Flat") -> None:
        self.store = FAISSDocumentStore(embedding_dim=embedding_dim, faiss_index_factory_str=faiss_index)

    def write_candidates(self, payloads: Sequence[Dict[str, Any]]) -> None:
        docs = [
            Document(
                content=p["payload"],
                meta={
                    "symbol": p["symbol"],
                    "candidate_date": p["candidate_date"],
                    "indicator": p["indicator"],
                },
            )
            for p in payloads
        ]
        self.store.write_documents(docs)

    def update_embeddings(self, embedder) -> None:
        # `embedder` should provide a .embed_documents(List[str]) -> np.ndarray API
        self.store.update_embeddings(retriever=embedder)


