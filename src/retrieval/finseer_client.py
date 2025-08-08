from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from src.schemas.timeseries import QueryBasic, QueryTechnical, Candidate
except Exception:  # pragma: no cover
    QueryBasic = QueryTechnical = Candidate = object  # type: ignore


PoolingStrategy = Literal["cls", "mean", "last_token"]


@dataclass
class FinSeerConfig:
    model_id: str = "TheFinAI/FinSeer"
    pooling: PoolingStrategy = "cls"
    normalize: bool = True
    max_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None
    trust_remote_code: bool = True


def _resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FinSeerEmbedder:
    def __init__(self, config: Optional[FinSeerConfig] = None) -> None:
        self.config = config or FinSeerConfig()
        self.device = _resolve_device(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=self.config.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_id, trust_remote_code=self.config.trust_remote_code
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        strategy = self.config.pooling
        if strategy == "cls":
            return last_hidden_state[:, 0]
        if strategy == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts
        if strategy == "last_token":
            lengths = attention_mask.sum(dim=1) - 1
            return last_hidden_state[torch.arange(last_hidden_state.size(0)), lengths]
        raise ValueError(f"Unsupported pooling strategy: {strategy}")

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = list(texts)
        if not texts_list:
            return np.zeros((0, 0), dtype=np.float32)
        all_embeddings: List[np.ndarray] = []
        bs = self.config.batch_size
        for i in range(0, len(texts_list), bs):
            chunk = texts_list[i : i + bs]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=self.config.max_length, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            pooled = self._pool(outputs.last_hidden_state, enc["attention_mask"])  # (B, H)
            emb = pooled.detach().cpu().numpy().astype(np.float32)
            if self.config.normalize:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb = emb / norms
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def to_text_query(q: Union["QueryBasic", "QueryTechnical"]) -> str:
        return q.to_paper_json()

    @staticmethod
    def to_text_candidate(c: "Candidate") -> str:
        return c.to_paper_json()

    def encode_query(self, q: Union["QueryBasic", "QueryTechnical"]) -> np.ndarray:
        return self.encode_one(self.to_text_query(q))

    def encode_candidates(self, candidates: Iterable["Candidate"]) -> np.ndarray:
        return self.encode([self.to_text_candidate(c) for c in candidates])


