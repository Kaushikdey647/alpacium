from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.schemas.timeseries import QueryBasic, QueryTechnical, Candidate
except Exception:  # pragma: no cover
    QueryBasic = QueryTechnical = Candidate = object  # type: ignore


@dataclass
class StockLLMConfig:
    model_id: str = "TheFinAI/StockLLM"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    device: Optional[str] = None
    trust_remote_code: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def _resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StockLLMGenerator:
    def __init__(self, config: Optional[StockLLMConfig] = None) -> None:
        self.config = config or StockLLMConfig()
        self.device = _resolve_device(self.config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=self.config.trust_remote_code
        )
        model_kwargs: Dict[str, Any] = {"trust_remote_code": self.config.trust_remote_code}
        if self.config.load_in_4bit:
            model_kwargs.update({"load_in_4bit": True})
        elif self.config.load_in_8bit:
            model_kwargs.update({"load_in_8bit": True})
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device in ("cuda", "mps") else None,
            **model_kwargs,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    def _build_prompt(self, query_json: str, candidates_json: Iterable[str]) -> str:
        parts: List[str] = []
        parts.append(
            "You are a model that predicts the next trading day movement for the query stock.\n"
            "Use the retrieved indicator sequences as context. Valid movements: rise, fall, freeze.\n"
            "Return ONLY strict JSON with keys: movement (one of rise/fall/freeze) and probabilities (object with keys rise/fall/freeze)."
        )
        parts.append("\nQuery:\n" + query_json)
        parts.append("\nRetrieved:")
        for i, cj in enumerate(candidates_json):
            parts.append(f"\nCandidate_{i+1}:\n{cj}")
        parts.append(
            "\nReturn JSON only in one line: {\"movement\": \"...\", \"probabilities\": {\"rise\": r, \"fall\": f, \"freeze\": z}}"
        )
        return "\n".join(parts)

    def generate_raw(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0.0,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return gen.strip()


