from __future__ import annotations

from typing import Any, Dict

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
    _LG = True
except Exception:  # pragma: no cover
    _LG = False

from src.schemas.timeseries import QueryBasic, QueryTechnical
from src.retrieval.faiss_index import FaissCandidateIndex
from src.llm.stockllm_client import StockLLMGenerator


def build_retrieval_graph(index: FaissCandidateIndex, generator: StockLLMGenerator) -> Any:
    if not _LG:
        raise RuntimeError("LangGraph not installed. Install `langgraph`.")

    def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state["query"]
        top_k = state.get("top_k", 5)
        filter_symbols = state.get("filter_symbols")
        state["candidates"] = index.query(q, top_k=top_k, filter_symbols=filter_symbols)
        return state

    def generate(state: Dict[str, Any]) -> Dict[str, Any]:
        q = state["query"]
        cand_payloads = [c["payload"] for c in state.get("candidates", [])]
        result = generator.predict(q.to_paper_json(), cand_payloads)
        state["result"] = result
        return state

    g = StateGraph(dict)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


