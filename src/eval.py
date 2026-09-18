"""Retrieval evaluation: recall@k and mean reciprocal rank (MRR) against a
small labeled eval set of (query, relevant doc ids) pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

from .retrievers import Retriever


@dataclass
class EvalQuery:
    query: str
    relevant_doc_ids: set[str]


@dataclass
class RetrieverEvalResult:
    name: str
    recall_at_k: float
    mrr: float
    k: int


def _reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _recall_at_k(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def evaluate_retriever(name: str, retriever: Retriever, eval_queries: list[EvalQuery], k: int = 5) -> RetrieverEvalResult:
    recalls = []
    reciprocal_ranks = []
    for eq in eval_queries:
        ranked = [sd.doc.doc_id for sd in retriever.retrieve(eq.query, k=k)]
        recalls.append(_recall_at_k(ranked, eq.relevant_doc_ids))
        reciprocal_ranks.append(_reciprocal_rank(ranked, eq.relevant_doc_ids))

    return RetrieverEvalResult(
        name=name,
        recall_at_k=sum(recalls) / len(recalls) if recalls else 0.0,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        k=k,
    )
