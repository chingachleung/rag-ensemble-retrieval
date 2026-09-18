import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import (
    DOCUMENTS,
    BM25Retriever,
    Document,
    EnsembleRetriever,
    EvalQuery,
    TFIDFRetriever,
    evaluate_retriever,
)
from src.eval import _recall_at_k, _reciprocal_rank
from src.reranker import IdentityReranker, LLMReranker


def test_bm25_retriever_returns_k_results():
    retriever = BM25Retriever(DOCUMENTS)
    results = retriever.retrieve("how do I reset my password", k=3)
    assert len(results) == 3


def test_bm25_retriever_finds_obviously_relevant_doc_top1():
    retriever = BM25Retriever(DOCUMENTS)
    results = retriever.retrieve("what is the API rate limit", k=1)
    assert results[0].doc.doc_id == "d9"


def test_tfidf_retriever_finds_obviously_relevant_doc_top1():
    retriever = TFIDFRetriever(DOCUMENTS)
    results = retriever.retrieve("export all of my account data", k=1)
    assert results[0].doc.doc_id == "d10"


def test_ensemble_penalizes_archived_doc_below_current_one():
    retriever = EnsembleRetriever(DOCUMENTS)
    top1 = retriever.retrieve("what's the current refund policy", k=1)[0]
    assert top1.doc.doc_id == "d3"
    assert top1.doc.source == "docs"


def test_ensemble_scores_are_sorted_descending():
    retriever = EnsembleRetriever(DOCUMENTS)
    results = retriever.retrieve("cancel my subscription", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recall_at_k_all_relevant_found():
    assert _recall_at_k(["a", "b", "c"], {"b"}) == 1.0


def test_recall_at_k_none_found():
    assert _recall_at_k(["a", "b", "c"], {"z"}) == 0.0


def test_reciprocal_rank_first_position():
    assert _reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_second_position():
    assert _reciprocal_rank(["a", "b"], {"b"}) == 0.5


def test_reciprocal_rank_not_found():
    assert _reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_evaluate_retriever_perfect_recall_on_easy_query():
    docs = [
        Document("x1", "Password reset", "Reset your password from the login screen."),
        Document("x2", "Unrelated topic", "This document is about something else entirely."),
    ]
    retriever = BM25Retriever(docs)
    eval_queries = [EvalQuery("reset my password", {"x1"})]
    result = evaluate_retriever("bm25", retriever, eval_queries, k=1)
    assert result.recall_at_k == 1.0
    assert result.mrr == 1.0


def test_llm_reranker_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reranker = LLMReranker(api_key=None)
    with pytest.raises(RuntimeError):
        reranker.rerank("q", [])


def test_identity_reranker_returns_input_unchanged():
    retriever = BM25Retriever(DOCUMENTS)
    candidates = retriever.retrieve("password", k=3)
    reranker = IdentityReranker()
    assert reranker.rerank("password", candidates) == candidates
