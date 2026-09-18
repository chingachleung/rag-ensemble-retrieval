"""End-to-end demo: compare BM25, TF-IDF, and the ensemble retriever on a
small labeled eval set, then show a concrete case where the ensemble's
business-logic boost (penalize archived docs) fixes a ranking mistake that
pure lexical scoring makes on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    DOCUMENTS,
    BM25Retriever,
    EnsembleRetriever,
    EvalQuery,
    TFIDFRetriever,
    evaluate_retriever,
)

EVAL_QUERIES = [
    EvalQuery("how do I get my money back", {"d3"}),
    EvalQuery("what's the current refund policy", {"d3"}),
    EvalQuery("the sync client is running slow", {"d6"}),
    EvalQuery("I forgot my password", {"d1"}),
    EvalQuery("how do I set up two-factor authentication", {"d2"}),
    EvalQuery("export all of my account data", {"d10"}),
    EvalQuery("what is the API rate limit", {"d9"}),
    EvalQuery("I want to cancel my subscription", {"d8"}),
    EvalQuery("where can I see my invoices", {"d7"}),
    EvalQuery("app feels sluggish, how do I fix it", {"d5"}),
]


def main() -> None:
    retrievers = {
        "BM25": BM25Retriever(DOCUMENTS),
        "TF-IDF": TFIDFRetriever(DOCUMENTS),
        "Ensemble": EnsembleRetriever(DOCUMENTS),
    }

    print(f"Corpus: {len(DOCUMENTS)} documents. Eval set: {len(EVAL_QUERIES)} labeled queries.\n")
    print(f"{'retriever':<10} {'recall@3':>10} {'MRR':>8}")
    for name, retriever in retrievers.items():
        result = evaluate_retriever(name, retriever, EVAL_QUERIES, k=3)
        print(f"{name:<10} {result.recall_at_k:>10.2f} {result.mrr:>8.2f}")

    # Concrete case: the archived (superseded) refund-policy doc shares a lot
    # of vocabulary with refund queries. Show BM25's raw top-1 vs. the
    # ensemble's business-logic-boosted top-1 for the same query.
    query = "what's the current refund policy"
    print(f"\nQuery: {query!r}")
    bm25_top1 = retrievers["BM25"].retrieve(query, k=1)[0]
    ensemble_top1 = retrievers["Ensemble"].retrieve(query, k=1)[0]
    print(f"  BM25 top-1:     [{bm25_top1.doc.doc_id}] {bm25_top1.doc.title!r} (source={bm25_top1.doc.source})")
    print(f"  Ensemble top-1: [{ensemble_top1.doc.doc_id}] {ensemble_top1.doc.title!r} (source={ensemble_top1.doc.source})")
    print(
        "  -> The ensemble's archive penalty demotes the superseded 2023 "
        "policy doc below the current one, even when lexical overlap alone "
        "would rank them close together or reversed."
    )


if __name__ == "__main__":
    main()
