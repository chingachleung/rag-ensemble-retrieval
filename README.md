# rag-ensemble-retrieval

A small, runnable retrieval-for-RAG pipeline: two lexical retrievers (BM25
and TF-IDF/cosine), a score-ensemble that blends them and applies a
business-logic boost, an optional LLM reranking pass, and a recall@k / MRR
evaluation harness against a labeled query set.

This is an original, from-scratch demo — not production code — built to show
the same retrieval-pipeline shape (multiple retrieval signals blended, then
business rules applied on top, then measured against a labeled eval set)
used in retrieval/RAG systems I've built professionally.

## Why this design

- **Two retrievers that fail differently.** BM25 (sparse, term-frequency
  based) and TF-IDF+cosine (vectorized lexical overlap) are both lexical
  methods, but they score and rank candidates differently on the same
  corpus — which is enough to demonstrate why ensembling multiple retrieval
  signals beats relying on one. (A production system would pair BM25 with a
  *neural* embedding retriever for real semantic recall on paraphrased
  queries; that's the natural next swap-in — see below.)
- **Ensemble = blended score + business rules, not just averaging.**
  `EnsembleRetriever` min-max normalizes both retrievers' scores so they're
  comparable, blends them, and then applies a business-logic penalty for
  archived/superseded documents. `examples/run_demo.py` shows a concrete
  case: a superseded 2023 refund-policy doc out-scores the current policy
  doc on pure lexical overlap (it repeats "refund policy" densely) — the
  ensemble's archive penalty fixes the ranking.
- **Recall@k and MRR, not eyeballing results.** `src/eval.py` implements
  both metrics against a small hand-labeled query set so retriever changes
  can be compared numerically instead of by spot-checking a few queries.
- **Reranking is a separate, optional stage.** `LLMReranker` is real,
  complete integration code for a second, more expensive pass over a
  retriever's short top-k candidate list — inactive by default (no bundled
  API key), same pattern as the other repos in this portfolio.

## What's real vs. mocked

| Component | Status |
|---|---|
| `BM25Retriever`, `TFIDFRetriever` | Real, run locally, no API key needed |
| `EnsembleRetriever` (score blending + archive penalty) | Real, fully functional |
| `evaluate_retriever` (recall@k, MRR) | Real, fully functional |
| `LLMReranker` | Real integration code, **inactive without an API key** |
| The 10-document corpus and 10-query eval set | Fictional placeholder content, small by design for a runnable demo |

Both retrievers here are lexical (term-overlap) methods, not neural
embeddings — that's a real limitation, not a simplification hidden from the
reader: a query that shares no vocabulary with the right document (true
paraphrase) will not be found by either. Swapping in a neural embedding
retriever as a third ensemble member is the natural next step and requires
no change to `EnsembleRetriever`'s blending logic — it already treats each
retriever as an interchangeable `Retriever`.

## Run it

```bash
pip install -r requirements.txt
python examples/run_demo.py
```

No API key needed for the default run.

To try the LLM reranking pass over a retriever's top-k instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
```

```python
from src import EnsembleRetriever, DOCUMENTS
from src.reranker import LLMReranker

retriever = EnsembleRetriever(DOCUMENTS)
candidates = retriever.retrieve("what's the current refund policy", k=5)
reranked = LLMReranker().rerank("what's the current refund policy", candidates)
```

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

## Layout

```
src/
  corpus.py         # small fictional document corpus
  retrievers.py       # BM25Retriever, TFIDFRetriever, EnsembleRetriever
  reranker.py           # IdentityReranker (default), LLMReranker
  eval.py                # recall@k, MRR, evaluate_retriever
examples/run_demo.py      # retriever comparison + archive-penalty case study
tests/                      # pytest suite
```
