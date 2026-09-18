"""Pluggable retrievers plus a score-ensemble that combines them.

Two retrieval strategies catch different things:

- BM25 is a sparse, keyword/term-frequency method. It's strong when the
  query shares exact vocabulary with the right document, and weak on
  paraphrase (a query about "slow sync" won't match a doc that only says
  "sluggish performance").
- TF-IDF + cosine similarity is also lexical, but the vectorization+cosine
  scoring rewards partial/fuzzy term overlap differently than BM25's term-
  frequency saturation curve does, so the two rank candidates differently
  even on the same corpus. (A real system would pair BM25 with a *neural*
  embedding retriever for true semantic recall; that's the natural next
  swap-in here -- see README.)

`EnsembleRetriever` normalizes and blends both retrievers' scores, then
applies a small business-logic boost (penalize archived/stale docs) before
returning the final ranking -- the same "combine multiple signals, then
apply business rules on top of the raw similarity score" shape used in
production retrieval-plus-reranking pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Document


@dataclass
class ScoredDoc:
    doc: Document
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[ScoredDoc]: ...


class BM25Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._corpus_tokens = [_tokenize(d.text + " " + d.title) for d in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, query: str, k: int = 5) -> list[ScoredDoc]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        return [ScoredDoc(doc, float(score)) for doc, score in ranked[:k]]


class TFIDFRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([d.text + " " + d.title for d in documents])

    def retrieve(self, query: str, k: int = 5) -> list[ScoredDoc]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        return [ScoredDoc(doc, float(score)) for doc, score in ranked[:k]]


def _normalize(scored: list[ScoredDoc]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1] so BM25's unbounded scores and
    TF-IDF cosine's [0, 1] scores can be blended on comparable terms."""
    if not scored:
        return {}
    values = [s.score for s in scored]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return {s.doc.doc_id: 1.0 for s in scored}
    return {s.doc.doc_id: (s.score - lo) / span for s in scored}


class EnsembleRetriever:
    """Blends BM25 and TF-IDF rankings, then applies a business-logic boost.

    Each sub-retriever is queried over the *full* corpus (not just its own
    top-k) so that a document either retriever considers weak still gets a
    fair combined score rather than being dropped from consideration just
    because it wasn't in one retriever's top-k.
    """

    def __init__(
        self,
        documents: list[Document],
        bm25_weight: float = 0.5,
        tfidf_weight: float = 0.5,
        archive_penalty: float = 0.6,
    ):
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.tfidf = TFIDFRetriever(documents)
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        self.archive_penalty = archive_penalty

    def retrieve(self, query: str, k: int = 5) -> list[ScoredDoc]:
        n = len(self.documents)
        bm25_scores = _normalize(self.bm25.retrieve(query, k=n))
        tfidf_scores = _normalize(self.tfidf.retrieve(query, k=n))

        combined: list[ScoredDoc] = []
        for doc in self.documents:
            blended = (
                self.bm25_weight * bm25_scores.get(doc.doc_id, 0.0)
                + self.tfidf_weight * tfidf_scores.get(doc.doc_id, 0.0)
            )
            # Business-logic boost: penalize archived/superseded docs so a
            # stale document doesn't outrank its current replacement purely
            # on lexical overlap.
            if doc.source == "archive":
                blended -= self.archive_penalty
            combined.append(ScoredDoc(doc, blended))

        combined.sort(key=lambda sd: sd.score, reverse=True)
        return combined[:k]
