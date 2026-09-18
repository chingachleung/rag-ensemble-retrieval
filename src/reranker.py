"""Optional LLM reranking pass over a retriever's top-k candidates.

Retrieval (BM25/TF-IDF/ensemble) is cheap and runs over the whole corpus;
reranking is expensive and only makes sense over a short candidate list.
`LLMReranker` is real, complete integration code for that second pass --
inactive by default (no bundled API key), same pattern as the other repos
in this portfolio.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

from .retrievers import ScoredDoc


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredDoc]) -> list[ScoredDoc]: ...


class IdentityReranker:
    """No-op reranker: returns the candidates unchanged. Used as the default
    so the pipeline runs end-to-end with no API key."""

    def rerank(self, query: str, candidates: list[ScoredDoc]) -> list[ScoredDoc]:
        return candidates


class LLMReranker:
    """Asks the model to reorder a short candidate list by actual relevance
    to the query, which can fix cases where lexical retrieval ranks a
    tangentially-related document above the one that actually answers the
    question."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def rerank(self, query: str, candidates: list[ScoredDoc]) -> list[ScoredDoc]:
        if not self.api_key:
            raise RuntimeError(
                "LLMReranker requires ANTHROPIC_API_KEY. Use IdentityReranker for the no-key demo path."
            )

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        listing = "\n".join(f"{i}. [{c.doc.doc_id}] {c.doc.title}: {c.doc.text}" for i, c in enumerate(candidates))
        prompt = (
            f"Query: {query}\n\nCandidate documents:\n{listing}\n\n"
            "Return a JSON array of candidate indices (the leading numbers), "
            "ordered from most to least relevant to the query. JSON only."
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        order = json.loads(response.content[0].text)
        return [candidates[i] for i in order]
