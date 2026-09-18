"""A small fictional document corpus to retrieve over.

The content is intentionally toy (a handful of product-docs-style passages)
-- the point of this repo is the retrieval architecture, not the corpus.
Swap `DOCUMENTS` for a real corpus and everything else keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "docs"
    recency_rank: int = 0  # 0 = most recent; higher = older. Used by the
    # business-logic boost in the ensemble reranker as a stand-in for "prefer
    # the current version of a doc over a stale one."


DOCUMENTS: list[Document] = [
    Document(
        "d1",
        "Resetting your password",
        "To reset your password, open the login screen and select 'Forgot "
        "password'. You'll receive an email with a reset link that expires "
        "in 24 hours.",
        recency_rank=0,
    ),
    Document(
        "d2",
        "Two-factor authentication setup",
        "Enable two-factor authentication from Settings > Security. You can "
        "use an authenticator app or SMS codes. We recommend an authenticator "
        "app since SMS can be intercepted.",
        recency_rank=0,
    ),
    Document(
        "d3",
        "Refund policy",
        "Purchases can be refunded in full within 30 days of the charge date. "
        "After 30 days, refunds are handled on a case-by-case basis by "
        "support.",
        recency_rank=0,
    ),
    Document(
        "d4",
        "Refund policy (2023 archived version)",
        "Purchases can be refunded in full within 14 days of the charge date. "
        "This document is superseded by the current refund policy.",
        source="archive",
        recency_rank=5,
    ),
    Document(
        "d5",
        "Clearing the local cache",
        "If the app is running slowly, clearing the local cache often helps. "
        "Go to Settings > Storage > Clear Cache. This does not delete your "
        "account data.",
        recency_rank=0,
    ),
    Document(
        "d6",
        "Sync client troubleshooting",
        "If the sync client is slow or stuck, check your network connection "
        "first, then try clearing its local cache from the sync client's "
        "settings menu.",
        recency_rank=0,
    ),
    Document(
        "d7",
        "Billing cycle and invoices",
        "Invoices are generated on the first of each billing cycle and "
        "emailed to the account owner. You can view past invoices under "
        "Billing > History.",
        recency_rank=0,
    ),
    Document(
        "d8",
        "Cancelling a subscription",
        "You can cancel your subscription at any time from Billing > "
        "Subscription > Cancel. Cancelling stops future charges; it does not "
        "automatically issue a refund for the current period.",
        recency_rank=0,
    ),
    Document(
        "d9",
        "API rate limits",
        "The API enforces a default rate limit of 100 requests per minute per "
        "API key. Exceeding this returns a 429 response with a Retry-After "
        "header.",
        recency_rank=0,
    ),
    Document(
        "d10",
        "Exporting your data",
        "You can export all of your account data as a zip archive from "
        "Settings > Privacy > Export Data. Exports are emailed as a download "
        "link within a few minutes.",
        recency_rank=0,
    ),
]
