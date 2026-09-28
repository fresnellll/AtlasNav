"""Leakage-isolated benchmark adapters.

Every adapter emits the same three-way contract:

``documents.jsonl`` or ``documents.parquet``
    Agent-visible canonical corpus input.
``questions.jsonl``
    Agent-visible query IDs and query text only.
``scoring.jsonl``
    Evaluator-only answers, qrels, decompositions, and metadata.

The physical separation is deliberate: Atlas construction and Router training
must never receive the evaluator-side file.
"""

from .prepare import prepare_benchmark

__all__ = ["prepare_benchmark"]
