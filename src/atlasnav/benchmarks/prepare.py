"""Public benchmark preparation dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def prepare_benchmark(adapter: str, output: Path, **options: Any) -> dict[str, Any]:
    """Prepare one supported benchmark without importing unrelated extras."""
    normalized = adapter.casefold().replace("_", "-")
    if normalized in {"beir", "trec-covid", "scifact", "arguana"}:
        from .retrieval import prepare_retrieval

        return prepare_retrieval(adapter=normalized, output=output, **options)
    if normalized in {"2wiki", "2wiki-global-400"}:
        from .twowiki import prepare_2wiki

        return prepare_2wiki(output=output, **options)
    if normalized in {"enterprise", "enterpriserag", "enterpriserag-bench"}:
        from .enterprise import prepare_enterprise

        return prepare_enterprise(output=output, **options)
    if normalized in {"phantomwiki", "phantomwiki-scale"}:
        from .phantomwiki import prepare_phantomwiki

        return prepare_phantomwiki(output=output, **options)
    if normalized in {"fanoutqa", "fanout-qa"}:
        from .fanoutqa import prepare_fanoutqa

        return prepare_fanoutqa(output=output, **options)
    raise ValueError(f"unknown benchmark adapter: {adapter}")
