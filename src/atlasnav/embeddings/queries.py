"""Resumable four-view query encoding for runtime construction."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from atlasnav.atlas.signatures import VIEWS, query_signatures
from atlasnav.embeddings.client import EmbeddingBatch, EmbeddingClient
from atlasnav.io import atomic_json, sha256_file, stable_json


@dataclass(frozen=True)
class QueryItem:
    query_index: int
    view_index: int
    query_id: str
    signature: str
    signature_sha256: str


def load_queries(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("query_id") is None or value.get("query") is None:
                raise ValueError(f"query_id/query object required at line {line_number}")
            rows.append({"query_id": str(value["query_id"]), "query": str(value["query"]).strip()})
    identifiers = [row["query_id"] for row in rows]
    if not rows or len(set(identifiers)) != len(identifiers) or any(not row["query"] for row in rows):
        raise ValueError("queries must be non-empty with unique IDs")
    return rows


class QueryCache:
    def __init__(self, path: Path, dimensions: int) -> None:
        self.dimensions = dimensions
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS queries(
              query_index INTEGER PRIMARY KEY, query_id TEXT NOT NULL UNIQUE, query TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings(
              query_index INTEGER NOT NULL, view_index INTEGER NOT NULL,
              query_id TEXT NOT NULL, signature_sha256 TEXT NOT NULL,
              dimensions INTEGER NOT NULL, vector_f16 BLOB NOT NULL,
              PRIMARY KEY(query_index,view_index)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS requests(
              request_id INTEGER PRIMARY KEY AUTOINCREMENT, rows INTEGER NOT NULL,
              input_tokens INTEGER NOT NULL, attempts INTEGER NOT NULL, model TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def register(self, index: int, row: dict[str, str]) -> None:
        prior = self.connection.execute(
            "SELECT query_id,query FROM queries WHERE query_index=?", (index,)
        ).fetchone()
        expected = (row["query_id"], row["query"])
        if prior is not None and tuple(prior) != expected:
            raise RuntimeError(f"query cache identity mismatch at {index}")
        if prior is None:
            self.connection.execute("INSERT INTO queries VALUES (?,?,?)", (index, *expected))

    def has(self, item: QueryItem) -> bool:
        row = self.connection.execute(
            "SELECT query_id,signature_sha256,dimensions FROM embeddings WHERE query_index=? AND view_index=?",
            (item.query_index, item.view_index),
        ).fetchone()
        if row is None:
            return False
        if tuple(row) != (item.query_id, item.signature_sha256, self.dimensions):
            raise RuntimeError(f"query embedding cache mismatch at {item.query_index}/{item.view_index}")
        return True

    def store(self, items: Sequence[QueryItem], response: EmbeddingBatch) -> None:
        if len(items) != len(response.vectors):
            raise RuntimeError("query embedding response cardinality mismatch")
        with self.connection:
            for item, vector in zip(items, response.vectors):
                if vector.shape != (self.dimensions,):
                    raise RuntimeError(f"query embedding dimension mismatch: {vector.shape}")
                self.connection.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
                    (
                        item.query_index, item.view_index, item.query_id,
                        item.signature_sha256, self.dimensions,
                        np.asarray(vector, dtype=np.float16).tobytes(order="C"),
                    ),
                )
            self.connection.execute(
                "INSERT INTO requests(rows,input_tokens,attempts,model) VALUES (?,?,?,?)",
                (len(items), response.input_tokens, response.attempts, response.model),
            )


async def build_query_embeddings(
    dataset: Path,
    output_directory: Path,
    client: EmbeddingClient,
    *,
    dimensions: int = 2560,
    batch_size: int = 20,
    maximum_concurrency: int = 30,
    ramp_start: int = 2,
    ramp_seconds: float = 60.0,
) -> dict[str, Any]:
    if not 1 <= ramp_start <= maximum_concurrency or batch_size < 1:
        raise ValueError("invalid query embedding batch/concurrency settings")
    dataset = dataset.resolve()
    rows = load_queries(dataset)
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if (output_directory / "manifest.json").exists():
        raise FileExistsError(f"query bundle is already finalized: {output_directory}")
    cache = QueryCache(output_directory / "embedding_cache.sqlite3", dimensions)
    work: list[QueryItem] = []
    for index, row in enumerate(rows):
        cache.register(index, row)
        for view_index, view in enumerate(VIEWS):
            signature = query_signatures(row["query"])[view]
            item = QueryItem(
                index, view_index, row["query_id"], signature,
                hashlib.sha256(signature.encode()).hexdigest(),
            )
            if not cache.has(item):
                work.append(item)
    cache.connection.commit()
    batches = [tuple(work[index:index + batch_size]) for index in range(0, len(work), batch_size)]
    started = time.monotonic()

    def concurrency() -> int:
        if ramp_seconds <= 0:
            return maximum_concurrency
        fraction = min(1.0, (time.monotonic() - started) / ramp_seconds)
        return min(maximum_concurrency, ramp_start + int((maximum_concurrency - ramp_start) * fraction))

    async def request(items: tuple[QueryItem, ...]) -> tuple[tuple[QueryItem, ...], EmbeddingBatch]:
        return items, await client.embed([item.signature for item in items])

    pending: set[asyncio.Task[tuple[tuple[QueryItem, ...], EmbeddingBatch]]] = set()
    try:
        for batch in batches:
            pending.add(asyncio.create_task(request(batch)))
            if len(pending) >= concurrency():
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    cache.store(*task.result())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                cache.store(*task.result())
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    queries = int(cache.connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0])
    embeddings = int(cache.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    requests, input_tokens = cache.connection.execute(
        "SELECT COUNT(*),COALESCE(SUM(input_tokens),0) FROM requests"
    ).fetchone()
    if queries != len(rows) or embeddings != len(rows) * len(VIEWS):
        raise RuntimeError(f"incomplete query bundle: queries={queries}, embeddings={embeddings}")
    arrays: dict[str, str] = {}
    for view_index, view in enumerate(VIEWS):
        path = output_directory / f"{view}.f16.npy"
        matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(queries, dimensions))
        for query_index, blob in cache.connection.execute(
            "SELECT query_index,vector_f16 FROM embeddings WHERE view_index=? ORDER BY query_index",
            (view_index,),
        ):
            matrix[int(query_index)] = np.frombuffer(blob, dtype=np.float16)
        matrix.flush()
        arrays[view] = path.name
    catalog = output_directory / "catalog.jsonl.gz"
    with gzip.open(catalog, "wt", encoding="utf-8") as stream:
        for query_index, query_id, query in cache.connection.execute(
            "SELECT query_index,query_id,query FROM queries ORDER BY query_index"
        ):
            stream.write(stable_json({"index": query_index, "query_id": query_id, "query": query}) + "\n")
    cache.connection.close()
    manifest = {
        "schema": "atlasnav_four_view_query_embeddings_v1",
        "finalized": True,
        "queries": queries,
        "views": list(VIEWS),
        "model": client.model,
        "dimensions_per_view": dimensions,
        "dataset_sha256": sha256_file(dataset),
        "query_catalog": catalog.name,
        "query_catalog_sha256": sha256_file(catalog),
        "query_embeddings": arrays,
        "query_embedding_sha256": {
            view: sha256_file(output_directory / path) for view, path in arrays.items()
        },
        "requests": int(requests),
        "input_tokens": int(input_tokens),
        "construction_reads_answers_qrels_or_trajectories": False,
        "wall_time_seconds": time.monotonic() - started,
    }
    atomic_json(output_directory / "manifest.json", manifest)
    return manifest
