"""Resumable four-view corpus encoding."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap

from atlasnav.atlas.signatures import VIEWS, document_signatures
from atlasnav.embeddings.client import EmbeddingBatch, EmbeddingClient
from atlasnav.io import atomic_json, sha256_file, stable_json


@dataclass(frozen=True)
class WorkItem:
    document_index: int
    view_index: int
    document_id: str
    signature: str
    signature_sha256: str


class Cache:
    def __init__(self, path: Path, dimensions: int) -> None:
        self.dimensions = dimensions
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
              document_index INTEGER PRIMARY KEY, document_id TEXT NOT NULL UNIQUE,
              title TEXT NOT NULL, url TEXT NOT NULL, domain TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embeddings (
              document_index INTEGER NOT NULL, view_index INTEGER NOT NULL,
              document_id TEXT NOT NULL, signature_sha256 TEXT NOT NULL,
              dimensions INTEGER NOT NULL, vector_f16 BLOB NOT NULL,
              PRIMARY KEY(document_index, view_index)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS requests (
              request_id INTEGER PRIMARY KEY AUTOINCREMENT, rows INTEGER NOT NULL,
              input_tokens INTEGER NOT NULL, attempts INTEGER NOT NULL,
              model TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def register_file(self, index: int, document_id: str, title: str, url: str, domain: str) -> None:
        expected = (document_id, title, url, domain)
        row = self.connection.execute(
            "SELECT document_id,title,url,domain FROM files WHERE document_index=?", (index,)
        ).fetchone()
        if row is not None and tuple(row) != expected:
            raise RuntimeError(f"embedding cache file mismatch at row {index}")
        if row is None:
            self.connection.execute("INSERT INTO files VALUES (?,?,?,?,?)", (index, *expected))

    def has(self, item: WorkItem) -> bool:
        row = self.connection.execute(
            "SELECT document_id,signature_sha256,dimensions FROM embeddings WHERE document_index=? AND view_index=?",
            (item.document_index, item.view_index),
        ).fetchone()
        if row is None:
            return False
        if tuple(row) != (item.document_id, item.signature_sha256, self.dimensions):
            raise RuntimeError(f"embedding cache signature mismatch at {item.document_index}/{item.view_index}")
        return True

    def store(self, items: Sequence[WorkItem], response: EmbeddingBatch) -> None:
        if len(items) != len(response.vectors):
            raise RuntimeError("embedding response cardinality mismatch")
        with self.connection:
            for item, vector in zip(items, response.vectors):
                if vector.shape != (self.dimensions,):
                    raise RuntimeError(f"embedding dimension mismatch: {vector.shape}")
                self.connection.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
                    (item.document_index, item.view_index, item.document_id, item.signature_sha256,
                     self.dimensions, np.asarray(vector, dtype=np.float16).tobytes(order="C")),
                )
            self.connection.execute(
                "INSERT INTO requests(rows,input_tokens,attempts,model) VALUES (?,?,?,?)",
                (len(items), response.input_tokens, response.attempts, response.model),
            )

    def counts(self) -> tuple[int, int, int, int]:
        files = int(self.connection.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        embeddings = int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
        requests, tokens = self.connection.execute("SELECT COUNT(*),COALESCE(SUM(input_tokens),0) FROM requests").fetchone()
        return files, embeddings, int(requests), int(tokens)


def _corpus(corpus_directory: Path) -> tuple[dict, Path]:
    manifest = json.loads((corpus_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "atlasnav_canonical_corpus_v1":
        raise RuntimeError("canonical corpus manifest is incompatible")
    path = corpus_directory / str(manifest["documents_file"])
    if sha256_file(path) != manifest["documents_sha256"]:
        raise RuntimeError("canonical corpus checksum mismatch")
    return manifest, path


def _items(parquet_path: Path, cache: Cache) -> Iterator[WorkItem]:
    index = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=128, columns=["docid", "text", "url"]):
        for row in batch.to_pylist():
            document_id = str(row["docid"])
            value = document_signatures(document_id, str(row["text"] or ""), str(row["url"] or ""))
            cache.register_file(index, document_id, value.title, str(row["url"] or ""), value.domain)
            for view_index, view in enumerate(VIEWS):
                signature = value.signatures[view]
                item = WorkItem(index, view_index, document_id, signature, hashlib.sha256(signature.encode()).hexdigest())
                if not cache.has(item):
                    yield item
            index += 1
            if index % 256 == 0:
                cache.connection.commit()
    cache.connection.commit()


async def build_embeddings(
    corpus_directory: Path,
    output_directory: Path,
    client: EmbeddingClient,
    dimensions: int = 2560,
    batch_size: int = 20,
    maximum_concurrency: int = 30,
    ramp_start: int = 2,
    ramp_seconds: float = 180.0,
) -> dict:
    if batch_size < 1 or maximum_concurrency < 1 or ramp_start < 1:
        raise ValueError("batch and concurrency settings must be positive")
    corpus_directory, output_directory = corpus_directory.resolve(), output_directory.resolve()
    corpus_manifest, parquet_path = _corpus(corpus_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    final_manifest = output_directory / "manifest.json"
    if final_manifest.exists():
        raise FileExistsError(f"embedding bundle is already finalized: {output_directory}")
    cache = Cache(output_directory / "embedding_cache.sqlite3", dimensions)
    started = time.monotonic()
    pending: asyncio.Queue[list[WorkItem] | None] = asyncio.Queue(maxsize=maximum_concurrency * 3)
    write_lock = asyncio.Lock()

    async def producer() -> None:
        batch: list[WorkItem] = []
        for item in _items(parquet_path, cache):
            batch.append(item)
            if len(batch) >= batch_size:
                await pending.put(batch)
                batch = []
        if batch:
            await pending.put(batch)
        for _ in range(maximum_concurrency):
            await pending.put(None)

    async def worker(worker_index: int) -> None:
        if worker_index >= ramp_start and ramp_seconds > 0:
            await asyncio.sleep(ramp_seconds * (worker_index - ramp_start + 1) / max(1, maximum_concurrency - ramp_start + 1))
        while True:
            items = await pending.get()
            try:
                if items is None:
                    return
                response = await client.embed([item.signature for item in items])
                async with write_lock:
                    cache.store(items, response)
                    _, completed, requests, tokens = cache.counts()
                    if requests % 50 == 0:
                        print(f"encoded={completed:,} requests={requests:,} input_tokens={tokens:,}", flush=True)
            finally:
                pending.task_done()

    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(producer())
            for index in range(maximum_concurrency):
                group.create_task(worker(index))
    except* Exception as errors:
        # TaskGroup cancels the producer and all sibling workers immediately,
        # so a provider failure cannot leave a full queue waiting forever.
        raise errors.exceptions[0]
    files, embeddings, requests, input_tokens = cache.counts()
    expected = int(corpus_manifest["documents"])
    if files != expected or embeddings != expected * len(VIEWS):
        raise RuntimeError(f"incomplete embedding cache: files={files}, embeddings={embeddings}")
    arrays: dict[str, str] = {}
    for view_index, view in enumerate(VIEWS):
        path = output_directory / f"{view}.f16.npy"
        matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(files, dimensions))
        cursor = cache.connection.execute(
            "SELECT document_index,vector_f16 FROM embeddings WHERE view_index=? ORDER BY document_index", (view_index,)
        )
        rows = 0
        for document_index, blob in cursor:
            matrix[int(document_index)] = np.frombuffer(blob, dtype=np.float16)
            rows += 1
        matrix.flush()
        if rows != files:
            raise RuntimeError(f"incomplete {view} matrix")
        arrays[view] = path.name
    catalog = output_directory / "catalog.jsonl.gz"
    with gzip.open(catalog, "wt", encoding="utf-8") as stream:
        for row in cache.connection.execute("SELECT document_index,document_id,title,url,domain FROM files ORDER BY document_index"):
            stream.write(stable_json({"index": row[0], "docid": row[1], "title": row[2], "url": row[3], "domain": row[4]}) + "\n")
    cache.connection.close()
    manifest = {
        "schema": "atlasnav_four_view_embeddings_v1",
        "finalized": True,
        "documents": files,
        "views": list(VIEWS),
        "model": client.model,
        "dimensions_per_view": dimensions,
        "vectors_concatenated_or_averaged": False,
        "construction_reads_evaluation_artifacts": False,
        "corpus_sha256": corpus_manifest["documents_sha256"],
        "catalog": catalog.name,
        "catalog_sha256": sha256_file(catalog),
        "view_embeddings": arrays,
        "view_embedding_sha256": {view: sha256_file(output_directory / path) for view, path in arrays.items()},
        "requests": requests,
        "input_tokens": input_tokens,
        "wall_time_seconds": time.monotonic() - started,
    }
    atomic_json(final_manifest, manifest)
    return manifest
