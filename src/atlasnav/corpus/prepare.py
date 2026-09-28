"""Normalize upstream documents and construct an exact full-text index."""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from atlasnav.io import atomic_json, sha256_file, stable_json
from atlasnav.text import clean_text, domain_from_url, extract_title


SCHEMA = pa.schema([
    pa.field("docid", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("url", pa.string(), nullable=False),
])


@dataclass(frozen=True)
class CanonicalDocument:
    docid: str
    text: str
    url: str = ""


def _parquet_parts(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        parts = sorted(path.glob("*.parquet"))
        if parts:
            return parts
    return []


def iter_upstream(path: Path) -> Iterator[CanonicalDocument]:
    parts = _parquet_parts(path)
    if parts:
        for part in parts:
            parquet = pq.ParquetFile(part)
            names = set(parquet.schema.names)
            if not {"docid", "text"}.issubset(names):
                raise ValueError(f"Parquet input lacks docid/text: {part}")
            columns = ["docid", "text"] + (["url"] if "url" in names else [])
            for batch in parquet.iter_batches(batch_size=1024, columns=columns):
                for row in batch.to_pylist():
                    yield CanonicalDocument(str(row["docid"]), str(row.get("text") or ""), str(row.get("url") or ""))
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("docid") is None or value.get("text") is None:
                raise ValueError(f"Expected docid/text object at {path}:{line_number}")
            yield CanonicalDocument(str(value["docid"]), str(value["text"]), str(value.get("url") or ""))


def prepare_corpus(input_path: Path, output_directory: Path, batch_size: int = 1024) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to replace corpus bundle: {output_directory}")
    temporary = output_directory.with_name(f".{output_directory.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    parquet_path = temporary / "documents.parquet"
    catalog_path = temporary / "catalog.jsonl.gz"
    writer = pq.ParquetWriter(parquet_path, SCHEMA, compression="zstd")
    seen: set[str] = set()
    batch: list[dict[str, str]] = []
    count = 0
    try:
        with gzip.open(catalog_path, "wt", encoding="utf-8") as catalog:
            for source in iter_upstream(input_path):
                docid = source.docid.strip()
                if not docid or docid in seen:
                    raise ValueError(f"empty or duplicate document ID: {docid!r}")
                seen.add(docid)
                text = clean_text(source.text)
                url = source.url.strip()
                batch.append({"docid": docid, "text": text, "url": url})
                catalog.write(stable_json({
                    "index": count,
                    "docid": docid,
                    "title": extract_title(text, docid),
                    "url": url,
                    "domain": domain_from_url(url),
                }) + "\n")
                count += 1
                if len(batch) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        writer.close()
        manifest = {
            "schema": "atlasnav_canonical_corpus_v1",
            "documents": count,
            "physical_fields": ["docid", "text", "url"],
            "construction_reads_evaluation_artifacts": False,
            "documents_file": parquet_path.name,
            "documents_sha256": sha256_file(parquet_path),
            "catalog_file": catalog_path.name,
            "catalog_sha256": sha256_file(catalog_path),
        }
        atomic_json(temporary / "manifest.json", manifest)
        temporary.replace(output_directory)
        return manifest
    except BaseException:
        writer.close()
        raise


def build_fulltext_index(corpus_directory: Path, output_path: Path) -> dict[str, Any]:
    corpus_directory = corpus_directory.resolve()
    manifest = json.loads((corpus_directory / "manifest.json").read_text(encoding="utf-8"))
    documents = corpus_directory / str(manifest["documents_file"])
    if sha256_file(documents) != manifest["documents_sha256"]:
        raise RuntimeError("canonical corpus checksum mismatch")
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to replace full-text index: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".building-{os.getpid()}")
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE documents(
              rowid INTEGER PRIMARY KEY, docid TEXT NOT NULL UNIQUE,
              title TEXT NOT NULL, url TEXT NOT NULL, text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(text, content='documents', content_rowid='rowid', tokenize='unicode61');
            """
        )
        rowid = 1
        parquet = pq.ParquetFile(documents)
        for batch in parquet.iter_batches(batch_size=1024, columns=["docid", "url", "text"]):
            rows = [
                (
                    rowid + offset,
                    str(row["docid"]),
                    extract_title(str(row["text"]), str(row["docid"])),
                    str(row["url"]),
                    str(row["text"]),
                )
                for offset, row in enumerate(batch.to_pylist())
            ]
            connection.executemany("INSERT INTO documents VALUES (?,?,?,?,?)", rows)
            rowid += len(rows)
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        connection.commit()
    finally:
        connection.close()
    temporary.replace(output_path)
    return {
        "schema": "atlasnav_fulltext_index_v1",
        "documents": rowid - 1,
        "corpus_sha256": manifest["documents_sha256"],
        "index_sha256": sha256_file(output_path),
    }
