"""Streaming access to frozen trajectory packages."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterator

import zstandard as zstd


def archive_members(path: Path) -> Iterator[tuple[str, bytes]]:
    with path.open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    stream = archive.extractfile(member)
                    if stream is not None:
                        yield member.name, stream.read()


def read_package_metadata(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest: dict[str, Any] | None = None
    index: list[dict[str, Any]] | None = None
    for name, data in archive_members(path):
        if name.endswith("/manifest.json"):
            manifest = json.loads(data)
        elif name.endswith("/per_query.jsonl") or name.endswith("/index.jsonl"):
            index = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if manifest is not None and index is not None:
            break
    if manifest is None or index is None:
        raise ValueError(f"trajectory package lacks manifest/index: {path}")
    return manifest, index

