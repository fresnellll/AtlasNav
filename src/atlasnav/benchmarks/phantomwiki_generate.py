"""Generate resumable independent PhantomWiki world shards for scale studies."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from atlasnav.io import atomic_json, sha256_file


OFFICIAL_REVISION = "5542d6b2ba9a76cc62bd107b71934f68339ade34"
QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')
SCHEMA = pa.schema([
    pa.field("docid", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("url", pa.string(), nullable=False),
])


def _revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _namespace(title: str, article: str, facts: list[str], world: int) -> tuple[str, str]:
    prefix = f"World-{world:03d}"
    literals = {title}
    for fact in facts:
        literals.update(match.group(1) for match in QUOTED.finditer(fact))
    mapping = {value: f"{prefix} {value}" for value in literals if value not in {"male", "female"}}
    pattern = re.compile("|".join(re.escape(value) for value in sorted(mapping, key=len, reverse=True)))
    return mapping[title], pattern.sub(lambda match: mapping[match.group(0)], article)


def _valid(directory: Path, world: int, revision: str) -> bool:
    parquet, manifest = directory / "data.parquet", directory / "manifest.json"
    if not parquet.is_file() or not manifest.is_file():
        return False
    value = json.loads(manifest.read_text(encoding="utf-8"))
    return (
        value.get("world") == world and value.get("official_revision") == revision
        and value.get("parquet_sha256") == sha256_file(parquet)
        and value.get("documents") == pq.ParquetFile(parquet).metadata.num_rows
    )


def generate_world_shards(
    *,
    official_repository: Path,
    output: Path,
    world_start: int = 20,
    world_stop: int = 200,
    expected_revision: str = OFFICIAL_REVISION,
    trees_per_world: int = 100,
    **_: Any,
) -> dict[str, Any]:
    """Generate ``[world_start, world_stop)`` using the official generator."""
    official_repository = official_repository.resolve()
    revision = _revision(official_repository)
    if revision != expected_revision:
        raise RuntimeError(f"PhantomWiki checkout is {revision}; expected {expected_revision}")
    source = official_repository / "src"
    sys.path.insert(0, str(source))
    try:
        from phantom_wiki.core.article import get_articles
        from phantom_wiki.facts import get_database
        from phantom_wiki.facts.attributes import db_generate_attributes
        from phantom_wiki.facts.family import db_generate_family
        from phantom_wiki.facts.friends import db_generate_friendships
    except ImportError as error:
        raise RuntimeError("install the pinned PhantomWiki generator dependencies first") from error

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for world in range(world_start, world_stop):
        directory = output / f"world_{world:03d}"
        if _valid(directory, world, revision):
            manifests.append(json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
            continue
        if directory.exists():
            raise FileExistsError(f"incomplete shard requires review: {directory}")
        directory.mkdir()
        seed = 1_000_000 + world
        friendship_seed = 2_000_000 + world
        database = get_database()
        db_generate_family(database, seed, False, False, str(directory), False, 20, 5, 50, 0.0,
                           trees_per_world)
        db_generate_friendships(database, 3, friendship_seed, False, str(directory))
        db_generate_attributes(database, seed)
        names = database.get_person_names()
        articles = get_articles(database, names)
        rows: list[dict[str, str]] = []
        for index, raw_title in enumerate(names):
            article, facts = articles[raw_title]
            title, isolated = _namespace(
                str(raw_title), str(article), [str(value) for value in facts], world
            )
            digest = hashlib.sha256(f"{world}\0{index}\0{title}".encode()).hexdigest()[:12]
            docid = f"PW{world:03d}{index:05d}{digest}"
            rows.append({
                "docid": docid, "text": f"Title: {title}\n{isolated.strip()}",
                "url": f"https://phantomwiki.invalid/world/{world:03d}/{docid}",
            })
        if len(rows) < 5_000:
            raise RuntimeError(f"world {world} generated only {len(rows)} documents")
        parquet = directory / "data.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), parquet, compression="zstd")
        manifest = {
            "schema": "atlasnav_phantomwiki_world_v1", "world": world,
            "documents": len(rows), "seed": seed, "friendship_seed": friendship_seed,
            "official_revision": revision, "parquet_sha256": sha256_file(parquet),
            "questions_generated": 0, "paid_api_calls": 0,
        }
        atomic_json(directory / "manifest.json", manifest)
        manifests.append(manifest)
    summary = {
        "schema": "atlasnav_phantomwiki_world_shards_v1", "world_start": world_start,
        "world_stop": world_stop, "worlds": len(manifests),
        "documents": sum(int(row["documents"]) for row in manifests),
        "official_revision": revision, "paid_api_calls": 0,
    }
    atomic_json(output / "manifest.json", summary)
    return summary
