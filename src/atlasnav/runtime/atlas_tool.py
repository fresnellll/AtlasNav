#!/usr/bin/env python3
"""Navigate the complete global file atlas from one query workspace."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gzip
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np


PARENT_RE = re.compile(r"^C(\d{3})$")
CHILD_RE = re.compile(r"^C(\d{3})/S(\d{2})$")
HANDLE_RE = re.compile(r"^D(.+)$")
TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
LEDGER_LOCK_FILENAME = ".lead_ledger.lock"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    overview = sub.add_parser("overview", help="Page through all parent regions.")
    overview.add_argument("--offset", type=int, default=0)
    overview.add_argument("--limit", type=int, default=20)
    expand = sub.add_parser("expand", help="Expand a parent or hierarchical child region.")
    expand.add_argument("address")
    expand.add_argument("--offset", type=int, default=0)
    expand.add_argument("--limit", type=int, default=20)
    files = sub.add_parser("files", help="List query-ranked files in any region.")
    files.add_argument("address")
    files.add_argument("--offset", type=int, default=0)
    files.add_argument("--limit", type=int, default=20)
    locate = sub.add_parser("locate", help="Find the complete map address of a canonical handle.")
    locate.add_argument("handle")
    search = sub.add_parser("search", help="Full-corpus or region-scoped lexical search.")
    search.add_argument("query")
    search.add_argument("--cluster")
    search.add_argument("--mode", choices=("all", "any", "phrase", "raw"), default="all")
    search.add_argument("--limit", type=int, default=20)
    route = sub.add_parser("route", help="Re-route a new lexical clue over the complete corpus.")
    route.add_argument("query")
    route.add_argument("--mode", choices=("all", "any", "phrase", "raw"), default="any")
    route.add_argument("--limit", type=int, default=12)
    leads = sub.add_parser("leads", help="Show the persistent surfaced-but-unresolved lead ledger.")
    leads.add_argument("--limit", type=int, default=15)
    leads.add_argument("--all", action="store_true", help="Include opened and dismissed leads.")
    open_doc = sub.add_parser("open", help="Read a canonical document and mark its lead opened.")
    open_doc.add_argument("handle")
    open_doc.add_argument("--find", help="Prefer line windows matching these literal terms.")
    open_doc.add_argument("--start", type=int, default=1)
    open_doc.add_argument("--lines", type=int, default=160)
    open_doc.add_argument("--context", type=int, default=2)
    dismiss = sub.add_parser("dismiss", help="Mark one surfaced lead as checked and irrelevant.")
    dismiss.add_argument("handle")
    dismiss.add_argument("--reason", default="does not satisfy the unresolved clue")
    return result


def clean(value: object, limit: int = 300) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())[:limit]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object expected: {path}")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def exclusive_ledger_lock(workspace: Path):
    """Serialize every load/mutate/save cycle for a query's shared ledger.

    Atlas commands are separate processes, and an agent can issue tool calls
    concurrently.  Atomic replacement prevents torn JSON but does not prevent
    two processes from loading the same old value and overwriting one another.
    Callers hold this lock only for the ledger transaction. Expensive retrieval
    remains concurrent, while each transaction reloads the latest snapshot
    before merging its mutation.
    """

    config = load_json(workspace / "WORKSPACE.json")
    state_dir = Path(str(config["state_dir"])).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / LEDGER_LOCK_FILENAME
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Atlas:
    def __init__(self) -> None:
        workspace = Path.cwd().resolve()
        self.workspace = workspace
        query_path = workspace / ".atlas_query.json"
        config_path = workspace / "WORKSPACE.json"
        if not query_path.is_file() or query_path.is_symlink() or not config_path.is_file():
            raise RuntimeError("run atlas.py inside a formal atlas query workspace")
        self.query = load_json(query_path)
        config = load_json(config_path)
        self.runtime = Path(str(config["atlas_runtime"])).resolve()
        manifest = load_json(self.runtime / "manifest.json")
        if manifest.get("schema") != "atlasnav_runtime_v1":
            raise RuntimeError("atlas runtime manifest mismatch")
        self.arm = str(config["arm"])
        self.state_dir = Path(str(config["state_dir"])).resolve()
        self.ledger_path = self.state_dir / "lead_ledger.json"
        self.query_index = int(self.query["query_index"])
        self.catalog = load_rows(self.runtime / "catalog.jsonl.gz")
        self.parent_cards = load_rows(self.runtime / "parent_cards.jsonl.gz")
        self.child_cards = load_rows(self.runtime / "child_cards.jsonl.gz")
        self.parent_assignment = np.load(
            self.runtime / "file_to_parent.i32.npy", mmap_mode="r"
        )
        self.leaf_assignment = np.load(self.runtime / "file_to_leaf.i32.npy", mmap_mode="r")
        self.parents = int(manifest.get("parents", len(self.parent_cards)))
        local_path = self.runtime / "leaf_local_index.i32.npy"
        parent_path = self.runtime / "leaf_to_parent.i32.npy"
        if local_path.is_file() and parent_path.is_file():
            self.leaf_local = np.load(local_path, mmap_mode="r")
            self.leaf_parent = np.load(parent_path, mmap_mode="r")
        else:
            children = int(manifest.get("children_per_parent", 10) or 10)
            leaves = len(self.child_cards)
            self.leaf_local = np.arange(leaves, dtype=np.int32) % children
            self.leaf_parent = np.arange(leaves, dtype=np.int32) // children
        self.leaf_by_address = {
            (int(self.leaf_parent[leaf]), int(self.leaf_local[leaf])): leaf
            for leaf in range(len(self.child_cards))
        }
        self.scores = np.load(self.runtime / "query_file_scores.f32.npy", mmap_mode="r")[
            self.query_index
        ]
        self.parent_order = [int(value) for value in self.query["parent_order"]]
        self.by_docid = {str(row["docid"]): index for index, row in enumerate(self.catalog)}
        self.ledger = self.load_ledger()

    def load_ledger(self) -> dict[str, Any]:
        with exclusive_ledger_lock(self.workspace):
            return self._load_ledger_unlocked()

    def _load_ledger_unlocked(self) -> dict[str, Any]:
        if self.ledger_path.exists():
            value = load_json(self.ledger_path)
            if value.get("schema") != "atlas_lead_ledger_v1":
                raise RuntimeError("atlas lead ledger schema mismatch")
            return value
        value: dict[str, Any] = {
            "schema": "atlas_lead_ledger_v1",
            "query_id": str(self.query.get("query_id")),
            "next_sequence": 1,
            "leads": {},
        }
        self.ledger = value
        initial = self.query.get("initial_leads")
        if isinstance(initial, list):
            self._add_leads_unlocked(initial, source="initial_route")
        self.save_ledger()
        return self.ledger

    def save_ledger(self) -> None:
        atomic_json(self.ledger_path, self.ledger)

    def add_leads(self, rows: list[dict[str, Any]], *, source: str) -> None:
        with exclusive_ledger_lock(self.workspace):
            self.ledger = self._load_ledger_unlocked()
            self._add_leads_unlocked(rows, source=source)
            self.save_ledger()

    def _add_leads_unlocked(self, rows: list[dict[str, Any]], *, source: str) -> None:
        leads = self.ledger["leads"]
        for row in rows:
            handle = str(row["handle"])
            sequence = int(self.ledger["next_sequence"])
            self.ledger["next_sequence"] = sequence + 1
            existing = leads.get(handle)
            if existing is None:
                existing = {
                    "handle": handle,
                    "status": "unresolved",
                    "title": clean(row.get("title"), 220),
                    "map_address": clean(row.get("map_address"), 20),
                    "preview": clean(row.get("preview"), 260),
                    "best_rank": int(row.get("rank", 999999)),
                    "first_sequence": sequence,
                    "last_sequence": sequence,
                    "sightings": 0,
                    "sources": [],
                }
                leads[handle] = existing
            existing["last_sequence"] = sequence
            existing["sightings"] = int(existing.get("sightings", 0)) + 1
            existing["best_rank"] = min(
                int(existing.get("best_rank", 999999)), int(row.get("rank", 999999))
            )
            if row.get("preview"):
                existing["preview"] = clean(row["preview"], 260)
            sources = existing.setdefault("sources", [])
            source_note = f"{source}:rank{int(row.get('rank', 0))}"
            if source_note not in sources:
                sources.append(source_note)
                del sources[:-6]

    def lead_status(self, handle: str) -> str:
        row = self.ledger["leads"].get(handle)
        return str(row.get("status")) if isinstance(row, dict) else "new"

    def parent_label(self, parent: int) -> str:
        card = self.parent_cards[parent]
        return clean(card.get("llm_label") or card.get("machine_label"), 160)

    def child_label(self, leaf: int) -> str:
        card = self.child_cards[leaf]
        return clean(card.get("llm_label") or card.get("machine_label"), 140)

    def address(self, file_index: int) -> str:
        parent = int(self.parent_assignment[file_index])
        leaf = int(self.leaf_assignment[file_index])
        return f"C{parent:03d}/S{int(self.leaf_local[leaf]):02d}"

    def best(self, members: np.ndarray, count: int) -> np.ndarray:
        if not len(members):
            return np.empty(0, dtype=np.int32)
        order = np.lexsort((members, -np.asarray(self.scores[members], dtype=np.float32)))
        return np.asarray(members[order[:count]], dtype=np.int32)

    def record(self, file_index: int) -> str:
        row = self.catalog[file_index]
        return (
            f"D{row['docid']}\t{self.address(file_index)}\t"
            f"{float(self.scores[file_index]):.8f}\t{clean(row.get('title'), 260)}\t"
            f"canonical://D{row['docid']}"
        )

    def preview(self, file_index: int, query: str | None = None, mode: str = "any") -> str:
        connection = sqlite3.connect(
            f"file:{self.runtime / 'fulltext.sqlite3'}?mode=ro", uri=True, timeout=30
        )
        try:
            if query:
                expression = self.expression(query, mode)
                row = connection.execute(
                    "SELECT snippet(documents_fts,0,'[',']',' … ',30) "
                    "FROM documents_fts WHERE rowid=? AND documents_fts MATCH ?",
                    (file_index + 1, expression),
                ).fetchone()
                if row is not None and clean(row[0]):
                    return clean(row[0], 260)
            row = connection.execute(
                "SELECT substr(text,1,600) FROM documents WHERE rowid=?",
                (file_index + 1,),
            ).fetchone()
            return clean(row[0] if row else "", 260)
        finally:
            connection.close()

    def overview(self, offset: int, limit: int) -> None:
        validate_page(offset, limit, max(100, self.parents))
        selected = self.parent_order[offset : offset + limit]
        print("route_rank\tparent\tfiles\tlabel\tbest_anchor")
        for route_rank, parent in enumerate(selected, offset + 1):
            members = np.flatnonzero(self.parent_assignment == parent)
            best = int(self.best(members, 1)[0])
            row = self.catalog[best]
            print(
                f"{route_rank}\tC{parent:03d}\t{len(members)}\t{self.parent_label(parent)}\t"
                f"D{row['docid']} {clean(row.get('title'), 120)}"
            )
        print(
            f"# shown={len(selected)} remaining={max(0, self.parents-offset-len(selected))}; "
            f"all {self.parents} regions remain reachable"
        )

    def parse_address(self, address: str) -> tuple[int, int | None]:
        child = CHILD_RE.fullmatch(address)
        if child:
            parent = int(child.group(1))
            local = int(child.group(2))
            if not 0 <= parent < self.parents or (parent, local) not in self.leaf_by_address:
                raise ValueError("invalid child address")
            return parent, int(self.leaf_by_address[(parent, local)])
        parent_match = PARENT_RE.fullmatch(address)
        if parent_match:
            parent = int(parent_match.group(1))
            if not 0 <= parent < self.parents:
                raise ValueError("invalid parent address")
            return parent, None
        raise ValueError("address must be C000 or C000/S00")

    def child_overview(self, parent: int) -> None:
        rows: list[tuple[float, int, int]] = []
        leaves = sorted(
            (leaf for leaf in range(len(self.child_cards)) if int(self.leaf_parent[leaf]) == parent),
            key=lambda leaf: int(self.leaf_local[leaf]),
        )
        for leaf in leaves:
            members = np.flatnonzero(self.leaf_assignment == leaf)
            best = int(self.best(members, 1)[0])
            rows.append((float(self.scores[best]), leaf, best))
        rows.sort(key=lambda value: (-value[0], value[1]))
        show_bridges = any(self.child_cards[leaf].get("bridges") for leaf in leaves)
        print(
            "rank\tchild\tfiles\tlabel\tbest_anchor\tstrong_cross_facet_links"
            if show_bridges else "rank\tchild\tfiles\tlabel\tbest_anchor"
        )
        for rank, (_, leaf, best) in enumerate(rows, 1):
            row = self.catalog[best]
            size = int(np.sum(self.leaf_assignment == leaf))
            rendered = (
                f"{rank}\tC{parent:03d}/S{int(self.leaf_local[leaf]):02d}\t{size}\t{self.child_label(leaf)}\t"
                f"D{row['docid']} {clean(row.get('title'), 120)}"
            )
            if show_bridges:
                rendered += "\t" + ",".join(
                    clean(value.get("target_address"), 20)
                    for value in self.child_cards[leaf].get("bridges", [])[:3]
                )
            print(rendered)
        print("# child regions are local and variable-width; they are not a flat global list")

    def files(self, address: str, offset: int, limit: int) -> None:
        validate_page(offset, limit, 50)
        parent, leaf = self.parse_address(address)
        members = (
            np.flatnonzero(self.parent_assignment == parent)
            if leaf is None
            else np.flatnonzero(self.leaf_assignment == leaf)
        )
        ordered = self.best(members, len(members))
        selected = ordered[offset : offset + limit]
        lead_rows: list[dict[str, Any]] = []
        print("handle\tmap_address\tk1_field_score\tstatus\ttitle\tcanonical_path")
        for rank, file_index in enumerate(selected, offset + 1):
            file_index = int(file_index)
            row = self.catalog[file_index]
            handle = f"D{row['docid']}"
            record = self.record(file_index).split("\t")
            print("\t".join(record[:3] + [self.lead_status(handle)] + record[3:]))
            lead_rows.append(
                {
                    "handle": handle,
                    "title": row.get("title"),
                    "map_address": record[1],
                    "rank": rank,
                    "preview": "",
                }
            )
        self.add_leads(lead_rows, source=f"files:{address}")
        print(
            f"# shown={len(selected)} remaining={max(0,len(ordered)-offset-len(selected))}; "
            "pagination reaches every file in this region; surfaced files were added to the lead ledger"
        )

    def expand(self, address: str, offset: int, limit: int) -> None:
        parent, leaf = self.parse_address(address)
        if self.arm in {"atlas_hier", "atlasnav"} and leaf is None and offset == 0:
            self.child_overview(parent)
            return
        self.files(address, offset, limit)

    def locate(self, handle: str) -> None:
        match = HANDLE_RE.fullmatch(handle)
        if match is None or match.group(1) not in self.by_docid:
            raise ValueError("unknown canonical handle")
        file_index = self.by_docid[match.group(1)]
        print("handle\tmap_address\ttitle\tcanonical_path")
        row = self.catalog[file_index]
        parent = int(self.parent_assignment[file_index])
        print(
            f"D{row['docid']}\t{self.address(file_index)}\t"
            f"{clean(row.get('title'),260)}\tcanonical://D{row['docid']}"
        )

    def expression(self, query: str, mode: str) -> str:
        if mode == "raw":
            if not query.strip():
                raise ValueError("raw expression is empty")
            return query.strip()
        if mode == "phrase":
            phrase = " ".join(query.split())
            if not phrase:
                raise ValueError("phrase is empty")
            return '"' + phrase.replace('"', '""') + '"'
        tokens = TOKEN_RE.findall(query)
        if not tokens:
            raise ValueError("query contains no searchable tokens")
        tokens = tokens[:16]
        quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
        return (" AND " if mode == "all" else " OR ").join(quoted)

    def lexical_rows(self, query: str, mode: str) -> list[sqlite3.Row]:
        connection = sqlite3.connect(
            f"file:{self.runtime / 'fulltext.sqlite3'}?mode=ro", uri=True, timeout=30
        )
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(
                "SELECT documents_fts.rowid rowid, documents.docid docid, documents.title title, "
                "bm25(documents_fts,1.0) score FROM documents_fts JOIN documents ON "
                "documents.rowid=documents_fts.rowid WHERE documents_fts MATCH ? ORDER BY score",
                (self.expression(query, mode),),
            ).fetchall()
        finally:
            connection.close()

    def search(self, query: str, cluster: str | None, mode: str, limit: int) -> None:
        validate_page(0, limit, 50)
        target_parent: int | None = None
        target_leaf: int | None = None
        if cluster:
            target_parent, target_leaf = self.parse_address(cluster)
        selected: list[sqlite3.Row] = []
        for row in self.lexical_rows(query, mode):
            file_index = int(row["rowid"]) - 1
            if target_parent is not None and int(self.parent_assignment[file_index]) != target_parent:
                continue
            if target_leaf is not None and int(self.leaf_assignment[file_index]) != target_leaf:
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
        lead_rows: list[dict[str, Any]] = []
        print("rank\thandle\tmap_address\tbm25\tstatus\ttitle\tmatch_preview\tcanonical_path")
        for rank, row in enumerate(selected, 1):
            file_index = int(row["rowid"]) - 1
            parent = int(self.parent_assignment[file_index])
            leaf = int(self.leaf_assignment[file_index])
            handle = f"D{row['docid']}"
            preview = self.preview(file_index, query, mode)
            print(
                f"{rank}\t{handle}\t{self.address(file_index)}\t"
                f"{float(row['score']):.8f}\t{self.lead_status(handle)}\t"
                f"{clean(row['title'],260)}\t{preview}\tcanonical://D{row['docid']}"
            )
            lead_rows.append(
                {
                    "handle": handle,
                    "title": row["title"],
                    "map_address": self.address(file_index),
                    "rank": rank,
                    "preview": preview,
                }
            )
        self.add_leads(lead_rows, source=f"search:{cluster or 'global'}")
        print(
            f"# shown={len(selected)}; search evaluated the canonical full-text index; "
            "results were queued in `python atlas.py leads`"
        )

    def route(self, query: str, mode: str, limit: int) -> None:
        validate_page(0, limit, max(100, self.parents))
        best: dict[int, sqlite3.Row] = {}
        counts: dict[int, int] = {}
        for row in self.lexical_rows(query, mode):
            file_index = int(row["rowid"]) - 1
            parent = int(self.parent_assignment[file_index])
            counts[parent] = counts.get(parent, 0) + 1
            best.setdefault(parent, row)
        ordered = sorted(best, key=lambda parent: (float(best[parent]["score"]), parent))[:limit]
        lead_rows: list[dict[str, Any]] = []
        print("rank\tparent\tmatching_files\tlabel\tbest_lexical_anchor\tmatch_preview")
        for rank, parent in enumerate(ordered, 1):
            row = best[parent]
            file_index = int(row["rowid"]) - 1
            leaf = int(self.leaf_assignment[file_index])
            preview = self.preview(file_index, query, mode)
            print(
                f"{rank}\tC{parent:03d}\t{counts[parent]}\t{self.parent_label(parent)}\t"
                f"D{row['docid']} {clean(row['title'],120)}\t{preview}"
            )
            lead_rows.append(
                {
                    "handle": f"D{row['docid']}",
                    "title": row["title"],
                    "map_address": self.address(file_index),
                    "rank": rank,
                    "preview": preview,
                }
            )
        self.add_leads(lead_rows, source="route")
        print(
            "# this re-route scanned the complete canonical FTS result set, not an initial pool; "
            "anchors were queued in `python atlas.py leads`"
        )

    def leads(self, limit: int, include_all: bool) -> None:
        validate_page(0, limit, 50)
        rows = list(self.ledger["leads"].values())
        if not include_all:
            rows = [row for row in rows if row.get("status") == "unresolved"]
        rows.sort(
            key=lambda row: (
                0 if row.get("status") == "unresolved" else 1,
                int(row.get("best_rank", 999999)),
                -int(row.get("last_sequence", 0)),
                str(row.get("handle")),
            )
        )
        selected = rows[:limit]
        print("handle\tstatus\tbest_rank\tsightings\tmap_address\ttitle\tlatest_preview")
        for row in selected:
            print(
                f"{row['handle']}\t{row['status']}\t{row['best_rank']}\t{row['sightings']}\t"
                f"{row.get('map_address','')}\t{clean(row.get('title'),180)}\t"
                f"{clean(row.get('preview'),260)}"
            )
        unresolved = sum(row.get("status") == "unresolved" for row in self.ledger["leads"].values())
        print(
            f"# shown={len(selected)} unresolved={unresolved}; open a credible lead with "
            "`python atlas.py open D123 --find \"decisive phrase\"` or explicitly dismiss it"
        )

    def require_handle(self, handle: str) -> tuple[str, int]:
        match = HANDLE_RE.fullmatch(handle)
        if match is None or match.group(1) not in self.by_docid:
            raise ValueError("unknown canonical handle")
        return f"D{match.group(1)}", self.by_docid[match.group(1)]

    def mark(self, handle: str, status: str, reason: str = "") -> int:
        handle, file_index = self.require_handle(handle)
        row = self.catalog[file_index]
        with exclusive_ledger_lock(self.workspace):
            self.ledger = self._load_ledger_unlocked()
            if handle not in self.ledger["leads"]:
                parent = int(self.parent_assignment[file_index])
                self._add_leads_unlocked(
                    [
                        {
                            "handle": handle,
                            "title": row.get("title"),
                            "map_address": self.address(file_index),
                            "rank": 999999,
                            "preview": "",
                        }
                    ],
                    source="direct",
                )
            lead = self.ledger["leads"][handle]
            if status == "opened":
                # Opening is an observed lifecycle fact and must not be erased
                # by a racing or later semantic dismissal.
                lead["opened"] = True
                lead["status"] = "opened"
            elif status == "dismissed":
                lead["dismissed"] = True
                if not lead.get("opened") and lead.get("status") != "opened":
                    lead["status"] = "dismissed"
            else:
                raise ValueError(f"unsupported lead status: {status}")
            if reason:
                reason_field = (
                    "post_open_dismissal_reason"
                    if lead.get("opened") and status == "dismissed"
                    else "resolution_reason"
                )
                lead[reason_field] = clean(reason, 300)
            self.save_ledger()
        return file_index

    def open_document(
        self, handle: str, find: str | None, start: int, lines: int, context: int
    ) -> None:
        if start < 1 or lines < 1 or lines > 400 or context < 0 or context > 10:
            raise ValueError("start must be >=1, lines in 1..400, and context in 0..10")
        handle, file_index = self.require_handle(handle)
        connection = sqlite3.connect(
            f"file:{self.runtime / 'fulltext.sqlite3'}?mode=ro", uri=True, timeout=30
        )
        try:
            row = connection.execute(
                "SELECT text FROM documents WHERE rowid=?", (file_index + 1,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise RuntimeError("canonical document is missing from the full-text index")
        text = str(row[0])
        body = text.splitlines()
        indexes: list[int] = []
        if find:
            tokens = [value.casefold() for value in TOKEN_RE.findall(find)]
            exact = find.casefold()
            indexes = [index for index, line in enumerate(body) if exact in line.casefold()]
            if not indexes and tokens:
                indexes = [
                    index
                    for index, line in enumerate(body)
                    if all(token in line.casefold() for token in tokens)
                ]
            if not indexes and tokens:
                indexes = [
                    index
                    for index, line in enumerate(body)
                    if any(token in line.casefold() for token in tokens)
                ]
        if indexes:
            chosen: list[int] = []
            for index in indexes[:12]:
                chosen.extend(range(max(0, index - context), min(len(body), index + context + 1)))
            line_ids = sorted(set(chosen))[:lines]
        else:
            line_ids = list(range(start - 1, min(len(body), start - 1 + lines)))
        print(f"# {handle} opened; title={clean(self.catalog[file_index].get('title'),220)}")
        if find and not indexes:
            print("# requested terms were not localized together; showing the requested/default line range")
        for index in line_ids:
            print(f"{index + 1:06d}\t{body[index]}")
        print(f"# shown_lines={len(line_ids)} document_lines={len(body)} lead_status=opened")
        # Commit only after the canonical body has been read and its output was
        # delivered to the parent process. A missing file or broken output pipe
        # must not create a false opened fact.
        sys.stdout.flush()
        self.mark(handle, "opened")

    def dismiss(self, handle: str, reason: str) -> None:
        handle, _ = self.require_handle(handle)
        self.mark(handle, "dismissed", reason)
        print(f"# {handle} marked dismissed: {clean(reason,300)}")


def validate_page(offset: int, limit: int, maximum: int) -> None:
    if offset < 0 or limit < 1 or limit > maximum:
        raise ValueError(f"offset must be nonnegative and limit must be in 1..{maximum}")


def main() -> None:
    args = parser().parse_args()
    atlas = Atlas()
    if args.command == "overview":
        atlas.overview(args.offset, args.limit)
    elif args.command == "expand":
        atlas.expand(args.address, args.offset, args.limit)
    elif args.command == "files":
        atlas.files(args.address, args.offset, args.limit)
    elif args.command == "locate":
        atlas.locate(args.handle)
    elif args.command == "search":
        atlas.search(args.query, args.cluster, args.mode, args.limit)
    elif args.command == "route":
        atlas.route(args.query, args.mode, args.limit)
    elif args.command == "leads":
        atlas.leads(args.limit, args.all)
    elif args.command == "open":
        atlas.open_document(args.handle, args.find, args.start, args.lines, args.context)
    else:
        atlas.dismiss(args.handle, args.reason)


if __name__ == "__main__":
    main()
