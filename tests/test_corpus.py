from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from atlasnav.corpus.prepare import build_fulltext_index, prepare_corpus


def test_prepare_and_search_canonical_corpus(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    source.write_text(
        json.dumps({"docid": "a", "text": "title: Cedar\nA cedar grows here.", "url": "https://example.org/a"}) + "\n"
        + json.dumps({"docid": "b", "text": "title: Birch\nA birch grows there."}) + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    manifest = prepare_corpus(source, corpus)
    assert manifest["documents"] == 2
    index = tmp_path / "fulltext.sqlite3"
    index_manifest = build_fulltext_index(corpus, index)
    assert index_manifest["documents"] == 2
    connection = sqlite3.connect(index)
    try:
        row = connection.execute(
            "SELECT documents.docid FROM documents_fts JOIN documents ON "
            "documents_fts.rowid=documents.rowid WHERE documents_fts MATCH 'cedar'"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("a",)
