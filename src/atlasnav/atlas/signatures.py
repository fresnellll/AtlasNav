"""Corpus-only Topic, Identity, Episode, and Relation signatures.

This module is intentionally isolated from benchmark questions, answers,
evidence judgments, trajectories, and correctness labels.  A document's title
and source domain are included because the Atlas clusters files, not passages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from atlasnav.text import (
    ENTITY_RE,
    NUMBER_RE,
    RELATION_TERMS,
    SENTENCE_RE,
    TIME_TERMS,
    YEAR_RE,
    clean_field,
    clean_text,
    domain_from_url,
    extract_title,
    tokens,
)


VIEWS = ("topic", "identity", "episode", "relation")
PLACE_EVENT_TERMS = {
    "battle", "campaign", "city", "conference", "country", "election", "event",
    "expedition", "festival", "france", "government", "incident", "meeting",
    "movement", "province", "revolution", "school", "tour", "treaty", "university",
    "war", "world", "china", "europe", "asia", "africa", "america",
}
IDENTITY_TERMS = {
    "album", "artist", "band", "book", "company", "film", "group", "institution",
    "organization", "person", "series", "team", "university", "work",
}


@dataclass(frozen=True)
class DocumentSignatures:
    document_id: str
    title: str
    domain: str
    signatures: dict[str, str]


def _sentences(text: str, title: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in SENTENCE_RE.split(clean_text(text)):
        value = clean_field(raw, 720)
        identity = value.casefold()
        if len(value) < 18 or identity in seen:
            continue
        seen.add(identity)
        values.append(value)
    return values or [clean_field(text or title, 720)]


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def _diverse(
    candidates: Iterable[tuple[float, int, str]],
    limit: int,
    similarity_ceiling: float = 0.68,
) -> list[str]:
    selected: list[str] = []
    selected_terms: list[set[str]] = []
    for _, _, sentence in sorted(candidates, key=lambda row: (-row[0], row[1])):
        current = tokens(sentence)
        if current and any(_jaccard(current, prior) >= similarity_ceiling for prior in selected_terms):
            continue
        selected.append(sentence)
        selected_terms.append(current)
        if len(selected) >= limit:
            break
    return selected


def _uniform(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return list(values)
    positions = np.linspace(0, len(values) - 1, num=limit, dtype=np.int64)
    return [values[int(position)] for position in positions]


def _render(
    view: str,
    document_id: str,
    title: str,
    domain: str,
    excerpts: list[str],
    maximum_characters: int,
) -> str:
    instruction = {
        "topic": "Represent this FILE by its overall subjects and independent content areas.",
        "identity": "Represent this FILE by named people, works, organizations, aliases, and identity-bearing titles.",
        "episode": "Represent this FILE by time periods, places, events, stages, and chronology.",
        "relation": "Represent this FILE by relations, roles, causes, comparisons, quantities, and distinguishing attributes.",
    }[view]
    rendered = (
        f"{instruction}\nView: {view}\nFile handle: D{document_id}\n"
        f"Filename/title: {title}\nSource domain: {domain}\nGrounded source excerpts:\n"
    )
    for excerpt in excerpts:
        block = f"- {excerpt}\n"
        if len(rendered) + len(block) > maximum_characters:
            remaining = maximum_characters - len(rendered) - 3
            if remaining >= 80:
                rendered += "- " + excerpt[:remaining]
            break
        rendered += block
    return rendered.strip()[:maximum_characters]


def document_signatures(
    document_id: str,
    text: str,
    url: str = "",
    topic_maximum_characters: int = 6144,
    facet_maximum_characters: int = 4096,
) -> DocumentSignatures:
    if topic_maximum_characters < 1024 or facet_maximum_characters < 1024:
        raise ValueError("signature limits must be at least 1024 characters")
    body = clean_text(text)
    title = extract_title(body, document_id)
    domain = domain_from_url(url)
    sentences = _sentences(body, title)
    ranked: dict[str, list[tuple[float, int, str]]] = {view: [] for view in VIEWS}
    for position, sentence in enumerate(sentences):
        sentence_terms = tokens(sentence)
        entities = len(ENTITY_RE.findall(sentence))
        years = len(YEAR_RE.findall(sentence))
        numbers = len(NUMBER_RE.findall(sentence))
        relations = len(sentence_terms & RELATION_TERMS)
        times = len(sentence_terms & TIME_TERMS)
        events = len(sentence_terms & PLACE_EVENT_TERMS)
        identities = len(sentence_terms & IDENTITY_TERMS)
        information = min(len(sentence_terms), 45)
        ranked["topic"].append((0.25 * information + entities + events, position, sentence))
        ranked["identity"].append((4.0 * entities + 1.5 * identities + 0.1 * information, position, sentence))
        ranked["episode"].append((4.0 * years + 2.0 * times + 1.5 * events + 0.35 * numbers, position, sentence))
        ranked["relation"].append((3.0 * relations + 1.25 * entities + 0.75 * numbers, position, sentence))
    excerpts = {
        "topic": _diverse(ranked["topic"], 8) + _uniform(sentences, 8),
        "identity": _diverse(ranked["identity"], 12) + _uniform(sentences, 3),
        "episode": _diverse(ranked["episode"], 12) + _uniform(sentences, 3),
        "relation": _diverse(ranked["relation"], 12) + _uniform(sentences, 3),
    }
    signatures = {
        view: _render(
            view,
            document_id,
            title,
            domain,
            excerpts[view],
            topic_maximum_characters if view == "topic" else facet_maximum_characters,
        )
        for view in VIEWS
    }
    return DocumentSignatures(document_id, title, domain, signatures)


def query_signatures(query: str) -> dict[str, str]:
    query = clean_field(query, 8000)
    if not query:
        raise ValueError("query is empty")
    instructions = {
        "topic": "Retrieve files by the question's overall subject and content areas.",
        "identity": "Retrieve files by named entities, aliases, works, organizations, and requested identity.",
        "episode": "Retrieve files by time period, place, event, stage, and chronology.",
        "relation": "Retrieve files by relations, roles, causes, comparisons, quantities, and exact distinguishing attributes.",
    }
    return {view: f"Instruct: {instructions[view]}\nQuery: {query}" for view in VIEWS}

