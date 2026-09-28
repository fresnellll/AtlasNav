"""Deterministic text normalization shared by construction and evaluation."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse


WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
TITLE_RE = re.compile(r"(?m)^title:\s*(.+?)\s*$", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)(?:s)?\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?\b", re.IGNORECASE)
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][\w'’.-]*)(?:\s+(?:[A-Z][\w'’.-]*|of|the|and|de|van|von)){1,5}\b"
)
TIME_TERMS = {
    "after", "before", "born", "century", "date", "died", "during", "early",
    "era", "founded", "late", "later", "month", "retired", "retirement", "since",
    "until", "when", "while", "year", "years", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october", "november",
    "december",
}
RELATION_TERMS = {
    "actor", "adapted", "alias", "appointed", "author", "award", "based", "became",
    "child", "created", "daughter", "directed", "director", "employed", "father",
    "founded", "founder", "husband", "joined", "known", "married", "member", "mother",
    "named", "partner", "played", "president", "produced", "producer", "related", "role",
    "son", "spouse", "starred", "studied", "succeeded", "teacher", "wife", "winner",
    "worked", "wrote", "writer",
}


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in value.splitlines()).strip()


def clean_field(value: object, limit: int = 400) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())[:limit]


def extract_title(text: str, document_id: str) -> str:
    match = TITLE_RE.search(text[:4096])
    if match and clean_field(match.group(1), 512):
        return clean_field(match.group(1), 512)
    for line in text.splitlines()[:20]:
        candidate = clean_field(line.lstrip("#- "), 512)
        if candidate and not candidate.startswith("---"):
            return candidate
    return f"Document {document_id}"


def domain_from_url(url: str) -> str:
    return (urlparse(url).hostname or "unknown-domain").lower()


def tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(value)}

