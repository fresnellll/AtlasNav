"""Environment and release-hygiene checks for a public research package."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

from .config import load_model_profile


FORBIDDEN_PUBLIC_PATTERNS = {
    # Keep the deny-list effective without embedding retired public-facing names
    # verbatim in the public source tree itself.
    "historical_method": re.compile(
        r"\b(?:" + "CAR" + r"TA|G" + r"4(?:\.1-V2)?|g" + r"41v2)\b", re.I
    ),
    "hosting_platform": re.compile(
        r"\b(?:" + "pp" + r"io|ohmy" + r"gpt)\b", re.I
    ),
    "local_workspace": re.compile(
        "/" + r"(?:da" + "ta|ho" + r"me)/[^\s\"']+"
    ),
    "credential": re.compile(
        r"(?<![A-Za-z0-9_])(?:ghp_|github_pat_|sk[-_])[A-Za-z0-9_.-]{32,}\b"
    ),
}


def public_tree_findings(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() in {
            ".npy", ".npz", ".parquet", ".gz", ".zst", ".whl", ".zip", ".tar",
        }:
            continue
        relative = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="replace")
        for kind, pattern in FORBIDDEN_PUBLIC_PATTERNS.items():
            count = len(pattern.findall(text)) + len(pattern.findall(relative.as_posix()))
            if count:
                findings.append(
                    {"path": relative.as_posix(), "kind": kind, "count": count}
                )
    return findings


def doctor(root: Path, profiles: list[Path] | None = None) -> dict[str, Any]:
    optional = {name: importlib.util.find_spec(name) is not None for name in (
        "faiss", "igraph", "leidenalg", "sklearn", "pyarrow", "zstandard"
    )}
    parsed_profiles = []
    for profile in profiles or []:
        value = load_model_profile(profile)
        parsed_profiles.append(
            {
                "id": value.profile_id,
                "model_id": value.model_id,
                "transport": value.transport,
                "currency": value.pricing.currency,
                "safe_release": value.safe_release.cost_threshold,
                "base_url_configured": bool(value.base_url),
                "api_key_configured": bool(value.api_key),
            }
        )
    findings = public_tree_findings(root)
    return {
        "schema": "atlasnav_doctor_v1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "root": "${ATLASNAV_REPOSITORY}",
        "optional_dependencies": optional,
        "profiles": parsed_profiles,
        "public_tree_findings": findings,
        "valid_public_tree": not findings,
    }
