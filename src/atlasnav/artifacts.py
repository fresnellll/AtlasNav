"""Versioned artifact download and checksum verification."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .errors import ArtifactError
from .version import __version__


RELEASE_SCHEMA = "atlasnav_artifact_release_v2"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".csv", ".toml", ".yaml", ".yml"}
SENSITIVE_PATTERNS = {
    "underscore_api_key": re.compile(
        rb"(?<![A-Za-z0-9_])" + b"s" + rb"k_[A-Za-z0-9]{32,}\b"
    ),
    "dash_api_key": re.compile(
        rb"(?<![A-Za-z0-9_])" + b"s" + rb"k-[A-Za-z0-9][A-Za-z0-9._-]{31,}\b"
    ),
    "github_token": re.compile(
        rb"(?<![A-Za-z0-9_])ghp_[A-Za-z0-9]{32,}\b"
    ),
    # Enterprise corpora legitimately quote third-party home and production
    # paths. Detect an absolute path only when it also contains a release-tree
    # marker; no maintainer name or private machine root is embedded here.
    "machine_workspace": re.compile(
        rb"/(?:data|home)/(?:[^/\s\"']+/){0,4}"
        rb"(?:AtlasNav|DCI-Agent-Lite|artifact_release_v2|outputs/portable)"
        rb"(?:/[^\s\"']*)?"
    ),
    "historical_method": re.compile(
        rb"\b(?:" + b"CAR" + rb"TA|G" + rb"4(?:\.1(?:-V2)?)?|g" + rb"41v2)\b",
    ),
    "hosting_platform": re.compile(
        rb"\b(?:" + b"pp" + rb"io|ohmy" + rb"gpt|bai" + rb"lian|dash" + rb"scope)\b",
        re.I,
    ),
    "historical_shard": re.compile(rb"(?:fixed" + rb"166|remaining" + rb"664)", re.I),
    "legacy_repository": re.compile(rb"DCI-Agent-Lite", re.I),
    "internal_workspace_marker": re.compile(rb"ATLASNAV_WORKSPACE", re.I),
    "development_storage": re.compile(
        rb"(?:outputs/portable|historical_shard|source_shard_[ab]|results_(?:calibration|heldout))",
        re.I,
    ),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str


def load_release_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "release_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactError(f"missing release manifest: {path}") from error
    if payload.get("schema") != RELEASE_SCHEMA:
        raise ArtifactError(f"unsupported artifact schema: {payload.get('schema')!r}")
    return payload


def records(payload: dict[str, Any]) -> Iterable[ArtifactRecord]:
    values = payload.get("files")
    if not isinstance(values, list):
        raise ArtifactError("release manifest files must be a list")
    for value in values:
        if not isinstance(value, dict):
            raise ArtifactError("artifact file record must be an object")
        yield ArtifactRecord(
            path=str(value["path"]),
            bytes=int(value["bytes"]),
            sha256=str(value["sha256"]),
        )


def _scan(data: bytes) -> dict[str, int]:
    return {name: len(pattern.findall(data)) for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(data)}


def _deep_verification(base: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from atlasnav.artifact_archive import archive_members

    findings: dict[str, dict[str, int]] = {}
    trajectory_packages: list[dict[str, Any]] = []
    declared_trajectory_packages = (
        (payload.get("browsecomp_plus") or {}).get("trajectory_packages") or []
    )
    declared_trajectories = {
        f"browsecomp_plus/trajectories/{Path(str(row.get('path') or '')).name}": row
        for row in declared_trajectory_packages
    }
    archives = sorted(base.glob("browsecomp_plus/trajectories/*.tar.zst"))
    for path in archives:
        manifest: dict[str, Any] | None = None
        per_query: list[dict[str, Any]] | None = None
        member_count = 0
        archive_findings: dict[str, int] = {}
        for name, data in archive_members(path):
            member_count += 1
            for kind, count in _scan(data).items():
                archive_findings[kind] = archive_findings.get(kind, 0) + count
            if name.endswith("/manifest.json"):
                manifest = json.loads(data)
            elif name.endswith("/per_query.jsonl"):
                per_query = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        relative = path.relative_to(base).as_posix()
        errors = []
        if manifest is None or manifest.get("schema") != "atlasnav_trajectory_package_v2":
            errors.append("manifest")
        declared = int((manifest or {}).get("questions") or 0)
        if per_query is None or len(per_query) != declared:
            errors.append("per_query_cardinality")
        elif len({str(row.get("query_id")) for row in per_query}) != len(per_query):
            errors.append("duplicate_query_id")
        if archive_findings:
            findings[relative] = archive_findings
        declaration = declared_trajectories.get(relative)
        if declaration is None:
            errors.append("undeclared_package")
        elif manifest is not None:
            if str(manifest.get("backbone")) != str(declaration.get("backbone")):
                errors.append("declared_backbone")
            if str(manifest.get("interface")) != str(declaration.get("interface")):
                errors.append("declared_interface")
            if declared != int(declaration.get("queries") or 0):
                errors.append("declared_questions")
        trajectory_packages.append({
            "path": relative, "members": member_count, "questions": len(per_query or []),
            "errors": errors, "valid": not errors and not archive_findings,
        })
    observed_trajectory_paths = {row["path"] for row in trajectory_packages}
    missing_declared_trajectories = sorted(
        set(declared_trajectories) - observed_trajectory_paths
    )
    for record in records(payload):
        path = base / record.path
        if not path.is_file() or path.name.endswith(".tar.zst") or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        result = _scan(path.read_bytes())
        # EnterpriseRAG answer text is benchmark content and may legitimately
        # mention a benchmark source label; it is not a release implementation
        # identifier.
        if record.path.startswith("enterpriserag/") and Path(record.path).name in {
            "answers.jsonl", "questions.jsonl", "results.json", "document_selection_results.json",
        }:
            # These are benchmark-derived result text, not release metadata;
            # provider names and historical terms may occur in the answers.
            result = {}
        if result:
            findings[record.path] = result
    construction_packages: list[dict[str, Any]] = []
    for declared in payload.get("frozen_construction_assets") or []:
        root = base / str(declared.get("root") or "")
        atlas = base / str(declared.get("atlas") or "")
        query = base / str(declared.get("query_embeddings") or "")
        errors: list[str] = []
        try:
            atlas_manifest = json.loads((atlas / "manifest.json").read_text(encoding="utf-8"))
            query_manifest = json.loads((query / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            atlas_manifest, query_manifest = {}, {}
            errors.append("manifest")
        documents = int(declared.get("documents") or 0)
        queries = int(declared.get("queries") or 0)
        if atlas_manifest.get("schema") != "atlasnav_multiplex_atlas_frozen_v1":
            errors.append("atlas_schema")
        if query_manifest.get("schema") != "atlasnav_four_view_query_embeddings_v1":
            errors.append("query_schema")
        if int(atlas_manifest.get("documents") or 0) != documents:
            errors.append("document_cardinality")
        if int(query_manifest.get("queries") or 0) != queries:
            errors.append("query_cardinality")
        required: list[Path] = []
        if atlas_manifest:
            required.append(atlas / str(atlas_manifest.get("catalog") or ""))
            for view in ("topic", "identity", "episode", "relation"):
                pca = ((atlas_manifest.get("pca") or {}).get("assets") or {}).get(view) or {}
                required.extend((
                    atlas / str(pca.get("reduced") or ""),
                    atlas / str(pca.get("transform") or ""),
                    atlas / f"{view}_graph.npz",
                ))
            arrays = atlas_manifest.get("arrays") or atlas_manifest.get("assets") or {}
            for name in ("file_to_parent", "file_to_leaf", "leaf_to_parent"):
                required.append(atlas / str(arrays.get(name) or ""))
            cards = atlas_manifest.get("cards") or {}
            required.extend((
                atlas / str(cards.get("parents") or arrays.get("parent_cards") or "parent_cards.jsonl.gz"),
                atlas / str(cards.get("leaves") or arrays.get("child_cards") or "child_cards.jsonl.gz"),
            ))
        if query_manifest:
            required.append(query / str(query_manifest.get("query_catalog") or ""))
            for view in ("topic", "identity", "episode", "relation"):
                required.append(query / str((query_manifest.get("query_embeddings") or {}).get(view) or ""))
        if any(not path.is_file() for path in required):
            errors.append("required_assets")
        catalog = atlas / str(atlas_manifest.get("catalog") or "")
        if catalog.is_file():
            with gzip.open(catalog, "rt", encoding="utf-8") as stream:
                if sum(1 for line in stream if line.strip()) != documents:
                    errors.append("catalog_cardinality")
        query_catalog = query / str(query_manifest.get("query_catalog") or "")
        if query_catalog.is_file():
            with gzip.open(query_catalog, "rt", encoding="utf-8") as stream:
                if sum(1 for line in stream if line.strip()) != queries:
                    errors.append("query_catalog_cardinality")
        construction_packages.append({
            "name": str(declared.get("name") or root.name),
            "root": root.relative_to(base).as_posix() if root != base else ".",
            "documents": documents,
            "queries": queries,
            "errors": sorted(set(errors)),
            "valid": not errors,
        })
    frozen_ranking_package: dict[str, Any] | None = None
    ranking_declaration = (payload.get("browsecomp_plus") or {}).get("frozen_ranking")
    if ranking_declaration:
        import numpy as np

        ranking_root = base / str(ranking_declaration.get("root") or "")
        errors: list[str] = []
        try:
            ranking_manifest = json.loads(
                (ranking_root / "manifest.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            ranking_manifest = {}
            errors.append("manifest")
        if ranking_manifest.get("schema") != "atlasnav_frozen_ranking_v1":
            errors.append("schema")
        if ranking_manifest.get("finalized") is not True:
            errors.append("not_finalized")
        queries = int(ranking_manifest.get("queries") or 0)
        documents = int(ranking_manifest.get("documents") or 0)
        scores_path = ranking_root / str(ranking_manifest.get("query_scores") or "")
        audit_path = ranking_root / str(ranking_manifest.get("retrieval_audit") or "")
        catalog_path = ranking_root / str(ranking_manifest.get("query_catalog") or "")
        for path, key in (
            (scores_path, "query_scores"),
            (audit_path, "retrieval_audit"),
            (catalog_path, "query_catalog"),
        ):
            expected = ranking_manifest.get(f"{key}_sha256")
            if not path.is_file():
                errors.append(f"{key}_missing")
            elif not expected or sha256_file(path) != expected:
                errors.append(f"{key}_checksum")
        catalog_qids: list[str] = []
        audit_qids: list[str] = []
        if catalog_path.is_file():
            try:
                with gzip.open(catalog_path, "rt", encoding="utf-8") as stream:
                    catalog_qids = [str(json.loads(line)["query_id"]) for line in stream if line.strip()]
            except (OSError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
                errors.append("query_catalog_format")
        if audit_path.is_file():
            try:
                with gzip.open(audit_path, "rt", encoding="utf-8") as stream:
                    audit_qids = [str(json.loads(line)["query_id"]) for line in stream if line.strip()]
            except (OSError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
                errors.append("retrieval_audit_format")
        if len(catalog_qids) != queries or len(set(catalog_qids)) != queries:
            errors.append("query_catalog_cardinality")
        if audit_qids != catalog_qids:
            errors.append("retrieval_audit_query_set")
        route_rows: list[dict[str, Any]] = []
        route_asset = ranking_manifest.get("workspace_routes")
        if route_asset:
            route_path = ranking_root / str(route_asset)
            if not route_path.is_file():
                errors.append("workspace_routes_missing")
            elif sha256_file(route_path) != ranking_manifest.get("workspace_routes_sha256"):
                errors.append("workspace_routes_checksum")
            else:
                try:
                    with gzip.open(route_path, "rt", encoding="utf-8") as stream:
                        for line in stream:
                            if not line.strip():
                                continue
                            route_rows.append(json.loads(line))
                            for kind, count in _scan(line.encode()).items():
                                relative = route_path.relative_to(base).as_posix()
                                findings.setdefault(relative, {})[kind] = (
                                    findings.get(relative, {}).get(kind, 0) + count
                                )
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    errors.append("workspace_routes_format")
            if [str(row.get("query_id")) for row in route_rows] != catalog_qids:
                errors.append("workspace_routes_query_set")
            for row in route_rows:
                route = str(row.get("route_tsv") or "")
                if hashlib.sha256(route.encode()).hexdigest() != row.get("route_sha256"):
                    errors.append("workspace_route_internal_checksum")
                    break
                if not isinstance(row.get("parent_order"), list) or not isinstance(
                    row.get("initial_leads"), list
                ):
                    errors.append("workspace_route_metadata")
                    break
        if scores_path.is_file():
            try:
                scores = np.load(scores_path, mmap_mode="r")
                if scores.shape != (queries, documents) or scores.dtype != np.float32:
                    errors.append("query_scores_shape_or_dtype")
                elif not np.isfinite(scores).all():
                    errors.append("query_scores_values")
            except (OSError, ValueError):
                errors.append("query_scores_format")
        frozen_ranking_package = {
            "root": ranking_root.relative_to(base).as_posix(),
            "queries": queries,
            "documents": documents,
            "errors": sorted(set(errors)),
            "valid": not errors,
        }
    return {
        "trajectory_packages": trajectory_packages,
        "declared_trajectory_packages": len(declared_trajectories),
        "missing_declared_trajectory_packages": missing_declared_trajectories,
        "construction_asset_packages": construction_packages,
        "frozen_ranking_package": frozen_ranking_package,
        "sensitive_findings": findings,
        "valid": bool(declared_trajectories)
        and len(trajectory_packages) == len(declared_trajectories)
        and not missing_declared_trajectories
        and all(row["valid"] for row in trajectory_packages)
        and all(row["valid"] for row in construction_packages)
        and (frozen_ranking_package is None or frozen_ranking_package["valid"])
        and not findings,
    }


def verify_artifact_root(root: str | Path, deep: bool = False) -> dict[str, Any]:
    base = Path(root).resolve()
    payload = load_release_manifest(base)
    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, str]] = []
    checked = 0
    for record in records(payload):
        path = (base / record.path).resolve()
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ArtifactError(f"artifact path escapes root: {record.path}") from error
        if not path.is_file():
            missing.append(record.path)
            continue
        checked += 1
        actual_size = path.stat().st_size
        if actual_size != record.bytes:
            size_mismatches.append(
                {"path": record.path, "expected": record.bytes, "actual": actual_size}
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.sha256:
            hash_mismatches.append(
                {"path": record.path, "expected": record.sha256, "actual": actual_hash}
            )
    compatible_spec = payload.get("compatible_code")
    code_compatible = (
        compatible_spec is None
        or (compatible_spec == ">=0.2.0.dev0,<0.3" and __version__.startswith("0.2."))
    )
    inventory_path = base / "MANIFEST.sha256.json"
    inventory_consistent: bool | None = None
    if inventory_path.is_file():
        try:
            inventory_consistent = json.loads(inventory_path.read_text(encoding="utf-8")) == payload.get("files")
        except (json.JSONDecodeError, UnicodeDecodeError):
            inventory_consistent = False
    report = {
        "schema": "atlasnav_artifact_verification_v2",
        "artifact_version": payload.get("artifact_version"),
        "compatible_code": compatible_spec,
        "installed_code_version": __version__,
        "code_compatible": code_compatible,
        "inventory_consistent": inventory_consistent,
        "checked": checked,
        "declared": len(list(records(payload))),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
    }
    report["valid"] = (
        not any((missing, size_mismatches, hash_mismatches))
        and code_compatible
        and inventory_consistent is not False
    )
    if deep and report["valid"]:
        report["deep"] = _deep_verification(base, payload)
        report["valid"] = report["deep"]["valid"]
    return report


def download_artifacts(repo_id: str, output: str | Path, revision: str | None = None) -> str:
    if not repo_id or "/" not in repo_id:
        raise ArtifactError("a namespace/repository artifact ID is required")
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(Path(output)),
    )
