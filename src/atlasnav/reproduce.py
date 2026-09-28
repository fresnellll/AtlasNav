"""Offline reproduction of released tables without model calls."""

from __future__ import annotations

import json
from statistics import mean
from pathlib import Path
from typing import Any

from atlasnav.artifact_archive import read_package_metadata
from atlasnav.evaluation.checkpoints import passive_checkpoints
from atlasnav.evaluation.metrics import endpoint_summary, evidence_blindness_summary
from atlasnav.evaluation.metrics import paired_accuracy
from atlasnav.io import atomic_json


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _checkpoint_reproduction(
    analysis: Path, endpoint_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Validate the frozen checkpoint protocol and replay passive endpoints.

    Frozen branch manifests record which queries had naturally terminated and
    which required a counterfactual finalization branch.  The artifact replay
    validates that every declared query occurs exactly once at every boundary.
    Passive checkpoint accuracy is then recomputed from the reported endpoint
    rows without making a model call.  These two objects are deliberately kept
    distinct: a passive replay must never be mislabeled as active Safe Release.
    """
    path = analysis / "checkpoints/checkpoint_manifest.jsonl"
    if not path.is_file():
        return None
    manifest_rows = _jsonl(path)
    endpoint_qids = {str(row["query_id"]) for row in endpoint_rows}
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    selection_is_outcome_blind = True
    for row in manifest_rows:
        kind = str(row.get("kind") or "")
        value = float(row.get("value"))
        grouped.setdefault((kind, value), []).append(row)
        selection_is_outcome_blind = selection_is_outcome_blind and not bool(
            row.get("selection_reads_gold_qrel_judge_correctness")
            or row.get("reads_gold_qrel_judge_correctness")
        )
    boundary_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for (kind, value), rows in sorted(grouped.items()):
        qids = [str(row.get("query_id")) for row in rows]
        dispositions: dict[str, int] = {}
        for row in rows:
            disposition = str(row.get("disposition") or "unknown")
            dispositions[disposition] = dispositions.get(disposition, 0) + 1
        if len(qids) != len(endpoint_qids):
            errors.append(f"{kind}:{value}:cardinality={len(qids)}")
        if len(set(qids)) != len(qids):
            errors.append(f"{kind}:{value}:duplicate_query_id")
        if set(qids) != endpoint_qids:
            errors.append(f"{kind}:{value}:query_set")
        boundary_rows.append({
            "kind": kind,
            "value": value,
            "queries": len(qids),
            "dispositions": dispositions,
        })
    turn_points = sorted({int(value) for kind, value in grouped if kind == "turn"})
    cost_points = sorted({float(value) for kind, value in grouped if kind == "cost"})
    passive = passive_checkpoints(
        endpoint_rows,
        turn_checkpoints=turn_points,
        cost_checkpoints=cost_points,
    )
    prefix_path = analysis / "checkpoints/prefix_funnel_summary.json"
    return {
        "schema": "atlasnav_frozen_checkpoint_reproduction_v1",
        "manifest_rows": len(manifest_rows),
        "boundaries": boundary_rows,
        "selection_is_outcome_blind": selection_is_outcome_blind,
        "manifest_valid": not errors and selection_is_outcome_blind,
        "manifest_errors": errors,
        "passive_endpoint_replay": passive,
        "active_checkpoint_accuracy_recomputed": False,
        "active_checkpoint_note": (
            "Counterfactual Safe Release requires recorded branch answers and judgments; "
            "the branch manifest is validated here but passive replay is not relabeled as active accuracy."
        ),
        "frozen_prefix_funnel": (
            json.loads(prefix_path.read_text(encoding="utf-8")) if prefix_path.is_file() else None
        ),
    }


def reproduce_browsecomp_plus(artifact_root: Path, output_directory: Path) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    root = artifact_root / "browsecomp_plus"
    packages = sorted((root / "trajectories").glob("*.tar.zst"))
    if not packages:
        raise FileNotFoundError(f"no trajectory packages under {root / 'trajectories'}")
    release_manifest_path = artifact_root / "release_manifest.json"
    if not release_manifest_path.is_file():
        raise FileNotFoundError(f"missing release manifest: {release_manifest_path}")
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    declared_packages = (release_manifest.get("browsecomp_plus") or {}).get(
        "trajectory_packages"
    ) or []
    declared_paths = {
        (root / "trajectories" / Path(str(row["path"])).name).resolve()
        for row in declared_packages
    }
    actual_paths = {path.resolve() for path in packages}
    if not declared_paths or actual_paths != declared_paths:
        missing = sorted(path.name for path in declared_paths - actual_paths)
        unexpected = sorted(path.name for path in actual_paths - declared_paths)
        raise ValueError(
            "BrowseComp-Plus trajectory package set disagrees with the release "
            f"manifest (missing={missing}, unexpected={unexpected})"
        )
    results: dict[str, Any] = {}
    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        manifest, rows = read_package_metadata(package)
        model = str(manifest["backbone"])
        interface = str(manifest["interface"])
        key = f"{model}/{interface}"
        if key in rows_by_key:
            raise ValueError(f"duplicate BrowseComp-Plus trajectory identity: {key}")
        rows_by_key[key] = rows
        # Some early archive manifests encoded a historical agent+judge total.
        # The current paper contract is uniform: query-time agent inference only.
        endpoint = endpoint_summary(
            rows,
            agent_currency=str(manifest.get("recorded_agent_cost_currency") or "") or None,
            judge_currency=str(manifest.get("recorded_judge_cost_currency") or "") or None,
            comparable_components=("agent",),
        )
        analysis = root / "analysis" / model / interface
        evidence_path = analysis / "evidence_blindness/final/per_query.jsonl"
        evidence = evidence_blindness_summary(_jsonl(evidence_path)) if evidence_path.is_file() else None
        checkpoints = _checkpoint_reproduction(analysis, rows)
        results[key] = {
            "package": package.relative_to(artifact_root).as_posix(),
            "trajectory_recomputable": True,
            "endpoint": endpoint,
            "evidence_blindness": evidence,
            "checkpoints": checkpoints,
        }
    comparisons: dict[str, Any] = {}
    for model in sorted({key.split("/", 1)[0] for key in rows_by_key}):
        atlas_key = f"{model}/atlasnav"
        if atlas_key not in rows_by_key:
            continue
        comparisons[model] = {}
        for interface in ("raw-dci", "dr-dci"):
            baseline_key = f"{model}/{interface}"
            if baseline_key in rows_by_key:
                comparisons[model][f"atlasnav_vs_{interface}"] = paired_accuracy(
                    rows_by_key[atlas_key], rows_by_key[baseline_key]
                )
    # The declared paper table is part of the checksummed artifact contract so
    # wheel installs never depend on a source-checkout-relative config path.
    expected_path = root / "paper_results.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    consistency: dict[str, Any] = {}
    for model, interfaces in expected["results"].items():
        atlas_key = f"{model}/atlasnav"
        atlas_cost = (
            float(results[atlas_key]["endpoint"]["paper_comparable_online_cost"])
            if atlas_key in results else None
        )
        for interface, target in interfaces.items():
            key = f"{model}/{interface}"
            if key not in results:
                continue
            observed = results[key]["endpoint"]
            expected_accuracy = float(target["accuracy_percent"])
            observed_accuracy = float(observed["accuracy_percent"])
            expected_normalized_cost = target.get("normalized_cost")
            observed_normalized_cost = (
                float(observed["paper_comparable_online_cost"]) / atlas_cost
                if expected_normalized_cost is not None and atlas_cost else None
            )
            accuracy_matches = round(observed_accuracy, 2) == expected_accuracy
            cost_matches = (
                True if expected_normalized_cost is None else
                round(float(observed_normalized_cost), 3) == float(expected_normalized_cost)
            )
            consistency[key] = {
                "expected_correct": int(target["correct"]),
                "observed_correct": int(observed["correct"]),
                "expected_accuracy_percent": expected_accuracy,
                "observed_accuracy_percent": observed_accuracy,
                "accuracy_matches": accuracy_matches,
                "expected_normalized_cost": expected_normalized_cost,
                "observed_normalized_cost": observed_normalized_cost,
                "cost_matches": cost_matches,
                "matches": (
                    int(target["correct"]) == int(observed["correct"])
                    and accuracy_matches and cost_matches
                ),
            }
    diagnostics_path = root / "paper_diagnostics.json"
    diagnostic_consistency: dict[str, Any] | None = None
    if diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        endpoint_cells: list[dict[str, Any]] = []
        for model, interfaces in diagnostics["endpoint_evidence_blindness_percent"].items():
            for interface, stages in interfaces.items():
                evidence = (results.get(f"{model}/{interface}") or {}).get("evidence_blindness")
                if evidence is None:
                    continue
                for stage, expected_values in stages.items():
                    for label, expected_value in zip(("any", "mean", "all"), expected_values):
                        observed_value = 100.0 * float(evidence[stage][label])
                        endpoint_cells.append({
                            "backbone": model, "interface": interface, "stage": stage,
                            "statistic": label, "expected_percent": float(expected_value),
                            "observed_percent": observed_value,
                            "matches": round(observed_value, 2) == float(expected_value),
                        })
        checkpoint_cells: list[dict[str, Any]] = []
        for kind, models in diagnostics["locate_all_blindness_percent"].items():
            for model, interfaces in models.items():
                for interface, points in interfaces.items():
                    analysis = root / "analysis" / model / interface
                    for value, expected_value in points.items():
                        suffix = str(int(float(value))) if kind == "turn" else str(value)
                        summary_path = (
                            analysis / "evidence_blindness/checkpoints"
                            / f"{kind}_{suffix}/summary.json"
                        )
                        if not summary_path.is_file():
                            checkpoint_cells.append({
                                "backbone": model, "interface": interface, "kind": kind,
                                "value": value, "expected_percent": float(expected_value),
                                "observed_percent": None, "matches": False,
                                "error": "missing frozen checkpoint summary",
                            })
                            continue
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        observed_value = 100.0 * (1.0 - float(summary["answer_evidence_all_rate"]))
                        checkpoint_cells.append({
                            "backbone": model, "interface": interface, "kind": kind,
                            "value": value, "expected_percent": float(expected_value),
                            "observed_percent": observed_value,
                            "matches": round(observed_value, 2) == float(expected_value),
                        })
        diagnostic_consistency = {
            "endpoint_cells": endpoint_cells,
            "checkpoint_cells": checkpoint_cells,
            "endpoint_cells_checked": len(endpoint_cells),
            "checkpoint_cells_checked": len(checkpoint_cells),
            "all_match": bool(endpoint_cells) and bool(checkpoint_cells)
            and all(row["matches"] for row in endpoint_cells + checkpoint_cells),
        }
    report = {
        "schema": "atlasnav_browsecomp_plus_reproduction_v1",
        "benchmark": "BrowseComp-Plus",
        "results": results,
        "paired_comparisons": comparisons,
        "paper_consistency": consistency,
        "paper_consistency_all_match": bool(consistency) and all(
            row["matches"] for row in consistency.values()
        ),
        "paper_diagnostic_consistency": diagnostic_consistency,
        "summary_only_paper_results": {
            model: value for model, value in expected["results"].items()
            if expected["trajectory_availability"].get(model) == "summary-only"
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    atomic_json(output_directory / "browsecomp_plus.json", report)
    return report


def _summarize_binary_rows(rows: list[dict[str, Any]], correct_key: str = "is_correct") -> dict[str, Any]:
    queries = len(rows)
    correct = sum(row.get(correct_key) is True for row in rows)
    return {
        "queries": queries, "correct": correct,
        "accuracy": correct / queries if queries else None,
        "agent_cost_cny": sum(float(row.get("agent_cost_cny") or row.get("agent_cost") or 0.0) for row in rows),
        "judge_cost_cny": sum(float(row.get("judge_cost_cny") or row.get("judge_cost") or 0.0) for row in rows),
        "turns": sum(int(row.get("turn_count") or row.get("turns") or 0) for row in rows),
    }


def reproduce_auxiliary(artifact_root: Path, output_directory: Path) -> dict[str, Any]:
    """Recompute released auxiliary endpoint metrics from per-query rows."""
    result: dict[str, Any] = {}
    wiki = artifact_root / "2wiki_global400/analysis"
    if wiki.is_dir():
        wiki_values = {
            interface: _summarize_binary_rows(_jsonl(wiki / f"{interface}_rows.jsonl"))
            for interface in ("atlasnav", "raw-dci", "dr-dci")
        }
        wiki_summary_path = wiki / "summary.json"
        if wiki_summary_path.is_file():
            summary = json.loads(wiki_summary_path.read_text(encoding="utf-8"))
            for public, internal in {
                "atlasnav": "atlasnav", "raw-dci": "dci", "dr-dci": "drdci",
            }.items():
                wiki_values[public]["judge_cost_cny"] = float(
                    summary["judge_cost_cny"][internal]
                )
                wiki_values[public]["online_query_embedding_cost_cny"] = (
                    float(summary["drdci_online_query_embedding"]["cost_cny"])
                    if internal == "drdci" else 0.0
                )
                wiki_values[public]["operating_cost_cny"] = (
                    wiki_values[public]["agent_cost_cny"]
                    + wiki_values[public]["online_query_embedding_cost_cny"]
                )
        result["2wiki_global400"] = wiki_values
    beir = artifact_root / "beir_sample50/analysis"
    if beir.is_dir():
        values: dict[str, Any] = {}
        for dataset in ("beir_scifact", "beir_arguana"):
            values[dataset] = {}
            for interface in ("atlasnav", "raw-dci", "dr-dci"):
                rows = _jsonl(beir / dataset / f"{interface}_rows.jsonl")
                values[dataset][interface] = {
                    "queries": len(rows),
                    "ndcg_at_10": mean(float(row["ndcg_at_10"]) for row in rows),
                    "recall_at_10": mean(float(row["recall_at_10"]) for row in rows),
                    "mrr_at_10": mean(float(row["mrr_at_10"]) for row in rows),
                    "agent_cost_cny": sum(float(row.get("agent_cost_cny") or 0.0) for row in rows),
                }
        result["beir_sample50"] = values
    fanout = artifact_root / "fanoutqa"
    if fanout.is_dir():
        values = {}
        for interface in ("atlasnav", "raw-dci", "dr-dci"):
            rows = _jsonl(fanout / interface / "per_query.jsonl")
            values[interface] = {
                "queries": len(rows),
                "loose_accuracy": mean(float(row["official_loose"]) for row in rows),
                "strict_accuracy": mean(row.get("official_strict") is True for row in rows),
                "agent_cost_cny": sum(float(row.get("agent_cost") or 0.0) for row in rows),
                "turns": sum(int(row.get("turn_count") or 0) for row in rows),
            }
        fanout_summary_path = fanout / "comparison/summary.json"
        if fanout_summary_path.is_file():
            summary = json.loads(fanout_summary_path.read_text(encoding="utf-8"))
            embedding = float(summary["dr_dci_online_query_embedding"]["cost_cny"])
            values["atlasnav"]["operating_cost_cny"] = values["atlasnav"]["agent_cost_cny"]
            values["raw-dci"]["operating_cost_cny"] = values["raw-dci"]["agent_cost_cny"]
            values["dr-dci"]["online_query_embedding_cost_cny"] = embedding
            values["dr-dci"]["operating_cost_cny"] = (
                values["dr-dci"]["agent_cost_cny"] + embedding
            )
        result["fanoutqa"] = values
    phantom = artifact_root / "phantomwiki/analysis/per_query.jsonl"
    if phantom.is_file():
        rows = _jsonl(phantom)
        values = {}
        for scale in sorted({str(row["scale"]) for row in rows}):
            values[scale] = {}
            for interface in ("atlasnav", "dci", "drdci"):
                selected = [row for row in rows if str(row["scale"]) == scale and row["method"] == interface]
                values[scale][interface] = _summarize_binary_rows(selected, "correct")
        result["phantomwiki"] = values
    trec = artifact_root / "trec_covid/analysis/per_query.jsonl"
    if trec.is_file():
        rows = _jsonl(trec)
        values = {}
        for method in sorted({str(row["method"]) for row in rows}):
            selected = [row for row in rows if row["method"] == method]
            values[method] = {
                "queries": len(selected),
                "ndcg_at_10": mean(float(row["ndcg_at_10"]) for row in selected),
                "agent_cost_cny": sum(float(row.get("agent_cost_cny") or 0.0) for row in selected),
                "turns": sum(int(row.get("turns") or 0) for row in selected),
            }
        trec_analysis_path = trec.parent / "analysis.json"
        if trec_analysis_path.is_file():
            endpoint_values = json.loads(
                trec_analysis_path.read_text(encoding="utf-8")
            )["endpoints"]
            for method, value in endpoint_values.items():
                if method in values:
                    values[method]["online_query_embedding_cost_cny"] = float(
                        value.get("online_query_embedding_cost_cny") or 0.0
                    )
                    values[method]["operating_cost_cny"] = float(
                        value["online_total_cost_cny"]
                    )
        result["trec_covid"] = values
    enterprise = artifact_root / "enterpriserag/results.json"
    enterprise_selection = artifact_root / "enterpriserag/document_selection/document_selection_results.json"
    if enterprise.is_file():
        value = json.loads(enterprise.read_text(encoding="utf-8"))
        selection = json.loads(enterprise_selection.read_text(encoding="utf-8")) if enterprise_selection.is_file() else None
        rows = value.get("questions") or []
        aggregate = value.get("aggregate_stats") or {}
        selection_aggregate = (selection or {}).get("aggregate_stats") or {}
        result["enterpriserag"] = {
            "queries": len(rows),
            "correctness": mean(row.get("answer_correct") is True for row in rows),
            "completeness_percent": mean(float(row.get("completeness_pct") or 0.0) for row in rows),
            "document_recall_percent": float(selection_aggregate.get("average_recall_pct") or aggregate.get("average_recall_pct") or 0.0),
            "invalid_extra_documents": float(selection_aggregate.get("average_invalid_extra_docs") or aggregate.get("average_invalid_extra_docs") or 0.0),
            "official_aggregate": aggregate,
            "selection_aggregate": selection_aggregate,
            "selection_protocol": "frozen answers with official document-selection semantics",
        }
    expected_path = artifact_root / "paper_auxiliary_results.json"
    if expected_path.is_file():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        cells: list[dict[str, Any]] = []

        def check(path: str, observed: float | int, target: float | int, digits: int = 2) -> None:
            cells.append({
                "path": path, "observed": observed, "expected": target,
                "matches": round(float(observed), digits) == float(target),
            })

        for scale, methods in expected["phantomwiki_accuracy_percent"].items():
            for method, target in methods.items():
                check(
                    f"phantomwiki/{scale}/{method}/accuracy_percent",
                    100.0 * result["phantomwiki"][scale][method]["accuracy"], target, 1,
                )
        for interface, targets in expected["2wiki_global400"].items():
            row = result["2wiki_global400"][interface]
            check(f"2wiki/{interface}/accuracy_percent", 100.0 * row["accuracy"], targets["accuracy_percent"])
            check(
                f"2wiki/{interface}/query_time_agent_inference_cost_cny",
                row["agent_cost_cny"], targets["query_time_agent_inference_cost_cny"],
            )
            check(
                f"2wiki/{interface}/operating_cost_cny",
                row["operating_cost_cny"], targets["operating_cost_cny"],
            )
            check(f"2wiki/{interface}/turns", row["turns"], targets["turns"], 0)
        for interface, targets in expected["fanoutqa"].items():
            row = result["fanoutqa"][interface]
            check(f"fanoutqa/{interface}/loose_percent", 100.0 * row["loose_accuracy"], targets["loose_percent"])
            check(f"fanoutqa/{interface}/strict_percent", 100.0 * row["strict_accuracy"], targets["strict_percent"])
            check(
                f"fanoutqa/{interface}/query_time_agent_inference_cost_cny",
                row["agent_cost_cny"], targets["query_time_agent_inference_cost_cny"],
            )
            check(
                f"fanoutqa/{interface}/operating_cost_cny",
                row["operating_cost_cny"], targets["operating_cost_cny"],
            )
            check(f"fanoutqa/{interface}/turns", row["turns"], targets["turns"], 0)
        for method, targets in expected["trec_covid"].items():
            row = result["trec_covid"][method]
            check(f"trec_covid/{method}/ndcg_at_10", row["ndcg_at_10"], targets["ndcg_at_10"], 4)
            if "query_time_agent_inference_cost_cny" in targets:
                check(
                    f"trec_covid/{method}/query_time_agent_inference_cost_cny",
                    row["agent_cost_cny"], targets["query_time_agent_inference_cost_cny"],
                )
            if "operating_cost_cny" in targets:
                check(
                    f"trec_covid/{method}/operating_cost_cny",
                    row["operating_cost_cny"], targets["operating_cost_cny"],
                )
        enterprise_expected = expected["enterpriserag"]
        if "enterpriserag" in result:
            enterprise_observed = result["enterpriserag"]["selection_aggregate"]
            for label, source in {
                "overall": "combined_correctness_completeness_score",
                "correctness_percent": "average_correctness_pct",
                "completeness_percent": "average_completeness_pct",
                "document_recall_percent": "average_recall_pct",
                "invalid_extra_documents": "average_invalid_extra_docs",
            }.items():
                check(f"enterpriserag/{label}", enterprise_observed[source], enterprise_expected[label])
        else:
            cells.append({
                "path": "enterpriserag",
                "observed": None,
                "expected": "frozen one-trajectory-per-question package",
                "matches": False,
                "reason": "final EnterpriseRAG package is not present",
            })
        result["paper_consistency"] = {
            "cells": cells, "cells_checked": len(cells),
            "all_match": bool(cells) and all(row["matches"] for row in cells),
        }
    atomic_json(output_directory / "auxiliary.json", result)
    return result


def reproduce(artifact_root: Path, output_directory: Path, suite: str = "paper") -> dict[str, Any]:
    if suite not in {"paper", "browsecomp_plus"}:
        raise ValueError(f"unknown reproduction suite: {suite}")
    report = reproduce_browsecomp_plus(artifact_root, output_directory)
    auxiliary = reproduce_auxiliary(artifact_root.resolve(), output_directory) if suite == "paper" else None
    summary = {
        "schema": "atlasnav_reproduction_v1", "suite": suite,
        "browsecomp_plus": report, "auxiliary": auxiliary,
    }
    atomic_json(output_directory / "reproduction.json", summary)
    return summary
