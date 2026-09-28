from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import zstandard as zstd

from atlasnav.reproduce import _checkpoint_reproduction, reproduce_browsecomp_plus


def _write_trajectory_package(
    path: Path, *, backbone: str, interface: str, rows: list[dict]
) -> None:
    root = "atlasnav_trajectory_package_v2"
    manifest = {
        "schema": "atlasnav_trajectory_package_v2",
        "benchmark": "BrowseComp-Plus",
        "questions": len(rows),
        "backbone": backbone,
        "interface": interface,
        "recorded_agent_cost_currency": "USD",
        "recorded_judge_cost_currency": "CNY",
        "paper_comparable_online_cost": {"components": ["agent"]},
    }
    payloads = {
        f"{root}/manifest.json": (json.dumps(manifest) + "\n").encode(),
        f"{root}/per_query.jsonl": b"".join(
            (json.dumps(row) + "\n").encode() for row in rows
        ),
    }
    with path.open("wb") as raw:
        with zstd.ZstdCompressor().stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for name, data in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))


def test_checkpoint_reproduction_validates_boundaries_and_keeps_passive_distinct(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    rows = []
    for query_id in ("q1", "q2"):
        rows.extend([
            {
                "query_id": query_id,
                "kind": "turn",
                "value": 15,
                "disposition": "reuse_natural_final",
                "selection_reads_gold_qrel_judge_correctness": False,
            },
            {
                "query_id": query_id,
                "kind": "cost",
                "value": 2.75,
                "disposition": "branch_safe_release",
                "selection_reads_gold_qrel_judge_correctness": False,
            },
        ])
    with (checkpoint / "checkpoint_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    endpoint = [
        {"query_id": "q1", "is_correct": True, "turn_count": 7, "recorded_agent_cost": 0.5},
        {"query_id": "q2", "is_correct": False, "turn_count": 20, "recorded_agent_cost": 3.0},
    ]
    report = _checkpoint_reproduction(tmp_path, endpoint)
    assert report is not None
    assert report["manifest_valid"] is True
    assert report["manifest_rows"] == 4
    assert report["active_checkpoint_accuracy_recomputed"] is False
    passive = report["passive_endpoint_replay"]
    assert passive["turn"][0]["correct_by_checkpoint"] == 1
    assert passive["cost"][0]["unfinished_counted_wrong"] == 1


def test_gold_reference_is_recomputed_from_declared_trajectory(tmp_path: Path) -> None:
    trajectories = tmp_path / "browsecomp_plus/trajectories"
    trajectories.mkdir(parents=True)
    package = trajectories / "gpt-5.6-luna_gold-document-reference_full830.tar.zst"
    rows = [
        {
            "query_id": "1", "is_correct": True, "terminal_valid": True,
            "turn_count": 2, "recorded_agent_cost": 0.01,
            "recorded_judge_cost": 0.001,
        },
        {
            "query_id": "2", "is_correct": False, "terminal_valid": True,
            "turn_count": 3, "recorded_agent_cost": 0.02,
            "recorded_judge_cost": 0.001,
        },
    ]
    _write_trajectory_package(
        package, backbone="gpt-5.6-luna",
        interface="gold-document-reference", rows=rows,
    )
    (tmp_path / "release_manifest.json").write_text(
        json.dumps({
            "browsecomp_plus": {"trajectory_packages": [{
                "path": package.name, "backbone": "gpt-5.6-luna",
                "interface": "gold-document-reference", "queries": 2,
            }]}
        }),
        encoding="utf-8",
    )
    (tmp_path / "browsecomp_plus/paper_results.json").write_text(
        json.dumps({
            "results": {"gpt-5.6-luna": {
                "gold-document-reference": {"correct": 1, "accuracy_percent": 50.0}
            }},
            "trajectory_availability": {"gpt-5.6-luna": "full"},
        }),
        encoding="utf-8",
    )
    report = reproduce_browsecomp_plus(tmp_path, tmp_path / "reproduced")
    endpoint = report["results"]["gpt-5.6-luna/gold-document-reference"]["endpoint"]
    assert endpoint["correct"] == 1
    assert endpoint["accuracy_percent"] == 50.0
    assert report["paper_consistency_all_match"] is True
