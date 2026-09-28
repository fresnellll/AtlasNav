"""Render a stable Markdown report from an AtlasNav evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlasnav.io import atomic_json


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def render_report(evaluation: Path, output: Path) -> dict[str, Any]:
    source = evaluation / "summary.json" if evaluation.is_dir() else evaluation
    value = json.loads(source.read_text(encoding="utf-8"))
    endpoint = value["endpoint"]
    lines = [
        "# AtlasNav evaluation report", "",
        "## Endpoint", "",
        "| Queries | Correct | Strict accuracy | Invalid terminals | Turns | Recorded cost |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {endpoint['queries']} | {endpoint['correct']} | {_percent(endpoint['strict_accuracy'])} | "
        f"{endpoint['invalid_terminals']} | {endpoint['turns']} | "
        f"{endpoint['recorded_online_cost'] if endpoint['recorded_online_cost'] is not None else 'mixed currencies'} "
        f"{endpoint.get('currency') or ''} |",
    ]
    evidence = value.get("evidence_blindness")
    if evidence:
        lines.extend([
            "", "## Five-stage evidence funnel", "",
            "Construction checks whether the annotated evidence is in the indexed corpus. Surface checks whether "
            "a supporting canonical document becomes model-visible. Open requires canonical body exposure. "
            "Locate requires an independently frozen supporting span to occur in that body exposure. Closure is "
            "the final judged answer.", "",
            "| Stage | Any realization | Macro slot recall | Micro slot recall | All realization | All blindness |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        construction = evidence.get("construction") or {}
        lines.append(
            f"| Construction | {_percent(1.0 - float(construction.get('any', 0.0)))} | — | — | "
            f"{_percent(1.0 - float(construction.get('all', 0.0)))} | {_percent(construction.get('all'))} |"
        )
        for stage in ("surface", "open", "locate"):
            row = evidence[stage]
            lines.append(
                f"| {stage.title()} | {_percent(row['realization_any'])} | {_percent(row['macro_recall'])} | "
                f"{_percent(row['micro_recall'])} | {_percent(row['realization_all'])} | {_percent(row['all'])} |"
            )
        closure = evidence.get("closure") or {}
        lines.append(
            f"| Closure | {_percent(closure.get('realization'))} | — | — | "
            f"{_percent(closure.get('realization'))} | {_percent(closure.get('blindness'))} |"
        )
    lines.extend([
        "", "## Accounting note", "",
        "Costs are reconstructed only from provider-reported token usage and the selected model profile. "
        "Different currencies are never silently summed. Offline Atlas construction is reported separately.", "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    result = {"schema": "atlasnav_markdown_report_v1", "source": str(source), "output": str(output)}
    atomic_json(output.with_suffix(output.suffix + ".json"), result)
    return result
