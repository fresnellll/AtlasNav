#!/usr/bin/env python3
"""Train and evaluate the CPU-only AtlasNav multi-positive risk Router."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from atlasnav.router.features import FEATURE_DIMENSIONS, VIEWS
from atlasnav.router.model import FACET_FLOOR, router_outputs, sigmoid, softmax
from atlasnav.io import sha256_file, atomic_json


DEFAULT_ARRAYS = Path("artifacts/router/training_arrays")
DEFAULT_OUTPUT = Path("artifacts/router/model")
SCHEMA = "atlasnav_linear_router_v1"
RETRIEVAL_TEMPERATURE = 0.002
BOTTLENECK_POSITIVE_TEMPERATURE = 0.003
BOTTLENECK_NEGATIVE_TEMPERATURE = 0.003
BOTTLENECK_MARGIN_TEMPERATURE = 0.004
BOTTLENECK_LAMBDA = 0.35
RISK_LAMBDA = 0.25
FACET_TARGET_LAMBDA = 0.12
ETA_TARGET_LAMBDA = 0.10
CONFIDENCE_TARGET_LAMBDA = 0.15
L2 = 2e-4


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--array-dir", type=Path, default=DEFAULT_ARRAYS)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--maxiter", type=int, default=100)
    result.add_argument("--audit-only", action="store_true")
    return result


def parameter_size(dimensions: int) -> int:
    return 4 * dimensions + 4 + dimensions + 1 + dimensions + 1


def unpack(flat: np.ndarray, dimensions: int) -> tuple[np.ndarray, ...]:
    cursor = 0
    facet_w = flat[cursor:cursor + 4 * dimensions].reshape(4, dimensions)
    cursor += 4 * dimensions
    facet_b = flat[cursor:cursor + 4]
    cursor += 4
    eta_w = flat[cursor:cursor + dimensions]
    cursor += dimensions
    eta_b = flat[cursor]
    cursor += 1
    confidence_w = flat[cursor:cursor + dimensions]
    cursor += dimensions
    confidence_b = flat[cursor]
    if cursor + 1 != len(flat):
        raise RuntimeError("AtlasNav parameter unpack mismatch")
    return facet_w, facet_b, eta_w, eta_b, confidence_w, confidence_b


def stable_masked_softmax(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, value, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    exponential = np.where(mask, np.exp(masked - maximum), 0.0)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def all_positive_loss_and_gradient(
    score: np.ndarray, positive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = score / RETRIEVAL_TEMPERATURE
    maximum = np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(logits - maximum)
    negative = ~positive
    negative_sum = np.sum(exponential * negative, axis=1, keepdims=True)
    denominator = negative_sum + exponential
    count = np.sum(positive, axis=1, keepdims=True)
    query_loss = np.sum(
        positive * (np.log(denominator + 1e-30) - (logits - maximum)), axis=1,
    ) / count[:, 0]
    inverse_positive_denominator = np.sum(
        np.where(positive, 1.0 / (denominator + 1e-30), 0.0), axis=1, keepdims=True,
    ) / count
    d_logits = np.where(
        negative,
        exponential * inverse_positive_denominator,
        np.where(positive, (exponential / (denominator + 1e-30) - 1.0) / count, 0.0),
    )
    # L-BFGS-B uses finite line-search differences that are much smaller than
    # the retrieval loss.  Keep the analytic objective in float64; casting it
    # to float32 here creates avoidable gradient noise and false convergence.
    return query_loss.astype(np.float64), (d_logits / RETRIEVAL_TEMPERATURE).astype(np.float64)


def bottleneck_loss_and_gradient(
    score: np.ndarray, positive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    negative = ~positive
    positive_soft = stable_masked_softmax(-score / BOTTLENECK_POSITIVE_TEMPERATURE, positive)
    negative_soft = stable_masked_softmax(score / BOTTLENECK_NEGATIVE_TEMPERATURE, negative)
    positive_value = -BOTTLENECK_POSITIVE_TEMPERATURE * np.log(
        np.sum(np.where(positive, np.exp(-score / BOTTLENECK_POSITIVE_TEMPERATURE), 0.0), axis=1) + 1e-30
    )
    negative_max = np.max(np.where(negative, score, -np.inf), axis=1)
    negative_value = negative_max + BOTTLENECK_NEGATIVE_TEMPERATURE * np.log(
        np.sum(
            np.where(
                negative,
                np.exp((score - negative_max[:, None]) / BOTTLENECK_NEGATIVE_TEMPERATURE),
                0.0,
            ),
            axis=1,
        ) + 1e-30
    )
    margin = (negative_value - positive_value) / BOTTLENECK_MARGIN_TEMPERATURE
    query_loss = np.logaddexp(0.0, margin)
    active = sigmoid(margin) / BOTTLENECK_MARGIN_TEMPERATURE
    gradient = active[:, None] * (negative_soft - positive_soft)
    return query_loss.astype(np.float64), gradient.astype(np.float64)


def forward(flat: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    dimensions = x.shape[1]
    facet_w, facet_b, eta_w, eta_b, confidence_w, confidence_b = unpack(flat, dimensions)
    facet_soft = softmax(x @ facet_w.T + facet_b)
    facet_gate = FACET_FLOOR + (1.0 - 4.0 * FACET_FLOOR) * facet_soft
    eta_unit = sigmoid(x @ eta_w + eta_b)
    eta_gate = 0.20 + 0.60 * eta_unit
    confidence = sigmoid(x @ confidence_w + confidence_b)
    facet = (1.0 - confidence[:, None]) * 0.25 + confidence[:, None] * facet_gate
    eta = 0.5 + confidence * (eta_gate - 0.5)
    channel = np.concatenate(
        (2.0 * eta[:, None] * facet, (2.0 * (1.0 - eta))[:, None]), axis=1,
    )
    return {
        "facet_soft": facet_soft,
        "facet_gate": facet_gate,
        "eta_unit": eta_unit,
        "eta_gate": eta_gate,
        "confidence": confidence,
        "facet": facet,
        "eta": eta,
        "channel": channel,
    }


def objective(
    flat: np.ndarray,
    x: np.ndarray,
    rrf: np.ndarray,
    positive: np.ndarray,
    sample_weight: np.ndarray,
    facet_target: np.ndarray,
    eta_target: np.ndarray,
    confidence_target: np.ndarray,
    baseline_query_loss: np.ndarray,
) -> tuple[float, np.ndarray]:
    output = forward(flat, x)
    score = np.einsum("qkc,qc->qk", rrf, output["channel"], optimize=True)
    all_loss, all_gradient = all_positive_loss_and_gradient(score, positive)
    bottleneck_loss, bottleneck_gradient = bottleneck_loss_and_gradient(score, positive)
    weight = np.asarray(sample_weight, dtype=np.float64)
    weight /= np.sum(weight)
    risk_active = all_loss > baseline_query_loss
    query_loss = all_loss + BOTTLENECK_LAMBDA * bottleneck_loss
    loss = float(np.sum(weight * query_loss))
    loss += RISK_LAMBDA * float(np.sum(weight * np.maximum(all_loss - baseline_query_loss, 0.0)))
    d_score = (
        all_gradient + BOTTLENECK_LAMBDA * bottleneck_gradient
        + RISK_LAMBDA * risk_active[:, None] * all_gradient
    ) * weight[:, None]
    d_channel = np.einsum("qk,qkc->qc", d_score, rrf, optimize=True)

    d_facet = d_channel[:, :4] * (2.0 * output["eta"][:, None])
    d_eta = 2.0 * np.sum(d_channel[:, :4] * output["facet"], axis=1) - 2.0 * d_channel[:, -1]
    d_facet_gate = d_facet * output["confidence"][:, None]
    d_confidence = np.sum(d_facet * (output["facet_gate"] - 0.25), axis=1)
    d_eta_gate = d_eta * output["confidence"]
    d_confidence += d_eta * (output["eta_gate"] - 0.5)
    centered = d_facet_gate - np.sum(d_facet_gate * output["facet_soft"], axis=1, keepdims=True)
    d_facet_logits = (1.0 - 4.0 * FACET_FLOOR) * output["facet_soft"] * centered
    d_eta_logits = d_eta_gate * 0.60 * output["eta_unit"] * (1.0 - output["eta_unit"])
    d_confidence_logits = d_confidence * output["confidence"] * (1.0 - output["confidence"])

    eps = 1e-7
    supervised_weight = weight[:, None]
    facet_ce = -np.sum(facet_target * np.log(output["facet_soft"] + eps), axis=1)
    loss += FACET_TARGET_LAMBDA * float(np.sum(weight * facet_ce))
    d_facet_logits += FACET_TARGET_LAMBDA * supervised_weight * (output["facet_soft"] - facet_target)
    eta_unit_target = np.clip((eta_target - 0.20) / 0.60, eps, 1.0 - eps)
    eta_bce = -eta_unit_target * np.log(output["eta_unit"] + eps) - (1.0 - eta_unit_target) * np.log(1.0 - output["eta_unit"] + eps)
    loss += ETA_TARGET_LAMBDA * float(np.sum(weight * eta_bce))
    d_eta_logits += ETA_TARGET_LAMBDA * weight * (output["eta_unit"] - eta_unit_target)
    confidence_target = np.clip(confidence_target, eps, 1.0 - eps)
    confidence_bce = -confidence_target * np.log(output["confidence"] + eps) - (1.0 - confidence_target) * np.log(1.0 - output["confidence"] + eps)
    loss += CONFIDENCE_TARGET_LAMBDA * float(np.sum(weight * confidence_bce))
    d_confidence_logits += CONFIDENCE_TARGET_LAMBDA * weight * (output["confidence"] - confidence_target)

    dimensions = x.shape[1]
    facet_w, _facet_b, eta_w, _eta_b, confidence_w, _confidence_b = unpack(flat, dimensions)
    gradient_facet_w = d_facet_logits.T @ x + L2 * facet_w
    gradient_facet_b = np.sum(d_facet_logits, axis=0)
    gradient_eta_w = d_eta_logits @ x + L2 * eta_w
    gradient_eta_b = np.sum(d_eta_logits)
    gradient_confidence_w = d_confidence_logits @ x + L2 * confidence_w
    gradient_confidence_b = np.sum(d_confidence_logits)
    loss += 0.5 * L2 * float(
        np.sum(facet_w ** 2) + np.sum(eta_w ** 2) + np.sum(confidence_w ** 2)
    )
    gradient = np.concatenate((
        gradient_facet_w.ravel(), gradient_facet_b,
        gradient_eta_w, np.asarray([gradient_eta_b]),
        gradient_confidence_w, np.asarray([gradient_confidence_b]),
    ))
    return loss, gradient.astype(np.float64)


def candidate_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    positions = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(
        positions, order,
        np.arange(1, scores.shape[1] + 1, dtype=np.int32)[None, :], axis=1,
    )
    return positions


def metrics(
    name: str,
    channel: np.ndarray,
    rrf: np.ndarray,
    positive: np.ndarray,
    mask: np.ndarray,
    kind: np.ndarray,
    rare: np.ndarray,
    confidence: np.ndarray | None = None,
    eta: np.ndarray | None = None,
    facet: np.ndarray | None = None,
    baseline_channel: np.ndarray | None = None,
) -> dict[str, Any]:
    indexes = np.flatnonzero(mask)
    scores = np.einsum("qkc,qc->qk", rrf[indexes], channel, optimize=True)
    positions = candidate_positions(scores)
    positive_positions = [positions[row][positive[index]] for row, index in enumerate(indexes)]
    worst = np.asarray([int(np.max(value)) for value in positive_positions], dtype=np.int32)
    mean_rr = np.asarray([float(np.mean(1.0 / value)) for value in positive_positions], dtype=np.float32)
    all30 = np.asarray([bool(np.all(value <= 30)) for value in positive_positions])
    result: dict[str, Any] = {
        "name": name,
        "queries": len(indexes),
        "worst_positive_mrr": float(np.mean(1.0 / worst)),
        "mean_positive_mrr": float(np.mean(mean_rr)),
        "all_positive_recall_at_10": float(np.mean([np.all(value <= 10) for value in positive_positions])),
        "all_positive_recall_at_30": float(np.mean(all30)),
        "mean_worst_positive_rank": float(np.mean(worst)),
        "by_kind": {},
    }
    for label, code in (("single", 1), ("pair", 2), ("triple", 3)):
        local = kind[indexes] == code
        result["by_kind"][label] = {
            "queries": int(np.sum(local)),
            "worst_positive_mrr": float(np.mean(1.0 / worst[local])) if np.any(local) else None,
            "all_positive_recall_at_30": float(np.mean(all30[local])) if np.any(local) else None,
            "mean_worst_positive_rank": float(np.mean(worst[local])) if np.any(local) else None,
        }
    local_rare = rare[indexes]
    result["rare_all_positive_recall_at_30"] = float(np.mean(all30[local_rare])) if np.any(local_rare) else None
    result["rare_queries"] = int(np.sum(local_rare))
    if confidence is not None and eta is not None and facet is not None:
        result.update({
            "confidence_mean": float(np.mean(confidence)),
            "confidence_min": float(np.min(confidence)),
            "confidence_max": float(np.max(confidence)),
            "eta_mean": float(np.mean(eta)),
            "eta_min": float(np.min(eta)),
            "eta_max": float(np.max(eta)),
            "facet_mean": np.mean(facet, axis=0).tolist(),
            "facet_min": np.min(facet, axis=0).tolist(),
        })
        if baseline_channel is not None:
            baseline_scores = np.einsum("qkc,qc->qk", rrf[indexes], baseline_channel, optimize=True)
            baseline_positions = candidate_positions(baseline_scores)
            baseline_worst = np.asarray([
                int(np.max(baseline_positions[row][positive[index]])) for row, index in enumerate(indexes)
            ])
            gain = 1.0 / worst - 1.0 / baseline_worst
            quantile = np.quantile(confidence, [0.25, 0.75])
            result["confidence_calibration"] = {
                "low_quartile_mean_gain": float(np.mean(gain[confidence <= quantile[0]])),
                "high_quartile_mean_gain": float(np.mean(gain[confidence >= quantile[1]])),
                "positive_gain_rate": float(np.mean(gain > 0)),
                "negative_gain_rate": float(np.mean(gain < 0)),
            }
    return result


def prefix_mask(train: np.ndarray, kind: np.ndarray, identifiers: list[str], fraction: float) -> np.ndarray:
    selected = np.zeros(len(train), dtype=bool)
    for code in (1, 2, 3):
        indexes = np.flatnonzero(train & (kind == code))
        ordered = sorted(indexes, key=lambda index: hashlib.sha256(identifiers[index].encode()).hexdigest())
        count = max(1, round(len(ordered) * fraction))
        selected[ordered[:count]] = True
    return selected


def train(args: argparse.Namespace) -> dict[str, Any]:
    arrays_dir = args.array_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("refusing to replace trained Router")
    manifest = json.loads((arrays_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "atlasnav_router_training_arrays_v1" or manifest.get("finalized") is not True:
        raise RuntimeError("AtlasNav finalized training arrays are required")
    arrays = {path.name.split(".")[0]: np.load(path, mmap_mode="r") for path in arrays_dir.glob("*.npy")}
    x_raw = np.asarray(arrays["features"], dtype=np.float32)
    rrf = np.asarray(arrays["channel_rrf"], dtype=np.float32)
    positive = np.asarray(arrays["positive_mask"], dtype=bool)
    split = np.asarray(arrays["split"], dtype=np.int8)
    kind = np.asarray(arrays["kind"], dtype=np.int8)
    rare = np.asarray(arrays["rare_query"], dtype=bool)
    sample_weight = np.asarray(arrays["sample_weight"], dtype=np.float32)
    facet_target = np.asarray(arrays["facet_targets"], dtype=np.float32)
    eta_target = np.asarray(arrays["eta_targets"], dtype=np.float32)
    confidence_target = np.asarray(arrays["confidence_targets"], dtype=np.float32)
    with gzip.open(arrays_dir / "query_ids.jsonl.gz", "rt", encoding="utf-8") as stream:
        identifiers = [json.loads(line)["pseudoquery_id"] for line in stream]
    train_mask, validation_mask, test_mask = split == 0, split == 1, split == 2
    mean = np.mean(x_raw[train_mask], axis=0)
    scale = np.std(x_raw[train_mask], axis=0)
    scale[scale < 1e-5] = 1.0
    x = np.clip((x_raw - mean) / scale, -8.0, 8.0).astype(np.float32)
    baseline_channel_all = np.tile(np.asarray([0.25, 0.25, 0.25, 0.25, 1.0], dtype=np.float32), (len(x), 1))
    baseline_score = np.einsum("qkc,qc->qk", rrf, baseline_channel_all, optimize=True)
    baseline_query_loss, _ = all_positive_loss_and_gradient(baseline_score, positive)
    baseline_validation = metrics(
        "uniform_safe_baseline", baseline_channel_all[validation_mask], rrf, positive, validation_mask, kind, rare,
    )
    baseline_test = metrics(
        "uniform_safe_baseline", baseline_channel_all[test_mask], rrf, positive, test_mask, kind, rare,
    )
    initial = np.zeros(parameter_size(x.shape[1]), dtype=np.float64)
    learning_curve: list[dict[str, Any]] = []
    final_flat: np.ndarray | None = None
    for fraction in (0.25, 0.50, 0.75, 1.00):
        current = prefix_mask(train_mask, kind, identifiers, fraction)
        print(f"AtlasNav optimize fraction={fraction:.2f} queries={int(np.sum(current))}", flush=True)
        result = minimize(
            objective,
            initial,
            args=(
                x[current], rrf[current], positive[current], sample_weight[current],
                facet_target[current], eta_target[current], confidence_target[current],
                baseline_query_loss[current],
            ),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": args.maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 30},
        )
        raw = forward(result.x, x[validation_mask])
        report = metrics(
            "router_uncalibrated", raw["channel"], rrf, positive, validation_mask, kind, rare,
            raw["confidence"], raw["eta"], raw["facet"], baseline_channel_all[validation_mask],
        )
        learning_curve.append({
            "fraction": fraction,
            "train_queries": int(np.sum(current)),
            "optimizer": {"success": bool(result.success), "iterations": int(result.nit), "loss": float(result.fun), "message": str(result.message)},
            "validation": report,
        })
        if fraction == 1.0:
            final_flat = result.x.copy()
        initial = result.x.copy()
    assert final_flat is not None
    facet_w, facet_b, eta_w, eta_b, confidence_w, confidence_b = unpack(final_flat, x.shape[1])

    trials: list[dict[str, Any]] = []
    for facet_temperature in (0.70, 1.0, 1.30):
        for eta_temperature in (0.70, 1.0, 1.30):
            for confidence_temperature in (0.70, 1.0, 1.30):
                for confidence_bias in (-1.0, 0.0, 1.0):
                    output_values = router_outputs(
                        x_raw[validation_mask], mean=mean, scale=scale,
                        facet_coefficients=facet_w, facet_intercept=facet_b,
                        eta_coefficients=eta_w, eta_intercept=eta_b,
                        confidence_coefficients=confidence_w, confidence_intercept=confidence_b,
                        facet_temperature=facet_temperature, eta_temperature=eta_temperature,
                        confidence_temperature=confidence_temperature, confidence_bias=confidence_bias,
                    )
                    report = metrics(
                        "router", output_values["channel"], rrf, positive, validation_mask, kind, rare,
                        output_values["confidence"], output_values["eta"], output_values["facet"],
                        baseline_channel_all[validation_mask],
                    )
                    single = report["by_kind"]["single"]
                    baseline_single = baseline_validation["by_kind"]["single"]
                    eligible = (
                        single["worst_positive_mrr"] >= baseline_single["worst_positive_mrr"] - 0.002
                        and report["rare_all_positive_recall_at_30"] >= baseline_validation["rare_all_positive_recall_at_30"] - 0.005
                        and 0.22 <= report["eta_mean"] <= 0.78
                    )
                    composite = (
                        math.log(max(report["worst_positive_mrr"], 1e-9))
                        + math.log(max(report["all_positive_recall_at_30"], 1e-9))
                        + 0.35 * math.log(max(report["by_kind"]["triple"]["all_positive_recall_at_30"], 1e-9))
                    )
                    trials.append({
                        "facet_temperature": facet_temperature,
                        "eta_temperature": eta_temperature,
                        "confidence_temperature": confidence_temperature,
                        "confidence_bias": confidence_bias,
                        "eligible": eligible,
                        "composite": composite,
                        "validation": report,
                    })
    eligible_trials = [trial for trial in trials if trial["eligible"]] or trials
    selected = max(eligible_trials, key=lambda trial: (trial["composite"], trial["validation"]["worst_positive_mrr"]))
    test_output = router_outputs(
        x_raw[test_mask], mean=mean, scale=scale,
        facet_coefficients=facet_w, facet_intercept=facet_b,
        eta_coefficients=eta_w, eta_intercept=eta_b,
        confidence_coefficients=confidence_w, confidence_intercept=confidence_b,
        facet_temperature=selected["facet_temperature"], eta_temperature=selected["eta_temperature"],
        confidence_temperature=selected["confidence_temperature"], confidence_bias=selected["confidence_bias"],
    )
    test_report = metrics(
        "router", test_output["channel"], rrf, positive, test_mask, kind, rare,
        test_output["confidence"], test_output["eta"], test_output["facet"], baseline_channel_all[test_mask],
    )
    gates = {
        "overall_worst_positive_mrr_gain_at_least_3pct": test_report["worst_positive_mrr"] >= 1.03 * baseline_test["worst_positive_mrr"],
        "pair_all_positive_recall_at_30_not_lower": test_report["by_kind"]["pair"]["all_positive_recall_at_30"] >= baseline_test["by_kind"]["pair"]["all_positive_recall_at_30"],
        "triple_all_positive_recall_at_30_not_lower": test_report["by_kind"]["triple"]["all_positive_recall_at_30"] >= baseline_test["by_kind"]["triple"]["all_positive_recall_at_30"],
        "single_mrr_not_lower": test_report["by_kind"]["single"]["worst_positive_mrr"] >= baseline_test["by_kind"]["single"]["worst_positive_mrr"] - 0.002,
        "rare_all_positive_recall_at_30_not_lower": test_report["rare_all_positive_recall_at_30"] >= baseline_test["rare_all_positive_recall_at_30"] - 1e-12,
        "high_confidence_gain_positive": test_report["confidence_calibration"]["high_quartile_mean_gain"] > 0,
        "eta_not_collapsed": 0.22 <= test_report["eta_mean"] <= 0.78 and test_report["eta_min"] > 0.19 and test_report["eta_max"] < 0.81,
        "facet_residual_floor": min(test_report["facet_min"]) >= FACET_FLOOR - 1e-6,
    }
    gates["passed"] = all(gates.values())
    output.mkdir(parents=True)
    model_path = output / "router_model.npz"
    np.savez_compressed(
        model_path,
        mean=mean.astype(np.float32), scale=scale.astype(np.float32),
        facet_coefficients=facet_w.astype(np.float32), facet_intercept=facet_b.astype(np.float32),
        eta_coefficients=eta_w.astype(np.float32), eta_intercept=np.float32(eta_b),
        confidence_coefficients=confidence_w.astype(np.float32), confidence_intercept=np.float32(confidence_b),
        facet_temperature=np.float32(selected["facet_temperature"]),
        eta_temperature=np.float32(selected["eta_temperature"]),
        confidence_temperature=np.float32(selected["confidence_temperature"]),
        confidence_bias=np.float32(selected["confidence_bias"]),
        views=np.asarray(VIEWS), channels=np.asarray((*VIEWS, "bm25")),
    )
    report = {
        "schema": SCHEMA,
        "finalized": True,
        "created_at_unix": time.time(),
        "parameters": parameter_size(x.shape[1]),
        "training_objective": {
            "all_positive": True,
            "bottleneck_lambda": BOTTLENECK_LAMBDA,
            "risk_lambda": RISK_LAMBDA,
            "facet_target_lambda": FACET_TARGET_LAMBDA,
            "eta_target_lambda": ETA_TARGET_LAMBDA,
            "confidence_target_lambda": CONFIDENCE_TARGET_LAMBDA,
            "l2": L2,
        },
        "learning_curve": learning_curve,
        "validation_baseline": baseline_validation,
        "selected": selected,
        "test": {"router": test_report, "uniform_safe_baseline": baseline_test},
        "mechanism_gates": gates,
        "model": model_path.name,
        "model_sha256": sha256_file(model_path),
        "array_manifest_sha256": sha256_file(arrays_dir / "manifest.json"),
        "evaluation_artifact_input": False,
        "gold_answer_qrel_judge_correctness_or_trajectory_input": False,
    }
    atomic_json(output / "manifest.json", report)
    return report


def audit(output: Path) -> dict[str, Any]:
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("schema") != SCHEMA or report.get("finalized") is not True:
        errors.append("manifest identity mismatch")
    model = output / str(report.get("model", ""))
    if not model.is_file() or sha256_file(model) != report.get("model_sha256"):
        errors.append("model hash mismatch")
    else:
        values = np.load(model)
        if values["facet_coefficients"].shape != (4, FEATURE_DIMENSIONS):
            errors.append("facet coefficient shape mismatch")
        if values["eta_coefficients"].shape != (FEATURE_DIMENSIONS,):
            errors.append("eta coefficient shape mismatch")
    result = {
        "schema": "atlasnav_linear_router_audit_v1",
        "passed": not errors,
        "errors": errors,
        "mechanism_gates": report.get("mechanism_gates"),
        "paid_api_calls": 0,
    }
    if errors:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parser().parse_args()
    output = args.output_dir.resolve()
    result = audit(output) if args.audit_only or output.exists() else train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
