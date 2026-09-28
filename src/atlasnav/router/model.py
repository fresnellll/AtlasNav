#!/usr/bin/env python3
"""Shared inference math for the AtlasNav risk-controlled five-channel Router."""

from __future__ import annotations

from typing import Any

import numpy as np

from atlasnav.router.features import FEATURE_DIMENSIONS, VIEWS


CHANNELS = (*VIEWS, "bm25")
FACET_FLOOR = 0.05
ETA_MINIMUM = 0.20
ETA_MAXIMUM = 0.80


def sigmoid(value: np.ndarray) -> np.ndarray:
    # Preserve float64 during CPU optimization while retaining float32 for the
    # deployed inference path.  Forcing every training forward pass to
    # float32 makes the L-BFGS line search locally discontinuous.
    dtype = np.result_type(np.asarray(value).dtype, np.float32)
    value = np.asarray(value, dtype=dtype)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    dtype = np.result_type(np.asarray(value).dtype, np.float32)
    value = np.asarray(value, dtype=dtype)
    shifted = value - np.max(value, axis=axis, keepdims=True)
    result = np.exp(shifted)
    return result / np.sum(result, axis=axis, keepdims=True)


def router_outputs(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    facet_coefficients: np.ndarray,
    facet_intercept: np.ndarray,
    eta_coefficients: np.ndarray,
    eta_intercept: float,
    confidence_coefficients: np.ndarray,
    confidence_intercept: float,
    facet_temperature: float = 1.0,
    eta_temperature: float = 1.0,
    confidence_temperature: float = 1.0,
    confidence_bias: float = 0.0,
) -> dict[str, np.ndarray]:
    if facet_temperature <= 0 or eta_temperature <= 0 or confidence_temperature <= 0:
        raise ValueError("AtlasNav temperatures must be positive")
    normalized = np.clip(
        (np.asarray(features, dtype=np.float32) - mean) / scale,
        -8.0,
        8.0,
    )
    facet_soft = softmax(
        (normalized @ facet_coefficients.T + facet_intercept) / facet_temperature,
    )
    facet_gate = FACET_FLOOR + (1.0 - len(VIEWS) * FACET_FLOOR) * facet_soft
    eta_unit = sigmoid((normalized @ eta_coefficients + float(eta_intercept)) / eta_temperature)
    eta_gate = ETA_MINIMUM + (ETA_MAXIMUM - ETA_MINIMUM) * eta_unit
    confidence = sigmoid(
        (normalized @ confidence_coefficients + float(confidence_intercept) + confidence_bias)
        / confidence_temperature
    )
    facet = (1.0 - confidence[:, None]) * 0.25 + confidence[:, None] * facet_gate
    eta = 0.5 + confidence * (eta_gate - 0.5)
    channel = np.concatenate(
        (2.0 * eta[:, None] * facet, (2.0 * (1.0 - eta))[:, None]),
        axis=1,
    )
    if not np.allclose(np.sum(channel, axis=1), 2.0, atol=2e-5):
        raise RuntimeError("AtlasNav channel mass invariant failed")
    return {
        "facet_gate": facet_gate.astype(np.float32),
        "facet": facet.astype(np.float32),
        "eta_gate": eta_gate.astype(np.float32),
        "eta": eta.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "channel": channel.astype(np.float32),
    }


def outputs_from_model(features: np.ndarray, model: Any) -> dict[str, np.ndarray]:
    return router_outputs(
        features,
        mean=np.asarray(model["mean"], dtype=np.float32),
        scale=np.asarray(model["scale"], dtype=np.float32),
        facet_coefficients=np.asarray(model["facet_coefficients"], dtype=np.float32),
        facet_intercept=np.asarray(model["facet_intercept"], dtype=np.float32),
        eta_coefficients=np.asarray(model["eta_coefficients"], dtype=np.float32),
        eta_intercept=float(model["eta_intercept"]),
        confidence_coefficients=np.asarray(model["confidence_coefficients"], dtype=np.float32),
        confidence_intercept=float(model["confidence_intercept"]),
        facet_temperature=float(model["facet_temperature"]),
        eta_temperature=float(model["eta_temperature"]),
        confidence_temperature=float(model["confidence_temperature"]),
        confidence_bias=float(model["confidence_bias"]),
    )
