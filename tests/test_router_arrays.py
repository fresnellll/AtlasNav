from __future__ import annotations

import numpy as np

from atlasnav.router.arrays import approximate_rrf, channel_targets


def test_approximate_rrf_is_monotone() -> None:
    calibration = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    candidates = np.asarray([[0.15, 0.35, 0.50]], dtype=np.float32)
    rrf, ranks = approximate_rrf(candidates, calibration, documents=100)
    assert ranks.tolist()[0] == sorted(ranks.tolist()[0], reverse=True)
    assert np.all(np.diff(rrf[0]) > 0)


def test_channel_targets_shapes_and_bounds() -> None:
    ranks = np.asarray([
        [[3, 4, 5, 6, 7], [5, 3, 9, 8, 4], [20, 30, 10, 12, 14]],
        [[7, 8, 4, 5, 6], [9, 6, 7, 4, 3], [21, 19, 30, 25, 20]],
    ], dtype=np.int32)
    positive = np.asarray([[True, True, False], [True, False, False]])
    rrf = 1.0 / (60.0 + ranks)
    result = channel_targets(ranks, positive, rrf)
    assert result["facet_targets"].shape == (2, 4)
    assert np.allclose(result["facet_targets"].sum(axis=1), 1.0)
    assert np.all((result["eta_targets"] >= 0.2) & (result["eta_targets"] <= 0.8))
    assert np.all((result["confidence_targets"] >= 0.0) & (result["confidence_targets"] <= 1.0))
