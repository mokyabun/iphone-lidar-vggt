from __future__ import annotations

import numpy as np

from vggt_lidar_scan.generative import _apply_transform, _best_metric_transform, _nearest_distances


def test_metric_alignment_recovers_scale_and_axis_orientation() -> None:
    rng = np.random.default_rng(42)
    target = rng.normal(size=(160, 3)) * np.array([0.04, 0.08, 0.025])
    target[:40] += np.array([0.025, -0.045, 0.018])
    rotation = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    source = target @ rotation * 4.25 + np.array([1.2, -0.7, 0.35])

    transform = _best_metric_transform(source, target)
    aligned = _apply_transform(source, transform)

    assert np.median(_nearest_distances(aligned, target)) < 1e-5
    np.testing.assert_allclose(
        np.percentile(aligned, 99, axis=0) - np.percentile(aligned, 1, axis=0),
        np.percentile(target, 99, axis=0) - np.percentile(target, 1, axis=0),
        rtol=0.02,
        atol=1e-4,
    )
