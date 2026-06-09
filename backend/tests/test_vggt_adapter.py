from __future__ import annotations

import numpy as np

from vggt_lidar_scan.vggt_adapter import _as_numpy


def test_as_numpy_accepts_numpy_array() -> None:
    value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    converted = _as_numpy(value)

    assert converted is value


def test_as_numpy_accepts_tensor_like_object() -> None:
    class TensorLike:
        def __init__(self) -> None:
            self.value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        def detach(self) -> TensorLike:
            return self

        def float(self) -> TensorLike:
            return self

        def cpu(self) -> TensorLike:
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    converted = _as_numpy(TensorLike())

    assert np.allclose(converted, [[1.0, 2.0, 3.0]])
