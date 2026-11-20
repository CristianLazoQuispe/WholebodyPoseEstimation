"""Moving mean/median smoothing of keypoint streams."""

from __future__ import annotations

import numpy as np


class MeanMedianFilter:
    """Online smoother: average of the moving mean and moving median over ``window_size`` frames."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self.buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self.buffer.clear()

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        self.buffer.append(np.asarray(keypoints, np.float32))
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        if len(self.buffer) < self.window_size:
            return np.asarray(keypoints, np.float32)
        arr = np.stack(self.buffer)
        return (arr.mean(axis=0) + np.median(arr, axis=0)) / 2
