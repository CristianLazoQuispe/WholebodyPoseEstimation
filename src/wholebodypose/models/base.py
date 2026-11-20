"""Common interface shared by every pose backend."""

from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 133


class BasePoseModel:
    """Base class for whole-body pose backends.

    Every backend takes an **RGB** ``uint8`` image of shape ``(H, W, 3)`` and returns

    * ``keypoints``: ``float32`` array ``(N, 133, 2)`` with pixel ``(x, y)`` coordinates in
      COCO-WholeBody order, and
    * ``scores``: ``float32`` array ``(N, 133)`` with per-keypoint confidences,

    where ``N`` is the number of detected people (``N == 0`` when nobody is found).
    Undetected keypoints are stored as ``(0, 0)`` with score ``0``.
    """

    name: str = "base"

    def __init__(self, use_thresholding: bool = False, kpt_thr: float = 0.5) -> None:
        self.use_thresholding = use_thresholding
        self.kpt_thr = kpt_thr

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate whole-body keypoints on an RGB image."""
        keypoints, scores = self._predict(image_rgb)
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, NUM_KEYPOINTS, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, NUM_KEYPOINTS)
        if self.use_thresholding:
            keypoints, scores = self.thresholding(keypoints, scores)
        return keypoints, scores

    __call__ = predict

    def _predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def thresholding(self, keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Binarize scores with ``kpt_thr`` and zero-out keypoints below it."""
        keypoints = keypoints.copy()
        low = scores < self.kpt_thr
        keypoints[low] = 0
        scores = np.where(low, 0.0, 1.0).astype(np.float32)
        return keypoints, scores
