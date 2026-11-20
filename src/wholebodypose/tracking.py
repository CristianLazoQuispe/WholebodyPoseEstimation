"""Lightweight multi-person tracking: stable person IDs across video frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pose_bbox(keypoints: np.ndarray, scores: np.ndarray, thr: float = 0.0) -> np.ndarray | None:
    """``[x0, y0, x1, y1]`` of the keypoints with ``score > thr`` (``None`` if there are none)."""
    vis = scores > thr
    if not vis.any():
        return None
    pts = keypoints[vis]
    return np.array([*pts.min(axis=0), *pts.max(axis=0)], np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    last_seen: int


class PoseTracker:
    """Greedy IoU tracker over pose bounding boxes.

    Each call to :meth:`update` receives the people of one frame and returns one integer ID per
    person. IDs persist while a person keeps overlapping (IoU >= ``iou_thr``) with a track seen in
    the last ``max_age`` frames; new people get new IDs. Good enough for the typical sign-language
    / dance video with a few people; it is not a re-identification model.

    Example:
        >>> tracker = PoseTracker()
        >>> ids = tracker.update(keypoints, scores)  # doctest: +SKIP
    """

    def __init__(self, iou_thr: float = 0.3, max_age: int = 15, score_thr: float = 0.0) -> None:
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.score_thr = score_thr
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_Track] = []
        self.frame = -1
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def update(self, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Assign track IDs to the ``N`` people of the current frame; returns ``(N,)`` int array."""
        self.frame += 1
        self.tracks = [t for t in self.tracks if self.frame - t.last_seen <= self.max_age]
        boxes = [pose_bbox(k, s, self.score_thr) for k, s in zip(keypoints, scores, strict=True)]
        ids = np.full(len(boxes), -1, np.int64)

        pairs = sorted(
            (
                (bbox_iou(b, t.bbox), i, j)
                for i, b in enumerate(boxes)
                if b is not None
                for j, t in enumerate(self.tracks)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        for iou, i, j in pairs:
            if iou < self.iou_thr:
                break
            if ids[i] != -1 or j in used_tracks:
                continue
            ids[i] = self.tracks[j].track_id
            self.tracks[j].bbox, self.tracks[j].last_seen = boxes[i], self.frame
            used_tracks.add(j)

        for i, b in enumerate(boxes):
            if ids[i] == -1:
                ids[i] = self._new_id()
                if b is not None:
                    self.tracks.append(_Track(int(ids[i]), b, self.frame))
        return ids
