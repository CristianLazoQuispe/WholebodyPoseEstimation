"""Keypoint subsets of the 135-point layout used by the released pose datasets.

The released HDF5 files (``*_135-{Train,Val,Test}.hdf5``) store ``data`` with shape ``(T, 2, 135)``:
COCO-WholeBody 0-132, plus 133 = mid-chest (mean of shoulders 5, 6) and 134 = mid-hip (mean of
hips 11, 12).

``SIGNERT_96`` is the 96-point subset used by SignERT / ConnectingPoints and PocketSign
(``*_96-*.hdf5`` files). It was produced from the same RTMPose parquet files as the 135-point
release, so ``data135[:, :, SIGNERT_96]`` reproduces the original ``*_96`` files bit for bit
(the ``*_96`` files have no ``width``/``height`` entries).
"""

from __future__ import annotations

import numpy as np

#: 96-point subset, as indices into the 135-point layout. Order matters (it is the stored order):
#: 11 upper-body points, 42 hand points (left 91-111, right 112-132), mid-chest (133), then 42 face
#: points: mouth 72-90, left eye, right eye, left eyebrow, right eyebrow, and mouth 71 last
#: (the original script swapped mid-chest with point 71).
SIGNERT_96: tuple[int, ...] = (
    *range(0, 11),  # nose, eyes, ears, shoulders, elbows, wrists
    *range(91, 133),  # left hand (91-111), right hand (112-132)
    133,  # mid-chest
    *range(72, 91),  # mouth (without 71)
    65, 66, 68, 70, 67, 69,  # left eye
    59, 61, 62, 63, 60, 64,  # right eye
    *range(45, 50),  # left eyebrow
    *range(40, 45),  # right eyebrow
    71,  # mouth corner
)  # fmt: skip

SUBSETS: dict[str, tuple[int, ...]] = {"signert96": SIGNERT_96}


def select_keypoints(data: np.ndarray, subset: str | tuple[int, ...] = "signert96", axis: int = -1) -> np.ndarray:
    """Select a keypoint subset from 135-point data.

    Args:
        data: Array whose ``axis`` has 135 keypoints, e.g. ``(T, 2, 135)`` as stored in the HDF5
            files (default ``axis=-1``) or ``(T, 135, 2)`` with ``axis=1``.
        subset: Name in ``SUBSETS`` or an explicit index tuple.
        axis: Keypoint axis.
    """
    idx = SUBSETS[subset] if isinstance(subset, str) else tuple(subset)
    data = np.asarray(data)
    if data.shape[axis] != 135:
        raise ValueError(f"expected 135 keypoints on axis {axis}, got shape {data.shape}")
    return np.take(data, idx, axis=axis)
