"""WholeBodyPose: unified COCO-WholeBody (133 keypoints) pose estimation.

Backends (RTMPose, ViTPose, MediaPipe) are imported lazily, so ``import wholebodypose`` only
needs NumPy and OpenCV.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wholebodypose")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+local"

_LAZY = {
    "PoseEstimator": "wholebodypose.api",
    "create_model": "wholebodypose.api",
    "available_backends": "wholebodypose.api",
    "BACKENDS": "wholebodypose.api",
    "Video2Pose": "wholebodypose.video.video2pose",
    "DrawerPose": "wholebodypose.utils.vision",
    "save_poses": "wholebodypose.io",
    "load_poses": "wholebodypose.io",
    "PoseSequenceFilter": "wholebodypose.filters.sequence",
    "FilteringNLMS": "wholebodypose.filters.filtering_nlms",
    "get_weight": "wholebodypose.weights",
    "PoseTracker": "wholebodypose.tracking",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'wholebodypose' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
