"""Save and load pose sequences (``.npz`` or ``.json``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def save_poses(path: str | Path, keypoints, scores, **meta) -> Path:
    """Save keypoints ``(T, 133, 2)`` and scores ``(T, 133)`` plus JSON-serializable metadata.

    The format is chosen from the suffix: ``.npz`` (compressed NumPy) or ``.json``.
    Multi-person sequences (lists of ``(N_t, 133, ...)`` arrays) are supported in both formats.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ragged = isinstance(keypoints, list) and len({np.shape(k) for k in keypoints}) > 1
    meta = {"format_version": FORMAT_VERSION, "layout": "coco_wholebody_133", **meta}

    if path.suffix == ".json":
        payload = {
            "meta": meta,
            "keypoints": [np.asarray(k).round(2).tolist() for k in keypoints],
            "scores": [np.asarray(s).round(4).tolist() for s in scores],
        }
        path.write_text(json.dumps(payload))
    elif path.suffix == ".npz":
        if ragged:  # flatten people over time + per-frame counts (no pickle needed to load)
            counts = np.array([len(k) for k in keypoints], np.int64)
            kp = np.concatenate([np.asarray(k, np.float32).reshape(-1, 133, 2) for k in keypoints])
            sc = np.concatenate([np.asarray(s, np.float32).reshape(-1, 133) for s in scores])
            np.savez_compressed(path, keypoints=kp, scores=sc, counts=counts, meta=json.dumps(meta))
        else:
            kp, sc = np.asarray(keypoints, np.float32), np.asarray(scores, np.float32)
            np.savez_compressed(path, keypoints=kp, scores=sc, meta=json.dumps(meta))
    else:
        raise ValueError(f"Unsupported format {path.suffix!r}; use .npz or .json")
    return path


def load_poses(path: str | Path) -> tuple[np.ndarray | list, np.ndarray | list, dict]:
    """Inverse of :func:`save_poses`. Returns ``(keypoints, scores, meta)``."""
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        kp = [np.asarray(k, np.float32) for k in payload["keypoints"]]
        sc = [np.asarray(s, np.float32) for s in payload["scores"]]
        if len({k.shape for k in kp}) <= 1:
            kp, sc = np.asarray(kp, np.float32), np.asarray(sc, np.float32)
        return kp, sc, payload["meta"]
    with np.load(path, allow_pickle=False) as data:
        kp, sc = data["keypoints"], data["scores"]
        meta = json.loads(str(data["meta"]))
        if "counts" in data.files:
            splits = np.cumsum(data["counts"])[:-1]
            kp, sc = np.split(kp, splits), np.split(sc, splits)
        return kp, sc, meta
