"""Read and write the ISLR pose HDF5 files published with the ICMV 2025 paper.

Schema (one file per split, e.g. ``WLASL100_135-Test.hdf5``)::

    /<i>                      one group per sample, named "0", "1", ..., "N-1"
        data        float64  (T, 2, 135)  x / width, y / height; undetected keypoints are 0.0
        width       int64    ()           original video width  (pixels)
        height      int64    ()           original video height (pixels)
        label       vlen str ()           gloss (UTF-8)
        video_name  vlen str ()           video identifier / path, as published
        scores      float32  (T, 135)     optional (``save_scores=True``), not in the published files

Keypoints 0-132 are COCO-WholeBody; 133 is the mid-chest (mean of shoulders 5 and 6) and 134 the
mid-hip (mean of hips 11 and 12). Every frame of the video is stored (no subsampling).

h5py iterates groups in alphabetical order ("0", "1", "10", ...); use :func:`iter_samples` or
:func:`read_hdf5`, which return samples in numeric order.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

NUM_COCO = 133
NUM_KEYPOINTS = 135
MID_CHEST, MID_HIP = 133, 134
SHOULDERS, HIPS = (5, 6), (11, 12)
SPLIT_SUFFIX = {"train": "Train", "val": "Val", "test": "Test"}


def _h5py():
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - depends on extras
        raise ImportError("HDF5 datasets need: pip install 'wholebodypose[datasets]'") from e
    return h5py


def add_midpoints(keypoints: np.ndarray, safe: bool = False) -> np.ndarray:
    """Append mid-chest (133) and mid-hip (134) to ``(T, 133, 2)`` keypoints -> ``(T, 135, 2)``.

    ``safe=False`` reproduces the published data: plain mean of the two endpoints, so a missing
    endpoint (stored as ``(0, 0)``) pulls the midpoint towards the origin. ``safe=True`` writes
    ``(0, 0)`` unless both endpoints are detected.
    """
    k = np.asarray(keypoints, dtype=np.float64)
    if k.ndim != 3 or k.shape[1:] != (NUM_COCO, 2):
        raise ValueError(f"expected (T, 133, 2) keypoints, got {k.shape}")
    mids = []
    for a, b in (SHOULDERS, HIPS):
        m = (k[:, a, :] + k[:, b, :]) / 2
        if safe:
            ok = np.any(k[:, a, :] != 0, axis=-1) & np.any(k[:, b, :] != 0, axis=-1)
            m = np.where(ok[:, None], m, 0.0)
        mids.append(m[:, None, :])
    return np.concatenate([k, *mids], axis=1)


def to_published_array(keypoints_px: np.ndarray, width: int, height: int, safe_midpoints: bool = False) -> np.ndarray:
    """Pixel keypoints ``(T, 133, 2)`` in the original frame -> published ``(T, 2, 135)`` float64.

    Undetected keypoints must be ``(0, 0)``; they stay exactly 0.0 after normalization.
    """
    k = np.asarray(keypoints_px, dtype=np.float64).reshape(-1, NUM_COCO, 2).copy()
    k[:, :, 0] /= float(width)
    k[:, :, 1] /= float(height)
    k = add_midpoints(k, safe=safe_midpoints)
    return np.ascontiguousarray(np.transpose(k, (0, 2, 1)))


def scores_135(scores: np.ndarray, safe_midpoints: bool = False) -> np.ndarray:
    """``(T, 133)`` scores -> ``(T, 135)`` float32; midpoint score = min (safe) or mean of endpoints."""
    s = np.asarray(scores, dtype=np.float32).reshape(-1, NUM_COCO)
    mids = []
    for a, b in (SHOULDERS, HIPS):
        mids.append((np.minimum if safe_midpoints else lambda x, y: (x + y) / 2)(s[:, a], s[:, b])[:, None])
    return np.concatenate([s, *mids], axis=1).astype(np.float32)


def write_sample(
    h5,
    index: int | str,
    data: np.ndarray,
    label: str,
    video_name: str,
    width: int,
    height: int,
    scores: np.ndarray | None = None,
    compression: str | None = None,
) -> None:
    """Write one sample group with the published layout.

    ``data`` is the ``(T, 2, 135)`` normalized array (see :func:`to_published_array`).
    ``compression`` (e.g. ``"gzip"``) is off by default to match the published files.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3 or data.shape[1:] != (2, NUM_KEYPOINTS):
        raise ValueError(f"expected (T, 2, 135) data, got {data.shape}")
    g = h5.create_group(str(index))
    g["video_name"] = str(video_name)
    g["label"] = str(label)
    kw = {"compression": compression} if compression and len(data) else {}
    g.create_dataset("data", data=data, **kw)
    g["width"] = np.int64(width)
    g["height"] = np.int64(height)
    if scores is not None:
        s = np.asarray(scores, dtype=np.float32)
        if s.shape != (data.shape[0], NUM_KEYPOINTS):
            raise ValueError(f"expected ({data.shape[0]}, 135) scores, got {s.shape}")
        g.create_dataset("scores", data=s, **kw)


def _decode(v) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def read_group(g) -> dict:
    """Read one sample group into a dict (strings decoded, arrays loaded)."""
    out = {
        "data": g["data"][()],
        "label": _decode(g["label"][()]),
        "video_name": _decode(g["video_name"][()]),
        "width": int(g["width"][()]),
        "height": int(g["height"][()]),
    }
    if "scores" in g:
        out["scores"] = g["scores"][()]
    return out


def group_names(h5) -> list[str]:
    """Group names in numeric order ("0", "1", "2", ...)."""
    return sorted(h5.keys(), key=lambda k: (0, int(k)) if k.isdigit() else (1, k))


def iter_samples(path: str | Path, with_data: bool = True) -> Iterator[dict]:
    """Yield samples of a published/built HDF5 file in numeric group order.

    ``with_data=False`` skips the arrays and adds ``T`` (number of frames) instead.
    """
    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        for name in group_names(h5):
            g = h5[name]
            if with_data:
                d = read_group(g)
            else:
                d = {k: _decode(g[k][()]) for k in ("label", "video_name")}
                d.update(width=int(g["width"][()]), height=int(g["height"][()]), T=int(g["data"].shape[0]))
            d["index"] = int(name) if name.isdigit() else name
            yield d


def read_hdf5(path: str | Path, with_data: bool = True) -> list[dict]:
    """All samples of a file as a list of dicts (see :func:`iter_samples`)."""
    return list(iter_samples(path, with_data=with_data))


def split_filename(prefix: str, split: str, num_keypoints: int = NUM_KEYPOINTS) -> str:
    """``("WLASL100", "test") -> "WLASL100_135-Test.hdf5"``."""
    return f"{prefix}_{num_keypoints}-{SPLIT_SUFFIX[split]}.hdf5"


def detected_mask(data: np.ndarray) -> np.ndarray:
    """``(T, 2, K)`` -> ``(T, K)`` bool, True where the keypoint is not (0, 0)."""
    return np.any(np.asarray(data) != 0, axis=1)


def compare_arrays(ref: np.ndarray, ours: np.ndarray, keypoints: slice | None = slice(0, NUM_COCO)) -> dict:
    """Agreement between two ``(T, 2, K)`` arrays of the same sample.

    Returns ``T_match``, ``detection_agreement`` (fraction of (frame, keypoint) where both agree on
    detected / undetected), and mean / median / max absolute coordinate difference over keypoints
    detected in both (normalized units). Frames are compared up to the shorter sequence.
    """
    ref, ours = np.asarray(ref), np.asarray(ours)
    T = min(len(ref), len(ours))
    a, b = ref[:T], ours[:T]
    if keypoints is not None:
        a, b = a[:, :, keypoints], b[:, :, keypoints]
    da, db = detected_mask(a), detected_mask(b)
    both = da & db
    diff = np.abs(a - b).transpose(0, 2, 1)[both]  # (n, 2)
    return {
        "T_ref": int(len(ref)),
        "T_ours": int(len(ours)),
        "T_match": bool(len(ref) == len(ours)),
        "detection_agreement": float((da == db).mean()) if da.size else float("nan"),
        "n_both": int(both.sum()),
        "mean_abs_diff": float(diff.mean()) if diff.size else float("nan"),
        "median_abs_diff": float(np.median(diff)) if diff.size else float("nan"),
        "max_abs_diff": float(diff.max()) if diff.size else float("nan"),
    }
