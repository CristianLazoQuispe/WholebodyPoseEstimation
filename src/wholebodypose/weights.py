"""Model weight resolution.

WholeBodyPose wraps third-party pose models whose weights are downloaded at run time.
Upstream links do disappear, so every file used by the library is mirrored on the
Hugging Face Hub (``WEIGHTS_REPO``) together with its SHA-256. Resolution order:

1. ``$WHOLEBODYPOSE_HOME`` (default ``~/.cache/wholebodypose``) if the file is already there;
2. the Hugging Face mirror;
3. the original upstream URL (``.zip`` archives are extracted).

Every download is checked against the pinned SHA-256 when one is known.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

WEIGHTS_REPO = os.environ.get("WHOLEBODYPOSE_WEIGHTS_REPO", "CristianLazoQuispe/wholebodypose-weights")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

_MMPOSE = "https://download.openmmlab.com/mmpose/v1/projects"
_ONEPOSE = "https://github.com/developer0hye/onepose/releases/download/1.0.0"
_MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class WeightFile:
    """A single model file and where to fetch it from."""

    key: str
    filename: str  # path inside the mirror repo and the local cache
    upstream: str  # original URL (may be a .zip that contains ``Path(filename).name``)
    sha256: str | None = None
    license: str = "Apache-2.0"
    source: str = ""


def _wf(key, filename, upstream, sha256=None, source=""):
    return WeightFile(key, filename, upstream, sha256, "Apache-2.0", source)


WEIGHTS: dict[str, WeightFile] = {
    w.key: w
    for w in [
        # --- RTMW whole-body (OpenMMLab, via rtmlib) --------------------------------------
        _wf(
            "yolox_m",
            "rtmpose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
            f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            None,
            "OpenMMLab mmpose",
        ),
        _wf(
            "yolox_tiny",
            "rtmpose/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
            f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            None,
            "OpenMMLab mmpose",
        ),
        _wf(
            "rtmw_x_384",
            "rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
            f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
            None,
            "OpenMMLab mmpose",
        ),
        _wf(
            "rtmw_x_256",
            "rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
            f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
            None,
            "OpenMMLab mmpose",
        ),
        _wf(
            "rtmw_l_256",
            "rtmpose/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx",
            f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
            None,
            "OpenMMLab mmpose",
        ),
        # --- ViTPose+ whole-body (half precision, via onepose) ----------------------------
        *[
            _wf(
                f"vitpose_{size}",
                f"vitpose/splitted_vitpose+_{size}_coco_wholebody_half.pth",
                f"{_ONEPOSE}/splitted_vitpose+_{size}_coco_wholebody_half.pth",
                None,
                "ViTPose / onepose",
            )
            for size in ("small", "base", "large", "huge")
        ],
        # --- MediaPipe Tasks models --------------------------------------------------------
        _wf(
            "mp_pose_heavy",
            "mediapipe/pose_landmarker_heavy.task",
            f"{_MEDIAPIPE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            None,
            "Google MediaPipe",
        ),
        _wf(
            "mp_pose_full",
            "mediapipe/pose_landmarker_full.task",
            f"{_MEDIAPIPE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            None,
            "Google MediaPipe",
        ),
        _wf(
            "mp_hand",
            "mediapipe/hand_landmarker.task",
            f"{_MEDIAPIPE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            None,
            "Google MediaPipe",
        ),
        _wf(
            "mp_face",
            "mediapipe/face_landmarker.task",
            f"{_MEDIAPIPE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            None,
            "Google MediaPipe",
        ),
    ]
}


def cache_dir() -> Path:
    """Local folder where weights are stored."""
    root = os.environ.get("WHOLEBODYPOSE_HOME") or Path.home() / ".cache" / "wholebodypose"
    return Path(root)


def sha256sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dst: Path, verbose: bool) -> None:
    if verbose:
        print(f"[wholebodypose] downloading {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".part")
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wholebodypose"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _extract_member(archive: Path, member_name: str, dst: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        matches = [n for n in zf.namelist() if Path(n).name == member_name]
        if not matches:  # fall back to the only .onnx inside
            matches = [n for n in zf.namelist() if n.endswith(Path(member_name).suffix)]
        if not matches:
            raise FileNotFoundError(f"{member_name} not found in {archive}")
        with zf.open(matches[0]) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)


def mirror_url(w: WeightFile) -> str:
    return f"{HF_ENDPOINT}/{WEIGHTS_REPO}/resolve/main/{w.filename}"


def get_weight(key: str, verbose: bool = True, offline: bool | None = None) -> Path:
    """Return a local path to weight ``key``, downloading it if needed."""
    if key not in WEIGHTS:
        raise KeyError(f"Unknown weight '{key}'. Available: {sorted(WEIGHTS)}")
    w = WEIGHTS[key]
    dst = cache_dir() / w.filename
    if offline is None:
        offline = os.environ.get("WHOLEBODYPOSE_OFFLINE", "0") == "1"

    if dst.exists():
        return dst
    if offline:
        raise FileNotFoundError(f"{dst} missing and WHOLEBODYPOSE_OFFLINE=1")

    errors = []
    for url in (mirror_url(w), w.upstream):
        try:
            if url.endswith(".zip"):
                with tempfile.TemporaryDirectory() as td:
                    archive = Path(td) / "w.zip"
                    _download(url, archive, verbose)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    _extract_member(archive, Path(w.filename).name, dst)
            else:
                _download(url, dst, verbose)
        except Exception as e:  # noqa: BLE001 - try the next source
            errors.append(f"{url}: {e}")
            continue
        if w.sha256 and sha256sum(dst) != w.sha256:
            dst.unlink(missing_ok=True)
            errors.append(f"{url}: sha256 mismatch")
            continue
        return dst
    raise RuntimeError(f"Could not fetch weight '{key}':\n  " + "\n  ".join(errors))
