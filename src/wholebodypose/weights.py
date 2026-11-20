"""Model weight resolution.

WholeBodyPose wraps third-party pose models whose weights are downloaded at run time.
Upstream links do disappear, so every file used by the library is mirrored on the
Hugging Face Hub (``WEIGHTS_REPO``) together with its SHA-256. Resolution order:

1. ``$WHOLEBODYPOSE_HOME`` (default ``~/.cache/wholebodypose``) if the file is already there;
2. the Hugging Face mirror;
3. the original upstream URL (``.zip`` archives are extracted), when there is one. Files produced by
   WholeBodyPose itself (the ViTPose ONNX exports) have no upstream and come only from the mirror.

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
_MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class WeightFile:
    """A single model file and where to fetch it from."""

    key: str
    filename: str  # path inside the mirror repo and the local cache
    upstream: str | None  # original URL (may be a .zip with ``Path(filename).name``); None = mirror only
    sha256: str | None = None
    license: str = "Apache-2.0"
    source: str = ""


def _wf(key, filename, upstream, sha256=None, source=""):
    return WeightFile(key, filename, upstream, sha256, "Apache-2.0", source)


# SHA-256 of the ViTPose+ ONNX exports of the onepose checkpoints (opset 17, dynamic
# batch, float32 compute, float16-stored weights). They exist only on the WholeBodyPose mirror.
_VITPOSE_ONNX_SHA256 = {
    "small": "2cc364025711805eeb0d4a613eaf4cebdb3ef69623c06bfc3fb494ed8816786b",
    "base": "61cb598570d01f4575b90e516332c2473a81ca0b17e94fa6ac9a66ebe07fdabe",
    "large": "64c250be2730cae07c4bb59a29752856cd1410c4e1f5770211de377b8e7bfe73",
    "huge": "b8605dfbc33ff292d4e05e922cf175a2387facc9cc93da87ca3304bdd6544718",
}


WEIGHTS: dict[str, WeightFile] = {
    w.key: w
    for w in [
        # --- YOLOX + RTMW whole-body (OpenMMLab) --------------------------------------------
        _wf(
            "yolox_m",
            "rtmpose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
            f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23",
            "OpenMMLab mmpose",
        ),
        _wf(
            "yolox_tiny",
            "rtmpose/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
            f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
            "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
            "OpenMMLab mmpose",
        ),
        _wf(
            "rtmw_x_384",
            "rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
            f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
            "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054",
            "OpenMMLab mmpose",
        ),
        _wf(
            "rtmw_x_256",
            "rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
            f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
            "9a9bc17c13ff0a1e37507f45a037a21bc410cd08374b0e664c57d0080843aa58",
            "OpenMMLab mmpose",
        ),
        _wf(
            "rtmw_l_256",
            "rtmpose/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx",
            f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
            "68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76",
            "OpenMMLab mmpose",
        ),
        # --- ViTPose+ whole-body exported to ONNX by WholeBodyPose (mirror only) ----------
        *[
            _wf(
                f"vitpose_{size}",
                f"vitpose/vitpose+_{size}_coco_wholebody.onnx",
                None,
                sha,
                "ViTPose / onepose, ONNX export by WholeBodyPose",
            )
            for size, sha in _VITPOSE_ONNX_SHA256.items()
        ],
        # --- MediaPipe Tasks models --------------------------------------------------------
        _wf(
            "mp_pose_heavy",
            "mediapipe/pose_landmarker_heavy.task",
            f"{_MEDIAPIPE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
            "Google MediaPipe",
        ),
        _wf(
            "mp_pose_full",
            "mediapipe/pose_landmarker_full.task",
            f"{_MEDIAPIPE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
            "Google MediaPipe",
        ),
        _wf(
            "mp_hand",
            "mediapipe/hand_landmarker.task",
            f"{_MEDIAPIPE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
            "Google MediaPipe",
        ),
        _wf(
            "mp_face",
            "mediapipe/face_landmarker.task",
            f"{_MEDIAPIPE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
            "Google MediaPipe",
        ),
    ]
}


def cache_dir() -> Path:
    """Local folder where weights are stored."""
    root = os.environ.get("WHOLEBODYPOSE_HOME") or Path.home() / ".cache" / "wholebodypose"
    return Path(root).expanduser()


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
    for url in (u for u in (mirror_url(w), w.upstream) if u):
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
