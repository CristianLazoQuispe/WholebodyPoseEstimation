"""RTMPose / RTMW whole-body backend (OpenMMLab models through ``rtmlib``)."""

from __future__ import annotations

import contextlib
import io
import os

import cv2
import numpy as np

from wholebodypose.models.base import BasePoseModel
from wholebodypose.weights import get_weight

# RTMW SimCC confidences are not calibrated: the 384x288 model scores ~5-8 on visible joints while
# the 256x192 models score ~1. Default thresholds keep roughly the same operating point.
DEFAULT_KPT_THR = {"performance": 2.5, "balanced": 0.35, "lightweight": 0.35}

# (detector weight, pose weight, detector input size, pose input size) per rtmlib mode
MODES = {
    "performance": ("yolox_m", "rtmw_x_384", (640, 640), (288, 384)),
    "balanced": ("yolox_m", "rtmw_x_256", (640, 640), (192, 256)),
    "lightweight": ("yolox_tiny", "rtmw_l_256", (416, 416), (192, 256)),
}

FOOT_LEFT_IDS = [11, 13, 15]
FOOT_RIGHT_IDS = [12, 14, 16]
HAND_RIGHT_IDS = [6, 8, 10, 112]
HAND_LEFT_IDS = [5, 7, 9, 91]


def _ort_has_cuda() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


class RTMPoseModel(BasePoseModel):
    """Multi-person whole-body pose with YOLOX + RTMW (COCO-WholeBody, 133 keypoints).

    Args:
        device: ``'auto'``, ``'cuda'`` or ``'cpu'``. Falls back to CPU when onnxruntime has no CUDA.
        backend: rtmlib inference backend (``'onnxruntime'``, ``'opencv'``, ``'openvino'``).
        mode: ``'performance'`` (paper setting), ``'balanced'`` or ``'lightweight'``.
        to_openpose: Return rtmlib's OpenPose-style ordering instead of COCO-WholeBody.
        use_thresholding: Binarize scores with ``kpt_thr`` (see :class:`BasePoseModel`).
        kpt_thr: Threshold on RTMW SimCC confidences; ``None`` uses ``DEFAULT_KPT_THR[mode]``
            (paper, performance mode: ``2.5``).
        filter_noise: Drop feet/hands whose anchoring limb joints are not confidently detected.
        verbose: Print loading information.
    """

    name = "rtmpose"

    def __init__(
        self,
        device: str = "auto",
        backend: str = "onnxruntime",
        mode: str = "performance",
        to_openpose: bool = False,
        use_thresholding: bool = False,
        kpt_thr: float | None = None,
        filter_noise: bool = False,
        verbose: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        kpt_thr = DEFAULT_KPT_THR[mode] if kpt_thr is None else kpt_thr
        super().__init__(use_thresholding=use_thresholding, kpt_thr=kpt_thr)
        try:
            from rtmlib import Wholebody
        except ImportError as e:  # pragma: no cover - depends on extras
            raise ImportError("RTMPose backend needs: pip install 'wholebodypose[rtmpose]'") from e

        if device in ("auto", "cuda"):
            device = "cuda" if _ort_has_cuda() else "cpu"
        self.device, self.backend, self.mode = device, backend, mode
        self.to_openpose = to_openpose
        self.filter_noise = filter_noise
        self.verbose = verbose

        if not verbose:
            os.environ.setdefault("ORT_LOG_LEVEL", "3")
            with contextlib.suppress(ImportError):
                import onnxruntime as ort

                ort.set_default_logger_severity(3)

        det_key, pose_key, det_size, pose_size = MODES[mode]
        det_path = str(get_weight(det_key, verbose=verbose))
        pose_path = str(get_weight(pose_key, verbose=verbose))

        sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with sink:
            self.model = Wholebody(
                det=det_path,
                det_input_size=det_size,
                pose=pose_path,
                pose_input_size=pose_size,
                to_openpose=to_openpose,
                backend=backend,
                device=device,
            )

    def _predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # rtmlib (like OpenMMLab) expects BGR input.
        keypoints, scores = self.model(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if keypoints.size == 0:
            return np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)
        if self.filter_noise:
            scores = self.filter_scores(scores)
        return keypoints, scores

    def _filter_part(self, scores: np.ndarray, anchor_ids, part_ids, extra_ids) -> None:
        for person in scores:  # in-place on each (133,) row
            if np.sum(person[anchor_ids] > self.kpt_thr) != len(anchor_ids):
                person[list(part_ids)] = 0
                person[extra_ids] = 0

    def filter_scores(self, scores: np.ndarray) -> np.ndarray:
        """Zero feet/hands whose anchoring joints (hip-knee-ankle, shoulder-elbow-wrist) are missing."""
        scores = scores.copy()
        self._filter_part(scores, FOOT_LEFT_IDS, range(17, 20), [15])
        self._filter_part(scores, FOOT_RIGHT_IDS, range(20, 23), [16])
        self._filter_part(scores, HAND_RIGHT_IDS, range(112, 133), [10])
        self._filter_part(scores, HAND_LEFT_IDS, range(91, 112), [9])
        return scores
