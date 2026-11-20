"""RTMPose / RTMW whole-body backend (OpenMMLab YOLOX + RTMW ONNX models) on ONNX Runtime.

Pre- and post-processing are vendored from rtmlib (:mod:`wholebodypose.models.rtmpose._onnx`).
"""

from __future__ import annotations

import contextlib
import os

import cv2
import numpy as np

from wholebodypose.models.base import BasePoseModel
from wholebodypose.models.onnx_session import ort_has_cuda
from wholebodypose.models.rtmpose._onnx.wholebody import MODE as _MODE_TABLE
from wholebodypose.weights import get_weight

# RTMW SimCC confidences are not calibrated: the 384x288 model scores ~5-8 on visible joints while
# the 256x192 models score ~1. Default thresholds keep roughly the same operating point.
DEFAULT_KPT_THR = {"performance": 2.5, "balanced": 0.35, "lightweight": 0.35}

# (detector weight, pose weight, detector input size, pose input size) per mode.
# performance = YOLOX-m + RTMW-l 384x288, balanced = YOLOX-m + RTMW-l 256x192,
# lightweight = YOLOX-tiny + RTMW-m 256x192.
MODES = {
    name: (m["det_key"], m["pose_key"], m["det_input_size"], m["pose_input_size"]) for name, m in _MODE_TABLE.items()
}

FOOT_LEFT_IDS = [11, 13, 15]
FOOT_RIGHT_IDS = [12, 14, 16]
HAND_RIGHT_IDS = [6, 8, 10, 112]
HAND_LEFT_IDS = [5, 7, 9, 91]


class RTMPoseModel(BasePoseModel):
    """Multi-person whole-body pose with YOLOX + RTMW (COCO-WholeBody, 133 keypoints).

    Args:
        device: ``'auto'``, ``'cuda'`` or ``'cpu'``. Falls back to CPU when onnxruntime has no CUDA.
        mode: ``'performance'`` (paper setting), ``'balanced'`` or ``'lightweight'``.
        to_openpose: Return the OpenPose-style ordering instead of COCO-WholeBody.
        use_thresholding: Binarize scores with ``kpt_thr`` (see :class:`BasePoseModel`).
        kpt_thr: Threshold on RTMW SimCC confidences; ``None`` uses ``DEFAULT_KPT_THR[mode]``
            (paper, performance mode: ``2.5``).
        filter_noise: Drop feet/hands whose anchoring limb joints are not confidently detected.
        verbose: Print loading information.
        legacy_rgb_input: Feed the RGB image to the models, which expect BGR (red and blue swapped),
            as the ICMV 2025 pipeline did. Off by default; only needed to regenerate the ICMV 2025 ISLR
            keypoints bit-for-bit (see ``docs/datasets``).
        intra_op_num_threads: ONNX Runtime CPU threads per session (``0`` = onnxruntime's default).
    """

    name = "rtmpose"

    def __init__(
        self,
        device: str = "auto",
        mode: str = "performance",
        to_openpose: bool = False,
        use_thresholding: bool = False,
        kpt_thr: float | None = None,
        filter_noise: bool = False,
        verbose: bool = True,
        legacy_rgb_input: bool = False,
        intra_op_num_threads: int = 0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        kpt_thr = DEFAULT_KPT_THR[mode] if kpt_thr is None else kpt_thr
        super().__init__(use_thresholding=use_thresholding, kpt_thr=kpt_thr)

        if device in ("auto", "cuda"):
            device = "cuda" if ort_has_cuda() else "cpu"
        self.device, self.mode = device, mode
        self.to_openpose = to_openpose
        self.filter_noise = filter_noise
        self.verbose = verbose
        self.legacy_rgb_input = legacy_rgb_input

        if not verbose:
            os.environ.setdefault("ORT_LOG_LEVEL", "3")
            with contextlib.suppress(ImportError):
                import onnxruntime as ort

                ort.set_default_logger_severity(3)

        det_key, pose_key, det_size, pose_size = MODES[mode]
        det_path = str(get_weight(det_key, verbose=verbose))
        pose_path = str(get_weight(pose_key, verbose=verbose))

        from wholebodypose.models.rtmpose._onnx import Wholebody

        self.model = Wholebody(
            det=det_path,
            pose=pose_path,
            det_input_size=det_size,
            pose_input_size=pose_size,
            to_openpose=to_openpose,
            device=device,
            intra_op_num_threads=intra_op_num_threads,
        )

    def _predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # RTMW and YOLOX (like OpenMMLab) expect BGR input; legacy_rgb_input feeds RGB as in ICMV 2025.
        image = image_rgb if self.legacy_rgb_input else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        keypoints, scores = self.model(image)
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
