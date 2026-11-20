"""ViTPose+ whole-body backend (through ``onepose``)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from wholebodypose.models.base import BasePoseModel, resolve_device
from wholebodypose.weights import get_weight

MODEL_NAMES = {
    "ViTPose+_small_coco_wholebody": "vitpose_small",
    "ViTPose+_base_coco_wholebody": "vitpose_base",
    "ViTPose+_large_coco_wholebody": "vitpose_large",
    "ViTPose+_huge_coco_wholebody": "vitpose_huge",
}


def _prefetch_onepose_weights(model_name: str, verbose: bool) -> None:
    """Place the checkpoint where onepose expects it, using the WholeBodyPose mirror."""
    import onepose.models.factory as factory

    cfg = factory.model_config[model_name]
    target = Path(factory.__file__).parent / "weights" / cfg["url"].split("/")[-1]
    if target.exists():
        return
    try:
        src = get_weight(MODEL_NAMES[model_name], verbose=verbose)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    except Exception:  # noqa: BLE001 - onepose will download from upstream itself
        pass


class VITPoseModel(BasePoseModel):
    """Top-down ViTPose+ (COCO-WholeBody, 133 keypoints).

    By default (paper setting) the whole frame is treated as one person box. With
    ``detector='yolox_m'`` (or ``'yolox_tiny'``) people are first detected with YOLOX (needs the
    ``[rtmpose]`` extra) and ViTPose runs on each person crop, giving multi-person output.

    Args:
        device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
        model_name: One of ``MODEL_NAMES`` (paper: ``ViTPose+_huge_coco_wholebody``).
        use_thresholding: Binarize scores with ``kpt_thr``.
        kpt_thr: Threshold on heatmap confidences (0-1).
        verbose: Print loading information.
        detector: ``None`` (full frame) or ``'yolox_m'`` / ``'yolox_tiny'``.
        max_people: Keep at most this many detections (largest boxes first); ``None`` = all.
        det_score_thr: YOLOX score threshold.
        box_padding: Relative padding added around each detected box before cropping.
    """

    name = "vitpose"

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "ViTPose+_huge_coco_wholebody",
        use_thresholding: bool = False,
        kpt_thr: float = 0.5,
        verbose: bool = True,
        detector: str | None = None,
        max_people: int | None = None,
        det_score_thr: float = 0.5,
        box_padding: float = 0.25,
    ) -> None:
        super().__init__(use_thresholding=use_thresholding, kpt_thr=kpt_thr)
        try:
            import onepose
        except ImportError as e:  # pragma: no cover - depends on extras
            raise ImportError("ViTPose backend needs: pip install 'wholebodypose[vitpose]'") from e
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {list(MODEL_NAMES)}, got {model_name!r}")

        self.device = resolve_device(device, framework="torch")
        self.model_name = model_name
        _prefetch_onepose_weights(model_name, verbose)
        self.model = onepose.create_model(model_name=model_name).to(self.device)

        self.max_people = max_people
        self.box_padding = box_padding
        self.detector = None
        if detector is not None:
            self.detector = _load_yolox(detector, self.device, det_score_thr, verbose)

    def _pose_on(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.model(image_bgr)
        return np.asarray(out["points"], np.float32), np.asarray(out["confidence"], np.float32).reshape(-1)

    def _predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # onepose applies its own BGR->RGB transform, so feed it BGR.
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if self.detector is None:
            k, s = self._pose_on(image_bgr)
            return k[None], s[None]

        h, w = image_bgr.shape[:2]
        boxes = np.asarray(self.detector(image_bgr), np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes[np.argsort(-areas)][: self.max_people]
        kpts, scores = [], []
        for x0, y0, x1, y1 in boxes:
            pw, ph = (x1 - x0) * self.box_padding / 2, (y1 - y0) * self.box_padding / 2
            cx0, cy0 = int(max(x0 - pw, 0)), int(max(y0 - ph, 0))
            cx1, cy1 = int(min(x1 + pw, w)), int(min(y1 + ph, h))
            if cx1 - cx0 < 8 or cy1 - cy0 < 8:
                continue
            k, s = self._pose_on(np.ascontiguousarray(image_bgr[cy0:cy1, cx0:cx1]))
            kpts.append(k + [cx0, cy0])
            scores.append(s)
        if not kpts:
            return np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)
        return np.stack(kpts).astype(np.float32), np.stack(scores).astype(np.float32)


DETECTORS = {"yolox_m": ("yolox_m", (640, 640)), "yolox_tiny": ("yolox_tiny", (416, 416))}


def _load_yolox(name: str, device: str, score_thr: float, verbose: bool):
    """rtmlib YOLOX person detector with WholeBodyPose-managed weights."""
    if name not in DETECTORS:
        raise ValueError(f"detector must be one of {list(DETECTORS)} or None, got {name!r}")
    try:
        from rtmlib import YOLOX
    except ImportError as e:  # pragma: no cover - depends on extras
        raise ImportError("The ViTPose person detector needs: pip install 'wholebodypose[rtmpose]'") from e
    key, size = DETECTORS[name]
    det = YOLOX(
        onnx_model=str(get_weight(key, verbose=verbose)),
        model_input_size=size,
        backend="onnxruntime",
        device="cuda" if device == "cuda" else "cpu",
    )
    det.score_thr = score_thr
    return det
