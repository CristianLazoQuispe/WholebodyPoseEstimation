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
    """Single-person top-down ViTPose+ on the full frame (COCO-WholeBody, 133 keypoints).

    Args:
        device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
        model_name: One of ``MODEL_NAMES`` (paper: ``ViTPose+_huge_coco_wholebody``).
        use_thresholding: Binarize scores with ``kpt_thr``.
        kpt_thr: Threshold on heatmap confidences (0-1).
        verbose: Print loading information.
    """

    name = "vitpose"

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "ViTPose+_huge_coco_wholebody",
        use_thresholding: bool = False,
        kpt_thr: float = 0.5,
        verbose: bool = True,
    ) -> None:
        super().__init__(use_thresholding=use_thresholding, kpt_thr=kpt_thr)
        try:
            import onepose
        except ImportError as e:  # pragma: no cover - depends on extras
            raise ImportError("ViTPose backend needs: pip install 'wholebodypose[vitpose]'") from e
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {list(MODEL_NAMES)}, got {model_name!r}")

        self.device = resolve_device(device)
        self.model_name = model_name
        _prefetch_onepose_weights(model_name, verbose)
        self.model = onepose.create_model(model_name=model_name).to(self.device)

    def _predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # onepose applies its own BGR->RGB transform, so feed it BGR.
        out = self.model(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        keypoints = np.asarray(out["points"], dtype=np.float32)[None]
        scores = np.asarray(out["confidence"], dtype=np.float32).reshape(1, -1)
        return keypoints, scores
