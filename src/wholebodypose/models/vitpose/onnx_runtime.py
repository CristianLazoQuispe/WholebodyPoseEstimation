"""ViTPose+ on ONNX Runtime, with NumPy/OpenCV pre- and post-processing (no PyTorch, no onepose).

This reproduces onepose's ``Model.forward`` for the COCO-WholeBody configs:

* box -> (center, scale) with the 192:256 aspect ratio and no padding (``onepose.transforms._box2cs``);
* affine warp to 192x256 (``cv2.warpAffine``, bilinear, no UDP), RGB, ``/255``, ImageNet mean/std;
* heatmap argmax + the +/-0.25 px "default" shift towards the higher neighbour, then the inverse
  transform (``onepose.functional.keypoints_from_heatmaps`` with ``post_process='default'``,
  ``use_udp=False``). onepose does not run flip test, so neither does this module.

The graph itself (``scripts/export_vitpose_onnx.py``) maps ``(N, 3, 256, 192)`` to ``(N, 133, 64, 48)``.
"""

from __future__ import annotations

import cv2
import numpy as np

INPUT_W, INPUT_H = 192, 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def box_to_center_scale(x: float, y: float, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y, w, h)`` box -> center and scale (in units of 200 px), expanded to the input aspect ratio."""
    aspect = INPUT_W / INPUT_H
    center = np.array([x + w * 0.5, y + h * 0.5], np.float32)
    if w > aspect * h:
        h = w * 1.0 / aspect
    elif w < aspect * h:
        w = h * aspect
    return center, np.array([w / 200.0, h / 200.0], np.float32)


def affine_transform(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """2x3 matrix mapping the box to the ``INPUT_W x INPUT_H`` crop (rotation 0, as in onepose)."""
    src_w = scale[0] * 200.0
    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, src_w * -0.5], np.float32)
    dst[0] = [INPUT_W * 0.5, INPUT_H * 0.5]
    dst[1] = dst[0] + np.array([0.0, INPUT_W * -0.5], np.float32)
    for pts in (src, dst):  # third point: (p0 - p1) rotated by 90 degrees around p1
        d = pts[0] - pts[1]
        pts[2] = pts[1] + np.array([-d[1], d[0]], np.float32)
    return cv2.getAffineTransform(src, dst)


def preprocess(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole image as the person box -> ``(3, 256, 192)`` float32 tensor, center, scale."""
    h, w = image_rgb.shape[:2]
    center, scale = box_to_center_scale(0, 0, w, h)
    crop = cv2.warpAffine(image_rgb, affine_transform(center, scale), (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)), center, scale


def decode_heatmaps(heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(N, K, H, W)`` heatmaps -> keypoints ``(N, K, 2)`` in image pixels and scores ``(N, K)``."""
    n, k, hh, ww = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    idx = np.argmax(flat, axis=2)
    maxvals = np.take_along_axis(flat, idx[..., None], axis=2)[..., 0]
    preds = np.stack([idx % ww, idx // ww], axis=-1).astype(np.float32)
    preds[maxvals <= 0] = -1

    px, py = preds[..., 0].astype(int), preds[..., 1].astype(int)
    inside = (px > 1) & (px < ww - 1) & (py > 1) & (py < hh - 1)
    pxc, pyc = np.clip(px, 1, ww - 2), np.clip(py, 1, hh - 2)
    ni, ki = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
    dx = heatmaps[ni, ki, pyc, pxc + 1] - heatmaps[ni, ki, pyc, pxc - 1]
    dy = heatmaps[ni, ki, pyc + 1, pxc] - heatmaps[ni, ki, pyc - 1, pxc]
    shift = np.stack([np.sign(dx), np.sign(dy)], axis=-1) * 0.25
    preds = preds + np.where(inside[..., None], shift, 0).astype(np.float32)

    size = np.asarray(scales, np.float32).reshape(n, 1, 2) * 200.0
    center = np.asarray(centers, np.float32).reshape(n, 1, 2)
    keypoints = preds * (size / np.array([ww, hh], np.float32)) + center - size * 0.5
    return keypoints.astype(np.float32), maxvals.astype(np.float32)


def ort_providers(device: str) -> list[str]:
    """ONNX Runtime providers for ``device`` (``'auto'``/``'cuda'`` use CUDA when it is available)."""
    import onnxruntime as ort

    if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class ViTPoseOnnx:
    """A ViTPose+ ONNX graph plus its NumPy pre/post-processing.

    Args:
        model_path: ``.onnx`` file exported by ``scripts/export_vitpose_onnx.py``.
        device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
        intra_op_num_threads: ONNX Runtime CPU threads (``0`` = its default).
    """

    def __init__(self, model_path: str, device: str = "auto", intra_op_num_threads: int = 0) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_num_threads
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=ort_providers(device))
        self.input_name = self.session.get_inputs()[0].name
        self.device = "cuda" if self.session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"

    def __call__(self, images_rgb: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Pose of the person filling each image -> keypoints ``(N, 133, 2)``, scores ``(N, 133)``."""
        if not images_rgb:
            return np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32)
        tensors, centers, scales = zip(*(preprocess(im) for im in images_rgb), strict=True)
        heatmaps = self.session.run(None, {self.input_name: np.stack(tensors)})[0]
        return decode_heatmaps(np.asarray(heatmaps, np.float32), np.stack(centers), np.stack(scales))
