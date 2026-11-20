# Vendored from rtmlib 0.0.16 (https://github.com/Tau-J/rtmlib), files
# rtmlib/tools/pose_estimation/rtmpose.py, pre_processings.py and post_processings.py.
#
# rtmlib: Copyright 2018-2020 Open-MMLab (as stated in its LICENSE); author Tao Jiang (Tau-J) and contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy is in LICENSES/Apache-2.0.txt and at
# http://www.apache.org/licenses/LICENSE-2.0. Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# Modified by the WholeBodyPose authors: the session comes from wholebodypose.models.onnx_session
# (ONNX Runtime only; rtmlib's OpenCV/OpenVINO backends and checkpoint download are dropped), only the
# 2D SimCC code used by RTMW is kept (no DARK/UDP, no 3D), the rotation helper is folded into
# get_warp_matrix (rotation is always 0 here) and the code is reformatted. The arithmetic and dtypes
# (float64 normalization, SimCC argmax, inverse transform) are unchanged, so keypoints and scores are
# bit-identical to rtmlib's.
"""RTMPose / RTMW top-down pose (SimCC heads) on ONNX Runtime + NumPy, as in rtmlib."""

from __future__ import annotations

import cv2
import numpy as np

from wholebodypose.models.onnx_session import create_session, session_device

# ImageNet mean/std in 0-255 units (MMPose RTMPose configs)
RTMPOSE_MEAN = (123.675, 116.28, 103.53)
RTMPOSE_STD = (58.395, 57.12, 57.375)


# ------------------------------------------------------------------------------- pre-processing
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(x1, y1, x2, y2)`` box(es) -> center ``(x, y)`` and scale ``(w, h)`` multiplied by ``padding``."""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotate ``a - b`` by 90 degrees anticlockwise around ``b``."""
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: tuple[int, int],
    shift: tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    """2x3 affine matrix that maps the box ``(center, scale)`` to ``output_size`` (``inv``: the reverse)."""
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = np.array([[cs, -sn], [sn, cs]]) @ np.array([0.0, src_w * -0.5])
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(
    input_size: tuple[int, int], bbox_scale: np.ndarray, bbox_center: np.ndarray, img: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop the box to ``input_size`` ``(w, h)``; the box is first expanded to that aspect ratio.

    Returns the warped image and the expanded box scale.
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    aspect_ratio = w / h
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        b_w > b_h * aspect_ratio, np.hstack([b_w, b_w / aspect_ratio]), np.hstack([b_h * aspect_ratio, b_h])
    )

    warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0, output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img, bbox_scale


# ------------------------------------------------------------------------------ post-processing
def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SimCC ``(N, K, Wx)``, ``(N, K, Wy)`` -> argmax locations ``(N, K, 2)`` and scores ``(N, K)``.

    The score is the mean of the x and y maxima; locations with score ``<= 0`` are set to ``-1``.
    """
    N, K, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.0] = -1

    return locs.reshape(N, K, 2), vals.reshape(N, K)


def convert_coco_to_openpose(keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COCO-WholeBody (133) -> OpenPose-style ordering with an added neck (134)."""
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:3] = np.where(
        keypoints_info[:, 5, 2:3] > keypoints_info[:, 6, 2:3], keypoints_info[:, 6, 2:3], keypoints_info[:, 5, 2:3]
    )
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    return keypoints_info[..., :2], keypoints_info[..., 2]


# --------------------------------------------------------------------------------------- model
class RTMPose:
    """Top-down RTMPose/RTMW on ONNX Runtime.

    ``__call__(bgr_image, bboxes)`` runs the model once per xyxy box (the whole image when ``bboxes`` is
    empty) and returns keypoints ``(N, K, 2)`` (float64) and scores ``(N, K)`` (float32).

    Args:
        onnx_model: Path to the ``.onnx`` file.
        model_input_size: ``(width, height)`` of the network input (rtmlib convention).
        mean, std: Per-channel normalization in 0-255 units (``None`` disables it).
        to_openpose: Return OpenPose-style ordering (134 points) instead of COCO-WholeBody.
        device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
        intra_op_num_threads: onnxruntime CPU threads (``0`` = its default).
    """

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple[int, int] = (192, 256),
        mean: tuple[float, float, float] | None = RTMPOSE_MEAN,
        std: tuple[float, float, float] | None = RTMPOSE_STD,
        to_openpose: bool = False,
        device: str = "cpu",
        intra_op_num_threads: int = 0,
    ) -> None:
        self.onnx_model = str(onnx_model)
        self.model_input_size = tuple(model_input_size)
        self.mean = None if mean is None else np.array(mean)
        self.std = None if std is None else np.array(std)
        self.to_openpose = to_openpose
        self.session = create_session(self.onnx_model, device, intra_op_num_threads)
        self.device = session_device(self.session)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, image: np.ndarray, bboxes=()) -> tuple[np.ndarray, np.ndarray]:
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img)
            kpts, score = self.postprocess(outputs, center, scale)
            keypoints.append(kpts)
            scores.append(score)

        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)
        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)
        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Box (padded by 1.25) -> normalized crop (float64 ``(h, w, 3)``), center, expanded scale."""
        center, scale = bbox_xyxy2cs(np.array(bbox), padding=1.25)
        resized_img, scale = top_down_affine(self.model_input_size, scale, center, img)
        if self.mean is not None:
            resized_img = (resized_img - self.mean) / self.std
        return resized_img, center, scale

    def inference(self, img: np.ndarray) -> list[np.ndarray]:
        """``(H, W, 3)`` crop -> ``[simcc_x, simcc_y]`` (``float32`` NCHW input, batch 1)."""
        x = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]
        return self.session.run(self.output_names, {self.input_name: x})

    def postprocess(
        self, outputs: list[np.ndarray], center: np.ndarray, scale: np.ndarray, simcc_split_ratio: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """SimCC decode, then map from the crop back to image pixels."""
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2
        return keypoints, scores
