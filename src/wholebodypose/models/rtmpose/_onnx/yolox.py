# Vendored from rtmlib 0.0.16 (https://github.com/Tau-J/rtmlib), files
# rtmlib/tools/object_detection/yolox.py and rtmlib/tools/object_detection/post_processings.py.
# rtmlib's YOLOX code is itself "modified from https://github.com/IDEA-Research/DWPose" (cv_ox_det.py).
#
# rtmlib: Copyright 2018-2020 Open-MMLab (as stated in its LICENSE); author Tao Jiang (Tau-J) and contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy is in LICENSES/Apache-2.0.txt and at
# http://www.apache.org/licenses/LICENSE-2.0. Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# Modified by the WholeBodyPose authors: the session comes from wholebodypose.models.onnx_session
# (ONNX Runtime only; rtmlib's OpenCV/OpenVINO backends and checkpoint download are dropped), the
# deprecated ``mode`` alias is removed, and the code is reformatted. The arithmetic (resize and
# padding, grid decoding, NumPy NMS) is unchanged so that boxes are bit-identical to rtmlib's.
"""YOLOX person detector (ONNX Runtime + NumPy), as in rtmlib."""

from __future__ import annotations

import cv2
import numpy as np

from wholebodypose.models.onnx_session import create_session, session_device


def nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list[int]:
    """Single class NMS implemented in NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float):
    """Class-aware NMS in NumPy.

    Args:
        boxes: ``(N, 4)`` boxes, xyxy.
        scores: ``(N, num_classes)`` per-class scores.
        nms_thr: IoU threshold.
        score_thr: Boxes with score ``<= score_thr`` are dropped before NMS.

    Returns:
        ``(dets, keep)``: ``dets`` ``(M, 6)`` rows ``(x1, y1, x2, y2, score, cls)`` and ``keep``, the
        indices of those rows in ``boxes``; ``(None, None)`` when nothing survives.
    """
    final_dets = []
    final_keep = []
    num_classes = scores.shape[1]
    all_indices = np.arange(boxes.shape[0])
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_indices = all_indices[valid_score_mask]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls_ind
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
            final_dets.append(dets)
            final_keep.append(valid_indices[keep])
    if len(final_dets) == 0:
        return None, None
    return np.concatenate(final_dets, 0), np.concatenate(final_keep, 0)


def yolox_preprocess(img: np.ndarray, model_input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Letterbox ``img`` (BGR ``uint8``) into ``model_input_size`` with gray (114) padding.

    Returns the padded ``uint8`` image and the resize ratio.
    """
    if img.shape[:2] == tuple(model_input_size[:2]):
        return img.copy(), 1.0
    if len(img.shape) == 3:
        padded_img = np.ones((model_input_size[0], model_input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(model_input_size, dtype=np.uint8) * 114

    ratio = min(model_input_size[0] / img.shape[0], model_input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
    padded_img[: padded_shape[0], : padded_shape[1]] = resized_img
    return padded_img, ratio


def yolox_postprocess(
    outputs: np.ndarray,
    ratio: float,
    model_input_size: tuple[int, int],
    nms_thr: float,
    score_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw YOLOX outputs ``(1, A, 5 + C)`` (or ``(1, A, 5)`` with NMS in the graph).

    Returns ``(boxes, class_ids)``: xyxy boxes in input-image pixels and their class indices.
    ``outputs`` is modified in place, as in rtmlib.
    """
    final_cls_inds = np.array([])
    if outputs.shape[-1] == 4 or outputs.shape[-1] > 5:
        # graph without NMS: decode the grid, then NumPy NMS
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]

        hsizes = [model_input_size[0] // stride for stride in strides]
        wsizes = [model_input_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides, strict=True):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        predictions = outputs[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets, _ = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)
        if dets is not None:
            final_boxes = dets[:, :4]
            final_cls_inds = dets[:, 5].astype(int)
        else:
            final_boxes = np.array([])
    elif outputs.shape[-1] == 5:
        # graph with NMS: (x1, y1, x2, y2, score)
        final_boxes, final_scores = outputs[0, :, :4], outputs[0, :, 4]
        final_boxes /= ratio
        final_boxes = final_boxes[final_scores > 0.3]
    else:
        raise ValueError(f"unexpected YOLOX output shape {outputs.shape}")
    return final_boxes, final_cls_inds


class YOLOX:
    """YOLOX detector on ONNX Runtime; ``__call__(bgr_image)`` returns xyxy person boxes.

    Args:
        onnx_model: Path to the ``.onnx`` file.
        model_input_size: ``(height, width)`` of the network input (rtmlib convention).
        nms_thr: NMS IoU threshold.
        score_thr: Detection score threshold (attribute, may be changed after construction).
        device: ``'auto'``, ``'cuda'`` or ``'cpu'``.
        intra_op_num_threads: onnxruntime CPU threads (``0`` = its default).
    """

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple[int, int] = (640, 640),
        nms_thr: float = 0.45,
        score_thr: float = 0.7,
        device: str = "cpu",
        intra_op_num_threads: int = 0,
    ) -> None:
        self.onnx_model = str(onnx_model)
        self.model_input_size = tuple(model_input_size)
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.session = create_session(self.onnx_model, device, intra_op_num_threads)
        self.device = session_device(self.session)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        return yolox_preprocess(img, self.model_input_size)

    def inference(self, img: np.ndarray) -> list[np.ndarray]:
        """``(H, W, 3)`` image -> raw graph outputs (``float32`` NCHW input, batch 1)."""
        x = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]
        return self.session.run(self.output_names, {self.input_name: x})

    def postprocess(self, outputs: np.ndarray, ratio: float = 1.0) -> np.ndarray:
        boxes, _ = yolox_postprocess(outputs, ratio, self.model_input_size, self.nms_thr, self.score_thr)
        return boxes

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Person boxes ``(M, 4)`` xyxy (an empty array when nobody is detected)."""
        image, ratio = self.preprocess(image_bgr)
        outputs = self.inference(image)[0]
        return self.postprocess(outputs, ratio)
