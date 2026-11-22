# Vendored from rtmlib 0.0.16 (https://github.com/Tau-J/rtmlib), file rtmlib/tools/solution/wholebody.py.
#
# rtmlib: Copyright 2018-2020 Open-MMLab (as stated in its LICENSE); author Tao Jiang (Tau-J) and contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy is in LICENSES/Apache-2.0.txt and at
# http://www.apache.org/licenses/LICENSE-2.0. Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# Modified by the WholeBodyPose authors: the mode table also names the WholeBodyPose weight keys
# (files are resolved by wholebodypose.weights, with the upstream URLs kept for reference), the
# models are the vendored ONNX Runtime YOLOX/RTMPose, ``format_result`` (OpenPose result objects) is
# dropped, and the code is reformatted. The detection -> per-box pose pipeline is unchanged.
"""YOLOX + RTMW whole-body pipeline (rtmlib's ``Wholebody`` solution)."""

from __future__ import annotations

import numpy as np

from wholebodypose.models.rtmpose._onnx.rtmw import RTMPose
from wholebodypose.models.rtmpose._onnx.yolox import YOLOX

_MMPOSE = "https://download.openmmlab.com/mmpose/v1/projects"

# rtmlib's mode table. Input sizes follow rtmlib: detector (height, width), pose (width, height).
MODE: dict[str, dict] = {
    "performance": {
        "det": f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "det_key": "yolox_m",
        "det_input_size": (640, 640),
        "pose": f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        "pose_key": "rtmw_x_384",  # RTMW-l, 384x288
        "pose_input_size": (288, 384),
    },
    "lightweight": {
        "det": f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        "det_key": "yolox_tiny",
        "det_input_size": (416, 416),
        "pose": f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
        "pose_key": "rtmw_l_256",  # RTMW-m, 256x192
        "pose_input_size": (192, 256),
    },
    "balanced": {
        "det": f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        "det_key": "yolox_m",
        "det_input_size": (640, 640),
        "pose": f"{_MMPOSE}/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        "pose_key": "rtmw_x_256",  # RTMW-l, 256x192
        "pose_input_size": (192, 256),
    },
}


class Wholebody:
    """Detect people with YOLOX, then run RTMW on every box.

    ``__call__(bgr_image)`` returns keypoints ``(N, 133, 2)`` (float64) and scores ``(N, 133)``
    (float32). As in rtmlib, when YOLOX finds nobody RTMW runs once on the whole image.

    Args:
        det, pose: Paths to the YOLOX and RTMW ``.onnx`` files.
        det_input_size: YOLOX input ``(height, width)``.
        pose_input_size: RTMW input ``(width, height)``.
        to_openpose: OpenPose-style output ordering.
        device: ``'auto'``, ``'cuda'``, ``'coreml'`` or ``'cpu'``.
        intra_op_num_threads: onnxruntime CPU threads for each session (``0`` = its default).
    """

    def __init__(
        self,
        det: str,
        pose: str,
        det_input_size: tuple[int, int] = (640, 640),
        pose_input_size: tuple[int, int] = (288, 384),
        to_openpose: bool = False,
        device: str = "cpu",
        intra_op_num_threads: int = 0,
    ) -> None:
        self.det_model = YOLOX(
            det, model_input_size=det_input_size, device=device, intra_op_num_threads=intra_op_num_threads
        )
        self.pose_model = RTMPose(
            pose,
            model_input_size=pose_input_size,
            to_openpose=to_openpose,
            device=device,
            intra_op_num_threads=intra_op_num_threads,
        )
        self.device = self.pose_model.device

    def __call__(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bboxes = self.det_model(image_bgr)
        return self.pose_model(image_bgr, bboxes=bboxes)
