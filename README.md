<div align="center">

# WholeBodyPose

One interface for RTMPose, ViTPose+ and MediaPipe that always returns the 133 COCO-WholeBody keypoints, plus the video pipeline that turns sign language videos into training-ready pose sequences (ICMV 2025, Proceedings of SPIE).

**Cristian Lazo-Quispe, Renato Castro-Cruz, Mauricio Salazar-Espinosa, Gissella Bejarano**

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square)](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/static/paper/WholeBodyPose_ICMV2025.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.1117%2F12.3096312-blue?style=flat-square)](https://doi.org/10.1117/12.3096312)
[![Project page](https://img.shields.io/badge/Project-Page-6f42c1?style=flat-square)](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Space-yellow?style=flat-square)](https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-pose--action--recognition-orange?style=flat-square)](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition)
[![PyPI](https://img.shields.io/pypi/v/wholebodypose?style=flat-square)](https://pypi.org/project/wholebodypose/)
[![CI](https://img.shields.io/github/actions/workflow/status/CristianLazoQuispe/WholebodyPoseEstimation/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green?style=flat-square)](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/LICENSE)

</div>

<img src="https://raw.githubusercontent.com/CristianLazoQuispe/WholebodyPoseEstimation/main/docs/assets/pipeline.svg" width="100%" alt="WholeBodyPose pipeline: input frames, pose backend (RTMPose, ViTPose+, MediaPipe), unified COCO-WholeBody output with 133 keypoints, optional temporal clean-up, training-ready pose sequences">

## Installation

```bash
pip install wholebodypose               # RTMPose + ViTPose (NumPy, OpenCV, ONNX Runtime)
pip install "wholebodypose[mediapipe]"  # + MediaPipe backend
pip install "wholebodypose[gpu]"        # CUDA ONNX Runtime (uninstall onnxruntime first)
pip install "wholebodypose[all]"        # MediaPipe, temporal filters, video codecs
```

Python 3.10–3.13. Weights download on first use. Details: [installation](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/installation.md).

| Backend | Install | Default model | Notes |
|:--|:--|:--|:--|
| `rtmpose` | core | RTMW-l + YOLOX-m (`mode="performance"`) | `mode="balanced"` / `"lightweight"` for speed |
| `vitpose` | core | ViTPose+-H | `model_name="ViTPose+_base_coco_wholebody"` for speed |
| `mediapipe` | `[mediapipe]` | BlazePose heavy + Hands + Face landmarker | CPU; mapped to the same 133 points |

## Image

```python
import cv2
from wholebodypose import PoseEstimator

est = PoseEstimator("rtmpose")  # or "vitpose", "mediapipe"
keypoints, scores = est.predict_image("image.jpg")  # (N, 133, 2) pixels, (N, 133)
cv2.imwrite("pose.png", est.draw(cv2.imread("image.jpg"), keypoints, scores))
```

## Video

```python
from wholebodypose import PoseEstimator

est = PoseEstimator("rtmpose", mode="lightweight")
res = est.process_video("video.mp4", "outputs/")
print(res["total_keypoints"].shape, res["filepath_result"])  # (T, 133, 2), outputs/video_rtmpose.mp4
```

Command line:

```bash
wholebodypose image image.jpg -b rtmpose
wholebodypose video video.mp4 -b vitpose --style openpose135   # -> outputs/video_vitpose.{mp4,npz}
```

## Keypoint layouts

| COCO-WholeBody 133 (default) | OpenPose-style 135 |
|:--:|:--:|
| <img src="https://cristianlazoquispe.github.io/WholebodyPoseEstimation/static/images/kpts_coco133.jpg" width="300" alt="COCO-WholeBody 133-keypoint skeleton"> | <img src="https://cristianlazoquispe.github.io/WholebodyPoseEstimation/static/images/kpts_openpose135.jpg" width="300" alt="OpenPose-style 135-keypoint skeleton"> |
| `PoseEstimator(..., mode_coco=True)`, `--style coco133` | `PoseEstimator(..., mode_coco=False)`, `--style openpose135` |

Every backend returns the same 133 points (body 0–16, feet 17–22, face 23–90, left hand 91–111,
right hand 112–132). The 135-point variant adds neck and pelvis midpoints, used by the released pose
datasets; see [keypoints](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/keypoints.md) to build it from the output.

## Documentation

[Quickstart](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/quickstart.md) ·
[backends](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/backends.md) ·
[keypoints](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/keypoints.md) ·
[video pipeline](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/video-pipeline.md) ·
[weights](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/weights.md) ·
[examples](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/examples)

## Intended use

WholeBodyPose is a research tool, provided as is, without warranty. Pose estimates can be wrong,
especially for occluded hands, fast motion and low-resolution video; do not use them for decisions about
individuals without human review. Several model weights were trained on non-commercial data (see License).

## Citation

If you use WholeBodyPose, the pose datasets or the demo in your work, please cite:

```bibtex
@inproceedings{lazo2026wholebodypose,
  title        = {WholeBodyPose: a unified end-to-end framework for sign language recognition and pose-based training data},
  author       = {Lazo-Quispe, Cristian and Castro-Cruz, Renato and Bejarano, Gissella and Salazar-Espinosa, Mauricio},
  booktitle    = {Eighteenth International Conference on Machine Vision (ICMV 2025)},
  volume       = {14114},
  pages        = {69--79},
  year         = {2026},
  organization = {SPIE},
  doi          = {10.1117/12.3096312}
}
```

Please also cite the backend you use ([backends](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/guide/backends.md#citing-the-backends)) and, for the
pose datasets, the source dataset of every subset.

## License

- **Code**: [Apache-2.0](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/LICENSE).
- **Pose datasets**: [CC BY-NC-SA 4.0](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/hf/dataset_cards/LICENSE.md), citation required.
- **Model weights** keep their own licenses. The whole-body models (RTMW, YOLOX, ViTPose+) were trained on
  non-commercial data; check their terms before commercial use ([THIRD_PARTY.md](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/THIRD_PARTY.md)).

Details, and what to cite: [docs/LICENSING.md](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/LICENSING.md).
