# WholeBodyPose

One interface for RTMPose, ViTPose+ and MediaPipe that always returns the 133 COCO-WholeBody
keypoints, plus the video pipeline that turns sign language videos into training-ready pose sequences
(ICMV 2025, Proceedings of SPIE).

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square)](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/static/paper/WholeBodyPose_ICMV2025.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.1117%2F12.3096312-blue?style=flat-square)](https://doi.org/10.1117/12.3096312)
[![Project page](https://img.shields.io/badge/Project-Page-6f42c1?style=flat-square)](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Space-yellow?style=flat-square)](https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-pose--action--recognition-orange?style=flat-square)](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition)
[![CI](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/actions/workflows/ci.yml/badge.svg)](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green?style=flat-square)](LICENSE)

> **Please cite.** If WholeBodyPose helps your work, cite the [paper](#citation)
> (doi:[10.1117/12.3096312](https://doi.org/10.1117/12.3096312)), the backend you used and, for the pose
> datasets, the source dataset of every subset.

<img src="docs/assets/pipeline.svg" width="100%" alt="WholeBodyPose pipeline: input frames, pose backend (RTMPose, ViTPose+, MediaPipe), unified COCO-WholeBody output with 133 keypoints, optional temporal clean-up, training-ready pose sequences">

## Installation

```bash
pip install "wholebodypose @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
```

The core install (NumPy, OpenCV, ONNX Runtime) runs RTMPose and ViTPose; no PyTorch needed.

| Extra | For |
|:--|:--|
| `[mediapipe]` | the MediaPipe backend |
| `[gpu]` | ONNX Runtime CUDA for RTMPose and ViTPose |
| `[all]` | MediaPipe, NLMS filtering and bundled ffmpeg |

Weights download on first use. GPU, from source, Docker: [docs/guide/installation.md](docs/guide/installation.md).

## Usage

**Image**

```python
import cv2
from wholebodypose import PoseEstimator

est = PoseEstimator("rtmpose")  # or "vitpose", "mediapipe"
keypoints, scores = est.predict_image("data/images/image.jpg")  # (N, 133, 2) pixels, (N, 133)
cv2.imwrite("image_pose.png", est.draw(cv2.imread("data/images/image.jpg"), keypoints, scores))
```

**Video**

```python
from wholebodypose import PoseEstimator, save_poses

est = PoseEstimator("rtmpose")
res = est.process_video("data/videos/videoTest.mp4", "outputs/", video_processing_fps=15)
print(res["total_keypoints"].shape, res["filepath_result"])  # (T, 133, 2), outputs/videoTest_rtmpose.mp4
save_poses("outputs/videoTest.npz", res["total_keypoints"], res["total_scores"], fps=res["processing_fps"])
```

```bash
wholebodypose video data/videos/videoTest.mp4 -b rtmpose --fps 15   # -> outputs/videoTest_rtmpose.{mp4,npz}
```

## Links

- Documentation: [docs/](docs/README.md) (backends, keypoint layout, video pipeline, filtering, weights)
- Datasets: [pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) on the Hugging Face Hub ([schema and loading](docs/datasets/README.md))
- Demo: [WholeBodyPoseDemo](https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo), no install needed
- Project page: [cristianlazoquispe.github.io/WholebodyPoseEstimation](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/)

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

Please also cite the backend you use ([docs/guide/backends.md](docs/guide/backends.md#citing-the-backends)) and,
for the pose datasets, the source dataset of every subset you use (WLASL, MS-ASL, ASL Citizen, LSA64, PUCP-DGI;
BibTeX in the [dataset card](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition)).

## License

- **Code**: [Apache-2.0](LICENSE).
- **Pose datasets**: [CC BY-NC-SA 4.0](hf/dataset_cards/LICENSE.md), citation required.
- **Model weights** keep their own licenses. The whole-body models (RTMW, YOLOX, ViTPose+) were trained on
  non-commercial data; check their terms before commercial use ([THIRD_PARTY.md](THIRD_PARTY.md)).

Details, and what to cite: [docs/LICENSING.md](docs/LICENSING.md).
