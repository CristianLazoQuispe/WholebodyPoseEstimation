# Third-party code and models

WholeBodyPose is MIT-licensed. It does not vendor third-party source code, but it depends on the following
projects and downloads (or mirrors on the Hugging Face Hub, unchanged) their model weights.

| Component | Used for | License | Source |
|:--|:--|:--|:--|
| rtmlib | RTMPose/RTMW inference | Apache-2.0 | https://github.com/Tau-J/rtmlib |
| RTMW, YOLOX (ONNX) weights | `rtmpose` backend | Apache-2.0 | https://github.com/open-mmlab/mmpose (download.openmmlab.com) |
| onepose | ViTPose inference | Apache-2.0 | https://github.com/developer0hye/onepose |
| ViTPose+ whole-body weights (fp16) | `vitpose` backend | Apache-2.0 | https://github.com/ViTAE-Transformer/ViTPose, converted by onepose |
| MediaPipe + Tasks models (`*.task`) | `mediapipe` backend | Apache-2.0 | https://github.com/google-ai-edge/mediapipe |
| padasip | NLMS filtering | MIT | https://github.com/matousc89/padasip |
| COCO-WholeBody keypoint definition | 133-keypoint layout and colors (`utils/coco133.py`, from MMPose) | Apache-2.0 / annotations CC BY 4.0 | https://github.com/jin-s13/COCO-WholeBody |

Weights mirrored in [`CristianLazoQuispe/wholebodypose-weights`](https://huggingface.co/CristianLazoQuispe/wholebodypose-weights)
are byte-identical copies, listed with their SHA-256 and original URL in its `MANIFEST.csv`; their original
licenses and authors apply. Please cite the original papers (see README) when you use a backend.

Example images and videos in `data/` and `hf/space/examples/` are stock media used for demonstration only.
