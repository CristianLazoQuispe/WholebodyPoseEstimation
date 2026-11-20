# Third-party code and models

WholeBodyPose's own code is licensed under Apache-2.0 ([`LICENSE`](LICENSE)); code contributed up to v1.0.0
is also available under MIT ([`LICENSES/MIT.txt`](LICENSES/MIT.txt)). Third-party components keep their own
licenses. Overview of every license: [docs/LICENSING.md](docs/LICENSING.md).

## Vendored code

| Files | Origin | License | Changes |
|:--|:--|:--|:--|
| `src/wholebodypose/models/rtmpose/_onnx/yolox.py` | [rtmlib](https://github.com/Tau-J/rtmlib) 0.0.16, `tools/object_detection/yolox.py` and `post_processings.py` (itself modified from [DWPose](https://github.com/IDEA-Research/DWPose)) | Apache-2.0 | ONNX Runtime only, shared session helper, reformatted; arithmetic unchanged |
| `src/wholebodypose/models/rtmpose/_onnx/rtmw.py` | rtmlib 0.0.16, `tools/pose_estimation/rtmpose.py`, `pre_processings.py`, `post_processings.py` | Apache-2.0 | ONNX Runtime only, 2D SimCC code only, reformatted; arithmetic unchanged |
| `src/wholebodypose/models/rtmpose/_onnx/wholebody.py` | rtmlib 0.0.16, `tools/solution/wholebody.py` | Apache-2.0 | mode table names WholeBodyPose weight keys; `format_result` dropped |
| `src/wholebodypose/models/vitpose/onnx_runtime.py` (pre-/post-processing) | [onepose](https://github.com/developer0hye/onepose) | Apache-2.0 | NumPy/OpenCV port |
| `src/wholebodypose/utils/coco133.py` | MMPose COCO-WholeBody dataset config (Copyright OpenMMLab) | Apache-2.0 | keypoint names and colors only |

rtmlib is by Tao Jiang and contributors ("Copyright 2018-2020 Open-MMLab"). Each vendored file carries the
Apache-2.0 notice and its changes; the license text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## Dependencies

| Component | Used for | License | Source |
|:--|:--|:--|:--|
| ONNX Runtime | RTMPose, YOLOX and ViTPose inference | MIT | https://github.com/microsoft/onnxruntime |
| MediaPipe | `mediapipe` backend | Apache-2.0 | https://github.com/google-ai-edge/mediapipe |
| padasip | NLMS filtering | MIT | https://github.com/matousc89/padasip |

## Model weights

| Weights | Backend | License | Training data | Commercial use | Source |
|:--|:--|:--|:--|:--|:--|
| RTMW-l / RTMW-m (ONNX) | `rtmpose` | Apache-2.0 | Cocktail14 (incl. COCO-WholeBody, UBody, Halpe, InterHand2.6M, ...) | not guaranteed: training data includes non-commercial datasets | https://github.com/open-mmlab/mmpose |
| YOLOX-m / YOLOX-tiny, Human-Art (ONNX) | `rtmpose`, ViTPose detector | Apache-2.0 | Human-Art (non-commercial) | not guaranteed | https://github.com/open-mmlab/mmpose |
| ViTPose+ whole-body ONNX (`vitpose/*.onnx`) | `vitpose` | Apache-2.0 (modified: converted to ONNX, no retraining) | COCO, AI Challenger, MPII, AP-10K, APT-36K, COCO-WholeBody | not guaranteed: MPII and COCO-WholeBody are non-commercial | https://github.com/ViTAE-Transformer/ViTPose (fp16 checkpoints by onepose), ONNX export on the WholeBodyPose mirror |
| MediaPipe Tasks models (`*.task`) | `mediapipe` | Apache-2.0 | Google | yes | https://github.com/google-ai-edge/mediapipe |
| COCO-WholeBody keypoint definition | 133-keypoint layout | annotations CC BY-NC 4.0 | — | — | https://github.com/jin-s13/COCO-WholeBody |

The whole-body weights are released by their authors under Apache-2.0, but were trained on datasets whose
terms restrict commercial use (COCO-WholeBody: CC BY-NC 4.0; MPII: no commercial use; Human-Art:
non-commercial). Check those terms before any commercial use. MediaPipe models have no such restriction.

The mirror [`CristianLazoQuispe/wholebodypose-weights`](https://huggingface.co/CristianLazoQuispe/wholebodypose-weights)
holds byte-identical copies plus the ViTPose ONNX conversions; `MANIFEST.csv` lists each file with its SHA-256
and original URL, and [`hf/WEIGHTS_LICENSE.md`](hf/WEIGHTS_LICENSE.md) states the terms above. Please cite the
original papers when you use a backend ([docs/guide/backends.md](docs/guide/backends.md#citing-the-backends)).

Example images and videos in `data/` and `hf/space/examples/` are stock media used for demonstration only.
