# License of the model weights

All files in this repository are third-party model weights released by their authors under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). They are redistributed unchanged, except
the ViTPose+ ONNX files (`vitpose/*.onnx`), which are modified versions of the Apache-2.0 checkpoints:
converted to ONNX, no retraining.

| Files | Authors | License | Training data | Commercial use |
|:--|:--|:--|:--|:--|
| `rtmpose/rtmw-*.onnx` | OpenMMLab | Apache-2.0 | Cocktail14 (incl. COCO-WholeBody, UBody, Halpe, InterHand2.6M) | not guaranteed |
| `rtmpose/yolox_*_humanart-*.onnx` | OpenMMLab | Apache-2.0 | Human-Art | not guaranteed |
| `vitpose/*.pth`, `vitpose/*.onnx` | ViTAE-Transformer (fp16 split by onepose) | Apache-2.0 | COCO, AI Challenger, MPII, AP-10K, APT-36K, COCO-WholeBody | not guaranteed |
| `mediapipe/*.task` | Google | Apache-2.0 | Google | yes |

**Training-data restrictions.** RTMW, the Human-Art YOLOX detectors and ViTPose+ were trained on datasets
whose terms restrict commercial use (COCO-WholeBody: CC BY-NC 4.0; MPII: no commercial use; Human-Art:
non-commercial). No claim is made that commercial use of these weights is permitted; check those terms before
any commercial use.

The weights are provided as is, without warranty of any kind (Apache-2.0, Sections 7-8). Cite the original
papers when you use a model (see `README.md`).
