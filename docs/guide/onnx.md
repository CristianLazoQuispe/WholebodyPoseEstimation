# ONNX Runtime models

RTMPose and ViTPose run on ONNX Runtime with NumPy/OpenCV pre- and post-processing. Both use one session
helper (`wholebodypose.models.onnx_session`) for the execution providers (CUDA or CoreML when the installed
`onnxruntime` has it, see [installation.md](installation.md#gpu) and [macOS](installation.md#macos-coreml)),
`intra_op_num_threads` and the
reported device. Arguments: [backends.md](backends.md).

## RTMPose

YOLOX + RTMW `.onnx` files from OpenMMLab. The pre- and post-processing are vendored from rtmlib 0.0.16
in `wholebodypose.models.rtmpose._onnx` (Apache-2.0, see [THIRD_PARTY.md](../../THIRD_PARTY.md)):

| Step | Module |
|:--|:--|
| YOLOX letterbox (gray 114 padding), grid decoding, NumPy NMS | `_onnx/yolox.py` |
| RTMW box -> center/scale (padding 1.25), affine crop, normalization; SimCC argmax, inverse transform | `_onnx/rtmw.py` |
| Detect, then one RTMW pass per box (whole image when nobody is detected); mode table | `_onnx/wholebody.py` |

Validation: on 62 frames (images in `data/images/` and `hf/space/examples/`, frames of 5 videos), the
vendored code and rtmlib 0.0.16 gave bit-identical boxes, keypoints and scores in 744/744 comparisons
(3 modes, BGR and RGB input, raw and `RTMPoseModel` outputs).

## ViTPose

| | |
|:--|:--|
| Input `input` | `(N, 3, 256, 192)` float32, RGB, `/255`, ImageNet mean/std; `N` is dynamic |
| Output `heatmaps` | `(N, 133, 64, 48)` float32 |
| Opset | 17 |
| Weights | stored as float16 and cast to float32 inside the graph; compute is float32 |

Processing outside the graph (`wholebodypose.models.vitpose.onnx_runtime`, ported from onepose):

1. person box (whole frame, or a padded YOLOX box) -> center and scale with the 192:256 aspect ratio;
2. affine warp to 192x256 (`cv2.warpAffine`, bilinear, no UDP);
3. heatmap argmax, a +/-0.25 px shift towards the higher neighbour, inverse transform to image pixels;
   the score is the heatmap maximum. No flip test.

In multi-person mode all person crops of a frame run as one batch.

Validation: against the original PyTorch model (onepose) on 62 inputs, all four sizes, full frame and
multi-person, the maximum keypoint difference was 0 px and the maximum score difference 5.1e-6.

## Quantization

int8 (dynamic) and fp16 variants of ViTPose+ base and RTMW-l 384x288 were evaluated on CPU and are not
used: int8 moves keypoints (ViTPose p95 10 px; RTMW changes half of the visibility decisions), and fp16 is
not faster on CPU.
