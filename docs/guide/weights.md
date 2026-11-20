# Model weights

WholeBodyPose wraps third-party models whose weights are downloaded at run time. Upstream links do
disappear, so every file the library uses is mirrored on the Hugging Face Hub in
[`CristianLazoQuispe/wholebodypose-weights`](https://huggingface.co/CristianLazoQuispe/wholebodypose-weights).
The logic is in `wholebodypose.weights`.


## Resolution order

`get_weight(key)` returns a local path, trying in order:

1. the local cache, `$WHOLEBODYPOSE_HOME/<filename>`;
2. the Hugging Face mirror, `$HF_ENDPOINT/$WHOLEBODYPOSE_WEIGHTS_REPO/resolve/main/<filename>`;
3. the original upstream URL (OpenMMLab `.zip` archives are extracted), if the file has one.

Downloads go to a temporary file first, so an interrupted download never leaves a broken file in the
cache. When a SHA-256 is pinned for a file, a mismatch deletes it and the next source is tried.

## Environment variables

| Variable | Default | Effect |
|:--|:--|:--|
| `WHOLEBODYPOSE_HOME` | `~/.cache/wholebodypose` | cache folder |
| `WHOLEBODYPOSE_OFFLINE` | `0` | `1`: never download; a missing file raises `FileNotFoundError` |
| `WHOLEBODYPOSE_WEIGHTS_REPO` | `CristianLazoQuispe/wholebodypose-weights` | mirror repository (e.g. your own copy) |
| `HF_ENDPOINT` | `https://huggingface.co` | Hub endpoint (e.g. a corporate proxy) |

`WHOLEBODYPOSE_WEIGHTS_REPO` and `HF_ENDPOINT` are read when `wholebodypose.weights` is imported;
`WHOLEBODYPOSE_HOME` and `WHOLEBODYPOSE_OFFLINE` on every call.

## Files

| Key | File (in the cache and the mirror) | Used by | Upstream |
|:--|:--|:--|:--|
| `yolox_m` | `rtmpose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` | RTMPose `performance`, `balanced`; ViTPose `detector="yolox_m"` | OpenMMLab |
| `yolox_tiny` | `rtmpose/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx` | RTMPose `lightweight`; ViTPose `detector="yolox_tiny"` | OpenMMLab |
| `rtmw_x_384` | `rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx` | RTMPose `performance` | OpenMMLab |
| `rtmw_x_256` | `rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx` | RTMPose `balanced` | OpenMMLab |
| `rtmw_l_256` | `rtmpose/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx` | RTMPose `lightweight` | OpenMMLab |
| `vitpose_{small,base,large,huge}` | `vitpose/splitted_vitpose+_<size>_coco_wholebody_half.pth` | ViTPose, `engine="torch"` | onepose releases |
| `vitpose_{small,base,large,huge}_onnx` | `vitpose/vitpose+_<size>_coco_wholebody.onnx` | ViTPose, `engine="onnx"` (default) | none: exported by WholeBodyPose, mirror only |
| `mp_pose_heavy` | `mediapipe/pose_landmarker_heavy.task` | MediaPipe Tasks, `model_complexity=2` | Google |
| `mp_pose_full` | `mediapipe/pose_landmarker_full.task` | MediaPipe Tasks, `model_complexity<2` | Google |
| `mp_hand` | `mediapipe/hand_landmarker.task` | MediaPipe Tasks | Google |
| `mp_face` | `mediapipe/face_landmarker.task` | MediaPipe Tasks | Google |

All of them are Apache-2.0 (see [THIRD_PARTY.md](../../THIRD_PARTY.md)). The RTMPose files are the ones
rtmlib downloads, read by the same (vendored) code, so outputs do not depend on whether rtmlib is installed. The MediaPipe `solutions`
engine ships its models inside the `mediapipe` wheel and downloads nothing.

The ViTPose `.onnx` files are written by
[`scripts/export_vitpose_onnx.py`](../../scripts/export_vitpose_onnx.py) from the onepose checkpoints
(opset 17, dynamic batch, float32 compute; the weights are stored as float16, which is lossless because the
checkpoints are float16). Re-running the script (PyTorch 2.15, onnx 1.23) reproduced the pinned SHA-256
byte for byte (checked for `small`). Sizes: small 46.5 MiB, base 171.8 MiB, large 588.8 MiB, huge 1215.8 MiB. See
[onnx.md](onnx.md).

ViTPose `engine="torch"`: `onepose` loads checkpoints from its own package folder (`onepose/models/weights/`). WholeBodyPose
fetches the file through `get_weight` and copies it there before building the model; if that fails,
onepose downloads it from its upstream URL itself.

## Pre-downloading and offline use

```bash
wholebodypose download                              # every file
wholebodypose download rtmw_x_384 yolox_m           # only RTMPose performance
WHOLEBODYPOSE_HOME=/data/wbp wholebodypose download mp_pose_heavy mp_hand mp_face
```

```python
from wholebodypose.weights import WEIGHTS, get_weight

path = get_weight("rtmw_x_384")
```

Then, on the offline machine, copy the cache folder and set

```bash
export WHOLEBODYPOSE_HOME=/data/wbp
export WHOLEBODYPOSE_OFFLINE=1
```

For ViTPose with `engine="torch"` offline, also run the model once online (or copy `onepose/models/weights/`), since onepose reads
from its own folder.

In CI, cache `~/.cache/wholebodypose` between runs (the repository's
[`ci.yml`](../../.github/workflows/ci.yml) does this with `actions/cache`).

## Your own mirror

The mirror layout is the `filename` column above. To host a copy, upload the cached files with the
same relative paths to a Hugging Face model repository and set
`WHOLEBODYPOSE_WEIGHTS_REPO=<user>/<repo>`.
