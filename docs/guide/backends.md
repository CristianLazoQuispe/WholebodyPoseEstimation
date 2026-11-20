# Backends

Every backend is a subclass of `wholebodypose.models.base.BasePoseModel` with the same contract:

- input: an **RGB** `uint8` image `(H, W, 3)`;
- output: `keypoints` `float32 (N, 133, 2)` in pixels and `scores` `float32 (N, 133)`, in
  [COCO-WholeBody order](keypoints.md);
- `N = 0` (empty arrays) when nobody is detected; undetected keypoints are `(0, 0)` with score `0`.

The registry `wholebodypose.api.BACKENDS` stores, for each backend, its class, its pip extra and its
**paper settings**. `PoseEstimator(backend)` and `create_model(backend)` start from those settings and
apply your keyword arguments on top:

```python
from wholebodypose import BACKENDS, PoseEstimator, create_model

BACKENDS["rtmpose"].defaults  # {'mode': 'performance', 'use_thresholding': True, 'filter_noise': True}
est = PoseEstimator("rtmpose", mode="lightweight")  # paper settings, except the model size
model = create_model("mediapipe", kpt_thr=0.3)  # the bare backend, without drawing/video helpers
```

| Backend | Paper settings (`BACKENDS[...].defaults`) | Extra |
|:--|:--|:--|
| `rtmpose` | `mode="performance"`, `use_thresholding=True`, `filter_noise=True` | none, core install (+ `gpu`) |
| `vitpose` | `model_name="ViTPose+_huge_coco_wholebody"`, `use_thresholding=True`, `kpt_thr=0.5` | none, core install (+ `gpu`) |
| `mediapipe` | `use_thresholding=True`, `kpt_thr=0.5` | `mediapipe` |

`create_model` also passes `device` (default `"auto"`) to RTMPose and ViTPose, and `verbose` to all.

## Overview

| Backend | Model | People | Runs on | Install extra |
|:--|:--|:-:|:--|:--|
| `rtmpose` | YOLOX-m + RTMW-l 384×288 (OpenMMLab) | multi | ONNX Runtime CPU / CUDA | core (+ `[gpu]`) |
| `vitpose` | ViTPose+ huge/large/base/small, top-down | single · multi with YOLOX crops | ONNX Runtime CPU / CUDA | core (+ `[gpu]`) |
| `mediapipe` | BlazePose heavy + Hands + Face landmarker | single · multi (`max_people`) | CPU | `[mediapipe]` |

## Multi-person

`PoseEstimator(backend, multiperson=True)` works with every backend. It applies
`wholebodypose.api.MULTIPERSON_DEFAULTS` on top of the paper settings: RTMPose is always multi-person;
ViTPose gets `detector="yolox_m"` (YOLOX person crops, core install); MediaPipe gets
`engine="tasks"` and `max_people=4`. `max_people` caps the number of people for ViTPose and MediaPipe.
In videos, every frame keeps all people and `process_video` adds stable person IDs (`track_ids`):

```python
est = PoseEstimator("vitpose", multiperson=True, max_people=4)
res = est.process_video("dance.mp4", "outputs/")
# res["total_keypoints"][t]: (N_t, 133, 2), res["track_ids"][t]: (N_t,)
```

The per-backend sections below describe the default (single-person) behavior.

## Thresholding

With `use_thresholding=True` (paper setting for all three backends), `BasePoseModel.thresholding`
compares each score with `kpt_thr`: keypoints below it become `(0, 0)` with score `0`, the others keep
their coordinates and get score `1`. Scores are therefore **binary** with the defaults. Pass
`use_thresholding=False` to get the raw confidences of the model.

The confidence scales differ between models, which is why the thresholds differ:

| Backend | Raw score | Default `kpt_thr` |
|:--|:--|:--|
| RTMPose `performance` (RTMW-l 384×288) | SimCC, not calibrated; about 5-8 on visible joints | **2.5** |
| RTMPose `balanced` / `lightweight` (256×192) | SimCC; about 1 on visible joints | **0.35** |
| ViTPose+ | heatmap maximum, 0-1 | 0.5 |
| MediaPipe | already 0/1 (landmark present and visible) | 0.5 |

For RTMPose, `kpt_thr=None` (the class default) selects the threshold from
`wholebodypose.models.rtmpose.model.DEFAULT_KPT_THR = {"performance": 2.5, "balanced": 0.35, "lightweight": 0.35}`.
The same threshold drives `filter_noise`.

## RTMPose (`rtmpose`)

`wholebodypose.models.rtmpose.model.RTMPoseModel`: a YOLOX person detector followed by the RTMW
whole-body model (OpenMMLab, [arXiv:2407.08634](https://arxiv.org/abs/2407.08634)) on ONNX Runtime.
**Multi-person.** The pre- and post-processing of [rtmlib](https://github.com/Tau-J/rtmlib) (letterbox,
NMS, affine crop, SimCC decoding) is vendored in `wholebodypose.models.rtmpose._onnx` (Apache-2.0, see
[THIRD_PARTY.md](../../THIRD_PARTY.md); [onnx.md](onnx.md#rtmpose)).

| `mode` | Detector | Pose model | Input |
|:--|:--|:--|:--|
| `"performance"` (paper) | YOLOX-m (640×640) | RTMW-l | 384×288 |
| `"balanced"` | YOLOX-m (640×640) | RTMW-l | 256×192 |
| `"lightweight"` | YOLOX-tiny (416×416) | RTMW-m | 256×192 |

Model names follow the [MMPose RTMW table](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#wholebody-2d-133-keypoints): the files `rtmw-dw-x-l_*` are **RTMW-l** (distilled from RTMW-x; 70.1 whole-body AP at 384×288) and `rtmw-dw-l-m_*` is **RTMW-m**. The paper calls the paper model "RTMW-x"; the weights are the RTMW-l files above (weight key `rtmw_x_384`).

| Argument | Default (class) | Notes |
|:--|:--|:--|
| `device` | `"auto"` | `cuda` only if ONNX Runtime has `CUDAExecutionProvider` (install `onnxruntime-gpu`), else `cpu` |
| `intra_op_num_threads` | `0` | ONNX Runtime CPU threads per session (`0` = its default) |
| `mode` | `"performance"` | see the table above |
| `to_openpose` | `False` | OpenPose-style ordering instead of COCO-WholeBody |
| `use_thresholding` | `False` (paper: `True`) | see [Thresholding](#thresholding) |
| `kpt_thr` | `None` → `DEFAULT_KPT_THR[mode]` | |
| `filter_noise` | `False` (paper: `True`) | zero a foot (and its ankle) unless hip, knee and ankle are all above `kpt_thr`, and a hand (and its wrist) unless shoulder, elbow, wrist and hand root are; applied to every person |
| `verbose` | `True` | |
| `legacy_rgb_input` | `False` | feed RGB to the models (which expect BGR), as the ICMV 2025 pipeline did; used by the `icmv2025` dataset preset |

RTMW is fed BGR internally, as the OpenMMLab models expect; you still pass RGB.

## ViTPose (`vitpose`)

`wholebodypose.models.vitpose.model.VITPoseModel`: ViTPose+ whole-body
([ViTPose++](https://arxiv.org/abs/2212.04246)), ONNX exports of the
[onepose](https://github.com/developer0hye/onepose) checkpoints on ONNX Runtime ([onnx.md](onnx.md#vitpose);
files: [weights.md](weights.md#files)). Top-down on the **full frame** by default (paper setting), so it
returns **one person** and works best when that person fills most of the frame, as in sign language
videos. With `detector="yolox_m"` (or `PoseEstimator(..., multiperson=True)`) it runs on YOLOX person
crops instead (the same YOLOX as RTMPose).

| Argument | Default (class) | Notes |
|:--|:--|:--|
| `device` | `"auto"` | `cuda` if ONNX Runtime has `CUDAExecutionProvider`, else `cpu` |
| `model_name` | `"ViTPose+_huge_coco_wholebody"` | also `ViTPose+_large_…`, `ViTPose+_base_…`, `ViTPose+_small_…` (`…` = `coco_wholebody`) |
| `use_thresholding` | `False` (paper: `True`) | |
| `kpt_thr` | `0.5` | heatmap confidence, 0-1 |
| `verbose` | `True` | |
| `detector` | `None` | `None` (full frame), `"yolox_m"` or `"yolox_tiny"` |
| `max_people`, `det_score_thr`, `box_padding` | `None`, `0.5`, `0.25` | detector mode only |
| `intra_op_num_threads` | `0` | ONNX Runtime CPU threads per session, ViTPose and YOLOX (`0` = its default) |

## MediaPipe (`mediapipe`)

`wholebodypose.models.mediapipe.model.MediapipeModel`: MediaPipe pose (BlazePose, 33 landmarks),
hands (21 per hand) and face mesh, mapped to COCO-WholeBody. **Single person, CPU.** The mapping:

- body: BlazePose nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles → COCO 0-16;
- feet: heel → heel (19, 22); MediaPipe's single "foot index" point → both toes (17-18, 20-21);
- face: 68 face-mesh vertices → COCO 23-90;
- hands: each detected hand goes to the left (91-111) or right (112-132) slot of the body wrist it is
  closest to; without a body, the (mirrored) MediaPipe handedness label is used.

MediaPipe returns presence, not a confidence: mapped keypoints get score `1`, missing ones `0`.

### Engines: `solutions` and `tasks`

| `engine` | API | Used |
|:--|:--|:--|
| `"solutions"` | legacy `mp.solutions` (`Pose`, `Hands`, `FaceMesh`); removed in MediaPipe 0.10.30 | in the paper |
| `"tasks"` | MediaPipe Tasks (`PoseLandmarker` heavy/full, `HandLandmarker`, `FaceLandmarker`); `.task` model files are downloaded like the other weights | every current MediaPipe release |
| `"auto"` (default) | `solutions` when `mp.solutions` exists, `tasks` otherwise | |

The Tasks face detector misses small faces in full-body shots, so the `tasks` engine searches the
face on a head crop built from the body pose and falls back to the full image. Results of the two
engines are close but not identical; to reproduce the paper exactly, use `mediapipe<0.10.30`.

| Argument | Default | Notes |
|:--|:--|:--|
| `models` | `("hands", "face", "pose")` | parts to run, 1-3 of them |
| `image_mode` | `True` | `True`: every frame is independent; `False`: tracking across frames (video / webcam) |
| `pose_config` | `DEFAULT_POSE_CONFIG` | `min_detection_confidence=0.3`, `min_tracking_confidence=0.3`, `model_complexity=2` (heavy; `<2` selects the `full` model with Tasks), `smooth_landmarks=True`, … |
| `hands_config` | `DEFAULT_HANDS_CONFIG` | `min_detection_confidence=0.3`, `min_tracking_confidence=0.3`, `max_num_hands=2`, `model_complexity=1` |
| `face_config` | `DEFAULT_FACE_CONFIG` | `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`, `max_num_faces=1`, `refine_landmarks=False` |
| `use_thresholding`, `kpt_thr` | `False`, `0.5` (paper: `True`, `0.5`) | |
| `engine` | `"auto"` | see above |
| `verbose` | `True` | |

Config dictionaries are merged into the defaults, so `pose_config={"model_complexity": 1}` changes only
that key. Call `model.close()` (or let it be garbage-collected) to release the MediaPipe graphs.

## Speed

Steady-state time per 640×427 frame on CPU (8 cores of an AMD EPYC 7662, ONNX Runtime 1.32, MediaPipe
Tasks engine), `PoseEstimator(..., device="cpu")` after three warm-up frames. Only the ratios matter.

| Backend | ms / frame |
|:--|--:|
| `mediapipe` (pose heavy + hands + face) | ~120 |
| `rtmpose`, `mode="lightweight"` | ~160 |
| `rtmpose`, `mode="balanced"` | ~290 |
| `rtmpose`, `mode="performance"` (paper) | ~310 |
| `vitpose`, `ViTPose+_small_coco_wholebody` | ~40 |
| `vitpose`, `ViTPose+_base_coco_wholebody` | ~75 |

- RTMPose and ViTPose are much faster on a GPU (`[gpu]` extra); MediaPipe stays on CPU.
- RTMPose time grows with the number of people (one pose pass per detected person).
- In `Video2Pose`, `resize_width=640` (default) and `video_processing_fps` are the main speed knobs.

## Adding a backend

Subclass `BasePoseModel`, implement `_predict(image_rgb) -> (keypoints, scores)` in COCO-WholeBody
order, and register it in `wholebodypose.api.BACKENDS`. See [CONTRIBUTING.md](../../CONTRIBUTING.md#adding-a-backend).

## Citing the backends

If you use WholeBodyPose, please cite the paper (BibTeX in the [README](../../README.md#citation)) and
also the backend you use, plus COCO-WholeBody for the keypoint layout:

```bibtex
@article{jiang2024rtmw,
  title   = {RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation},
  author  = {Jiang, Tao and Xie, Xinchen and Li, Yining},
  journal = {arXiv preprint arXiv:2407.08634},
  year    = {2024}
}
@article{xu2024vitpose++,
  title   = {ViTPose++: Vision Transformer for Generic Body Pose Estimation},
  author  = {Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {46},
  number  = {2},
  pages   = {1212--1230},
  year    = {2024},
  doi     = {10.1109/TPAMI.2023.3330016}
}
@article{lugaresi2019mediapipe,
  title   = {MediaPipe: A Framework for Building Perception Pipelines},
  author  = {Lugaresi, Camillo and others},
  journal = {arXiv preprint arXiv:1906.08172},
  year    = {2019}
}
@inproceedings{jin2020whole,
  title     = {Whole-Body Human Pose Estimation in the Wild},
  author    = {Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle = {ECCV},
  year      = {2020}
}
```

## Acknowledgements

WholeBodyPose builds on [rtmlib](https://github.com/Tau-J/rtmlib) / [MMPose](https://github.com/open-mmlab/mmpose),
[onepose](https://github.com/developer0hye/onepose) / [ViTPose](https://github.com/ViTAE-Transformer/ViTPose),
[MediaPipe](https://github.com/google-ai-edge/mediapipe) and [padasip](https://github.com/matousc89/padasip).
See [THIRD_PARTY.md](../../THIRD_PARTY.md) for licenses.
