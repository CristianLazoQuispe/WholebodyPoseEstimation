# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- ViTPose on ONNX Runtime: `VITPoseModel(engine="auto" | "onnx" | "torch")` and the `[vitpose-onnx]` extra
  (only `onnxruntime`, no PyTorch or onepose). Pre- and post-processing are NumPy/OpenCV ports of onepose's;
  multi-person crops run as one batch. `auto` picks ONNX when the `.onnx` file can be found, else PyTorch.
- `scripts/export_vitpose_onnx.py` (opset 17, dynamic batch) and the weight keys
  `vitpose_{small,base,large,huge}_onnx` with pinned SHA-256 (mirror only, no upstream URL).
- `scripts/vitpose_onnx_study.py` and `docs/guide/onnx.md`: torch-vs-ONNX agreement, CPU latency and an
  int8/fp16 quantization study for ViTPose-base and RTMW-x.

### Changed
- Project page: the pipeline diagram is now the hero figure; the teaser video (and its generator code) is
  removed. New sections: datasets (one row per published subset with links to every split file and label
  map), data processing (annotation checks, subset rules, reproducibility), and ISLR models (paper Table 3
  with links to each model's original code). The weights button is marked "soon" while the mirror is private.

## [1.0.0] - 2026-11-19

First stable release, consolidating the library, project page and demo of the ICMV 2025 paper in one repository.

### Added
- Multi-person for every backend: `PoseEstimator(backend, multiperson=True, max_people=...)`. ViTPose runs
  on YOLOX person crops (`detector="yolox_m"`), MediaPipe detects up to `max_people` bodies and attaches
  hands/faces to the nearest body. RTMPose was already multi-person.
- `PoseTracker` (`wholebodypose.tracking`): stable person IDs across frames; `Video2Pose` returns
  `track_ids` and draws `#id` labels in multi-person mode; the CLI stores them in the pose file metadata.
- `wholebodypose.datasets` and `wholebodypose-build-islr` (extra `[datasets]`): read/write the published ISLR
  pose HDF5 schema, adapters for WLASL, MS-ASL, ASL Citizen, LSA64 and PUCP-DGI that rebuild the published
  sample lists, and a resumable multi-process/multi-GPU video -> HDF5 builder. Preset `icmv2025` reproduces
  the released keypoints; `RTMPoseModel(legacy_rgb_input=True)` replays the RGB-to-rtmlib input it needs.
- `PoseEstimator` / `create_model`: one entry point for every backend, with the paper settings as defaults.
- `wholebodypose` command line tool (`image`, `video`, `backends`, `download`).
- `save_poses` / `load_poses` for `.npz` and `.json`, including multi-person sequences.
- Weight resolution with a Hugging Face mirror, upstream fallback and SHA-256 checks (`wholebodypose.weights`).
- MediaPipe Tasks engine (works with every current MediaPipe release; the legacy `mp.solutions` API was
  removed in MediaPipe 0.10.30). `engine="auto"` keeps the paper's `solutions` engine when available.
- `Video2Pose`: `resize_width`, `max_frames`, `progress` callback, `web_compatible` H.264 output, model reuse.
- `ensure_constant_fps`: variable-frame-rate inputs (webcam/browser WebM) are re-encoded before processing.
- Documentation (`docs/`), runnable `examples/`, Docker images that install the package, pre-commit, CONTRIBUTING,
  issue/PR templates, Dependabot, release workflow.
- Hugging Face tooling (`hf/`): weight mirror with pinned SHA-256 for every file, model card, dataset card,
  and the demo Space (`hf/space/`) rebuilt on the v1 API and deployed + smoke-tested by CI.
- Project page (`website/`) deployed to GitHub Pages; the old `wholebodypose.github.io` page redirects to it.
- Test suite (unit tests without downloads + `slow` real-backend tests), CI, GitHub Pages deployment,
  automatic Hugging Face Space deployment.

### Changed
- Backends are optional extras; `import wholebodypose` only needs NumPy and OpenCV. Python 3.10-3.13, NumPy 2.
- All backends take **RGB** input. RTMPose and ViTPose now receive BGR internally, as their models expect
  (0.1.0 fed them RGB).
- Backends return `N = 0` people (empty arrays) when nobody is detected.
- RTMPose: no PyTorch dependency; `filter_noise` applies to every detected person; the default keypoint
  threshold depends on the mode (`performance` 2.5, `balanced`/`lightweight` 0.35), because the 256x192
  RTMW models produce confidences on a different scale.
- MediaPipe: hands are assigned to the left/right COCO slots by proximity to the body wrists (0.1.0 used the
  detection order); feet map heel -> heel and foot index -> toes; face is searched on a head crop so that
  small faces are found; per-instance configuration (0.1.0 mutated module-level defaults).
- `Video2Pose`: output FPS defaults to the effective processing FPS (0.1.0 always wrote 15 fps, so videos
  played at the wrong speed); rotation metadata handled by OpenCV; correct writer size with rotation and
  solid backgrounds; outlier cleaning reimplemented in NumPy.
- NLMS filter is seeded (`FilteringNLMS(seed=0)`) and no longer prints debug output.

### Removed
- ISLR training notebooks and experiments (`codes/`, `results/`); they remain available at tag `archive/pre-v1.0`.
- `filters/filtering_kalman.py` (unused, initialized state at import time) and `models/rtmpose/refine.py`.
- `MediapipeModel.draw_mediapipe` and the `ffmpeg-python` dependency.

## [0.1.0] - 2025

Version used for the ICMV 2025 paper (tag `archive/pre-v1.0`).
