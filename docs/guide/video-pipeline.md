# Video pipeline

`wholebodypose.video.Video2Pose` turns a video file into a keypoint sequence and, optionally, a
rendered video. `PoseEstimator.process_video` and `wholebodypose video` are thin wrappers around it.

```python
from wholebodypose import PoseEstimator, Video2Pose

# through PoseEstimator (reuses its model)
res = PoseEstimator("rtmpose").process_video("sign.mp4", "outputs/", video_processing_fps=15)

# or directly
runner = Video2Pose(model_name="mediapipe", mode_coco=True, verbose=False)
res = runner.predict("sign.mp4", folder_results=None, max_frames=100)  # keypoints only
```

## Constructor

`Video2Pose(mode_coco=True, model_name="rtmpose", verbose=True, model=None, device="auto", **model_kwargs)`

| Argument | Meaning |
|:--|:--|
| `mode_coco` | drawing style: 133-point skeleton, or `False` for the [135-point](keypoints.md#the-135-point-variant) one |
| `model_name` | backend built with `create_model` (paper defaults + `model_kwargs`); ignored when `model` is given |
| `model` | an already built backend (any `BasePoseModel`), e.g. to process many videos with one model |
| `device` | device for the backend built from `model_name` |
| `**model_kwargs` | backend overrides, e.g. `mode="lightweight"` |

## `predict` options

`Video2Pose.predict(filepath, folder_results=None, **options)`; `PoseEstimator.process_video(filepath, folder_results=None, **options)` takes the same options.

| Option | Default | Meaning |
|:--|:--|:--|
| `folder_results` | `None` | write `<stem>_<backend>.mp4` here; `None` skips rendering |
| `video_processing_fps` | `None` | sub-sample to about this FPS: every `ceil(fps / video_processing_fps)`-th frame is processed |
| `max_frames` | `None` | stop after this many **processed** frames |
| `resize_width` | `640` | resize frames to this width before inference (paper setting); `None` keeps the original size. **Keypoints are in the resized frame** |
| `multiperson` | `False` | `False`: keep the person with the largest bounding box in each frame; `True`: keep everyone |
| `output_video_fps` | processing FPS | FPS of the rendered video; the default keeps the original playback speed |
| `web_compatible` | `True` | re-encode the rendered video to H.264 / yuv420p / faststart with ffmpeg (from `imageio-ffmpeg`, `[video]` extra, or `ffmpeg` on `PATH`); without ffmpeg the OpenCV `mp4v` file is kept |
| `save_as_mov` | `False` | also write a `.mov` copy |
| `background_plot` | `None` | draw on a solid background: `red`, `blue`, `green`, `white`, `black` |
| `line_width`, `radius` | `2`, `1` | skeleton drawing style (keypoints with score ≥ 0.5 are drawn) |
| `show` | `False` | display frames in an OpenCV window (`q` stops) |
| `progress` | `None` | callback `progress(processed, expected_total)` after every frame |

Rotation metadata (phone videos) is applied: by OpenCV when it supports
`CAP_PROP_ORIENTATION_AUTO`, otherwise from `ffprobe` when available.

Each frame goes through the backend and through `clean_outliers`, which resets a "phantom cluster"
of keypoints collapsed near one corner (a failure mode of some backends) to `(0, 0)`.

## Returned dictionary

| Key | Value |
|:--|:--|
| `total_keypoints` | single person: `float32 (T, 133, 2)`; multi-person: list of `T` arrays `(N_t, 133, 2)` |
| `total_scores` | single person: `float32 (T, 133)`; multi-person: list of `(N_t, 133)` |
| `n_frames` | `T`, the number of processed frames |
| `processing_fps` | effective FPS of the sequence (`fps / frame_skip`) |
| `frame_size` | `(width, height)` of the processed (resized) frames, `None` if no frame was read |
| `filepath`, `width`, `height`, `fps`, `total_frames`, `duration` | input video properties |
| `backend` | backend name |
| `filepath_result` | rendered MP4 (only when `folder_results` is set) |
| `filepath_result_mov` | MOV copy (only with `save_as_mov=True` and ffmpeg) |

In single-person mode, frames without a detection are all zeros, so `T` always equals the number of
processed frames.

## Saving and loading

```python
from wholebodypose import save_poses, load_poses

save_poses(
    "clip.npz",
    res["total_keypoints"],
    res["total_scores"],
    backend="rtmpose",
    fps=res["processing_fps"],
    frame_size=res["frame_size"],
)
keypoints, scores, meta = load_poses("clip.npz")
```

`save_poses(path, keypoints, scores, **meta)` picks the format from the suffix; any JSON-serializable
keyword is stored in `meta`, together with `format_version` (currently `1`) and
`layout="coco_wholebody_133"`.

| Format | Content |
|:--|:--|
| `.npz` (compressed) | `keypoints` `float32 (T, 133, 2)`, `scores` `float32 (T, 133)`, `meta` (JSON string). Multi-person sequences with different `N_t` are stored flat, `(ΣN_t, 133, 2)`, plus `counts` `(T,)`. Loads with `allow_pickle=False` |
| `.json` | `{"meta": {...}, "keypoints": [...], "scores": [...]}`; coordinates rounded to 0.01 px, scores to 1e-4 |

`load_poses(path) -> (keypoints, scores, meta)` returns arrays when every frame has the same shape and
lists of per-frame arrays for ragged multi-person sequences.

From NumPy only:

```python
import json, numpy as np

d = np.load("clip.npz")
keypoints, scores, meta = d["keypoints"], d["scores"], json.loads(str(d["meta"]))
```

## Filtering

Three tools, all working on single-person sequences `(T, K, 2)`:

### `PoseSequenceFilter`: drop unusable frames

`wholebodypose.filters.sequence.PoseSequenceFilter(left_hand_indices=None, right_hand_indices=None)`

| Method | Removes frames… | Parameter (default) |
|:--|:--|:--|
| `filter_invalid_frames(kp, threshold)` | where the fraction of `(0, 0)` keypoints is ≥ `threshold` | `0.7` |
| `filter_by_motion(kp, motion_threshold)` | whose mean keypoint displacement from the previous frame is ≤ `motion_threshold` px (the first frame is kept) | `1.0` |
| `filter_by_hand_visibility(kp, visibility_threshold)` | where neither hand's mean coordinate magnitude exceeds the threshold (needs the hand indices) | `0.01` |
| `apply_all_filters(kp, invalid_threshold, motion_threshold, visibility_threshold, apply_hand_filter=True)` | the three above, in that order | |

Each method returns `(kept_keypoints, mask)` and always keeps at least two frames.
`apply_all_filters` returns `(kept_keypoints, {"invalid": m1, "motion": m2, "visibility": m3})`; each
mask indexes the **output of the previous stage**, so recover the kept frame indices with

```python
import numpy as np
from wholebodypose import PoseSequenceFilter

flt = PoseSequenceFilter(left_hand_indices=np.arange(91, 112), right_hand_indices=np.arange(112, 133))
kept, masks = flt.apply_all_filters(keypoints)
idx = np.arange(len(keypoints))
for m in masks.values():
    idx = idx[m]
kept_scores = scores[idx]
```

The filters only see keypoints; select the matching scores with the indices as above.

### `FilteringNLMS`: the paper's temporal smoothing

`wholebodypose.FilteringNLMS(num_keypoints=135, mu=1.05, seed=0)`, `[filters]` extra (padasip).

```python
from wholebodypose import FilteringNLMS

kp_f, sc_f = FilteringNLMS(num_keypoints=133).apply(keypoints, scores)  # (T, 133, 2), (T, 133)
```

Every coordinate and score of every keypoint has a Normalized Least-Mean-Squares adaptive filter
(seeded by `seed`) whose input is the time step and the confidence. Per frame and keypoint:

- score ≥ 0.5: the observation is kept;
- 0.25 ≤ score < 0.5: the output is `0.5 · observation + 0.5 · prediction`;
- score < 0.25: the last kept position is held, with its score decreasing by 0.05 per frame. The
  memory is refreshed while that score stays ≥ 0.25 and expires 5 frames after the last refresh,
  so a gap after a confident keypoint (score 1) is filled for about 20 frames; after that the
  keypoint is `(0, 0)` with score 0.

The filters always adapt to the observation, not to their output. Use it on raw confidences
(`use_thresholding=False`) to benefit from the middle band; with binary scores it acts as a
short-gap filler. `num_keypoints` defaults to 135 (dataset layout): pass `133` for library output.
The online building blocks (`initialize_adaptive_filter`, `KeypointMemory`, `track_keypoints` in
`wholebodypose.filters.filtering_nlms`) can run frame by frame.

### `MeanMedianFilter`: moving smoothing

`wholebodypose.filters.filtering_mean.MeanMedianFilter(window_size=5)` is an online smoother: call
it once per frame with `(K, 2)` keypoints; once the window is full it returns
`(moving mean + moving median) / 2` over the last `window_size` frames (before that, the input
unchanged). `reset()` clears the window.

```python
import numpy as np
from wholebodypose.filters.filtering_mean import MeanMedianFilter

f = MeanMedianFilter(window_size=5)
smoothed = np.stack([f(k) for k in keypoints])
```
