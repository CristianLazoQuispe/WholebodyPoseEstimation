# Quick start

Assumes `pip install "wholebodypose[rtmpose,mediapipe]"` (see [installation.md](installation.md)).
Sample inputs are in the repository: `data/images/image.jpg`, `data/videos/videoTest.mp4`.

## Command line

```bash
wholebodypose backends                                    # backends and their pip extras
wholebodypose image data/images/image.jpg -b mediapipe    # -> outputs/image_mediapipe.png, .json
wholebodypose video data/videos/videoTest.mp4 -b rtmpose --fps 15
                                                          # -> outputs/videoTest_rtmpose.mp4, .npz
wholebodypose download                                    # pre-fetch every weight file
```

| Option (`image` and `video`) | Default | Meaning |
|:--|:--|:--|
| `--backend, -b` | `rtmpose` | `rtmpose`, `vitpose` or `mediapipe` |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--style` | `coco133` | skeleton drawing: `coco133`, or `openpose135` (adds neck and pelvis) |
| `--output, -o` | `outputs` | output folder |
| `--format` | `json` (image), `npz` (video) | keypoint file format |
| `--quiet, -q` | off | no loading messages |

| Option (`video` only) | Default | Meaning |
|:--|:--|:--|
| `--fps` | input FPS | sub-sample the input to about this FPS |
| `--max-frames` | all | stop after this many processed frames |
| `--resize-width` | `640` | frames are resized to this width before inference (paper); `0` keeps the size |
| `--multiperson` | off | keep every person (default: the largest one per frame) |
| `--background` | none | draw on a solid color (`black`, `white`, `red`, `green`, `blue`) |
| `--no-render` | off | only write the keypoint file |

`wholebodypose image` stores the result as one frame holding `N` people: keypoints `(1, N, 133, 2)`.
`wholebodypose video` stores `(T, 133, 2)` (single person) or a ragged multi-person sequence; see
[video-pipeline.md](video-pipeline.md#saving-and-loading).

## Python

```python
import cv2
from wholebodypose import PoseEstimator

est = PoseEstimator("rtmpose")                     # paper defaults; or "vitpose", "mediapipe"

# from a file
keypoints, scores = est.predict_image("data/images/image.jpg")
print(keypoints.shape, scores.shape)               # (N, 133, 2) pixels (x, y), (N, 133)

# from an array: every backend takes RGB uint8
bgr = cv2.imread("data/images/image.jpg")
keypoints, scores = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
cv2.imwrite("drawn.png", est.draw(bgr, keypoints, scores))
```

- `N` is the number of people (`0` when nobody is found; RTMPose is multi-person, ViTPose and MediaPipe
  return at most one).
- Undetected keypoints are `(0, 0)` with score `0`. With the paper defaults (`use_thresholding=True`)
  scores are binary: `1` for kept keypoints, `0` otherwise.
- `est.model` is the underlying backend object; `est(image)` is the same as `est.predict(image)`.

Override any backend argument by keyword:

```python
PoseEstimator("rtmpose", mode="lightweight", device="cpu")       # faster RTMW-l model
PoseEstimator("vitpose", model_name="ViTPose+_base_coco_wholebody")
PoseEstimator("mediapipe", image_mode=False)                     # tracking mode for video streams
PoseEstimator("rtmpose", use_thresholding=False)                 # raw confidences
```

All options and their defaults are in [backends.md](backends.md).

### Video

```python
from wholebodypose import PoseEstimator, save_poses, load_poses

est = PoseEstimator("rtmpose")
res = est.process_video("data/videos/videoTest.mp4", "outputs/", video_processing_fps=15)
res["total_keypoints"].shape           # (T, 133, 2)
res["filepath_result"]                 # outputs/videoTest_rtmpose.mp4

save_poses("outputs/videoTest.npz", res["total_keypoints"], res["total_scores"], fps=res["processing_fps"])
keypoints, scores, meta = load_poses("outputs/videoTest.npz")
```

Pass `None` instead of a folder to skip rendering. Everything about videos, file formats and
filtering is in [video-pipeline.md](video-pipeline.md).

## Next

- [`examples/`](../../examples/README.md): image, video, filtering and webcam scripts.
- [keypoints.md](keypoints.md): what each of the 133 indices is.
