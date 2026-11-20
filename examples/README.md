# Examples

Short, self-contained scripts. Run them from the repository root after installing the package with
the backend you want (`pip install -e ".[rtmpose,mediapipe,filters,video]"`, or `".[all]"`).
Weights are downloaded on first use (see [docs/guide/weights.md](../docs/guide/weights.md)).

| Script | What it does | Extras |
|:--|:--|:--|
| [`image_inference.py`](image_inference.py) | One image → keypoints, per-part visibility, a drawn PNG and a JSON file | backend |
| [`video_to_npz.py`](video_to_npz.py) | Video → `(T, 133, 2)` sequence in `.npz` (+ rendered MP4) | backend, `video` (optional) |
| [`filter_sequence.py`](filter_sequence.py) | `.npz` → NLMS smoothing, frame filtering (`PoseSequenceFilter`), mean/median smoothing | `filters` for `--nlms` |
| [`webcam_realtime.py`](webcam_realtime.py) | Live webcam / stream with an optional online NLMS or mean/median filter | backend, `filters` for `--smooth nlms` |

```bash
# 1. an image (MediaPipe is the lightest backend; RTMPose is the paper default)
python examples/image_inference.py data/images/image.jpg --backend mediapipe
python examples/image_inference.py data/images/image.jpg --backend rtmpose --mode lightweight

# 2. a video, sub-sampled to 15 fps, first 60 processed frames
python examples/video_to_npz.py data/videos/videoTest.mp4 --backend rtmpose --fps 15 --max-frames 60
#    -> outputs/videoTest_rtmpose.npz, outputs/videoTest_rtmpose.mp4

# 3. clean the sequence
python examples/filter_sequence.py outputs/videoTest_rtmpose.npz --nlms
#    -> outputs/videoTest_rtmpose_filtered.npz

# 4. real time (q quits); --source also takes a video file or a stream URL
python examples/webcam_realtime.py --backend mediapipe --smooth nlms
python examples/webcam_realtime.py --source data/videos/videoTest.mp4 --no-show --max-frames 50
```

Every script takes `--help`. The same tasks are available without Python through the
`wholebodypose` command (`wholebodypose image ...`, `wholebodypose video ...`); see
[docs/guide/quickstart.md](../docs/guide/quickstart.md).
