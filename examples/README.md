# Examples

Run from the repository root. Weights are downloaded on first use ([weights](../docs/guide/weights.md)).

| Script | Task | Install |
|:--|:--|:--|
| [`image.py`](image.py) | image → keypoints + drawn PNG | `wholebodypose` |
| [`video.py`](video.py) | video → `(T, 133, 2)` keypoints in `.npz` + rendered MP4 | `wholebodypose` |
| [`dataset.py`](dataset.py) | folder of labelled videos → HDF5 dataset | `wholebodypose[datasets]` |
| [`webcam.py`](webcam.py) | live webcam pose with FPS overlay | `wholebodypose[webcam]` |

```bash
python examples/image.py data/images/image.jpg --backend rtmpose
python examples/video.py data/videos/videoTest.mp4 --backend rtmpose --max-frames 60
python examples/dataset.py videos/ outputs/my_dataset.hdf5
python examples/webcam.py --backend rtmpose --mode lightweight
```

`--backend mediapipe` needs `wholebodypose[mediapipe]`. `webcam.py` opens a window, which the headless OpenCV
of the core install cannot do: `pip install "wholebodypose[webcam]"`. On macOS, allow camera access for your terminal.
