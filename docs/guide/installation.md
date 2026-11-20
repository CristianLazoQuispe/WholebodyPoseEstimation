# Installation

Python 3.10 to 3.13. The core package depends on NumPy, OpenCV (headless) and ONNX Runtime, which is all
that RTMPose and ViTPose need.

```bash
pip install "wholebodypose @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
pip install "wholebodypose[mediapipe] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
```

## Extras

| Extra | Installs | Needed for |
|:--|:--|:--|
| (core) | `numpy`, `opencv-python-headless`, `onnxruntime` | `PoseEstimator("rtmpose")`, `PoseEstimator("vitpose")` |
| `mediapipe` | `mediapipe` | `PoseEstimator("mediapipe")` |
| `video` | `imageio-ffmpeg` | re-encoding rendered videos to browser-playable H.264 |
| `filters` | `padasip` | `FilteringNLMS` (temporal smoothing of the paper) |
| `datasets` | `h5py` | `wholebodypose.datasets`, `wholebodypose-build-islr` |
| `gpu` | `onnxruntime-gpu` | RTMPose / ViTPose on CUDA (see [GPU](#gpu)) |
| `demo` | `gradio` + `mediapipe,video` | the Gradio app in `hf/space/` |
| `all` | `mediapipe,filters,video` | |
| `dev` | `pytest`, `ruff`, `pre-commit`, `build`, `h5py`, `huggingface_hub` | development |

`wholebodypose backends` lists the backends, the install command each one needs, the ONNX Runtime
execution providers and the installed extras.

## GPU

`device="auto"` (the default) uses CUDA when the installed ONNX Runtime has it and CPU otherwise.
`onnxruntime-gpu` provides the same `onnxruntime` module as the CPU wheel of the core install, so
uninstall the CPU wheel first:

```bash
pip install "wholebodypose[gpu] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
pip uninstall -y onnxruntime onnxruntime-gpu && pip install "onnxruntime-gpu>=1.17"
python -c "import onnxruntime as o; print(o.get_available_providers())"   # must list CUDAExecutionProvider
```

`onnxruntime-gpu` must match your CUDA / cuDNN version
([requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).
MediaPipe runs on CPU.

## From source (development)

```bash
git clone https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git
cd WholebodyPoseEstimation
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest -m "not slow"        # unit tests: no backend, no download needed
pytest -m slow              # real backends on data/images/image.jpg (downloads weights)
```

The unit tests replace the backends with a fake model; the `slow` tests run the real backends and skip
MediaPipe when it is not installed.

## Docker

`docker/` has a CPU and a GPU image that install the package from the repository:

```bash
docker build -f docker/Dockerfile.cpu -t wholebodypose:cpu .
docker run --rm -v "$PWD:/workspace" wholebodypose:cpu \
    wholebodypose video data/videos/videoTest.mp4 -b rtmpose --max-frames 30
```

See [`docker/README.md`](../../docker/README.md) for the GPU image, Compose and Jupyter.

## Demo app

The [demo Space](https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo) (upload a video
or an image, pick a backend, get the skeleton overlay and the keypoints) is built from
[`hf/space/`](../../hf/space) by CI. To run it locally:

```bash
pip install -e ".[demo]" && python hf/space/app.py   # core (RTMPose, ViTPose) + MediaPipe
```

## Check the installation

```bash
wholebodypose --version
wholebodypose backends
wholebodypose image data/images/image.jpg -b mediapipe    # -> outputs/image_mediapipe.{png,json}
```

Model weights are fetched on first use; see [weights.md](weights.md) to pre-download them or to
work offline.
