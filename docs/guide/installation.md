# Installation

Python 3.10 to 3.13. The core package depends on NumPy, OpenCV (headless) and ONNX Runtime, which is all
that RTMPose and ViTPose need.

```bash
pip install wholebodypose
pip install "wholebodypose[mediapipe]"
```

## Extras

| Extra | Installs | Needed for |
|:--|:--|:--|
| (core) | `numpy`, `opencv-python-headless`, `onnxruntime` | `PoseEstimator("rtmpose")`, `PoseEstimator("vitpose")` |
| `mediapipe` | `mediapipe` | `PoseEstimator("mediapipe")` |
| `video` | `imageio-ffmpeg` | re-encoding rendered videos to browser-playable H.264 |
| `filters` | `padasip` | `FilteringNLMS` (temporal smoothing of the paper) |
| `datasets` | `h5py` | `wholebodypose.datasets`, `wholebodypose-build-islr` |
| `webcam` | `opencv-python` | OpenCV with windows (`cv2.imshow`) for [`examples/webcam.py`](../../examples/webcam.py) |
| `gpu` | `onnxruntime-gpu` | RTMPose / ViTPose on CUDA (see [GPU](#gpu)) |
| `demo` | `gradio` + `mediapipe,video` | the Gradio app in `hf/space/` |
| `all` | `mediapipe,filters,video` | |
| `dev` | `pytest`, `ruff`, `pre-commit`, `build`, `h5py`, `huggingface_hub` | development |

`wholebodypose backends` lists the backends, the install command each one needs, the ONNX Runtime
execution providers and the installed extras.

## GPU

`device="auto"` (the default) uses CUDA when the installed ONNX Runtime has it, then CoreML
([macOS](#macos-coreml)), then CPU.
`onnxruntime-gpu` provides the same `onnxruntime` module as the CPU wheel of the core install, so
uninstall the CPU wheel first:

```bash
pip install "wholebodypose[gpu]"
pip uninstall -y onnxruntime onnxruntime-gpu && pip install "onnxruntime-gpu[cuda,cudnn]>=1.21"
wholebodypose backends   # must list CUDAExecutionProvider
```

The `[cuda,cudnn]` extras install the CUDA and cuDNN libraries as pip packages, so only the NVIDIA driver
is needed; WholeBodyPose loads them when CUDA is requested. With a system-wide CUDA / cuDNN install,
`onnxruntime-gpu` must match its version
([requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).
MediaPipe runs on CPU.

## macOS (CoreML)

The `onnxruntime` wheel for macOS ships the CoreML execution provider, so the core install already uses the
Apple GPU / Neural Engine: `device="auto"` (or `"coreml"`) runs RTMW and YOLOX on CoreML, with CPU fallback
for unsupported nodes. ViTPose itself stays on CPU (its graph is slower on CoreML), but its YOLOX detector
uses CoreML. The first session takes 2-4 s to compile the CoreML model.

CoreML computes in reduced precision: keypoints match the CPU ones within a few pixels (an RTMW SimCC bin),
not bit for bit. Pass `device="cpu"` to reproduce CPU outputs exactly.

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
