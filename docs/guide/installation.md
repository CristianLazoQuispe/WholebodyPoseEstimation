# Installation

WholeBodyPose supports Python 3.10 to 3.13. The core package depends on NumPy, OpenCV (headless) and
ONNX Runtime, which is all that RTMPose (YOLOX + RTMW) and ViTPose (ONNX engine) need: no PyTorch,
rtmlib or onepose. MediaPipe and the other features are optional extras.

## From GitHub

```bash
pip install "wholebodypose @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"          # RTMPose + ViTPose
pip install "wholebodypose[all] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"     # + MediaPipe, filters, ffmpeg
```

### Extras

| Extra | Installs | Needed for |
|:--|:--|:--|
| (core) | `numpy`, `opencv-python-headless`, `onnxruntime` | `PoseEstimator("rtmpose")`, `PoseEstimator("vitpose")` |
| `mediapipe` | `mediapipe` | `PoseEstimator("mediapipe")` |
| `filters` | `padasip` | `FilteringNLMS` (temporal smoothing of the paper) |
| `video` | `imageio-ffmpeg` | a bundled `ffmpeg` that re-encodes rendered videos to browser-playable H.264 |
| `gpu` | `onnxruntime-gpu` | RTMPose / ViTPose on CUDA (see [GPU](#gpu)) |
| `vitpose-torch` (alias `torch`) | `torch` | ViTPose reference engine `engine="torch"`; also needs onepose (below) |
| `datasets` | `h5py`, `pandas` | `wholebodypose.datasets`, `wholebodypose-build-islr` |
| `demo` | `gradio` + `mediapipe,video` | the Gradio app in `hf/space/` |
| `all` | `mediapipe,filters,video` | everything except `gpu`, `vitpose-torch`, `demo` and `dev` |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `pre-commit`, `build` | development |

`rtmpose`, `vitpose` and `vitpose-onnx` are kept as names for compatibility with 1.0; they now only
install `onnxruntime`, which the core already has.

`wholebodypose backends` lists the backends, the install command each one needs, the ONNX Runtime
version and execution providers, and which optional packages are present. Asking for a backend whose
extra is missing raises an `ImportError` that names the `pip install` command.

### Reference engines (optional)

The ONNX engines give the same outputs as the original libraries (see [onnx.md](onnx.md)). To run the
original code paths for comparison:

- **RTMPose through rtmlib:** `pip install rtmlib`, then `RTMPoseModel(engine="rtmlib")`. Also needed for
  rtmlib's `backend="opencv"` / `"openvino"`.
- **ViTPose through PyTorch + onepose:** onepose is not on PyPI and PyPI does not allow packages to depend
  on a git URL, so install it by hand at the pinned commit:

  ```bash
  pip install "wholebodypose[vitpose-torch]"   # torch
  pip install "onepose @ git+https://github.com/developer0hye/onepose.git@dcb645fb52d2f3055dff893965d484e20b917c0f"
  ```

  then `VITPoseModel(engine="torch")` (or `engine="auto"`, which falls back to torch when the `.onnx`
  weight cannot be fetched).

## GPU

`device="auto"` (the default) uses CUDA when it is available and falls back to CPU otherwise.

- **RTMPose and ViTPose** run on ONNX Runtime. CUDA needs `onnxruntime-gpu`, which provides the same
  `onnxruntime` module as the CPU wheel that the core install pulls in. If both are installed, whichever
  was installed last wins, so remove both and reinstall only the GPU build:

  ```bash
  pip install "wholebodypose[gpu] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
  pip uninstall -y onnxruntime onnxruntime-gpu && pip install "onnxruntime-gpu>=1.17"
  python -c "import onnxruntime as o; print(o.get_available_providers())"   # must list CUDAExecutionProvider
  ```

  The `onnxruntime-gpu` build must match your CUDA / cuDNN version (see the
  [ONNX Runtime requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).
  `wholebodypose backends` prints the available providers.
- **ViTPose `engine="torch"`** (reference) runs on PyTorch: install the CUDA build of `torch` for your
  driver first (<https://pytorch.org/get-started/locally/>).
- **MediaPipe** always runs on CPU in this library.

## From source (development)

```bash
git clone https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git
cd WholebodyPoseEstimation
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest -m "not slow"        # unit tests: no backend, no download needed
pytest -m slow              # real backends on data/images/image.jpg (downloads weights)
```

`pip install -e ".[dev]"` alone already runs RTMPose and ViTPose (ONNX). The unit tests replace the
backends with a fake model; the `slow` tests run the real backends and skip what is not installed. To
also run the comparisons against the reference engines, install `rtmlib`, a CPU PyTorch
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) and onepose as shown above.

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
