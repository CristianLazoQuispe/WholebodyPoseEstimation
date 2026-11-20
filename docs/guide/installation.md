# Installation

WholeBodyPose supports Python 3.10 to 3.13. The core package depends only on NumPy and
OpenCV (headless); each pose backend is an optional extra, so you install only what you run.

## From GitHub

```bash
pip install "wholebodypose[all] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
```

`git` must be available: the ViTPose extra pulls [onepose](https://github.com/developer0hye/onepose)
from GitHub at a pinned commit.

### Extras

| Extra | Installs | Needed for |
|:--|:--|:--|
| `rtmpose` | `rtmlib`, `onnxruntime` | `PoseEstimator("rtmpose")` (paper default backend) |
| `vitpose` | `torch`, `onepose` | `PoseEstimator("vitpose")` |
| `mediapipe` | `mediapipe` | `PoseEstimator("mediapipe")` |
| `filters` | `padasip` | `FilteringNLMS` (temporal smoothing of the paper) |
| `video` | `imageio-ffmpeg` | a bundled `ffmpeg` that re-encodes rendered videos to browser-playable H.264 |
| `gpu` | `onnxruntime-gpu` | RTMPose on CUDA |
| `demo` | `gradio` + `rtmpose,mediapipe,vitpose,video` | the Gradio app in `hf/space/` |
| `all` | `rtmpose,vitpose,mediapipe,filters,video` | everything except `gpu`, `demo` and `dev` |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `pre-commit`, `build` | development |

Examples:

```bash
pip install "wholebodypose[mediapipe] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"           # CPU only, light
pip install "wholebodypose[rtmpose,filters,video] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
```

`wholebodypose backends` lists the backends and the extra each one needs. Asking for a backend
whose extra is missing raises an `ImportError` that names the `pip install` command.

## GPU

`device="auto"` (the default) uses CUDA when it is available and falls back to CPU otherwise.

- **RTMPose** runs on ONNX Runtime. CUDA needs `onnxruntime-gpu`, which provides the same
  `onnxruntime` module as the CPU package, so only one of them may be installed:

  ```bash
  pip install "wholebodypose[rtmpose] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
  pip uninstall -y onnxruntime && pip install onnxruntime-gpu
  python -c "import onnxruntime as o; print(o.get_available_providers())"   # must list CUDAExecutionProvider
  ```

  The `onnxruntime-gpu` build must match your CUDA / cuDNN version (see the
  [ONNX Runtime requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)).
- **ViTPose** runs on PyTorch. Install the CUDA build of `torch` for your driver first
  (<https://pytorch.org/get-started/locally/>), then the `vitpose` extra.
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

On a CPU-only machine, `pip install torch --index-url https://download.pytorch.org/whl/cpu` before
`".[all,dev]"` avoids downloading the CUDA wheels of PyTorch.

`pip install -e ".[dev]"` alone (no backend) is enough to work on the core: the unit tests replace the
backends with a fake model, and the tests that need a real backend are skipped.

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
pip install -e ".[demo]" && python hf/space/app.py   # [demo] includes all three backends
```

## Check the installation

```bash
wholebodypose --version
wholebodypose backends
wholebodypose image data/images/image.jpg -b mediapipe    # -> outputs/image_mediapipe.{png,json}
```

Model weights are fetched on first use; see [weights.md](weights.md) to pre-download them or to
work offline.
