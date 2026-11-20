# Docker

Two images that install WholeBodyPose from this repository (`pip install ".[all]"`). Build them
from the **repository root** (the build context must contain `pyproject.toml` and `src/`).

| File | Base | Backends |
|:--|:--|:--|
| [`Dockerfile.cpu`](Dockerfile.cpu) | `python:3.12-slim` | RTMPose and ViTPose (ONNX Runtime CPU), MediaPipe |
| [`Dockerfile.gpu`](Dockerfile.gpu) | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` | RTMPose and ViTPose (`onnxruntime-gpu`), MediaPipe (CPU) |

Both run as an unprivileged user, use `/workspace` as working directory, cache weights in
`$WHOLEBODYPOSE_HOME` (`~/.cache/wholebodypose` of that user) and default to `wholebodypose --help`.

## Build and run

```bash
docker build -f docker/Dockerfile.cpu -t wholebodypose:cpu .
docker run --rm -v "$PWD:/workspace" wholebodypose:cpu \
    wholebodypose image data/images/image.jpg -b mediapipe          # -> outputs/image_mediapipe.png

docker build -f docker/Dockerfile.gpu -t wholebodypose:gpu .
docker run --rm --gpus all -v "$PWD:/workspace" wholebodypose:gpu \
    wholebodypose video data/videos/videoTest.mp4 -b rtmpose --device cuda
```

Keep downloaded weights between runs with a volume: `-v wbp-weights:/home/wbp/.cache/wholebodypose`
(CPU image) or `-v wbp-weights:/home/ubuntu/.cache/wholebodypose` (GPU image).

Build arguments:

| Arg | Default | Image |
|:--|:--|:--|
| `EXTRAS` | `all` | both; e.g. `--build-arg EXTRAS=video` for a smaller image without MediaPipe |
| `INSTALL_JUPYTER` | `false` | both; `true` adds JupyterLab |
| `PYTHON_VERSION` | `3.12` | CPU |
| `CUDA_IMAGE` | CUDA 12.6 + cuDNN 9 | GPU; match your driver |

## Compose

```bash
docker compose -f docker/docker-compose-cpu.yaml build
docker compose -f docker/docker-compose-cpu.yaml run --rm wholebodypose \
    wholebodypose video data/videos/videoTest.mp4 -b rtmpose --max-frames 30
docker compose -f docker/docker-compose-cpu.yaml up jupyter     # JupyterLab on http://localhost:8888
```

`docker-compose-gpu.yaml` is the same with the GPU image and an NVIDIA device reservation. Both mount
the repository on `/workspace` and keep the weights in the named volume `weights`.
