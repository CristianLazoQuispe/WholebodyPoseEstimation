# Docker

Two images that install WholeBodyPose from this repository (`pip install ".[all]"`). Build them
from the **repository root** (the build context must contain `pyproject.toml` and `src/`).

| File | Base | Backends |
|:--|:--|:--|
| [`Dockerfile.cpu`](Dockerfile.cpu) | `python:3.12-slim` | RTMPose (ONNX Runtime CPU), ViTPose (PyTorch CPU), MediaPipe |
| [`Dockerfile.gpu`](Dockerfile.gpu) | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` | RTMPose (`onnxruntime-gpu`), ViTPose (PyTorch cu126), MediaPipe (CPU) |

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
| `EXTRAS` | `all` | both; e.g. `--build-arg EXTRAS=rtmpose,video` for a smaller image |
| `INSTALL_JUPYTER` | `false` | both; `true` adds JupyterLab |
| `PYTHON_VERSION` | `3.12` | CPU |
| `CUDA_IMAGE`, `TORCH_INDEX` | CUDA 12.6 / `cu126` | GPU; change both together to match your driver |

## Compose

```bash
docker compose -f docker/docker-compose-cpu.yaml build
docker compose -f docker/docker-compose-cpu.yaml run --rm wholebodypose \
    wholebodypose video data/videos/videoTest.mp4 -b rtmpose --max-frames 30
docker compose -f docker/docker-compose-cpu.yaml up jupyter     # JupyterLab on http://localhost:8888
```

`docker-compose-gpu.yaml` is the same with the GPU image and an NVIDIA device reservation. Both mount
the repository on `/workspace` and keep the weights in the named volume `weights`.

## Webcam inside a container

Linux: `docker run --device /dev/video0 ...`. Docker Desktop (Windows / macOS) cannot pass a webcam,
so stream it from the host with [`stream.py`](stream.py) (needs only OpenCV on the host):

```bash
python docker/stream.py --port 8080                                          # on the host
docker run --rm -v "$PWD:/workspace" wholebodypose:cpu \
    python examples/webcam_realtime.py --source http://host.docker.internal:8080/ --no-show
```

On Linux hosts add `--add-host=host.docker.internal:host-gateway` to `docker run`.
