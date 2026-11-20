# Documentation

The repository [README](../README.md) is the overview. The guides below go one level deeper; every
default and signature they quote is taken from the code in [`src/wholebodypose/`](../src/wholebodypose).

```text
docs/
├── guide/
│   ├── installation.md     extras, CPU / GPU, from source, Docker
│   ├── quickstart.md       command line and Python in five minutes
│   ├── backends.md         RTMPose, ViTPose, MediaPipe: models, paper defaults, thresholds, speed
│   ├── keypoints.md        COCO-WholeBody 133 layout and the 135-point variant
│   ├── video-pipeline.md   Video2Pose options and outputs, save/load formats, filtering
│   └── weights.md          Hugging Face mirror, cache, offline use
├── datasets/
│   ├── README.md           ISLR pose datasets: HDF5 schema, loading, rebuilding from videos
│   └── wlasl.md            WLASL provenance and processing
└── assets/                 figures used by the README, the guides and the project page
```

| I want to… | read |
|:--|:--|
| try it without installing anything | the [demo Space](https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo) |
| install the package (CPU, GPU, one backend, Docker) | [guide/installation.md](guide/installation.md) |
| get keypoints from an image or a video right now | [guide/quickstart.md](guide/quickstart.md), [`examples/`](../examples/README.md) |
| choose a backend, or know what the defaults are | [guide/backends.md](guide/backends.md) |
| know which index is the left wrist / the right hand / the face | [guide/keypoints.md](guide/keypoints.md) |
| turn sign language videos into training sequences | [guide/video-pipeline.md](guide/video-pipeline.md) |
| smooth keypoints (NLMS) or drop bad frames | [guide/video-pipeline.md#filtering](guide/video-pipeline.md#filtering) |
| run on a machine without internet, or mirror the weights | [guide/weights.md](guide/weights.md) |
| use the pose datasets of the paper | [datasets/README.md](datasets/README.md), the [dataset card](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) |
| rebuild the pose datasets from the raw videos | [datasets/README.md#rebuilding-from-videos](datasets/README.md#rebuilding-from-videos) |
| add a backend or send a fix | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| read the paper | the [project page](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/) |
