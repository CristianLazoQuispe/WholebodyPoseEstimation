---
title: WholeBodyPose Demo
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Whole-body 133-keypoint pose with RTMPose/ViTPose/MediaPipe
models:
  - CristianLazoQuispe/wholebodypose-weights
datasets:
  - CristianLazoQuispe/pose-action-recognition
tags:
  - pose-estimation
  - whole-body
  - coco-wholebody
  - sign-language
  - rtmpose
  - vitpose
  - mediapipe
  - computer-vision
---

# WholeBodyPose demo

Interactive demo of **[WholeBodyPose](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation)**
(ICMV 2025): whole-body pose estimation that returns the same **133 COCO-WholeBody keypoints**
(body, feet, face, both hands) from every backend.

| Backend (`backend=`) | Model | People |
|:--|:--|:-:|
| RTMPose (`rtmpose`) | YOLOX-m + RTMW-l 384×288, paper setting | multi |
| RTMPose-lite (`rtmpose_lite`) | YOLOX-tiny + RTMW-m 256×192, default for videos on CPU | multi |
| MediaPipe (`mediapipe`) | BlazePose heavy + Hands + Face Mesh | single |
| ViTPose+ (`vitpose`) | ViTPose+ whole-body (base on CPU, huge on GPU) | single |

On 2 CPU cores RTMW-l processed 0.5-0.9 frames/s in our tests, so on CPU hardware the Video tab defaults to
RTMPose-lite (about 2-3x faster on 2 CPU cores in our tests); the Image tab uses the paper model.

Upload a video (or record one with the webcam) or an image, choose the backend, the skeleton style
(COCO-WholeBody 133 or OpenPose-style 135), the drawing options and the processing FPS, and get

- the rendered skeleton video (H.264 MP4) or image (PNG), named `<input>_<backend>_pose.*`,
- the keypoints as `<input>_<backend>.npz` and `.json` (`wholebodypose.load_poses(path)` reads both),
- a summary (frames, people, processing time).

The *About* tab links the paper, the ISLR results and code, and every split of the pose datasets.
Videos are limited to their first 15 seconds. The app can also be called programmatically:

```python
from gradio_client import Client, handle_file

client = Client("CristianLazoQuispe/WholeBodyPoseDemo")
video, files, summary = client.predict(
    video=handle_file("sign.mp4"), backend="rtmpose_lite", processing_fps=10, api_name="/video"
)
image, files, summary = client.predict(image=handle_file("person.jpg"), backend="rtmpose", api_name="/image")
```

This Space is deployed automatically from
[`hf/space/`](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/tree/main/hf/space) in the GitHub
repository; please open issues and pull requests there. Run it locally with

```bash
git clone https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git && cd WholebodyPoseEstimation
pip install -e ".[demo,vitpose]" && python hf/space/app.py
```

- 🌐 Project page: <https://cristianlazoquispe.github.io/WholebodyPoseEstimation/>
- 🤟 Pose datasets: [CristianLazoQuispe/pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition)

## Citation

```bibtex
@inproceedings{lazo2026wholebodypose,
  title        = {WholeBodyPose: a unified end-to-end framework for sign language recognition and pose-based training data},
  author       = {Lazo-Quispe, Cristian and Castro-Cruz, Renato and Bejarano, Gissella and Salazar-Espinosa, Mauricio},
  booktitle    = {Eighteenth International Conference on Machine Vision (ICMV 2025)},
  volume       = {14114},
  pages        = {69--79},
  year         = {2026},
  organization = {SPIE},
  doi          = {10.1117/12.3096312}
}
```
