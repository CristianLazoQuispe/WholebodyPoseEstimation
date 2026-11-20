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

# 🧠 WholeBodyPose demo

Interactive demo of **[WholeBodyPose](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation)**
(ICMV 2025): whole-body pose estimation that returns the same **133 COCO-WholeBody keypoints**
(body, feet, face, both hands) from three backends.

| Backend | Model | People |
|:--|:--|:-:|
| ⚡ RTMPose | YOLOX-m + RTMW-x 384×288 | multi |
| 🔷 MediaPipe | BlazePose heavy + Hands + Face Mesh | single |
| 🎯 ViTPose+ | ViTPose+ whole-body (base on CPU, huge on GPU) | single |

Upload a video (or record one with the webcam) or an image, choose the backend, the skeleton style
(COCO-WholeBody 133 or OpenPose-style 135), the drawing options and the processing FPS, and get

- the rendered skeleton video (H.264 MP4) or image,
- the keypoints as `.npz` and `.json` (`wholebodypose.load_poses(path)` reads both),
- a short summary (frames, people, processing time).

Videos are limited to their first 15 seconds. The app can also be called programmatically:

```python
from gradio_client import Client, handle_file

client = Client("CristianLazoQuispe/WholeBodyPoseDemo")
video, files, summary = client.predict(
    video=handle_file("sign.mp4"), backend="mediapipe", processing_fps=10, api_name="/video"
)
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
- 📦 Weights mirror: [CristianLazoQuispe/wholebodypose-weights](https://huggingface.co/CristianLazoQuispe/wholebodypose-weights)

## Citation

```bibtex
@inproceedings{lazo2026wholebodypose,
  title        = {WholeBodyPose: a unified end-to-end framework for sign language recognition and pose-based training data},
  author       = {Lazo-Quispe, Cristian and Castro-Cruz, Renato and Bejarano, Gissella and Salazar-Espinosa, Mauricio},
  booktitle    = {Eighteenth International Conference on Machine Vision (ICMV 2025)},
  volume       = {14114},
  pages        = {69--79},
  year         = {2026},
  organization = {SPIE}
}
```
