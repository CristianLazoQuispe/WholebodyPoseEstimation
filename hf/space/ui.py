"""Static texts and styling of the WholeBodyPose demo (kept apart from the app logic)."""

REPO_URL = "https://github.com/CristianLazoQuispe/WholebodyPoseEstimation"
PAGE_URL = "https://cristianlazoquispe.github.io/WholebodyPoseEstimation/"
PAPER_URL = f"{PAGE_URL}static/paper/WholeBodyPose_ICMV2025.pdf"
DATASETS_URL = "https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition"
WEIGHTS_URL = "https://huggingface.co/CristianLazoQuispe/wholebodypose-weights"

CSS = """
.gradio-container { max-width: 1400px !important; margin: auto !important; }
#header, #badges { text-align: center; }
.badge { display: inline-block; padding: 5px 12px; margin: 3px; border-radius: 20px;
         font-size: 12px; font-weight: 600; color: white; text-decoration: none; }
"""

HEADER_MD = """
# 🧠 WholeBodyPose: Unified Whole-Body Pose Estimation

**RTMPose · ViTPose+ · MediaPipe → the same 133 COCO-WholeBody keypoints (body, feet, face, hands)**

Upload a video or an image (or record with your webcam), pick a backend and download the keypoints.
"""


def _badge(label: str, url: str, color: str, text_color: str = "white") -> str:
    return (
        f'<a class="badge" style="background:{color};color:{text_color}" href="{url}" '
        f'target="_blank" rel="noopener">{label}</a>'
    )


BADGES_HTML = (
    '<div id="badges">'
    + _badge("📄 ICMV 2025 paper", PAPER_URL, "#b31b1b")
    + _badge("🌐 Project page", PAGE_URL, "#6f42c1")
    + _badge("💻 GitHub", REPO_URL, "#24292f")
    + _badge("🤟 Pose datasets", DATASETS_URL, "#ffcc00", "#333")
    + _badge("📦 Model weights", WEIGHTS_URL, "#ff9800")
    + "</div>"
)

MODEL_INFO_MD = """
| Backend | Model | People | Notes |
|:--|:--|:-:|:--|
| ⚡ **RTMPose** | YOLOX-m + RTMW-x 384×288 | multi | Detector + whole-body model, paper default |
| 🔷 **MediaPipe** | BlazePose heavy + Hands + Face Mesh | single | Lightweight, fastest on CPU |
| 🎯 **ViTPose+** | ViTPose+ whole-body (base on CPU, huge on GPU) | single | Full-frame transformer, largest model |

Single-person backends (and RTMPose with *multi-person* off) keep the largest person in each frame.
**OpenPose-135** only changes the drawing: it adds neck and pelvis points; the downloaded keypoints are always
COCO-WholeBody 133 `(T, 133, 2)` pixel coordinates in the processed frame (640 px wide) with scores `(T, 133)`.
Load them with `wholebodypose.load_poses(path)`.
"""

ABOUT_MD = """
## 🌟 Why WholeBodyPose?

| | |
|:--|:--|
| ⚡ **3 backends, 1 format** | RTMPose, ViTPose+ and MediaPipe all return COCO-WholeBody 133 keypoints |
| 🔄 **Reproducible** | One pipeline from raw video to training-ready pose sequences, weights mirrored on the Hub |
| 📈 **Better recognition** | Up to +6.5% accuracy on isolated sign language recognition benchmarks |
| 🌍 **Multilingual data** | Pose sequences for American 🇺🇸, Argentinian 🇦🇷 and Peruvian 🇵🇪 sign language datasets |

### 📊 Benchmark (WLASL100, top-1 accuracy)

| Model | Original | + WholeBodyPose | Δ |
|:--|:-:|:-:|:-:|
| SPOTER | 63.18% | **67.05%** | +3.87 |
| GCN-BERT | 60.15% | **66.67%** | +6.52 |
| SILT | 62.79% | **66.67%** | +3.88 |
"""

CITATION_MD = f"""
### 📦 Use it in Python

```bash
pip install "wholebodypose[all] @ git+{REPO_URL}.git"
```

```python
from wholebodypose import PoseEstimator, save_poses

est = PoseEstimator("rtmpose")                                    # or "vitpose", "mediapipe"
res = est.process_video("sign.mp4", "outputs/", video_processing_fps=15)
save_poses("sign.npz", res["total_keypoints"], res["total_scores"], fps=res["processing_fps"])
```

### 🤟 Pose datasets

Pose sequences extracted with WholeBodyPose for WLASL100, WLASL300, MSASL100, ASL-Citizen, LSA64 and
PUCP-DGI30 are available at [CristianLazoQuispe/pose-action-recognition]({DATASETS_URL}).

### 📚 Citation

```bibtex
@inproceedings{{lazo2026wholebodypose,
  title        = {{WholeBodyPose: a unified end-to-end framework for sign language recognition and pose-based training data}},
  author       = {{Lazo-Quispe, Cristian and Castro-Cruz, Renato and Bejarano, Gissella and Salazar-Espinosa, Mauricio}},
  booktitle    = {{Eighteenth International Conference on Machine Vision (ICMV 2025)}},
  volume       = {{14114}},
  pages        = {{69--79}},
  year         = {{2026}},
  organization = {{SPIE}}
}}
```

Please also cite the backend you use (RTMW, ViTPose++ or MediaPipe); see [docs/guide/backends.md]({REPO_URL}/blob/main/docs/guide/backends.md#citing-the-backends).

**Authors**: Cristian Lazo-Quispe, Renato Castro-Cruz (Universidad Nacional de Ingeniería, Peru),
Gissella Bejarano (Marist University, USA), Mauricio Salazar-Espinosa (Pontificia Universidad Católica del Perú).
"""

FOOTER_MD = f"""
<div align="center" style="color:#888">

Made with ❤️ for the sign language AI community · Questions or bugs? [Open an issue]({REPO_URL}/issues)

</div>
"""
