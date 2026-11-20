"""Static texts and styling of the WholeBodyPose demo (kept apart from the app logic)."""

REPO_URL = "https://github.com/CristianLazoQuispe/WholebodyPoseEstimation"
PAGE_URL = "https://cristianlazoquispe.github.io/WholebodyPoseEstimation/"
PAPER_URL = f"{PAGE_URL}static/paper/WholeBodyPose_ICMV2025.pdf"
DATASETS_URL = "https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition"
DATASET_DOCS_URL = f"{REPO_URL}/tree/main/docs/datasets"

CSS = """
.gradio-container { max-width: 1280px !important; margin: auto !important; }
#wbp-header { text-align: center; padding: 8px 0 4px; }
#wbp-header h1 { font-size: 1.9rem; margin: 0 0 6px; }
#wbp-header p { margin: 0 auto 10px; max-width: 820px; color: var(--body-text-color); opacity: .8; }
#wbp-links { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
#wbp-links a { padding: 5px 14px; border-radius: 999px; border: 1px solid var(--border-color-primary);
               color: var(--body-text-color); text-decoration: none; font-size: 0.875rem; font-weight: 600; }
#wbp-links a:hover { border-color: var(--color-accent); color: var(--color-accent); }
.summary { min-height: 1.5em; }
.about { max-width: 980px; margin: 0 auto; min-width: 0; }
.about table { font-size: 0.875rem; display: block; overflow-x: auto; border: none !important;
               max-width: calc(100vw - 80px); }
.about pre { overflow-x: auto; max-width: calc(100vw - 80px); }
.about td, .about th { padding: 4px 8px !important; min-width: 5.5em; overflow-wrap: normal; word-break: normal; }
.about th, .about td:first-child { white-space: nowrap; }
"""

HEADER_HTML = f"""
<div id="wbp-header">
  <h1>WholeBodyPose</h1>
  <p>Whole-body pose estimation: RTMPose, ViTPose+ and MediaPipe return the same 133 COCO-WholeBody
  keypoints (body, feet, face, hands). Upload a video or an image and download the keypoints.</p>
  <div id="wbp-links">
    <a href="{PAPER_URL}" target="_blank" rel="noopener">Paper (ICMV 2025)</a>
    <a href="{PAGE_URL}" target="_blank" rel="noopener">Project page</a>
    <a href="{REPO_URL}" target="_blank" rel="noopener">Code</a>
    <a href="{DATASETS_URL}" target="_blank" rel="noopener">Pose datasets</a>
  </div>
</div>
"""

# (subset, folder on the dataset repo, sign language, classes, clips): counts from the dataset card,
# which reads them from the HDF5 files.
DATASETS = [
    ("WLASL100", "ISLR/WLASL/WLASL100", "ASL", 100, 2038),
    ("WLASL300", "ISLR/WLASL/WLASL300", "ASL", 300, 5117),
    ("WLASL2000", "ISLR/WLASL/WLASL2000", "ASL", 2000, 21083),
    ("MSASL100", "ISLR/MSASL/MSASL100", "ASL", 100, 5721),
    ("MSASL200", "ISLR/MSASL/MSASL200", "ASL", 200, 9696),
    ("MSASL500", "ISLR/MSASL/MSASL500", "ASL", 500, 17781),
    ("MSASL1000", "ISLR/MSASL/MSASL1000", "ASL", 1000, 25461),
    ("ASLCitizen100", "ISLR/ASLCITIZEN/ASLCitizen100", "ASL", 100, 3544),
    ("ASLCitizen300", "ISLR/ASLCITIZEN/ASLCitizen300", "ASL", 300, 9885),
    ("ASLCitizen1000", "ISLR/ASLCITIZEN/ASLCitizen1000", "ASL", 1000, 31585),
    ("ASLCitizen2731", "ISLR/ASLCITIZEN/ASLCitizen2731", "ASL", 2731, 83399),
    ("LSA64", "ISLR/LSA64", "Argentinian SL", 64, 3200),
    ("DGI_PUCP30", "ISLR/DGI_PUCP30", "Peruvian SL", 30, 715),
]


def _dataset_rows() -> str:
    rows = []
    for name, folder, lang, classes, clips in DATASETS:
        splits = " · ".join(
            f"[{split}]({DATASETS_URL}/blob/main/{folder}/{name}_135-{split}.hdf5)"
            for split in ("Train", "Val", "Test")
        )
        rows.append(f"| [{name}]({DATASETS_URL}/tree/main/{folder}) | {lang} | {classes:,} | {clips:,} | {splits} |")
    return "\n".join(rows)


# (model, paper reference, original, + WholeBodyPose, delta, code link): ICMV 2025 paper, Table 3.
ISLR_RESULTS = [
    ("Pose-TGCN", "55.43", "58.14", "+2.71", "[dxli94/WLASL](https://github.com/dxli94/WLASL/tree/master/code/TGCN)"),
    ("GCN-BERT", "60.15", "66.67", "+6.52", "[paper](https://arxiv.org/abs/2012.00781) (no code release found)"),
    (
        "SILT",
        "62.79",
        "66.67",
        "+3.88",
        "[cristinalunaj/InterpretableTransformer_SignLanguage]"
        "(https://github.com/cristinalunaj/InterpretableTransformer_SignLanguage)",
    ),
    ("SPOTER", "63.18", "67.05", "+3.87", "[maty-bohacek/spoter](https://github.com/maty-bohacek/spoter)"),
]


def _results_rows() -> str:
    return "\n".join(f"| {m} | {a} | **{b}** | {d} | {code} |" for m, a, b, d, code in ISLR_RESULTS)


ABOUT_MD = f"""
## What this demo does

Video frames are resized to 640 px wide (paper setting) and sampled at the chosen processing FPS; images are
used as they are. All backends return the same **133 COCO-WholeBody keypoints** per person (17 body, 6 feet,
68 face, 42 hands) in pixel coordinates of the processed frame, with a score per keypoint (0 when not detected).

| Backend | Model | People | Notes |
|:--|:--|:-:|:--|
| RTMPose | YOLOX-m + RTMW-l 384×288 | multi | Paper setting |
| RTMPose-lite | YOLOX-tiny + RTMW-m 256×192 | multi | About 2–3× faster than RTMW-l 384 on 2 CPU cores in our tests; default for videos on CPU |
| MediaPipe | BlazePose heavy + Hands + Face Mesh | single | Fastest on CPU |
| ViTPose+ | ViTPose+ whole-body (base on CPU, huge on GPU) | single | Full-frame transformer |

With *multi-person* off (and for single-person backends) the largest person of each frame is kept.
**Downloads**: `<input>_<backend>.npz` and `.json` hold `keypoints` `(T, 133, 2)` and `scores` `(T, 133)`
(ragged `(T, N, 133, 2)` in multi-person mode) plus the FPS and frame size; read them with
`wholebodypose.load_poses(path)`. The *OpenPose-style 135* option only changes the drawing (it adds a neck
and a pelvis point).

## Sign language recognition results (paper)

Top-1 per-instance accuracy (%) on WLASL100, models trained from scratch on pose inputs only, original
pose input vs. WholeBodyPose (RTMPose, 135 keypoints). The ISLR models are not part of this demo and no
checkpoints are released; the links point to their original code.

| Model | Original | + WholeBodyPose | Δ | Code |
|:--|--:|--:|--:|:--|
{_results_rows()}

## Pose datasets

The pose sequences used in the paper are public at
[CristianLazoQuispe/pose-action-recognition]({DATASETS_URL}): one HDF5 file per split with
135 keypoints per frame (COCO-WholeBody 133 + neck + pelvis), coordinates normalized to `[0, 1]`.

| Subset | Language | Classes | Clips | Splits |
|:--|:--|--:|--:|:--|
{_dataset_rows()}

The 96-keypoint subset used in older experiments is a column subset of these files
(`SIGNERT_96` in `wholebodypose.utils.keypoint_subsets`). To rebuild the datasets from the videos, see the
[dataset docs]({DATASET_DOCS_URL}) (`wholebodypose-build-islr`).

## Use it in Python

```bash
pip install "wholebodypose[all] @ git+{REPO_URL}.git"
```

```python
from wholebodypose import PoseEstimator, save_poses

est = PoseEstimator("rtmpose")  # or "vitpose", "mediapipe"
res = est.process_video("sign.mp4", "outputs/", video_processing_fps=15)
save_poses("sign.npz", res["total_keypoints"], res["total_scores"], fps=res["processing_fps"])
```

## Citation

```bibtex
@inproceedings{{lazo2026wholebodypose,
  title        = {{WholeBodyPose: a unified end-to-end framework for sign language recognition and pose-based training data}},
  author       = {{Lazo-Quispe, Cristian and Castro-Cruz, Renato and Bejarano, Gissella and Salazar-Espinosa, Mauricio}},
  booktitle    = {{Eighteenth International Conference on Machine Vision (ICMV 2025)}},
  volume       = {{14114}},
  pages        = {{69--79}},
  year         = {{2026}},
  organization = {{SPIE}},
  doi          = {{10.1117/12.3096312}}
}}
```

Please also cite the backend you use (RTMW, ViTPose++ or MediaPipe); see [docs/guide/backends.md]({REPO_URL}/blob/main/docs/guide/backends.md#citing-the-backends).

**Authors**: Cristian Lazo-Quispe, Renato Castro-Cruz (Universidad Nacional de Ingeniería, Peru),
Gissella Bejarano (Marist University, USA), Mauricio Salazar-Espinosa (Pontificia Universidad Católica del Perú).
"""

FOOTER_MD = f"""
<div align="center" style="opacity:.7; font-size:.875rem">

WholeBodyPose · [Report an issue]({REPO_URL}/issues)

</div>
"""
