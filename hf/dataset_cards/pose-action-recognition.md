---
annotations_creators: [manual]
language_creators: [found]
language: [en, es]
multilinguality: [multilingual]
task_categories: [video-classification, keypoint-detection, other]
task_ids: [pose-estimation, multi-class-classification]
license: other
license_name: research-only
license_link: LICENSE.md
pretty_name: Pose-Based Action Recognition (135 keypoints)
size_categories: [100K<n<1M]
tags:
- sign-language-recognition
- isolated-sign-language-recognition
- pose-estimation
- whole-body
- coco-wholebody
- skeleton
- hdf5
---

# Pose-Based Action & Sign Language Datasets (135 keypoints)

Pose sequences (2D whole-body keypoints) for **isolated sign language recognition (ISLR)**, extracted from
six public video benchmarks and stored as HDF5. Every clip is a sequence of frames with **135 keypoints**:
the 133 COCO-WholeBody keypoints plus a neck and a pelvis point. No video or image is included.

| Resource | Link |
|:--|:--|
| Pose extraction library | [WholeBodyPose](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation) ([project page](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/), [demo](https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo)) |
| Model weights used by the library | [CristianLazoQuispe/wholebodypose-weights](https://huggingface.co/CristianLazoQuispe/wholebodypose-weights) |

## Contents

| Subset | Folder | Train | Val | Test | Total | Classes | Frames per clip (min / train median / max) | Size |
|:--|:--|--:|--:|--:|--:|--:|:--|--:|
| WLASL100 | `ISLR/WLASL/WLASL100/` | 1,442 | 338 | 258 | 2,038 | 100 | 12 / 60 / 203 | 0.28 GB |
| WLASL300 | `ISLR/WLASL/WLASL300/` | 3,549 | 900 | 668 | 5,117 | 300 | 12 / 59 / 203 | 0.69 GB |
| WLASL2000 | `ISLR/WLASL/WLASL2000/` | 14,289 | 3,916 | 2,878 | 21,083 | 2,000 | 9 / 59 / 247 | 2.81 GB |
| MSASL100 | `ISLR/MSASL/MSASL100/` | 3,774 | 1,190 | 757 | 5,721 | 100 | 10 / 85 / 270 | 1.09 GB |
| MSASL200 | `ISLR/MSASL/MSASL200/` | 6,296 | 2,041 | 1,359 | 9,696 | 200 | 10 / 87 / 290 | 1.87 GB |
| MSASL1000 | `ISLR/MSASL/MSASL1000/` | 16,002 | 5,287 | 4,172 | 25,461 | 1,000 | 9 / 85 / 290 | 4.84 GB |
| ASLCitizen100 | `ISLR/ASLCITIZEN/ASLCitizen100/` | 1,707 | 393 | 1,444 | 3,544 | 100 | 3 / 75 / 505 | 0.62 GB |
| ASLCitizen300 | `ISLR/ASLCITIZEN/ASLCitizen300/` | 4,699 | 1,128 | 4,058 | 9,885 | 300 | 3 / 74 / 540 | 1.73 GB |
| ASLCitizen2731 | `ISLR/ASLCITIZEN/ASLCitizen2731/` | 40,154 | 10,304 | 32,941 | 83,399 | 2,731 | 3 / 75 / 680 | 15.13 GB |
| LSA64 | `ISLR/LSA64/` | 2,560 | 320 | 320 | 3,200 | 64 | 14 / 78 / 201 | 0.58 GB |
| DGI_PUCP30 | `ISLR/DGI_PUCP30/` | 500 | 108 | 107 | 715 | 30 | 1 / 13 / 69 | 0.03 GB |

Counts are read from the HDF5 files themselves (one HDF5 group per clip). In WLASL2000 the test split
contains 1,999 of the 2,000 glosses. Languages: American Sign Language (WLASL, MS-ASL, ASL Citizen),
Argentinian Sign Language (LSA64), Peruvian Sign Language (PUCP-DGI30); labels are English glosses, except
the Spanish glosses of PUCP-DGI30.

## File layout

```
pose-action-recognition/
└── ISLR/                                    # Isolated Sign Language Recognition
    ├── WLASL/
    │   ├── WLASL100/
    │   │   ├── WLASL100_135-Train.hdf5
    │   │   ├── WLASL100_135-Val.hdf5
    │   │   ├── WLASL100_135-Test.hdf5
    │   │   └── wlasl_100_maplabels.json
    │   ├── WLASL300/                        # same pattern
    │   └── WLASL2000/
    ├── MSASL/
    │   ├── MSASL100/   MSASL200/   MSASL1000/
    ├── ASLCITIZEN/
    │   ├── ASLCitizen100/   ASLCitizen300/   ASLCitizen2731/
    ├── LSA64/
    │   ├── LSA64_135-{Train,Val,Test}.hdf5
    │   └── lsa64_64_maplabels.json
    └── DGI_PUCP30/
        ├── DGI_PUCP30_135-{Train,Val,Test}.hdf5
        └── dgi_pucp_30_maplabels.json
```

Every subset folder holds `<Subset>_135-Train.hdf5`, `<Subset>_135-Val.hdf5`, `<Subset>_135-Test.hdf5` and a
`*_maplabels.json` label map.

## Data format

Each HDF5 file has one group per clip, named `"0"`, `"1"`, … (string indices). Each group contains:

| Dataset | Shape / type | Content |
|:--|:--|:--|
| `data` | `(T, 2, 135)`, `float64` | x and y of the 135 keypoints for each of the `T` frames (note the axis order: coordinates before keypoints) |
| `label` | scalar, bytes | gloss of the clip, e.g. `b"accident"` |
| `video_name` | scalar, bytes | path of the source video on the machine that processed it (identifies the clip in the original dataset) |
| `width`, `height` | scalar, `int64` | frame size of the source video in pixels |

**Coordinates.** x and y are normalized by the frame width and height (values lie in `[0, 1]`, with a few
slightly above 1 for points extrapolated outside the frame). Multiply by `width` / `height` to get pixels.
A keypoint that was not detected is stored as `(0, 0)`. No confidence scores are stored.

**Keypoints.** Indices `0–132` follow COCO-WholeBody order: `0–16` body, `17–22` feet, `23–90` face,
`91–111` left hand, `112–132` right hand. The two extra points are:

* `133` neck = midpoint of the shoulders (`5`, `6`);
* `134` pelvis = midpoint of the hips (`11`, `12`).

We checked this on the files: in every frame where both shoulders (hips) are detected, point 133 (134)
equals their midpoint exactly.

**Label maps.** Each `*_maplabels.json` has two dictionaries. Note that their names are swapped with
respect to their content: `"id_to_label"` maps gloss → integer id (`{"accident": 0, ...}`) and
`"label_to_id"` maps the id as a string → gloss (`{"0": "accident", ...}`). Use the gloss stored in each
clip's `label` with the `"id_to_label"` dictionary to get its class index.

**Source frame sizes.** WLASL clips come from 256×256 videos; LSA64 and PUCP-DGI30 from 1920×1080 videos;
ASL Citizen mostly from 640×480 (some 960×540 or 480×640); MS-ASL from per-clip crops of varying size
(`width`/`height` differ per clip).

## Splits

The `Train` / `Val` / `Test` files of WLASL, MS-ASL and ASL Citizen follow the official splits of each
benchmark. LSA64 uses a signer-independent split and PUCP-DGI30 a stratified random split (as described in
the WholeBodyPose README).

## Loading

```python
import json

import h5py
import numpy as np
from huggingface_hub import hf_hub_download

repo = "CristianLazoQuispe/pose-action-recognition"
h5_path = hf_hub_download(repo, "ISLR/WLASL/WLASL100/WLASL100_135-Train.hdf5", repo_type="dataset")
map_path = hf_hub_download(repo, "ISLR/WLASL/WLASL100/wlasl_100_maplabels.json", repo_type="dataset")
gloss_to_id = json.load(open(map_path))["id_to_label"]  # gloss -> id (see "Label maps")

samples = []
with h5py.File(h5_path, "r") as f:
    for key in f:  # "0", "1", ...
        g = f[key]
        xy = g["data"][()].transpose(0, 2, 1)  # (T, 135, 2), normalized x, y
        gloss = g["label"][()].decode()
        size = (int(g["width"][()]), int(g["height"][()]))
        samples.append((xy.astype(np.float32), gloss_to_id[gloss], size))

xy, y, (w, h) = samples[0]
pixels = xy * np.array([w, h], dtype=np.float32)  # pixel coordinates (undetected points stay at 0, 0)
coco133 = xy[:, :133]  # drop neck and pelvis -> plain COCO-WholeBody
```

To download a whole subset:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "CristianLazoQuispe/pose-action-recognition",
    repo_type="dataset",
    allow_patterns=["ISLR/LSA64/*"],
    local_dir="pose-action-recognition",
)
```

## License: research use only

These pose files are released **for non-commercial academic research only**. They were produced for an
academic paper (WholeBodyPose, ICMV 2025) from sign language video datasets whose own terms forbid
commercial use (e.g. WLASL's C-UDA and LSA64's CC BY-NC-SA 4.0). By downloading them you agree to:

1. use them only for non-commercial research and teaching;
2. cite WholeBodyPose and the original source dataset of every subset you use;
3. follow the terms of each source dataset (below), which take precedence for its subset;
4. not redistribute the files commercially or try to re-identify the signers.

## Source datasets and their terms

These files are **derived data**: keypoints computed from the videos of the datasets below. The videos are
not redistributed. The terms of each source dataset still apply to its subset; read them before use, and
cite the original dataset.

| Subset | Source dataset | Terms of the source dataset |
|:--|:--|:--|
| WLASL* | [WLASL](https://github.com/dxli94/WLASL) (Li et al., WACV 2020) | Computational Use of Data Agreement (C-UDA); "intended for academic and computational use only. No commercial usage is allowed." |
| MSASL* | [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/) (Vaezi Joze & Koller, BMVC 2019) | See the original dataset terms. |
| ASLCitizen* | [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Desai et al., 2023) | See the original dataset license; for commercial use the authors ask to contact ASL_Citizen@microsoft.com. |
| LSA64 | [LSA64](https://facundoq.github.io/datasets/lsa64/) (Ronchetti et al., CACIC 2016) | CC BY-NC-SA 4.0: non-commercial, attribution required, derivatives under the same license. |
| DGI_PUCP30 | PUCP-DGI30 (Pontificia Universidad Católica del Perú) | See the original dataset terms. |

## Citation

If you use this dataset structure:

```bibtex
@misc{signert2025,
  title={SignERT: Efficient Transformer-based Recognition with 135 Keypoints},
  author={Lazo, Cristian and Renato Castro},
  year={2025},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

If you use the pose extraction pipeline:

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

Please also cite the source datasets you use:

```bibtex
@inproceedings{li2020word,
  title     = {Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author    = {Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle = {The IEEE Winter Conference on Applications of Computer Vision},
  pages     = {1459--1469},
  year      = {2020}
}
@inproceedings{joze2019msasl,
  title     = {MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language},
  author    = {Vaezi Joze, Hamid Reza and Koller, Oscar},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2019}
}
@article{desai2023aslcitizen,
  title   = {ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author  = {Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O. and Milan, Vanessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard E. and Daum{\'e} III, Hal and Lu, Alex X. and Caselli, Naomi and Bragg, Danielle},
  journal = {arXiv preprint arXiv:2304.05934},
  year    = {2023}
}
@inproceedings{ronchetti2016lsa64,
  title     = {LSA64: A Dataset of Argentinian Sign Language},
  author    = {Ronchetti, Franco and Quiroga, Facundo and Estrebou, C{\'e}sar and Lanzarini, Laura and Rosete, Alejandro},
  booktitle = {XXII Congreso Argentino de Ciencias de la Computaci{\'o}n (CACIC)},
  year      = {2016}
}
@inproceedings{jin2020whole,
  title     = {Whole-Body Human Pose Estimation in the Wild},
  author    = {Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle = {ECCV},
  year      = {2020}
}
```
