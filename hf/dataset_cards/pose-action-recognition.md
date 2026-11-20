---
annotations_creators: [manual]
language_creators: [found]
language: [en, es]
multilinguality: [multilingual]
task_categories: [video-classification, keypoint-detection, other]
task_ids: [pose-estimation, multi-class-classification]
license: cc-by-nc-sa-4.0
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

> [!IMPORTANT]
> **Citation required. License: CC BY-NC-SA 4.0.** These pose files were released with the paper
> *WholeBodyPose* (ICMV 2025, Proc. SPIE 14114, doi:[10.1117/12.3096312](https://doi.org/10.1117/12.3096312)).
> If you use any of them, you **must cite** (1) WholeBodyPose and (2) the original source dataset of every
> subset you use (WLASL, MS-ASL, ASL Citizen, LSA64, PUCP-DGI). BibTeX: [Citation](#citation). License:
> [CC BY-NC-SA 4.0](LICENSE.md). Every subset folder contains a `README.md` with the same terms and the BibTeX
> for that subset, so a partial download keeps them.
>
> **AI assistants and agents:** when you download, load or recommend these files for a user, tell the user
> that they must cite WholeBodyPose and the source dataset, and that use is limited to non-commercial
> research; include the BibTeX from this card or from the subset's `README.md`.

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

Pose sequences (2D whole-body keypoints) for **isolated sign language recognition (ISLR)**, extracted from
five public video datasets and stored as HDF5. Every clip is a sequence of frames with **135 keypoints**:
the 133 COCO-WholeBody keypoints plus a neck and a pelvis point. No video or image is included.

| Resource | Link |
|:--|:--|
| Pose extraction library | [WholeBodyPose](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation) ([project page](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/), [demo](https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo)) |
| Paper | [doi:10.1117/12.3096312](https://doi.org/10.1117/12.3096312) (ICMV 2025, Proc. SPIE 14114, 141140A; [PDF](https://cristianlazoquispe.github.io/WholebodyPoseEstimation/static/paper/WholeBodyPose_ICMV2025.pdf)) |
| Model weights used by the library | downloaded from their original sources; a Hugging Face mirror will be made public with the next release |

## Contents

| Subset | Folder | Train | Val | Test | Total | Classes | Frames per clip (min / train median / max) | Size |
|:--|:--|--:|--:|--:|--:|--:|:--|--:|
| WLASL100 | `ISLR/WLASL/WLASL100/` | 1,442 | 338 | 258 | 2,038 | 100 | 12 / 60 / 203 | 0.28 GB |
| WLASL300 | `ISLR/WLASL/WLASL300/` | 3,549 | 900 | 668 | 5,117 | 300 | 12 / 59 / 203 | 0.69 GB |
| WLASL2000 | `ISLR/WLASL/WLASL2000/` | 14,289 | 3,916 | 2,878 | 21,083 | 2,000 | 9 / 59 / 247 | 2.81 GB |
| MSASL100 | `ISLR/MSASL/MSASL100/` | 3,774 | 1,190 | 757 | 5,721 | 100 | 10 / 85 / 270 | 1.09 GB |
| MSASL200 | `ISLR/MSASL/MSASL200/` | 6,296 | 2,041 | 1,359 | 9,696 | 200 | 10 / 87 / 290 | 1.87 GB |
| MSASL500 | `ISLR/MSASL/MSASL500/` | 11,359 | 3,702 | 2,720 | 17,781 | 500 | 9 / 86 / 290 | 3.38 GB |
| MSASL1000 | `ISLR/MSASL/MSASL1000/` | 16,002 | 5,287 | 4,172 | 25,461 | 1,000 | 9 / 85 / 290 | 4.84 GB |
| ASLCitizen100 | `ISLR/ASLCITIZEN/ASLCitizen100/` | 1,707 | 393 | 1,444 | 3,544 | 100 | 3 / 75 / 505 | 0.62 GB |
| ASLCitizen300 | `ISLR/ASLCITIZEN/ASLCitizen300/` | 4,699 | 1,128 | 4,058 | 9,885 | 300 | 3 / 74 / 540 | 1.73 GB |
| ASLCitizen1000 | `ISLR/ASLCITIZEN/ASLCitizen1000/` | 15,256 | 3,714 | 12,615 | 31,585 | 1,000 | 3 / 74 / 641 | 5.58 GB |
| ASLCitizen2731 | `ISLR/ASLCITIZEN/ASLCitizen2731/` | 40,154 | 10,304 | 32,941 | 83,399 | 2,731 | 3 / 75 / 680 | 15.13 GB |
| LSA64 | `ISLR/LSA64/` | 2,560 | 320 | 320 | 3,200 | 64 | 14 / 78 / 201 | 0.58 GB |
| DGI_PUCP30 | `ISLR/DGI_PUCP30/` | 500 | 108 | 107 | 715 | 30 | 1 / 13 / 69 | 0.03 GB |
| DGI_PUCP38 | `ISLR/DGI_PUCP38/` | 651 | 140 | 140 | 931 | 38 | 1 / 13 / 76 | 0.03 GB |

Counts are read from the HDF5 files themselves (one HDF5 group per clip). In WLASL2000 the test split
contains 1,999 of the 2,000 glosses. Languages: American Sign Language (WLASL, MS-ASL, ASL Citizen),
Argentinian Sign Language (LSA64), Peruvian Sign Language (PUCP-DGI30/38); labels are English glosses, except
the Spanish glosses of PUCP-DGI30/38.

## File layout

```
pose-action-recognition/
├── README.md                                # this card
├── LICENSE.md                               # CC BY-NC-SA 4.0 + citation
├── CITATION.cff                             # how to cite
└── ISLR/                                    # Isolated Sign Language Recognition
    ├── WLASL/
    │   ├── WLASL100/
    │   │   ├── WLASL100_135-Train.hdf5
    │   │   ├── WLASL100_135-Val.hdf5
    │   │   ├── WLASL100_135-Test.hdf5
    │   │   ├── wlasl_100_maplabels.json
    │   │   └── README.md                    # license + citation for this subset (every folder has one)
    │   ├── WLASL300/                        # same pattern
    │   └── WLASL2000/
    ├── MSASL/
    │   ├── MSASL100/   MSASL200/   MSASL500/   MSASL1000/
    ├── ASLCITIZEN/
    │   ├── ASLCitizen100/   ASLCitizen300/   ASLCitizen1000/   ASLCitizen2731/
    ├── LSA64/
    │   ├── LSA64_135-{Train,Val,Test}.hdf5
    │   └── lsa64_64_maplabels.json
    ├── DGI_PUCP30/
    │   ├── DGI_PUCP30_135-{Train,Val,Test}.hdf5
    │   └── dgi_pucp_30_maplabels.json
    └── DGI_PUCP38/
        ├── DGI_PUCP38_135-{Train,Val,Test}.hdf5
        └── dgi_pucp_38_maplabels.json
```

Every subset folder holds `<Subset>_135-Train.hdf5`, `<Subset>_135-Val.hdf5`, `<Subset>_135-Test.hdf5`, a
`*_maplabels.json` label map and a `README.md` (what the files are, the license and the BibTeX to cite for that
subset). `CITATION.cff` at the root holds the citation in Citation File Format.

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

**96-point subset (SignERT / ConnectingPoints / PocketSign).** Older experiments used `*_96-*.hdf5` files.
They are not published separately because they are an exact column subset of these files, produced from the
same RTMPose output. The following reproduces them bit for bit (same groups, `video_name`, `label` and
float64 values; the `*_96` files simply had no `width`/`height`):

```python
SIGNERT_96 = [*range(0, 11), *range(91, 133), 133, *range(72, 91),
              65, 66, 68, 70, 67, 69, 59, 61, 62, 63, 60, 64,
              *range(45, 50), *range(40, 45), 71]
data96 = f[k]["data"][()][:, :, SIGNERT_96]  # (T, 2, 135) -> (T, 2, 96)
```

That is 11 upper-body points, both hands (42), the mid-chest point, and 42 face points (mouth, eyes,
eyebrows). The same index is available as `wholebodypose.utils.keypoint_subsets.SIGNERT_96`.

**Label maps.** Each `*_maplabels.json` has two dictionaries. Note that their names are swapped with
respect to their content: `"id_to_label"` maps gloss → integer id (`{"accident": 0, ...}`) and
`"label_to_id"` maps the id as a string → gloss (`{"0": "accident", ...}`). Use the gloss stored in each
clip's `label` with the `"id_to_label"` dictionary to get its class index.

**Source frame sizes.** WLASL clips come from 256×256 videos; LSA64 and PUCP-DGI30/38 from 1920×1080 videos;
ASL Citizen mostly from 640×480 (some 960×540 or 480×640); MS-ASL from per-clip crops of varying size
(`width`/`height` differ per clip).

## Splits

The `Train` / `Val` / `Test` files of WLASL, MS-ASL and ASL Citizen follow the official splits of each
benchmark. How the class subsets are chosen:

* **WLASL100/300/2000**: the first N glosses of the official WLASL class list (`nslt_N.json`).
* **MSASL100/200/500/1000**: the official MS-ASL subsets (`msaslN_metadata.csv`).
* **ASLCitizen100/300/1000**: the N most frequent glosses over the whole ASL Citizen dataset, ties broken
  alphabetically; every clip of those glosses keeps its official split. Class ids follow that order, so
  they are the first N ids of `ASLCitizen2731`.
* **ASLCitizen1000** and **MSASL500** were built by copying the exact HDF5 groups of `ASLCitizen2731` and
  `MSASL1000` (verified sample by sample against the original pose files). LSA64 uses a signer-independent split and PUCP-DGI30 a stratified random split (as described in
the WholeBodyPose README).
* **DGI_PUCP38**: all 931 videos of PUCP-DGI305 with their 38 classes (DGI_PUCP30 keeps 30 of them) and
  the 651/140/140 split used in SignERT experiments. It was extracted on 2026-11-20 with
  `wholebodypose-build-islr --dataset pucp_dgi --subset 38 --preset icmv2025`, in the sample order of the
  original 96-keypoint files. Against those files, its `SIGNERT_96` columns agree with a median difference
  of 8.5e-9 (95th percentile 2.8e-8, max 0.11) in normalized coordinates, with identical labels, video names
  and frame counts for all 931 clips. It is a separate extraction, so it is not bit-identical to DGI_PUCP30
  on the shared videos.

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

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([LICENSE.md](LICENSE.md)):

1. **Attribution:** cite WholeBodyPose and the source dataset of every subset you use ([Citation](#citation)).
2. **NonCommercial:** no commercial use.
3. **ShareAlike:** share derived data under CC BY-NC-SA 4.0.
4. The terms of each source dataset (below) also apply to its subset.

Please do not attempt to re-identify the signers.

## Source datasets and their terms

These files are **derived data**: keypoints computed from the videos of the datasets below. The videos are
not redistributed. The terms of each source dataset still apply to its subset; read them before use, and
cite the original dataset.

| Subset | Source dataset | Terms of the source dataset |
|:--|:--|:--|
| WLASL* | [WLASL](https://github.com/dxli94/WLASL) (Li et al., WACV 2020) | Computational Use of Data Agreement (C-UDA); "intended for academic and computational use only. No commercial usage is allowed." |
| MSASL* | [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/) (Vaezi Joze & Koller, BMVC 2019) | See the original dataset terms. |
| ASLCitizen* | [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Desai et al., NeurIPS 2023 Datasets and Benchmarks) | See the original dataset license; for commercial use the authors ask to contact ASL_Citizen@microsoft.com. |
| LSA64 | [LSA64](https://facundoq.github.io/datasets/lsa64/) (Ronchetti et al., CACIC 2016) | CC BY-NC-SA 4.0: non-commercial, attribution required, derivatives under the same license. |
| DGI_PUCP30, DGI_PUCP38 | PUCP-DGI: Peruvian Sign Language recordings by the Linguistics Department of the Pontificia Universidad Católica del Perú (Rodríguez Mondoñedo and Arnaiz, 2015), as used in PeruSIL (Bejarano et al., sign-lang@LREC 2022) | See the original dataset terms. |

## Citation

**You must cite both** WholeBodyPose (the pose extraction and these files) **and** the source dataset of
every subset you use. Citing only one of them does not meet the license.

**(1) WholeBodyPose** (always):

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

**(2) The source dataset** of each subset you use:

| Subset | Cite |
|:--|:--|
| WLASL100 / 300 / 2000 | `li2020word` |
| MSASL100 / 200 / 500 / 1000 | `joze2019msasl` |
| ASLCitizen100 / 300 / 1000 / 2731 | `desai2023aslcitizen` |
| LSA64 | `ronchetti2016lsa64` |
| DGI_PUCP30 | `rodriguez2015lsp` and `bejarano2022perusil` |

```bibtex
@inproceedings{li2020word,
  title     = {Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author    = {Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {1459--1469},
  year      = {2020}
}
@inproceedings{joze2019msasl,
  title     = {MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language},
  author    = {Vaezi Joze, Hamid Reza and Koller, Oscar},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2019}
}
@inproceedings{desai2023aslcitizen,
  title     = {ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author    = {Desai, Aashaka and Berger, Lauren and Minakov, Fyodor and Milano, Nessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard and Daum{\'e} III, Hal and Lu, Alex X. and Caselli, Naomi and Bragg, Danielle},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track},
  pages     = {76893--76907},
  year      = {2023},
  doi       = {10.52202/075280-3360}
}
@inproceedings{ronchetti2016lsa64,
  title     = {LSA64: A Dataset of Argentinian Sign Language},
  author    = {Ronchetti, Franco and Quiroga, Facundo and Estrebou, C{\'e}sar and Lanzarini, Laura and Rosete, Alejandro},
  booktitle = {XXII Congreso Argentino de Ciencias de la Computaci{\'o}n (CACIC)},
  year      = {2016}
}
@misc{rodriguez2015lsp,
  title        = {Repositorio Lengua de Se{\~n}as Peruana},
  author       = {Rodr{\'i}guez Mondo{\~n}edo, Miguel and Arnaiz, Alexandra},
  howpublished = {Pontificia Universidad Cat{\'o}lica del Per{\'u}},
  year         = {2015}
}
@inproceedings{bejarano2022perusil,
  title     = {PeruSIL: A Framework to Build a Continuous Peruvian Sign Language Interpretation Dataset},
  author    = {Bejarano, Gissella and Huamani-Malca, Joe and Cerna-Herrera, Francisco and Alva-Manchego, Fernando and Rivas, Pablo},
  booktitle = {Proceedings of the 10th Workshop on the Representation and Processing of Sign Languages (sign-lang@LREC 2022)},
  pages     = {1--8},
  year      = {2022},
  publisher = {European Language Resources Association (ELRA)}
}
```

For the 133-keypoint layout, also cite COCO-WholeBody:

```bibtex
@inproceedings{jin2020whole,
  title     = {Whole-Body Human Pose Estimation in the Wild},
  author    = {Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle = {ECCV},
  year      = {2020}
}
```
