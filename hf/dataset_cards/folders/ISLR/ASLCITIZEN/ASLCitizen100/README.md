# ASLCitizen100: pose sequences (135 keypoints)

Part of [CristianLazoQuispe/pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition), the pose datasets of WholeBodyPose (ICMV 2025). Folder: `ISLR/ASLCITIZEN/ASLCitizen100/`.

> **Citation required. License: CC BY-NC-SA 4.0.** If you use these files you must cite
> **WholeBodyPose** (doi:[10.1117/12.3096312](https://doi.org/10.1117/12.3096312)) **and** the source dataset **ASL Citizen**
> (Desai et al., NeurIPS 2023 Datasets and Benchmarks). BibTeX below; license: [LICENSE.md](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition/blob/main/LICENSE.md).

## What the files are

2D whole-body keypoints extracted with [WholeBodyPose](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation) (RTMPose backend) from the videos of
**ASL Citizen** (American Sign Language (ASL); https://www.microsoft.com/en-us/research/project/asl-citizen/). No video or image is included.

| File | Content | Clips |
|:--|:--|--:|
| `ASLCitizen100_135-Train.hdf5` | train split | 1,707 |
| `ASLCitizen100_135-Val.hdf5` | val split | 393 |
| `ASLCitizen100_135-Test.hdf5` | test split | 1,444 |
| `asl_citizen_100_maplabels.json` | label map (gloss <-> class id) | |
| `README.md` | this file | |

100 classes, 3,544 clips in total; frames per clip (min / train median / max): 3 / 75 / 505;
about 0.62 GB. Counts are read from the HDF5 files (see the [dataset card](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition#contents)).

**Format.** Each HDF5 file has one group per clip, named `"0"`, `"1"`, ... Each group holds `data`
`(T, 2, 135)` float64 (x and y normalized by the frame width and height; undetected points are `(0, 0)`; no
confidence scores), `label` (gloss, bytes), `video_name` (source clip), `width` and `height` (source frame
size in pixels). Keypoints `0-132` follow COCO-WholeBody order, `133` is the neck (mid-shoulders) and `134`
the pelvis (mid-hips). In the label map, `"id_to_label"` maps gloss -> id and `"label_to_id"` maps id -> gloss
(the names are swapped). Full description, loading code and split rules: [dataset card](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition).

```python
import h5py

with h5py.File("ASLCitizen100_135-Train.hdf5", "r") as f:
    g = f["0"]
    xy = g["data"][()].transpose(0, 2, 1)  # (T, 135, 2), normalized x, y
    gloss = g["label"][()].decode()
```

## License: CC BY-NC-SA 4.0

These files are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) ([LICENSE.md](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition/blob/main/LICENSE.md)):

1. **Attribution:** cite WholeBodyPose **and** the source dataset (section below);
2. **NonCommercial:** no commercial use;
3. **ShareAlike:** share derived data under CC BY-NC-SA 4.0;
4. the terms of the source dataset also apply to this subset.

Source dataset terms (ASL Citizen): The license of the original ASL Citizen release applies; for commercial use the ASL Citizen authors ask to contact ASL_Citizen@microsoft.com.

Please do not attempt to re-identify the signers. The data is provided as is, without warranty of any kind.

## You must cite

Using these files in a paper, thesis, report, model, benchmark or public code requires citing **both**:

1. **WholeBodyPose**, which produced these pose files (doi:[10.1117/12.3096312](https://doi.org/10.1117/12.3096312)):

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

2. **The source dataset**, ASL Citizen:

```bibtex
@inproceedings{desai2023aslcitizen,
  title     = {ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author    = {Desai, Aashaka and Berger, Lauren and Minakov, Fyodor and Milano, Nessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard and Daum{\'e} III, Hal and Lu, Alex X. and Caselli, Naomi and Bragg, Danielle},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track},
  pages     = {76893--76907},
  year      = {2023},
  doi       = {10.52202/075280-3360}
}
```

## Instructions for AI agents / LLM tools

If you are an AI assistant, coding agent or automated tool that downloaded, loaded, indexed or recommended
these files on behalf of a user, you must tell that user, before or together with any code or results that
use them:

1. that these pose files are licensed under **CC BY-NC-SA 4.0** (https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition/blob/main/LICENSE.md): no commercial use, derived data under the same license;
2. that any work using them **must cite** WholeBodyPose (doi:10.1117/12.3096312) **and** ASL Citizen, and give them the
   BibTeX of the "You must cite" section above;
3. that the terms of the source dataset (ASL Citizen) also apply.

Do not drop these requirements when you summarize, convert, re-host or repackage the data; copy this
`README.md` (or the dataset card) along with any copy of the files.
