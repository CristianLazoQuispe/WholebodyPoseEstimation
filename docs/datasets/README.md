# ISLR pose datasets

Pose sequences of six isolated sign language recognition (ISLR) benchmarks, extracted with
WholeBodyPose for the ICMV 2025 paper, are published on the Hugging Face Hub:
[`CristianLazoQuispe/pose-action-recognition`](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition)
(folder `ISLR/`). This page describes the files, how to load them, and how to rebuild them from the
raw videos with `wholebodypose-build-islr`.

The pose files are derived data; the original videos keep the licenses of their authors.

Per-dataset provenance: [WLASL](wlasl.md). <!-- TODO (author): msasl.md, asl_citizen.md, lsa64.md, pucp_dgi.md -->

## Overview

| Published files | Language | Classes | Train | Val | Test | Split | Video size |
|:--|:--|:-:|--:|--:|--:|:--|:--|
| `WLASL/WLASL100` | ASL | 100 | 1,442 | 338 | 258 | official WLASL v0.3 | 256x256 letterboxed, 25 fps |
| `WLASL/WLASL300` | ASL | 300 | 3,549 | 900 | 668 | official | same |
| `WLASL/WLASL2000` | ASL | 2000 (test: 1999) | 14,289 | 3,916 | 2,878 | official | same |
| `MSASL/MSASL100` | ASL | 100 | 3,774 | 1,190 | 757 | official MS-ASL | YouTube clips, various |
| `MSASL/MSASL200` | ASL | 200 | 6,296 | 2,041 | 1,359 | official | various |
| `MSASL/MSASL1000` | ASL | 1000 | 16,002 | 5,287 | 4,172 | official | various |
| `ASLCITIZEN/ASLCitizen100` | ASL | 100 | 1,707 | 393 | 1,444 | official ASL Citizen | 640x480 (some 960x540) |
| `ASLCITIZEN/ASLCitizen300` | ASL | 300 | 4,699 | 1,128 | 4,058 | official | same |
| `ASLCITIZEN/ASLCitizen2731` | ASL | 2731 | 40,154 | 10,304 | 32,941 | official | same (4 portrait 480x640) |
| `LSA64` | Argentinian SL | 64 | 2,560 | 320 | 320 | signer-independent (signer 9 val, 10 test) | 1920x1080 |
| `DGI_PUCP30` | Peruvian SL | 30 | 500 | 108 | 107 | stratified random (PUCP-DGI305 split) | 1920x1080 |

Subsets: WLASL-N = the first N glosses of `WLASL_v0.3.json`; MSASL-N = official MS-ASL labels `< N`;
ASLCitizen-N = the N most frequent glosses (ties by name); DGI_PUCP30 = 30 of the 38
DGI305AEC_SignERT glosses (`wholebodypose.datasets.sources.PUCP_DGI_30`).

## HDF5 schema

One file per split: `<Prefix>_135-{Train,Val,Test}.hdf5`, plus `<name>_<N>_maplabels.json`.

```text
/<i>                       one group per sample, "0" .. "N-1"
    data        float64    (T, 2, 135)  x / width, y / height; undetected keypoints are exactly 0.0
    width       int64      ()           ORIGINAL video width in pixels
    height      int64      ()           ORIGINAL video height in pixels
    label       vlen str   ()           gloss (UTF-8)
    video_name  vlen str   ()           video identifier as published (see below)
```

- `T` is the number of decoded frames: every frame, no temporal subsampling.
- Keypoints 0-132 are COCO-WholeBody ([layout](../guide/keypoints.md)); 133 = mean of the
  shoulders (5, 6) = mid-chest, 134 = mean of the hips (11, 12) = mid-hip. A midpoint whose
  endpoint is missing is the mean with (0, 0) (rare; `--safe-midpoints` writes 0 instead).
- No compression, no attributes. h5py lists groups alphabetically ("0", "1", "10", ...): iterate
  with `wholebodypose.datasets.iter_samples`, which uses numeric order.
- `video_name`: absolute paths of the authors' machine for MS-ASL (`/data/MSASL/<split>/...`),
  ASL Citizen (`/data/ASL-Citizen/ASL_Citizen/videos/...`), LSA64 (`/data/LSA64_dataset/LSA64/...`)
  and PUCP-DGI (`/data/DGI305AEC_SignERT/split/<split>/...`); for WLASL the Kaggle layout
  `rgb/WLASL{N}/{split}/<id>.mp4`. Treat them as identifiers.
- `*_maplabels.json`: `{"id_to_label": {gloss: int}, "label_to_id": {"int": gloss}}`
  (`id_to_label` maps gloss -> int despite its name). **ASLCitizen300's map is not contiguous**
  (values 0..999): the original script numbered the glosses of all ASL Citizen subsets together.
  Remap with `sorted(set(values))` if your loss needs 0..299.
- Optional fields of rebuilt files (not in the published ones): `scores` float32 `(T, 135)` with
  `--save-scores`.

## Loading

```python
from wholebodypose.datasets import read_hdf5, iter_samples  # pip install "wholebodypose[datasets]"

samples = read_hdf5("ISLR/WLASL/WLASL100/WLASL100_135-Test.hdf5")
s = samples[0]
s["data"].shape, s["label"], s["video_name"], s["width"], s["height"]
# ((35, 2, 135), 'accident', 'rgb/WLASL100/test/00625.mp4', 256, 256)

xy = s["data"].transpose(0, 2, 1)  # (T, 135, 2) normalized
detected = (xy != 0).any(-1)  # (T, 135) bool
pixels = xy * [s["width"], s["height"]]  # back to pixels of the original video

for s in iter_samples(path, with_data=False):  # metadata only, adds s["T"]
    ...
```

Plain h5py:

```python
import h5py, json

with h5py.File(path) as f:
    for k in sorted(f, key=int):
        g = f[k]
        data, label = g["data"][()], g["label"][()].decode()
label_map = json.load(open("wlasl_100_maplabels.json"))["id_to_label"]  # gloss -> int
```

## Rebuilding from videos

```bash
pip install "wholebodypose[rtmpose,gpu,datasets]"     # [rtmpose] alone runs on CPU

# 1. check the sample lists (no video is read)
wholebodypose-build-islr --dataset lsa64 --root /path/to/LSA64_dataset --list

# 2. extract and write the three split files + the label map
wholebodypose-build-islr --dataset lsa64 --root /path/to/LSA64_dataset \
    --preset icmv2025 --out out/LSA64 --workers 4 --devices cuda:0,cuda:1 \
    --verify /path/to/published/ISLR/LSA64          # optional: compare with the published files
```

| Dataset (`--dataset`) | `--subset` | `--root` must contain |
|:--|:--|:--|
| `wlasl` | 100, 300, 1000, 2000 | `info.json` (= `WLASL_v0.3.json`) and `videos/<id>.mp4` (the 256x256 clips of `wlasl-processed.zip` / Kaggle `quanbkit1/wlasl-rgb-video`) |
| `msasl` | 100, 200, 500, 1000 | `{train,val,test}/<ytid>_<start>_<end>-msasl-<text>.mp4`; official `MSASL_{train,val,test}.json` in `--root` or `--annotations` |
| `asl_citizen` | 100, 300, 1000, 2731 | `splits/{train,val,test}.csv`, `videos/*.mp4` |
| `lsa64` | 64 | `lsa64_full_metadata.csv`, `LSA64/*.mp4` |
| `pucp_dgi` | 30 | `DGI305AEC_SignERT_metadata.csv`, `split/{train,val,test}/<gloss>_<k>_<id>.mp4` |

- **Resumable:** each video is cached as `.npz` (original-resolution pixels + scores) in
  `--cache-dir` (default `<out>/.cache/<dataset>-<settings hash>`); a re-run only extracts what is
  missing. `done.txt` lists finished videos, `errors.jsonl` the failures (the run continues; the
  failed samples are left out of the split file and reported). Share one `--cache-dir` between
  subsets (100 / 300 / 2000) to extract each video once.
- **Parallel:** `--workers N` starts N processes; `--devices cuda:0,cuda:1` assigns GPUs
  round-robin (needs `onnxruntime-gpu`). On CPU, onnxruntime already uses every core per process.
- `--limit 5` writes the first 5 samples of each split (smoke test); `--splits test` one split.
- `--order-like <published dir>`: write the groups in the published order (only differs for WLASL,
  whose original order came from a directory walk).

Python API: `wholebodypose.datasets.build.build_dataset(...)`, `extract_video(...)`,
`sources.load_samples(dataset, subset, root)` -> `(samples, label_map)`.

### Presets

| | `icmv2025` (reproduces the release) | `v1` (recommended for new data) |
|:--|:--|:--|
| model | RTMW-l 384x288 + YOLOX-m, rtmlib `performance` | same |
| colour fed to rtmlib | **RGB** (`legacy_rgb_input=True`, the original bug) | BGR (correct) |
| resolution | original (no resize) | original |
| person | first detection | largest bounding box |
| threshold / filter | score < 2.5 -> (0, 0); hand/foot chain filter | same |
| midpoints | plain mean (`--safe-midpoints` to change) | same |

Any preset field can be overridden: `--no-legacy-rgb-input`, `--resize-width 640`,
`--person largest`, `--kpt-thr 3`.

## Sample lists

The adapters in `wholebodypose.datasets.sources` rebuild the published sample lists from the
original metadata. Checked against **every** published file (33 files): per split the sample
counts, the set of `video_name`, every `label` and the `*_maplabels.json` content are identical;
the group order is identical too, except WLASL (see above). Dataset-specific rules that the
adapters encode:

- **MS-ASL**: file names come from the official json `text` field (the `/data/MSASL/*_metadata.csv`
  files use `clean_text` and do not match the downloaded files); YouTube ids saved with a URL
  suffix (`TwkGS9TjUX8&t=21s_...`) are resolved on disk; duplicated train clips are kept once; the
  never-downloaded `xfVNJxKWqeg_004007_004007-msasl-right.mp4` is dropped; the label's gloss is the
  `clean_text` of its last row. `MSASL200` was internally called group "300".
- **ASL Citizen**: rows of `splits/{train,val,test}.csv`, stably sorted by gloss name; label maps
  share one global numbering (see the schema notes).
- **LSA64**: split by signer; the sign `None` is published as `None_` (pandas read it as NaN).
- **PUCP-DGI**: files renamed `<gloss>_<k>_<row id>.mp4`, 30 glosses, labels 0..29 in original
  label order.

## Reproducing the published keypoints

10 videos of the **test** split of each dataset (evenly spaced groups), up to 30 evenly spaced
frames per video, CPU (onnxruntime 1.32, rtmlib 0.0.16); the published data was extracted on GPU.
`T match`: published `T` equals the number of frames decoded by OpenCV; `W/H`: published
width/height equal the video's. Differences are in normalized units over keypoints 0-132 detected in
both; detection agreement = fraction of (frame, keypoint) pairs where both agree on detected / not.

| input to rtmlib, resize width, person, threshold | dataset | n videos | T match | W/H match | frames | median abs diff | mean abs diff | detection agreement |
|---|---|---|---|---|---|---|---|---|
| BGR, none, first, 2.5 | ASLCitizen100 | 10 | 100% | 100% | 300 | 9.11e-04 | 2.15e-03 | 0.9904 |
| BGR, none, first, 2.5 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 5.49e-04 | 9.46e-04 | 1.0000 |
| BGR, none, first, 2.5 | LSA64 | 10 | 100% | 100% | 293 | 5.41e-04 | 9.50e-04 | 0.9981 |
| BGR, none, first, 2.5 | MSASL100 | 10 | 100% | 100% | 300 | 7.88e-04 | 1.25e-03 | 0.9972 |
| BGR, none, first, 2.5 | WLASL100 | 10 | 100% | 100% | 294 | 6.09e-04 | 1.10e-03 | 0.9969 |
| RGB, none, first, 2.5 | ASLCitizen100 | 10 | 100% | 100% | 300 | 6.00e-06 | 1.16e-04 | 0.9996 |
| RGB, none, first, 2.5 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 1.27e-08 | 3.57e-05 | 1.0000 |
| RGB, none, first, 2.5 | LSA64 | 10 | 100% | 100% | 293 | 2.50e-06 | 3.03e-05 | 0.9999 |
| RGB, none, first, 2.5 | MSASL100 | 10 | 100% | 100% | 300 | 7.43e-06 | 6.57e-05 | 0.9998 |
| RGB, none, first, 2.5 | WLASL100 | 10 | 100% | 100% | 294 | 3.17e-08 | 6.23e-05 | 0.9996 |
| RGB, 640, first, 2.5 | ASLCitizen100 | 10 | 100% | 100% | 300 | 6.00e-06 | 1.16e-04 | 0.9996 |
| RGB, 640, first, 2.5 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 9.00e-04 | 1.08e-03 | 0.9997 |
| RGB, 640, first, 2.5 | LSA64 | 10 | 100% | 100% | 293 | 8.64e-04 | 8.52e-04 | 0.9982 |
| RGB, 640, first, 2.5 | MSASL100 | 10 | 100% | 100% | 300 | 4.58e-04 | 9.25e-04 | 0.9989 |
| RGB, 640, first, 2.5 | WLASL100 | 10 | 100% | 100% | 294 | 1.33e-03 | 1.26e-03 | 0.9986 |
| BGR, 640, first, 2.5 | ASLCitizen100 | 10 | 100% | 100% | 300 | 9.11e-04 | 2.15e-03 | 0.9904 |
| BGR, 640, first, 2.5 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 8.48e-04 | 1.36e-03 | 0.9999 |
| BGR, 640, first, 2.5 | LSA64 | 10 | 100% | 100% | 293 | 8.14e-04 | 1.19e-03 | 0.9978 |
| BGR, 640, first, 2.5 | MSASL100 | 10 | 100% | 100% | 300 | 9.51e-04 | 1.42e-03 | 0.9967 |
| BGR, 640, first, 2.5 | WLASL100 | 10 | 100% | 100% | 294 | 1.23e-03 | 1.65e-03 | 0.9963 |
| BGR, none, largest, 2.5 | ASLCitizen100 | 10 | 100% | 100% | 300 | 9.11e-04 | 2.15e-03 | 0.9904 |
| BGR, none, largest, 2.5 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 5.49e-04 | 9.46e-04 | 1.0000 |
| BGR, none, largest, 2.5 | LSA64 | 10 | 100% | 100% | 293 | 5.41e-04 | 9.50e-04 | 0.9981 |
| BGR, none, largest, 2.5 | MSASL100 | 10 | 100% | 100% | 300 | 7.88e-04 | 1.25e-03 | 0.9972 |
| BGR, none, largest, 2.5 | WLASL100 | 10 | 100% | 100% | 294 | 6.09e-04 | 1.10e-03 | 0.9969 |
| RGB, none, largest, 2.5 | ASLCitizen100 | 10 | 100% | 100% | 300 | 6.00e-06 | 1.22e-04 | 0.9993 |
| RGB, none, largest, 2.5 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 1.27e-08 | 3.57e-05 | 1.0000 |
| RGB, none, largest, 2.5 | LSA64 | 10 | 100% | 100% | 293 | 2.50e-06 | 3.03e-05 | 0.9999 |
| RGB, none, largest, 2.5 | MSASL100 | 10 | 100% | 100% | 300 | 7.43e-06 | 6.57e-05 | 0.9998 |
| RGB, none, largest, 2.5 | WLASL100 | 10 | 100% | 100% | 294 | 3.17e-08 | 6.57e-05 | 0.9996 |
| RGB, none, first, 2.0 | ASLCitizen100 | 10 | 100% | 100% | 300 | 6.00e-06 | 1.16e-04 | 0.9664 |
| RGB, none, first, 2.0 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 1.27e-08 | 3.57e-05 | 0.9998 |
| RGB, none, first, 2.0 | LSA64 | 10 | 100% | 100% | 293 | 2.50e-06 | 3.03e-05 | 0.9958 |
| RGB, none, first, 2.0 | MSASL100 | 10 | 100% | 100% | 300 | 7.43e-06 | 6.57e-05 | 0.9725 |
| RGB, none, first, 2.0 | WLASL100 | 10 | 100% | 100% | 294 | 3.17e-08 | 6.24e-05 | 0.9816 |
| RGB, none, first, 3.0 | ASLCitizen100 | 10 | 100% | 100% | 300 | 5.91e-06 | 7.16e-05 | 0.9686 |
| RGB, none, first, 3.0 | DGI_PUCP30 | 10 | 100% | 100% | 161 | 1.27e-08 | 3.50e-05 | 0.9934 |
| RGB, none, first, 3.0 | LSA64 | 10 | 100% | 100% | 293 | 2.49e-06 | 2.93e-05 | 0.9888 |
| RGB, none, first, 3.0 | MSASL100 | 10 | 100% | 100% | 300 | 7.43e-06 | 6.35e-05 | 0.9883 |
| RGB, none, first, 3.0 | WLASL100 | 10 | 100% | 100% | 294 | 3.14e-08 | 5.51e-05 | 0.9832 |

End-to-end check of the library path (`wholebodypose-build-islr --preset icmv2025 --splits test
--limit 3 --order-like <published> --verify <published>`, all frames of the first 3 published test
groups, CPU):

| dataset | n | T match | median of per-sample median abs diff | median detection agreement |
|---|--:|--:|--:|--:|
| WLASL100 | 3 | 100% | 2.98e-08 | 1.0000 |
| MSASL100 | 3 | 100% | 5.52e-06 | 0.9997 |
| ASLCitizen100 | 3 | 100% | 6.26e-06 | 0.9993 |
| LSA64 | 3 | 100% | 4.03e-06 | 0.9999 |
| DGI_PUCP30 | 3 | 100% | 8.63e-06 | 1.0000 |

Conclusions:

- The published data is reproduced by **RGB input to rtmlib, original resolution, first detection,
  threshold 2.5 with the chain filter**: preset `icmv2025` (detection agreement 99.96-100 %, median
  difference 1e-8 to 7e-6, mean 3e-5 to 1.2e-4 normalized). The remaining differences are
  GPU-vs-CPU floating point noise that occasionally flips a score across the threshold.
- The correct BGR input changes the keypoints by ~5e-4 to 9e-4 (normalized, median) and flips more
  detections (ASL Citizen: 99.0 % agreement); resizing to width 640 (the pre-1.0 `Video2Pose`
  default) was not used for these files. (The sampled ASL Citizen videos are 640 wide, so the
  resize is a no-op there.)
- Thresholds 2.0 or 3.0 instead of 2.5 drop the detection agreement to 96.6-99.6 %: 2.5 is the
  published threshold.
- First vs largest person: identical on these single-signer clips.
- `T`, width and height match for every checked video.
