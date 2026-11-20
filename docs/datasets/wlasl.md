# WLASL (Word-Level American Sign Language)

> Status: provenance reconstructed from code, notebooks and data on disk (Nov 2026).
> Statements marked **[verified]** were checked by re-computation. Items marked **TODO (author)**
> still need the author's input; they are collected in [Open questions](#6-open-questions-todo-author).

## 1. Summary

| Subset | Classes (train / val / test) | Train | Val | Test | Total |
|:--|:-:|--:|--:|--:|--:|
| WLASL100 | 100 / 100 / 100 | 1,442 | 338 | 258 | 2,038 |
| WLASL300 | 300 / 300 / 300 | 3,549 | 900 | 668 | 5,117 |
| WLASL2000 | 2000 / 2000 / **1999** | 14,289 | 3,916 | 2,878 | 21,083 |

- **Splits:** the official WLASL v0.3 splits. A sample's split is the `split` field of
  `WLASL_v0.3.json`, and WLASL-N is the instances of the first N glosses in that file.
  **[verified: 25,138 / 25,138 samples]**
- **No samples are dropped:** all 21,083 instances of WLASL v0.3 are present.
- **Files:** `ISLR/WLASL/WLASL{100,300,2000}/WLASL{N}_135-{Train,Val,Test}.hdf5` and
  `wlasl_{N}_maplabels.json` (gloss -> int in the official class order: `book`=0, `drink`=1,
  `computer`=2, ...). Despite its name, the key `id_to_label` maps gloss -> int.
- **Pose:** RTMPose-WholeBody (rtmlib `Wholebody`, mode `performance`: YOLOX-m HumanArt detector +
  RTMW-l 384x288) on every frame, 133 COCO-WholeBody keypoints plus mid-chest (133) and mid-hip
  (134). Schema: [README](README.md#hdf5-schema).

## 2. Sources

1. **Annotations:** `WLASL_v0.3.json`, `nslt_{100,300,1000,2000}.json` and `wlasl_class_list.txt`
   from the official repository https://github.com/dxli94/WLASL (class list pinned to commit `ac00e6b`).
2. **Videos:** a pre-processed copy of all 21,083 WLASL v0.3 instances, distributed on Kaggle as
   `quanbkit1/wlasl-rgb-video`, with layout `rgb/WLASL{100,300,1000,2000}/{train,val,test}/<video_id>.mp4`.
   This is where the `video_name` strings come from (e.g. `rgb/WLASL100/train/14675.mp4`).
   - Every video is **256x256 at 25 fps**, 9 to 247 frames (median 58).
   - Each clip is already **trimmed in time** to `frame_start` / `frame_end`.
   - The **whole frame is resized keeping its aspect ratio and letterboxed** (black bars) into
     256x256; the clips are *not* cropped to the WLASL bbox.
3. **Local mirror used for verification:** `wlasl-processed.zip` (2.9 GB, internal timestamps
   2022-02-07) and its extraction `WLASL/` (`videos/*.mp4`, `frames/<gloss>/<id>.png`, `info.json`).
   - `info.json` has the same content as `WLASL_v0.3.json`.
   - The zip and the extracted folder are byte-identical (sha256 of all 42,167 members).
   - Every HDF5 sample has exactly the frame count and size of its video in this mirror, and
     re-running the extractor on it reproduces the published keypoints (section 3.2), so it is
     equivalent to the copy used.
   - It is **not** the `risangbaskoro/wlasl-processed` Kaggle release (about 11,980 videos and a
     `missing.txt`).
4. **Cross-check labels:** the WLASL label files of the Uni-Sign codebase
   (`labels-2000.{train,dev,test}`, gzipped pickles of `{id: {name, gloss, text, video_path}}`).

## 3. Processing pipeline (what was actually done)

### 3.1 Split and label validation

Original code: `LargeScaleSLR/codes/data_preparation/wlasl/2.split_dataset.ipynb`,
`src/utils/datasets/wlasl/wlasl.py`.

1. Flatten `WLASL_v0.3.json` into one row per instance (21,083 rows: gloss, bbox, fps,
   frame_start / frame_end, signer_id, source, split, url).
2. Parse the four `nslt_*.json` files (41,425 rows of id / subset / group) and join them to v0.3 by
   `video_id`.
3. **Inconsistencies found and fixed:**
   - 12 video ids appear in `nslt_*.json` but have no gloss in `WLASL_v0.3.json`: `12209, 48251,
     39347, 16096, 20138, 57839, 60721, 09500, 47639, 20065, 51153, 13422`. Those 19 nslt rows were
     dropped (41,425 -> 41,406). After the fix the unique-id count is exactly 21,083.
   - `wlasl_class_list.txt` misspells `lose` as `losear`; it was corrected
     (`wlasl_class_list_cleaned.txt`), so the class list and v0.3 have the same 2000 glosses.
4. **Completeness checks:** for every group (100/300/1000/2000) and split, the set of video files on
   disk equals the set of labelled ids in both directions, and the merged table has no NaN gloss or
   label. Output: `videos_info.csv` (id, subset, group, gloss, label, relative_path).
5. Class-balance EDA per group and split (`3.eda.ipynb`).

The rows of `videos_info.csv` follow a directory walk of the Kaggle `rgb/` tree, which is why the
group order inside the HDF5 files is neither gloss nor id order. The new adapter yields the
samples in `info.json` order; `wholebodypose-build-islr --order-like <published dir>` copies the
published order when an identical file is wanted.

### 3.2 Pose extraction

Original code: `LargeScaleSLR/codes/data_preparation/wlasl/4_videos2pose.py`,
`src/utils/pose_estimation/video2pose.py`, with `RTMPoseModel` from WholeBodyPose
`archive/pre-v1.0` (commit `72f9307`).

- The extractor ran once per unique video (21,083). Subsets reuse those poses, so a video in several
  subsets has identical poses in each.
- Every frame is decoded with OpenCV at the stored 256x256 size (no resize), converted BGR -> RGB
  and passed to rtmlib `Wholebody(mode='performance', backend='onnxruntime', device='cuda')`.
  rtmlib expects BGR, so the model saw red and blue swapped (see Known issues).
- `keypoints[0]` is kept: the first detection, not the largest person.
- Noise filter: if a limb chain is not fully confident the whole hand or foot is invalidated (e.g.
  if any of shoulder, elbow, wrist, hand-root is below the threshold, the 21 hand points and the
  wrist are zeroed).
- Thresholding: any keypoint with score < 2.5 (rtmlib SimCC score) is set to (0, 0).
- Each video was cached to `keypoints/video_<id>.json` (commit `3e7d32e`, 2024-05-28), merged into
  `wlasl_pose.parquet`, joined back to each subset -> `wlasl_<N>_pose_{train,val,test}_NEW.parquet`
  plus `wlasl_<N>_maplabels.json`.

**[verified]** Re-running these settings with the new library
(`wholebodypose-build-islr --preset icmv2025`, CPU) on 10 WLASL100 test videos reproduces the
published keypoints: median absolute difference ~3e-8 (normalized), mean ~6e-5 (about 0.015 px;
the original ran on GPU) and 99.96 % detection agreement. Feeding BGR (the correct input) or
resizing to width 640 first does *not* reproduce the data (median ~6e-4 and ~1.3e-3). Full table:
[README](README.md#reproducing-the-published-keypoints).

### 3.3 HDF5 export

Original code: `SignLanguageRecognition/Notebooks/old/create_datasets_135.py` (Jul 2025); the
96-keypoint precursor is `Notebooks/old/WLASL_splits.ipynb` (Apr 2025).

- `x /= width`, `y /= height` with the 256x256 video size; append 133 = mean(5, 6) and
  134 = mean(11, 12); transpose to `(T, 2, 135)` float64. Undetected points stay 0.0.
- Store `label` (gloss), `video_name` (Kaggle relative path), `width` / `height` (int64).
- **[verified]** For all 25,138 samples `T` equals the video frame count, `width` / `height` = 256
  and `label` equals the v0.3 gloss.
- **[verified]** The 96-keypoint (`WLASL*_96`) and the 135-keypoint files come from the same
  parquet: the shared keypoints are identical.

### 3.4 Independent cross-check against Uni-Sign

Code: `Notebooks/old/AnalysisWLASL.ipynb`.

- WLASL100 equals the 100 most frequent glosses, and the WLASL100 test ids match Uni-Sign's test
  ids 258/258.
- `labels-{100,300,1000}.*` were derived from Uni-Sign's `labels-2000.*`.
- **[verified]** The id sets of `labels-{100,300,2000}.{train,dev,test}` equal the HDF5 id sets for
  every split, and the `text` field matches the gloss.

## 4. Known issues and caveats

- **WLASL2000 test has 1,999 classes:** `caterpillar` has no test instance in WLASL v0.3 (inherited
  from the official split; train and val have all 2000 glosses).
- **Third-party 256x256 letterboxed videos**, not the original source videos:
  coordinates are normalized by the padded 256x256 canvas (not the original frame or the signer
  bbox), `width` / `height` are 256, and the hands have a low effective resolution.
- **BGR/RGB swap:** RGB frames were fed to rtmlib, which expects BGR. The released keypoints were
  produced this way and the swap changes the predictions (measured above). The library keeps the
  correct BGR input by default; `RTMPoseModel(legacy_rgb_input=True)` / `--preset icmv2025` replays
  the swap.
- **Person selection:** the first detection is used, not the largest person. On the 10 validation
  videos both choices give the same person (identical results).
- **Missing parts:** in WLASL2000-test, left-hand points are all zero in 23.0 % of frames and
  right-hand points in 10.7 %; 272 of 170,113 frames have no keypoints at all.
- **Midpoints with missing inputs:** if exactly one shoulder (or hip) is undetected the midpoint is
  the mean with a (0, 0) point and is biased (1 frame in WLASL2000 test for the shoulders).
  `--safe-midpoints` writes (0, 0) instead.
- **WLASL1000 is not released.** It can be built (`--subset 1000`, official first-1000 glosses:
  13,168 samples). The local Uni-Sign-derived `labels-1000.*` was built by gloss frequency and
  differs by 3 glosses (13,171 samples).
- `video_name` paths (`rgb/WLASL.../...`) refer to the Kaggle layout. In the mirror the file is
  `WLASL/videos/<id>.mp4`.
- Group order inside the HDF5 is not sorted by gloss or id; use `label` and `video_name`.

## 5. Rebuild with the library

```bash
pip install "wholebodypose[rtmpose,gpu,datasets]"
# videos: the 21,083 clips of Kaggle quanbkit1/wlasl-rgb-video or the identical wlasl-processed.zip,
# extracted so that <root>/videos/<id>.mp4 and <root>/info.json exist
wholebodypose-build-islr --dataset wlasl --subset 100 --root /path/to/WLASL \
    --preset icmv2025 --out ISLR/WLASL/WLASL100 --workers 4 --devices cuda:0,cuda:1 \
    [--order-like /path/to/published/ISLR/WLASL/WLASL100] [--verify /path/to/published/ISLR/WLASL/WLASL100]
# repeat with --subset 300 and --subset 2000 (reuse the cache with --cache-dir to extract each video once)
```

Acceptance checks (all satisfied by the adapter, see [README](README.md#sample-lists)):
counts match section 1, `T` equals the video frame count, splits equal the v0.3 `split` field,
WLASL2000 test has 1999 classes, label map in `wlasl_class_list.txt` order with `losear` -> `lose`.

## 6. Open questions (TODO author)

- **TODO (author):** who produced the 256x256 / 25 fps / letterboxed / trimmed videos, and with which
  tool? Candidates: the Kaggle uploader `quanbkit1`, or the dataset behind `wlasl-processed.zip`.
  Were frames resampled to 25 fps or was the source fps kept (`info.json` says 25 for every instance)?
- **TODO (author):** exact Kaggle dataset name / version of `wlasl-processed.zip` (2022-02-07), and
  whether `quanbkit1/wlasl-rgb-video` is byte-identical to it (only frame counts, sizes and
  re-extracted poses were verified, not file hashes).
- **TODO (author):** rtmlib / onnxruntime-gpu / CUDA versions and the GPU of the remote machine
  (`/home/cc/...`, probably Chameleon Cloud); extraction date (around May 2024, judging from
  LargeScaleSLR commits `3e7d32e` / `eabc15b`).
- **TODO (author):** the `*_NEW.parquet` files and the per-video JSON caches are deleted, so the raw
  rtmlib scores are lost. Keep or regenerate them (`--save-scores`)?
- **TODO (author):** manual or visual quality control beyond the notebooks (visual inspection of
  poses or videos, time spent).
- **TODO (author):** why the first detection instead of the largest person, and whether the RGB/BGR
  swap should be kept for future releases.
- **TODO (author):** should the 12 orphan nslt ids and the `losear` fix be reported upstream?

## 7. Citation

```bibtex
@inproceedings{li2020word,
  title     = {Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author    = {Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {1459--1469},
  year      = {2020}
}
```

WLASL is released under the Computational Use of Data Agreement (C-UDA) for academic,
non-commercial use. Only pose keypoints are redistributed, not videos.

## Website blurb

**WLASL.** We use all 21,083 instances of WLASL v0.3 (Li et al., WACV 2020) with the official splits
for WLASL100, WLASL300 and WLASL2000. Before extracting poses we checked the official annotations
against each other: matching `WLASL_v0.3.json` with the `nslt_*.json` subset files revealed 12 subset
entries with no gloss, which we dropped, and we fixed a misspelled class name (`losear` -> `lose`).
We confirmed that every annotated instance has a video and every video has an annotation, for every
subset and split, and checked our splits against an independent public label set (Uni-Sign). We ran
RTMPose-WholeBody on every frame of the trimmed 256x256, 25 fps clips, with no temporal subsampling.
Low-confidence keypoints (score < 2.5) and incomplete hand or foot chains are set to zero instead of
being guessed. Each sample has 135 keypoints: the 133 COCO-WholeBody points plus mid-chest and
mid-hip, normalized by the frame size. As in the official split, WLASL2000-test covers 1,999 classes
(`caterpillar` has no test sample).
