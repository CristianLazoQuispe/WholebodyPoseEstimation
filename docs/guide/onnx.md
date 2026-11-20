# ONNX Runtime engines

Since 1.1 the core install (NumPy, OpenCV, ONNX Runtime) runs both ONNX backends without rtmlib, PyTorch or
onepose:

- **RTMPose** (`RTMPoseModel(engine="onnx")`, default): YOLOX + RTMW `.onnx` files from OpenMMLab with the
  pre- and post-processing of rtmlib vendored in `wholebodypose.models.rtmpose._onnx` (Apache-2.0, see
  [THIRD_PARTY.md](../../THIRD_PARTY.md)). Outputs are bit-identical to rtmlib ([below](#rtmpose-vendored-code-vs-rtmlib)).
- **ViTPose** (`VITPoseModel(engine="onnx")`, default): an ONNX export of the onepose checkpoints with
  NumPy/OpenCV ports of onepose's pre- and post-processing. Keypoints identical to PyTorch, scores
  within float32 rounding ([below](#vitpose-agreement-with-pytorch)).

Both share one session helper (`wholebodypose.models.onnx_session`): execution providers (CUDA when the
installed `onnxruntime` build has it, see [installation.md](installation.md#gpu)), `intra_op_num_threads`
and the reported device. See [backends.md](backends.md) for the arguments.

## RTMPose: vendored code vs rtmlib

`RTMPoseModel` no longer imports rtmlib. The code it used from rtmlib 0.0.16 is vendored in
`wholebodypose.models.rtmpose._onnx`, with the arithmetic and dtypes kept as they were:

| Step | rtmlib source | Vendored in |
|:--|:--|:--|
| YOLOX letterbox (gray 114 padding), grid decoding, NumPy NMS | `object_detection/yolox.py`, `post_processings.py` | `_onnx/yolox.py` |
| RTMW box -> center/scale (padding 1.25), aspect-ratio expansion, affine crop, float64 normalization | `pose_estimation/rtmpose.py`, `pre_processings.py` | `_onnx/rtmw.py` |
| SimCC argmax (score = mean of the x/y maxima), inverse transform | `pose_estimation/post_processings.py` | `_onnx/rtmw.py` |
| `Wholebody` (detect, then one RTMW pass per box; whole image when nobody is detected) and its mode table | `solution/wholebody.py` | `_onnx/wholebody.py` |

What changed is the plumbing only: sessions come from the shared helper (ONNX Runtime only; rtmlib's
OpenCV/OpenVINO backends need `engine="rtmlib"`), weights come from `wholebodypose.weights`, and unused
code (RTMO, 3D, DARK/UDP, `format_result`) is not copied. `legacy_rgb_input` (the `icmv2025` dataset
preset) works the same with both engines: it only decides which colour order is passed in.

### Agreement

`python scripts/validate_onnx_stack.py compare --frames 10 --threads 8`: both implementations on the same
`.onnx` files and CPU, on the 12 images of `data/images/` and `hf/space/examples/*.jpg` plus the first 10
frames of each of the 5 videos (`data/videos/videoTest.mp4`, `hf/space/examples/*.mp4`), i.e. 62 frames
with 106 detected people in total (32 frames with two or more people). Each frame is run with BGR input
(library default) and with RGB input (`legacy_rgb_input=True`), and once more end to end through
`RTMPoseModel(engine="onnx")` vs `RTMPoseModel(engine="rtmlib")` with the paper settings
(`use_thresholding=True`, `filter_noise=True`) for both values of `legacy_rgb_input`.

| Mode | Raw `Wholebody` outputs (BGR + RGB) | `RTMPoseModel` outputs | Max abs difference |
|:--|--:|--:|--:|
| `performance` | 124 / 124 identical | 124 / 124 identical | 0 |
| `balanced` | 124 / 124 identical | 124 / 124 identical | 0 |
| `lightweight` | 124 / 124 identical | 124 / 124 identical | 0 |

"Identical" means `np.array_equal` on the YOLOX boxes, the keypoints (float64 before the cast in
`RTMPoseModel`) and the scores. The ViTPose multi-person path was checked the same way, with the vendored
YOLOX and with rtmlib's YOLOX as detector in front of ViTPose+ small: boxes, keypoints and scores
identical on all 62 frames for both `yolox_m` and `yolox_tiny`. A second run with onnxruntime's default
thread count (`--threads 0`, `lightweight`) was identical as well. Software: onnxruntime 1.32.0,
rtmlib 0.0.16, NumPy 2.4.6, OpenCV 5.1.0. None of these frames had zero detections, so the "whole
image when nobody is detected" branch is covered only by the code being the same, not by this run.

`tests/test_rtmpose_onnx.py` repeats a smaller version (two frames, every mode, both inputs) as `slow`
tests when rtmlib is installed, and compares the NumPy steps alone on random inputs in the fast suite.

**Dataset builder.** `wholebodypose-build-islr --dataset pucp_dgi --subset 30 --splits test --limit 5
--preset icmv2025 --save-scores --order-like <published> --verify <published>` was run once with the
vendored engine and once with `engine="rtmlib"` (CPU): the two HDF5 files are byte-identical (`cmp`), and
both give the same `--verify` result against the published `DGI_PUCP30_135-Test.hdf5` (5 samples, T match
100 %, median abs diff 5.76e-06, detection agreement 1.0000).

### CPU latency

`taskset -c 0-7 python scripts/validate_onnx_stack.py latency --threads 8 --repeats 10`: one
`Wholebody` call per frame (`data/images/image.jpg` and the first 3 frames of
each of the 5 videos, 16 frames), the two implementations run back to back on each frame, in alternating
order, 10 rounds, 160 calls each,
on the shared machine described below (load average about 240-270 during the run).

| Mode | Vendored, median (ms) | rtmlib, median (ms) | Per-call ratio vendored / rtmlib, median (IQR) |
|:--|--:|--:|--:|
| `performance` | 620 | 570 | 1.03 (0.89-1.22) |
| `balanced` | 545 | 559 | 1.00 (0.81-1.20) |
| `lightweight` | 265 | 261 | 1.01 (0.87-1.16) |

The same graphs run with the same session options, so equal speed is expected; the per-call ratios are
centred on 1 and their spread is the machine load.


## ViTPose: what the graph contains

| | |
|:--|:--|
| Input `input` | `(N, 3, 256, 192)` float32, RGB, `/255`, ImageNet mean/std; `N` is dynamic |
| Output `heatmaps` | `(N, 133, 64, 48)` float32 |
| Opset | 17 |
| Weights | stored as float16 and cast to float32 inside the graph; compute is float32 |
| Export | [`scripts/export_vitpose_onnx.py`](../../scripts/export_vitpose_onnx.py) (PyTorch 2.15 TorchScript exporter, onnx 1.23) |

The onepose checkpoints are float16 and PyTorch upcasts them to float32 when it loads them, so every
weight is exactly representable in float16: storing them as float16 is lossless (the script checks each
tensor and would keep a non-exact one in float32; none were found). The `.onnx` files are therefore the
size of the checkpoints instead of twice that size:

| Model | `.pth` (onepose, float16) | `.onnx` |
|:--|--:|--:|
| ViTPose+ small | 46.5 MiB | 46.5 MiB |
| ViTPose+ base | 171.8 MiB | 171.8 MiB |
| ViTPose+ large | 588.7 MiB | 588.8 MiB |
| ViTPose+ huge | 1215.6 MiB | 1215.8 MiB |

Every `.onnx` file is a single file (no external data) and is pinned by SHA-256 in
`wholebodypose.weights` (keys `vitpose_<size>_onnx`). Re-running the export script reproduced the
pinned hash for `small`.

Processing outside the graph (`wholebodypose.models.vitpose.onnx_runtime`), same as onepose:

1. person box (whole frame, or a padded YOLOX box) → center and scale with the 192:256 aspect ratio, no
   extra padding;
2. affine warp to 192×256 (`cv2.warpAffine`, bilinear, no UDP);
3. heatmap argmax, a ±0.25 px shift towards the higher neighbour (`post_process="default"`), inverse
   transform to image pixels; the score is the heatmap maximum.

onepose does not run flip test, so neither engine does. In multi-person mode the ONNX engine runs all
person crops of a frame as one batch; the PyTorch engine runs them one by one.

## ViTPose: agreement with PyTorch

`python scripts/vitpose_onnx_study.py compare`: both engines on CPU, on the 12 images of `data/images/` and
`hf/space/examples/*.jpg` plus 10 evenly spaced frames of each of 5 videos (`data/videos/videoTest.mp4`
and `hf/space/examples/*.mp4`), i.e. 62 inputs. Full frame: one person per input (8246 keypoints);
multi-person (`detector="yolox_m"`): the 50 video frames, 89 people (11837 keypoints).

| Model | Mode | Keypoint distance, median / max (px) | Score difference, median / max |
|:--|:--|--:|--:|
| small | full frame | 0 / 0 | 1.2e-7 / 3.2e-6 |
| small | multi-person | 0 / 0 | 1.2e-7 / 2.4e-6 |
| base | full frame | 0 / 0 | 1.2e-7 / 3.8e-6 |
| base | multi-person | 0 / 0 | 8.9e-8 / 1.3e-6 |
| large | full frame | 0 / 0 | 1.2e-7 / 4.5e-6 |
| large | multi-person | 0 / 0 | 1.2e-7 / 4.2e-6 |
| huge | full frame | 0 / 0 | 1.2e-7 / 5.1e-6 |
| huge | multi-person | 0 / 0 | 1.2e-7 / 1.7e-6 |

The keypoints are identical on every input and every model; scores differ by float32 rounding only, and
both engines found the same number of people on every frame. With `use_thresholding=True` (paper
setting) the outputs are therefore the same up to a score that sits within ~5e-6 of `kpt_thr`.

## ViTPose: CPU latency

`taskset -c 120-127 python scripts/vitpose_onnx_study.py bench --threads 8 --repeats 20`: time of
`model.predict(image)` on `data/images/image.jpg` (960×640), full frame (one ViTPose pass), 8 threads
for PyTorch (`torch.set_num_threads`) and ONNX Runtime (`intra_op_num_threads`), after 3 warm-up calls.
All ten configurations were timed round-robin (one call each per round, 20 rounds), so they saw the same
load.

Machine: 2× AMD EPYC 7662 64-Core Processor (128 cores, 256 threads, 1 TB RAM), Python 3.11,
PyTorch 2.15.0+cpu, ONNX Runtime 1.32.0, NumPy 2.4.6. **The machine was shared and fully loaded during
the measurement (load average 220-290 on 256 hardware threads)**, so the absolute times are higher and
noisier than on an idle machine; compare ratios within a row, not with [backends.md](backends.md#speed).

| Model | PyTorch median (min), ms | ONNX Runtime median (min), ms | median ratio |
|:--|--:|--:|--:|
| ViTPose+ small | 349 (210) | 152 (66) | 2.3× |
| ViTPose+ base | 616 (273) | 234 (176) | 2.6× |
| ViTPose+ large | 1462 (863) | 663 (463) | 2.2× |
| ViTPose+ huge (paper) | 2439 (1751) | 1321 (955) | 1.8× |
| RTMPose `performance` (YOLOX-m + RTMW-l 384×288), reference | — | 692 (574) | |
| RTMPose `lightweight` (YOLOX-tiny + RTMW-m 256×192), reference | — | 305 (260) | |

ONNX Runtime was faster for every size in this run. The ratio under load is not a promise for an idle
machine; rerun the command on your hardware.

## Quantization study

`python scripts/vitpose_onnx_study.py quant` then `quant-latency` (8 threads, `taskset -c 120-127`), for the
two models where it matters most: ViTPose+ base (a candidate for a cheaper Space) and RTMW-l 384×288 (the
paper's RTMPose model). Variants, all made with ONNX Runtime 1.32 tools:

- **int8 dynamic**: `quantize_dynamic(weight_type=QUInt8)` after `quant_pre_process`; weights int8,
  activations quantized at run time;
- **fp16**: `onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)`.

For ViTPose the shipped graph stores float16 weights behind `Cast` nodes, which quantizers do not see, so
the study first expands it to a plain float32 graph (343.5 MiB, numerically the same model).

Deviation is measured against the float32 model on the 62 inputs above (ViTPose: full frame, 8246
keypoints; RTMW-l: on the YOLOX-m boxes of the float32 run, 106 people, 14098 keypoints). "Visible" means
score above the paper threshold of the float32 model (ViTPose 0.5, RTMW-l 2.5); "mask agreement" is
the fraction of keypoints whose above/below-threshold decision is the same.

Latency is one `session.run` on a batch of one, timed round-robin over the variants (30 rounds, median
(min)); the machine was loaded as above (load average 180-250).

| Model | Variant | Size (MiB) | Latency, ms | Visible kpts: median / p95 / max distance (px) | Visible kpts > 0.5 px | Mask agreement |
|:--|:--|--:|--:|--:|--:|--:|
| ViTPose+ base | float32 (shipped, fp16-stored) | 171.8 | 149 (133) | reference | | |
| ViTPose+ base | float32 (plain) | 343.5 | 149 (130) | reference | | |
| ViTPose+ base | int8 dynamic | 98.8 | 137 (126) | 0 / 10.0 / 457.1 | 17.6 % | 98.36 % |
| ViTPose+ base | fp16 | 171.9 | 157 (141) | 0 / 0 / 13.3 | 0.09 % | 99.99 % |
| RTMW-l 384×288 | float32 (shipped) | 218.7 | 246 (147) | reference | | |
| RTMW-l 384×288 | int8 dynamic | 54.9 | 492 (270) | 7.0 / 242.6 / 2848.7 | 95.8 % | 50.71 % |
| RTMW-l 384×288 | fp16 | 109.4 | 283 (132) | 0 / 0 / 7.3 | 1.34 % | 99.99 % |

Recommendations (CPU, ONNX Runtime):

- **ViTPose+ int8 dynamic: no.** 17.6 % of the visible keypoints move by more than 0.5 px (p95 10 px)
  and 1.6 % of the keypoints change visibility, for 8 % lower median latency.
- **ViTPose+ fp16: no.** Nearly identical keypoints, but the shipped float32 graph already has the same
  size (171.8 vs 171.9 MiB) and fp16 is not faster on CPU.
- **RTMW-l int8 dynamic: no.** The SimCC outputs do not survive dynamic quantization: half of the
  visibility decisions change (the number of visible keypoints drops from 13440 to 6517) and it is
  slower than float32.
- **RTMW-l fp16: not by default.** It halves the file (218.7 → 109.4 MiB) with near-identical keypoints
  (visible: 1.3 % move by more than 0.5 px, max 7.3 px; 2 of 14098 visibility decisions change), but it
  is not faster on CPU and the paper numbers were produced with the float32 file.

