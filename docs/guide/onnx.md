# ViTPose on ONNX Runtime

`VITPoseModel(engine="onnx")` runs ViTPose+ without PyTorch or onepose: the network is an ONNX graph
and the pre- and post-processing are NumPy/OpenCV ports of onepose's. The install is
`pip install "wholebodypose[vitpose-onnx]"` (only `onnxruntime` on top of NumPy and OpenCV; add
`[rtmpose]` for the multi-person detector). `engine="auto"` (default) uses it whenever the `.onnx` file
can be found; see [backends.md](backends.md#vitpose-vitpose).

## What the graph contains

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

## Agreement with PyTorch

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

## CPU latency

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
  is not faster on CPU and the paper numbers were produced with the float32 file. It could be offered as
  an opt-in download-size option, or evaluated on GPU (not measured here).

Static (calibrated) int8 quantization was not tried.
