---
license: other
license_name: apache-2.0-with-dataset-restrictions
license_link: WEIGHTS_LICENSE.md
library_name: wholebodypose
pipeline_tag: keypoint-detection
tags:
- pose-estimation
- whole-body
- coco-wholebody
- onnx
- mediapipe
- vitpose
- rtmpose
- sign-language
---

# WholeBodyPose model weights (mirror)

This repository is a **byte-identical mirror** of the third-party model files that the
[WholeBodyPose](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation) library downloads at run time.
It exists only so that the results of the WholeBodyPose paper (ICMV 2025) stay reproducible if an upstream
link moves or disappears. **We did not train, fine-tune or modify any of these models.** The only files we
produced are the ViTPose+ ONNX exports (`vitpose/*.onnx`), converted from the upstream checkpoints without
retraining. All credit goes to the original authors, listed below; their licenses and terms apply.

| Resource | Link |
|:--|:--|
| Code | https://github.com/CristianLazoQuispe/WholebodyPoseEstimation |
| Project page | https://cristianlazoquispe.github.io/WholebodyPoseEstimation/ |
| Demo | https://huggingface.co/spaces/CristianLazoQuispe/WholeBodyPoseDemo |
| Pose dataset | https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition |

## Original models and authors

| Files | Model | Authors / maintainers | Paper | Original source | License |
|:--|:--|:--|:--|:--|:--|
| `rtmpose/rtmw-*.onnx` | RTMW (RTMPose for whole-body), ONNX export | Tao Jiang, Xinchen Xie, Yining Li (OpenMMLab) | [RTMW, arXiv:2407.08634](https://arxiv.org/abs/2407.08634); [RTMPose, arXiv:2303.07399](https://arxiv.org/abs/2303.07399) | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) (download.openmmlab.com), as used by [rtmlib](https://github.com/Tau-J/rtmlib) | Apache-2.0 |
| `rtmpose/yolox_*_humanart-*.onnx` | YOLOX-m / YOLOX-tiny person detectors trained on Human-Art, ONNX export | OpenMMLab (MMPose RTMPose project); YOLOX by Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun | [YOLOX, arXiv:2107.08430](https://arxiv.org/abs/2107.08430); [Human-Art, CVPR 2023](https://arxiv.org/abs/2303.02760) | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) (download.openmmlab.com) | Apache-2.0 |
| `vitpose/splitted_vitpose+_*_coco_wholebody_half.pth` | ViTPose+ (S/B/L/H) COCO-WholeBody heads, fp16 checkpoints split per dataset head | Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao (ViTAE-Transformer); fp16 split by [developer0hye/onepose](https://github.com/developer0hye/onepose) | [ViTPose++, IEEE TPAMI 2024](https://arxiv.org/abs/2212.04246); [ViTPose, NeurIPS 2022](https://arxiv.org/abs/2204.12484) | [onepose release 1.0.0](https://github.com/developer0hye/onepose/releases/tag/1.0.0), from [ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | Apache-2.0 |
| `vitpose/vitpose+_*_coco_wholebody.onnx` | The ViTPose+ (S/B/L/H) COCO-WholeBody models above, exported to ONNX (opset 17, dynamic batch) by WholeBodyPose's [`scripts/export_vitpose_onnx.py`](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/scripts/export_vitpose_onnx.py) | Same as ViTPose+ (export by the WholeBodyPose authors) | Same as ViTPose+ | This mirror only (no upstream file) | Apache-2.0 (as the source checkpoints) |
| `mediapipe/*.task` | MediaPipe Tasks Pose (BlazePose GHUM heavy/full), Hand and Face landmarkers, float16 | Google (MediaPipe team) | [MediaPipe, arXiv:1906.08172](https://arxiv.org/abs/1906.08172); [BlazePose, arXiv:2006.10204](https://arxiv.org/abs/2006.10204); [MediaPipe Hands, arXiv:2006.10214](https://arxiv.org/abs/2006.10214); [Face mesh, arXiv:1907.06724](https://arxiv.org/abs/1907.06724) | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe), `storage.googleapis.com/mediapipe-models/.../float16/latest/` | Apache-2.0 (see the [MediaPipe model cards](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models)) |

The models were trained by their authors on third-party datasets (e.g. COCO, COCO-WholeBody, Human-Art and
the other datasets of each paper), which have their own terms. Check those terms, and the upstream
repositories, before any commercial use.

## Files

Every file is stored at the same relative path the library uses in its local cache. `MANIFEST.csv` lists,
for each file, its library key, path, size in bytes, SHA-256, original URL, license and source. The same
SHA-256 values are pinned in
[`wholebodypose/weights.py`](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/src/wholebodypose/weights.py),
and every download is checked against them.

<!-- FILES_TABLE -->

How we checked that the mirror is identical to the upstream files:

* ViTPose+: the MD5 of each `.pth` equals the `hash` that onepose itself checks
  (`onepose.models.factory.model_config`).
* RTMW / YOLOX: the `.onnx` extracted from the upstream `onnx_sdk` `.zip` archive is byte-identical to the file
  rtmlib downloads and caches (`~/.cache/rtmlib/hub/checkpoints`).
* MediaPipe: the files are the upstream `float16/latest` downloads. Google may update `latest`; this mirror
  keeps the version the paper's code was run with (for `pose_landmarker_full.task` it already differs from
  the `float16/1` file).

## How to use

You normally do not need to touch this repository. The library resolves every model in this order:
local cache, then this mirror, then the original upstream URL, and checks the SHA-256 each time.

```bash
pip install "wholebodypose[all] @ git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git"
wholebodypose download            # pre-fetch all files into ~/.cache/wholebodypose
```

```python
from wholebodypose import PoseEstimator

est = PoseEstimator("rtmpose")  # or "vitpose", "mediapipe"
keypoints, scores = est.predict_image("person.jpg")  # (N, 133, 2), (N, 133)
```

Environment variables: `WHOLEBODYPOSE_HOME` (cache folder, default `~/.cache/wholebodypose`),
`WHOLEBODYPOSE_WEIGHTS_REPO` (another mirror), `WHOLEBODYPOSE_OFFLINE=1` (never download).

To fetch a single file directly:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "CristianLazoQuispe/wholebodypose-weights", "rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
)
```

## Citation

If these files help your work, please cite the original papers of the model you use, and WholeBodyPose
(doi:[10.1117/12.3096312](https://doi.org/10.1117/12.3096312)) if you use the library:

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
@article{jiang2024rtmw,
  title   = {RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation},
  author  = {Jiang, Tao and Xie, Xinchen and Li, Yining},
  journal = {arXiv preprint arXiv:2407.08634},
  year    = {2024}
}
@article{jiang2023rtmpose,
  title   = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  author  = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  journal = {arXiv preprint arXiv:2303.07399},
  year    = {2023}
}
@article{ge2021yolox,
  title   = {YOLOX: Exceeding YOLO Series in 2021},
  author  = {Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal = {arXiv preprint arXiv:2107.08430},
  year    = {2021}
}
@inproceedings{ju2023humanart,
  title     = {Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
  author    = {Ju, Xuan and Zeng, Ailing and Wang, Jianan and Xu, Qiang and Zhang, Lei},
  booktitle = {CVPR},
  year      = {2023}
}
@article{xu2024vitposeplus,
  title   = {ViTPose++: Vision Transformer for Generic Body Pose Estimation},
  author  = {Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {46},
  number  = {2},
  pages   = {1212--1230},
  year    = {2024},
  doi     = {10.1109/TPAMI.2023.3330016}
}
@inproceedings{xu2022vitpose,
  title     = {ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author    = {Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
@article{lugaresi2019mediapipe,
  title   = {MediaPipe: A Framework for Building Perception Pipelines},
  author  = {Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon and McClanahan, Chris and Uboweja, Esha and Hays, Michael and Zhang, Fan and Chang, Chuo-Ling and Yong, Ming Guang and Lee, Juhyun and Chang, Wan-Teh and Hua, Wei and Georg, Manfred and Grundmann, Matthias},
  journal = {arXiv preprint arXiv:1906.08172},
  year    = {2019}
}
@article{bazarevsky2020blazepose,
  title   = {BlazePose: On-device Real-time Body Pose tracking},
  author  = {Bazarevsky, Valentin and Grishchenko, Ivan and Raveendran, Karthik and Zhu, Tyler and Zhang, Fan and Grundmann, Matthias},
  journal = {arXiv preprint arXiv:2006.10204},
  year    = {2020}
}
@inproceedings{jin2020whole,
  title     = {Whole-Body Human Pose Estimation in the Wild},
  author    = {Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle = {ECCV},
  year      = {2020}
}
```

## License

Every file keeps the Apache-2.0 license of its original release ([WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md)).
The ViTPose ONNX files are modified versions (converted to ONNX, no retraining) of the Apache-2.0 checkpoints.
This mirror claims no authorship of the models.

**Training data.** RTMW, the Human-Art YOLOX detectors and ViTPose+ were trained on datasets whose terms
restrict commercial use (COCO-WholeBody: CC BY-NC 4.0; MPII: no commercial use; Human-Art: non-commercial).
Commercial use of these weights is not guaranteed; check those terms first. The MediaPipe models have no such
restriction.

The WholeBodyPose library is licensed under Apache-2.0
([docs/LICENSING.md](https://github.com/CristianLazoQuispe/WholebodyPoseEstimation/blob/main/docs/LICENSING.md)).
