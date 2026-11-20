# Licensing

| Part | License | Commercial use | Derivatives | What users must do | Why |
|:--|:--|:--|:--|:--|:--|
| **Code** (library, CLI, demo, scripts) | [Apache-2.0](../LICENSE) (code up to v1.0.0 also [MIT](../LICENSES/MIT.txt)) | yes, as is, without warranty | yes, under any license | keep the copyright, `LICENSE` and `NOTICE`; state changes | standard open-source license of the pose-estimation ecosystem (MMPose, rtmlib, ViTPose) |
| **Pose datasets** ([pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition)) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | no | yes, under CC BY-NC-SA 4.0 | cite WholeBodyPose and the source dataset of every subset; follow the source dataset terms | the source videos are non-commercial; LSA64 is ShareAlike |
| **ISLR model weights trained by the authors** | CC BY-NC-SA 4.0 | no | yes, under CC BY-NC-SA 4.0 | cite WholeBodyPose; follow the terms of the training data | trained on the pose datasets above |
| **Third-party whole-body weights** (RTMW, YOLOX Human-Art, ViTPose+, incl. ONNX conversions) | Apache-2.0, as released by their authors | not guaranteed | per Apache-2.0 | check the training-data terms before commercial use | trained on non-commercial data (COCO-WholeBody CC BY-NC 4.0, MPII, Human-Art) |
| **MediaPipe models** | Apache-2.0 | yes | per Apache-2.0 | keep the license notice | released by Google under Apache-2.0 |

Principles:

1. No part is distributed under terms more permissive than its sources.
2. The terms of each source dataset or model also apply.
3. Everything is provided as is, without warranty or liability (Apache-2.0 §7–8, CC BY-NC-SA 4.0 §5).

Third-party code and weights, with sources: [THIRD_PARTY.md](../THIRD_PARTY.md).

## How to cite

Cite **(1) the paper**, **(2) the backend** you used and, if you use the pose datasets, **(3) the source
dataset** of every subset.

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

Plain text: C. Lazo-Quispe, R. Castro-Cruz, G. Bejarano, M. Salazar-Espinosa, "WholeBodyPose: a unified
end-to-end framework for sign language recognition and pose-based training data," Proc. SPIE 14114,
Eighteenth International Conference on Machine Vision (ICMV 2025), 141140A (25 February 2026),
https://doi.org/10.1117/12.3096312. GitHub also offers it through "Cite this repository"
([`CITATION.cff`](../CITATION.cff)).

**(2) The backend**: RTMW, ViTPose++ or MediaPipe, plus COCO-WholeBody for the keypoint layout. BibTeX in
[guide/backends.md](guide/backends.md#citing-the-backends).

**(3) The source datasets**: WLASL (Li et al., WACV 2020), MS-ASL (Vaezi Joze and Koller, BMVC 2019),
ASL Citizen (Desai et al., NeurIPS 2023 Datasets and Benchmarks), LSA64 (Ronchetti et al., CACIC 2016) and
PUCP-DGI (Pontificia Universidad Católica del Perú). BibTeX in the
[dataset card](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) and in the
`README.md` of each subset folder.
