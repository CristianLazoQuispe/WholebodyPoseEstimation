# Hugging Face tooling

Scripts and cards for the Hugging Face repositories of WholeBodyPose. Nothing here runs in CI or at
install time; uploads are manual and need a token with write access.

| Path | Purpose | Hub repo |
|:--|:--|:--|
| `upload_weights.py` | stage, upload and verify the weights mirror | [`CristianLazoQuispe/wholebodypose-weights`](https://huggingface.co/CristianLazoQuispe/wholebodypose-weights) (model) |
| `model_card.md` | model card of the weights mirror (template; the file table is filled in by `--build`) | same |
| `upload_dataset_card.py` | check and push a dataset card (`README.md` only) | [`CristianLazoQuispe/pose-action-recognition`](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) (dataset) |
| `dataset_cards/pose-action-recognition.md` | dataset card | same |
| `dataset_cards/LICENSE.md` | CC BY-NC-SA 4.0 license of the pose datasets | same |
| `make_dataset_folder_readmes.py` | writes `dataset_cards/folders/`: a `README.md` per dataset folder (files, license, BibTeX, note for AI agents) and the root `CITATION.cff` | same |
| `_staging/` | build output (git-ignored) | |

Install what the upload / remote-verify modes need: `pip install huggingface_hub`. The token is read from
`HF_TOKEN` or `HUGGINGFACE_TOKEN`, or from `--env-file` (default `.env`), and is never printed.

## Weights mirror

`src/wholebodypose/weights.py` (`WEIGHTS`) is the single source of truth: every file has a key, its path in
the mirror and the local cache, the original URL, license, source and a pinned **SHA-256**. The library
downloads from the mirror first, falls back to the upstream URL and rejects any file whose SHA-256 differs.

```bash
# 1. get every file into the local cache through the library (mirror -> upstream)
wholebodypose download                      # or: python -m wholebodypose.cli download

# 2. print the plan; --hash also checks every SHA-256 against WEIGHTS
python hf/upload_weights.py --dry-run --hash

# 3. stage: hard links at mirror paths + MANIFEST.csv + README.md (model card)
python hf/upload_weights.py --build         # --fetch downloads missing files first
python hf/upload_weights.py --verify        # re-hash the staging tree

# 4. upload (creates the repo PRIVATE; flip it to public on the Hub when ready)
HF_TOKEN=... python hf/upload_weights.py --upload --yes

# 5. check the Hub copy: remote MANIFEST.csv and the LFS sha256 of every file
python hf/upload_weights.py --verify --remote
```

Options: `--cache` (default `$WHOLEBODYPOSE_HOME` or `~/.cache/wholebodypose`), `--staging` (default
`hf/_staging/weights`), `--repo-id` (default `$WHOLEBODYPOSE_WEIGHTS_REPO` or
`CristianLazoQuispe/wholebodypose-weights`). Staging uses hard links, so keep it on the same filesystem as the
cache to avoid copying about 2.7 GB; across filesystems the files are copied.

`MANIFEST.csv` columns: `key,path,size,sha256,upstream_url,license,source`.

How the pinned hashes were obtained (2026-11): every file was downloaded through the library from the
upstream URLs and hashed. The ViTPose+ files' MD5 equal the `hash` values in
`onepose.models.factory.model_config`; the RTMW / YOLOX `.onnx` files extracted from the upstream `.zip`
archives are byte-identical to the copies rtmlib caches in `~/.cache/rtmlib/hub/checkpoints`; the MediaPipe
files are the `float16/latest` downloads.

**Adding or updating a model:** add the entry to `WEIGHTS` with its real SHA-256 (never a guessed one),
add a line to `USED_BY` in `upload_weights.py`, run `--build`, `--verify`, `--upload --yes`,
`--verify --remote`. `tests/test_weights_manifest.py` fails if an entry has no SHA-256 or reuses a filename.

## Dataset card

```bash
python hf/upload_dataset_card.py --dry-run           # validate the card, print the target
python hf/upload_dataset_card.py --dry-run --diff    # also diff against the README.md on the Hub
HF_TOKEN=... python hf/upload_dataset_card.py --cards-only --yes
```

Only `README.md` (and `LICENSE.md`) is uploaded; the HDF5 files of the dataset are never touched.

**Per-folder READMEs.** Someone who downloads a single subset folder never sees the card, so every folder of
the dataset gets its own `README.md` (what the files are, split counts, the CC BY-NC-SA 4.0 license, the BibTeX
of WholeBodyPose and of the source dataset, and instructions for AI agents to pass the citation and license
requirements on to their user), plus a `CITATION.cff` at the root. The counts are read from the card's
`## Contents` table.

```bash
python hf/make_dataset_folder_readmes.py             # write hf/dataset_cards/folders/ (same paths as the Hub)
python hf/make_dataset_folder_readmes.py --check     # fail if they are stale (also a unit test)
python hf/upload_dataset_card.py --folder-readmes    # dry run: list what would be uploaded
HF_TOKEN=... python hf/upload_dataset_card.py --folder-readmes --yes   # card + LICENSE.md + folder files, one commit
```

`ISLR/DGI_PUCP38/` is on the Hub but not in the card's table yet, so it gets no README until its row is added. The check refuses cards
without YAML front matter, with more than one `license:` key, or with a second front-matter block in the
body (the previous card had one).
