"""Dataset adapters: raw videos + official metadata -> the sample lists of the published HDF5 files.

Each adapter returns a list of :class:`Sample` in the exact order of the published files (group
"0" is the first sample of the split), so that a rebuild writes the same groups with the same
``video_name`` and ``label``. The rules below were reverse-engineered from the scripts that made
the ICMV 2025 data and checked against every published file (see ``docs/datasets``).

==============  =====================  ===================================================
dataset         subsets                metadata used (``root`` defaults in ``DEFAULT_ROOTS``)
==============  =====================  ===================================================
``wlasl``       100, 300, 1000, 2000   ``WLASL/info.json`` (== ``WLASL_v0.3.json``), 256x256 clips
``msasl``       100, 200, 500, 1000    ``MSASL/msasl{N}_metadata.csv`` + clips in ``MSASL/{split}``
``asl_citizen`` 100, 300, 1000, 2731   ``ASL_Citizen/splits/{train,val,test}.csv``
``lsa64``       64                     ``lsa64_full_metadata.csv`` (signer 9 = val, 10 = test)
``pucp_dgi``    30                     ``DGI305AEC_SignERT_metadata.csv`` (30-gloss subset)
==============  =====================  ===================================================
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    """One video of a dataset split."""

    video_path: str  # where the video is read from
    video_name: str  # string stored in the HDF5 ``video_name`` (as published)
    label: str  # gloss stored in the HDF5 ``label``
    split: str  # 'train' | 'val' | 'test'


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    subsets: tuple[int, ...]
    folder: str  # published folder under ISLR/
    prefix: str  # published file prefix, formatted with the subset
    maplabels: str  # published label-map filename, formatted with the subset
    nested: bool  # published files live in ISLR/<folder>/<prefix>/ (True) or ISLR/<folder>/ (False)


SPECS: dict[str, DatasetSpec] = {
    "wlasl": DatasetSpec("wlasl", (100, 300, 1000, 2000), "WLASL", "WLASL{n}", "wlasl_{n}_maplabels.json", True),
    "msasl": DatasetSpec("msasl", (100, 200, 500, 1000), "MSASL", "MSASL{n}", "msasl_{n}_maplabels.json", True),
    "asl_citizen": DatasetSpec(
        "asl_citizen", (100, 300, 1000, 2731), "ASLCITIZEN", "ASLCitizen{n}", "asl_citizen_{n}_maplabels.json", True
    ),
    "lsa64": DatasetSpec("lsa64", (64,), "LSA64", "LSA64", "lsa64_64_maplabels.json", False),
    "pucp_dgi": DatasetSpec("pucp_dgi", (30,), "DGI_PUCP30", "DGI_PUCP{n}", "dgi_pucp_{n}_maplabels.json", False),
}
ALIASES = {"asl-citizen": "asl_citizen", "aslcitizen": "asl_citizen", "dgi_pucp": "pucp_dgi", "pucp-dgi": "pucp_dgi"}

# Subsets that were published on Hugging Face (the others can be built but have no reference).
PUBLISHED = {
    "wlasl": (100, 300, 2000),
    "msasl": (100, 200, 1000),
    "asl_citizen": (100, 300, 2731),
    "lsa64": (64,),
    "pucp_dgi": (30,),
}

DEFAULT_ROOTS = {
    "wlasl": "/data/cristian/paper_2025/WLASL_videos/WLASL",
    "msasl": "/data/MSASL",
    "asl_citizen": "/data/ASL-Citizen/ASL_Citizen",
    "lsa64": "/data/LSA64_dataset",
    "pucp_dgi": "/data/DGI305AEC_SignERT",
}
# Official MS-ASL json files (MSASL_{train,val,test}.json); /data/MSASL/*_metadata.csv do NOT match
# the downloaded file names (they use ``clean_text`` instead of ``text``).
DEFAULT_MSASL_ANNOTATIONS = "/home/va0831/Projects/2027/LargeScaleSLR/data/videos/msasl"

# The 30 PUCP-DGI glosses used in the paper (DGI305AEC_SignERT has 38).
PUCP_DGI_30 = (
    "amigo", "ayudar", "casa", "cuanto", "dinero", "dormir", "escribir", "idea", "saber", "sentir",
    "sol", "tres", "ya", "cien", "ir", "ahora", "comprar", "decir", "esperar", "guardar",
    "hacer", "niño", "pensar", "comer", "ver", "hola", "no", "yo", "igual", "mi",
)  # fmt: skip

# MS-ASL clip that is listed in the metadata but was never downloaded (start == end frame).
MSASL_MISSING = ("xfVNJxKWqeg_004007_004007-msasl-right.mp4",)


def canonical(dataset: str) -> str:
    key = dataset.lower().replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in SPECS:
        raise ValueError(f"unknown dataset {dataset!r}; choose one of {sorted(SPECS)}")
    return key


def default_subset(dataset: str) -> int:
    return SPECS[canonical(dataset)].subsets[0]


def published_prefix(dataset: str, subset: int) -> str:
    """File prefix of the published HDF5, e.g. ``('wlasl', 100) -> 'WLASL100'``."""
    return SPECS[canonical(dataset)].prefix.format(n=subset)


def published_dir(islr_root: str | Path, dataset: str, subset: int) -> Path:
    """Folder that holds the published files, e.g. ``ISLR/WLASL/WLASL100``."""
    spec = SPECS[canonical(dataset)]
    d = Path(islr_root) / spec.folder
    return d / spec.prefix.format(n=subset) if spec.nested else d


def maplabels_filename(dataset: str, subset: int) -> str:
    return SPECS[canonical(dataset)].maplabels.format(n=subset)


def _read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stable_sort(rows: list, key) -> list:
    return sorted(rows, key=key)  # Python's sort is stable, like pandas' multi-column sort_values


# --------------------------------------------------------------------------- WLASL
def wlasl(subset: int = 100, root: str | Path | None = None, annotations: str | Path | None = None):
    """WLASL v0.3: the instances of the first ``subset`` glosses of ``WLASL_v0.3.json``.

    ``root`` holds ``videos/<id>.mp4`` (the 256x256 / 25 fps letterboxed clips of
    ``wlasl-processed.zip`` == Kaggle ``quanbkit1/wlasl-rgb-video``) and ``info.json``.
    ``video_name`` follows the Kaggle layout ``rgb/WLASL{N}/{split}/{id}.mp4``.

    Order: gloss order of the json, then instance order. The published files use a different
    (unrecoverable) group order; pass ``order_like=`` to :func:`build_dataset` to copy it.
    """
    root = Path(root or DEFAULT_ROOTS["wlasl"])
    with open(annotations or root / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    samples, labels = [], {}
    for entry in info[:subset]:
        gloss = entry["gloss"]
        labels.setdefault(gloss, len(labels))
        for inst in entry["instances"]:
            vid, split = inst["video_id"], inst["split"]
            samples.append(
                Sample(str(root / "videos" / f"{vid}.mp4"), f"rgb/WLASL{subset}/{split}/{vid}.mp4", gloss, split)
            )
    return samples, labels


# --------------------------------------------------------------------------- MS-ASL
def _msasl_resolve(path: str, listing: dict[str, list[str]]) -> str:
    """Map a metadata path to the downloaded file whose YouTube id carries a URL suffix.

    E.g. ``train/TwkGS9TjUX8_014261_014440-msasl-grandmother.mp4`` was saved as
    ``train/TwkGS9TjUX8&t=21s_014261_014440-msasl-grandmother.mp4`` and ``not like`` as
    ``not_like``. Same rule as the original notebook (11-character YouTube id prefix).
    """
    d, name = os.path.split(path)
    prefix, suffix = name[:11], name[12:].replace(" ", "_")
    for f in listing.get(d, []):
        if f.startswith(prefix) and "_".join(f[11:].split("_")[1:]) == suffix:
            return os.path.join(d, f)
    return path


def msasl_metadata(annotations: str | Path) -> list[dict]:
    """Rows of the official MS-ASL ``MSASL_{train,val,test}.json`` with the clip file names.

    A clip of ``url=...watch?v=<id>``, frames ``start``-``end`` and text ``text`` is stored as
    ``MSASL/<split>/<id>_<start:06d>_<end:06d>-msasl-<text>.mp4`` (the downloader's naming; note
    that it uses ``text``, not ``clean_text``). The gloss is ``clean_text``.
    """
    rows = []
    for split in SPLITS:
        with open(Path(annotations) / f"MSASL_{split}.json", encoding="utf-8") as f:
            for item in json.load(f):
                vid = item["url"].split("watch?v=")[-1].split("&")[0]
                fname = f"{vid}_{item['start']:06d}_{item['end']:06d}-msasl-{item['text']}.mp4"
                rows.append(
                    {"subset": split, "gloss": item["clean_text"], "label": item["label"],
                     "relative_path": f"MSASL/{split}/{fname}"}
                )  # fmt: skip
    return rows


def msasl(
    subset: int = 100,
    root: str | Path | None = None,
    annotations: str | Path | None = None,
    check_exists: bool = True,
):
    """MS-ASL: official train/val/test json restricted to ``label < subset``.

    ``annotations`` is the folder with ``MSASL_{train,val,test}.json`` (default: ``root``, then
    :data:`DEFAULT_MSASL_ANNOTATIONS`). Rules of the paper: gloss = ``clean_text`` of the label's
    last row; paths whose YouTube id was saved with a URL suffix (``&t=21s``) are resolved on disk;
    duplicated train paths are kept once (last); the never-downloaded clip in
    :data:`MSASL_MISSING` is dropped. ``video_name`` is ``/data/MSASL/{split}/<file>`` as published.

    Note: the published ``MSASL200`` is subset 200 (it was called group "300" internally).
    """
    root = Path(root or DEFAULT_ROOTS["msasl"])
    if annotations is None:
        annotations = root if (root / "MSASL_train.json").exists() else DEFAULT_MSASL_ANNOTATIONS
    rows = [r for r in msasl_metadata(annotations) if int(r["label"]) < subset]
    label2gloss = {int(r["label"]): r["gloss"] for r in rows}  # last occurrence wins (dict(zip))
    listing: dict[str, list[str]] = {}
    out = []
    for r in rows:
        rel = r["relative_path"]  # 'MSASL/train/<file>.mp4'
        local = str(root / Path(rel).relative_to("MSASL")) if rel.startswith("MSASL/") else str(root / rel)
        if check_exists and not os.path.exists(local):
            d = os.path.dirname(local)
            if d not in listing:
                listing[d] = sorted(os.listdir(d)) if os.path.isdir(d) else []
            local = _msasl_resolve(local, listing)
        name = "/data/MSASL/" + os.path.relpath(local, root)
        out.append((r["subset"], int(r["label"]), local, name))
    train = {}
    for s, lab, local, name in out:  # drop_duplicates(keep='last') keeps the position of the last row
        if s == "train":
            train.pop(name, None)
            train[name] = (s, lab, local, name)
    rows2 = list(train.values()) + [o for o in out if o[0] != "train"]
    samples = [
        Sample(local, name, label2gloss[lab], s)
        for s, lab, local, name in rows2
        if os.path.basename(name) not in MSASL_MISSING
    ]
    labels = {label2gloss[k]: k for k in sorted(label2gloss)}
    return samples, labels


# --------------------------------------------------------------------------- ASL Citizen
def _asl_citizen_rows(root: Path) -> list[tuple[str, str, str]]:
    rows = []
    for split in SPLITS:
        for r in _read_csv(root / "splits" / f"{split}.csv"):
            rows.append((split, r["Video file"], r["Gloss"]))
    return rows


def _top_glosses(rows, n: int) -> set[str]:
    counts = Counter(g for _, _, g in rows)
    return {g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]}


def asl_citizen(subset: int = 100, root: str | Path | None = None):
    """ASL Citizen official splits; subsets are the ``subset`` most frequent glosses (ties: name).

    Samples are sorted (stably) by gloss name, as in the paper. Label map: the published maps share
    one global numbering (subsets 100 -> 1000 -> 2731 -> 300 in string order), so
    ``ASLCitizen300``'s labels are **not** contiguous (0..999); it is reproduced as published.
    """
    root = Path(root or DEFAULT_ROOTS["asl_citizen"])
    rows = _asl_citizen_rows(root)
    order = {g: i for i, g in enumerate(sorted({g for _, _, g in rows}))}
    groups = {}
    for n in sorted({100, 300, 1000, 2731, subset}, key=str):  # '100' < '1000' < '2731' < '300'
        keep = _top_glosses(rows, n) if n < len(order) else set(order)
        groups[n] = _stable_sort([r for r in rows if r[2] in keep], key=lambda r: order[r[2]])
    global_map: dict[str, int] = {}
    for n in groups:
        for _, _, g in groups[n]:
            global_map.setdefault(g, len(global_map))
    pub = "/data/ASL-Citizen/ASL_Citizen/videos/"
    samples = [Sample(str(root / "videos" / f), pub + f, g, s) for s, f, g in groups[subset]]
    glosses = sorted({s.label for s in samples}, key=global_map.get)
    return samples, {g: global_map[g] for g in glosses}


# --------------------------------------------------------------------------- LSA64
def lsa64(subset: int = 64, root: str | Path | None = None):
    """LSA64 (64 signs x 10 signers x 5 repetitions); split by signer: 9 = val, 10 = test.

    Files are ``SSS_PPP_RRR.mp4`` (sign, signer/person, repetition). The sign ``None`` is read as
    a missing value by pandas and was published as ``None_``; that spelling is kept.
    """
    if subset != 64:
        raise ValueError("LSA64 only has subset 64")
    root = Path(root or DEFAULT_ROOTS["lsa64"])
    samples, labels = [], {}
    for r in _read_csv(root / "lsa64_full_metadata.csv"):
        gloss = r["gloss"] if r["gloss"] not in ("", "None", "NaN", "nan", "NA", "null") else "None_"
        fname = r["relative_path"].replace("\\", "/").split("/")[-1]
        signer = int(fname.split("_")[1])
        split = "test" if signer == 10 else "val" if signer == 9 else "train"
        labels.setdefault(gloss, int(r["label"]))
        samples.append(Sample(str(root / "LSA64" / fname), f"/data/LSA64_dataset/LSA64/{fname}", gloss, split))
    return samples, labels


# --------------------------------------------------------------------------- PUCP-DGI
def pucp_dgi(subset: int = 30, root: str | Path | None = None):
    """PUCP-DGI305 (DGI305AEC_SignERT) restricted to the 30 glosses of :data:`PUCP_DGI_30`.

    Files were renamed ``<gloss>_<k>.mp4 -> <gloss>_<k>_<row id>.mp4`` inside ``split/<split>/``.
    Samples are sorted (stably) by the original label; labels are re-numbered 0..29 in that order.
    """
    if subset != 30:
        raise ValueError("PUCP-DGI only has subset 30")
    root = Path(root or DEFAULT_ROOTS["pucp_dgi"])
    rows = []
    for r in _read_csv(root / "DGI305AEC_SignERT_metadata.csv"):
        if r["gloss"] not in PUCP_DGI_30:
            continue
        stem = Path(r["relative_path"]).stem
        fname = f"{stem}_{r['id']}.mp4"
        rows.append((int(r["label"]), r["gloss"], r["subset"], fname))
    rows = _stable_sort(rows, key=lambda r: r[0])
    labels: dict[str, int] = {}
    samples = []
    for _, gloss, split, fname in rows:
        labels.setdefault(gloss, len(labels))
        samples.append(
            Sample(
                str(root / "split" / split / fname),
                f"/data/DGI305AEC_SignERT/split/{split}/{fname}",
                gloss,
                split,
            )
        )
    return samples, labels


ADAPTERS = {"wlasl": wlasl, "msasl": msasl, "asl_citizen": asl_citizen, "lsa64": lsa64, "pucp_dgi": pucp_dgi}


def load_samples(dataset: str, subset: int | None = None, root: str | Path | None = None, **kwargs):
    """``(samples, label_map)`` for a dataset subset; ``label_map`` is gloss -> int as published."""
    key = canonical(dataset)
    subset = default_subset(key) if subset is None else int(subset)
    if subset not in SPECS[key].subsets:
        raise ValueError(f"{key} subsets are {SPECS[key].subsets}, got {subset}")
    return ADAPTERS[key](subset, root=root, **kwargs)


def iter_samples(dataset: str, subset: int | None = None, root: str | Path | None = None, split: str | None = None):
    """Yield ``(video_path, video_name, label, split)`` tuples (optionally for one split)."""
    samples, _ = load_samples(dataset, subset, root)
    for s in samples:
        if split is None or s.split == split:
            yield s.video_path, s.video_name, s.label, s.split


def by_split(samples: list[Sample]) -> dict[str, list[Sample]]:
    out: dict[str, list[Sample]] = {s: [] for s in SPLITS}
    for s in samples:
        out[s.split].append(s)
    return out


def maplabels_json(label_map: dict[str, int]) -> dict:
    """Published ``*_maplabels.json`` content (``id_to_label`` is gloss -> int despite its name)."""
    items = sorted(label_map.items(), key=lambda kv: kv[1])
    return {"id_to_label": dict(items), "label_to_id": {str(v): k for k, v in items}}
