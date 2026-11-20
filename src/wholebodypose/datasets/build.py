"""Video -> ISLR pose HDF5 builder (``wholebodypose-build-islr``).

Pipeline per video: decode every frame with OpenCV -> (optional resize) -> pose backend ->
one person per frame -> keypoints in **original-resolution pixels** -> cached as ``.npz``.
When every video of a dataset subset is cached, the split files are written with the published
schema (:mod:`wholebodypose.datasets.hdf5`): ``x / width``, ``y / height``, plus mid-chest and
mid-hip, ``(T, 2, 135)``.

The cache makes runs resumable (finished videos are skipped) and lets several processes or GPUs
share the work: ``--workers 8 --devices cuda:0,cuda:1`` starts 8 processes, alternating GPUs.
Failures are appended to ``errors.jsonl`` in the cache folder and do not stop the run.

Presets
-------
``icmv2025``  reproduces the published ICMV 2025 data: RTMW performance mode, threshold 2.5,
              noise filter, the first detection, original resolution and the *legacy RGB input*
              (rtmlib expects BGR; the original code passed RGB).
``v1``        the same model and thresholds with the colour bug fixed (BGR to rtmlib) and the
              largest person instead of the first detection. Recommended for new datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from wholebodypose.datasets import hdf5 as H
from wholebodypose.datasets import sources as S

NUM_COCO = 133


@dataclass(frozen=True)
class BuildConfig:
    """Everything that changes the extracted keypoints."""

    backend: str = "rtmpose"
    model_kwargs: dict = field(default_factory=dict)
    resize_width: int | None = None  # resize frames to this width before inference (None: original)
    person: str = "first"  # 'first' detection (paper) or 'largest' bounding box
    safe_midpoints: bool = False  # midpoints only when both endpoints are detected
    save_scores: bool = False  # add per-group 'scores' (T, 135) float32
    compression: str | None = None  # e.g. 'gzip'; None matches the published files

    def cache_tag(self) -> str:
        """Short hash of the settings that affect raw keypoints (cache sub-folder name)."""
        key = json.dumps(
            {"backend": self.backend, "model": self.model_kwargs, "resize": self.resize_width, "person": self.person},
            sort_keys=True,
        )
        return hashlib.sha1(key.encode()).hexdigest()[:10]


_RTM_PAPER = {"mode": "performance", "use_thresholding": True, "filter_noise": True, "kpt_thr": 2.5}
PRESETS: dict[str, BuildConfig] = {
    "icmv2025": BuildConfig(model_kwargs={**_RTM_PAPER, "legacy_rgb_input": True}, person="first"),
    "v1": BuildConfig(model_kwargs={**_RTM_PAPER, "legacy_rgb_input": False}, person="largest"),
}


# --------------------------------------------------------------------------- extraction
def select_person(keypoints: np.ndarray, scores: np.ndarray, person: str = "first"):
    """One ``(133, 2)`` / ``(133,)`` pose from ``(N, 133, 2)`` / ``(N, 133)``; zeros if ``N == 0``."""
    if len(keypoints) == 0:
        return np.zeros((NUM_COCO, 2), np.float32), np.zeros(NUM_COCO, np.float32)
    if person == "first":
        i = 0
    elif person == "largest":
        from wholebodypose.video.video2pose import select_largest_person

        i = select_largest_person(keypoints, scores)
    else:
        raise ValueError(f"person must be 'first' or 'largest', got {person!r}")
    return np.asarray(keypoints[i], np.float32), np.asarray(scores[i], np.float32)


def extract_video(model, path: str, resize_width: int | None = None, person: str = "first", max_frames=None) -> dict:
    """Run ``model`` on every frame of ``path``.

    Returns ``keypoints`` ``(T, 133, 2)`` float32 in pixels of the **original** frame (undetected
    keypoints stay ``(0, 0)``), ``scores`` ``(T, 133)``, and ``width``, ``height``, ``fps`` from
    the container (as in the published files).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    kpts, scs = [], []
    try:
        while max_frames is None or len(kpts) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            sx = sy = 1.0
            if resize_width and resize_width != w:
                nh = int(h * (resize_width / w))
                frame = cv2.resize(frame, (resize_width, nh))
                sx, sy = w / resize_width, h / nh
            k, s = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            k, s = select_person(np.asarray(k), np.asarray(s), person)
            if sx != 1.0 or sy != 1.0:
                k = k * np.array([sx, sy], np.float32)
            kpts.append(k)
            scs.append(s)
    finally:
        cap.release()
    if not kpts:
        raise ValueError(f"no frames decoded from {path}")
    return {
        "keypoints": np.stack(kpts).astype(np.float32),
        "scores": np.stack(scs).astype(np.float32),
        "width": width,
        "height": height,
        "fps": fps,
    }


# --------------------------------------------------------------------------- cache
def cache_key(video_name: str) -> str:
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(video_name).stem)[:60]
    return f"{stem}-{hashlib.sha1(video_name.encode('utf-8')).hexdigest()[:12]}.npz"


def _save_npz(path: Path, res: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, keypoints=res["keypoints"], scores=res["scores"], width=res["width"], height=res["height"],
             fps=res.get("fps", 0.0))  # fmt: skip
    os.replace(tmp, path)


def load_cached(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- workers
_WORKER: dict = {}


def _make_model(config: BuildConfig, device: str, verbose: bool = False):
    from wholebodypose.api import create_model

    return create_model(config.backend, device=device, verbose=verbose, **config.model_kwargs)


def _worker_init(config: BuildConfig, devices: list[str], counter) -> None:
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    dev = devices[i % len(devices)]
    if dev.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = dev.split(":", 1)[1]
        dev = "cuda"
    _WORKER["model"] = _make_model(config, dev)
    _WORKER["config"] = config


def _process(job: tuple[str, str, str]) -> tuple[str, str | None, int]:
    video_path, video_name, out = job
    cfg: BuildConfig = _WORKER["config"]
    try:
        res = extract_video(_WORKER["model"], video_path, cfg.resize_width, cfg.person)
        _save_npz(Path(out), res)
        return video_name, None, len(res["keypoints"])
    except Exception as e:  # noqa: BLE001 - logged and reported, the run continues
        return video_name, f"{type(e).__name__}: {e}", 0


class _Progress:
    def __init__(self, total: int, every: float = 10.0, stream=sys.stderr) -> None:
        self.total, self.done, self.err, self.t0, self.last, self.every = total, 0, 0, time.time(), 0.0, every
        self.stream = stream

    def update(self, ok: bool) -> None:
        self.done += 1
        self.err += 0 if ok else 1
        now = time.time()
        if now - self.last >= self.every or self.done == self.total:
            self.last = now
            rate = self.done / max(now - self.t0, 1e-9)
            eta = (self.total - self.done) / rate if rate else float("inf")
            print(f"[build-islr] {self.done}/{self.total} videos, {self.err} errors, "
                  f"{rate:.2f} vid/s, ETA {eta / 60:.1f} min", file=self.stream, flush=True)  # fmt: skip


def extract_all(
    samples: list[S.Sample],
    cache_dir: str | Path,
    config: BuildConfig,
    model=None,
    workers: int = 1,
    devices: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Extract (or reuse) the cached keypoints of every unique video; returns name -> cache file.

    ``model``: an already-built model (single process; used by tests). Otherwise ``workers``
    processes each build their own model on ``devices`` (round-robin).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    jobs = []
    for s in samples:
        if s.video_name in files:
            continue
        f = cache_dir / cache_key(s.video_name)
        files[s.video_name] = f
        if not f.exists():
            jobs.append((s.video_path, s.video_name, str(f)))
    done_log = cache_dir / "done.txt"
    err_log = cache_dir / "errors.jsonl"
    if verbose:
        print(f"[build-islr] {len(files)} videos, {len(files) - len(jobs)} cached, {len(jobs)} to extract "
              f"-> {cache_dir}", file=sys.stderr, flush=True)  # fmt: skip
    if not jobs:
        return files
    prog = _Progress(len(jobs)) if verbose else None

    def _record(name, err, T):
        if err:
            with open(err_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"video_name": name, "error": err, "time": time.time()}) + "\n")
        else:
            with open(done_log, "a", encoding="utf-8") as fh:
                fh.write(f"{name}\t{T}\n")
        if prog:
            prog.update(err is None)

    if model is not None or workers <= 1:
        if model is None:
            dev = (devices or ["auto"])[0]
            _worker_init(config, [dev], _Counter())
        else:
            _WORKER.update(model=model, config=config)
        for job in jobs:
            _record(*_process(job))
        return files

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ctx.Pool(workers, initializer=_worker_init, initargs=(config, devices or ["auto"], counter)) as pool:
        for res in pool.imap_unordered(_process, jobs, chunksize=1):
            _record(*res)
    return files


class _Counter:  # stand-in for multiprocessing.Value in the single-process path
    def __init__(self) -> None:
        self.value = 0

    def get_lock(self):
        import contextlib

        return contextlib.nullcontext()


# --------------------------------------------------------------------------- assembly
def reference_order(samples: list[S.Sample], reference_hdf5: str | Path) -> list[S.Sample]:
    """Reorder ``samples`` like the groups of a reference file (unknown samples go last)."""
    pos = {d["video_name"]: d["index"] for d in H.iter_samples(reference_hdf5, with_data=False)}
    big = len(pos) + len(samples)
    keyed = sorted(enumerate(samples), key=lambda t: pos.get(t[1].video_name, big + t[0]))
    return [s for _, s in keyed]


def write_split(
    samples: list[S.Sample], cache_files: dict[str, Path], out_path: str | Path, config: BuildConfig
) -> dict:
    """Write one split file from cached keypoints, in ``samples`` order. Returns a small report."""
    h5py = H._h5py()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.hdf5")
    written, missing = 0, []
    with h5py.File(tmp, "w") as h5:
        for s in samples:
            f = cache_files.get(s.video_name)
            if f is None or not f.exists():
                missing.append(s.video_name)
                continue
            c = load_cached(f)
            w, h = int(c["width"]), int(c["height"])
            data = H.to_published_array(c["keypoints"], w, h, safe_midpoints=config.safe_midpoints)
            scores = H.scores_135(c["scores"], config.safe_midpoints) if config.save_scores else None
            H.write_sample(h5, written, data, s.label, s.video_name, w, h, scores=scores,
                           compression=config.compression)  # fmt: skip
            written += 1
    os.replace(tmp, out_path)
    return {"file": str(out_path), "samples": written, "missing": missing}


def build_dataset(
    dataset: str,
    subset: int | None,
    out_dir: str | Path,
    root: str | Path | None = None,
    config: BuildConfig | str = "icmv2025",
    splits: tuple[str, ...] = S.SPLITS,
    cache_dir: str | Path | None = None,
    workers: int = 1,
    devices: list[str] | None = None,
    model=None,
    limit: int | None = None,
    order_like: str | Path | None = None,
    source_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Build ``<prefix>_135-{Train,Val,Test}.hdf5`` and ``*_maplabels.json`` for one subset.

    ``order_like``: folder with published files of the same subset; groups are then written in
    the published order (only needed for WLASL, whose original order came from a directory walk).
    ``limit``: first N samples of each split (smoke tests).
    """
    key = S.canonical(dataset)
    subset = S.default_subset(key) if subset is None else int(subset)
    if isinstance(config, str):
        config = PRESETS[config]
    samples, label_map = S.load_samples(key, subset, root, **(source_kwargs or {}))
    per_split = S.by_split(samples)
    prefix = S.published_prefix(key, subset)
    chosen = {}
    for sp in splits:
        items = per_split[sp]
        if order_like:
            ref = Path(order_like) / H.split_filename(prefix, sp)
            if ref.exists():
                items = reference_order(items, ref)
        chosen[sp] = items[:limit] if limit else items
    cache_dir = Path(cache_dir) if cache_dir else Path(out_dir) / ".cache" / f"{key}-{config.cache_tag()}"
    every = [s for sp in splits for s in chosen[sp]]
    files = extract_all(every, cache_dir, config, model=model, workers=workers, devices=devices, verbose=verbose)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": key, "subset": subset, "config": asdict(config), "splits": {}}
    for sp in splits:
        report["splits"][sp] = write_split(chosen[sp], files, out_dir / H.split_filename(prefix, sp), config)
    with open(out_dir / S.maplabels_filename(key, subset), "w", encoding="utf-8") as f:
        json.dump(S.maplabels_json(label_map), f, indent=4)
    if verbose:
        for r in report["splits"].values():
            print(f"[build-islr] {r['file']}: {r['samples']} samples, {len(r['missing'])} missing", file=sys.stderr)
    return report


def verify_against(built_dir: str | Path, published_dir: str | Path, prefix: str, splits=S.SPLITS) -> list[dict]:
    """Compare built files with published ones, matching samples by ``video_name``."""
    rows = []
    for sp in splits:
        a, b = Path(published_dir) / H.split_filename(prefix, sp), Path(built_dir) / H.split_filename(prefix, sp)
        if not (a.exists() and b.exists()):
            continue
        ref = {d["video_name"]: d for d in H.iter_samples(a)}
        for d in H.iter_samples(b):
            r = ref.get(d["video_name"])
            if r is None:
                rows.append({"split": sp, "video_name": d["video_name"], "in_reference": False})
                continue
            m = H.compare_arrays(r["data"], d["data"])
            m.update(split=sp, video_name=d["video_name"], in_reference=True, label_match=r["label"] == d["label"],
                     size_match=(r["width"], r["height"]) == (d["width"], d["height"]))  # fmt: skip
            rows.append(m)
    return rows


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wholebodypose-build-islr",
        description="Build ISLR pose HDF5 files (published ICMV 2025 schema) from raw videos.",
    )
    p.add_argument("--dataset", required=True, help=f"one of {sorted(S.SPECS)}")
    p.add_argument("--subset", type=int, default=None, help="vocabulary size, e.g. 100 / 300 / 2000 / 2731")
    p.add_argument("--root", default=None, help="dataset root (videos + metadata); see docs/datasets")
    p.add_argument("--annotations", default=None, help="MS-ASL: folder with MSASL_{train,val,test}.json")
    p.add_argument("--out", default=None, help="output folder")
    p.add_argument("--preset", default="icmv2025", choices=sorted(PRESETS))
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda (single process)")
    p.add_argument("--devices", default=None, help="comma list for workers, e.g. cuda:0,cuda:1")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cache-dir", default=None, help="per-video cache (resumable); default <out>/.cache/...")
    p.add_argument("--limit", type=int, default=None, help="first N samples per split (smoke test)")
    p.add_argument("--order-like", default=None, help="folder with published files: copy their group order")
    p.add_argument("--resize-width", type=int, default=None, help="override preset (0 = original size)")
    p.add_argument("--person", choices=["first", "largest"], default=None, help="override preset")
    p.add_argument("--kpt-thr", type=float, default=None, help="override preset threshold")
    rgb = p.add_mutually_exclusive_group()
    rgb.add_argument("--legacy-rgb-input", dest="legacy_rgb", action="store_true", default=None)
    rgb.add_argument("--no-legacy-rgb-input", dest="legacy_rgb", action="store_false")
    p.add_argument("--safe-midpoints", action="store_true", help="midpoints only when both endpoints exist")
    p.add_argument("--save-scores", action="store_true", help="add per-group 'scores' (T, 135)")
    p.add_argument("--compression", default=None, help="HDF5 compression, e.g. gzip (default: none)")
    p.add_argument("--list", action="store_true", help="only print the sample counts per split")
    p.add_argument("--verify", default=None, help="folder with published files: compare after building")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def config_from_args(args) -> BuildConfig:
    cfg = PRESETS[args.preset]
    mk = dict(cfg.model_kwargs)
    if args.kpt_thr is not None:
        mk["kpt_thr"] = args.kpt_thr
    if args.legacy_rgb is not None:
        mk["legacy_rgb_input"] = args.legacy_rgb
    upd = {"model_kwargs": mk}
    if args.resize_width is not None:
        upd["resize_width"] = args.resize_width or None
    if args.person:
        upd["person"] = args.person
    upd.update(safe_midpoints=args.safe_midpoints, save_scores=args.save_scores, compression=args.compression)
    return replace(cfg, **upd)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    key = S.canonical(args.dataset)
    subset = S.default_subset(key) if args.subset is None else args.subset
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    kw = {"annotations": args.annotations} if args.annotations and key in ("msasl", "wlasl") else {}
    if args.list:
        samples, labels = S.load_samples(key, subset, args.root, **kw)
        per = S.by_split(samples)
        for sp in splits:
            print(f"{S.published_prefix(key, subset)} {sp}: {len(per[sp])} samples, "
                  f"{len({s.label for s in per[sp]})} classes")  # fmt: skip
        print(f"label map: {len(labels)} glosses")
        return 0
    if not args.out:
        print("--out is required (unless --list)", file=sys.stderr)
        return 2
    config = config_from_args(args)
    devices = [d.strip() for d in args.devices.split(",")] if args.devices else [args.device]
    build_dataset(
        key, subset, args.out, root=args.root, config=config, splits=splits, cache_dir=args.cache_dir,
        workers=args.workers, devices=devices, limit=args.limit, order_like=args.order_like,
        source_kwargs=kw, verbose=not args.quiet,
    )  # fmt: skip
    if args.verify:
        rows = [r for r in verify_against(args.out, args.verify, S.published_prefix(key, subset), splits)
                if r.get("in_reference")]  # fmt: skip
        if rows:
            agg = {k: float(np.nanmedian([r[k] for r in rows])) for k in ("median_abs_diff", "detection_agreement")}
            tm = np.mean([r["T_match"] for r in rows])
            print(f"verify: {len(rows)} samples, T match {tm:.1%}, median |diff| {agg['median_abs_diff']:.2e}, "
                  f"detection agreement {agg['detection_agreement']:.4f}")  # fmt: skip
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
