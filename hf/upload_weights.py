"""Stage, upload and verify the WholeBodyPose model-weights mirror on the Hugging Face Hub.

The mirror (``CristianLazoQuispe/wholebodypose-weights`` by default, i.e. ``WEIGHTS_REPO`` in
``wholebodypose.weights``) holds byte-identical copies of every third-party file listed in
``wholebodypose.weights.WEIGHTS``, at the same relative path the library uses for its local cache
(``rtmpose/...onnx``, ``vitpose/...pth``, ``mediapipe/...task``). Nothing is re-trained or converted.

Modes (they can be combined; with none, ``--dry-run`` is assumed):

``--dry-run``
    Print the plan: every file, where it is in the local cache, its size and whether its SHA-256
    matches the pinned value. Writes nothing.
``--build``
    Build ``hf/_staging/weights`` (``--staging``): hard links (copies across filesystems) of the
    cached files at their mirror paths, ``MANIFEST.csv`` (key, path, size, sha256, upstream_url,
    license, source) and the model card ``README.md`` (from ``hf/model_card.md``). Every file is
    hashed and must match the pin in ``WEIGHTS``; ``--fetch`` downloads missing files first through
    the library (mirror, then upstream).
``--upload``
    Create the repo (always ``private=True``, ``exist_ok=True``) and ``upload_folder`` the staging
    tree. Needs ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN`` / ``--env-file``) and ``--yes``. Making the repo
    public is a manual step on the Hub.
``--verify``
    Without ``--remote``: re-hash the staging tree and compare it with ``WEIGHTS`` and the staged
    ``MANIFEST.csv``. With ``--remote``: download ``MANIFEST.csv`` from the Hub and compare it, and
    the LFS sha256 / size of every remote file, with ``WEIGHTS``.

    python hf/upload_weights.py --dry-run
    python hf/upload_weights.py --build [--fetch]
    python hf/upload_weights.py --verify
    HF_TOKEN=... python hf/upload_weights.py --upload --yes
    python hf/upload_weights.py --verify --remote

The token is never printed.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from wholebodypose.weights import WEIGHTS, WEIGHTS_REPO, WeightFile, cache_dir, get_weight, sha256sum  # noqa: E402

DEFAULT_STAGING = HERE / "_staging" / "weights"
MODEL_CARD = HERE / "model_card.md"
FILES_MARKER = "<!-- FILES_TABLE -->"
MANIFEST_FIELDS = ["key", "path", "size", "sha256", "upstream_url", "license", "source"]

# Which library setting loads each file (shown in the model card).
USED_BY = {
    "yolox_m": "`rtmpose` detector, modes `performance` (paper default) and `balanced`",
    "yolox_tiny": "`rtmpose` detector, mode `lightweight`",
    "rtmw_x_384": "`rtmpose` pose model, mode `performance` (paper default)",
    "rtmw_x_256": "`rtmpose` pose model, mode `balanced`",
    "rtmw_l_256": "`rtmpose` pose model, mode `lightweight`",
    "vitpose_small": "`vitpose`, `model_name='ViTPose+_small_coco_wholebody'`",
    "vitpose_base": "`vitpose`, `model_name='ViTPose+_base_coco_wholebody'`",
    "vitpose_large": "`vitpose`, `model_name='ViTPose+_large_coco_wholebody'`",
    "vitpose_huge": "`vitpose` (paper default)",
    "vitpose_small_onnx": "`vitpose`, ONNX engine, `model_name='ViTPose+_small_coco_wholebody'`",
    "vitpose_base_onnx": "`vitpose`, ONNX engine, `model_name='ViTPose+_base_coco_wholebody'`",
    "vitpose_large_onnx": "`vitpose`, ONNX engine, `model_name='ViTPose+_large_coco_wholebody'`",
    "vitpose_huge_onnx": "`vitpose`, ONNX engine (paper model)",
    "mp_pose_heavy": "`mediapipe` body, `model_complexity=2` (paper default)",
    "mp_pose_full": "`mediapipe` body, `model_complexity` < 2",
    "mp_hand": "`mediapipe` hands",
    "mp_face": "`mediapipe` face",
}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def _source(cache: Path, w: WeightFile, fetch: bool) -> Path | None:
    src = cache / w.filename
    if not src.exists() and fetch:
        old = os.environ.get("WHOLEBODYPOSE_HOME")
        os.environ["WHOLEBODYPOSE_HOME"] = str(cache)
        try:
            get_weight(w.key, verbose=True)
        finally:
            if old is None:
                os.environ.pop("WHOLEBODYPOSE_HOME", None)
            else:
                os.environ["WHOLEBODYPOSE_HOME"] = old
    return src if src.exists() else None


def _check_registry() -> list[str]:
    problems = []
    names = [w.filename for w in WEIGHTS.values()]
    if len(set(names)) != len(names):
        problems.append("duplicate filenames in WEIGHTS")
    for w in WEIGHTS.values():
        if not w.sha256:
            problems.append(f"{w.key}: no pinned sha256 in WEIGHTS")
    return problems


def dry_run(args) -> int:
    cache = Path(args.cache)
    print(f"repo     : {args.repo_id} (model, private on creation)")
    print(f"cache    : {cache}")
    print(f"staging  : {args.staging}")
    total, missing, bad = 0, 0, 0
    for w in WEIGHTS.values():
        src = cache / w.filename
        if not src.exists():
            missing += 1
            print(f"  MISSING  {w.filename}")
            continue
        size = src.stat().st_size
        total += size
        status = "ok" if not args.hash else ("sha256 ok" if sha256sum(src) == w.sha256 else "SHA256 MISMATCH")
        bad += status == "SHA256 MISMATCH"
        print(f"  {status:9s} {human(size):>9s}  {w.filename}")
    problems = _check_registry()
    for p in problems:
        print(f"  PROBLEM  {p}")
    print(f"{len(WEIGHTS)} files, {human(total)} present, {missing} missing, {bad} mismatched")
    print(
        "plan     : --build -> staging tree + MANIFEST.csv + README.md; --upload --yes -> create_repo + upload_folder"
    )
    return int(bool(missing or bad or problems))


def _link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return "linked"
        dst.unlink()
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def files_table(rows: list[dict]) -> str:
    lines = [
        "| File | Size | SHA-256 | Used by | Upstream |",
        "|:--|--:|:--|:--|:--|",
    ]
    for r in rows:
        upstream = f"[link]({r['upstream_url']})" if r["upstream_url"] else "exported by WholeBodyPose"
        lines.append(
            f"| `{r['path']}` | {human(int(r['size']))} | `{r['sha256'][:12]}…` | {USED_BY.get(r['key'], '')} "
            f"| {upstream} |"
        )
    return "\n".join(lines)


def build(args) -> int:
    problems = _check_registry()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    cache, staging = Path(args.cache), Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for w in WEIGHTS.values():
        src = _source(cache, w, args.fetch)
        if src is None:
            failed.append(f"{w.key}: missing in {cache} (use --fetch)")
            continue
        digest = sha256sum(src)
        if digest != w.sha256:
            failed.append(f"{w.key}: sha256 {digest} != pinned {w.sha256}")
            continue
        how = _link(src, staging / w.filename)
        size = src.stat().st_size
        print(f"  {how:6s} {human(size):>9s}  {w.filename}")
        rows.append(
            {
                "key": w.key,
                "path": w.filename,
                "size": size,
                "sha256": digest,
                "upstream_url": w.upstream or "",
                "license": w.license,
                "source": w.source,
            }
        )
    if failed:
        print("\n".join(failed), file=sys.stderr)
        return 1
    with open(staging / "MANIFEST.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    card = MODEL_CARD.read_text()
    if FILES_MARKER not in card:
        print(f"{MODEL_CARD} has no {FILES_MARKER} marker", file=sys.stderr)
        return 1
    (staging / "README.md").write_text(card.replace(FILES_MARKER, files_table(rows)))
    total = sum(int(r["size"]) for r in rows)
    print(f"staged {len(rows)} files ({human(total)}) + MANIFEST.csv + README.md in {staging}")
    return 0


def _token(env_file: str | None) -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok and env_file and Path(env_file).exists():
        for ln in Path(env_file).read_text().splitlines():
            for k in ("HUGGINGFACE_TOKEN=", "HF_TOKEN="):
                if ln.strip().startswith(k):
                    tok = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        raise SystemExit("no HF_TOKEN / HUGGINGFACE_TOKEN in the environment or --env-file")
    return tok


def _hub():
    try:
        import huggingface_hub
    except ImportError as e:
        raise SystemExit("this mode needs: pip install huggingface_hub") from e
    return huggingface_hub


def upload(args) -> int:
    if not args.yes:
        print("--upload writes to the Hugging Face Hub; re-run with --yes to confirm", file=sys.stderr)
        return 2
    staging = Path(args.staging)
    if not (staging / "MANIFEST.csv").exists():
        print(f"{staging} is not built; run --build first", file=sys.stderr)
        return 1
    if verify_local(args) != 0:
        print("staging does not verify; refusing to upload", file=sys.stderr)
        return 1
    hub = _hub()
    api = hub.HfApi(token=_token(args.env_file))
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(staging),
        commit_message=args.message,
        ignore_patterns=[".*", "**/.*"],
    )
    print(f"uploaded {staging} -> https://huggingface.co/{args.repo_id}")
    return 0


def _read_manifest(path: Path) -> dict[str, dict]:
    with open(path, newline="") as fh:
        return {r["path"]: r for r in csv.DictReader(fh)}


def _compare_manifest(manifest: dict[str, dict]) -> list[str]:
    errs = []
    expected = {w.filename: w for w in WEIGHTS.values()}
    for path in sorted(set(expected) - set(manifest)):
        errs.append(f"MANIFEST lacks {path}")
    for path in sorted(set(manifest) - set(expected)):
        errs.append(f"MANIFEST has unknown {path}")
    for path in sorted(set(expected) & set(manifest)):
        if manifest[path]["sha256"] != expected[path].sha256:
            errs.append(f"MANIFEST sha256 differs for {path}")
    return errs


def verify_local(args) -> int:
    staging = Path(args.staging)
    man = staging / "MANIFEST.csv"
    if not man.exists():
        print(f"{man} missing; run --build first", file=sys.stderr)
        return 1
    manifest = _read_manifest(man)
    errs = _compare_manifest(manifest)
    for w in WEIGHTS.values():
        f = staging / w.filename
        if not f.exists():
            errs.append(f"staging lacks {w.filename}")
        elif str(f.stat().st_size) != manifest.get(w.filename, {}).get("size"):
            errs.append(f"size differs for {w.filename}")
        elif sha256sum(f) != w.sha256:
            errs.append(f"sha256 differs for {w.filename}")
    if not (staging / "README.md").exists():
        errs.append("staging lacks README.md")
    for e in errs:
        print(f"  FAIL {e}")
    print(f"verify (local {staging}): {len(WEIGHTS) - len(errs)}/{len(WEIGHTS)} files ok" if not errs else "FAILED")
    return int(bool(errs))


def verify_remote(args) -> int:
    hub = _hub()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    api = hub.HfApi(token=tok)
    man = hub.hf_hub_download(args.repo_id, "MANIFEST.csv", repo_type="model", token=tok)
    errs = _compare_manifest(_read_manifest(Path(man)))
    info = api.model_info(args.repo_id, files_metadata=True, token=tok)
    remote = {s.rfilename: s for s in info.siblings}
    for w in WEIGHTS.values():
        s = remote.get(w.filename)
        if s is None:
            errs.append(f"remote lacks {w.filename}")
            continue
        lfs = getattr(s, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha is None and isinstance(lfs, dict):
            sha = lfs.get("sha256")
        if sha is None:  # not stored with LFS: download and hash
            sha = sha256sum(hub.hf_hub_download(args.repo_id, w.filename, repo_type="model", token=tok))
        if sha != w.sha256:
            errs.append(f"remote sha256 differs for {w.filename}")
    for e in errs:
        print(f"  FAIL {e}")
    print(f"verify (remote {args.repo_id}): " + ("ok" if not errs else "FAILED"))
    return int(bool(errs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=WEIGHTS_REPO)
    ap.add_argument("--cache", default=str(cache_dir()), help="local weights cache ($WHOLEBODYPOSE_HOME)")
    ap.add_argument("--staging", default=str(DEFAULT_STAGING))
    ap.add_argument("--env-file", default=".env", help="file with HF_TOKEN= / HUGGINGFACE_TOKEN=")
    ap.add_argument("--message", default="Mirror WholeBodyPose model weights (byte-identical)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--hash", action="store_true", help="with --dry-run: also check sha256")
    ap.add_argument("--build", action="store_true", help="build staging tree + MANIFEST.csv + README.md")
    ap.add_argument("--fetch", action="store_true", help="with --build: download missing files first")
    ap.add_argument("--upload", action="store_true", help="create the private repo and upload staging")
    ap.add_argument("--yes", action="store_true", help="confirm --upload")
    ap.add_argument("--verify", action="store_true", help="verify staging (or the Hub with --remote)")
    ap.add_argument("--remote", action="store_true", help="with --verify: check the Hub repo")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.build or args.upload or args.verify):
        args.dry_run = True
    rc = 0
    if args.dry_run:
        rc |= dry_run(args)
    if args.build:
        rc |= build(args)
    if args.upload and args.dry_run:
        print(f"dry run: would create_repo({args.repo_id!r}, private=True) and upload_folder({args.staging!r})")
    elif args.upload and not rc:
        rc |= upload(args)
    if args.verify:
        rc |= verify_remote(args) if args.remote else verify_local(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
