"""Push an updated dataset card (``README.md`` only) to a Hugging Face dataset repo.

Cards live in ``hf/dataset_cards/<repo-name>.md``. Only the card is uploaded; the data files of the
dataset are never touched.

``--dry-run`` (default)
    Check the card (YAML front matter present, a single ``license:`` key, no duplicated front matter),
    print where it would go and, if the current remote card is reachable (``--diff``), a unified diff.
``--cards-only``
    Upload the card as ``README.md`` of the dataset repo in one commit. Needs ``HF_TOKEN`` (or
    ``HUGGINGFACE_TOKEN`` / ``--env-file``) and ``--yes``.

    python hf/upload_dataset_card.py --dry-run
    python hf/upload_dataset_card.py --dry-run --diff
    HF_TOKEN=... python hf/upload_dataset_card.py --cards-only --yes

``--folder-readmes``
    Also handle the per-folder ``README.md`` files and the root ``CITATION.cff`` written by
    ``hf/make_dataset_folder_readmes.py`` into ``hf/dataset_cards/folders/`` (same paths as on the Hub). They are
    regenerated in memory from the card and compared with the committed files first; stale files are a
    problem. Without ``--yes`` this is a dry run that lists what would be uploaded; with ``--yes`` the card,
    ``LICENSE.md`` and these files go up in one commit.

    python hf/upload_dataset_card.py --folder-readmes               # dry run
    HF_TOKEN=... python hf/upload_dataset_card.py --folder-readmes --yes

The token is never printed.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "CristianLazoQuispe/pose-action-recognition"


def card_path(repo_id: str) -> Path:
    return HERE / "dataset_cards" / f"{repo_id.split('/')[-1]}.md"


def check_card(text: str) -> list[str]:
    """Return problems found in a dataset card."""
    problems = []
    if not text.startswith("---\n"):
        problems.append("card does not start with YAML front matter")
        return problems
    end = text.find("\n---\n", 4)
    if end < 0:
        problems.append("YAML front matter is not closed")
        return problems
    meta = text[4:end]
    if len(re.findall(r"^license:", meta, flags=re.M)) != 1:
        problems.append("front matter must have exactly one 'license:' key")
    if re.search(r"^---\s*\nlicense:", text[end + 5 :], flags=re.M):
        problems.append("a second front-matter block with 'license:' appears in the body")
    if "pretty_name:" not in meta:
        problems.append("front matter has no pretty_name")
    return problems


def folder_files(card_text: str) -> tuple[dict[str, Path], list[str]]:
    """{path in repo: local file} for the per-folder READMEs + CITATION.cff, and problems found."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_dataset_folder_readmes", HERE / "make_dataset_folder_readmes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up while the class is created
    spec.loader.exec_module(mod)
    expected = mod.build(card_text)
    files, problems = {}, []
    for rel, text in expected.items():
        local = mod.OUT / rel
        if not local.exists() or local.read_text(encoding="utf-8") != text:
            problems.append(f"{local} is missing or stale: run python hf/make_dataset_folder_readmes.py")
        files[rel] = local
    return files, problems


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


def remote_card(repo_id: str) -> str:
    hub = _hub()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    path = hub.hf_hub_download(repo_id, "README.md", repo_type="dataset", token=tok)
    return Path(path).read_text()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--card", default=None, help="card file (default: hf/dataset_cards/<repo-name>.md)")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--message", default="Update dataset card")
    ap.add_argument("--dry-run", action="store_true", help="check the card and print the plan (default)")
    ap.add_argument("--diff", action="store_true", help="with --dry-run: diff against the remote README.md")
    ap.add_argument("--cards-only", action="store_true", help="upload the card as README.md")
    ap.add_argument("--yes", action="store_true", help="confirm --cards-only / --folder-readmes")
    ap.add_argument(
        "--folder-readmes",
        action="store_true",
        help="also the per-folder README.md files and CITATION.cff (dry run unless --yes)",
    )
    args = ap.parse_args(argv)

    card = Path(args.card) if args.card else card_path(args.repo_id)
    text = card.read_text()
    problems = check_card(text)
    for p in problems:
        print(f"PROBLEM {p}")
    print(f"card    : {card} ({len(text.splitlines())} lines)")
    print(f"target  : https://huggingface.co/datasets/{args.repo_id} -> README.md")

    extra: dict[str, Path] = {}
    if args.folder_readmes:
        extra, extra_problems = folder_files(text)
        for p in extra_problems:
            print(f"PROBLEM {p}")
        problems += extra_problems
        for rel in extra:
            print(f"folder  : {rel}")
        print(f"{len(extra)} folder files")

    if args.dry_run or not (args.cards_only or (args.folder_readmes and args.yes)):
        if args.diff:
            old = remote_card(args.repo_id)
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), text.splitlines(True), "remote", "local"))
        print("dry run: nothing uploaded")
        return int(bool(problems))

    if problems:
        print("refusing to upload a card with problems", file=sys.stderr)
        return 1
    if not args.yes:
        print("uploading writes to the Hugging Face Hub; re-run with --yes to confirm", file=sys.stderr)
        return 2
    hub = _hub()
    api = hub.HfApi(token=_token(args.env_file))
    ops = [hub.CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card))]
    license_file = card.parent / "LICENSE.md"
    if "/LICENSE.md" in text and license_file.exists():
        ops.append(hub.CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj=str(license_file)))
    ops += [hub.CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path)) for rel, path in extra.items()]
    api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops, commit_message=args.message)
    print(f"uploaded {', '.join(o.path_in_repo for o in ops)} -> {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
