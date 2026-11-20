"""The weight registry is fully pinned, and the hf/ tooling builds and verifies a mirror offline."""

import csv
import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

from wholebodypose import weights

HF_DIR = Path(__file__).resolve().parents[1] / "hf"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HF_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_weight_has_a_sha256():
    for key, w in weights.WEIGHTS.items():
        assert w.sha256 and re.fullmatch(r"[0-9a-f]{64}", w.sha256), key


def test_filenames_and_hashes_are_unique():
    names = [w.filename for w in weights.WEIGHTS.values()]
    assert len(set(names)) == len(names)
    assert len({Path(n).name for n in names}) == len(names)
    hashes = [w.sha256 for w in weights.WEIGHTS.values()]
    assert len(set(hashes)) == len(hashes)


def test_every_weight_has_license_and_source():
    for key, w in weights.WEIGHTS.items():
        assert w.license and w.source, key


def test_build_and_verify_staging_offline(tmp_path, monkeypatch):
    up = _load("upload_weights")
    cache, staging = tmp_path / "cache", tmp_path / "staging"
    fake = {}
    for k, w in list(weights.WEIGHTS.items())[:3]:
        data = f"fake {k}".encode()
        (cache / w.filename).parent.mkdir(parents=True, exist_ok=True)
        (cache / w.filename).write_bytes(data)
        fake[k] = weights.WeightFile(k, w.filename, w.upstream, hashlib.sha256(data).hexdigest(), w.license, w.source)
    monkeypatch.setattr(up, "WEIGHTS", fake)
    argv = ["--cache", str(cache), "--staging", str(staging)]

    assert up.main(["--dry-run", "--hash", *argv]) == 0
    assert up.main(["--build", *argv]) == 0
    with open(staging / "MANIFEST.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["key"] for r in rows] == list(fake)
    assert set(rows[0]) == set(up.MANIFEST_FIELDS)
    readme = (staging / "README.md").read_text()
    assert up.FILES_MARKER not in readme and rows[0]["path"] in readme
    assert up.main(["--verify", *argv]) == 0
    assert up.main(["--upload", *argv]) == 2  # refuses without --yes, before any network access

    (staging / rows[0]["path"]).write_bytes(b"tampered")
    assert up.main(["--verify", *argv]) == 1


def test_build_rejects_wrong_hash(tmp_path, monkeypatch):
    up = _load("upload_weights")
    w = weights.WEIGHTS["mp_hand"]
    (tmp_path / w.filename).parent.mkdir(parents=True)
    (tmp_path / w.filename).write_bytes(b"not the real model")
    monkeypatch.setattr(up, "WEIGHTS", {"mp_hand": w})
    assert up.main(["--build", "--cache", str(tmp_path), "--staging", str(tmp_path / "s")]) == 1


@pytest.mark.parametrize(
    "text, ok",
    [
        ("---\nlicense: mit\npretty_name: x\n---\n# body\n", True),
        ("# no front matter\n", False),
        ("---\nlicense: mit\npretty_name: x\n---\n# body\n---\nlicense: mit\n---\n", False),
    ],
)
def test_dataset_card_checks(text, ok):
    assert (not _load("upload_dataset_card").check_card(text)) is ok


def test_shipped_dataset_card_is_valid():
    card = HF_DIR / "dataset_cards" / "pose-action-recognition.md"
    assert _load("upload_dataset_card").check_card(card.read_text()) == []
