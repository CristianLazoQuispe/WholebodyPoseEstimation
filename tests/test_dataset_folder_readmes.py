"""The per-folder dataset READMEs are generated from the dataset card and carry the citation obligation."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

HF_DIR = Path(__file__).resolve().parents[1] / "hf"
CARD = HF_DIR / "dataset_cards" / "pose-action-recognition.md"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HF_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("make_dataset_folder_readmes")


@pytest.fixture(scope="module")
def card_text():
    return CARD.read_text(encoding="utf-8")


def test_card_lists_13_subsets_of_five_sources(gen, card_text):
    subsets = gen.read_subsets(card_text)
    assert len(subsets) == 14  # 14 subsets x Train/Val/Test = 42 split files
    assert {s.source.name for s in subsets} == {"WLASL", "MS-ASL", "ASL Citizen", "LSA64", "PUCP-DGI"}


def test_every_folder_gets_a_readme_with_citations(gen, card_text):
    files = gen.build(card_text)
    for s in gen.read_subsets(card_text):
        text = files[f"{s.folder}/README.md"]
        assert "lazo2026wholebodypose" in text and "10.1117/12.3096312" in text
        key = re.match(r"@\w+\{(\w+),", s.source.bibtex).group(1)
        assert key in text  # source dataset BibTeX
        assert "Instructions for AI agents" in text and "CC BY-NC-SA 4.0" in text
        assert f"{s.name}_135-Train.hdf5" in text and s.maplabels in text
        assert f"{s.train:,}" in text and f"{s.test:,}" in text
    for parent in ("ISLR", "ISLR/WLASL", "ISLR/MSASL", "ISLR/ASLCITIZEN"):
        assert f"{parent}/README.md" in files
    assert "CITATION.cff" in files


def test_label_map_names_match_the_hub(gen, card_text):
    names = {s.name: s.maplabels for s in gen.read_subsets(card_text)}
    assert names["WLASL100"] == "wlasl_100_maplabels.json"
    assert names["ASLCitizen2731"] == "asl_citizen_2731_maplabels.json"
    assert names["LSA64"] == "lsa64_64_maplabels.json"
    assert names["DGI_PUCP30"] == "dgi_pucp_30_maplabels.json"


def test_committed_files_are_up_to_date(gen):
    assert gen.main(["--check"]) == 0


def test_bibtex_braces_are_balanced(gen):
    for src in gen.SOURCES.values():
        assert src.bibtex.count("{") == src.bibtex.count("}"), src.name


def test_upload_folder_readmes_is_a_dry_run_by_default(capsys):
    up = _load("upload_dataset_card")
    assert up.main(["--folder-readmes", "--env-file", "/nonexistent"]) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing uploaded" in out and "ISLR/LSA64/README.md" in out
