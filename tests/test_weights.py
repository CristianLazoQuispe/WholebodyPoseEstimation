import hashlib
import zipfile

import pytest

from wholebodypose import weights


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WHOLEBODYPOSE_HOME", str(tmp_path))
    return tmp_path


def test_registry_is_consistent():
    for key, w in weights.WEIGHTS.items():
        assert w.key == key and "/" in w.filename and w.upstream.startswith("https://")
    assert weights.mirror_url(weights.WEIGHTS["mp_hand"]).endswith("/resolve/main/mediapipe/hand_landmarker.task")


def test_cached_file_is_used(home, monkeypatch):
    w = weights.WEIGHTS["mp_hand"]
    (home / w.filename).parent.mkdir(parents=True)
    (home / w.filename).write_bytes(b"x")
    monkeypatch.setattr(weights, "_download", lambda *a, **k: pytest.fail("should not download"))
    assert weights.get_weight("mp_hand") == home / w.filename


def test_falls_back_to_upstream_and_extracts_zip(home, monkeypatch):
    calls = []

    def fake_download(url, dst, verbose):
        calls.append(url)
        if "huggingface" in url:
            raise OSError("mirror down")
        with zipfile.ZipFile(dst, "w") as zf:
            zf.writestr("pkg/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx", b"onnx")

    monkeypatch.setattr(weights, "_download", fake_download)
    path = weights.get_weight("rtmw_l_256", verbose=False)
    assert path.read_bytes() == b"onnx" and len(calls) == 2


def test_sha_mismatch_is_rejected(home, monkeypatch):
    w = weights.WEIGHTS["mp_face"]
    bad = weights.WeightFile(w.key, w.filename, w.upstream, sha256=hashlib.sha256(b"good").hexdigest())
    monkeypatch.setitem(weights.WEIGHTS, "mp_face", bad)

    def evil(url, dst, verbose):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"evil")

    monkeypatch.setattr(weights, "_download", evil)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        weights.get_weight("mp_face", verbose=False)
    assert not (home / w.filename).exists()


def test_offline_and_unknown(home):
    with pytest.raises(FileNotFoundError):
        weights.get_weight("mp_hand", offline=True)
    with pytest.raises(KeyError):
        weights.get_weight("nope")
