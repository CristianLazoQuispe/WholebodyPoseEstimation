"""ISLR datasets: HDF5 schema, dataset adapters and the builder (synthetic data, fake model)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import FakePoseModel
from wholebodypose.datasets import build as B
from wholebodypose.datasets import hdf5 as H
from wholebodypose.datasets import sources as S

h5py = pytest.importorskip("h5py")


# --------------------------------------------------------------------------- helpers
def _video(path: Path, n: int = 6, size=(160, 120)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    assert w.isOpened()
    for t in range(n):
        f = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.rectangle(f, (10 + 3 * t, 30), (40 + 3 * t, 90), (0, 255, 0), -1)
        w.write(f)
    w.release()
    return path


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


@pytest.fixture
def wlasl_root(tmp_path) -> Path:
    root = tmp_path / "WLASL"
    info = [
        {"gloss": "book", "instances": [{"video_id": "001", "split": "train"}, {"video_id": "002", "split": "test"}]},
        {"gloss": "drink", "instances": [{"video_id": "003", "split": "val"}, {"video_id": "004", "split": "train"}]},
        {"gloss": "zzz", "instances": [{"video_id": "005", "split": "train"}]},
    ]
    root.mkdir()
    (root / "info.json").write_text(json.dumps(info))
    for i, vid in enumerate(["001", "002", "003", "004", "005"]):
        _video(root / "videos" / f"{vid}.mp4", n=4 + i)
    return root


# --------------------------------------------------------------------------- hdf5
def test_to_published_array_normalizes_and_keeps_zeros():
    k = np.zeros((3, 133, 2), np.float32)
    k[:, 5] = [100, 50]
    k[:, 6] = [300, 150]
    k[0, 11] = [200, 400]  # hip 12 missing in frame 0
    d = H.to_published_array(k, 400, 200)
    assert d.shape == (3, 2, 135) and d.dtype == np.float64
    np.testing.assert_allclose(d[0, :, 5], [0.25, 0.25])
    np.testing.assert_allclose(d[0, :, H.MID_CHEST], [0.5, 0.5])
    np.testing.assert_allclose(d[0, :, H.MID_HIP], [0.25, 1.0])  # biased mean, as published
    assert (d[:, :, 0] == 0).all()
    safe = H.to_published_array(k, 400, 200, safe_midpoints=True)
    assert (safe[0, :, H.MID_HIP] == 0).all()
    np.testing.assert_allclose(safe[0, :, H.MID_CHEST], [0.5, 0.5])


def test_add_midpoints_rejects_bad_shape():
    with pytest.raises(ValueError):
        H.add_midpoints(np.zeros((2, 135, 2)))


def test_write_read_roundtrip_matches_published_schema(tmp_path):
    path = tmp_path / "X_135-Test.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5:
        for i in range(12):  # > 10 so that alphabetical != numeric order
            data = H.to_published_array(rng.uniform(1, 100, (5 + i, 133, 2)), 640, 480)
            H.write_sample(h5, i, data, f"gloss{i}", f"/videos/v{i}.mp4", 640, 480)
    with h5py.File(path, "r") as h5:
        g = h5["3"]
        assert set(g) == {"data", "width", "height", "label", "video_name"}
        assert g["data"].dtype == np.float64 and g["data"].shape == (8, 2, 135)
        assert g["data"].compression is None and g["data"].chunks is None
        assert g["width"].dtype == np.int64 and g["width"].shape == ()
        assert h5py.check_string_dtype(g["label"].dtype).encoding == "utf-8"
        assert isinstance(g["label"][()], bytes)
    samples = H.read_hdf5(path)
    assert [s["index"] for s in samples] == list(range(12))
    assert samples[11]["label"] == "gloss11" and samples[11]["video_name"] == "/videos/v11.mp4"
    assert samples[11]["data"].shape == (16, 2, 135) and samples[11]["width"] == 640
    meta = H.read_hdf5(path, with_data=False)
    assert meta[2]["T"] == 7 and "data" not in meta[2]


def test_scores_and_compression_options(tmp_path):
    path = tmp_path / "s.hdf5"
    k = np.ones((4, 133, 2))
    s = np.full((4, 133), 0.5, np.float32)
    with h5py.File(path, "w") as h5:
        H.write_sample(
            h5, 0, H.to_published_array(k, 10, 10), "a", "v", 10, 10, scores=H.scores_135(s), compression="gzip"
        )
        with pytest.raises(ValueError):
            H.write_sample(h5, 1, H.to_published_array(k, 10, 10), "a", "v", 10, 10, scores=np.zeros((3, 135)))
    with h5py.File(path, "r") as h5:
        assert h5["0/scores"].shape == (4, 135) and h5["0/scores"].dtype == np.float32
        assert h5["0/data"].compression == "gzip"


def test_compare_arrays():
    a = np.zeros((4, 2, 135))
    a[:, :, :10] = 0.5
    b = a.copy()
    b[:, :, 0] += 0.01
    b[0, :, 1] = 0  # one detection disagreement
    m = H.compare_arrays(a, b)
    assert m["T_match"] and m["n_both"] == 4 * 10 - 1
    assert m["detection_agreement"] == pytest.approx(1 - 1 / (4 * 133))
    assert m["max_abs_diff"] == pytest.approx(0.01)
    assert m["median_abs_diff"] == 0.0
    assert not H.compare_arrays(a, b[:3])["T_match"]


def test_split_filename():
    assert H.split_filename("WLASL100", "test") == "WLASL100_135-Test.hdf5"


# --------------------------------------------------------------------------- sources
def test_wlasl_adapter(wlasl_root):
    samples, labels = S.load_samples("wlasl", 100, wlasl_root)
    assert labels == {"book": 0, "drink": 1, "zzz": 2}
    assert [s.video_name for s in samples][:2] == ["rgb/WLASL100/train/001.mp4", "rgb/WLASL100/test/002.mp4"]
    assert samples[0].video_path == str(wlasl_root / "videos" / "001.mp4")
    assert {k: len(v) for k, v in S.by_split(samples).items()} == {"train": 3, "val": 1, "test": 1}
    with pytest.raises(ValueError):
        S.load_samples("wlasl", 123, wlasl_root)


def test_lsa64_adapter(tmp_path):
    rows = [[0, "train", "Opaque", 0, r"LSA64_dataset\LSA64\001_001_001.mp4"],
            [1, "train", "None", 5, r"LSA64_dataset\LSA64\006_009_001.mp4"],
            [2, "train", "Opaque", 0, r"LSA64_dataset\LSA64\001_010_002.mp4"]]  # fmt: skip
    _csv(tmp_path / "lsa64_full_metadata.csv", ["id", "subset", "gloss", "label", "relative_path"], rows)
    samples, labels = S.load_samples("lsa64", 64, tmp_path)
    assert [s.split for s in samples] == ["train", "val", "test"]
    assert samples[1].label == "None_" and labels == {"Opaque": 0, "None_": 5}
    assert samples[2].video_name == "/data/LSA64_dataset/LSA64/001_010_002.mp4"


def test_pucp_dgi_adapter(tmp_path):
    rows = [[0, "train", "amigo", 1, "DGI305AEC_SignERT/split/train/amigo_1.mp4"],
            [1, "train", "ahora", 0, "DGI305AEC_SignERT/split/train/ahora_1.mp4"],
            [2, "test", "bien", 3, "DGI305AEC_SignERT/split/test/bien_1.mp4"],  # not in the 30
            [3, "val", "ahora", 0, "DGI305AEC_SignERT/split/val/ahora_2.mp4"]]  # fmt: skip
    _csv(tmp_path / "DGI305AEC_SignERT_metadata.csv", ["id", "subset", "gloss", "label", "relative_path"], rows)
    samples, labels = S.load_samples("pucp-dgi", 30, tmp_path)
    assert [s.video_name for s in samples] == [
        "/data/DGI305AEC_SignERT/split/train/ahora_1_1.mp4",
        "/data/DGI305AEC_SignERT/split/val/ahora_2_3.mp4",
        "/data/DGI305AEC_SignERT/split/train/amigo_1_0.mp4",
    ]
    assert labels == {"ahora": 0, "amigo": 1}


def test_asl_citizen_adapter_and_noncontiguous_labels(tmp_path):
    hdr = ["Participant ID", "Video file", "Gloss", "ASL-LEX Code"]
    # glosses: A x3, B x2, C x1 -> top-1 = {A}; the label map is global across subsets
    _csv(tmp_path / "splits" / "train.csv", hdr, [["P1", "1-C.mp4", "C", ""], ["P1", "2-A.mp4", "A", ""]])
    _csv(tmp_path / "splits" / "val.csv", hdr, [["P2", "3-B.mp4", "B", ""], ["P2", "4-A.mp4", "A", ""]])
    _csv(tmp_path / "splits" / "test.csv", hdr, [["P3", "5-B.mp4", "B", ""], ["P3", "6-A.mp4", "A", ""]])
    samples, labels = S.asl_citizen(2731, tmp_path)
    assert [s.video_name.split("/")[-1] for s in samples] == ["2-A.mp4", "4-A.mp4", "6-A.mp4", "3-B.mp4", "5-B.mp4",
                                                              "1-C.mp4"]  # fmt: skip
    assert samples[0].video_path == str(tmp_path / "videos" / "2-A.mp4")
    assert labels == {"A": 0, "B": 1, "C": 2}


def test_msasl_adapter(tmp_path):
    def item(vid, s, e, text, clean, label):
        return {"url": f"https://www.youtube.com/watch?v={vid}", "start": s, "end": e, "text": text,
                "clean_text": clean, "label": label}  # fmt: skip

    train = [item("AAAAAAAAAAA", 0, 10, "hello", "hello", 0), item("BBBBBBBBBBB", 5, 9, "not like", "dislike", 1),
             item("AAAAAAAAAAA", 0, 10, "hello", "hello", 0), item("xfVNJxKWqeg", 4007, 4007, "right", "right", 0),
             item("CCCCCCCCCCC", 0, 5, "big", "big", 150)]  # fmt: skip
    val = [item("DDDDDDDDDDD", 1, 2, "hello", "hello", 0)]
    for name, data in (("train", train), ("val", val), ("test", [])):
        (tmp_path / f"MSASL_{name}.json").write_text(json.dumps(data))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "BBBBBBBBBBB&t=3s_000005_000009-msasl-not_like.mp4").touch()
    samples, labels = S.load_samples("msasl", 100, tmp_path)
    names = [s.video_name for s in samples]
    assert names == [
        "/data/MSASL/train/BBBBBBBBBBB&t=3s_000005_000009-msasl-not_like.mp4",
        "/data/MSASL/train/AAAAAAAAAAA_000000_000010-msasl-hello.mp4",
        "/data/MSASL/val/DDDDDDDDDDD_000001_000002-msasl-hello.mp4",
    ]
    assert [s.label for s in samples] == ["dislike", "hello", "hello"]
    assert labels == {"hello": 0, "dislike": 1}


def test_maplabels_json_and_published_names():
    m = S.maplabels_json({"b": 1, "a": 0})
    assert m == {"id_to_label": {"a": 0, "b": 1}, "label_to_id": {"0": "a", "1": "b"}}
    assert S.published_prefix("pucp_dgi", 30) == "DGI_PUCP30"
    assert S.published_dir("ISLR", "wlasl", 100) == Path("ISLR/WLASL/WLASL100")
    assert S.published_dir("ISLR", "lsa64", 64) == Path("ISLR/LSA64")
    assert S.maplabels_filename("asl_citizen", 300) == "asl_citizen_300_maplabels.json"
    with pytest.raises(ValueError):
        S.canonical("nope")


# --------------------------------------------------------------------------- builder
def test_select_person():
    k = np.zeros((2, 133, 2), np.float32)
    k[0, :2] = [[0, 0], [1, 1]]
    k[1, :2] = [[0, 0], [10, 10]]
    s = np.ones((2, 133), np.float32)
    assert B.select_person(k, s, "first")[0][1, 0] == 1
    assert B.select_person(k, s, "largest")[0][1, 0] == 10
    z, zs = B.select_person(np.zeros((0, 133, 2)), np.zeros((0, 133)))
    assert z.shape == (133, 2) and not z.any() and not zs.any()
    with pytest.raises(ValueError):
        B.select_person(k, s, "middle")


def test_extract_video_maps_back_to_original_resolution(synthetic_video):
    model = FakePoseModel()
    full = B.extract_video(model, synthetic_video)
    assert full["keypoints"].shape == (20, 133, 2) and (full["width"], full["height"]) == (160, 120)
    small = B.extract_video(model, synthetic_video, resize_width=80, max_frames=3)
    assert small["keypoints"].shape == (3, 133, 2)
    # the fake pose scales with the image, so mapping back recovers the full-resolution pose
    np.testing.assert_allclose(small["keypoints"][0], full["keypoints"][0], rtol=1e-4)


def test_extract_video_missing_file(tmp_path):
    with pytest.raises(OSError):
        B.extract_video(FakePoseModel(), tmp_path / "nope.mp4")


def test_build_dataset_resumable_and_logs_errors(wlasl_root, tmp_path):
    (wlasl_root / "videos" / "005.mp4").unlink()  # one broken sample
    model = FakePoseModel()
    out = tmp_path / "out"
    rep = B.build_dataset("wlasl", 100, out, root=wlasl_root, model=model, verbose=False)
    assert rep["splits"]["train"]["samples"] == 2 and rep["splits"]["train"]["missing"] == [
        "rgb/WLASL100/train/005.mp4"
    ]
    cache = next((out / ".cache").iterdir())
    assert len((cache / "errors.jsonl").read_text().splitlines()) == 1
    assert len((cache / "done.txt").read_text().splitlines()) == 4
    calls = model.calls
    assert calls == 4 + 5 + 6 + 7  # every frame of the 4 readable videos

    train = H.read_hdf5(out / "WLASL100_135-Train.hdf5")
    assert [s["video_name"] for s in train] == ["rgb/WLASL100/train/001.mp4", "rgb/WLASL100/train/004.mp4"]
    assert train[0]["data"].shape == (4, 2, 135) and train[0]["width"] == 160 and train[0]["label"] == "book"
    assert 0 < train[0]["data"][:, :, :133].max() <= 1
    maps = json.loads((out / "wlasl_100_maplabels.json").read_text())
    assert maps["id_to_label"] == {"book": 0, "drink": 1, "zzz": 2}

    B.build_dataset("wlasl", 100, out, root=wlasl_root, model=model, verbose=False)  # resume: nothing to do
    assert model.calls == calls

    rows = B.verify_against(out, out, "WLASL100")
    assert rows and all(r["T_match"] and r["max_abs_diff"] == 0 for r in rows)


def test_build_options_scores_order_and_limit(wlasl_root, tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    with h5py.File(ref / "WLASL100_135-Train.hdf5", "w") as h5:  # published order: 004, 001
        for i, vid in enumerate(["004", "001"]):
            H.write_sample(h5, i, np.zeros((1, 2, 135)), "x", f"rgb/WLASL100/train/{vid}.mp4", 1, 1)
    cfg = B.BuildConfig(save_scores=True, safe_midpoints=True)
    out = tmp_path / "out"
    B.build_dataset("wlasl", 100, out, root=wlasl_root, config=cfg, model=FakePoseModel(), splits=("train",),
                    order_like=ref, verbose=False)  # fmt: skip
    train = H.read_hdf5(out / "WLASL100_135-Train.hdf5")
    assert [s["video_name"][-7:] for s in train] == ["004.mp4", "001.mp4", "005.mp4"]
    assert train[0]["scores"].shape == (train[0]["data"].shape[0], 135)
    B.build_dataset("wlasl", 100, tmp_path / "lim", root=wlasl_root, model=FakePoseModel(), splits=("train",),
                    limit=1, verbose=False)  # fmt: skip
    assert len(H.read_hdf5(tmp_path / "lim" / "WLASL100_135-Train.hdf5")) == 1


def test_presets_and_cache_tag():
    ic, v1 = B.PRESETS["icmv2025"], B.PRESETS["v1"]
    assert ic.model_kwargs["legacy_rgb_input"] and ic.person == "first" and ic.resize_width is None
    assert ic.model_kwargs["kpt_thr"] == 2.5 and ic.model_kwargs["filter_noise"]
    assert not v1.model_kwargs["legacy_rgb_input"] and v1.person == "largest"
    assert ic.cache_tag() != v1.cache_tag()
    assert B.BuildConfig(save_scores=True).cache_tag() == B.BuildConfig().cache_tag()


def test_cli_list_and_build(wlasl_root, tmp_path, monkeypatch, capsys):
    assert B.main(["--dataset", "wlasl", "--subset", "100", "--root", str(wlasl_root), "--list"]) == 0
    out = capsys.readouterr().out
    assert "WLASL100 train: 3 samples, 3 classes" in out and "label map: 3 glosses" in out
    assert B.main(["--dataset", "wlasl", "--root", str(wlasl_root)]) == 2  # --out missing

    seen = {}

    def fake_make_model(config, device, verbose=False):
        seen.update(config=config, device=device)
        return FakePoseModel()

    monkeypatch.setattr(B, "_make_model", fake_make_model)
    argv = ["--dataset", "wlasl", "--subset", "100", "--root", str(wlasl_root), "--out", str(tmp_path / "o"),
            "--splits", "test", "--device", "cpu", "--no-legacy-rgb-input", "--kpt-thr", "3", "--person", "largest",
            "--resize-width", "0", "-q", "--verify", str(tmp_path / "o")]  # fmt: skip
    assert B.main(argv) == 0
    cfg = seen["config"]
    assert seen["device"] == "cpu" and cfg.person == "largest" and cfg.resize_width is None
    assert cfg.model_kwargs["kpt_thr"] == 3 and cfg.model_kwargs["legacy_rgb_input"] is False
    assert (tmp_path / "o" / "WLASL100_135-Test.hdf5").exists()
    assert "verify: 1 samples, T match 100.0%" in capsys.readouterr().out


def test_rtmpose_legacy_rgb_flag(monkeypatch):
    """The legacy flag passes the RGB image to the models unchanged; the default converts to BGR."""
    from wholebodypose.models.rtmpose import model as M

    seen = []
    obj = M.RTMPoseModel.__new__(M.RTMPoseModel)
    obj.filter_noise, obj.use_thresholding, obj.kpt_thr = False, False, 2.5
    obj.model = lambda img: (seen.append(img.copy()) or np.zeros((1, 133, 2)), np.zeros((1, 133)))
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0] = 255  # red in RGB
    obj.legacy_rgb_input = False
    obj.predict(img)
    obj.legacy_rgb_input = True
    obj.predict(img)
    assert seen[0][0, 0].tolist() == [0, 0, 255] and seen[1][0, 0].tolist() == [255, 0, 0]


# --------------------------------------------------------------------------- published data (slow)
PUBLISHED = Path("/data/cristian/paper_2025/pose-action-recognition/ISLR/WLASL/WLASL100/WLASL100_135-Test.hdf5")
WLASL_VIDEOS = Path("/data/cristian/paper_2025/WLASL_videos/WLASL/videos")


@pytest.mark.slow
@pytest.mark.skipif(not (PUBLISHED.exists() and WLASL_VIDEOS.exists()), reason="published ISLR data not available")
def test_icmv2025_preset_reproduces_published_sample():
    from wholebodypose.api import create_model

    ref = next(H.iter_samples(PUBLISHED))
    video = WLASL_VIDEOS / ref["video_name"].split("/")[-1]
    cfg = B.PRESETS["icmv2025"]
    model = create_model(cfg.backend, device="cpu", verbose=False, **cfg.model_kwargs)
    res = B.extract_video(model, video, cfg.resize_width, cfg.person, max_frames=6)
    ours = H.to_published_array(res["keypoints"], res["width"], res["height"])
    m = H.compare_arrays(ref["data"][:6], ours)
    assert (res["width"], res["height"]) == (ref["width"], ref["height"])
    assert m["detection_agreement"] > 0.99
    assert m["median_abs_diff"] < 1e-4
