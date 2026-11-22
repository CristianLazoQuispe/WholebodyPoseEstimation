import pytest

from tests.conftest import FakePoseModel
from wholebodypose import api, cli
from wholebodypose.io import load_poses


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(api, "create_model", lambda *a, **k: FakePoseModel())


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    out = capsys.readouterr().out
    assert "wholebodypose" in out
    assert "10.1117/12.3096312" in out


def test_help_asks_to_cite(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "Please cite" in out and "10.1117/12.3096312" in out


def test_backends(capsys):
    assert cli.main(["backends"]) == 0
    out = capsys.readouterr().out
    assert "rtmpose" in out and "wholebodypose[mediapipe]" in out and "onnxruntime" in out


def test_video_command(fake_backend, synthetic_video, tmp_path, capsys):
    assert cli.main(["video", str(synthetic_video), "-o", str(tmp_path), "-q", "--max-frames", "4"]) == 0
    k, s, meta = load_poses(tmp_path / "synthetic_rtmpose.npz")
    assert k.shape == (4, 133, 2) and meta["backend"] == "rtmpose"
    assert list(tmp_path.glob("synthetic_*.mp4"))  # named after the backend that ran


def test_image_command(fake_backend, sample_image_path, tmp_path):
    assert cli.main(["image", str(sample_image_path), "-o", str(tmp_path), "-q", "-b", "mediapipe"]) == 0
    k, s, _ = load_poses(tmp_path / "image_mediapipe.json")
    assert k.shape == (1, 1, 133, 2)
    assert (tmp_path / "image_mediapipe.png").exists()


def test_video_tracking_flag(monkeypatch, synthetic_video, tmp_path):
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return FakePoseModel()

    monkeypatch.setattr(api, "create_model", fake)
    args = ["video", str(synthetic_video), "-o", str(tmp_path), "-q", "--max-frames", "2", "--no-render"]
    assert cli.main(args + ["-b", "mediapipe", "--tracking"]) == 0
    assert seen["image_mode"] is False
    assert cli.main(args + ["-b", "rtmpose", "--tracking"]) == 2
