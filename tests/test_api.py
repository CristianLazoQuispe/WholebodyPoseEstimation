import numpy as np
import pytest

import wholebodypose
from wholebodypose import api


def test_version_and_lazy_exports():
    assert wholebodypose.__version__
    assert set(wholebodypose.__all__) >= {"PoseEstimator", "Video2Pose", "save_poses", "load_poses"}
    assert wholebodypose.PoseEstimator is api.PoseEstimator
    with pytest.raises(AttributeError):
        _ = wholebodypose.does_not_exist


def test_registry_lists_paper_backends():
    assert api.available_backends() == ["rtmpose", "vitpose", "mediapipe"]
    for b in api.BACKENDS.values():
        assert ":" in b.target
    # RTMPose and ViTPose (ONNX) run on the core install; MediaPipe is an extra
    assert api.BACKENDS["rtmpose"].extra == "" and api.BACKENDS["vitpose"].extra == ""
    assert api.BACKENDS["mediapipe"].extra == "mediapipe"
    assert api.install_command("") == "pip install wholebodypose"


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        api.create_model("openpose")


def test_create_model_merges_defaults(monkeypatch):
    seen = {}

    class Dummy:
        def __init__(self, **kw):
            seen.update(kw)

    import types

    mod = types.ModuleType("dummy_backend")
    mod.Dummy = Dummy
    monkeypatch.setitem(__import__("sys").modules, "dummy_backend", mod)
    monkeypatch.setitem(api.BACKENDS, "dummy", api.Backend("dummy", "dummy_backend:Dummy", "x", "d", {"a": 1, "b": 2}))
    api.create_model("dummy", device="cpu", b=3)
    assert seen == {"a": 1, "b": 3, "device": "cpu", "verbose": True}


def test_missing_extra_message(monkeypatch):
    monkeypatch.setitem(api.BACKENDS, "ghost", api.Backend("ghost", "not_installed_mod:X", "ghost", "d"))
    with pytest.raises(ImportError, match=r"wholebodypose\[ghost\]"):
        api.create_model("ghost")


def test_pose_estimator_with_fake(monkeypatch, fake_model, image_rgb):
    monkeypatch.setattr(api, "create_model", lambda *a, **k: fake_model)
    est = api.PoseEstimator("rtmpose")
    k, s = est(image_rgb)
    assert k.shape == (1, 133, 2) and s.shape == (1, 133)
    assert k.dtype == np.float32
    drawn = est.draw(image_rgb, k, s)
    assert drawn.shape == image_rgb.shape and not np.array_equal(drawn, image_rgb)
    empty = est.draw(image_rgb, np.zeros((0, 133, 2)), np.zeros((0, 133)))
    assert np.array_equal(empty, image_rgb)


def test_thresholding(fake_model):
    fake_model.use_thresholding = True
    fake_model.kpt_thr = 0.5
    k = np.ones((1, 133, 2), np.float32)
    s = np.linspace(0, 1, 133, dtype=np.float32)[None]
    k2, s2 = fake_model.thresholding(k, s)
    assert set(np.unique(s2)) <= {0.0, 1.0}
    assert np.all(k2[s < 0.5] == 0) and np.all(k2[s >= 0.5] == 1)
    assert np.all(k == 1), "thresholding must not mutate its input"
