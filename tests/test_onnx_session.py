from wholebodypose.models import onnx_session


class _FakeOrt:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def preload_dlls(self):
        self.calls += 1
        if self.fail:
            raise OSError("no CUDA libraries")


def test_cuda_libs_load_once(monkeypatch):
    monkeypatch.setattr(onnx_session, "_CUDA_LIBS_LOADED", False)
    ort = _FakeOrt()
    onnx_session._load_cuda_libs(ort)
    onnx_session._load_cuda_libs(ort)
    assert ort.calls == 1


def test_cuda_libs_failure_is_ignored(monkeypatch):
    monkeypatch.setattr(onnx_session, "_CUDA_LIBS_LOADED", False)
    onnx_session._load_cuda_libs(_FakeOrt(fail=True))  # falls back silently, like a system CUDA install
    monkeypatch.setattr(onnx_session, "_CUDA_LIBS_LOADED", False)
    onnx_session._load_cuda_libs(object())  # onnxruntime < 1.21 has no preload_dlls


def test_cpu_device_uses_cpu_provider():
    assert onnx_session.ort_providers("cpu") == ["CPUExecutionProvider"]


def _providers(monkeypatch, *names):
    monkeypatch.setattr(onnx_session, "_available_providers", lambda: [*names, "CPUExecutionProvider"])


def test_auto_prefers_cuda_then_coreml(monkeypatch):
    _providers(monkeypatch, "CUDAExecutionProvider", "CoreMLExecutionProvider")
    assert onnx_session.resolve_device("auto") == "cuda"
    _providers(monkeypatch, "CoreMLExecutionProvider")
    assert onnx_session.resolve_device("auto") == "coreml"
    _providers(monkeypatch)
    assert onnx_session.resolve_device("auto") == "cpu"


def test_missing_provider_falls_back_to_cpu(monkeypatch):
    _providers(monkeypatch)
    assert onnx_session.resolve_device("cuda") == "cpu"
    assert onnx_session.resolve_device("coreml") == "cpu"
    assert onnx_session.ort_providers("coreml") == ["CPUExecutionProvider"]


def test_coreml_providers_carry_model_format(monkeypatch):
    _providers(monkeypatch, "CoreMLExecutionProvider")
    assert onnx_session.ort_providers("coreml", "MLProgram") == [
        ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}),
        "CPUExecutionProvider",
    ]
    assert onnx_session.resolve_device("cpu") == "cpu"
