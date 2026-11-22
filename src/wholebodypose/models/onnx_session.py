"""ONNX Runtime session helper shared by every ONNX model (YOLOX, RTMW, ViTPose).

One place decides the execution providers, the CPU thread count and the reported device, so the
RTMPose and ViTPose backends behave the same on CPU-only, CUDA and macOS (CoreML) builds of ``onnxruntime``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

_CUDA_LIBS_LOADED = False


def _load_cuda_libs(ort) -> None:
    """Load the CUDA / cuDNN libraries shipped as pip packages (``onnxruntime-gpu[cuda,cudnn]``).

    ``onnxruntime.preload_dlls()`` (ONNX Runtime >= 1.21) finds them in site-packages; without it the CUDA
    provider cannot load cuDNN and ONNX Runtime falls back to CPU. System-wide CUDA installs keep working.
    """
    global _CUDA_LIBS_LOADED
    if _CUDA_LIBS_LOADED:
        return
    _CUDA_LIBS_LOADED = True
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        with contextlib.suppress(Exception):
            preload()


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return ort.get_available_providers()


def ort_has_cuda() -> bool:
    """True when the installed onnxruntime build has the CUDA execution provider."""
    return "CUDAExecutionProvider" in _available_providers()


def ort_has_coreml() -> bool:
    """True when the installed onnxruntime build has the CoreML execution provider (the macOS wheels)."""
    return "CoreMLExecutionProvider" in _available_providers()


def resolve_device(device: str) -> str:
    """``'cuda'``, ``'coreml'`` or ``'cpu'``: the device that ``device`` runs on with this onnxruntime build.

    ``'auto'`` picks CUDA, then CoreML, then CPU. ``'cuda'`` and ``'coreml'`` fall back to CPU when the
    provider is missing; any other value means CPU.
    """
    if device in ("auto", "cuda") and ort_has_cuda():
        return "cuda"
    if device in ("auto", "coreml") and ort_has_coreml():
        return "coreml"
    return "cpu"


def ort_providers(device: str, coreml_format: str = "NeuralNetwork") -> list:
    """Execution providers for ``device`` (see :func:`resolve_device`), always with CPU fallback.

    ``coreml_format`` is the CoreML model format, ``'NeuralNetwork'`` or ``'MLProgram'``: the fastest one
    depends on the graph (YOLOX only runs as ``'MLProgram'``).
    """
    device = resolve_device(device)
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "coreml":
        return [("CoreMLExecutionProvider", {"ModelFormat": coreml_format}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(
    model_path: str | Path,
    device: str = "auto",
    intra_op_num_threads: int = 0,
    log_level: int = 3,
    coreml_format: str = "NeuralNetwork",
):
    """Open an ``onnxruntime.InferenceSession``.

    Args:
        model_path: ``.onnx`` file.
        device: ``'auto'``, ``'cuda'``, ``'coreml'`` or ``'cpu'`` (see :func:`resolve_device`).
        intra_op_num_threads: CPU threads per operator; ``0`` keeps onnxruntime's default.
        log_level: onnxruntime log severity (``3`` = errors only).
        coreml_format: CoreML model format when the session runs on CoreML (see :func:`ort_providers`).

    Every other session option keeps onnxruntime's default (full graph optimization).
    """
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover - onnxruntime is a core dependency
        raise ImportError("ONNX models need onnxruntime: pip install onnxruntime (or onnxruntime-gpu)") from e

    providers = ort_providers(device, coreml_format)
    if providers[0] == "CUDAExecutionProvider":
        _load_cuda_libs(ort)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(intra_op_num_threads)
    opts.log_severity_level = log_level
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


def session_device(session) -> str:
    """``'cuda'``, ``'coreml'`` or ``'cpu'`` from the session's first provider."""
    return {"CUDAExecutionProvider": "cuda", "CoreMLExecutionProvider": "coreml"}.get(session.get_providers()[0], "cpu")
