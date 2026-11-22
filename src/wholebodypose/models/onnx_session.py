"""ONNX Runtime session helper shared by every ONNX model (YOLOX, RTMW, ViTPose).

One place decides the execution providers, the CPU thread count and the reported device, so the
RTMPose and ViTPose backends behave the same on CPU-only and CUDA builds of ``onnxruntime``.
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


def ort_has_cuda() -> bool:
    """True when the installed onnxruntime build has the CUDA execution provider."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def ort_providers(device: str) -> list[str]:
    """Execution providers for ``device``.

    ``'auto'`` and ``'cuda'`` use CUDA (with CPU fallback for unsupported nodes) when the installed
    onnxruntime build has it, and the CPU provider otherwise; any other value means CPU.
    """
    if device in ("auto", "cuda") and ort_has_cuda():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_session(model_path: str | Path, device: str = "auto", intra_op_num_threads: int = 0, log_level: int = 3):
    """Open an ``onnxruntime.InferenceSession``.

    Args:
        model_path: ``.onnx`` file.
        device: ``'auto'``, ``'cuda'`` or ``'cpu'`` (see :func:`ort_providers`).
        intra_op_num_threads: CPU threads per operator; ``0`` keeps onnxruntime's default.
        log_level: onnxruntime log severity (``3`` = errors only).

    Every other session option keeps onnxruntime's default (full graph optimization).
    """
    try:
        import onnxruntime as ort
    except ImportError as e:  # pragma: no cover - onnxruntime is a core dependency
        raise ImportError("ONNX models need onnxruntime: pip install onnxruntime (or onnxruntime-gpu)") from e

    providers = ort_providers(device)
    if providers[0] == "CUDAExecutionProvider":
        _load_cuda_libs(ort)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(intra_op_num_threads)
    opts.log_severity_level = log_level
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


def session_device(session) -> str:
    """``'cuda'`` when the session's first provider is CUDA, else ``'cpu'``."""
    return "cuda" if session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
