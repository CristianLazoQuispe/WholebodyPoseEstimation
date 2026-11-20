"""ONNX Runtime session helper shared by every ONNX model (YOLOX, RTMW, ViTPose).

One place decides the execution providers, the CPU thread count and the reported device, so the
RTMPose and ViTPose backends behave the same on CPU-only and CUDA builds of ``onnxruntime``.
"""

from __future__ import annotations

from pathlib import Path


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

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(intra_op_num_threads)
    opts.log_severity_level = log_level
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=ort_providers(device))


def session_device(session) -> str:
    """``'cuda'`` when the session's first provider is CUDA, else ``'cpu'``."""
    return "cuda" if session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
