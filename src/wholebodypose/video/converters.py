"""Video re-encoding helpers built on the ``ffmpeg`` executable.

The executable is taken from ``imageio-ffmpeg`` (``pip install 'wholebodypose[video]'``,
ships a static binary) or, failing that, from ``ffmpeg`` on ``PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str | None:
    """Return a usable ffmpeg executable, or ``None``."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return shutil.which("ffmpeg")


def _run(cmd: list[str]) -> bool:
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def reencode_for_web(input_path: str | Path, output_path: str | Path | None = None, crf: int = 23) -> tuple[bool, str]:
    """Re-encode a video to H.264 / yuv420p / faststart so browsers can play it.

    Returns ``(success, output_path)``. If ffmpeg is unavailable the input path is returned.
    """
    exe = ffmpeg_exe()
    src = Path(input_path)
    if exe is None:
        return False, str(src)
    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_web.mp4")
    ok = _run(
        [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dst),
        ]
    )
    return ok, str(dst if ok else src)


def convert_mp4_to_mov(input_path: str | Path, output_dir: str | Path, overwrite: bool = True) -> tuple[bool, str]:
    """Convert an MP4 file to MOV (QuickTime) container."""
    exe = ffmpeg_exe()
    src = Path(input_path)
    if exe is None:
        raise RuntimeError("ffmpeg not found. Install with: pip install 'wholebodypose[video]'")
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}.mov"
    cmd = [exe, "-loglevel", "error", "-i", str(src), "-f", "mov"] + (["-y"] if overwrite else ["-n"]) + [str(dst)]
    return _run(cmd), str(dst)
