"""Compress a video chunk to a small proxy for the Jina video embedding API."""

from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path


def make_video_input(
    chunk_path: Path,
    max_width: int = 640,
    crf: int = 28,
    max_seconds: float = 3.0,
) -> dict:
    """Return a Jina video input dict with a short 640px proxy as base64."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        proxy_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(chunk_path),
                "-t",
                str(max_seconds),
                "-vf",
                f"scale='if(gt(iw,ih),{max_width},-2)':'if(gt(iw,ih),-2,{max_width})'",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "veryfast",
                "-an",
                "-movflags",
                "+faststart",
                proxy_path,
            ],
            check=True,
        )
        data = base64.b64encode(Path(proxy_path).read_bytes()).decode("ascii")
    finally:
        Path(proxy_path).unlink(missing_ok=True)
    return {"video": data}
