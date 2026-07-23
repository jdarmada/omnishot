"""
Scene-based video chunking.

Jina v5-omni samples 32 frames from any clip. Scene detection keeps each
indexed unit short enough that those frames actually cover the shot.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from scenedetect import AdaptiveDetector, detect


@dataclass
class Chunk:
    clip_id: str
    chunk_id: str
    path: Path
    start_sec: float
    end_sec: float
    strategy: str = "scene"

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def chunk_video(
    video_path: Path,
    out_dir: Path,
    min_scene_len_sec: float = 1.5,
) -> list[Chunk]:
    """Split a video on scene boundaries and write sub-clips to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = video_path.stem

    scenes = detect(
        str(video_path),
        AdaptiveDetector(
            adaptive_threshold=3.0,
            min_scene_len=int(min_scene_len_sec * 24),
        ),
    )
    ranges = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    if not ranges:
        ranges = [(0.0, _duration(video_path))]

    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(ranges):
        chunk_path = out_dir / f"{clip_id}__scene__{i:03d}.mp4"
        _ffmpeg_extract(video_path, chunk_path, start, end)
        chunks.append(
            Chunk(
                clip_id=clip_id,
                chunk_id=f"{clip_id}__scene__{i:03d}",
                path=chunk_path,
                start_sec=start,
                end_sec=end,
            )
        )
    return chunks


def _duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(out.strip())


def _ffmpeg_extract(src: Path, dst: Path, start: float, end: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(src),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(dst),
        ],
        check=True,
    )
