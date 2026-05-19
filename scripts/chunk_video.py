"""
Video chunking strategies for the b-roll corpus.

This is one of the talk's central experiments: v5-omni samples 32 evenly-spaced
frames from any video clip you hand it. For a 3-minute clip, that's one frame
per ~5.6 seconds — entire shots get skipped. Scene detection lets us index
sub-clips that fit inside the 32-frame budget.

Three strategies you can compare in `notebooks/02_chunking_experiment.ipynb`:
  - "whole"     : embed the whole clip, accept the 32-frame sampling
  - "scene"     : split with PySceneDetect, embed each scene separately
  - "fixed30"   : naive 30-second windows (the "lazy" baseline)
"""

from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from scenedetect import detect, ContentDetector, AdaptiveDetector

Strategy = Literal["whole", "scene", "fixed30"]


@dataclass
class Chunk:
    """One indexable unit of video."""
    clip_id: str        # parent clip identifier
    chunk_id: str       # unique chunk identifier
    path: Path          # absolute path to the chunk file on disk
    start_sec: float
    end_sec: float
    strategy: Strategy

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def chunk_video(
    video_path: Path,
    out_dir: Path,
    strategy: Strategy = "scene",
    min_scene_len_sec: float = 1.5,
) -> list[Chunk]:
    """Chunk a single video file and write sub-clips to out_dir. Returns chunk records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = video_path.stem

    if strategy == "whole":
        return [Chunk(
            clip_id=clip_id, chunk_id=clip_id, path=video_path,
            start_sec=0.0, end_sec=_duration(video_path), strategy="whole",
        )]

    if strategy == "scene":
        scenes = detect(
            str(video_path),
            AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=int(min_scene_len_sec * 24)),
        )
        # scenes is a list of (start_timecode, end_timecode)
        ranges = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
        if not ranges:  # single-scene video
            ranges = [(0.0, _duration(video_path))]

    elif strategy == "fixed30":
        total = _duration(video_path)
        ranges = [(t, min(t + 30.0, total)) for t in _frange(0, total, 30.0)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    chunks = []
    for i, (start, end) in enumerate(ranges):
        chunk_path = out_dir / f"{clip_id}__{strategy}__{i:03d}.mp4"
        _ffmpeg_extract(video_path, chunk_path, start, end)
        chunks.append(Chunk(
            clip_id=clip_id,
            chunk_id=f"{clip_id}__{strategy}__{i:03d}",
            path=chunk_path,
            start_sec=start,
            end_sec=end,
            strategy=strategy,
        ))
    return chunks


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _duration(path: Path) -> float:
    """ffprobe-based duration in seconds."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.strip())


def _ffmpeg_extract(src: Path, dst: Path, start: float, end: float) -> None:
    """Stream-copy a sub-clip. Fast and lossless when keyframes align;
    falls back to re-encode if not."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(src),
        "-c", "copy", "-avoid_negative_ts", "make_zero",
        str(dst),
    ], check=True)


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    t = start
    while t < stop:
        yield t
        t += step


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("./chunks"))
    p.add_argument("--strategy", choices=["whole", "scene", "fixed30"], default="scene")
    args = p.parse_args()

    chunks = chunk_video(args.video, args.out, args.strategy)
    for c in chunks:
        print(f"  {c.chunk_id}  {c.start_sec:6.2f}s → {c.end_sec:6.2f}s  ({c.duration:5.2f}s)")
    print(f"Wrote {len(chunks)} chunks to {args.out}")
