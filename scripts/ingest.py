"""
End-to-end ingestion pipeline:

    clips/ → chunks → Jina API (embed) → Elasticsearch (dense_vector kNN)

Each chunk is compressed to a 640px proxy (CRF 28, no audio) and passed
to the Jina API for 32-frame visual embedding. Vectors are stored as
dense_vector and searched via kNN.

Usage:
    python ingest.py --clips ./clips --strategy scene
    python ingest.py --clips ./clips --index-type int8_hnsw
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Generator

from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from chunk_video import chunk_video, Strategy
from embed_jina import JinaClient, EmbedConfig
from index_elastic import (
    es_client, create_index, bulk_index, index_name, ChunkDoc,
)

load_dotenv()

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def discover_clips(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def load_metadata(metadata_file: Path | None) -> dict[str, dict]:
    if not metadata_file or not metadata_file.exists():
        return {}
    return json.loads(metadata_file.read_text())


def make_video_input(chunk_path: Path, max_width: int = 640, crf: int = 28) -> dict:
    """Compress chunk to a small proxy and return as Jina video input dict."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        proxy_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(chunk_path),
                "-vf", f"scale='if(gt(iw,ih),{max_width},-2)':'if(gt(iw,ih),-2,{max_width})'",
                "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
                "-an", "-movflags", "+faststart",
                proxy_path,
            ],
            check=True,
        )
        data = base64.b64encode(Path(proxy_path).read_bytes()).decode("ascii")
    finally:
        Path(proxy_path).unlink(missing_ok=True)
    return {"video": data}


# ---------------------------------------------------------------------------
# Core pipeline — yields structured progress events so callers (CLI or the
# FastAPI SSE endpoint) can display live progress.
# ---------------------------------------------------------------------------

def run_ingest(
    clips_dir: Path,
    chunks_dir: Path = Path("./chunks"),
    strategy: str = "scene",
    index_type: str = "hnsw",
    dims: int = 1024,
    batch_size: int = 8,
    metadata: dict | None = None,
) -> Generator[dict, None, None]:
    """Run the full pipeline and yield progress events.

    Each event is a dict with at least:
        step   — "discover" | "chunk" | "compress" | "embed" | "index"
        status — "running" | "done" | "error"
        current, total  — progress counters
    """
    meta = metadata or {}
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover ──────────────────────────────────────────────────────────
    yield {"step": "discover", "status": "running", "current": 0, "total": 0}
    clips = discover_clips(clips_dir)
    if not clips:
        yield {"step": "discover", "status": "error",
               "message": f"No videos found in {clips_dir}"}
        return
    yield {"step": "discover", "status": "done",
           "current": len(clips), "total": len(clips)}

    # ── 2. Chunk ─────────────────────────────────────────────────────────────
    yield {"step": "chunk", "status": "running", "current": 0, "total": len(clips)}
    all_chunks = []
    for i, clip in enumerate(clips):
        all_chunks.extend(chunk_video(clip, chunks_dir, strategy))
        yield {"step": "chunk", "status": "running",
               "current": i + 1, "total": len(clips),
               "scenes": len(all_chunks)}
    yield {"step": "chunk", "status": "done",
           "current": len(all_chunks), "total": len(all_chunks)}

    # ── 3. Compress ───────────────────────────────────────────────────────────
    yield {"step": "compress", "status": "running",
           "current": 0, "total": len(all_chunks)}
    inputs: list[dict] = []
    valid_chunks = []
    for i, chunk in enumerate(all_chunks):
        try:
            inputs.append(make_video_input(chunk.path))
            valid_chunks.append(chunk)
        except Exception as exc:
            yield {"step": "compress", "status": "running",
                   "current": i + 1, "total": len(all_chunks),
                   "warn": str(exc)}
            continue
        yield {"step": "compress", "status": "running",
               "current": i + 1, "total": len(all_chunks)}
    yield {"step": "compress", "status": "done",
           "current": len(valid_chunks), "total": len(all_chunks)}

    # ── 4. Embed ──────────────────────────────────────────────────────────────
    jina = JinaClient()
    cfg  = EmbedConfig(dimensions=dims)
    batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
    embeddings: list[list[float]] = []

    yield {"step": "embed", "status": "running",
           "current": 0, "total": len(valid_chunks)}
    for i, batch in enumerate(batches):
        vecs = jina.embed(batch, task="retrieval.passage", config=cfg)
        embeddings.extend(vecs)
        done = min((i + 1) * batch_size, len(valid_chunks))
        yield {"step": "embed", "status": "running",
               "current": done, "total": len(valid_chunks)}
    yield {"step": "embed", "status": "done",
           "current": len(embeddings), "total": len(valid_chunks)}

    # ── 5. Index ──────────────────────────────────────────────────────────────
    docs: list[ChunkDoc] = []
    for chunk, emb in zip(valid_chunks, embeddings):
        m = meta.get(chunk.clip_id, {})
        docs.append(ChunkDoc(
            chunk_id=chunk.chunk_id,
            clip_id=chunk.clip_id,
            path=str(chunk.path),
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            duration=chunk.duration,
            strategy=chunk.strategy,
            uploaded_at=m.get("uploaded_at", dt.date.today().isoformat()),
            uploader=m.get("uploader", "unknown"),
            tags=m.get("tags", []),
            transcript=m.get("transcript"),
            embedding=emb,
        ))

    yield {"step": "index", "status": "running",
           "current": 0, "total": len(docs)}
    es   = es_client()
    name = index_name(strategy, index_type)
    create_index(es, name, dims=dims, index_type=index_type)
    n = bulk_index(es, name, docs)
    yield {"step": "index", "status": "done",
           "current": n, "total": len(docs), "index": name}


# ---------------------------------------------------------------------------
# CLI entry-point — wraps run_ingest with a simple tqdm-style output
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips",       type=Path, required=True)
    ap.add_argument("--chunks-dir",  type=Path, default=Path("./chunks"))
    ap.add_argument("--metadata",    type=Path)
    ap.add_argument("--strategy",    choices=["whole", "scene", "fixed30"], default="scene")
    ap.add_argument("--index-type",
                    choices=["hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw"],
                    default="hnsw",
                    help="Vector index quantization (default: hnsw / float32)")
    ap.add_argument("--batch-size",  type=int, default=8,
                    help="Inputs per Jina API call (default: 8)")
    ap.add_argument("--dims",        type=int, default=1024,
                    help="Matryoshka truncation: 32-1024 (default: 1024)")
    args = ap.parse_args()

    metadata = load_metadata(args.metadata)

    for event in run_ingest(
        args.clips, args.chunks_dir, args.strategy,
        args.index_type, args.dims, args.batch_size, metadata,
    ):
        step    = event["step"]
        status  = event["status"]
        current = event.get("current", 0)
        total   = event.get("total", 0)

        if status == "done":
            extra = f" → '{event['index']}'" if step == "index" else ""
            print(f"  ✓ {step}: {current}/{total}{extra}")
        elif status == "error":
            print(f"  ✗ {step}: {event.get('message', 'unknown error')}")
        elif total > 0:
            pct = current / total * 100
            print(f"  {step}: {current}/{total}  ({pct:.0f}%)", end="\r", flush=True)


if __name__ == "__main__":
    main()
