"""
End-to-end ingestion pipeline:

    clips/ → chunks → Jina API (embed) → Elasticsearch (dense_vector kNN)

Each chunk is compressed to a 640px proxy (CRF 28, no audio) and passed
to the Jina API for 32-frame visual embedding. Vectors are stored as
dense_vector and searched via kNN.

Embedding cache: use --cache to save vectors after the first run so that
re-indexing into a different index type (int8, int4, bbq) skips the Jina
API entirely and goes straight to ES bulk indexing.

Usage:
    # First run — embeds and caches vectors, indexes as float32 HNSW
    python ingest.py --clips ./clips --cache ./chunks/.embed_cache.json

    # Subsequent runs — reads cache, skips Jina API, just re-indexes
    python ingest.py --clips ./clips --index-type int8_hnsw --cache ./chunks/.embed_cache.json
    python ingest.py --clips ./clips --index-type int4_hnsw --cache ./chunks/.embed_cache.json
    python ingest.py --clips ./clips --index-type bbq_hnsw  --cache ./chunks/.embed_cache.json
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
import sys
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Generator

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


def make_video_input(chunk_path: Path, max_width: int = 640, crf: int = 28,
                     max_seconds: float = 3.0) -> dict:
    """Compress chunk to a small proxy and return as Jina video input dict."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        proxy_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(chunk_path),
                "-t", str(max_seconds),
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
# Chunk manifest — maps clip_id → list of chunk dicts so re-runs skip
# PySceneDetect's detect() call (which reads every frame of every video)
# ---------------------------------------------------------------------------

def load_chunk_manifest(chunks_dir: Path) -> dict[str, list[dict]]:
    p = chunks_dir / ".chunks_manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_chunk_manifest(chunks_dir: Path, manifest: dict[str, list[dict]]) -> None:
    p = chunks_dir / ".chunks_manifest.json"
    p.write_text(json.dumps(manifest))


def chunks_from_manifest(records: list[dict]) -> list:
    """Reconstruct Chunk objects from manifest records, skipping missing files."""
    from chunk_video import Chunk
    result = []
    for r in records:
        p = Path(r["path"])
        if p.exists():
            result.append(Chunk(
                clip_id=r["clip_id"], chunk_id=r["chunk_id"], path=p,
                start_sec=r["start_sec"], end_sec=r["end_sec"],
                strategy=r["strategy"],
            ))
    return result


# ---------------------------------------------------------------------------
# Embedding cache — maps chunk_id → vector so re-indexing skips Jina API
# ---------------------------------------------------------------------------

def load_embed_cache(path: Path | None) -> dict[str, list[float]]:
    if path and path.exists():
        return json.loads(path.read_text())
    return {}


def save_embed_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.write_text(json.dumps(cache))


def matryoshka_truncate(vec: list[float], dims: int) -> list[float]:
    """Slice first `dims` dimensions and renormalize — valid because Jina v5-omni
    is Matryoshka-trained so the first N dims form a coherent subspace."""
    truncated = vec[:dims]
    norm = math.sqrt(sum(x * x for x in truncated))
    return [x / norm for x in truncated] if norm > 0 else truncated


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_ingest(
    clips_dir: Path,
    chunks_dir: Path = Path("./chunks"),
    strategy: str = "scene",
    index_type: str = "hnsw",
    dims: int = 1024,
    batch_size: int = 8,
    metadata: dict | None = None,
    embed_cache_path: Path | None = None,
) -> Generator[dict, None, None]:
    """Run the full pipeline and yield progress events.

    Each event is a dict with at least:
        step   — "discover" | "chunk" | "compress" | "embed" | "index"
        status — "running" | "done" | "error"
        current, total  — progress counters
    """
    meta          = metadata or {}
    embed_cache   = load_embed_cache(embed_cache_path)
    chunk_manifest = load_chunk_manifest(chunks_dir)
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
    manifest_updated = False
    for i, clip in enumerate(clips):
        clip_id = clip.stem
        if clip_id in chunk_manifest:
            # Re-use cached chunk records — skip PySceneDetect
            cached = chunks_from_manifest(chunk_manifest[clip_id])
            if cached:
                all_chunks.extend(cached)
                yield {"step": "chunk", "status": "running",
                       "current": i + 1, "total": len(clips),
                       "scenes": len(all_chunks)}
                continue
        # Not cached (or files missing) — run scene detection
        new_chunks = chunk_video(clip, chunks_dir, strategy)
        all_chunks.extend(new_chunks)
        chunk_manifest[clip_id] = [
            {"clip_id": c.clip_id, "chunk_id": c.chunk_id, "path": str(c.path),
             "start_sec": c.start_sec, "end_sec": c.end_sec, "strategy": c.strategy}
            for c in new_chunks
        ]
        manifest_updated = True
        yield {"step": "chunk", "status": "running",
               "current": i + 1, "total": len(clips),
               "scenes": len(all_chunks)}
    if manifest_updated:
        save_chunk_manifest(chunks_dir, chunk_manifest)
    yield {"step": "chunk", "status": "done",
           "current": len(all_chunks), "total": len(all_chunks)}

    # Split chunks into cached vs uncached
    cached_chunks   = [c for c in all_chunks if c.chunk_id in embed_cache]
    uncached_chunks = [c for c in all_chunks if c.chunk_id not in embed_cache]
    all_cached      = len(uncached_chunks) == 0

    # ── 3. Compress ───────────────────────────────────────────────────────────
    if all_cached:
        yield {"step": "compress", "status": "done",
               "current": len(all_chunks), "total": len(all_chunks), "cached": True}
    else:
        yield {"step": "compress", "status": "running",
               "current": 0, "total": len(uncached_chunks)}
        inputs: list[dict] = []
        valid_uncached: list = []
        for i, chunk in enumerate(uncached_chunks):
            try:
                inputs.append(make_video_input(chunk.path))
                valid_uncached.append(chunk)
            except Exception as exc:
                yield {"step": "compress", "status": "running",
                       "current": i + 1, "total": len(uncached_chunks),
                       "warn": str(exc)}
                continue
            yield {"step": "compress", "status": "running",
                   "current": i + 1, "total": len(uncached_chunks)}
        yield {"step": "compress", "status": "done",
               "current": len(valid_uncached), "total": len(uncached_chunks)}

    # ── 4. Embed ──────────────────────────────────────────────────────────────
    if all_cached:
        yield {"step": "embed", "status": "done",
               "current": len(all_chunks), "total": len(all_chunks), "cached": True}
    else:
        jina = JinaClient()
        cfg  = EmbedConfig(dimensions=dims)
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]

        yield {"step": "embed", "status": "running",
               "current": 0, "total": len(valid_uncached)}
        for i, batch in enumerate(batches):
            vecs = jina.embed(batch, task="retrieval.passage", config=cfg)
            for chunk, vec in zip(valid_uncached[i * batch_size:], vecs):
                embed_cache[chunk.chunk_id] = vec
            done = min((i + 1) * batch_size, len(valid_uncached))
            yield {"step": "embed", "status": "running",
                   "current": done, "total": len(valid_uncached)}

        # Persist cache so next index type skips this entirely
        if embed_cache_path:
            save_embed_cache(embed_cache_path, embed_cache)

        yield {"step": "embed", "status": "done",
               "current": len(valid_uncached), "total": len(valid_uncached)}

    # ── 5. Index ──────────────────────────────────────────────────────────────
    docs: list[ChunkDoc] = []
    for chunk in all_chunks:
        emb = embed_cache.get(chunk.chunk_id)
        if emb is None:
            continue  # failed to compress/embed
        if dims < len(emb):
            emb = matryoshka_truncate(emb, dims)
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
    name = index_name(strategy, index_type, dims)
    create_index(es, name, dims=dims, index_type=index_type)
    n = bulk_index(es, name, docs)
    yield {"step": "index", "status": "done",
           "current": n, "total": len(docs), "index": name}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips",       type=Path, required=True)
    ap.add_argument("--chunks-dir",  type=Path, default=Path("./chunks"))
    ap.add_argument("--metadata",    type=Path)
    ap.add_argument("--strategy",    choices=["whole", "scene", "fixed30"], default="scene")
    ap.add_argument("--index-type",
                    choices=["hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw"],
                    default="hnsw")
    ap.add_argument("--batch-size",  type=int, default=8)
    ap.add_argument("--dims",        type=int, default=1024)
    ap.add_argument("--cache",       type=Path, default=Path("./chunks/.embed_cache.json"),
                    help="Cache file for embeddings (default: ./chunks/.embed_cache.json)")
    args = ap.parse_args()

    metadata = load_metadata(args.metadata)

    for event in run_ingest(
        args.clips, args.chunks_dir, args.strategy,
        args.index_type, args.dims, args.batch_size,
        metadata, args.cache,
    ):
        step    = event["step"]
        status  = event["status"]
        current = event.get("current", 0)
        total   = event.get("total", 0)

        if status == "done":
            cached = " (from cache)" if event.get("cached") else ""
            extra  = f" → '{event['index']}'" if step == "index" else ""
            print(f"  ✓ {step}: {current}/{total}{cached}{extra}")
        elif status == "error":
            print(f"  ✗ {step}: {event.get('message', 'unknown error')}")
        elif total > 0:
            pct = current / total * 100
            print(f"  {step}: {current}/{total}  ({pct:.0f}%)", end="\r", flush=True)


if __name__ == "__main__":
    main()
