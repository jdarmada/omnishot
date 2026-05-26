"""
End-to-end ingestion pipeline:

    clips/ → chunks → Jina API (embed) → Elasticsearch (dense_vector kNN)

Each chunk is compressed to a 640px proxy (CRF 28, no audio) and passed
to the Jina API for 32-frame visual embedding. Vectors are stored as
dense_vector and searched via kNN.

Usage:
    python ingest.py --clips ./clips --strategy scene
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--chunks-dir", type=Path, default=Path("./chunks"))
    ap.add_argument("--metadata", type=Path)
    ap.add_argument("--strategy", choices=["whole", "scene", "fixed30"], default="scene")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="inputs per Jina API call (default: 8)")
    ap.add_argument("--dims", type=int, default=1024,
                    help="Matryoshka truncation: 32-1024 (default: 1024)")
    args = ap.parse_args()

    clips = discover_clips(args.clips)
    if not clips:
        sys.exit(f"No videos found under {args.clips}")
    print(f"Found {len(clips)} source clips")

    metadata = load_metadata(args.metadata)
    jina = JinaClient()
    cfg  = EmbedConfig(dimensions=args.dims)

    # 1. Chunk
    all_chunks = []
    for clip in tqdm(clips, desc="Chunking"):
        all_chunks.extend(chunk_video(clip, args.chunks_dir, args.strategy))
    print(f"Produced {len(all_chunks)} chunks")

    # 2. Build visual inputs (compress each chunk to a small proxy)
    print(f"Batch size: {args.batch_size}")
    inputs = []
    valid_chunks = []
    for chunk in tqdm(all_chunks, desc="Preparing inputs"):
        try:
            inp = make_video_input(chunk.path)
            inputs.append(inp)
            valid_chunks.append(chunk)
        except Exception as e:
            print(f"  ⚠ input prep failed for {chunk.chunk_id}: {e}")

    # 3. Embed in batches via Jina API
    embeddings: list[list[float]] = []
    batches = [inputs[i:i+args.batch_size] for i in range(0, len(inputs), args.batch_size)]
    for batch in tqdm(batches, desc="Embedding (Jina API)"):
        vecs = jina.embed(batch, task="retrieval.passage", config=cfg)
        embeddings.extend(vecs)
    print(f"Got {len(embeddings)} embeddings")

    # 4. Build docs
    docs: list[ChunkDoc] = []
    for chunk, emb in zip(valid_chunks, embeddings):
        meta = metadata.get(chunk.clip_id, {})
        docs.append(ChunkDoc(
            chunk_id=chunk.chunk_id,
            clip_id=chunk.clip_id,
            path=str(chunk.path),
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            duration=chunk.duration,
            strategy=chunk.strategy,
            uploaded_at=meta.get("uploaded_at", dt.date.today().isoformat()),
            uploader=meta.get("uploader", "unknown"),
            tags=meta.get("tags", []),
            transcript=meta.get("transcript"),
            embedding=emb,
        ))

    # 5. Index — no inference calls during indexing, bulk is fast again
    es   = es_client()
    name = index_name(args.strategy)
    create_index(es, name, dims=args.dims)
    print(f"Indexing {len(docs)} chunks into '{name}'...")
    n = bulk_index(es, name, tqdm(docs, desc="Indexing"))
    print(f"✓ Indexed {n}/{len(docs)} chunks into '{name}'")


if __name__ == "__main__":
    main()
