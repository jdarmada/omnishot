"""
End-to-end ingestion pipeline:

    clips/ → chunks → Jina embeddings → Elasticsearch

This is the script the talk opens with — "50 lines and you've got multimodal
video search." (The rest of the talk is everything you discover after that.)

Usage:
    python ingest.py --clips ./clips --strategy scene --dims 1024 --quant float

You can re-run with different (strategy, dims, quant) to build out the
benchmark matrix; each combination lands in its own index.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as: python scripts/ingest.py
sys.path.insert(0, str(Path(__file__).parent))

from chunk_video import chunk_video, Strategy
from embed_jina import JinaClient, EmbedConfig, video_input
from index_elastic import (
    es_client, create_index, bulk_index, index_name, ChunkDoc,
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def discover_clips(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def load_metadata(metadata_file: Path | None) -> dict[str, dict]:
    """Optional per-clip metadata: uploader, tags, uploaded_at, transcript.
    Keyed by clip filename stem. Returns {} if no file provided."""
    if not metadata_file or not metadata_file.exists():
        return {}
    return json.loads(metadata_file.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True, help="dir of source clips")
    ap.add_argument("--chunks-dir", type=Path, default=Path("./chunks"))
    ap.add_argument("--metadata", type=Path, help="optional metadata JSON")
    ap.add_argument("--strategy", choices=["whole", "scene", "fixed30"], default="scene")
    ap.add_argument("--dims", type=int, default=1024)
    ap.add_argument("--quant", choices=["float", "int8", "bbq"], default="float")
    ap.add_argument("--model", default="jina-embeddings-v5-omni-small")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    clips = discover_clips(args.clips)
    if not clips:
        sys.exit(f"No videos found under {args.clips}")
    print(f"Found {len(clips)} source clips")

    metadata = load_metadata(args.metadata)

    # 1. Chunk
    all_chunks = []
    for clip in tqdm(clips, desc="Chunking"):
        all_chunks.extend(chunk_video(clip, args.chunks_dir, args.strategy))
    print(f"Produced {len(all_chunks)} chunks")

    # 2. Embed (in batches — the Jina API accepts up to 8 video inputs/req)
    jina = JinaClient()
    cfg = EmbedConfig(model=args.model, dimensions=args.dims, embedding_type="float")
    docs: list[ChunkDoc] = []

    for i in tqdm(range(0, len(all_chunks), args.batch_size), desc="Embedding"):
        batch = all_chunks[i:i + args.batch_size]
        inputs = [video_input(c.path) for c in batch]
        vectors = jina.embed(inputs, task="retrieval.passage", config=cfg)

        for chunk, vec in zip(batch, vectors):
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
                embedding=vec,
            ))

    # 3. Index
    es = es_client()
    name = index_name(args.strategy, args.dims, args.quant)
    create_index(es, name, args.dims, args.quant)
    n = bulk_index(es, name, docs)
    print(f"✓ Indexed {n}/{len(docs)} chunks into '{name}'")


if __name__ == "__main__":
    main()
