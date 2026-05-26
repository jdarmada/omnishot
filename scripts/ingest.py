"""
End-to-end ingestion pipeline:

    clips/ → chunks → keyframe (data URI) → Elasticsearch (semantic_text)

With semantic_text, Elasticsearch calls the inference endpoint internally
at index time — no manual embedding step needed here. We just extract a
representative keyframe from each chunk, encode it as a data URI, and pass
it as the `content` field. ES handles the rest.

Usage:
    python ingest.py --clips ./clips --strategy scene
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv

# Allow running as: python scripts/ingest.py
sys.path.insert(0, str(Path(__file__).parent))

from chunk_video import chunk_video, Strategy
from embed_elastic import video_input
from index_elastic import (
    es_client, create_index, bulk_index, index_name, ChunkDoc,
)

load_dotenv()

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def discover_clips(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def load_metadata(metadata_file: Path | None) -> dict[str, dict]:
    """Optional per-clip metadata: uploader, tags, uploaded_at, transcript."""
    if not metadata_file or not metadata_file.exists():
        return {}
    return json.loads(metadata_file.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True, help="dir of source clips")
    ap.add_argument("--chunks-dir", type=Path, default=Path("./chunks"))
    ap.add_argument("--metadata", type=Path, help="optional metadata JSON")
    ap.add_argument("--strategy", choices=["whole", "scene", "fixed30"], default="scene")
    ap.add_argument("--inference-id", default=None,
                    help="ES inference endpoint ID (overrides ES_INFERENCE_ID)")
    args = ap.parse_args()

    inference_id = args.inference_id or os.environ.get("ES_INFERENCE_ID")
    if not inference_id:
        sys.exit("Set ES_INFERENCE_ID env var or pass --inference-id")

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

    # 2. Build docs — extract keyframe as data URI, let ES embed via semantic_text
    docs: list[ChunkDoc] = []
    for chunk in tqdm(all_chunks, desc="Building content"):
        try:
            vi = video_input(chunk.path)          # {"image": "<base64>"}
            content = f"data:image/jpeg;base64,{vi['image']}"
        except Exception as e:
            print(f"  ⚠ keyframe extraction failed for {chunk.chunk_id}: {e}")
            continue

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
            content=content,
        ))

    # 3. Index — ES calls the inference endpoint per document via semantic_text
    es = es_client()
    name = index_name(args.strategy)
    create_index(es, name, inference_id)
    print(f"Indexing {len(docs)} chunks into '{name}' (ES embeds each one)...")
    n = bulk_index(es, name, tqdm(docs, desc="Indexing"))
    print(f"✓ Indexed {n}/{len(docs)} chunks into '{name}'")


if __name__ == "__main__":
    main()
