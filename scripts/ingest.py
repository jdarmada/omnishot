"""
Lightweight batch ingest: clips → scene chunks → Jina embed → Elasticsearch.

Use this to preload a folder. The live app also auto-ingests anything dropped
into WATCH_DIR; this script is for one-shot corpus builds.

Usage:
    python scripts/ingest.py --clips ./clips
    python scripts/ingest.py --clips ./clips --cache ./chunks/.embed_cache.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / ".env")

from lib.chunk_video import chunk_video  # noqa: E402
from lib.embed_jina import EmbedConfig, JinaClient  # noqa: E402
from lib.index_elastic import (  # noqa: E402
    ChunkDoc,
    bulk_index,
    create_index,
    es_client,
)
from lib.video_proxy import make_video_input  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
INDEX_DEFAULT = "broll"


def discover_clips(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def load_cache(path: Path | None) -> dict[str, list[float]]:
    if path and path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


def clip_key(p: Path) -> str:
    st = p.stat()
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"


def load_watch_manifest(chunks_dir: Path) -> dict:
    p = chunks_dir / ".manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_watch_manifest(chunks_dir: Path, manifest: dict) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / ".manifest.json").write_text(json.dumps(manifest))


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-ingest video clips into ES")
    ap.add_argument("--clips", type=Path, required=True, help="Folder of source videos")
    ap.add_argument("--chunks-dir", type=Path, default=Path("./chunks"))
    ap.add_argument("--index", default=INDEX_DEFAULT)
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional embed cache JSON (skips Jina on re-runs)",
    )
    args = ap.parse_args()

    clips = discover_clips(args.clips)
    if not clips:
        print(f"No videos found in {args.clips}")
        sys.exit(1)

    es = es_client()
    jina = JinaClient()
    cfg = EmbedConfig()
    create_index(es, args.index, dims=1024)
    cache = load_cache(args.cache)
    manifest = load_watch_manifest(args.chunks_dir)

    print(f"Ingesting {len(clips)} clips → index '{args.index}'")
    total_chunks = 0

    for clip in tqdm(clips, desc="clips"):
        chunks = chunk_video(clip, args.chunks_dir)
        docs: list[ChunkDoc] = []
        for c in chunks:
            if c.chunk_id in cache:
                vec = cache[c.chunk_id]
            else:
                try:
                    inp = make_video_input(c.path)
                    [vec] = jina.embed([inp], task="retrieval.passage", config=cfg)
                    cache[c.chunk_id] = vec
                    if args.cache:
                        save_cache(args.cache, cache)
                except Exception as e:
                    print(f"  ⚠ embed failed for {c.chunk_id}: {e}")
                    continue
            docs.append(
                ChunkDoc(
                    chunk_id=c.chunk_id,
                    clip_id=c.clip_id,
                    path=str(c.path),
                    start_sec=c.start_sec,
                    end_sec=c.end_sec,
                    duration=c.duration,
                    strategy="scene",
                    uploaded_at=time.strftime("%Y-%m-%d"),
                    uploader="ingest",
                    tags=[],
                    transcript=None,
                    embedding=vec,
                )
            )
        if docs:
            bulk_index(es, args.index, docs)
            total_chunks += len(docs)
            try:
                rel = str(clip.resolve().relative_to(args.clips.resolve()))
            except ValueError:
                rel = clip.name
            manifest[rel] = {
                "key": clip_key(clip),
                "source": str(clip.resolve()),
                "chunk_ids": [d.chunk_id for d in docs],
                "chunk_paths": {d.chunk_id: d.path for d in docs},
            }
            save_watch_manifest(args.chunks_dir, manifest)

    es.indices.refresh(index=args.index)
    print(f"Done. Indexed {total_chunks} chunks into '{args.index}'.")
    print(f"Watcher manifest updated at {args.chunks_dir / '.manifest.json'}")


if __name__ == "__main__":
    main()
