"""
B-Roll Search backend — FastAPI service.

Endpoints:
    GET  /                    → serve the search UI
    POST /api/search          → query (text) → ranked clips with metadata
    GET  /api/clip/{id}       → stream the underlying clip file
    GET  /api/stats           → index stats for the corner status badge
    GET  /api/indices         → list all broll-* indices
    GET  /api/ingest/stream   → SSE stream for live ingest pipeline progress

Designed to be readable on stage.
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Allow imports from sibling scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from index_elastic import es_client, semantic_search, hybrid_search, index_name  # noqa: E402
from embed_jina import JinaClient, EmbedConfig  # noqa: E402
from ingest import run_ingest  # noqa: E402


app = FastAPI(title="B-Roll Search")

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Defaults — overridable per request
DEFAULT_INDEX = os.environ.get("BROLL_INDEX", index_name("scene", "hnsw", 1024))

es   = es_client()
jina = JinaClient()
_embed_cfg = EmbedConfig()

# One worker so concurrent ingest jobs don't stomp each other
_ingest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return (FRONTEND_DIR / "index.html").read_text()


class SearchRequest(BaseModel):
    query: str
    k: int = 12
    # Optional metadata filters — drives the "uploaded this week" demo
    uploaded_after: Optional[str] = None    # ISO date
    uploader: Optional[str] = None
    max_duration: Optional[float] = None
    tags: Optional[List[str]] = None
    # Hybrid vs. pure vector
    hybrid: bool = False
    index: Optional[str] = None


class Hit(BaseModel):
    chunk_id: str
    clip_id: str
    path: str
    start_sec: float
    end_sec: float
    duration: float
    score: float
    uploader: str = ""
    uploaded_at: str = ""
    tags: list[str] = []


class SearchResponse(BaseModel):
    hits: list[Hit]
    latency_ms: float   # total = embed_ms + search_ms
    embed_ms: float
    search_ms: float
    index_used: str


@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    index = req.index or DEFAULT_INDEX

    # Build filters
    filters = []
    if req.uploaded_after:
        filters.append({"range": {"uploaded_at": {"gte": req.uploaded_after}}})
    if req.uploader:
        filters.append({"term": {"uploader": req.uploader}})
    if req.max_duration:
        filters.append({"range": {"duration": {"lte": req.max_duration}}})
    if req.tags:
        filters.append({"terms": {"tags": req.tags}})

    try:
        t0 = time.perf_counter()
        [query_vector] = jina.embed([req.query], task="retrieval.query", config=_embed_cfg)
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        if req.hybrid:
            hits = hybrid_search(es, index, req.query, query_vector,
                                 k=req.k, filter_clauses=filters)
        else:
            hits = semantic_search(es, index, query_vector,
                                   k=req.k, filter_clauses=filters)
        search_ms = (time.perf_counter() - t1) * 1000
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")

    return SearchResponse(
        hits=[Hit(
            chunk_id=h["chunk_id"], clip_id=h["clip_id"], path=h["path"],
            start_sec=h["start_sec"], end_sec=h["end_sec"], duration=h["duration"],
            score=h["_score"], uploader=h.get("uploader", ""),
            uploaded_at=h.get("uploaded_at", ""), tags=h.get("tags", []),
        ) for h in hits],
        latency_ms=embed_ms + search_ms,
        embed_ms=embed_ms,
        search_ms=search_ms,
        index_used=index,
    )


@app.get("/api/clip/{chunk_id}")
async def get_clip(chunk_id: str):
    """Stream a clip from disk. In production you'd use signed S3 URLs."""
    res = es.get(index=DEFAULT_INDEX, id=chunk_id, _source=["path"])
    path = Path(res["_source"]["path"])
    if not path.exists():
        raise HTTPException(404, f"Clip file missing: {path}")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/stats")
async def stats(index: str = Query(default=DEFAULT_INDEX)):
    if not es.indices.exists(index=index):
        return {"index": index, "exists": False}
    count = es.count(index=index)["count"]
    # _stats API is not available on serverless — return count only
    return {
        "index": index,
        "exists": True,
        "doc_count": count,
        "size_mb": None,
    }


@app.get("/api/pick-folder")
async def pick_folder():
    """Open the native OS folder-picker dialog and return the chosen path.
    Only meaningful when the server is running on the same machine as the browser."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            r = subprocess.run(
                ["osascript", "-e", "POSIX path of (choose folder)"],
                capture_output=True, text=True,
            )
            path = r.stdout.strip().rstrip("/")
        elif system == "Linux":
            # Try zenity (GNOME) then kdialog (KDE)
            for cmd in [
                ["zenity", "--file-selection", "--directory", "--title=Choose clips folder"],
                ["kdialog", "--getexistingdirectory", "."],
            ]:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0:
                    path = r.stdout.strip()
                    break
            else:
                raise HTTPException(400, "No folder picker found (install zenity or kdialog)")
        elif system == "Windows":
            ps = ("Add-Type -AssemblyName System.Windows.Forms;"
                  "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
                  "if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}")
            r = subprocess.run(["powershell", "-Command", ps],
                                capture_output=True, text=True)
            path = r.stdout.strip()
        else:
            raise HTTPException(400, f"Unsupported platform: {system}")
    except FileNotFoundError as e:
        raise HTTPException(400, f"Dialog tool not found: {e}")

    if not path:
        raise HTTPException(204, "No folder selected")
    return {"path": path}


@app.get("/api/indices")
async def list_indices():
    """List all broll-* indices so the UI can let you switch configs live."""
    all_idx = es.indices.get(index="broll-*", expand_wildcards="open")
    return sorted(all_idx.keys())


@app.get("/api/ingest/stream")
async def ingest_stream(
    clips: str,
    chunks_dir: str = "./chunks",
    strategy: str = "scene",
    index_type: str = "hnsw",
    dims: int = 1024,
    batch_size: int = 8,
    cache: str = "",
):
    """SSE stream that runs the ingest pipeline and pushes progress events.

    Each event is JSON:  {"step": "...", "status": "running|done|error", ...}
    A final {"step": "done", "status": "done"} is sent when complete.

    Pass cache=<path> to load/save embeddings so re-running with a different
    index_type skips the Jina API entirely.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[dict | None] = asyncio.Queue()

    embed_cache_path = Path(cache) if cache else None

    def _run() -> None:
        try:
            for event in run_ingest(
                Path(clips), Path(chunks_dir),
                strategy, index_type, dims, batch_size,
                None, embed_cache_path,
            ):
                loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait,
                {"step": "error", "status": "error", "message": str(exc)})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

    loop.run_in_executor(_ingest_pool, _run)

    async def generate():
        while True:
            event = await q.get()
            if event is None:
                yield f"data: {json.dumps({'step': 'done', 'status': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
