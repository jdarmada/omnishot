"""
B-Roll Search backend — FastAPI service.

Endpoints:
    GET  /              → serve the search UI
    POST /api/search    → query (text) → ranked clips with metadata
    GET  /api/clip/{id} → stream the underlying clip file
    GET  /api/stats     → index stats for the corner status badge

Designed to be tiny so the talk's demo code is readable on stage.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Allow imports from sibling scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from index_elastic import es_client, semantic_search, hybrid_search, index_name  # noqa: E402


app = FastAPI(title="B-Roll Search")

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Defaults — overridable per request
DEFAULT_INDEX = os.environ.get("BROLL_INDEX", index_name("scene"))

es = es_client()


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
    latency_ms: float
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

    # Search — ES embeds the query via semantic_text automatically
    t_search = time.perf_counter()
    try:
        if req.hybrid:
            hits = hybrid_search(es, index, req.query, k=req.k, filter_clauses=filters)
        else:
            hits = semantic_search(es, index, req.query, k=req.k, filter_clauses=filters)
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")
    search_ms = (time.perf_counter() - t_search) * 1000

    return SearchResponse(
        hits=[Hit(
            chunk_id=h["chunk_id"], clip_id=h["clip_id"], path=h["path"],
            start_sec=h["start_sec"], end_sec=h["end_sec"], duration=h["duration"],
            score=h["_score"], uploader=h.get("uploader", ""),
            uploaded_at=h.get("uploaded_at", ""), tags=h.get("tags", []),
        ) for h in hits],
        latency_ms=search_ms,
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


@app.get("/api/indices")
async def list_indices():
    """List all broll-* indices so the UI can let you switch configs live."""
    all_idx = es.indices.get(index="broll-*", expand_wildcards="open")
    return sorted(all_idx.keys())
