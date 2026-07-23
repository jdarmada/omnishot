"""
omnishot-ts — folder-watch b-roll search.

Links one library folder on disk. Videos added there are indexed; files
removed from the folder leave the search corpus. Chunks stay in CHUNKS_DIR.

Usage:
    uvicorn backend.app:app --reload --port 8001

Env:
    WATCH_DIR   initial library folder   (default: ./clips)
    CHUNKS_DIR  where chunk files go     (default: ./chunks)
    JINA_API_KEY, ES_URL, ES_API_KEY     as usual (.env)
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.chunk_video import chunk_video  # noqa: E402
from lib.embed_jina import EmbedConfig, JinaClient  # noqa: E402
from lib.index_elastic import (  # noqa: E402
    ChunkDoc,
    bulk_index,
    create_index,
    es_client,
    knn_search,
)
from lib.video_proxy import make_video_input  # noqa: E402

load_dotenv(ROOT / ".env")

CHUNKS_DIR = Path(os.environ.get("CHUNKS_DIR", "./chunks")).resolve()
INDEX = os.environ.get("BROLL_INDEX", "broll-demo")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MANIFEST = CHUNKS_DIR / ".demo_manifest.json"
LIBRARY_CFG = CHUNKS_DIR / ".library.json"
SCAN_EVERY = 4.0
FRONTEND_DIST = ROOT / "frontend" / "dist"
DEFAULT_LIBRARY = Path(os.environ.get("WATCH_DIR", "./clips")).resolve()

app = FastAPI(title="omnishot-ts")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

es = es_client()
jina = JinaClient()
_cfg = EmbedConfig()

_lib_lock = threading.Lock()
_watch_dir = DEFAULT_LIBRARY
_library_generation = 0  # bumped when library path changes → watcher resets

status = {"clips": 0, "chunks": 0, "state": "starting", "current": None}
events: list[dict] = []


def log_event(msg: str) -> None:
    events.append({"t": time.strftime("%H:%M:%S"), "msg": msg})
    del events[:-8]


def _load_library_path() -> Path:
    if LIBRARY_CFG.exists():
        try:
            raw = json.loads(LIBRARY_CFG.read_text()).get("path")
            if raw:
                return Path(raw).expanduser().resolve()
        except Exception:
            pass
    return DEFAULT_LIBRARY


def _save_library_path(path: Path) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_CFG.write_text(json.dumps({"path": str(path)}))


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _save_manifest(m: dict) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m))


def _clip_key(p: Path) -> str:
    st = p.stat()
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"


def _watch_key(p: Path, watch_dir: Path) -> str:
    try:
        return str(p.resolve().relative_to(watch_dir))
    except ValueError:
        return p.name


def _clear_corpus(manifest: dict) -> None:
    """Remove every indexed clip and its chunk files (used when switching libraries)."""
    for name in list(manifest.keys()):
        _remove_clip(name, manifest, quiet=True)
    _save_manifest({})
    try:
        if es.indices.exists(index=INDEX):
            es.delete_by_query(
                index=INDEX,
                query={"match_all": {}},
                refresh=True,
                conflicts="proceed",
            )
    except Exception as e:
        print(f"  ⚠ index clear: {e}")


def _ingest_clip(clip: Path, watch_dir: Path, manifest: dict) -> None:
    status.update(state="processing", current=clip.name)
    chunks = chunk_video(clip, CHUNKS_DIR)
    docs = []
    for c in chunks:
        try:
            inp = make_video_input(c.path)
            [vec] = jina.embed([inp], task="retrieval.passage", config=_cfg)
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
                uploader="demo",
                tags=[],
                transcript=None,
                embedding=vec,
            )
        )
    if docs:
        bulk_index(es, INDEX, docs)
        es.indices.refresh(index=INDEX)
    key = _watch_key(clip, watch_dir)
    manifest[key] = {
        "key": _clip_key(clip),
        "source": str(clip),
        "chunk_ids": [d.chunk_id for d in docs],
        "chunk_paths": {d.chunk_id: d.path for d in docs},
    }
    _save_manifest(manifest)
    log_event(f"{key} → {len(docs)} scenes indexed, searchable")
    print(f"  ✓ {key}: {len(docs)} chunks indexed")


def _remove_clip(name: str, manifest: dict, quiet: bool = False) -> None:
    entry = manifest.pop(name, None)
    if not entry:
        return
    for cid in entry["chunk_ids"]:
        try:
            es.delete(index=INDEX, id=cid)
        except Exception:
            pass
        p = Path(entry["chunk_paths"].get(cid, ""))
        if p.exists() and CHUNKS_DIR in p.resolve().parents:
            p.unlink(missing_ok=True)
    try:
        es.indices.refresh(index=INDEX)
    except Exception:
        pass
    _save_manifest(manifest)
    if not quiet:
        log_event(f"{name} removed from the index")
        print(f"  ✗ {name}: removed from index")


def _refresh_status(manifest: dict) -> None:
    status["clips"] = len(manifest)
    status["chunks"] = sum(len(e["chunk_ids"]) for e in manifest.values())


def set_library(path: Path, *, clear: bool = True) -> Path:
    """Retarget the watcher to a new library folder."""
    global _watch_dir, _library_generation
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Folder does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    with _lib_lock:
        if path == _watch_dir:
            return _watch_dir
        old = _watch_dir
        if clear:
            _clear_corpus(_load_manifest())
        _watch_dir = path
        _library_generation += 1
        _save_library_path(path)
        status.update(clips=0, chunks=0, state="switching", current=None)
        log_event(f"library → {path}")
        print(f"  ↪ library changed: {old} → {path}")
        return _watch_dir


def watcher() -> None:
    global _watch_dir
    create_index(es, INDEX, dims=1024)
    with _lib_lock:
        _watch_dir = _load_library_path()
        if _watch_dir == DEFAULT_LIBRARY:
            _watch_dir.mkdir(parents=True, exist_ok=True)
        _save_library_path(_watch_dir)

    generation = -1
    manifest: dict = {}
    pending_sizes: dict[str, int] = {}
    watch_dir = _watch_dir

    while True:
        try:
            with _lib_lock:
                current_gen = _library_generation
                watch_dir = _watch_dir

            if current_gen != generation:
                generation = current_gen
                pending_sizes = {}
                manifest = _load_manifest()
                _refresh_status(manifest)
                status.update(state="watching", current=None)

            if not watch_dir.is_dir():
                status.update(state=f"error: library missing ({watch_dir})", current=None)
                time.sleep(SCAN_EVERY)
                continue

            on_disk = {
                _watch_key(p, watch_dir): p
                for p in watch_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            }

            for name in [n for n in manifest if n not in on_disk]:
                _remove_clip(name, manifest)

            for name, p in on_disk.items():
                known = manifest.get(name)
                if known and known["key"] == _clip_key(p):
                    continue
                size = p.stat().st_size
                if pending_sizes.get(name) != size:
                    pending_sizes[name] = size
                    continue
                del pending_sizes[name]
                if known:
                    _remove_clip(name, manifest)
                _ingest_clip(p, watch_dir, manifest)

            _refresh_status(manifest)
            if status.get("state") != "processing":
                status.update(state="watching", current=None)
        except Exception as e:
            print(f"watcher error: {e}")
            status.update(state=f"error: {e}", current=None)
        time.sleep(SCAN_EVERY)


threading.Thread(target=watcher, daemon=True).start()


class SearchRequest(BaseModel):
    query: str
    k: int = 9


class ImageSearchRequest(BaseModel):
    image_b64: str
    k: int = 9


class LibraryRequest(BaseModel):
    path: str


@app.get("/api/status")
async def api_status():
    with _lib_lock:
        watch = str(_watch_dir)
    return {
        **status,
        "watch_dir": watch,
        "events": list(reversed(events)),
    }


@app.get("/api/pick-folder")
async def pick_folder():
    """Open a native OS folder dialog (server machine) and return the path."""
    system = platform.system()
    try:
        if system == "Darwin":
            r = subprocess.run(
                ["osascript", "-e", "POSIX path of (choose folder)"],
                capture_output=True,
                text=True,
            )
            path = r.stdout.strip().rstrip("/")
        elif system == "Linux":
            path = ""
            for cmd in [
                ["zenity", "--file-selection", "--directory", "--title=Choose library folder"],
                ["kdialog", "--getexistingdirectory", "."],
            ]:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0:
                    path = r.stdout.strip()
                    break
            if not path:
                raise HTTPException(
                    400, "No folder picker found (install zenity or kdialog)"
                )
        elif system == "Windows":
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
                "if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}"
            )
            r = subprocess.run(
                ["powershell", "-Command", ps], capture_output=True, text=True
            )
            path = r.stdout.strip()
        else:
            raise HTTPException(400, f"Unsupported platform: {system}")
    except FileNotFoundError as e:
        raise HTTPException(400, f"Dialog tool not found: {e}") from e

    if not path:
        raise HTTPException(400, "No folder selected")
    return {"path": path}


@app.post("/api/library")
async def set_library_api(req: LibraryRequest):
    """Point the watcher at a library folder. Clears the previous corpus."""
    try:
        path = set_library(Path(req.path), clear=True)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except NotADirectoryError as e:
        raise HTTPException(400, str(e)) from e
    return {"watch_dir": str(path), "cleared": True}


@app.post("/api/library/pick")
async def pick_and_set_library():
    """Native folder picker, then retarget the library in one step."""
    picked = await pick_folder()
    return await set_library_api(LibraryRequest(path=picked["path"]))


def _hits_payload(hits, exclude_id: str | None = None, k: int = 9):
    out = []
    seen_clips: set[str] = set()
    for h in hits:
        if h["chunk_id"] == exclude_id:
            continue
        if h["clip_id"] in seen_clips:
            continue
        seen_clips.add(h["clip_id"])
        out.append(
            {
                "chunk_id": h["chunk_id"],
                "clip_id": h["clip_id"],
                "score": h["_score"],
                "duration": h["duration"],
                "start_sec": h["start_sec"],
                "end_sec": h["end_sec"],
            }
        )
    return out[:k]


@app.post("/api/similar/{chunk_id}")
async def similar(chunk_id: str):
    try:
        doc = es.get(index=INDEX, id=chunk_id, source_includes=["embedding"])
        vec = doc["_source"]["embedding"]
        t0 = time.perf_counter()
        hits = knn_search(es, INDEX, vec, k=50, num_candidates=100)
        search_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        raise HTTPException(500, f"Similar search failed: {e}") from e
    return {
        "hits": _hits_payload(hits, exclude_id=chunk_id),
        "embed_ms": 0.0,
        "search_ms": search_ms,
    }


@app.post("/api/search_image")
async def search_image(req: ImageSearchRequest):
    try:
        t0 = time.perf_counter()
        [qv] = jina.embed(
            [{"image": req.image_b64}], task="retrieval.query", config=_cfg
        )
        embed_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        hits = knn_search(es, INDEX, qv, k=50, num_candidates=100)
        search_ms = (time.perf_counter() - t1) * 1000
    except Exception as e:
        raise HTTPException(500, f"Image search failed: {e}") from e
    return {
        "hits": _hits_payload(hits, k=req.k),
        "embed_ms": embed_ms,
        "search_ms": search_ms,
    }


@app.post("/api/search")
async def search(req: SearchRequest):
    try:
        t0 = time.perf_counter()
        [qv] = jina.embed([req.query], task="retrieval.query", config=_cfg)
        embed_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        hits = knn_search(es, INDEX, qv, k=50, num_candidates=100)
        search_ms = (time.perf_counter() - t1) * 1000
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}") from e
    return {
        "hits": _hits_payload(hits, k=req.k),
        "embed_ms": embed_ms,
        "search_ms": search_ms,
    }


@app.get("/api/clip/{chunk_id}")
async def get_clip(chunk_id: str):
    res = es.get(index=INDEX, id=chunk_id, _source=["path"])
    path = Path(res["_source"]["path"])
    if not path.exists():
        raise HTTPException(404, "chunk file missing")
    return FileResponse(path, media_type="video/mp4")


def _reveal_in_file_manager(path: Path) -> None:
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
    elif system == "Windows":
        subprocess.run(["explorer", "/select,", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=False)


@app.post("/api/reveal/{chunk_id}")
async def reveal(chunk_id: str):
    res = es.get(index=INDEX, id=chunk_id, _source=["clip_id"])
    clip_id = res["_source"]["clip_id"]
    manifest = _load_manifest()
    for entry in manifest.values():
        src = Path(entry["source"])
        if src.stem == clip_id and src.exists():
            _reveal_in_file_manager(src)
            return {"revealed": str(src)}
    raise HTTPException(404, "source clip not found in library folder")


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
