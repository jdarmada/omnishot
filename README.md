# omnishot-ts

Standalone multimodal b-roll search: drop videos in a folder, search by text or image, jump to the source clip in your file manager.

Python FastAPI backend + TypeScript (Vite) frontend. Embeddings via **Jina v5-omni-small**; kNN via **Elasticsearch HNSW**.

This is the editor-facing demo extracted from the [omnishot](https://github.com/your-username/omnishot) benchmark repo — no quantization matrix, no A/B compare UI.

```
omnishot-ts/
├── backend/
│   ├── app.py              # FastAPI: folder watch + search APIs
│   ├── requirements.txt
│   └── lib/                # chunk → embed → index helpers
├── frontend/               # Vite + TypeScript UI
├── scripts/
│   ├── download_pexels.py  # stock footage (needs PEXELS_API_KEY)
│   ├── download_youtube.py # longer clips via yt-dlp
│   └── ingest.py           # one-shot batch ingest
├── docker-compose.yml      # local Elasticsearch 9.x
└── .env.example
```

---

## Prerequisites

- **Python 3.9+**
- **Node.js 18+** (frontend)
- **ffmpeg** (chunking + proxy compression)
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt-get install ffmpeg
  ```
- **Docker Desktop** (recommended for local Elasticsearch)
- A free **[Jina API key](https://jina.ai)**

---

## Setup

```bash
git clone <this-repo>
cd omnishot-ts

# Python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Config
cp .env.example .env
# edit .env — at minimum set JINA_API_KEY
```

| Variable | Required | Description |
|---|---|---|
| `JINA_API_KEY` | Yes | Embeddings for ingest + search |
| `ES_URL` | Yes | `http://localhost:9200` or your cloud URL |
| `ES_API_KEY` | No | Needed for secured / cloud clusters |
| `PEXELS_API_KEY` | No | Only for `download_pexels.py` |
| `WATCH_DIR` | No | Folder the app watches (default `./clips`) |
| `CHUNKS_DIR` | No | Scene chunk output (default `./chunks`) |
| `BROLL_INDEX` | No | ES index name (default `broll-demo`) |

---

## Start Elasticsearch

```bash
docker compose up -d
curl http://localhost:9200   # should return cluster info
```

Or point `ES_URL` / `ES_API_KEY` at Elastic Cloud and skip Docker.

---

## Run the app

**Terminal 1 — backend** (from repo root, venv active):

```bash
uvicorn backend.app:app --reload --port 8001
```

**Terminal 2 — frontend**:

```bash
cd frontend && npm run dev
```

Open **http://localhost:5173**.

The Vite dev server proxies `/api` to the backend. Drop `.mp4` / `.mov` / `.mkv` / `.webm` files into `./clips` — within a few seconds they are scene-chunked, embedded, and searchable.

### What you can do in the UI

- **Text search** — describe the shot visually
- **Image search** — drag a reference image onto the search bar
- **≈ More** — find visually similar clips (reuses the stored vector, no Jina call)
- **Reveal** — open the source file in Finder / Explorer / file manager

### Production-style (single port)

```bash
cd frontend && npm run build && cd ..
uvicorn backend.app:app --port 8001
```

Then open **http://localhost:8001** (backend serves `frontend/dist`).

---

## Download videos

Downloads land under `./clips/<category>/`. The live app watches `./clips` recursively via name, and batch ingest walks the tree — either path works.

### Option A: Pexels (short stock clips)

Needs `PEXELS_API_KEY` in `.env`.

```bash
# ~50 clips across 6 categories (default)
python scripts/download_pexels.py --out ./clips --total 50

# Specific categories
python scripts/download_pexels.py --out ./clips --total 30 --categories nature urban animals
```

### Option B: YouTube (longer clips, more scenes each)

Uses `yt-dlp`; no API key. Clips are filtered to roughly 2–20 minutes.

```bash
python scripts/download_youtube.py --out ./clips --total 20

python scripts/download_youtube.py --out ./clips --total 40 --categories nature animals
```

Both scripts checkpoint progress (`.checkpoint.json` / `.yt_checkpoint.json`) so interrupted runs resume cleanly.

If yt-dlp hits 403s, update it: `pip install -U yt-dlp`.

---

## Ingest videos

### Path 1 — live folder watch (default)

1. Start the backend
2. Copy videos into `./clips` (or `WATCH_DIR`)
3. Watch the status line: clips are indexed automatically

No extra command needed.

### Path 2 — batch ingest (preload)

Useful for a large folder before you open the UI, or to rebuild the index with an embed cache:

```bash
python scripts/ingest.py --clips ./clips --cache ./chunks/.embed_cache.json
```

This writes the same `broll-demo` index and a `.demo_manifest.json` so the folder watcher will not re-embed those files when the app starts.

Re-runs with `--cache` skip Jina calls for chunks already embedded.

---

## How it works

```
clips/  →  scene chunk (PySceneDetect)  →  640px proxy  →  Jina embed
                                                         →  Elasticsearch kNN
```

Query text or an image is embedded with `retrieval.query` into the same space as video passages (`retrieval.passage`), then searched with HNSW.

---

## Troubleshooting

**`Connection refused` on port 9200** — Docker isn't running, or ES is still starting. Wait ~15s after `docker compose up -d`.

**`compatible-with` version error** — Install an ES 9.x client: `pip install "elasticsearch>=9.0.0"`.

**Backend offline in the UI** — Confirm uvicorn is on port 8001; Vite proxies `/api` there.

**Reveal does nothing on Linux** — Opens the parent folder via `xdg-open` (macOS uses Finder `open -R`, Windows uses Explorer `/select`).
