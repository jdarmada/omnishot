# omnishot-ts

[![CI](https://github.com/jdarmada/omnishot-ts/actions/workflows/ci.yml/badge.svg)](https://github.com/jdarmada/omnishot-ts/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Standalone multimodal b-roll search: link any folder of footage, search by text or image, jump to the source clip in your file manager.

Python FastAPI backend + TypeScript (Vite) frontend. Embeddings via **Jina v5-omni-small**; kNN via **Elasticsearch HNSW**.

This is the editor-facing demo extracted from the [omnishot](https://github.com/jdarmada/omnishot) benchmark repo — no quantization matrix, no A/B compare UI.

## Quick start (Docker)

The fastest way to try it — one command runs Elasticsearch + the app:

```bash
git clone https://github.com/jdarmada/omnishot-ts
cd omnishot-ts
cp .env.example .env      # set JINA_API_KEY (free key at jina.ai)
docker compose up -d
```

Open **http://localhost:8001** and drop video files into `./clips`.

> Docker mode watches the mounted `./clips` folder. The native folder picker
> and Reveal button need the backend running directly on your machine — see
> [Run the app](#run-the-app) below for the local setup.

```
omnishot-ts/
├── backend/
│   ├── app.py              # FastAPI: library watch + search APIs
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

**Layout of data**

| What | Where | Who owns it |
|---|---|---|
| Source videos (your library) | Any folder you link | You — add/delete in Finder |
| Scene chunks + embeddings | `./chunks` + Elasticsearch | App (derived; safe to wipe) |
| Default starter folder | `./clips` | Convenience default until you change it |

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
git clone https://github.com/jdarmada/omnishot-ts
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
| `WATCH_DIR` | No | Initial library folder (default `./clips`) |
| `CHUNKS_DIR` | No | Scene chunk output (default `./chunks`) |
| `BROLL_INDEX` | No | ES index name (default `broll-demo`) |

The linked library path is also persisted in `chunks/.library.json` after you change it in the UI.

---

## Start Elasticsearch

```bash
docker compose up -d elasticsearch
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

### Link a library folder

1. Click **Change folder…** — a native OS dialog opens on the machine running the backend
2. Pick any folder of `.mp4` / `.mov` / `.mkv` / `.webm` footage
3. The app indexes that folder (and subfolders). Switching folders clears the previous search corpus and rebuilds from the new path

**Add clips:** copy or save videos into the linked folder.  
**Remove from search:** move or delete the file in the file manager — the watcher drops it from Elasticsearch within a few seconds.

You can also set the path via API without the dialog:

```bash
curl -X POST http://localhost:8001/api/library \
  -H 'Content-Type: application/json' \
  -d '{"path":"/Users/you/Movies/B-roll"}'
```

### What you can do in the UI

- **Text search** — describe the shot visually
- **Image search** — click **Image** to upload a reference, or drag one onto the search bar
- **≈ More** — find visually similar clips (reuses the stored vector, no Jina call)
- **Reveal** — open the source file in Finder / Explorer / file manager (the original in your library)

### Production-style (single port)

```bash
cd frontend && npm run build && cd ..
uvicorn backend.app:app --port 8001
```

Then open **http://localhost:8001** (backend serves `frontend/dist`).

---

## Download videos

Point downloads at your library folder (or at `./clips`, then **Change folder…** to that path).

### Option A: Pexels (short stock clips)

Needs `PEXELS_API_KEY` in `.env`.

```bash
python scripts/download_pexels.py --out ./clips --total 50
python scripts/download_pexels.py --out ~/Movies/B-roll --total 30 --categories nature urban
```

### Option B: YouTube (longer clips, more scenes each)

```bash
python scripts/download_youtube.py --out ./clips --total 20
python scripts/download_youtube.py --out ~/Movies/B-roll --categories nature animals
```

Both scripts checkpoint progress so interrupted runs resume cleanly. If yt-dlp hits 403s: `pip install -U yt-dlp`.

---

## Batch ingest (optional preload)

For a large folder before opening the UI:

```bash
python scripts/ingest.py --clips ~/Movies/B-roll --cache ./chunks/.embed_cache.json
```

Then set the same path as the library (UI or `POST /api/library`) so the watcher does not re-embed files already in the manifest.

---

## How it works

```
library folder  →  scene chunk (PySceneDetect)  →  640px proxy  →  Jina embed
                                                                →  Elasticsearch kNN
```

Membership follows the filesystem: files in the linked folder are searchable; files removed from it leave the corpus. Chunk files under `./chunks` are app-owned derivatives.

---

## Security

This is a **local tool** — it opens file dialogs and your file manager on the
machine running the backend, and it has no authentication. It binds to
`127.0.0.1` by default. **Do not expose it to the public internet.** See
[SECURITY.md](SECURITY.md) for the full threat model and how to report
vulnerabilities.

---

## Contributing

Dev setup, test commands, and extension points (including how to swap in a
different embedding provider) are in [CONTRIBUTING.md](CONTRIBUTING.md).
Releases are tracked in [CHANGELOG.md](CHANGELOG.md).

```bash
pytest                        # backend tests — no ES or API keys needed
ruff check .                  # Python lint
cd frontend && npm run lint   # TypeScript lint
```

---

## Troubleshooting

**`Connection refused` on port 9200** — Docker isn't running, or ES is still starting. Wait ~15s after `docker compose up -d`.

**`compatible-with` version error** — Install an ES 9.x client: `pip install "elasticsearch>=9.0.0"`.

**Backend offline in the UI** — Confirm uvicorn is on port 8001; Vite proxies `/api` there.

**Change folder… does nothing** — The dialog runs on the **server** host (same machine as uvicorn). On Linux install `zenity` or `kdialog`. Cancelled dialogs are ignored.

**Reveal does nothing on Linux** — Opens the parent folder via `xdg-open` (macOS uses Finder `open -R`, Windows uses Explorer `/select`).
