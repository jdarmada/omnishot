# Contributing to omnishot-ts

Thanks for your interest! This document covers dev setup, testing, and where things live.

## Dev setup

```bash
git clone https://github.com/jdarmada/omnishot-ts
cd omnishot-ts

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt -r backend/requirements-dev.txt

# Frontend
cd frontend && npm install && cd ..

# Elasticsearch
docker compose up -d elasticsearch

# Config
cp .env.example .env   # set JINA_API_KEY
```

Run the backend and frontend in two terminals:

```bash
uvicorn backend.app:app --reload --port 8001
cd frontend && npm run dev
```

## Tests and linting

```bash
pytest                       # backend tests (no ES or Jina needed)
ruff check .                 # Python lint
cd frontend && npm run lint  # TypeScript lint
cd frontend && npm run build # typecheck + production build
```

Tests run with `OMNISHOT_DISABLE_WATCHER=1` and fake credentials (see `backend/tests/conftest.py`), so they never touch Elasticsearch or the Jina API. Please keep it that way — mock at the module boundary.

CI runs all of the above plus a Docker image build on every PR.

## Project layout

```
backend/app.py        FastAPI app: library watcher + search endpoints
backend/lib/          pipeline pieces: chunking, embedding, indexing, proxies
backend/tests/        pytest suite
frontend/src/         Vite + TypeScript UI (api.ts = typed API client)
scripts/              standalone download / batch-ingest CLIs
```

## Extension points

**Embeddings** are isolated in `backend/lib/embed_jina.py`. To add a different
embedding provider (e.g. a local CLIP/SigLIP model), implement the same
interface — `embed(inputs, task, config) -> list[list[float]]` where inputs are
`{"text": ...}`, `{"image": <b64>}`, or `{"video": <b64>}` dicts — and swap the
client construction in `backend/app.py` and `scripts/ingest.py`. Contributions
making this a proper pluggable interface are welcome.

**Index schema** lives in `backend/lib/index_elastic.py` (`create_index`). The
demo uses a single float32 HNSW index; the upstream
[omnishot](https://github.com/jdarmada/omnishot) benchmark explores
quantization and dimension trade-offs if you want data before changing this.

## Pull requests

- Keep PRs focused; one concern per PR.
- Add or update tests for behavior changes.
- Run the lint/test commands above before pushing.
- Note in the PR description if you touched the security-sensitive endpoints
  (`/api/reveal`, `/api/pick-folder`, `/api/clip`, `/api/library`).
