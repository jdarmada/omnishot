# B-Roll Search — Multimodal Video Retrieval Demo

A demo + benchmark toolkit for the AIEWF talk on production multimodal video search using **Jina v5-omni** embeddings and **Elasticsearch**.

The pitch: a video editor at Elastic spends hours scrubbing b-roll to find the shot she needs. With v5-omni and a real search engine, she shouldn't have to. This repo is everything needed to demo that, benchmark it honestly, and present numbers.

## What's in here

```
broll-search/
├── scripts/        # Ingestion + indexing scripts
│   ├── chunk_video.py        # PySceneDetect-based chunking
│   ├── embed_jina.py         # Jina API client (v5-omni)
│   ├── index_elastic.py      # Bulk-index into Elasticsearch
│   └── ingest.py             # End-to-end: clips → ES
├── notebooks/      # Eval and benchmark notebooks
│   ├── 01_baseline_retrieval.ipynb   # Sanity-check the pipeline
│   ├── 02_chunking_experiment.ipynb  # Native vs. scene-chunked
│   ├── 03_compression_bench.ipynb    # Matryoshka × BBQ × rerank
│   └── 04_hybrid_search.ipynb        # Vector + filter + BM25
├── app/            # Lightweight web UI for live demo
│   ├── backend.py            # FastAPI: query → Elasticsearch
│   └── frontend/index.html   # Editor-facing search UI
├── data/           # Eval set (queries.json, relevance.json)
├── clips/          # Sample b-roll (gitignored)
└── embeddings/     # Cached embeddings for offline benchmarks
```

## Quick start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set credentials
export JINA_API_KEY=jina_...
export ES_URL=https://your-cluster.es.io
export ES_API_KEY=...

# 3. Drop a few clips into ./clips/ then:
python scripts/ingest.py --clips ./clips --strategy scene

# 4. Run the demo UI
uvicorn app.backend:app --reload
# Open http://localhost:8000
```

## What each piece is for

| File | Purpose in the talk |
|---|---|
| `scripts/ingest.py` | "The naive solution is 50 lines" — the demo opener |
| `notebooks/02_chunking_experiment.ipynb` | The 32-frame sampling limit, scene chunking vs. native |
| `notebooks/03_compression_bench.ipynb` | The money chart: storage × recall × latency |
| `notebooks/04_hybrid_search.ipynb` | "Find an outdoor shot uploaded this week" |
| `app/` | The live demo — what the editor would actually use |

See `TALK_NOTES.md` for the slide-by-slide mapping.
