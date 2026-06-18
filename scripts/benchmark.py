"""
Benchmark kNN recall, latency, and index size across all broll-* index types.

Recall@k is computed against the float32 HNSW baseline (broll-*-jina, no suffix).
Query vectors are cached so re-runs don't hit the Jina API again.

Usage:
    python benchmark.py                      # benchmark all broll-* indices
    python benchmark.py --k 10 --runs 20    # tune k and timing iterations
    python benchmark.py --compare           # print table from all saved runs
    python benchmark.py --compare --k 5    # filter saved runs by k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from embed_jina import JinaClient, EmbedConfig
from index_elastic import es_client, knn_search

load_dotenv()

RESULTS_DIR  = Path(__file__).parent.parent / "results"
QUERIES_FILE = Path(__file__).parent / "eval_queries.json"
VECTOR_CACHE = Path(__file__).parent / ".query_vectors_cache.json"


# ---------------------------------------------------------------------------
# Query loading + embedding
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def embed_queries(queries: list[dict], force: bool = False) -> list[list[float]]:
    """Embed all queries and cache to disk — re-runs skip the API call."""
    texts = [q["query"] for q in queries]

    if not force and VECTOR_CACHE.exists():
        cached = json.loads(VECTOR_CACHE.read_text())
        if cached.get("queries") == texts:
            print(f"  Using cached vectors for {len(texts)} queries")
            return cached["vectors"]

    print(f"  Embedding {len(texts)} queries via Jina API…")
    jina = JinaClient()
    vectors = jina.embed(texts, task="retrieval.query", config=EmbedConfig())
    VECTOR_CACHE.write_text(json.dumps({"queries": texts, "vectors": vectors}))
    return vectors


# ---------------------------------------------------------------------------
# Index discovery
# ---------------------------------------------------------------------------

def get_broll_indices(es) -> list[str]:
    """Return all broll-* indices. Float32 baseline first."""
    all_idx = es.indices.get(index="broll-*", expand_wildcards="open")
    indices  = sorted(all_idx.keys())
    baseline = [i for i in indices if i.endswith("-float32")]
    rest     = [i for i in indices if not i.endswith("-float32")]
    return baseline + rest


def index_label(index: str) -> str:
    """Human-readable label for display."""
    if index.endswith("-float32"):   return "float32 HNSW (baseline)"
    if index.endswith("-int8"):      return "int8 HNSW  (4× smaller)"
    if index.endswith("-int4"):      return "int4 HNSW  (8× smaller)"
    if index.endswith("-bbq"):       return "BBQ HNSW   (32× smaller)"
    return index


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_index(
    es,
    index: str,
    query_vectors: list[list[float]],
    k: int = 10,
    runs: int = 20,
) -> dict:
    """Run each query `runs` times, collect latencies and top-k hits."""
    latencies_ms: list[float] = []
    hits_per_query: list[list[str]] = []

    for vec in tqdm(query_vectors, desc=f"  {index}", leave=False):
        first_hits: list[str] = []
        for run in range(runs):
            t0   = time.perf_counter()
            hits = knn_search(es, index, vec, k=k)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            if run == 0:
                first_hits = [h["chunk_id"] for h in hits]
        hits_per_query.append(first_hits)

    return {
        "index":          index,
        "label":          index_label(index),
        "latencies_ms":   latencies_ms,
        "hits_per_query": hits_per_query,
        "k":              k,
        "runs":           runs,
    }


def compute_recall(
    candidate_hits: list[list[str]],
    ground_truth_hits: list[list[str]],
    k: int,
) -> float:
    """Mean recall@k across all queries vs the float32 ground truth."""
    recalls = [
        len(set(c[:k]) & set(g[:k])) / k
        for c, g in zip(candidate_hits, ground_truth_hits)
    ]
    return float(np.mean(recalls))


def get_index_stats(es, index: str) -> dict:
    try:
        count = es.count(index=index)["count"]
    except Exception:
        count = None

    size_mb = seg_mb = None
    try:
        stats   = es.indices.stats(index=index)
        pri     = stats["indices"][index]["primaries"]
        size_mb = round(pri["store"]["size_in_bytes"] / 1024 / 1024, 1)
        # Segment memory includes the HNSW graph — this is what quantization shrinks
        seg_mb  = round(pri["segments"]["memory_in_bytes"] / 1024 / 1024, 2)
    except Exception:
        pass  # serverless — _stats not available

    return {"doc_count": count, "size_mb": size_mb, "seg_mb": seg_mb}


# ---------------------------------------------------------------------------
# Main benchmark run
# ---------------------------------------------------------------------------

def run_benchmarks(
    k: int = 10,
    runs: int = 20,
    queries_path: Path = QUERIES_FILE,
) -> dict:
    print("\n── Queries ──────────────────────────────────")
    queries      = load_queries(queries_path)
    query_vectors = embed_queries(queries)
    print(f"  {len(queries)} queries loaded")

    es      = es_client()
    indices = get_broll_indices(es)
    if not indices:
        sys.exit("No broll-* indices found. Run ingest first.")
    print(f"\n── Indices ──────────────────────────────────")
    for i in indices:
        print(f"  {i}")

    print(f"\n── Running  k={k}  runs_per_query={runs} ──────")
    results: dict[str, dict] = {}
    for index in indices:
        print(f"\n  {index}")
        r           = benchmark_index(es, index, query_vectors, k=k, runs=runs)
        r["stats"]  = get_index_stats(es, index)
        results[index] = r

    # Recall against float32 baseline
    baseline = next((i for i in indices if i.endswith("-float32")), indices[0])
    gt_hits  = results[baseline]["hits_per_query"]
    for index, r in results.items():
        r["recall_at_k"] = (
            1.0 if index == baseline
            else compute_recall(r["hits_per_query"], gt_hits, k)
        )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".3f", suffix=""):
    return f"{val:{fmt}}{suffix}" if val is not None else "—"


def print_table(results: dict, k: int = 10):
    col = [36, 12, 10, 10, 8, 10, 11]
    header = (
        f"{'Index':<{col[0]}}"
        f"{'Recall@'+str(k):<{col[1]}}"
        f"{'p50':>{col[2]}}"
        f"{'p99':>{col[3]}}"
        f"{'Docs':>{col[4]}}"
        f"{'Size MB':>{col[5]}}"
        f"{'Seg MB':>{col[6]}}"
    )
    sep = "─" * sum(col)
    print(f"\n{sep}\n{header}\n{sep}")

    for idx, r in results.items():
        lats  = np.array(r["latencies_ms"])
        p50   = np.percentile(lats, 50)
        p99   = np.percentile(lats, 99)
        rec   = r.get("recall_at_k")
        docs  = r["stats"].get("doc_count")
        size  = r["stats"].get("size_mb")
        seg   = r["stats"].get("seg_mb")
        print(
            f"{idx:<{col[0]}}"
            f"{_fmt(rec):<{col[1]}}"
            f"{_fmt(p50, '.1f', 'ms'):>{col[2]}}"
            f"{_fmt(p99, '.1f', 'ms'):>{col[3]}}"
            f"{str(docs) if docs else '—':>{col[4]}}"
            f"{_fmt(size, '.1f', ' MB'):>{col[5]}}"
            f"{_fmt(seg, '.2f', ' MB'):>{col[6]}}"
        )
    print(sep)


def save_results(results: dict, k: int) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    path = RESULTS_DIR / f"{ts}_k{k}.json"

    serialisable = {}
    for idx, r in results.items():
        serialisable[idx] = {
            "label":        r["label"],
            "recall_at_k":  r.get("recall_at_k"),
            "latencies_ms": r["latencies_ms"],
            "stats":        r["stats"],
            "k":            r["k"],
            "runs":         r["runs"],
        }
    path.write_text(json.dumps({"k": k, "indices": serialisable}, indent=2))
    print(f"\n  Saved → {path.relative_to(Path.cwd())}")
    return path


# ---------------------------------------------------------------------------
# Compare saved runs
# ---------------------------------------------------------------------------

def compare_saved(results_dir: Path = RESULTS_DIR, k_filter: int | None = None):
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print("No saved results found in results/")
        return
    for f in files:
        data = json.loads(f.read_text())
        if k_filter and data["k"] != k_filter:
            continue
        print(f"\n{'━'*70}\n{f.name}  (k={data['k']})")
        _print_saved_table(data)


def _print_saved_table(data: dict):
    k       = data["k"]
    indices = data["indices"]
    col     = [36, 12, 10, 10, 10]
    header  = (
        f"{'Index':<{col[0]}}"
        f"{'Recall@'+str(k):<{col[1]}}"
        f"{'p50':>{col[2]}}"
        f"{'p99':>{col[3]}}"
        f"{'Size MB':>{col[4]}}"
    )
    sep = "─" * sum(col)
    print(f"{sep}\n{header}\n{sep}")
    for idx, r in indices.items():
        lats = np.array(r["latencies_ms"])
        p50  = np.percentile(lats, 50)
        p99  = np.percentile(lats, 99)
        rec  = r.get("recall_at_k")
        size = r["stats"].get("size_mb")
        print(
            f"{idx:<{col[0]}}"
            f"{_fmt(rec):<{col[1]}}"
            f"{_fmt(p50, '.1f', 'ms'):>{col[2]}}"
            f"{_fmt(p99, '.1f', 'ms'):>{col[3]}}"
            f"{_fmt(size, '.1f', ' MB'):>{col[4]}}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark broll-* ES indices")
    ap.add_argument("--k",       type=int, default=10,
                    help="Top-k to evaluate recall against (default: 10)")
    ap.add_argument("--runs",    type=int, default=20,
                    help="Timing runs per query (default: 20)")
    ap.add_argument("--queries", type=Path, default=QUERIES_FILE,
                    help="Path to eval_queries.json")
    ap.add_argument("--compare", action="store_true",
                    help="Print comparison table from all saved results")
    ap.add_argument("--no-save", action="store_true",
                    help="Don't write results to disk")
    args = ap.parse_args()

    if args.compare:
        compare_saved(k_filter=args.k if args.k != 10 else None)
        return

    results = run_benchmarks(k=args.k, runs=args.runs, queries_path=args.queries)
    print_table(results, k=args.k)
    if not args.no_save:
        save_results(results, k=args.k)


if __name__ == "__main__":
    main()
