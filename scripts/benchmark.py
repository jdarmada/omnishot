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
    """Return all broll-* indices, sorted by dims desc then quant type.
    Float32 at each dim level comes first (used as per-dim baseline)."""
    all_idx  = es.indices.get(index="broll-*", expand_wildcards="open")
    indices  = sorted(all_idx.keys(), key=lambda i: (-(parse_dims(i) or 0), i))
    return indices


def parse_dims(index: str) -> int | None:
    """Extract dims from index name e.g. broll-scene-1024d-float32 → 1024."""
    for part in index.split("-"):
        if part.endswith("d") and part[:-1].isdigit():
            return int(part[:-1])
    return None


def index_label(index: str) -> str:
    """Human-readable label for display."""
    dims = parse_dims(index) or "?"
    if index.endswith("-float32"):   return f"float32  {dims}d  (baseline)"
    if index.endswith("-int8"):      return f"int8     {dims}d"
    if index.endswith("-int4"):      return f"int4     {dims}d"
    if index.endswith("-bbq"):       return f"BBQ      {dims}d"
    return index


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _truncate_normalize(vec: list[float], dims: int) -> list[float]:
    import math
    t = vec[:dims]
    n = math.sqrt(sum(x * x for x in t))
    return [x / n for x in t] if n > 0 else t


def benchmark_index(
    es,
    index: str,
    query_vectors: list[list[float]],
    k: int = 10,
    runs: int = 20,
    num_candidates: int = 100,
) -> dict:
    """Run each query `runs` times, collect latencies and top-k hits."""
    dims = parse_dims(index)
    vecs = (
        [_truncate_normalize(v, dims) for v in query_vectors]
        if dims and dims < len(query_vectors[0])
        else query_vectors
    )

    latencies_ms: list[float] = []
    hits_per_query: list[list[str]] = []

    for vec in tqdm(vecs, desc=f"  {index}", leave=False):
        first_hits: list[str] = []
        for run in range(runs):
            t0   = time.perf_counter()
            hits = knn_search(es, index, vec, k=k, num_candidates=num_candidates)
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
        "num_candidates": num_candidates,
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

    size_mb = None
    try:
        stats   = es.indices.stats(index=index)
        pri     = stats["indices"][index]["primaries"]
        size_mb = round(pri["store"]["size_in_bytes"] / 1024 / 1024, 1)
    except Exception:
        pass

    return {"doc_count": count, "size_mb": size_mb, "dims": parse_dims(index)}


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

    # Recall against the highest-dim float32 index as ground truth
    float32_indices = [i for i in indices if i.endswith("-float32")]
    baseline = max(float32_indices, key=lambda i: parse_dims(i) or 0, default=indices[0])
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
    col = [38, 12, 8, 10, 10, 8, 10]
    header = (
        f"{'Index':<{col[0]}}"
        f"{'Recall@'+str(k):<{col[1]}}"
        f"{'Dims':>{col[2]}}"
        f"{'p50':>{col[3]}}"
        f"{'p99':>{col[4]}}"
        f"{'Docs':>{col[5]}}"
        f"{'Size MB':>{col[6]}}"
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
        dims  = r["stats"].get("dims")
        print(
            f"{idx:<{col[0]}}"
            f"{_fmt(rec):<{col[1]}}"
            f"{str(dims) if dims else '—':>{col[2]}}"
            f"{_fmt(p50, '.1f', 'ms'):>{col[3]}}"
            f"{_fmt(p99, '.1f', 'ms'):>{col[4]}}"
            f"{str(docs) if docs else '—':>{col[5]}}"
            f"{_fmt(size, '.1f', ' MB'):>{col[6]}}"
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
    col     = [38, 12, 8, 10, 10, 10]
    header  = (
        f"{'Index':<{col[0]}}"
        f"{'Recall@'+str(k):<{col[1]}}"
        f"{'Dims':>{col[2]}}"
        f"{'p50':>{col[3]}}"
        f"{'p99':>{col[4]}}"
        f"{'Size MB':>{col[5]}}"
    )
    sep = "─" * sum(col)
    print(f"{sep}\n{header}\n{sep}")
    for idx, r in indices.items():
        lats = np.array(r["latencies_ms"])
        p50  = np.percentile(lats, 50)
        p99  = np.percentile(lats, 99)
        rec  = r.get("recall_at_k")
        dims = r["stats"].get("dims")
        size = r["stats"].get("size_mb")
        print(
            f"{idx:<{col[0]}}"
            f"{_fmt(rec):<{col[1]}}"
            f"{str(dims) if dims else '—':>{col[2]}}"
            f"{_fmt(p50, '.1f', 'ms'):>{col[3]}}"
            f"{_fmt(p99, '.1f', 'ms'):>{col[4]}}"
            f"{_fmt(size, '.1f', ' MB'):>{col[5]}}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# num_candidates sweep — shows recall recovery curve
# ---------------------------------------------------------------------------

def run_candidates_sweep(
    k: int = 10,
    candidates_list: list[int] = None,
    queries_path: Path = QUERIES_FILE,
) -> dict:
    """For each index, benchmark recall at multiple num_candidates values.

    Returns {index: {nc: {recall, p50, p99}}}
    """
    candidates_list = candidates_list or [50, 100, 200, 500]

    print("\n── Queries ──────────────────────────────────")
    queries       = load_queries(queries_path)
    query_vectors = embed_queries(queries)
    print(f"  {len(queries)} queries loaded")

    es      = es_client()
    indices = get_broll_indices(es)
    if not indices:
        sys.exit("No broll-* indices found. Run ingest first.")

    # Ground truth: highest-dim float32
    float32_indices = [i for i in indices if i.endswith("-float32")]
    baseline = max(float32_indices, key=lambda i: parse_dims(i) or 0, default=indices[0])
    print(f"\n  Ground truth: {baseline}")
    print(f"  Sweeping num_candidates: {candidates_list}")

    # Get ground truth hits (use largest nc for most accurate ground truth)
    gt_result = benchmark_index(es, baseline, query_vectors,
                                k=k, runs=3, num_candidates=max(candidates_list))
    gt_hits   = gt_result["hits_per_query"]

    sweep: dict[str, dict] = {}
    for index in indices:
        sweep[index] = {}
        for nc in candidates_list:
            r = benchmark_index(es, index, query_vectors, k=k, runs=5, num_candidates=nc)
            lats = np.array(r["latencies_ms"])
            recall = (1.0 if index == baseline
                      else compute_recall(r["hits_per_query"], gt_hits, k))
            sweep[index][nc] = {
                "recall": recall,
                "p50":    float(np.percentile(lats, 50)),
                "p99":    float(np.percentile(lats, 99)),
            }
            print(f"  {index}  nc={nc}  recall={recall:.3f}  p50={np.percentile(lats,50):.1f}ms")

    return sweep


def print_candidates_table(sweep: dict, candidates_list: list[int], k: int = 10):
    nc_w  = 9
    idx_w = 38
    lat_w = 14
    total_w = idx_w + nc_w * len(candidates_list) + lat_w
    sep = "─" * total_w

    header = f"{'Index':<{idx_w}}"
    for nc in candidates_list:
        header += f"{'nc='+str(nc):>{nc_w}}"
    header += f"{'p50@nc='+str(candidates_list[-1]):>{lat_w}}"

    print(f"\n{sep}\n{header}\n{sep}")
    for index, nc_results in sweep.items():
        row = f"{index:<{idx_w}}"
        for nc in candidates_list:
            rec = nc_results.get(nc, {}).get("recall")
            row += f"{_fmt(rec, '.3f'):>{nc_w}}"
        p50 = nc_results.get(candidates_list[-1], {}).get("p50")
        row += f"{_fmt(p50, '.1f', 'ms'):>{lat_w}}"
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark broll-* ES indices")
    ap.add_argument("--k",          type=int, default=10,
                    help="Top-k to evaluate recall against (default: 10)")
    ap.add_argument("--runs",       type=int, default=20,
                    help="Timing runs per query (default: 20)")
    ap.add_argument("--queries",    type=Path, default=QUERIES_FILE,
                    help="Path to eval_queries.json")
    ap.add_argument("--compare",    action="store_true",
                    help="Print comparison table from all saved results")
    ap.add_argument("--no-save",    action="store_true",
                    help="Don't write results to disk")
    ap.add_argument("--candidates", nargs="+", type=int, default=None,
                    metavar="N",
                    help="Sweep num_candidates values e.g. --candidates 50 100 200 500")
    args = ap.parse_args()

    if args.candidates:
        sweep = run_candidates_sweep(k=args.k, candidates_list=args.candidates,
                                     queries_path=args.queries)
        print_candidates_table(sweep, args.candidates, k=args.k)
        return

    if args.compare:
        compare_saved(k_filter=args.k if args.k != 10 else None)
        return

    results = run_benchmarks(k=args.k, runs=args.runs, queries_path=args.queries)
    print_table(results, k=args.k)
    if not args.no_save:
        save_results(results, k=args.k)


if __name__ == "__main__":
    main()
