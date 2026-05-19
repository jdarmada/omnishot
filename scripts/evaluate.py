"""
Evaluation harness for the b-roll search benchmark.

This is the code behind the "money chart" — recall vs. storage across configs.
Designed to be called from notebooks, but can also run standalone.

Metrics computed per (config, query) and aggregated:
    - recall@k        : did >=1 relevant chunk appear in top-k?
    - mrr             : reciprocal rank of first relevant hit (0 if missed)
    - precision@k     : fraction of top-k that are relevant
    - latency_ms      : wall-clock query time at p50 / p95

Storage cost is computed from the embedding dim × bytes-per-element × N docs.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np


@dataclass
class QueryResult:
    query_id: str
    config_label: str
    hits: list[str]                # chunk_ids in rank order
    latency_ms: float
    relevant: list[str]            # ground truth

    @property
    def first_relevant_rank(self) -> int | None:
        for i, h in enumerate(self.hits):
            if h in self.relevant:
                return i + 1  # 1-indexed
        return None

    def recall_at(self, k: int) -> float:
        return 1.0 if any(h in self.relevant for h in self.hits[:k]) else 0.0

    def precision_at(self, k: int) -> float:
        if k == 0 or not self.hits:
            return 0.0
        return sum(1 for h in self.hits[:k] if h in self.relevant) / k

    @property
    def mrr(self) -> float:
        r = self.first_relevant_rank
        return 1.0 / r if r else 0.0


@dataclass
class ConfigReport:
    label: str
    dims: int
    quantization: str               # 'float' | 'int8' | 'bbq'
    num_docs: int
    results: list[QueryResult] = field(default_factory=list)

    @property
    def bytes_per_vec(self) -> int:
        return {
            "float": self.dims * 4,
            "int8":  self.dims * 1,
            "bbq":   max(1, self.dims // 8),
        }[self.quantization]

    @property
    def total_storage_mb(self) -> float:
        return self.bytes_per_vec * self.num_docs / (1024 * 1024)

    def summary(self) -> dict:
        if not self.results:
            return {"label": self.label, "n_queries": 0}
        latencies = [r.latency_ms for r in self.results]
        return {
            "label":            self.label,
            "dims":             self.dims,
            "quantization":     self.quantization,
            "n_queries":        len(self.results),
            "recall@1":         np.mean([r.recall_at(1) for r in self.results]),
            "recall@5":         np.mean([r.recall_at(5) for r in self.results]),
            "recall@10":        np.mean([r.recall_at(10) for r in self.results]),
            "mrr":              np.mean([r.mrr for r in self.results]),
            "precision@5":      np.mean([r.precision_at(5) for r in self.results]),
            "latency_p50_ms":   median(latencies),
            "latency_p95_ms":   float(np.percentile(latencies, 95)),
            "storage_mb":       round(self.total_storage_mb, 2),
            "bytes_per_vec":    self.bytes_per_vec,
        }


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def load_eval_set(queries_path: Path, relevance_path: Path) -> tuple[list[dict], dict]:
    queries = json.loads(queries_path.read_text())["queries"]
    relevance = json.loads(relevance_path.read_text())["relevance"]
    return queries, relevance


def run_config(
    label: str,
    dims: int,
    quantization: str,
    num_docs: int,
    queries: list[dict],
    relevance: dict[str, list[str]],
    search_fn: Callable[[str], list[dict]],
    k: int = 10,
) -> ConfigReport:
    """
    search_fn takes a query string and returns ranked hits (list of dicts
    with 'chunk_id'). It handles embedding the query AND the kNN call.
    """
    report = ConfigReport(label=label, dims=dims, quantization=quantization, num_docs=num_docs)

    for q in queries:
        t0 = time.perf_counter()
        hits = search_fn(q["text"])
        dt_ms = (time.perf_counter() - t0) * 1000

        report.results.append(QueryResult(
            query_id=q["id"],
            config_label=label,
            hits=[h["chunk_id"] for h in hits[:k]],
            latency_ms=dt_ms,
            relevant=relevance.get(q["id"], []),
        ))
    return report
