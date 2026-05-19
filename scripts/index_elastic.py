"""
Elasticsearch index management for the b-roll corpus.

We use dense_vector with cosine similarity and Better Binary Quantization
(BBQ) as the index type for the compressed configurations. Metadata fields
power the hybrid-search story ("outdoor shot, under 15s, uploaded this week").

Index naming convention: `broll-{strategy}-{dims}-{quantization}`
  e.g.  broll-scene-1024-float, broll-scene-256-bbq

Each config gets its own index so we can apples-to-apples benchmark them.
"""

from __future__ import annotations
import os
from dataclasses import asdict, dataclass
from typing import Iterable

from elasticsearch import Elasticsearch, helpers


@dataclass
class ChunkDoc:
    """One indexable record."""
    chunk_id: str
    clip_id: str
    path: str
    start_sec: float
    end_sec: float
    duration: float
    strategy: str
    # Metadata for hybrid search
    uploaded_at: str        # ISO date — drives "uploaded this week"
    uploader: str
    tags: list[str]
    transcript: str | None  # if you've run STT — feeds BM25
    embedding: list[float]


def es_client() -> Elasticsearch:
    return Elasticsearch(
        os.environ["ES_URL"],
        api_key=os.environ["ES_API_KEY"],
        request_timeout=60,
    )


def index_name(strategy: str, dims: int, quant: str) -> str:
    return f"broll-{strategy}-{dims}-{quant}"


def create_index(es: Elasticsearch, name: str, dims: int, quantization: str = "float") -> None:
    """
    quantization options:
      - 'float'  : standard HNSW with float32. Reference quality.
      - 'int8'   : int8_hnsw — ~4x storage savings, negligible recall loss
      - 'bbq'    : bbq_hnsw — Better Binary Quantization, ~32x savings
    """
    if es.indices.exists(index=name):
        return

    index_type = {
        "float": "hnsw",
        "int8":  "int8_hnsw",
        "bbq":   "bbq_hnsw",
    }[quantization]

    mappings = {
        "properties": {
            "chunk_id":    {"type": "keyword"},
            "clip_id":     {"type": "keyword"},
            "path":        {"type": "keyword", "index": False},
            "start_sec":   {"type": "float"},
            "end_sec":     {"type": "float"},
            "duration":    {"type": "float"},
            "strategy":    {"type": "keyword"},
            "uploaded_at": {"type": "date"},
            "uploader":    {"type": "keyword"},
            "tags":        {"type": "keyword"},
            "transcript":  {"type": "text", "analyzer": "english"},
            "embedding":   {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": index_type},
            },
        }
    }
    es.indices.create(index=name, mappings=mappings)


def bulk_index(es: Elasticsearch, name: str, docs: Iterable[ChunkDoc]) -> int:
    """Bulk-index. Returns number of docs successfully indexed."""
    def actions():
        for d in docs:
            yield {
                "_op_type": "index",
                "_index": name,
                "_id": d.chunk_id,
                "_source": asdict(d),
            }
    success, errors = helpers.bulk(es, actions(), raise_on_error=False)
    if errors:
        print(f"⚠ {len(errors)} indexing errors (first: {errors[0]})")
    return success


def knn_search(
    es: Elasticsearch,
    index: str,
    query_vector: list[float],
    k: int = 10,
    num_candidates: int = 100,
    filter_clauses: list[dict] | None = None,
) -> list[dict]:
    """Pure kNN with optional pre-filter."""
    knn = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
    }
    if filter_clauses:
        knn["filter"] = {"bool": {"must": filter_clauses}}

    res = es.search(index=index, knn=knn, size=k, _source_excludes=["embedding"])
    return [
        {**hit["_source"], "_score": hit["_score"]}
        for hit in res["hits"]["hits"]
    ]


def hybrid_search(
    es: Elasticsearch,
    index: str,
    query_text: str,
    query_vector: list[float],
    k: int = 10,
    filter_clauses: list[dict] | None = None,
) -> list[dict]:
    """RRF fusion of BM25 over transcript + kNN over visual embedding,
    with optional metadata filter applied to both."""
    must = filter_clauses or []
    res = es.search(
        index=index,
        size=k,
        _source_excludes=["embedding"],
        retriever={
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"bool": {
                        "must": [{"match": {"transcript": query_text}}] + must
                    }}}},
                    {"knn": {
                        "field": "embedding",
                        "query_vector": query_vector,
                        "k": k * 3,
                        "num_candidates": 100,
                        **({"filter": {"bool": {"must": must}}} if must else {}),
                    }},
                ],
                "rank_window_size": 50,
                "rank_constant": 60,
            }
        },
    )
    return [
        {**hit["_source"], "_score": hit["_score"]}
        for hit in res["hits"]["hits"]
    ]
