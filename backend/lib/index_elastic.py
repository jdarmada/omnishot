"""
Elasticsearch helpers for the demo index (float32 HNSW, 1024-d cosine).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger("omnishot")


@dataclass
class ChunkDoc:
    chunk_id: str
    clip_id: str
    path: str
    start_sec: float
    end_sec: float
    duration: float
    strategy: str
    uploaded_at: str
    uploader: str
    tags: list
    transcript: str | None
    embedding: list


def es_client() -> Elasticsearch:
    url = os.environ.get("ES_URL")
    if not url:
        raise RuntimeError(
            "ES_URL is not set. Copy .env.example to .env and set ES_URL "
            "(http://localhost:9200 for local Docker Elasticsearch)."
        )
    kwargs: dict = {"request_timeout": 120}
    api_key = os.environ.get("ES_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return Elasticsearch(url, **kwargs)


def create_index(
    es: Elasticsearch,
    name: str,
    dims: int = 1024,
) -> None:
    if es.indices.exists(index=name):
        return

    mappings = {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "clip_id": {"type": "keyword"},
            "path": {"type": "keyword", "index": False},
            "start_sec": {"type": "float"},
            "end_sec": {"type": "float"},
            "duration": {"type": "float"},
            "strategy": {"type": "keyword"},
            "uploaded_at": {"type": "date"},
            "uploader": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "transcript": {"type": "text", "analyzer": "english"},
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw"},
            },
        }
    }
    es.indices.create(index=name, mappings=mappings)
    logger.info("Created index '%s' (%d-d cosine hnsw)", name, dims)


def bulk_index(es: Elasticsearch, name: str, docs: Iterable[ChunkDoc]) -> int:
    actions = (
        {"_index": name, "_id": d.chunk_id, "_source": asdict(d)} for d in docs
    )
    success, errors = helpers.bulk(es, actions, raise_on_error=False)
    if errors:
        logger.warning("%d bulk indexing errors", len(errors))
        for e in errors[:3]:
            logger.warning("  %s", e)
    return success


def knn_search(
    es: Elasticsearch,
    index: str,
    query_vector: list[float],
    k: int = 10,
    num_candidates: int = 100,
) -> list[dict]:
    res = es.search(
        index=index,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates,
        },
        size=k,
        source_excludes=["embedding"],
    )
    return [
        {**hit["_source"], "_score": hit["_score"]} for hit in res["hits"]["hits"]
    ]
