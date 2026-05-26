"""
Elasticsearch index management for the b-roll corpus.

We use semantic_text with the Jina v5-omni inference endpoint so that
Elasticsearch handles embedding at both index and query time. This means:

  - At ingest: pass each keyframe as a data URI in the `content` field.
    ES calls the inference endpoint with the correct typed image format
    automatically — no manual embedding calls needed.

  - At search: pass the text query to a `semantic` query.
    ES embeds it and runs the similarity search automatically.

Index naming convention: `broll-{strategy}-semantic`
  e.g.  broll-scene-semantic
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
    content: str            # data URI of keyframe — ES embeds this via semantic_text


def es_client() -> Elasticsearch:
    return Elasticsearch(
        os.environ["ES_URL"],
        api_key=os.environ["ES_API_KEY"],
        request_timeout=120,
    )


def index_name(strategy: str) -> str:
    return f"broll-{strategy}-semantic"


def create_index(es: Elasticsearch, name: str, inference_id: str) -> None:
    """Create index with a semantic_text field for automatic multimodal embedding."""
    if es.indices.exists(index=name):
        return

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
            "content": {
                "type": "semantic_text",
                "inference_id": inference_id,
            },
        }
    }
    es.indices.create(index=name, mappings=mappings)


def bulk_index(es: Elasticsearch, name: str, docs: Iterable[ChunkDoc]) -> int:
    """Index documents one at a time.

    With semantic_text, ES calls the inference endpoint inline for each document.
    Batching all 80+ docs in one request reliably exceeds any reasonable timeout,
    so we index individually with a generous per-doc timeout instead.
    """
    success = 0
    errors = 0
    for d in docs:
        try:
            es.index(index=name, id=d.chunk_id, document=asdict(d), request_timeout=60)
            success += 1
        except Exception as e:
            print(f"  ⚠ indexing error for {d.chunk_id}: {e}")
            errors += 1
    if errors:
        print(f"⚠ {errors} indexing errors")
    return success


def semantic_search(
    es: Elasticsearch,
    index: str,
    query: str,
    k: int = 10,
    filter_clauses: list[dict] | None = None,
) -> list[dict]:
    """Semantic search via the semantic_text field. ES embeds the query automatically."""
    semantic_clause = {"semantic": {"field": "content", "query": query}}

    if filter_clauses:
        query_body = {
            "bool": {
                "must": [semantic_clause],
                "filter": filter_clauses,
            }
        }
    else:
        query_body = semantic_clause

    res = es.search(
        index=index,
        query=query_body,
        size=k,
        source_excludes=["content"],  # don't return the data URI in results
    )
    return [
        {**hit["_source"], "_score": hit["_score"]}
        for hit in res["hits"]["hits"]
    ]


def hybrid_search(
    es: Elasticsearch,
    index: str,
    query_text: str,
    k: int = 10,
    filter_clauses: list[dict] | None = None,
) -> list[dict]:
    """RRF fusion of BM25 over transcript + semantic search over visual embedding."""
    must = filter_clauses or []

    bm25_retriever = {
        "standard": {
            "query": {
                "bool": {
                    "must": [{"match": {"transcript": query_text}}],
                    **({"filter": must} if must else {}),
                }
            }
        }
    }
    semantic_retriever = {
        "standard": {
            "query": {
                "bool": {
                    "must": [{"semantic": {"field": "content", "query": query_text}}],
                    **({"filter": must} if must else {}),
                }
            }
        }
    }

    res = es.search(
        index=index,
        size=k,
        source_excludes=["content"],
        retriever={
            "rrf": {
                "retrievers": [bm25_retriever, semantic_retriever],
                "rank_window_size": 50,
                "rank_constant": 60,
            }
        },
    )
    return [
        {**hit["_source"], "_score": hit["_score"]}
        for hit in res["hits"]["hits"]
    ]
