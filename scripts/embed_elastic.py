"""
Elastic inference service client for the jina-omni embedding endpoint.

Calls the Elasticsearch inference API rather than the Jina platform directly.
The inference endpoint is configured once in your cluster; this client just
hits it. Model, dimensions, and quantization are set on the endpoint — not
per-request.

Required env vars:
    ES_URL            Elasticsearch cluster URL
    ES_API_KEY        Cluster API key
    ES_INFERENCE_ID   Inference endpoint ID (e.g. "jina-omni-embeddings")

Jina task  →  Elastic input_type
  retrieval.passage  →  ingest
  retrieval.query    →  search
  text-matching      →  (omitted — uses endpoint default)
"""

from __future__ import annotations
import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from index_elastic import es_client

Modality = Literal["text", "image", "audio", "video"]
Task = Literal["retrieval.query", "retrieval.passage", "text-matching"]
EmbeddingType = Literal["float", "binary", "base64"]

_TASK_TO_INPUT_TYPE: dict[str, str] = {
    "retrieval.passage": "ingest",
    "retrieval.query":   "search",
}


@dataclass
class EmbedConfig:
    """Kept for interface compatibility. model/dimensions live on the endpoint."""
    model: str = "jina-omni"
    dimensions: int = 1024
    embedding_type: EmbeddingType = "float"
    normalized: bool = True

    @property
    def label(self) -> str:
        return f"{self.model.split('-')[-1]}_{self.dimensions}d_{self.embedding_type}"


class ElasticInferenceClient:
    def __init__(self, inference_id: str | None = None):
        self.inference_id = inference_id or os.environ.get("ES_INFERENCE_ID")
        if not self.inference_id:
            raise RuntimeError("Set ES_INFERENCE_ID env var or pass inference_id.")
        self.es = es_client()

    def embed(
        self,
        inputs: Sequence[dict | str],
        task: Task = "retrieval.passage",
        config: EmbedConfig | None = None,
        retries: int = 3,
    ) -> list[list[float]]:
        """
        inputs: plain strings or modality dicts {"text"|"image"|"audio"|"video": ...}
        Returns one embedding vector per input, in input order.
        """
        serialized = [self._serialize(i) for i in inputs]
        body: dict = {"input": serialized}
        input_type = _TASK_TO_INPUT_TYPE.get(task)
        if input_type:
            body["task_settings"] = {"input_type": input_type}

        resp = self.es.inference.inference(
            inference_id=self.inference_id,
            body=body,
        )
        embeddings = resp.get("text_embedding") or resp.get("sparse_embedding") or []
        return [e["embedding"] for e in sorted(embeddings, key=lambda x: x["index"])]

    @staticmethod
    def _serialize(item: dict | str) -> str:
        """Flatten modality dict to the bare value (base64 string or text)."""
        if isinstance(item, str):
            return item
        for key in ("text", "video", "image", "audio"):
            if key in item:
                return item[key]
        raise ValueError(f"Unrecognized input format: {item!r}")


# ---------------------------------------------------------------------------
# Modality helpers — identical interface to embed_jina for drop-in use
# ---------------------------------------------------------------------------

def text_input(t: str) -> dict:
    return {"text": t}


def image_input(path: str | Path) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"image": data}


def video_input(path_or_url: str | Path) -> dict:
    """URL is preferred for large clips; local files are base64-encoded."""
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        return {"video": s}
    data = base64.b64encode(Path(s).read_bytes()).decode("ascii")
    return {"video": data}


def audio_input(path_or_url: str | Path) -> dict:
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        return {"audio": s}
    data = base64.b64encode(Path(s).read_bytes()).decode("ascii")
    return {"audio": data}
