"""
Elastic inference service client for the jina-omni embedding endpoint.

Calls the Elasticsearch inference API rather than the Jina platform directly.
The inference endpoint is configured once in your cluster; this client just
hits it. Model, dimensions, and quantization are set on the endpoint — not
per-request.

Video inputs are not sent as raw video — the EIS gateway has a hard ~30s
timeout that video processing reliably exceeds. Instead, video_input() extracts
a representative keyframe at the midpoint using ffmpeg and embeds it as an
image. Image payloads are ~200KB and process in well under a second.

Required env vars:
    ES_URL            Elasticsearch cluster URL
    ES_API_KEY        Cluster API key
    ES_INFERENCE_ID   Inference endpoint ID (e.g. ".jina-embeddings-v5-omni-small")

Jina task  →  Elastic input_type
  retrieval.passage  →  ingest
  retrieval.query    →  search
  text-matching      →  (omitted — uses endpoint default)
"""

from __future__ import annotations
import base64
import os
import subprocess
import tempfile
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
        # EIS returns {"embeddings": [{"embedding": [...]}, ...]} in order
        embeddings = resp.get("embeddings") or resp.get("text_embedding") or []
        return [e["embedding"] for e in embeddings]

    @staticmethod
    def _serialize(item: dict | str) -> str:
        """Flatten modality dict to the bare value (base64 string or text)."""
        if isinstance(item, str):
            return item
        if "image" in item:
            val = item["image"]
            # Wrap raw base64 in a data URI so the inference endpoint treats
            # it as an image, not as a text string to embed.
            if not val.startswith(("http://", "https://", "data:")):
                return f"data:image/jpeg;base64,{val}"
            return val
        for key in ("text", "video", "audio"):
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
    """Extract mid-clip keyframe and return as an image input.

    The EIS gateway times out on raw video payloads. A single JPEG frame
    captures the visual content and stays well within size and latency limits.
    """
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        # Remote URL: pass through as image URL (assumes Jina can fetch it)
        return {"image": s}
    path = Path(s)
    # Get duration via ffprobe, seek to midpoint
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip() or "0")
    seek = duration / 2.0

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seek), "-i", str(path),
             "-frames:v", "1", "-q:v", "3", tmp_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        data = base64.b64encode(Path(tmp_path).read_bytes()).decode("ascii")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"image": data}


def audio_input(path_or_url: str | Path) -> dict:
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        return {"audio": s}
    data = base64.b64encode(Path(s).read_bytes()).decode("ascii")
    return {"audio": data}
