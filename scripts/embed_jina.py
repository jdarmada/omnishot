"""
Jina v5-omni embedding client.

Single endpoint, four modalities. Text/image/audio/video all return embeddings
in the same 1024-d (small) or 768-d (nano) space. Critical knobs:
- `task`: retrieval.query for text queries, retrieval.passage for documents
- `dimensions`: Matryoshka truncation (32-1024 for small, 32-768 for nano)
- `embedding_type`: 'float' for baseline, 'binary' for BBQ-equivalent
"""

from __future__ import annotations
import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import requests

JINA_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_MODEL = "jina-embeddings-v5-omni-small"

Modality = Literal["text", "image", "audio", "video"]
Task = Literal["retrieval.query", "retrieval.passage", "text-matching"]
EmbeddingType = Literal["float", "binary", "base64"]


@dataclass
class EmbedConfig:
    """One config = one row in the benchmark table."""
    model: str = DEFAULT_MODEL
    dimensions: int = 1024            # Matryoshka truncation
    embedding_type: EmbeddingType = "float"
    normalized: bool = True

    @property
    def label(self) -> str:
        return f"{self.model.split('-')[-1]}_{self.dimensions}d_{self.embedding_type}"


class JinaClient:
    def __init__(self, api_key: str | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set JINA_API_KEY env var or pass api_key.")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def embed(
        self,
        inputs: Sequence[dict | str],
        task: Task,
        config: EmbedConfig | None = None,
        retries: int = 3,
    ) -> list[list[float]]:
        """
        Inputs are either plain strings (text) or dicts:
            {"text": "..."} or {"image": "<base64>"} or {"video": "<url>"} ...
        Returns one embedding per input.
        """
        cfg = config or EmbedConfig()
        payload = {
            "model": cfg.model,
            "task": task,
            "dimensions": cfg.dimensions,
            "embedding_type": cfg.embedding_type,
            "normalized": cfg.normalized,
            "input": [self._normalize(i) for i in inputs],
        }
        for attempt in range(retries):
            try:
                r = self.session.post(JINA_URL, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()["data"]
                return [d["embedding"] for d in data]
            except requests.HTTPError as e:
                if e.response.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

    @staticmethod
    def _normalize(item: dict | str) -> dict:
        if isinstance(item, str):
            return {"text": item}
        return item


# -----------------------------------------------------------------------------
# Modality helpers — turn a local file path into the right input dict
# -----------------------------------------------------------------------------

def text_input(t: str) -> dict:
    return {"text": t}


def image_input(path: str | Path) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"image": data}


def video_input(path_or_url: str | Path) -> dict:
    """v5-omni accepts video as URL or base64. URL is much cheaper for >5MB clips."""
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
