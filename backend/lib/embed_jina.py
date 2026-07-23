"""
Jina v5-omni embedding client.

Text, image, and video share one embedding space. Use task=retrieval.query
for search queries and task=retrieval.passage for indexed chunks.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import requests

JINA_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_MODEL = "jina-embeddings-v5-omni-small"

Task = Literal["retrieval.query", "retrieval.passage", "text-matching"]
EmbeddingType = Literal["float", "binary", "base64"]


@dataclass
class EmbedConfig:
    model: str = DEFAULT_MODEL
    dimensions: int = 1024
    embedding_type: EmbeddingType = "float"
    normalized: bool = True


class JinaClient:
    def __init__(self, api_key: str | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "JINA_API_KEY is not set. Copy .env.example to .env and add "
                "your key (free tier at https://jina.ai)."
            )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def embed(
        self,
        inputs: Sequence[dict | str],
        task: Task,
        config: EmbedConfig | None = None,
        retries: int = 3,
    ) -> list[list[float]]:
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
                if (
                    e.response is not None
                    and e.response.status_code in (429, 500, 502, 503)
                    and attempt < retries - 1
                ):
                    time.sleep(2**attempt)
                    continue
                raise

    @staticmethod
    def _normalize(item: dict | str) -> dict:
        if isinstance(item, str):
            return {"text": item}
        return item
