"""
Zero-shot chunk categorization using the shared text-video embedding space.

Each category is a short text description embedded once (anchors are cached
on disk). Chunks are assigned to the closest anchor by cosine similarity at
ingest time, reusing the embedding they already have — no extra API calls.
Chunks that don't clear the similarity floor land in "other".

Customize by writing {"label": "description", ...} to CHUNKS_DIR/.categories.json.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("omnishot")

FALLBACK = "other"
MIN_SIM = float(os.environ.get("CATEGORY_MIN_SIM", "0.15"))

DEFAULT_CATEGORIES: dict[str, str] = {
    "nature": "nature and landscape footage: forests, mountains, sky, fields, natural scenery",
    "water": "water footage: oceans, waves, rivers, waterfalls, rain, underwater shots",
    "people": "footage of people: portraits, crowds, faces, hands, people working or talking",
    "urban": "urban city footage: streets, buildings, skylines, traffic, city life at day or night",
    "animals": "animal and wildlife footage: pets, birds, marine life, insects, wild animals",
    "food": "food and cooking footage: meals, ingredients, kitchens, restaurants, drinks",
    "technology": "technology footage: computers, screens, devices, servers, robots, machinery",
    "sports": "sports and action footage: athletes, exercise, running, competition, outdoor activity",
    "transport": "transportation footage: cars, trains, planes, boats, roads, travel",
    "abstract": "abstract footage: textures, patterns, light effects, smoke, slow motion details, backgrounds",
}


def load_categories(chunks_dir: Path) -> dict[str, str]:
    custom = chunks_dir / ".categories.json"
    if custom.exists():
        try:
            data = json.loads(custom.read_text())
            if isinstance(data, dict) and data:
                return {str(k): str(v) for k, v in data.items()}
        except Exception as e:
            logger.warning("ignoring invalid %s: %s", custom, e)
    return dict(DEFAULT_CATEGORIES)


class CategoryIndex:
    """Holds one anchor vector per category label."""

    def __init__(self, anchors: dict[str, list[float]], min_sim: float = MIN_SIM):
        self.anchors = anchors
        self.min_sim = min_sim

    @property
    def labels(self) -> list[str]:
        return list(self.anchors)

    def classify(self, embedding: list[float]) -> tuple[str, float]:
        """Return (label, similarity) of the best anchor, or (FALLBACK, best)."""
        best_label: str | None = None
        best_sim = -1.0
        for label, anchor in self.anchors.items():
            sim = sum(a * b for a, b in zip(anchor, embedding))
            if sim > best_sim:
                best_label, best_sim = label, sim
        if best_label is None or best_sim < self.min_sim:
            return FALLBACK, best_sim
        return best_label, best_sim


def build_category_index(jina, cfg, chunks_dir: Path) -> CategoryIndex:
    """Embed category descriptions (or load them from the on-disk cache)."""
    cats = load_categories(chunks_dir)
    cache_path = chunks_dir / ".category_anchors.json"
    key = hashlib.sha256(
        json.dumps([cfg.model, cfg.dimensions, cats], sort_keys=True).encode()
    ).hexdigest()

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("key") == key:
                return CategoryIndex(cached["anchors"])
        except Exception:
            pass

    labels = list(cats)
    vecs = jina.embed([cats[label] for label in labels], task="retrieval.query", config=cfg)
    anchors = dict(zip(labels, vecs))
    chunks_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"key": key, "anchors": anchors}))
    logger.info("embedded %d category anchors", len(anchors))
    return CategoryIndex(anchors)
