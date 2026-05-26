"""
Embedding quality diagnostic for the b-roll corpus.

Pulls every chunk's semantic scores against a set of anchor concepts,
then visualises two things:

  1. Heatmap  — chunk × concept score matrix (what does each clip "look like"?)
  2. UMAP     — chunks projected to 2-D by score matrix (do similar clips cluster?)

Because semantic_text doesn't expose raw vectors in _source, we probe the space
indirectly: embed each anchor text via the inference API, run a semantic search
to get its score against every doc, and use those N-dimensional score vectors
as a proxy for position in the embedding space.

Usage:
    python scripts/diagnose_embeddings.py
    python scripts/diagnose_embeddings.py --anchors "cats,dogs,ocean,fire,city"
    python scripts/diagnose_embeddings.py --out ./reports
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from index_elastic import es_client, index_name

DEFAULT_ANCHORS = [
    "cat", "dog", "bird", "wildlife animal",
    "ocean waves", "beach sunset", "underwater",
    "city skyline at night", "street traffic",
    "person running", "crowd of people",
    "forest trees nature", "mountains snow",
    "fire flames", "rain storm",
    "food cooking kitchen",
    "drone aerial view",
]


def score_all_docs(es, index: str, query: str, size: int = 200) -> dict[str, float]:
    """Return {chunk_id: score} for every doc in the index for this query."""
    res = es.search(
        index=index,
        query={"semantic": {"field": "content", "query": query}},
        size=size,
        source_includes=["chunk_id", "clip_id"],
    )
    return {
        hit["_source"]["chunk_id"]: hit["_score"]
        for hit in res["hits"]["hits"]
    }


def build_score_matrix(es, index: str, anchors: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a [n_docs × n_anchors] score matrix."""
    print(f"Probing {len(anchors)} anchor concepts against index '{index}'...")

    # Get full doc list from the first query
    first = score_all_docs(es, index, anchors[0])
    chunk_ids = sorted(first.keys())

    matrix = np.zeros((len(chunk_ids), len(anchors)), dtype=float)
    matrix[:, 0] = [first.get(c, 0.0) for c in chunk_ids]

    for j, anchor in enumerate(anchors[1:], start=1):
        scores = score_all_docs(es, index, anchor)
        matrix[:, j] = [scores.get(c, 0.0) for c in chunk_ids]
        print(f"  [{j+1}/{len(anchors)}] {anchor}")

    return matrix, chunk_ids, anchors


def short_label(chunk_id: str) -> str:
    """Turn a long chunk_id into a readable short label."""
    parts = chunk_id.split("__")
    clip = parts[0]
    # Keep just the numeric prefix and last segment
    clip_short = clip.split("-")[0].split("_")[0]
    idx = parts[-1] if len(parts) > 1 else ""
    return f"{clip_short}_{idx}" if idx else clip_short


def plot_heatmap(matrix: np.ndarray, chunk_ids: list[str],
                 anchors: list[str], out_path: Path) -> None:
    labels = [short_label(c) for c in chunk_ids]
    fig, ax = plt.subplots(figsize=(max(14, len(anchors) * 0.8),
                                    max(10, len(chunk_ids) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(anchors)))
    ax.set_xticklabels(anchors, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(chunk_ids)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Chunk × Concept score heatmap\n"
                 "(brighter = higher semantic similarity)", pad=12)
    plt.colorbar(im, ax=ax, shrink=0.6, label="semantic score")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → heatmap saved: {out_path}")


def plot_umap(matrix: np.ndarray, chunk_ids: list[str], out_path: Path) -> None:
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(10, len(chunk_ids) - 1),
                            min_dist=0.3, random_state=42)
        embedding = reducer.fit_transform(matrix)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        embedding = PCA(n_components=2, random_state=42).fit_transform(matrix)
        method = "PCA"

    # Colour by clip (group chunks from the same source clip)
    clip_ids = [c.split("__")[0] for c in chunk_ids]
    unique_clips = sorted(set(clip_ids))
    cmap = cm.get_cmap("tab20", len(unique_clips))
    colours = [cmap(unique_clips.index(c)) for c in clip_ids]

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                         c=colours, s=60, alpha=0.8, linewidths=0.4,
                         edgecolors="white")

    # Label each point with a short id
    for i, (x, y) in enumerate(embedding):
        ax.annotate(short_label(chunk_ids[i]), (x, y),
                    fontsize=5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_title(f"{method} projection of chunk embeddings\n"
                 "(colour = source clip; proximity = semantic similarity)")
    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")

    # Legend: one entry per unique clip
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(i), markersize=7, label=short_label(c))
        for i, c in enumerate(unique_clips)
    ]
    ax.legend(handles=legend_elements, fontsize=5,
              loc="upper left", ncol=3, framealpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {method} plot saved: {out_path}")


def top_matches(matrix: np.ndarray, chunk_ids: list[str],
                anchors: list[str], topk: int = 3) -> None:
    """Print the top-k chunks for each anchor concept."""
    print("\n── Top matches per anchor ──────────────────────────────")
    for j, anchor in enumerate(anchors):
        col = matrix[:, j]
        top_idx = np.argsort(col)[::-1][:topk]
        hits = [(chunk_ids[i], col[i]) for i in top_idx]
        print(f"\n  '{anchor}'")
        for cid, score in hits:
            print(f"    {score:.4f}  {cid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.environ.get("BROLL_INDEX", index_name("scene")))
    ap.add_argument("--anchors", default=None,
                    help="comma-separated list of concept strings to probe")
    ap.add_argument("--out", type=Path, default=Path("./reports"),
                    help="directory to write plot images")
    ap.add_argument("--topk", type=int, default=3,
                    help="top-k chunks to print per anchor")
    args = ap.parse_args()

    anchors = [a.strip() for a in args.anchors.split(",")] \
        if args.anchors else DEFAULT_ANCHORS

    args.out.mkdir(parents=True, exist_ok=True)

    es = es_client()
    matrix, chunk_ids, anchors = build_score_matrix(es, args.index, anchors)

    print(f"\nMatrix shape: {matrix.shape}  (chunks × anchors)")
    print(f"Score range: {matrix.min():.4f} – {matrix.max():.4f}")
    print(f"Score std:   {matrix.std():.4f}  (higher = more discriminative)")

    top_matches(matrix, chunk_ids, anchors, args.topk)

    print("\nGenerating plots...")
    plot_heatmap(matrix, chunk_ids, anchors, args.out / "heatmap.png")
    plot_umap(matrix, chunk_ids, args.out / "umap.png")
    print("\nDone. Open the images in ./reports/ to inspect embedding quality.")


if __name__ == "__main__":
    main()
