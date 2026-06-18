"""
Download a diverse b-roll corpus from Pexels.

Queries span 8 visual categories so the corpus covers the same range as
eval_queries.json — a homogeneous corpus makes all quantization methods
look equally good, defeating the benchmark.

Checkpointing: completed (term, page) pairs are saved to <out>/.checkpoint.json
so interrupted runs resume without re-downloading anything.

Requirements:
    pip install requests
    PEXELS_API_KEY in .env

Usage:
    python download_pexels.py --out ./clips
    python download_pexels.py --out ./clips --total 5000
    python download_pexels.py --out ./clips --total 5000 --pages 3 --quality hd
    python download_pexels.py --out ./clips --categories animals nature
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Search terms — 8 categories × ~12 terms = ~96 queries.
# At 80 clips/page × 1 page each → up to ~7680 unique clips before dedup.
# Diverse within each category: different subjects, lighting, motion types.
# ---------------------------------------------------------------------------
SEARCH_TERMS: dict[str, list[str]] = {
    "animals": [
        "cats indoors", "dogs running outside", "birds in flight",
        "horses galloping", "fish underwater", "wild deer forest",
        "butterflies flowers", "penguins", "bears wildlife",
        "rabbits grass", "lions savanna", "dolphins ocean",
    ],
    "nature": [
        "aerial forest", "ocean waves crashing", "mountain sunrise",
        "waterfall slow motion", "desert sand dunes", "snow covered forest",
        "tropical beach", "river flowing", "northern lights aurora",
        "storm clouds", "fog morning valley", "autumn leaves falling",
    ],
    "people": [
        "person typing laptop", "crowd walking city", "chef cooking kitchen",
        "child playing outdoors", "elderly people walking", "people dancing",
        "artist painting studio", "musician playing", "friends laughing",
        "woman reading book", "man exercising gym", "couple walking",
    ],
    "urban": [
        "city traffic night", "coffee shop interior", "construction site",
        "subway train", "neon signs rain", "rooftop city view",
        "street food market", "airport terminal", "library interior",
        "skyscraper glass", "parking garage", "bridge city",
    ],
    "sport": [
        "runner trail", "cyclist road", "swimmer pool",
        "basketball court", "soccer match", "tennis player",
        "rock climbing", "surfing waves", "skiing snow",
        "boxing training", "yoga practice", "skateboarding",
    ],
    "food": [
        "cooking close up hands", "restaurant food plating", "farmers market",
        "coffee pour over", "baking bread", "sushi chef",
        "pasta making", "fruit slicing", "cocktail bar",
        "street food vendor", "pizza oven", "chocolate melting",
    ],
    "tech": [
        "server room data center", "hands on keyboard", "smartphone screen",
        "circuit board close up", "robot arm factory", "drone flying",
        "electric car charging", "coding monitor screen",
        "3d printer working", "microscope lab", "satellite dish", "solar panels",
    ],
    "abstract": [
        "bokeh lights blurred", "smoke slow motion", "water surface ripple",
        "fire flames closeup", "paint dropping in water", "sand falling",
        "rain drops window", "soap bubbles", "glitter falling",
        "ink diffusing water", "confetti slow motion", "light rays forest",
    ],
}

QUALITY_SIZES = {
    "sd":  ("sd",  480),
    "hd":  ("hd",  1280),  # recommended — good balance of size and quality
    "fhd": ("hd",  1920),
}

# Pexels free tier: 200 requests/hour → 1 req per 18s to stay safe.
# We use 2s between requests — fast enough, safe enough.
REQUEST_DELAY = 2.0


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def pexels_search(
    query: str, api_key: str, per_page: int = 80, page: int = 1
) -> tuple[list[dict], int]:
    """Return (videos, total_results)."""
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": api_key},
        params={"query": query, "per_page": per_page, "page": page},
        timeout=15,
    )
    if resp.status_code == 429:
        raise RuntimeError("Pexels rate limit hit — wait an hour or use a paid key")
    resp.raise_for_status()
    data = resp.json()
    return data.get("videos", []), data.get("total_results", 0)


def best_file_url(video: dict, quality: str) -> str | None:
    target_label, min_width = QUALITY_SIZES[quality]
    files = video.get("video_files", [])
    candidates = [f for f in files if f.get("quality") == target_label] or files
    candidates.sort(key=lambda f: abs(f.get("width", 0) - min_width))
    return candidates[0]["link"] if candidates else None


def download_file(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return True
    except Exception as e:
        dest.unlink(missing_ok=True)
        tqdm.write(f"    ⚠ {dest.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Checkpoint — tracks completed (category, term, page) tuples and seen IDs
# ---------------------------------------------------------------------------

def load_checkpoint(out: Path) -> dict:
    p = out / ".checkpoint.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"done": [], "seen_ids": []}


def save_checkpoint(out: Path, cp: dict) -> None:
    (out / ".checkpoint.json").write_text(json.dumps(cp))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",        type=Path, default=Path("./clips"))
    ap.add_argument("--total",        type=int,  default=500,
                    help="Total clips to download (default: 500)")
    ap.add_argument("--per-category", type=int,  default=None,
                    help="Max clips per category (default: total // num_categories)")
    ap.add_argument("--per-page",     type=int,  default=80,
                    help="Results per API call, max 80 (default: 80)")
    ap.add_argument("--pages",        type=int,  default=1,
                    help="Pages to fetch per search term (default: 1)")
    ap.add_argument("--quality",      choices=["sd", "hd", "fhd"], default="hd")
    ap.add_argument("--categories",   nargs="*", default=None,
                    help="Subset of categories (default: all 8)")
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--reset",        action="store_true",
                    help="Ignore existing checkpoint and start fresh")
    args = ap.parse_args()

    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        sys.exit("PEXELS_API_KEY not set in .env")

    categories = args.categories or list(SEARCH_TERMS)
    unknown = set(categories) - set(SEARCH_TERMS)
    if unknown:
        sys.exit(f"Unknown categories: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)

    cp = {} if args.reset else load_checkpoint(args.out)
    done_keys: set[str] = set(cp.get("done", []))
    seen_ids:  set[int] = set(cp.get("seen_ids", []))

    target      = args.total
    per_cat_cap = args.per_category or (target // len(categories))

    # Count files already on disk per category so we respect per_cat_cap
    # even when restarting after a partial run.
    cat_counts: dict[str, int] = {
        cat: len(list((args.out / cat).glob("*.mp4")))
        if (args.out / cat).exists() else 0
        for cat in categories
    }

    total_terms = sum(len(SEARCH_TERMS[c]) for c in categories)
    print(f"Categories   : {len(categories)} ({', '.join(categories)})")
    print(f"Per category : {per_cat_cap}")
    print(f"Search terms : {total_terms}")
    print(f"Pages/term   : {args.pages}  ({args.per_page}/page)")
    print(f"Target       : {target} clips at {args.quality} quality")
    print(f"On disk      : { {c: n for c, n in cat_counts.items() if n} }")
    if args.dry_run:
        print("\n[dry-run] would search:")
        for cat in categories:
            for term in SEARCH_TERMS[cat]:
                for pg in range(1, args.pages + 1):
                    key = f"{cat}|{term}|{pg}"
                    status = "✓" if key in done_keys else " "
                    print(f"  [{status}] [{cat}] {term!r}  page {pg}")
        return

    downloaded = 0
    api_calls  = 0

    outer = tqdm(
        [(cat, term, pg)
         for cat in categories
         for term in SEARCH_TERMS[cat]
         for pg in range(1, args.pages + 1)],
        desc="Progress",
        unit="page",
    )

    for cat, term, page in outer:
        if len(seen_ids) >= target:
            tqdm.write(f"Reached target of {target} clips.")
            break

        if cat_counts.get(cat, 0) >= per_cat_cap:
            continue  # this category is full — move on

        key = f"{cat}|{term}|{page}"
        if key in done_keys:
            continue

        outer.set_postfix(cat=cat[:8], term=term[:18], clips=len(seen_ids))

        try:
            videos, total_results = pexels_search(
                term, api_key, per_page=args.per_page, page=page
            )
            api_calls += 1
        except RuntimeError as e:
            tqdm.write(f"  ✗ {e}")
            break
        except Exception as e:
            tqdm.write(f"  ⚠ API error for {term!r} p{page}: {e}")
            time.sleep(REQUEST_DELAY * 3)
            continue

        cat_dir = args.out / cat
        cat_dir.mkdir(exist_ok=True)

        for v in videos:
            if len(seen_ids) >= target:
                break
            vid_id = v["id"]
            if vid_id in seen_ids:
                continue

            url = best_file_url(v, args.quality)
            if not url:
                continue

            dest = cat_dir / f"{vid_id}.mp4"
            if dest.exists():
                seen_ids.add(vid_id)
                continue

            if download_file(url, dest):
                seen_ids.add(vid_id)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                downloaded += 1

        done_keys.add(key)
        save_checkpoint(args.out, {"done": list(done_keys), "seen_ids": list(seen_ids)})

        # Skip delay on last page of a term if no more pages exist
        if total_results > page * args.per_page:
            time.sleep(REQUEST_DELAY)

    print(f"\n✓ Downloaded {downloaded} new clips  ({len(seen_ids)} total in corpus)")
    print(f"  API calls this run : {api_calls}")
    print(f"  Output             : {args.out.resolve()}")
    print(f"  Checkpoint saved   : {args.out / '.checkpoint.json'}")


if __name__ == "__main__":
    main()
