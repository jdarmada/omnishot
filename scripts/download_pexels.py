"""
Download a small b-roll corpus from Pexels.

Lightweight version of the omnishot downloader: 6 categories, fewer terms.
Checkpointing resumes interrupted runs.

Requirements:
    pip install requests python-dotenv tqdm
    PEXELS_API_KEY in .env

Usage:
    python scripts/download_pexels.py --out ./clips --total 50
    python scripts/download_pexels.py --out ./clips --total 100 --categories nature urban
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

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SEARCH_TERMS: dict[str, list[str]] = {
    "animals": [
        "cats indoors", "dogs running outside", "birds in flight",
        "horses galloping", "fish underwater", "wild deer forest",
    ],
    "nature": [
        "aerial forest", "ocean waves crashing", "mountain sunrise",
        "waterfall slow motion", "desert sand dunes", "tropical beach",
    ],
    "people": [
        "person typing laptop", "crowd walking city", "chef cooking kitchen",
        "people dancing", "musician playing", "friends laughing",
    ],
    "urban": [
        "city traffic night", "coffee shop interior", "subway train",
        "neon signs rain", "street food market", "rooftop city view",
    ],
    "sport": [
        "runner trail", "cyclist road", "swimmer pool",
        "surfing waves", "yoga practice", "skateboarding",
    ],
    "abstract": [
        "bokeh lights blurred", "smoke slow motion", "water surface ripple",
        "fire flames closeup", "ink diffusing water", "light rays forest",
    ],
}

QUALITY_SIZES = {
    "sd": ("sd", 480),
    "hd": ("hd", 1280),
    "fhd": ("hd", 1920),
}
REQUEST_DELAY = 2.0


def pexels_search(
    query: str, api_key: str, per_page: int = 40, page: int = 1
) -> tuple[list[dict], int]:
    r = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": api_key},
        params={"query": query, "per_page": per_page, "page": page},
        timeout=30,
    )
    if r.status_code == 429:
        raise RuntimeError("Pexels rate limit hit — wait and retry")
    r.raise_for_status()
    data = r.json()
    return data.get("videos", []), data.get("total_results", 0)


def best_file_url(video: dict, quality: str) -> str | None:
    want_q, want_w = QUALITY_SIZES[quality]
    files = video.get("video_files") or []
    ranked = sorted(
        files,
        key=lambda f: (
            0 if f.get("quality") == want_q else 1,
            abs((f.get("width") or 0) - want_w),
        ),
    )
    return ranked[0].get("link") if ranked else None


def download_file(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        tqdm.write(f"  ⚠ download failed: {e}")
        dest.unlink(missing_ok=True)
        return False


def load_checkpoint(out: Path) -> dict:
    p = out / ".checkpoint.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_checkpoint(out: Path, cp: dict) -> None:
    (out / ".checkpoint.json").write_text(json.dumps(cp))


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Pexels stock footage")
    ap.add_argument("--out", type=Path, default=Path("./clips"))
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--per-category", type=int, default=None)
    ap.add_argument("--per-page", type=int, default=40)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--quality", choices=["sd", "hd", "fhd"], default="hd")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        sys.exit("PEXELS_API_KEY not set in .env")

    categories = args.categories or list(SEARCH_TERMS)
    unknown = set(categories) - set(SEARCH_TERMS)
    if unknown:
        sys.exit(f"Unknown categories: {unknown}. Valid: {sorted(SEARCH_TERMS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    cp = {} if args.reset else load_checkpoint(args.out)
    done_keys: set[str] = set(cp.get("done", []))
    seen_ids: set[int] = set(cp.get("seen_ids", []))
    per_cat_cap = args.per_category or max(1, args.total // len(categories))

    cat_counts: dict[str, int] = {
        cat: len(list((args.out / cat).glob("*.mp4")))
        if (args.out / cat).exists()
        else 0
        for cat in categories
    }

    print(f"Categories : {', '.join(categories)}")
    print(f"Target     : {args.total} clips @ {args.quality}")
    if args.dry_run:
        for cat in categories:
            for term in SEARCH_TERMS[cat]:
                print(f"  [{cat}] {term}")
        return

    downloaded = 0
    pages = [
        (cat, term, pg)
        for cat in categories
        for term in SEARCH_TERMS[cat]
        for pg in range(1, args.pages + 1)
    ]

    for cat, term, page in tqdm(pages, desc="Progress", unit="page"):
        if len(seen_ids) >= args.total:
            break
        if cat_counts.get(cat, 0) >= per_cat_cap:
            continue
        key = f"{cat}|{term}|{page}"
        if key in done_keys:
            continue

        try:
            videos, total_results = pexels_search(
                term, api_key, per_page=args.per_page, page=page
            )
        except Exception as e:
            tqdm.write(f"  ⚠ {e}")
            time.sleep(REQUEST_DELAY * 3)
            continue

        cat_dir = args.out / cat
        cat_dir.mkdir(exist_ok=True)

        for v in videos:
            if len(seen_ids) >= args.total or cat_counts.get(cat, 0) >= per_cat_cap:
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
        if total_results > page * args.per_page:
            time.sleep(REQUEST_DELAY)

    print(f"\n✓ Downloaded {downloaded} new clips ({len(seen_ids)} total)")
    print(f"  Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
