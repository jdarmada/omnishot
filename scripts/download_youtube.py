"""
Download a diverse b-roll corpus from YouTube using yt-dlp.

Targets longer-form content (3–30 min) that produces multiple chunks per
download — documentaries, educational footage, and nature films that Pexels
doesn't carry.  Complements download_pexels.py for benchmark diversity.

Checkpointing: completed (category, query, index) tuples are written to
<out>/.yt_checkpoint.json so interrupted runs resume cleanly.

Requirements:
    pip install yt-dlp

Usage:
    python download_youtube.py --out ./clips
    python download_youtube.py --out ./clips --total 100 --per-category 10
    python download_youtube.py --out ./clips --categories medical_science weather
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Search terms — 12 categories × 4-6 queries.
# Queries lean toward documentary and educational content (more scenes/clip).
# ---------------------------------------------------------------------------
SEARCH_TERMS: dict[str, list[str]] = {
    "animals": [
        "wildlife documentary Africa animals",
        "ocean marine life documentary footage",
        "bird behavior nature documentary",
        "predator prey interaction wildlife film",
        "insect macro nature documentary",
    ],
    "nature": [
        "nature timelapse documentary landscape",
        "forest ecosystem documentary footage",
        "volcano geology documentary film",
        "arctic wilderness polar documentary",
        "rainforest biodiversity documentary",
    ],
    "people": [
        "street life urban documentary photography",
        "cultural traditions people documentary",
        "human stories documentary portrait",
        "community everyday life documentary",
    ],
    "urban": [
        "city documentary urban exploration footage",
        "metropolitan architecture tour documentary",
        "city life street documentary film",
        "downtown urban scenery documentary",
    ],
    "sport": [
        "extreme sports documentary footage",
        "athletic training endurance documentary",
        "outdoor adventure sport film",
        "competitive sport training behind scenes",
    ],
    "food": [
        "culinary arts cooking documentary",
        "farm to table food production documentary",
        "traditional cuisine preparation documentary",
        "artisan food craft documentary footage",
    ],
    "tech": [
        "technology innovation documentary footage",
        "engineering manufacturing documentary",
        "robotics automation documentary film",
        "space technology science documentary",
    ],
    "abstract": [
        "macro photography slow motion nature",
        "fluid dynamics slow motion camera footage",
        "abstract light art installation footage",
        "slow motion water drops macro closeup",
    ],
    "medical_science": [
        "medical procedure educational footage surgery",
        "laboratory science experiment documentary",
        "biology anatomy educational footage",
        "medical imaging diagnostic technology",
        "pharmaceutical research laboratory footage",
    ],
    "architecture": [
        "architecture documentary building design",
        "interior design spaces documentary",
        "historic architecture restoration documentary",
        "modern architecture tour building film",
        "sustainable architecture green building documentary",
    ],
    "weather": [
        "storm chasing documentary footage tornado",
        "extreme weather phenomena documentary",
        "weather timelapse lightning storm footage",
        "hurricane typhoon documentary footage",
    ],
    "transportation": [
        "train journey railway documentary footage",
        "aviation documentary airport behind scenes",
        "port shipping maritime documentary",
        "highway infrastructure aerial footage",
        "public transit documentary city transportation",
    ],
}

MIN_DURATION  = 180    # 3 minutes — ensures multiple scenes
MAX_DURATION  = 300    # 5 minutes — keeps file sizes manageable
MAX_HEIGHT    = 720    # 720p cap — good quality, reasonable size
RESULTS_PER_QUERY = 5  # yt-dlp ytsearchN: prefix


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(out: Path) -> dict:
    p = out / ".yt_checkpoint.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"done": [], "downloaded_ids": []}


def save_checkpoint(out: Path, cp: dict) -> None:
    (out / ".yt_checkpoint.json").write_text(json.dumps(cp))


# ---------------------------------------------------------------------------
# Duration filter for yt-dlp match_filter
# ---------------------------------------------------------------------------

def _duration_filter(info: dict, *, incomplete: bool = False) -> str | None:
    duration = info.get("duration") or 0
    if duration and duration < MIN_DURATION:
        return f"Too short ({duration}s < {MIN_DURATION}s)"
    if duration and duration > MAX_DURATION:
        return f"Too long ({duration}s > {MAX_DURATION}s)"
    return None


# ---------------------------------------------------------------------------
# Download one search query worth of videos
# ---------------------------------------------------------------------------

def download_query(
    query: str,
    cat_dir: Path,
    downloaded_ids: set[str],
    per_cat_remaining: int,
    dry_run: bool = False,
) -> list[str]:
    """Download up to per_cat_remaining videos matching query.

    Returns list of video IDs successfully downloaded.
    """
    try:
        import yt_dlp
    except ImportError:
        sys.exit("yt-dlp not installed. Run: pip install yt-dlp")

    new_ids: list[str] = []
    n = min(RESULTS_PER_QUERY, per_cat_remaining)
    search_url = f"ytsearch{n}:{query}"

    ydl_opts: dict = {
        "format": (
            f"bestvideo[height<={MAX_HEIGHT}][ext=mp4]"
            f"/bestvideo[height<={MAX_HEIGHT}]"
            f"/best[height<={MAX_HEIGHT}]"
        ),
        "outtmpl": str(cat_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "match_filter": _duration_filter,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
    }

    if dry_run:
        print(f"      [dry-run] would search: {search_url!r}")
        return []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=True)
            if not info:
                return []
            entries = info.get("entries") or [info]
            for entry in entries:
                if not entry:
                    continue
                vid_id = entry.get("id", "")
                if vid_id and vid_id not in downloaded_ids:
                    # Check file actually landed on disk
                    matches = list(cat_dir.glob(f"{vid_id}.*"))
                    if matches:
                        new_ids.append(vid_id)
    except Exception as exc:
        print(f"      ⚠ yt-dlp error for {query!r}: {exc}")

    return new_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Download YouTube video corpus for b-roll benchmarking"
    )
    ap.add_argument("--out",          type=Path, default=Path("./clips"),
                    help="Output directory (default: ./clips)")
    ap.add_argument("--total",        type=int,  default=200,
                    help="Total videos to download (default: 200)")
    ap.add_argument("--per-category", type=int,  default=None,
                    help="Max videos per category (default: total // num_categories)")
    ap.add_argument("--categories",   nargs="*", default=None,
                    help="Subset of categories to download (default: all 12)")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Print what would be downloaded without actually downloading")
    ap.add_argument("--reset",        action="store_true",
                    help="Ignore existing checkpoint and start fresh")
    args = ap.parse_args()

    categories = args.categories or list(SEARCH_TERMS)
    unknown = set(categories) - set(SEARCH_TERMS)
    if unknown:
        sys.exit(f"Unknown categories: {unknown}. Valid: {sorted(SEARCH_TERMS)}")

    args.out.mkdir(parents=True, exist_ok=True)

    cp = {} if args.reset else load_checkpoint(args.out)
    done_keys:       set[str] = set(cp.get("done", []))
    downloaded_ids:  set[str] = set(cp.get("downloaded_ids", []))

    per_cat_cap = args.per_category or max(1, args.total // len(categories))

    # Count already-downloaded videos per category (survive restarts)
    cat_counts: dict[str, int] = {
        cat: sum(1 for vid_id in downloaded_ids
                 if (args.out / cat / f"{vid_id}.mp4").exists()
                 or any((args.out / cat).glob(f"{vid_id}.*")))
        if (args.out / cat).exists() else 0
        for cat in categories
    }

    total_queries = sum(len(SEARCH_TERMS[c]) for c in categories)
    print(f"Categories   : {len(categories)}  ({', '.join(categories)})")
    print(f"Per category : {per_cat_cap}")
    print(f"Search terms : {total_queries}")
    print(f"Results/term : {RESULTS_PER_QUERY}")
    print(f"Target       : {args.total}  videos  ({MIN_DURATION}–{MAX_DURATION}s, ≤{MAX_HEIGHT}p)")
    print(f"Checkpoint   : {len(downloaded_ids)} already downloaded")
    print()

    new_total = 0

    for cat in categories:
        cat_dir = args.out / cat
        cat_dir.mkdir(exist_ok=True)

        cat_downloaded = cat_counts.get(cat, 0)
        queries = SEARCH_TERMS[cat]

        print(f"[{cat}]  {cat_downloaded}/{per_cat_cap} on disk")

        for qi, query in enumerate(queries):
            if len(downloaded_ids) >= args.total:
                print("  Reached total target.")
                break

            if cat_downloaded >= per_cat_cap:
                print("  Category full.")
                break

            key = f"{cat}|{qi}|{query}"
            if key in done_keys:
                print(f"  ✓ (cached) {query!r}")
                continue

            remaining = per_cat_cap - cat_downloaded
            print(f"  → {query!r}  (need {remaining} more)")

            new_ids = download_query(
                query, cat_dir, downloaded_ids,
                per_cat_remaining=remaining,
                dry_run=args.dry_run,
            )

            for vid_id in new_ids:
                downloaded_ids.add(vid_id)
                cat_downloaded += 1
                new_total += 1
                print(f"    ✓ {vid_id}")

            done_keys.add(key)
            if not args.dry_run:
                save_checkpoint(args.out, {
                    "done":           list(done_keys),
                    "downloaded_ids": list(downloaded_ids),
                })

            # Brief pause between queries — be a polite yt-dlp user
            if not args.dry_run and qi < len(queries) - 1:
                time.sleep(1.0)

        print()

    print(f"✓ Downloaded {new_total} new videos  ({len(downloaded_ids)} total)")
    print(f"  Output     : {args.out.resolve()}")
    print(f"  Checkpoint : {args.out / '.yt_checkpoint.json'}")


if __name__ == "__main__":
    main()
