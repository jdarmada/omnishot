"""
Download longer b-roll clips from YouTube via yt-dlp.

Lightweight version: 6 categories, fewer queries. Checkpointing resumes
interrupted runs. No API key required.

Usage:
    python scripts/download_youtube.py --out ./clips --total 20
    python scripts/download_youtube.py --out ./clips --categories nature animals
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SEARCH_TERMS: dict[str, list[str]] = {
    "animals": [
        "wildlife documentary Africa animals",
        "ocean marine life documentary footage",
        "pets dogs cats funny compilation",
    ],
    "nature": [
        "nature timelapse documentary landscape",
        "mountain landscape drone footage 4k",
        "ocean waves beach cinematic b-roll",
    ],
    "people": [
        "street life urban documentary photography",
        "people working office b-roll footage",
        "musicians street performance footage",
    ],
    "urban": [
        "city 4k drone footage aerial",
        "night city neon lights b-roll",
        "urban timelapse traffic 4k",
    ],
    "sport": [
        "surfing skateboarding 4k footage",
        "marathon running race footage",
        "gym workout fitness footage 4k",
    ],
    "abstract": [
        "ink in water 4k slow motion",
        "smoke fire slow motion footage",
        "bubbles liquid macro cinematic",
    ],
}

MIN_DURATION = 120
MAX_DURATION = 1200
MAX_HEIGHT = 720
RESULTS_PER_QUERY = 3


def _progress_hook(d: dict) -> None:
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "").strip()
        name = Path(d.get("filename", "")).name[:40]
        print(f"\r      {name}  {pct}    ", end="", flush=True)
    elif d["status"] == "finished":
        print()


def load_checkpoint(out: Path) -> dict:
    p = out / ".yt_checkpoint.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"done": [], "downloaded_ids": []}


def save_checkpoint(out: Path, cp: dict) -> None:
    (out / ".yt_checkpoint.json").write_text(json.dumps(cp))


def _duration_filter(info: dict, *, incomplete: bool = False) -> str | None:
    duration = info.get("duration") or 0
    if info.get("is_live") or info.get("was_live"):
        return "Live content"
    if not duration:
        return None if incomplete else "Unknown duration"
    if duration < MIN_DURATION:
        return f"Too short ({duration}s < {MIN_DURATION}s)"
    if duration > MAX_DURATION:
        return f"Too long ({duration}s > {MAX_DURATION}s)"
    return None


def download_query(
    query: str,
    cat_dir: Path,
    downloaded_ids: set[str],
    per_cat_remaining: int,
    dry_run: bool = False,
) -> list[str]:
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
        "progress_hooks": [_progress_hook],
        "ignoreerrors": True,
        "match_filter": _duration_filter,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": 300 * 1024 * 1024,
        "sleep_interval": 3,
        "max_sleep_interval": 8,
        "sleep_interval_requests": 1,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }

    if dry_run:
        print(f"      [dry-run] would search: {search_url!r}")
        return []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=True)
            if not info:
                return []
            for entry in info.get("entries") or [info]:
                if not entry:
                    continue
                vid_id = entry.get("id", "")
                if vid_id and vid_id not in downloaded_ids:
                    if list(cat_dir.glob(f"{vid_id}.*")):
                        new_ids.append(vid_id)
    except Exception as exc:
        print(f"      ⚠ yt-dlp error for {query!r}: {exc}")

    return new_ids


def main() -> None:
    global RESULTS_PER_QUERY
    ap = argparse.ArgumentParser(description="Download YouTube b-roll clips")
    ap.add_argument("--out", type=Path, default=Path("./clips"))
    ap.add_argument("--total", type=int, default=20)
    ap.add_argument("--per-category", type=int, default=None)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--per-query", type=int, default=RESULTS_PER_QUERY)
    args = ap.parse_args()

    RESULTS_PER_QUERY = args.per_query
    categories = args.categories or list(SEARCH_TERMS)
    unknown = set(categories) - set(SEARCH_TERMS)
    if unknown:
        sys.exit(f"Unknown categories: {unknown}. Valid: {sorted(SEARCH_TERMS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    cp = {} if args.reset else load_checkpoint(args.out)
    done_keys: set[str] = set(cp.get("done", []))
    downloaded_ids: set[str] = set(cp.get("downloaded_ids", []))
    per_cat_cap = args.per_category or max(1, args.total // len(categories))

    print(f"Categories : {', '.join(categories)}")
    print(f"Target     : {args.total} videos ({MIN_DURATION}–{MAX_DURATION}s)")

    new_total = 0
    for cat in categories:
        cat_dir = args.out / cat
        cat_dir.mkdir(exist_ok=True)
        cat_downloaded = sum(
            1
            for vid_id in downloaded_ids
            if any((cat_dir).glob(f"{vid_id}.*"))
        )
        print(f"[{cat}]  {cat_downloaded}/{per_cat_cap} on disk")

        for qi, query in enumerate(SEARCH_TERMS[cat]):
            if len(downloaded_ids) >= args.total or cat_downloaded >= per_cat_cap:
                break
            key = f"{cat}|{qi}|{query}|n{RESULTS_PER_QUERY}"
            if key in done_keys:
                continue

            remaining = per_cat_cap - cat_downloaded
            print(f"  → {query!r}")
            new_ids = download_query(
                query,
                cat_dir,
                downloaded_ids,
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
                save_checkpoint(
                    args.out,
                    {"done": list(done_keys), "downloaded_ids": list(downloaded_ids)},
                )
            if not args.dry_run and qi < len(SEARCH_TERMS[cat]) - 1:
                time.sleep(5.0 + random.uniform(0, 5))

    print(f"✓ Downloaded {new_total} new videos ({len(downloaded_ids)} total)")
    print(f"  Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
