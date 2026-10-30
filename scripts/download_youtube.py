"""
Download longer b-roll clips from YouTube via yt-dlp.

Use the built-in categories or a JSON theme. Checkpointing resumes
interrupted runs. No API key required.

Usage:
    python scripts/download_youtube.py --out ./clips --total 20
    python scripts/download_youtube.py --out ./clips --categories nature animals
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

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
COOKIE_BROWSERS = ("brave", "chrome", "chromium", "edge", "firefox", "opera", "safari", "vivaldi", "whale")


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


def _duration_filter(
    info: dict, *, incomplete: bool = False,
    min_duration: int = MIN_DURATION, max_duration: int = MAX_DURATION,
) -> str | None:
    duration = info.get("duration") or 0
    if info.get("is_live") or info.get("was_live"):
        return "Live content"
    if not duration:
        return None if incomplete else "Unknown duration"
    if duration < min_duration:
        return f"Too short ({duration}s < {min_duration}s)"
    if duration > max_duration:
        return f"Too long ({duration}s > {max_duration}s)"
    return None


def _media_exists(cat_dir: Path, vid_id: str) -> bool:
    """Count completed media files, excluding partial downloads and sidecars."""
    return any(
        path.is_file() and path.stem == vid_id
        and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}
        and path.stat().st_size > 0
        for path in cat_dir.iterdir()
    ) if cat_dir.exists() else False


def download_query(
    query: str,
    cat_dir: Path,
    downloaded_ids: set[str],
    per_cat_remaining: int,
    dry_run: bool = False,
    *,
    per_query: int = RESULTS_PER_QUERY,
    min_duration: int = MIN_DURATION,
    max_duration: int = MAX_DURATION,
    cookies_from_browser: str | None = None,
    cookies: Path | None = None,
) -> list[str]:
    n = min(per_query, per_cat_remaining)
    if n <= 0:
        return []
    search_url = f"ytsearch{n}:{query}"
    if dry_run:
        print(f"      [dry-run] would search: {search_url!r}")
        return []
    try:
        import yt_dlp
    except ImportError:
        sys.exit('yt-dlp not installed. Run: python -m pip install "yt-dlp[default]"')

    reserved: set[str] = set()
    completed: set[str] = set()

    def match_filter(info: dict, *, incomplete: bool = False) -> str | None:
        reason = _duration_filter(
            info, incomplete=incomplete, min_duration=min_duration, max_duration=max_duration,
        )
        if reason:
            return reason
        vid_id = info.get("id")
        if vid_id in downloaded_ids or vid_id in completed:
            return "Already downloaded"
        if not incomplete and vid_id:
            # yt-dlp can validate the same entry more than once before downloading.
            if vid_id not in reserved:
                if len(reserved) >= n:
                    return "Download quota reached"
                reserved.add(vid_id)
        return None

    def progress_hook(data: dict) -> None:
        _progress_hook(data)
        if data["status"] == "finished":
            vid_id = data.get("info_dict", {}).get("id")
            if vid_id in reserved:
                completed.add(vid_id)

    ydl_opts: dict = {
        "format": (
            f"bestvideo[height<={MAX_HEIGHT}][ext=mp4]"
            f"/bestvideo[height<={MAX_HEIGHT}]"
            f"/best[height<={MAX_HEIGHT}]"
        ),
        "outtmpl": str(cat_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "js_runtimes": {"deno": {}, "node": {}},
        "progress_hooks": [progress_hook],
        "ignoreerrors": True,
        "match_filter": match_filter,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": 300 * 1024 * 1024,
        "sleep_interval": 3,
        "max_sleep_interval": 8,
        "sleep_interval_requests": 1,
    }
    if cookies_from_browser is not None:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    elif cookies is not None:
        ydl_opts["cookiefile"] = str(cookies.expanduser())

    new_ids: list[str] = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=True)
            if not info:
                return []
            for entry in info.get("entries") or [info]:
                if not entry:
                    continue
                vid_id = entry.get("id", "")
                if (vid_id in reserved and vid_id not in downloaded_ids
                        and vid_id not in new_ids and _media_exists(cat_dir, vid_id)):
                    new_ids.append(vid_id)
    except Exception as exc:
        print(f"      ⚠ yt-dlp error for {query!r}: {exc}")
    return new_ids


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_theme(path: Path) -> dict[str, list[str]]:
    """Read and validate categories and queries from a JSON theme."""
    theme = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(theme, dict) or not theme:
        raise ValueError("theme must be a nonempty JSON object")
    for category, queries in theme.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", category):
            raise ValueError(f"invalid category name: {category!r}")
        if (not isinstance(queries, list) or not queries
                or any(not isinstance(query, str) or not query.strip() for query in queries)):
            raise ValueError(f"category {category!r} must contain nonempty query strings")
    return theme


def main() -> None:
    ap = argparse.ArgumentParser(description="Download YouTube b-roll clips")
    ap.add_argument("--out", type=Path, default=Path("./clips"))
    ap.add_argument("--theme", type=Path, help="JSON mapping categories to search query lists")
    ap.add_argument("--total", type=positive_int, default=20)
    ap.add_argument("--per-category", type=positive_int, default=None)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Preview all queries without network or writes")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--per-query", type=positive_int, default=RESULTS_PER_QUERY)
    ap.add_argument("--min-duration", type=positive_int, default=MIN_DURATION)
    ap.add_argument("--max-duration", type=positive_int, default=MAX_DURATION)
    authentication = ap.add_mutually_exclusive_group()
    authentication.add_argument(
        "--cookies-from-browser",
        choices=COOKIE_BROWSERS,
        help="Use the selected browser's signed-in session for downloads",
    )
    authentication.add_argument("--cookies", type=Path, help="Read a Netscape-format cookie file for downloads")
    args = ap.parse_args()
    if args.min_duration > args.max_duration:
        ap.error("--min-duration must not exceed --max-duration")
    try:
        terms = load_theme(args.theme) if args.theme else SEARCH_TERMS
    except (OSError, ValueError) as exc:
        ap.error(f"Cannot load theme: {exc}")
    categories = list(dict.fromkeys(args.categories or terms))
    unknown = set(categories) - set(terms)
    if unknown:
        ap.error(f"Unknown categories: {sorted(unknown)}. Valid: {sorted(terms)}")

    print(f"Categories : {', '.join(categories)}")
    print(f"Target     : {args.total} videos ({args.min_duration}–{args.max_duration}s)")
    print(f"Per query  : {args.per_query} results")
    if args.per_category:
        print(f"Per category: {args.per_category} videos maximum")
    if args.dry_run:
        for cat in categories:
            for query in terms[cat]:
                print(f"  [{cat}] {query}")
        print(f"  Output: {args.out.resolve()}")
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        ap.error('python-dotenv not installed. Run: python -m pip install "yt-dlp[default]" python-dotenv')
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    if args.cookies_from_browser is None and args.cookies is None:
        browser = os.environ.get("YOUTUBE_COOKIES_FROM_BROWSER", "").strip()
        cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
        if browser and cookie_file:
            ap.error("Set only one of YOUTUBE_COOKIES_FROM_BROWSER and YOUTUBE_COOKIES_FILE")
        args.cookies_from_browser = browser or None
        args.cookies = Path(cookie_file) if cookie_file else None
    if args.cookies_from_browser is not None and args.cookies_from_browser not in COOKIE_BROWSERS:
        ap.error("YOUTUBE_COOKIES_FROM_BROWSER must name a supported browser; see --help")
    if args.cookies is not None and not args.cookies.expanduser().is_file():
        ap.error("--cookies must name an existing cookie file")

    args.out.mkdir(parents=True, exist_ok=True)
    cp = {} if args.reset else load_checkpoint(args.out)
    done_keys: set[str] = set(cp.get("done", []))
    downloaded_ids: set[str] = set(cp.get("downloaded_ids", []))
    existing = {
        cat: sum(_media_exists(args.out / cat, vid_id) for vid_id in downloaded_ids)
        for cat in categories
    }
    remaining_total = max(0, args.total - len(downloaded_ids))
    base, extra = divmod(remaining_total, len(categories))
    caps = {
        cat: args.per_category if args.per_category is not None else existing[cat] + base + (i < extra)
        for i, cat in enumerate(categories)
    }

    new_total = 0
    for cat in categories:
        cat_downloaded = existing[cat]
        print(f"[{cat}]  {cat_downloaded}/{caps[cat]} on disk")
        for qi, query in enumerate(terms[cat]):
            remaining = min(caps[cat] - cat_downloaded, args.total - len(downloaded_ids))
            if remaining <= 0:
                break
            key = f"{cat}|{qi}|{query}|n{args.per_query}|{args.min_duration}-{args.max_duration}"
            if key in done_keys:
                continue
            cat_dir = args.out / cat
            cat_dir.mkdir(exist_ok=True)
            print(f"  → {query!r}")
            new_ids = download_query(
                query, cat_dir, downloaded_ids, per_cat_remaining=remaining,
                per_query=args.per_query, min_duration=args.min_duration, max_duration=args.max_duration,
                cookies_from_browser=args.cookies_from_browser, cookies=args.cookies,
            )
            added = 0
            for vid_id in new_ids:
                if vid_id in downloaded_ids or added >= remaining:
                    continue
                downloaded_ids.add(vid_id)
                cat_downloaded += 1
                new_total += 1
                added += 1
                print(f"    ✓ {vid_id}")
            if added:
                done_keys.add(key)
                save_checkpoint(
                    args.out,
                    {"done": sorted(done_keys), "downloaded_ids": sorted(downloaded_ids)},
                )
            if (qi < len(terms[cat]) - 1 and cat_downloaded < caps[cat]
                    and len(downloaded_ids) < args.total):
                time.sleep(5.0 + random.uniform(0, 5))

    print(f"✓ Downloaded {new_total} new videos ({len(downloaded_ids)} total)")
    print(f"  Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
