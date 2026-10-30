"""Offline regression tests for the YouTube downloader."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "download_youtube.py"
SPEC = importlib.util.spec_from_file_location("download_youtube", SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DownloaderTests(unittest.TestCase):
    def run_main(self, args):
        with patch.object(sys, "argv", [str(SCRIPT), *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            downloader.main()

    def test_dry_run_without_dependencies_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            result = subprocess.run(
                [sys.executable, "-S", str(SCRIPT), "--out", str(out), "--dry-run"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("wildlife documentary", result.stdout)
            self.assertFalse(out.exists())
            out.mkdir()
            checkpoint = out / ".yt_checkpoint.json"
            checkpoint.write_text('{"done": [], "downloaded_ids": ["existing"]}')
            before = checkpoint.read_bytes()
            subprocess.run(
                [sys.executable, "-S", str(SCRIPT), "--out", str(out), "--dry-run"],
                capture_output=True, check=True,
            )
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertEqual(list(out.iterdir()), [checkpoint])

    def test_theme_category_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            theme = Path(tmp) / "theme.json"
            theme.write_text(json.dumps({"ski": ["ski powder"], "mountain": ["snowy village"]}))
            result = subprocess.run(
                [sys.executable, "-S", str(SCRIPT), "--theme", str(theme), "--categories", "ski",
                 "--min-duration", "10", "--max-duration", "60", "--dry-run"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("ski powder", result.stdout)
            self.assertIn("10–60s", result.stdout)
            self.assertNotIn("snowy village", result.stdout)
            self.assertNotIn("wildlife documentary", result.stdout)

    def test_help_without_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, check=True,
        )
        self.assertIn("--theme", result.stdout)

    def test_invalid_themes_and_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            theme = Path(tmp) / "theme.json"
            for value in ({}, [], {"../escape": ["snow"]}, {"snow": []}, {"snow": [" "]}, {"snow": [1]}):
                theme.write_text(json.dumps(value))
                with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        self.run_main(["--theme", theme, "--dry-run"])
                    self.assertEqual(caught.exception.code, 2)
            for args in (["--total", "0"], ["--per-query", "-1"], ["--per-category", "0"],
                         ["--min-duration", "20", "--max-duration", "10"], ["--categories", "unknown"]):
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        self.run_main([*args, "--dry-run"])
                    self.assertEqual(caught.exception.code, 2)

    def test_total_remainder_and_resume_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls = []

            def fake_download(query, cat_dir, ids, per_cat_remaining, **kwargs):
                count = min(kwargs["per_query"], per_cat_remaining)
                calls.append((cat_dir.name, count))
                fresh = [f"{cat_dir.name}-{len(ids)}-{i}" for i in range(count)]
                for vid_id in fresh:
                    (cat_dir / f"{vid_id}.mp4").write_bytes(b"video")
                return fresh

            terms = {"a": ["one"], "b": ["two"], "c": ["three"]}
            with patch.object(downloader, "SEARCH_TERMS", terms), patch.object(
                downloader, "download_query", side_effect=fake_download,
            ), patch.object(downloader.time, "sleep"):
                self.run_main(["--out", out, "--total", "8"])
                self.assertEqual(calls, [("a", 3), ("b", 3), ("c", 2)])
                self.assertEqual(len(downloader.load_checkpoint(out)["downloaded_ids"]), 8)
                calls.clear()
                self.run_main(["--out", out, "--total", "8"])
                self.assertEqual(calls, [])

    def test_explicit_cap_never_exceeds_global_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_download(query, cat_dir, ids, per_cat_remaining, **kwargs):
                calls.append(per_cat_remaining)
                return [f"{cat_dir.name}-{i}" for i in range(per_cat_remaining)]

            with patch.object(downloader, "SEARCH_TERMS", {"a": ["one"], "b": ["two"]}), patch.object(
                downloader, "download_query", side_effect=fake_download,
            ):
                self.run_main(["--out", tmp, "--total", "5", "--per-category", "4", "--per-query", "4"])
            self.assertEqual(calls, [4, 1])
            self.assertEqual(len(downloader.load_checkpoint(Path(tmp))["downloaded_ids"]), 5)

    def test_failed_query_is_not_checkpointed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(downloader, "SEARCH_TERMS", {"ski": ["snow"]}), patch.object(
                downloader, "download_query", return_value=[],
            ) as query:
                self.run_main(["--out", tmp])
                self.run_main(["--out", tmp])
            self.assertEqual(query.call_count, 2)
            self.assertFalse((Path(tmp) / ".yt_checkpoint.json").exists())

    def test_match_filter_deduplicates_and_ignores_partials(self):
        with tempfile.TemporaryDirectory() as tmp:
            cat_dir = Path(tmp)
            statuses = []

            class FakeYoutubeDL:
                def __init__(self, opts):
                    self.opts = opts

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def extract_info(self, url, download):
                    self_url = "ytsearch3:snow"
                    if url != self_url:
                        raise AssertionError(url)
                    entries = []
                    for vid_id in ["old", "new", "new", "partial", "short"]:
                        entry = {"id": vid_id, "duration": 5 if vid_id == "short" else 30}
                        reason = self.opts["match_filter"](entry, incomplete=False)
                        statuses.append((vid_id, reason))
                        if reason:
                            continue
                        if self.opts["match_filter"](entry, incomplete=False) is not None:
                            raise AssertionError("Repeated validation rejected before download")
                        suffix = ".mp4.part" if vid_id == "partial" else ".mp4"
                        (cat_dir / f"{vid_id}{suffix}").write_bytes(b"video")
                        if vid_id != "partial":
                            self.opts["progress_hooks"][0]({"status": "finished", "info_dict": entry})
                        entries.append(entry)
                    return {"entries": entries}

            module = types.SimpleNamespace(YoutubeDL=FakeYoutubeDL)
            with patch.dict(sys.modules, {"yt_dlp": module}):
                result = downloader.download_query(
                    "snow", cat_dir, {"old"}, 3, min_duration=10, max_duration=60,
                )
            self.assertEqual(result, ["new"])
            self.assertEqual(statuses[0][1], "Already downloaded")
            self.assertEqual(statuses[2][1], "Already downloaded")
            self.assertIn("Too short", statuses[-1][1])

    def test_download_exception_returns_no_ids(self):
        class FailingYoutubeDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, *args, **kwargs):
                raise RuntimeError("Unavailable video")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"yt_dlp": types.SimpleNamespace(YoutubeDL=FailingYoutubeDL)},
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(downloader.download_query("snow", Path(tmp), set(), 1), [])

    def test_duration_filter_uses_custom_bounds(self):
        self.assertIsNone(downloader._duration_filter({"duration": 30}, min_duration=10, max_duration=60))
        self.assertIn("Too long", downloader._duration_filter({"duration": 90}, min_duration=10, max_duration=60))
        self.assertEqual(downloader._duration_filter({"is_live": True}), "Live content")


if __name__ == "__main__":
    unittest.main()
