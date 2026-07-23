from pathlib import Path

import pytest

from backend import app as app_module


def test_set_library_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        app_module.set_library(tmp_path / "does-not-exist")


def test_set_library_rejects_file(tmp_path: Path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"not a folder")
    with pytest.raises(NotADirectoryError):
        app_module.set_library(f)


def test_set_library_same_path_is_noop(monkeypatch):
    # Re-pointing at the current library must not touch Elasticsearch.
    def boom(*args, **kwargs):
        raise AssertionError("corpus should not be cleared for a no-op switch")

    monkeypatch.setattr(app_module, "_clear_corpus", boom)
    current = app_module._watch_dir
    current.mkdir(parents=True, exist_ok=True)
    assert app_module.set_library(current) == current


def test_watch_key_relative_to_library(tmp_path: Path):
    clip = tmp_path / "sub" / "clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"x")
    assert app_module._watch_key(clip, tmp_path) == "sub/clip.mp4"


def test_watch_key_outside_library_falls_back_to_name(tmp_path: Path):
    outside = tmp_path / "clip.mp4"
    outside.write_bytes(b"x")
    other = tmp_path / "library"
    other.mkdir()
    assert app_module._watch_key(outside, other) == "clip.mp4"


def test_clip_key_changes_with_content(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"one")
    key1 = app_module._clip_key(clip)
    clip.write_bytes(b"one two three")
    key2 = app_module._clip_key(clip)
    assert key1 != key2
