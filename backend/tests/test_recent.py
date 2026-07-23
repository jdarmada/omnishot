import json

from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _write_manifest(entries: dict) -> None:
    app_module.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    app_module.MANIFEST.write_text(json.dumps(entries))


def _manifest_entry(clip: str) -> dict:
    return {
        "key": f"{clip}.mp4:1:1",
        "source": f"/library/{clip}.mp4",
        "chunk_ids": [f"{clip}__scene__000"],
        "chunk_paths": {f"{clip}__scene__000": f"/chunks/{clip}__scene__000.mp4"},
    }


class FakeES:
    def __init__(self, docs_by_id: dict):
        self.docs_by_id = docs_by_id

    def mget(self, index: str, ids: list, source_excludes=None):
        return {
            "docs": [
                {"found": cid in self.docs_by_id, "_source": self.docs_by_id.get(cid, {})}
                for cid in ids
            ]
        }


def _doc(clip: str) -> dict:
    return {
        "chunk_id": f"{clip}__scene__000",
        "clip_id": clip,
        "duration": 2.0,
        "start_sec": 0.0,
        "end_sec": 2.0,
    }


def test_recent_empty_manifest():
    _write_manifest({})
    r = client.get("/api/recent")
    assert r.status_code == 200
    assert r.json() == {"hits": []}


def test_recent_returns_newest_first(monkeypatch):
    _write_manifest(
        {
            "old.mp4": _manifest_entry("old"),
            "mid.mp4": _manifest_entry("mid"),
            "new.mp4": _manifest_entry("new"),
        }
    )
    monkeypatch.setattr(
        app_module,
        "es",
        FakeES({f"{c}__scene__000": _doc(c) for c in ("old", "mid", "new")}),
    )
    r = client.get("/api/recent?k=2")
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert [h["clip_id"] for h in hits] == ["new", "mid"]


def test_recent_skips_missing_docs(monkeypatch):
    _write_manifest({"gone.mp4": _manifest_entry("gone"), "here.mp4": _manifest_entry("here")})
    monkeypatch.setattr(app_module, "es", FakeES({"here__scene__000": _doc("here")}))
    r = client.get("/api/recent")
    hits = r.json()["hits"]
    assert [h["clip_id"] for h in hits] == ["here"]
