import json

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.lib.categories import FALLBACK, CategoryIndex, load_categories

client = TestClient(app_module.app)


# ---------------------------------------------------------------------------
# CategoryIndex.classify
# ---------------------------------------------------------------------------

ANCHORS = {
    "nature": [1.0, 0.0, 0.0],
    "urban": [0.0, 1.0, 0.0],
}


def test_classify_picks_closest_anchor():
    idx = CategoryIndex(ANCHORS, min_sim=0.1)
    label, sim = idx.classify([0.9, 0.1, 0.0])
    assert label == "nature"
    assert sim > 0.8


def test_classify_falls_back_below_threshold():
    idx = CategoryIndex(ANCHORS, min_sim=0.5)
    label, _ = idx.classify([0.0, 0.0, 1.0])
    assert label == FALLBACK


def test_classify_empty_anchors():
    idx = CategoryIndex({}, min_sim=0.1)
    assert idx.classify([1.0, 0.0])[0] == FALLBACK


# ---------------------------------------------------------------------------
# load_categories
# ---------------------------------------------------------------------------

def test_load_categories_defaults(tmp_path):
    cats = load_categories(tmp_path)
    assert "nature" in cats and len(cats) >= 5


def test_load_categories_custom_override(tmp_path):
    (tmp_path / ".categories.json").write_text(
        json.dumps({"drone shots": "aerial drone footage from above"})
    )
    cats = load_categories(tmp_path)
    assert cats == {"drone shots": "aerial drone footage from above"}


def test_load_categories_ignores_invalid_json(tmp_path):
    (tmp_path / ".categories.json").write_text("not json {")
    cats = load_categories(tmp_path)
    assert "nature" in cats


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class FakeES:
    def __init__(self, agg_buckets=None, search_hits=None):
        self.agg_buckets = agg_buckets or []
        self.search_hits = search_hits or []

    def search(self, index=None, size=None, aggs=None, query=None, **kwargs):
        if aggs is not None:
            return {"aggregations": {"clips_per_cat": {"buckets": self.agg_buckets}}}
        return {"hits": {"hits": self.search_hits}}


def test_categories_endpoint_sorts_other_last(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "es",
        FakeES(
            agg_buckets=[
                {"key": "other", "clips": {"value": 9}},
                {"key": "nature", "clips": {"value": 5}},
                {"key": "", "clips": {"value": 2}},
                {"key": "urban", "clips": {"value": 7}},
            ]
        ),
    )
    r = client.get("/api/categories")
    assert r.status_code == 200
    cats = r.json()["categories"]
    assert [c["label"] for c in cats] == ["urban", "nature", "other"]


def _search_hit(clip: str, chunk: int) -> dict:
    return {
        "_score": None,
        "_source": {
            "chunk_id": f"{clip}__scene__{chunk:03d}",
            "clip_id": clip,
            "duration": 2.0,
            "start_sec": 0.0,
            "end_sec": 2.0,
            "uploaded_at": "2026-07-23",
        },
    }


def test_category_endpoint_dedupes_and_includes_date(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "es",
        FakeES(search_hits=[_search_hit("a", 0), _search_hit("a", 1), _search_hit("b", 0)]),
    )
    r = client.get("/api/category/nature")
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert [h["clip_id"] for h in hits] == ["a", "b"]
    assert hits[0]["uploaded_at"] == "2026-07-23"
    assert hits[0]["score"] == 0.0
