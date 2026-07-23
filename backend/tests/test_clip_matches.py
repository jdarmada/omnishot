from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


def _hit(clip: str, scene: int, score: float) -> dict:
    return {
        "chunk_id": f"{clip}__scene__{scene:03d}",
        "clip_id": clip,
        "_score": score,
        "duration": 2.0,
        "start_sec": scene * 2.0,
        "end_sec": scene * 2.0 + 2.0,
        "uploaded_at": "2026-07-23",
    }


def test_unknown_qid_returns_410():
    r = client.post(
        "/api/clip_matches", json={"qid": "nope", "clip_id": "a"}
    )
    assert r.status_code == 410


def test_expands_clip_excluding_card_chunk(monkeypatch):
    qid = app_module._remember_qvec([0.1, 0.2])
    captured = {}

    def fake_knn(es, index, vec, k, num_candidates, filter_clauses=None):
        captured["filter"] = filter_clauses
        captured["vec"] = vec
        return [_hit("a", 0, 0.9), _hit("a", 3, 0.8), _hit("a", 7, 0.7)]

    monkeypatch.setattr(app_module, "knn_search", fake_knn)
    r = client.post(
        "/api/clip_matches",
        json={"qid": qid, "clip_id": "a", "exclude_chunk": "a__scene__000"},
    )
    assert r.status_code == 200
    hits = r.json()["hits"]
    # Card chunk excluded, siblings ranked, no per-clip dedup applied
    assert [h["chunk_id"] for h in hits] == ["a__scene__003", "a__scene__007"]
    assert captured["filter"] == [{"term": {"clip_id": "a"}}]
    assert captured["vec"] == [0.1, 0.2]


def test_qvec_cache_evicts_oldest():
    first = app_module._remember_qvec([1.0])
    for _ in range(app_module._QVEC_CACHE_MAX):
        app_module._remember_qvec([0.0])
    assert first not in app_module._qvec_cache
    assert len(app_module._qvec_cache) == app_module._QVEC_CACHE_MAX
