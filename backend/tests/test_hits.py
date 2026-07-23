from backend.app import _hits_payload


def make_hit(chunk_id: str, clip_id: str, score: float = 1.0) -> dict:
    return {
        "chunk_id": chunk_id,
        "clip_id": clip_id,
        "_score": score,
        "duration": 2.0,
        "start_sec": 0.0,
        "end_sec": 2.0,
    }


def test_dedupes_to_one_chunk_per_clip():
    hits = [
        make_hit("a__scene__000", "a", 0.9),
        make_hit("a__scene__001", "a", 0.8),
        make_hit("b__scene__000", "b", 0.7),
    ]
    out = _hits_payload(hits)
    assert [h["chunk_id"] for h in out] == ["a__scene__000", "b__scene__000"]


def test_keeps_first_hit_per_clip():
    hits = [
        make_hit("a__scene__003", "a", 0.95),
        make_hit("a__scene__000", "a", 0.5),
    ]
    out = _hits_payload(hits)
    assert len(out) == 1
    assert out[0]["chunk_id"] == "a__scene__003"
    assert out[0]["score"] == 0.95


def test_excludes_requested_chunk():
    hits = [
        make_hit("a__scene__000", "a"),
        make_hit("b__scene__000", "b"),
    ]
    out = _hits_payload(hits, exclude_id="a__scene__000")
    assert [h["chunk_id"] for h in out] == ["b__scene__000"]


def test_limits_to_k():
    hits = [make_hit(f"c{i}__scene__000", f"c{i}") for i in range(20)]
    out = _hits_payload(hits, k=5)
    assert len(out) == 5


def test_payload_shape():
    out = _hits_payload([make_hit("a__scene__000", "a", 0.42)])
    assert out[0] == {
        "chunk_id": "a__scene__000",
        "clip_id": "a",
        "score": 0.42,
        "duration": 2.0,
        "start_sec": 0.0,
        "end_sec": 2.0,
        "uploaded_at": None,
    }
