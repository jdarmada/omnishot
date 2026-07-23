from fastapi.testclient import TestClient

from backend import app as app_module

client = TestClient(app_module.app)


class FakeES:
    def __init__(self, ping_ok: bool = True):
        self._ping_ok = ping_ok

    def ping(self):
        return self._ping_ok


def test_health_reports_es_up(monkeypatch):
    monkeypatch.setattr(app_module, "es", FakeES(ping_ok=True))
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["elasticsearch"] is True


def test_health_survives_es_down(monkeypatch):
    class ExplodingES:
        def ping(self):
            raise ConnectionError("no cluster")

    monkeypatch.setattr(app_module, "es", ExplodingES())
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["elasticsearch"] is False


def test_status_shape():
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("clips", "chunks", "state", "watch_dir", "events"):
        assert key in body


def test_set_library_api_rejects_missing_path():
    r = client.post("/api/library", json={"path": "/definitely/not/a/real/path"})
    assert r.status_code == 404
