import csv
import io

import pytest
from fastapi.testclient import TestClient

from wafer_inspect import api
from wafer_inspect.config import settings
from wafer_inspect.model import Prediction
from wafer_inspect.review_queue import ReviewQueue


def _pred(label="good", conf=0.5, unc=0.9):
    return Prediction(label, conf, unc, {"good": conf, "particle": 0.2, "scratch": 0.1})


def test_queue_orders_by_uncertainty_labels_and_exports():
    q = ReviewQueue(":memory:")
    a = q.add("r1", _pred(unc=0.3), b"png-a")
    b = q.add("r2", _pred(unc=0.9), b"png-b")
    assert [r["id"] for r in q.list("pending")] == [b, a]
    assert q.image(a) == b"png-a" and q.image(999) is None
    assert q.label(a, "scratch") and not q.label(999, "good")
    with pytest.raises(ValueError):
        q.label(a, "void")
    assert q.stats() == {"pending": 1, "labelled": 1}
    rows = list(csv.DictReader(io.StringIO(q.export_csv("labelled"))))
    assert len(rows) == 1 and rows[0]["true_label"] == "scratch" and rows[0]["image_url"].endswith("/image")


@pytest.fixture
def client(monkeypatch, tmp_path, model_file):
    monkeypatch.setattr(settings, "MODEL_PATH", model_file)
    monkeypatch.setattr(settings, "QUEUE_DB_PATH", str(tmp_path / "q.sqlite3"))
    monkeypatch.setattr(settings, "SAMPLES_DIR", str(tmp_path / "samples"))
    with TestClient(api.app) as c:
        yield c


def _post(client, png, **kw):
    return client.post("/predict", files={"file": ("w.png", png, "image/png")}, **kw)


def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "healthy"
    assert client.get("/readyz").status_code == 200
    assert client.get("/").status_code == 200


def test_not_ready_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(settings, "QUEUE_DB_PATH", str(tmp_path / "q.sqlite3"))
    with TestClient(api.app) as c:
        assert c.get("/healthz").json()["status"] == "degraded"
        assert c.get("/readyz").status_code == 503


def test_predict_confident_is_not_queued(client, wafer_png, monkeypatch):
    monkeypatch.setattr(settings, "REVIEW_CONFIDENCE_THRESHOLD", 0.0)
    body = _post(client, wafer_png(), headers={"X-Request-ID": "rid-1"}).json()
    assert body["needs_review"] is False and body["queue_id"] is None
    assert body["request_id"] == "rid-1"
    assert client.get("/queue/stats").json()["pending"] == 0


def test_low_confidence_is_queued_exported_and_labelled(client, wafer_png, monkeypatch):
    monkeypatch.setattr(settings, "REVIEW_CONFIDENCE_THRESHOLD", 1.01)  # everything is "uncertain"
    body = _post(client, wafer_png("particle", 0.2)).json()
    assert body["needs_review"] is True and body["queue_id"] == 1
    assert 0.0 <= body["uncertainty"] <= 1.0

    pending = client.get("/queue").json()
    assert len(pending) == 1 and pending[0]["predicted_label"] == body["label"]
    exported = client.get("/queue/export?format=csv")
    assert exported.headers["content-type"].startswith("text/csv")
    assert "predicted_label" in exported.text.splitlines()[0]
    assert client.get("/queue/1/image").headers["content-type"] == "image/png"

    assert client.post("/queue/1/label", json={"label": "scratch"}).json()["status"] == "labelled"
    assert client.get("/queue").json() == []
    assert client.get("/queue/export?status=labelled").json()[0]["true_label"] == "scratch"
    assert client.post("/queue/1/label", json={"label": "void"}).status_code == 422
    assert client.post("/queue/99/label", json={"label": "good"}).status_code == 404
    assert client.get("/queue/export?format=xml").status_code == 400


def test_bad_uploads(client, monkeypatch):
    assert _post(client, b"not an image").status_code == 400
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10)
    assert _post(client, b"x" * 100).status_code == 413
    assert client.get("/queue/5/image").status_code == 404


def test_samples_endpoints(client, tmp_path):
    folder = tmp_path / "samples"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"\x89PNG")
    assert client.get("/samples").json() == ["a.png"]
    assert client.get("/samples/a.png").status_code == 200
    assert client.get("/samples/..%2Fsecret.png").status_code == 404
    assert client.get("/samples/a.txt").status_code == 404
    assert client.get("/metrics-report").status_code in (200, 404)
