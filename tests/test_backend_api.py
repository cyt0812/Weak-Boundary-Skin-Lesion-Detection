import io

from fastapi.testclient import TestClient
from PIL import Image

from backend import main


def png_bytes(size=(64, 48), color=(120, 80, 60)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_mock_mode(monkeypatch):
    monkeypatch.setattr(main, "MOCK_MODE", True)
    monkeypatch.setattr(main, "model", None)
    with TestClient(main.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["mock_mode"] is True
    assert response.json()["status"] == "ok"


def test_models_endpoint_lists_both_tasks():
    with TestClient(main.app) as client:
        response = client.get("/api/models")
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["models"]]
    assert ids == ["classification", "segmentation"]


def test_mock_predict_contract(monkeypatch):
    monkeypatch.setattr(main, "MOCK_MODE", True)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "mock"
    assert body["model_version"] == "mock-classifier-v1"
    assert len(body["scores"]) == 7
    assert body["predicted_class"] in [item["code"] for item in body["scores"]]
    assert body["latency_ms"] >= 0


def test_mock_segment_contract(monkeypatch):
    monkeypatch.setattr(main, "MOCK_MODE", True)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/segment",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "mock"
    assert body["model_version"] == "mock-ellipse-v1"
    assert body["mask"].startswith("data:image/png;base64,")
    assert body["overlay"].startswith("data:image/png;base64,")
    assert 0 <= body["foreground_ratio"] <= 100
    assert "SYNTHETIC MOCK MASK" in body["notice"]


def test_rejects_non_image_content_type():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("sample.txt", b"not an image", "text/plain")},
        )
    assert response.status_code == 400


def test_rejects_invalid_image_bytes():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/segment",
            files={"file": ("broken.png", b"not really a png", "image/png")},
        )
    assert response.status_code == 400


def test_evaluation_endpoint_does_not_invent_metrics():
    with TestClient(main.app) as client:
        response = client.get("/api/evaluate")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["metrics"] is None
