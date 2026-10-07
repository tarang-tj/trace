import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tracecv.api import app, detector


@pytest.fixture
def client(monkeypatch):
    directory = next((path for path in ["artifacts/trace", "artifacts/pilot"]
                      if all(Path(path, filename).exists()
                             for filename in ("model.pt", "calibration.json", "config.json"))), None)
    if directory is None:
        pytest.skip("Train a detector to run checkpoint-backed integration checks")
    monkeypatch.setenv("TRACE_MODEL", directory)
    detector.cache_clear()
    with TestClient(app) as value:
        yield value
    detector.cache_clear()


def image_bytes(size=(96, 96)):
    image = Image.fromarray(np.full((*size[::-1], 3), 128, np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_real_checkpoint_serves_evidence_and_reobservation(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ready" and len(health["checkpoint_sha256"]) == 64
    payload = {"seed": 2000021, "scenario": "moved"}
    before = client.post("/api/inspect-scene", json=payload)
    assert before.status_code == 200
    result = before.json()
    assert 0 <= result["changed_fraction"] <= 1
    assert result["model_sha256"] == health["checkpoint_sha256"]
    assert result["probability_overlay"].startswith("data:image/png;base64,")
    assert "mask" not in result["diagnostics"]  # Hidden labels never enter planner inputs.
    after = client.post("/api/inspect-scene", json={**payload, "action": "relight"}).json()
    assert after["reference"] == result["reference"]
    assert after["observation"] != result["observation"]


def test_upload_validation_and_valid_pair(client):
    files = {"reference": ("a.png", image_bytes(), "image/png"),
             "observation": ("b.png", image_bytes(), "image/png")}
    response = client.post("/api/inspect", files=files)
    assert response.status_code == 200
    assert response.json()["source"] == "upload"
    bad = client.post("/api/inspect", files={**files, "observation": ("b.png", image_bytes((64, 96)), "image/png")})
    assert bad.status_code == 422
    bad = client.post("/api/inspect", files={**files, "observation": ("b.png", b"not an image", "image/png")})
    assert bad.status_code == 422


def test_missing_checkpoint_and_invalid_scene(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_MODEL", str(tmp_path))
    detector.cache_clear()
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 503
        assert client.post("/api/scene", json={"scenario": "invalid"}).status_code == 422
        assert client.post("/api/scene", json={"seed": -1}).status_code == 422
    detector.cache_clear()
