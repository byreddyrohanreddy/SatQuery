"""
Tests for the FastAPI endpoints and frontend integration.
"""

from fastapi.testclient import TestClient
from PIL import Image
import io

from satquery.api import app

client = TestClient(app)


def _make_image_bytes(width=100, height=100, color=(100, 100, 100), fmt="JPEG"):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "anthropic_api_key_set" in data
    assert "routing_mode" in data


def test_serve_frontend_files():
    # Test index.html via both / and /index.html
    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "SatQuery" in r_index.text

    r_index_direct = client.get("/index.html")
    assert r_index_direct.status_code == 200
    assert "SatQuery" in r_index_direct.text

    # Test results.html
    r_results = client.get("/results.html")
    assert r_results.status_code == 200

    # Test style.css
    r_css = client.get("/style.css")
    assert r_css.status_code == 200

    # Test script.js
    r_js = client.get("/script.js")
    assert r_js.status_code == 200


def test_serve_sample_images():
    r = client.get("/samples/sample_optical.jpg")
    assert r.status_code == 200


def test_query_endpoint_frontend_format():
    # Frontend sends image1 and query
    img_bytes = _make_image_bytes(color=(40, 90, 200))
    response = client.post(
        "/query",
        data={"query": "Where is the water?"},
        files={"image1": ("optical.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "visual_evidence" in data
    assert data["visual_evidence"]["type"] == "bounding_box"
    assert len(data["visual_evidence"]["boxes"]) > 0
    # Coordinates must be normalized (0.0 - 1.0)
    box = data["visual_evidence"]["boxes"][0]
    assert 0.0 <= box["x"] <= 1.0
    assert 0.0 <= box["y"] <= 1.0
    assert 0.0 <= box["width"] <= 1.0
    assert 0.0 <= box["height"] <= 1.0
    assert data["execution_trace"]["task_selected"] == "grounding"
    assert "model_called" in data["execution_trace"]


def test_query_endpoint_dual_images():
    img1 = _make_image_bytes(color=(50, 50, 50))
    img2 = _make_image_bytes(color=(200, 200, 200))
    response = client.post(
        "/query",
        data={"query": "What changed between these two images?"},
        files={
            "image1": ("before.jpg", img1, "image/jpeg"),
            "image2": ("after.jpg", img2, "image/jpeg"),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["visual_evidence"]["type"] == "change_regions"
    assert data["execution_trace"]["task_selected"] == "change_analysis"


def test_query_endpoint_clarification():
    # Only 1 image provided, but query asks for change detection (needs 2)
    img_bytes = _make_image_bytes()
    response = client.post(
        "/query",
        data={"query": "What changed between these images?"},
        files={"image1": ("test.jpg", img_bytes, "image/jpeg")},
    )
    # 400 Bad Request with friendly detail message for frontend banner
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "requires at least 2 image(s)" in data["detail"]


def test_query_endpoint_no_files():
    response = client.post(
        "/query",
        data={"query": "What is this?"},
    )
    assert response.status_code in (400, 422)


def test_query_endpoint_sample_tif():
    import os
    tif_path = "samples/sample.tif"
    if os.path.exists(tif_path):
        with open(tif_path, "rb") as f:
            response = client.post(
                "/query",
                data={"query": "What is in this image?"},
                files={"image1": ("sample.tif", f.read(), "image/tiff")},
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "execution_trace" in data
        assert "image_previews" in data
        assert len(data["image_previews"]) > 0
        assert data["image_previews"][0].startswith("data:image/jpeg;base64,")

