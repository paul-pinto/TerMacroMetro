from __future__ import annotations

import os

os.environ["PULSOBO_DISABLE_TRANSFORMER"] = "1"

from fastapi.testclient import TestClient

from api.observatory import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] in {
            "ok",
            "degraded",
        }

        assert "model_ready" in data


def test_models_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/models")

        assert response.status_code == 200

        data = response.json()

        assert "classic_model" in data
        assert "transformer_model" in data
        assert "topic_model" in data


def test_dashboard_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/dashboard")

        assert response.status_code == 200

        data = response.json()

        assert data["total_documents"] > 0
        assert "pulsobo_index" in data
        assert "topics" in data
        assert "indicators" in data


def test_analyze_endpoint() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "texto": (
                    "El BCB informó que la escasez de divisas "
                    "afecta las reservas internacionales."
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "resultado_consolidado" in data
        assert "tema" in data
        assert "contexto_boliviano" in data


def test_analyze_rejects_short_text() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze",
            json={
                "texto": "Muy corto"
            },
        )

        assert response.status_code == 422
