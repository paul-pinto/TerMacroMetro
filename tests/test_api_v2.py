from fastapi.testclient import TestClient

from api.observatory import app


client = TestClient(app)


def test_v2_overview() -> None:
    response = client.get(
        "/api/v2/overview"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["version"] == "2.0"
    assert "snapshot" in payload
    assert "temporal_quality" in payload
    assert "event_analysis" in payload


def test_v2_events() -> None:
    response = client.get(
        "/api/v2/events?limit=5"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "items" in payload
    assert len(payload["items"]) <= 5


def test_v2_topics() -> None:
    response = client.get(
        "/api/v2/topics?limit=5"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "items" in payload
    assert len(payload["items"]) <= 5


def test_v2_indicators() -> None:
    response = client.get(
        "/api/v2/indicators?limit=5"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "items" in payload
    assert len(payload["items"]) <= 5


def test_v2_entities() -> None:
    response = client.get(
        "/api/v2/entities?limit=5"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "items" in payload
    assert len(payload["items"]) <= 5


def test_v2_emerging_signals() -> None:
    response = client.get(
        "/api/v2/emerging-signals"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "topics" in payload
    assert "indicators" in payload
    assert "entities" in payload


def test_v2_event_source_filter() -> None:
    response = client.get(
        "/api/v2/events"
        "?minimum_sources=2"
    )

    assert response.status_code == 200

    for event in response.json()["items"]:
        assert event["unique_sources"] >= 2
