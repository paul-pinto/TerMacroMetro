from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

EVENTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_events.json"
)

EVENT_RANKING_PATH = (
    PROJECT_ROOT
    / "reports"
    / "event_relevance_ranking.json"
)


router = APIRouter(
    prefix="/api/v2",
    tags=["TerMacroMetro v2"],
)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "El artefacto solicitado todavía "
                "no fue generado."
            ),
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "El artefacto generado contiene "
                "datos inválidos."
            ),
        ) from error


def dashboard_data() -> dict[str, Any]:
    payload = read_json(
        DASHBOARD_PATH
    )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail="El dashboard no es un objeto JSON.",
        )

    return payload


def limited_items(
    items: Any,
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    return [
        item
        for item in items[:limit]
        if isinstance(item, dict)
    ]


@router.get("/overview")
def get_overview() -> dict[str, Any]:
    dashboard = dashboard_data()

    snapshot = dashboard.get(
        "daily_snapshot",
        {},
    )

    temporal = dashboard.get(
        "temporal_quality",
        {},
    )

    event_analysis = dashboard.get(
        "event_analysis",
        {},
    )

    return {
        "version": "2.0",
        "snapshot": snapshot,
        "temporal_quality": temporal,
        "event_analysis": {
            "documents": event_analysis.get(
                "documents",
                0,
            ),
            "events": event_analysis.get(
                "events",
                0,
            ),
            "multi_source_events": (
                event_analysis.get(
                    "multi_source_events",
                    0,
                )
            ),
            "compression_ratio": (
                event_analysis.get(
                    "compression_ratio",
                    0.0,
                )
            ),
            "similarity_threshold": (
                event_analysis.get(
                    "similarity_threshold",
                    0.0,
                )
            ),
        },
        "operational_window": dashboard.get(
            "operational_window",
            {},
        ),
    }


@router.get("/temporal-quality")
def get_temporal_quality() -> dict[str, Any]:
    dashboard = dashboard_data()

    return {
        "temporal_quality": dashboard.get(
            "temporal_quality",
            {},
        ),
        "operational_window": dashboard.get(
            "operational_window",
            {},
        ),
    }


@router.get("/events")
def get_events(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    minimum_sources: int = Query(
        default=1,
        ge=1,
        le=50,
    ),
) -> dict[str, Any]:
    payload = read_json(
        EVENTS_PATH
    )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                "El reporte de eventos "
                "no es un objeto JSON."
            ),
        )

    items = payload.get(
        "items",
        [],
    )

    filtered = [
        item
        for item in items
        if isinstance(item, dict)
        and int(
            item.get(
                "unique_sources",
                0,
            )
        ) >= minimum_sources
    ]

    return {
        "documents": payload.get(
            "documents",
            0,
        ),
        "events": payload.get(
            "events",
            0,
        ),
        "multi_source_events": payload.get(
            "multi_source_events",
            0,
        ),
        "compression_ratio": payload.get(
            "compression_ratio",
            0.0,
        ),
        "returned": min(
            len(filtered),
            limit,
        ),
        "items": filtered[:limit],
    }


@router.get("/topics")
def get_topics(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    momentum: str | None = Query(
        default=None,
    ),
) -> dict[str, Any]:
    payload = read_json(
        EVENT_RANKING_PATH
    )

    items = payload.get(
        "topics",
        [],
    )

    if momentum:
        normalized = momentum.strip().lower()

        items = [
            item
            for item in items
            if isinstance(item, dict)
            and str(
                item.get(
                    "momentum",
                    "",
                )
            ).lower() == normalized
        ]

    return {
        "ranking_unit": payload.get(
            "ranking_unit",
            "economic_events",
        ),
        "window_hours": payload.get(
            "window_hours",
            72,
        ),
        "top_topic": payload.get(
            "top_topic",
        ),
        "items": limited_items(
            items,
            limit,
        ),
    }


@router.get("/indicators")
def get_indicators(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    momentum: str | None = Query(
        default=None,
    ),
) -> dict[str, Any]:
    payload = read_json(
        EVENT_RANKING_PATH
    )

    items = payload.get(
        "indicators",
        [],
    )

    if momentum:
        normalized = momentum.strip().lower()

        items = [
            item
            for item in items
            if isinstance(item, dict)
            and str(
                item.get(
                    "momentum",
                    "",
                )
            ).lower() == normalized
        ]

    return {
        "ranking_unit": payload.get(
            "ranking_unit",
            "economic_events",
        ),
        "window_hours": payload.get(
            "window_hours",
            72,
        ),
        "top_indicator": payload.get(
            "top_indicator",
        ),
        "items": limited_items(
            items,
            limit,
        ),
    }


@router.get("/entities")
def get_entities(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    momentum: str | None = Query(
        default=None,
    ),
) -> dict[str, Any]:
    payload = read_json(
        EVENT_RANKING_PATH
    )

    items = payload.get(
        "entities",
        [],
    )

    if momentum:
        normalized = momentum.strip().lower()

        items = [
            item
            for item in items
            if isinstance(item, dict)
            and str(
                item.get(
                    "momentum",
                    "",
                )
            ).lower() == normalized
        ]

    return {
        "ranking_unit": payload.get(
            "ranking_unit",
            "economic_events",
        ),
        "window_hours": payload.get(
            "window_hours",
            72,
        ),
        "top_entity": payload.get(
            "top_entity",
        ),
        "items": limited_items(
            items,
            limit,
        ),
    }


@router.get("/emerging-signals")
def get_emerging_signals(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
) -> dict[str, Any]:
    payload = read_json(
        EVENT_RANKING_PATH
    )

    result: dict[str, list[dict[str, Any]]] = {}

    for dimension in (
        "topics",
        "indicators",
        "entities",
    ):
        items = payload.get(
            dimension,
            [],
        )

        result[dimension] = [
            item
            for item in items
            if isinstance(item, dict)
            and item.get(
                "momentum"
            ) == "emergente"
        ][:limit]

    return {
        "window_hours": payload.get(
            "window_hours",
            72,
        ),
        "ranking_unit": payload.get(
            "ranking_unit",
            "economic_events",
        ),
        **result,
    }
