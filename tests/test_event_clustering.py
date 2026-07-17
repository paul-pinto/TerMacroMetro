from __future__ import annotations

import json

import pandas as pd

from src.event_clustering import (
    EventClusteringConfig,
    cluster_news,
    summarize_events,
)


def build_test_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "titulo": (
                    "ASFI anuncia devolución "
                    "de depósitos en dólares"
                ),
                "texto": (
                    "La autoridad informó que "
                    "los bancos devolverán "
                    "depósitos en dólares."
                ),
                "fuente": "ABI",
                "fecha": (
                    "2026-07-16T10:00:00Z"
                ),
                "effective_at": (
                    "2026-07-16T10:00:00Z"
                ),
                "stress_score": 40,
                "tema": "Sistema financiero",
                "indicadores": json.dumps(
                    [
                        "sistema financiero",
                        "tipo de cambio",
                    ]
                ),
                "entidades": json.dumps(
                    ["ASFI"]
                ),
                "url": "https://example.com/1",
                "tono_consolidado": "neutral",
            },
            {
                "titulo": (
                    "Comienza devolución de "
                    "ahorros en dólares"
                ),
                "texto": (
                    "Los bancos iniciaron la "
                    "devolución de depósitos "
                    "en moneda estadounidense."
                ),
                "fuente": "Unitel",
                "fecha": (
                    "2026-07-16T11:00:00Z"
                ),
                "effective_at": (
                    "2026-07-16T11:00:00Z"
                ),
                "stress_score": 45,
                "tema": "Sistema financiero",
                "indicadores": json.dumps(
                    ["sistema financiero"]
                ),
                "entidades": json.dumps(
                    ["ASFI"]
                ),
                "url": "https://example.com/2",
                "tono_consolidado": "neutral",
            },
            {
                "titulo": (
                    "Productores de trigo "
                    "solicitan nuevas tecnologías"
                ),
                "texto": (
                    "El sector agrícola busca "
                    "incrementar la producción "
                    "nacional de trigo."
                ),
                "fuente": "Economy",
                "fecha": (
                    "2026-07-16T12:00:00Z"
                ),
                "effective_at": (
                    "2026-07-16T12:00:00Z"
                ),
                "stress_score": 25,
                "tema": "Sector productivo",
                "indicadores": json.dumps(
                    ["producción"]
                ),
                "entidades": json.dumps(
                    []
                ),
                "url": "https://example.com/3",
                "tono_consolidado": "favorable",
            },
        ]
    )


def test_similar_news_are_clustered() -> None:
    df = build_test_dataframe()

    result = cluster_news(
        df,
        config=EventClusteringConfig(
            similarity_threshold=0.20,
        ),
    )

    assert (
        result.loc[0, "event_id"]
        == result.loc[1, "event_id"]
    )

    assert (
        result.loc[0, "event_size"]
        == 2
    )

    assert (
        result.loc[2, "event_size"]
        == 1
    )


def test_event_summary_counts_sources() -> None:
    df = build_test_dataframe()

    clustered = cluster_news(
        df,
        config=EventClusteringConfig(
            similarity_threshold=0.20,
        ),
    )

    events = summarize_events(
        clustered
    )

    financial = events[
        events["dominant_topic"]
        == "Sistema financiero"
    ].iloc[0]

    assert financial["documents"] == 2
    assert financial["unique_sources"] == 2
    assert (
        financial["dominant_entity"]
        == "ASFI"
    )


def test_unrelated_news_remain_separate() -> None:
    df = build_test_dataframe()

    result = cluster_news(
        df,
        config=EventClusteringConfig(
            similarity_threshold=0.60,
        ),
    )

    assert result["event_id"].nunique() >= 2
