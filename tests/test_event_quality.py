from __future__ import annotations

import json

import pandas as pd

from src.event_clustering import (
    EventClusteringConfig,
    cluster_news,
    summarize_events,
)


def sample_news() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "titulo": (
                    "ASFI inicia devolución "
                    "de depósitos en dólares"
                ),
                "texto": (
                    "Los bancos iniciaron la devolución "
                    "de ahorros en moneda extranjera."
                ),
                "fuente": "ABI",
                "effective_at": (
                    "2026-07-16T10:00:00Z"
                ),
                "stress_score": 42,
                "tema": "Sistema financiero",
                "indicadores": json.dumps(
                    ["sistema financiero"]
                ),
                "entidades": json.dumps(
                    ["ASFI"]
                ),
                "url": "https://example.com/1",
                "tono_consolidado": "neutral",
            },
            {
                "titulo": (
                    "Bancos devuelven ahorros "
                    "retenidos en dólares"
                ),
                "texto": (
                    "Comenzó la devolución de depósitos "
                    "en moneda estadounidense."
                ),
                "fuente": "Unitel",
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
                    "Productores aumentan "
                    "la siembra de trigo"
                ),
                "texto": (
                    "El sector agrícola busca elevar "
                    "la producción nacional."
                ),
                "fuente": "Economy",
                "effective_at": (
                    "2026-07-16T12:00:00Z"
                ),
                "stress_score": 20,
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


def test_all_documents_receive_event_id() -> None:
    df = sample_news()

    clustered = cluster_news(
        df,
        config=EventClusteringConfig(
            similarity_threshold=0.20,
        ),
    )

    assert len(clustered) == len(df)
    assert clustered[
        "event_id"
    ].notna().all()

    assert (
        clustered["event_id"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )


def test_event_document_sum_is_consistent() -> None:
    df = sample_news()

    clustered = cluster_news(
        df,
        config=EventClusteringConfig(
            similarity_threshold=0.20,
        ),
    )

    events = summarize_events(
        clustered
    )

    assert int(
        events["documents"].sum()
    ) == len(df)

    assert events[
        "event_id"
    ].is_unique


def test_event_summary_has_sources() -> None:
    df = sample_news()

    clustered = cluster_news(
        df,
        config=EventClusteringConfig(
            similarity_threshold=0.20,
        ),
    )

    events = summarize_events(
        clustered
    )

    assert (
        events["unique_sources"] >= 1
    ).all()

    assert (
        events["event_title"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )
