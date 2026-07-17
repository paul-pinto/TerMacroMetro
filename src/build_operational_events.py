from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.event_clustering import (
    EventClusteringConfig,
    cluster_news,
    summarize_events,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_operativas.csv"
)

CLUSTERED_NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_eventos.csv"
)

EVENTS_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "eventos_bolivia_operativos.csv"
)

EVENTS_JSON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_events.json"
)

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

SIMILARITY_THRESHOLD = 0.48


def dataframe_records(
    df: pd.DataFrame,
) -> list[dict]:
    clean = df.copy()

    clean = clean.where(
        pd.notna(clean),
        None,
    )

    return clean.to_dict(
        orient="records"
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {INPUT_PATH}"
        )

    news_df = pd.read_csv(
        INPUT_PATH
    )

    if news_df.empty:
        raise RuntimeError(
            "El corpus operativo está vacío."
        )

    clustered_df = cluster_news(
        news_df,
        config=EventClusteringConfig(
            similarity_threshold=(
                SIMILARITY_THRESHOLD
            ),
            max_days_apart=3,
        ),
    )

    events_df = summarize_events(
        clustered_df
    )

    CLUSTERED_NEWS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVENTS_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clustered_df.to_csv(
        CLUSTERED_NEWS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    events_df.to_csv(
        EVENTS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    events_payload = {
        "documents": int(
            len(clustered_df)
        ),
        "events": int(
            len(events_df)
        ),
        "compression_ratio": round(
            (
                1
                - len(events_df)
                / len(clustered_df)
            )
            * 100,
            2,
        ),
        "similarity_threshold": (
            SIMILARITY_THRESHOLD
        ),
        "multi_source_events": int(
            (
                events_df[
                    "unique_sources"
                ]
                >= 2
            ).sum()
        ),
        "items": dataframe_records(
            events_df
        ),
    }

    EVENTS_JSON_PATH.write_text(
        json.dumps(
            events_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if DASHBOARD_PATH.exists():
        dashboard = json.loads(
            DASHBOARD_PATH.read_text(
                encoding="utf-8"
            )
        )
    else:
        dashboard = {}

    dashboard[
        "event_analysis"
    ] = events_payload

    dashboard[
        "top_economic_events"
    ] = dataframe_records(
        events_df.head(10)
    )

    DASHBOARD_PATH.write_text(
        json.dumps(
            dashboard,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Noticias operativas:",
        len(clustered_df),
    )

    print(
        "Eventos económicos:",
        len(events_df),
    )

    print(
        "Eventos multifuente:",
        events_payload[
            "multi_source_events"
        ],
    )

    print(
        "Compresión:",
        f"{events_payload['compression_ratio']}%",
    )

    print(
        "Umbral de similitud:",
        SIMILARITY_THRESHOLD,
    )

    print(
        "Eventos CSV:",
        EVENTS_CSV_PATH,
    )

    print(
        "Eventos JSON:",
        EVENTS_JSON_PATH,
    )


if __name__ == "__main__":
    main()
