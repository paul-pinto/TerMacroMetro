from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.event_relevance import (
    build_event_rankings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "eventos_bolivia_operativos.csv"
)

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "event_relevance_ranking.json"
)

WINDOW_HOURS = 72


def eligible_leader(
    ranking: list[dict],
    *,
    minimum_events: int = 2,
    minimum_sources: int = 2,
) -> dict | None:
    for item in ranking:
        if (
            int(
                item.get(
                    "current_events",
                    0,
                )
            )
            >= minimum_events
            and int(
                item.get(
                    "unique_sources",
                    0,
                )
            )
            >= minimum_sources
        ):
            return item

    return ranking[0] if ranking else None


def main() -> None:
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {EVENTS_PATH}"
        )

    if not DASHBOARD_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {DASHBOARD_PATH}"
        )

    events_df = pd.read_csv(
        EVENTS_PATH
    )

    if events_df.empty:
        raise RuntimeError(
            "No existen eventos económicos."
        )

    rankings = build_event_rankings(
        events_df,
        previous_events=None,
        window_hours=WINDOW_HOURS,
        limit=10,
    )

    rankings["top_topic"] = (
        eligible_leader(
            rankings["topics"],
        )
    )

    rankings["top_indicator"] = (
        eligible_leader(
            rankings["indicators"],
        )
    )

    rankings["top_entity"] = (
        eligible_leader(
            rankings["entities"],
            minimum_events=1,
            minimum_sources=2,
        )
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            rankings,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dashboard = json.loads(
        DASHBOARD_PATH.read_text(
            encoding="utf-8"
        )
    )

    dashboard[
        "event_relevance_ranking"
    ] = rankings

    snapshot = dashboard.get(
        "daily_snapshot",
        {},
    )

    top_topic = rankings.get(
        "top_topic"
    )

    top_indicator = rankings.get(
        "top_indicator"
    )

    top_entity = rankings.get(
        "top_entity"
    )

    snapshot[
        "event_top_topic"
    ] = (
        top_topic.get("name")
        if top_topic
        else ""
    )

    snapshot[
        "event_top_indicator"
    ] = (
        top_indicator.get("name")
        if top_indicator
        else ""
    )

    snapshot[
        "event_top_entity"
    ] = (
        top_entity.get("name")
        if top_entity
        else ""
    )

    snapshot[
        "economic_events"
    ] = int(
        len(events_df)
    )

    snapshot[
        "multi_source_events"
    ] = int(
        (
            pd.to_numeric(
                events_df[
                    "unique_sources"
                ],
                errors="coerce",
            )
            .fillna(0)
            >= 2
        ).sum()
    )

    dashboard[
        "daily_snapshot"
    ] = snapshot

    DASHBOARD_PATH.write_text(
        json.dumps(
            dashboard,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Eventos económicos:",
        len(events_df),
    )

    print(
        "Tema por eventos:",
        (
            top_topic.get("name")
            if top_topic
            else "sin datos"
        ),
    )

    print(
        "Indicador por eventos:",
        (
            top_indicator.get("name")
            if top_indicator
            else "sin datos"
        ),
    )

    print(
        "Entidad por eventos:",
        (
            top_entity.get("name")
            if top_entity
            else "sin datos"
        ),
    )

    print(
        "Ranking:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
