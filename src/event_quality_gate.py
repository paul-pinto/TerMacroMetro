from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPERATIONAL_NEWS_PATH = (
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

EVENT_RANKING_PATH = (
    PROJECT_ROOT
    / "reports"
    / "event_relevance_ranking.json"
)

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

REQUIRED_CLUSTERED_COLUMNS = {
    "event_id",
    "event_size",
    "titulo",
    "fuente",
}

REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "event_title",
    "documents",
    "unique_sources",
    "sources",
    "effective_at",
    "average_stress",
    "maximum_stress",
    "dominant_topic",
    "dominant_indicator",
    "dominant_entity",
    "representative_url",
}

MAX_SINGLE_EVENT_SHARE = 0.50
MIN_EVENT_RATIO = 0.20


def fail(message: str) -> None:
    print(
        f"[FAIL] {message}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def require_file(path: Path) -> None:
    if not path.exists():
        fail(
            f"No existe el artefacto requerido: {path}"
        )

    if path.stat().st_size == 0:
        fail(
            f"El artefacto está vacío: {path}"
        )


def parse_json(path: Path) -> dict:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as error:
        fail(
            f"JSON inválido en {path}: {error}"
        )

    if not isinstance(payload, dict):
        fail(
            f"Se esperaba un objeto JSON en {path}"
        )

    return payload


def main() -> None:
    for path in (
        OPERATIONAL_NEWS_PATH,
        CLUSTERED_NEWS_PATH,
        EVENTS_CSV_PATH,
        EVENTS_JSON_PATH,
        EVENT_RANKING_PATH,
        DASHBOARD_PATH,
    ):
        require_file(path)

    operational_df = pd.read_csv(
        OPERATIONAL_NEWS_PATH
    )

    clustered_df = pd.read_csv(
        CLUSTERED_NEWS_PATH
    )

    events_df = pd.read_csv(
        EVENTS_CSV_PATH
    )

    if operational_df.empty:
        fail(
            "El corpus operativo no contiene noticias."
        )

    if clustered_df.empty:
        fail(
            "El corpus agrupado no contiene noticias."
        )

    if events_df.empty:
        fail(
            "No se generaron eventos económicos."
        )

    clustered_missing = (
        REQUIRED_CLUSTERED_COLUMNS
        - set(clustered_df.columns)
    )

    if clustered_missing:
        fail(
            "Faltan columnas en noticias agrupadas: "
            f"{sorted(clustered_missing)}"
        )

    event_missing = (
        REQUIRED_EVENT_COLUMNS
        - set(events_df.columns)
    )

    if event_missing:
        fail(
            "Faltan columnas en eventos: "
            f"{sorted(event_missing)}"
        )

    operational_count = len(
        operational_df
    )

    clustered_count = len(
        clustered_df
    )

    event_count = len(
        events_df
    )

    if clustered_count != operational_count:
        fail(
            "El clustering perdió o duplicó noticias: "
            f"operativas={operational_count}, "
            f"agrupadas={clustered_count}"
        )

    if clustered_df["event_id"].isna().any():
        fail(
            "Existen noticias sin event_id."
        )

    if (
        clustered_df["event_id"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        fail(
            "Existen noticias con event_id vacío."
        )

    event_document_sum = int(
        pd.to_numeric(
            events_df["documents"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    if event_document_sum != operational_count:
        fail(
            "La suma de documentos por evento no coincide "
            "con el corpus operativo: "
            f"eventos={event_document_sum}, "
            f"operativas={operational_count}"
        )

    unique_event_ids = int(
        clustered_df["event_id"].nunique()
    )

    if unique_event_ids != event_count:
        fail(
            "El número de event_id únicos no coincide "
            "con el resumen de eventos: "
            f"ids={unique_event_ids}, "
            f"eventos={event_count}"
        )

    if events_df["event_id"].duplicated().any():
        fail(
            "Existen event_id duplicados "
            "en el resumen de eventos."
        )

    empty_titles = (
        events_df["event_title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_titles.any():
        fail(
            "Existen eventos sin título representativo."
        )

    event_sizes = pd.to_numeric(
        events_df["documents"],
        errors="coerce",
    ).fillna(0)

    if (event_sizes <= 0).any():
        fail(
            "Existen eventos con cero documentos."
        )

    largest_event = int(
        event_sizes.max()
    )

    largest_event_share = (
        largest_event
        / operational_count
    )

    if (
        operational_count >= 10
        and largest_event_share
        > MAX_SINGLE_EVENT_SHARE
    ):
        fail(
            "Un solo evento concentra demasiadas noticias: "
            f"{largest_event_share * 100:.2f}%"
        )

    event_ratio = (
        event_count
        / operational_count
    )

    if (
        operational_count >= 10
        and event_ratio < MIN_EVENT_RATIO
    ):
        fail(
            "El clustering parece demasiado agresivo: "
            f"{event_count} eventos para "
            f"{operational_count} noticias."
        )

    unique_sources = pd.to_numeric(
        events_df["unique_sources"],
        errors="coerce",
    ).fillna(0)

    if (unique_sources <= 0).any():
        fail(
            "Existen eventos sin fuentes asociadas."
        )

    event_payload = parse_json(
        EVENTS_JSON_PATH
    )

    if int(
        event_payload.get(
            "documents",
            -1,
        )
    ) != operational_count:
        fail(
            "bolivia_events.json contiene un número "
            "incorrecto de documentos."
        )

    if int(
        event_payload.get(
            "events",
            -1,
        )
    ) != event_count:
        fail(
            "bolivia_events.json contiene un número "
            "incorrecto de eventos."
        )

    ranking_payload = parse_json(
        EVENT_RANKING_PATH
    )

    for dimension in (
        "topics",
        "indicators",
        "entities",
    ):
        value = ranking_payload.get(
            dimension
        )

        if not isinstance(value, list):
            fail(
                f"El ranking de {dimension} "
                "no es una lista."
            )

    dashboard = parse_json(
        DASHBOARD_PATH
    )

    if "event_analysis" not in dashboard:
        fail(
            "El dashboard no contiene event_analysis."
        )

    if (
        "event_relevance_ranking"
        not in dashboard
    ):
        fail(
            "El dashboard no contiene "
            "event_relevance_ranking."
        )

    snapshot = dashboard.get(
        "daily_snapshot",
        {}
    )

    if not isinstance(snapshot, dict):
        fail(
            "daily_snapshot no es un objeto."
        )

    if int(
        snapshot.get(
            "economic_events",
            -1,
        )
    ) != event_count:
        fail(
            "daily_snapshot no contiene el número "
            "correcto de eventos económicos."
        )

    multi_source_events = int(
        (
            unique_sources >= 2
        ).sum()
    )

    print(
        "[OK] Calidad de eventos aprobada."
    )

    print(
        "Noticias operativas:",
        operational_count,
    )

    print(
        "Eventos económicos:",
        event_count,
    )

    print(
        "Eventos multifuente:",
        multi_source_events,
    )

    print(
        "Mayor evento:",
        largest_event,
        "noticias",
    )

    print(
        "Concentración máxima:",
        f"{largest_event_share * 100:.2f}%",
    )

    print(
        "Relación eventos/noticias:",
        f"{event_ratio:.2f}",
    )


if __name__ == "__main__":
    main()
