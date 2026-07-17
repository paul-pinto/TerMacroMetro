from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.temporal_quality import (
    TemporalPolicy,
    build_operational_corpus,
    temporal_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYZED_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_analizadas.csv"
)

OPERATIONAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_operativas.csv"
)

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

RECENT_WINDOW_HOURS = 72


def safe_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    return [
        column
        for column in columns
        if column in df.columns
    ]


def main() -> None:
    if not ANALYZED_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {ANALYZED_PATH}"
        )

    df = pd.read_csv(ANALYZED_PATH)

    if df.empty:
        raise ValueError(
            "El corpus analizado está vacío."
        )

    policy = TemporalPolicy(
        recent_window_hours=(
            RECENT_WINDOW_HOURS
        )
    )

    enriched, operational = (
        build_operational_corpus(
            df,
            policy=policy,
        )
    )

    enriched.to_csv(
        ANALYZED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    operational.to_csv(
        OPERATIONAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    audit = temporal_audit(
        enriched,
        operational,
        policy=policy,
    )

    if DASHBOARD_PATH.exists():
        dashboard = json.loads(
            DASHBOARD_PATH.read_text(
                encoding="utf-8"
            )
        )
    else:
        dashboard = {}

    news_columns = safe_columns(
        operational,
        [
            "titulo",
            "fuente",
            "fecha",
            "published_at",
            "effective_at",
            "age_hours",
            "tema",
            "tono_consolidado",
            "stress_score",
            "stress_nivel",
            "url",
        ],
    )

    top_news = (
        operational[
            news_columns
        ]
        .head(30)
        .where(
            pd.notna(
                operational[
                    news_columns
                ]
            ),
            None,
        )
        .to_dict(
            orient="records"
        )
    )

    stress_columns = safe_columns(
        operational,
        [
            "titulo",
            "fuente",
            "fecha",
            "published_at",
            "effective_at",
            "age_hours",
            "stress_score",
            "stress_nivel",
            "tema",
            "tono_consolidado",
            "url",
        ],
    )

    highest_stress_df = (
        operational
        .sort_values(
            [
                "stress_score",
                "effective_at",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        [
            stress_columns
        ]
        .head(10)
    )

    highest_stress_df = (
        operational
        .sort_values(
            [
                "stress_score",
                "effective_at",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        [
            stress_columns
        ]
        .head(10)
    )

    highest_stress = (
        highest_stress_df
        .where(
            pd.notna(
                highest_stress_df
            ),
            None,
        )
        .to_dict(
            orient="records"
        )
    )

    dashboard["news"] = top_news
    dashboard[
        "highest_stress_news"
    ] = highest_stress

    dashboard["temporal_quality"] = audit
    dashboard["operational_window"] = {
        "hours": RECENT_WINDOW_HOURS,
        "description": (
            "Indicadores actuales calculados "
            "sobre noticias publicadas o "
            "recolectadas durante las últimas "
            f"{RECENT_WINDOW_HOURS} horas."
        ),
    }

    DASHBOARD_PATH.write_text(
        json.dumps(
            dashboard,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Corpus histórico:",
        audit["historical_documents"],
    )

    print(
        "Corpus operativo:",
        audit["operational_documents"],
    )

    print(
        "Noticias excluidas:",
        audit["excluded_documents"],
    )

    print(
        "Ventana operativa:",
        f"{RECENT_WINDOW_HOURS} horas",
    )

    print(
        "Archivo operativo:",
        OPERATIONAL_PATH,
    )


if __name__ == "__main__":
    main()


