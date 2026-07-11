from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYZED_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_analizadas.csv"
)

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "history"
)

HISTORY_CSV_PATH = (
    HISTORY_DIR
    / "pulsobo_daily.csv"
)

HISTORY_JSON_PATH = (
    HISTORY_DIR
    / "pulsobo_daily.json"
)


OPTIMISM_LEXICON = {
    "inversión": 4,
    "inversion": 4,
    "crecimiento": 4,
    "recuperación": 5,
    "recuperacion": 5,
    "reactivación": 5,
    "reactivacion": 5,
    "mejora": 3,
    "mejoró": 3,
    "mejoro": 3,
    "empleo": 3,
    "exportaciones": 3,
    "superávit": 5,
    "superavit": 5,
    "financiamiento": 3,
    "cooperación": 2,
    "cooperacion": 2,
    "desarrollo": 2,
    "producción": 3,
    "produccion": 3,
    "industrialización": 4,
    "industrializacion": 4,
    "competitividad": 3,
    "estabilidad": 4,
    "abastecimiento": 2,
    "garantiza": 3,
    "fortalece": 3,
    "impulsa": 3,
    "incremento": 2,
    "aumento de ingresos": 4,
    "nuevos empleos": 5,
    "capital": 2,
}


def normalize_text(value: object) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or "").lower(),
    ).strip()


def count_optimism_signals(
    text: str,
) -> tuple[int, dict[str, int]]:
    normalized = normalize_text(text)

    weighted_score = 0
    detected: dict[str, int] = {}

    for term, weight in OPTIMISM_LEXICON.items():
        occurrences = len(
            re.findall(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                normalized,
            )
        )

        if occurrences:
            detected[term] = occurrences
            weighted_score += occurrences * weight

    return weighted_score, detected


def safe_json_list(
    value: object,
) -> list[str]:
    try:
        result = json.loads(str(value))

        if isinstance(result, list):
            return [
                str(item)
                for item in result
                if str(item).strip()
            ]

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        pass

    return []


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return round(
        numerator / denominator * 100,
        2,
    )


def build_optimism_index(
    lexical_average: float,
    favorable_percentage: float,
    low_stress_percentage: float,
) -> dict[str, Any]:
    lexical_component = min(
        lexical_average * 6.0,
        100.0,
    )

    score = (
        lexical_component * 0.50
        + favorable_percentage * 0.35
        + low_stress_percentage * 0.15
    )

    score = round(
        max(0.0, min(100.0, score)),
        2,
    )

    if score >= 70:
        level = "alto"
        interpretation = (
            "El corpus presenta una concentración elevada "
            "de señales de recuperación, inversión y crecimiento."
        )
    elif score >= 45:
        level = "moderado"
        interpretation = (
            "Existen señales favorables, aunque conviven "
            "con presión e incertidumbre económica."
        )
    elif score >= 25:
        level = "bajo"
        interpretation = (
            "Las señales de optimismo son limitadas dentro "
            "del corpus económico analizado."
        )
    else:
        level = "muy bajo"
        interpretation = (
            "Predominan textos sin señales claras de mejora "
            "o recuperación económica."
        )

    return {
        "score": score,
        "level": level,
        "interpretation": interpretation,
        "components": {
            "lexical_component": round(
                lexical_component,
                2,
            ),
            "favorable_percentage": round(
                favorable_percentage,
                2,
            ),
            "low_stress_percentage": round(
                low_stress_percentage,
                2,
            ),
        },
        "weights": {
            "lexical_component": 0.50,
            "favorable_percentage": 0.35,
            "low_stress_percentage": 0.15,
        },
        "disclaimer": (
            "Indicador experimental basado en señales lingüísticas "
            "del corpus. No representa una medición macroeconómica oficial."
        ),
    }


def top_counter(
    counter: Counter[str],
    limit: int = 5,
) -> dict[str, int]:
    return dict(
        counter.most_common(limit)
    )


def main() -> None:
    if not ANALYZED_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {ANALYZED_PATH}"
        )

    if not DASHBOARD_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {DASHBOARD_PATH}"
        )

    df = pd.read_csv(
        ANALYZED_PATH
    )

    required = {
        "titulo",
        "texto",
        "tono_consolidado",
        "stress_score",
        "stress_nivel",
        "tema",
        "entidades",
        "indicadores",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas: {sorted(missing)}"
        )

    total = len(df)

    if total == 0:
        raise RuntimeError(
            "El corpus analizado está vacío."
        )

    optimism_scores: list[int] = []
    optimism_terms: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    entities: Counter[str] = Counter()
    indicators: Counter[str] = Counter()

    for _, row in df.iterrows():
        document = (
            f"{row['titulo']}. {row['texto']}"
        )

        lexical_score, detected = (
            count_optimism_signals(document)
        )

        optimism_scores.append(
            lexical_score
        )

        optimism_terms.update(
            detected
        )

        topic = str(
            row.get("tema", "")
        ).strip()

        if topic:
            topics[topic] += 1

        entities.update(
            safe_json_list(
                row.get("entidades", "[]")
            )
        )

        indicators.update(
            safe_json_list(
                row.get("indicadores", "[]")
            )
        )

    favorable_count = int(
        (
            df["tono_consolidado"]
            == "favorable"
        ).sum()
    )

    unfavorable_count = int(
        (
            df["tono_consolidado"]
            == "desfavorable"
        ).sum()
    )

    neutral_count = int(
        (
            df["tono_consolidado"]
            == "neutral"
        ).sum()
    )

    low_stress_count = int(
        df["stress_nivel"].isin(
            ["bajo", "moderado"]
        ).sum()
    )

    favorable_percentage = percentage(
        favorable_count,
        total,
    )

    unfavorable_percentage = percentage(
        unfavorable_count,
        total,
    )

    neutral_percentage = percentage(
        neutral_count,
        total,
    )

    low_stress_percentage = percentage(
        low_stress_count,
        total,
    )

    lexical_average = round(
        sum(optimism_scores)
        / len(optimism_scores),
        2,
    )

    optimism_index = build_optimism_index(
        lexical_average=lexical_average,
        favorable_percentage=favorable_percentage,
        low_stress_percentage=low_stress_percentage,
    )

    with DASHBOARD_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dashboard = json.load(file)

    dashboard["optimism_index"] = (
        optimism_index
    )

    dashboard["optimism_signals"] = (
        top_counter(
            optimism_terms,
            limit=15,
        )
    )

    generated_at = datetime.now(
        timezone.utc
    )

    snapshot = {
        "date": generated_at.date().isoformat(),
        "generated_at": generated_at.isoformat(),
        "documents": total,
        "pulsobo_index": float(
            dashboard.get(
                "pulsobo_index",
                {},
            ).get(
                "score",
                0.0,
            )
        ),
        "pulsobo_level": str(
            dashboard.get(
                "pulsobo_index",
                {},
            ).get(
                "level",
                "sin datos",
            )
        ),
        "optimism_index": optimism_index[
            "score"
        ],
        "optimism_level": optimism_index[
            "level"
        ],
        "stress_average": round(
            float(
                df["stress_score"].mean()
            ),
            2,
        ),
        "favorable_percentage": (
            favorable_percentage
        ),
        "neutral_percentage": (
            neutral_percentage
        ),
        "unfavorable_percentage": (
            unfavorable_percentage
        ),
        "top_topic": (
            topics.most_common(1)[0][0]
            if topics
            else ""
        ),
        "top_entity": (
            entities.most_common(1)[0][0]
            if entities
            else ""
        ),
        "top_indicator": (
            indicators.most_common(1)[0][0]
            if indicators
            else ""
        ),
        "top_topics": top_counter(topics),
        "top_entities": top_counter(entities),
        "top_indicators": top_counter(indicators),
        "top_optimism_signals": top_counter(
            optimism_terms
        ),
    }

    dashboard["daily_snapshot"] = snapshot

    with DASHBOARD_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dashboard,
            file,
            ensure_ascii=False,
            indent=2,
        )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_row = {
        key: value
        for key, value in snapshot.items()
        if not isinstance(
            value,
            dict,
        )
    }

    if HISTORY_CSV_PATH.exists():
        history_df = pd.read_csv(
            HISTORY_CSV_PATH
        )
    else:
        history_df = pd.DataFrame()

    history_df = pd.concat(
        [
            history_df,
            pd.DataFrame(
                [history_row]
            ),
        ],
        ignore_index=True,
    )

    history_df = (
        history_df
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
    )

    history_df.to_csv(
        HISTORY_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    history_records = (
        history_df
        .where(
            pd.notna(history_df),
            None,
        )
        .to_dict(orient="records")
    )

    with HISTORY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Documentos:",
        total,
    )

    print(
        "MacroScore:",
        snapshot["pulsobo_index"],
    )

    print(
        "Optimismo:",
        optimism_index["score"],
        f"({optimism_index['level']})",
    )

    print(
        "Tensión promedio:",
        snapshot["stress_average"],
    )

    print(
        "Favorable:",
        favorable_percentage,
    )

    print(
        "Desfavorable:",
        unfavorable_percentage,
    )

    print(
        "Histórico:",
        HISTORY_CSV_PATH,
    )


if __name__ == "__main__":
    main()

