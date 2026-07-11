from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

from src.inference import EconomicAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_analizadas.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)


def percentage_counts(
    counter: Counter[str],
    total: int,
) -> dict[str, dict[str, int | float]]:
    return {
        key: {
            "count": int(value),
            "percentage": round(
                value / total * 100,
                2,
            ) if total else 0.0,
        }
        for key, value in counter.most_common()
    }


def calcular_pulsobo_index(
    stress_average: float,
    unfavorable_percentage: float,
    high_stress_percentage: float,
    disagreement_percentage: float,
) -> dict[str, Any]:
    """
    Índice textual experimental de 0 a 100.

    No mide directamente el estado real de la economía.
    Resume la intensidad negativa observada en el corpus.
    """
    score = (
        stress_average * 0.50
        + unfavorable_percentage * 0.30
        + high_stress_percentage * 0.10
        + disagreement_percentage * 0.10
    )

    score = round(
        max(0.0, min(100.0, score)),
        2,
    )

    if score >= 75:
        level = "crítico"
        interpretation = (
            "Alta concentración de noticias desfavorables "
            "y señales textuales de tensión económica."
        )
    elif score >= 60:
        level = "alto"
        interpretation = (
            "El corpus refleja una presión económica elevada."
        )
    elif score >= 40:
        level = "moderado"
        interpretation = (
            "Existen señales mixtas con tensión económica relevante."
        )
    else:
        level = "bajo"
        interpretation = (
            "La tensión textual agregada del corpus es reducida."
        )

    return {
        "score": score,
        "level": level,
        "interpretation": interpretation,
        "components": {
            "stress_average": round(stress_average, 2),
            "unfavorable_percentage": round(
                unfavorable_percentage,
                2,
            ),
            "high_stress_percentage": round(
                high_stress_percentage,
                2,
            ),
            "model_disagreement_percentage": round(
                disagreement_percentage,
                2,
            ),
        },
        "weights": {
            "stress_average": 0.50,
            "unfavorable_percentage": 0.30,
            "high_stress_percentage": 0.10,
            "model_disagreement_percentage": 0.10,
        },
        "disclaimer": (
            "Indicador experimental basado en el lenguaje de las "
            "noticias analizadas. No constituye una medición "
            "macroeconómica oficial ni una recomendación financiera."
        ),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No existe el corpus: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "titulo",
        "texto",
        "fuente",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas: {sorted(missing)}"
        )

    df = (
        df
        .dropna(
            subset=[
                "titulo",
                "texto",
                "fuente",
            ]
        )
        .reset_index(drop=True)
    )

    analyzer = EconomicAnalyzer(
        cargar_transformer=True,
    )

    rows: list[dict[str, Any]] = []

    tones: Counter[str] = Counter()
    nb_tones: Counter[str] = Counter()
    transformer_tones: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    entities: Counter[str] = Counter()
    indicators: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    stress_levels: Counter[str] = Counter()

    stress_scores: list[float] = []
    agreements = 0
    disagreements = 0

    total = len(df)

    for index, row in df.iterrows():
        titulo = str(row["titulo"]).strip()
        texto = str(row["texto"]).strip()
        documento = f"{titulo}. {texto}"

        result = analyzer.analizar(documento)

        consolidated = result["resultado_consolidado"]
        nb = result["modelos"]["naive_bayes"]
        transformer = result["modelos"]["transformer"]
        topic = result["tema"]
        context = result["contexto_boliviano"]
        stress = context["stress_lexico"]

        tone = consolidated["tono_informativo"]

        tones[tone] += 1
        nb_tones[nb["tono_informativo"]] += 1
        transformer_tones[
            transformer["tono_informativo"]
        ] += 1

        topics[topic["nombre"]] += 1
        sources[str(row["fuente"])] += 1
        stress_levels[stress["nivel"]] += 1

        entities.update(
            context["entidades"]
        )

        indicators.update(
            context["indicadores"]
        )

        stress_scores.append(
            float(stress["score"])
        )

        agreement = consolidated.get(
            "coincidencia_modelos"
        )

        if agreement is True:
            agreements += 1
        elif agreement is False:
            disagreements += 1

        rows.append(
            {
                **row.to_dict(),
                "tono_consolidado": tone,
                "confianza_consolidada": consolidated[
                    "confianza"
                ],
                "modelo_seleccionado": consolidated[
                    "modelo_seleccionado"
                ],
                "coincidencia_modelos": agreement,
                "tono_naive_bayes": nb[
                    "tono_informativo"
                ],
                "confianza_naive_bayes": nb[
                    "confianza"
                ],
                "tono_transformer": transformer[
                    "tono_informativo"
                ],
                "confianza_transformer": transformer[
                    "confianza"
                ],
                "topic_id": topic["topic_id"],
                "tema": topic["nombre"],
                "confianza_tema": topic[
                    "confianza"
                ],
                "entidades": json.dumps(
                    context["entidades"],
                    ensure_ascii=False,
                ),
                "indicadores": json.dumps(
                    context["indicadores"],
                    ensure_ascii=False,
                ),
                "stress_score": stress["score"],
                "stress_nivel": stress["nivel"],
                "terminos_riesgo": json.dumps(
                    stress["riesgo_detectado"],
                    ensure_ascii=False,
                ),
                "terminos_alivio": json.dumps(
                    stress["alivio_detectado"],
                    ensure_ascii=False,
                ),
            }
        )

        print(
            f"Procesadas: {index + 1}/{total}",
            end="\r",
        )

    print()

    output_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    highest_stress = (
        output_df
        .sort_values(
            "stress_score",
            ascending=False,
        )
        [
            [
                "titulo",
                "fuente",
                "stress_score",
                "stress_nivel",
                "tema",
                "tono_consolidado",
                "url",
            ]
        ]
        .head(10)
        .to_dict(orient="records")
    )

    news = (
        output_df
        [
            [
                "titulo",
                "fuente",
                "tema",
                "tono_consolidado",
                "stress_score",
                "url",
            ]
        ]
        .head(30)
        .to_dict(orient="records")
    )

    stress_average = round(
        mean(stress_scores),
        2,
    )

    unfavorable_percentage = (
        tones.get("desfavorable", 0)
        / total
        * 100
        if total
        else 0.0
    )

    high_stress_count = (
        stress_levels.get("alto", 0)
        + stress_levels.get("crítico", 0)
    )

    high_stress_percentage = (
        high_stress_count
        / total
        * 100
        if total
        else 0.0
    )

    disagreement_percentage = (
        disagreements
        / total
        * 100
        if total
        else 0.0
    )

    pulsobo_index = calcular_pulsobo_index(
        stress_average=stress_average,
        unfavorable_percentage=unfavorable_percentage,
        high_stress_percentage=high_stress_percentage,
        disagreement_percentage=disagreement_percentage,
    )

    summary = {
        "total_documents": total,
        "pulsobo_index": pulsobo_index,
        "sources": percentage_counts(
            sources,
            total,
        ),
        "tones": percentage_counts(
            tones,
            total,
        ),
        "naive_bayes_tones": percentage_counts(
            nb_tones,
            total,
        ),
        "transformer_tones": percentage_counts(
            transformer_tones,
            total,
        ),
        "topics": percentage_counts(
            topics,
            total,
        ),
        "entities": dict(
            entities.most_common(15)
        ),
        "indicators": dict(
            indicators.most_common(15)
        ),
        "stress": {
            "average": stress_average,
            "minimum": round(
                min(stress_scores),
                2,
            ),
            "maximum": round(
                max(stress_scores),
                2,
            ),
            "levels": percentage_counts(
                stress_levels,
                total,
            ),
        },
        "model_agreement": {
            "agreements": agreements,
            "disagreements": disagreements,
            "agreement_rate": round(
                agreements / total * 100,
                2,
            ) if total else 0.0,
        },
        "highest_stress_news": highest_stress,
        "news": news,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nNoticias procesadas: {total}")
    print(
        "Stress promedio:",
        summary["stress"]["average"],
    )
    print(
        "Coincidencia de modelos:",
        f"{summary['model_agreement']['agreement_rate']}%",
    )

    print("\nTonos:")

    for name, values in summary["tones"].items():
        print(
            f"  {name}: "
            f"{values['count']} "
            f"({values['percentage']}%)"
        )

    print("\nTemas:")

    for name, values in summary["topics"].items():
        print(
            f"  {name}: "
            f"{values['count']} "
            f"({values['percentage']}%)"
        )

    print(f"\nCSV:\n{OUTPUT_PATH}")
    print(f"\nDashboard JSON:\n{SUMMARY_PATH}")


if __name__ == "__main__":
    main()

