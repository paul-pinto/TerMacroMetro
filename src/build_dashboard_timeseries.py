from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
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

MIN_DATE = pd.Timestamp("2025-01-01")
MAX_DATE = pd.Timestamp.now().normalize()
MIN_DOCUMENTS_PER_PERIOD = 2


MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def parse_spanish_date(
    value: object,
) -> pd.Timestamp | None:
    """
    Normaliza fechas de portales bolivianos.

    Las fechas ISO-8601 se interpretan primero sin dayfirst y se
    convierten a la zona horaria de Bolivia. Esto evita transformar
    2026-07-10 en 2026-10-07.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "nat",
    }:
        return None

    # ISO-8601:
    # 2026-07-10
    # 2026-07-10T18:00:43+00:00
    # 2026-07-10T14:00:00-04:00
    if re.match(
        r"^\d{4}-\d{2}-\d{2}(?:[T\s].*)?$",
        text,
    ):
        try:
            parsed = pd.to_datetime(
                text,
                errors="raise",
                utc=True,
            )

            return (
                pd.Timestamp(parsed)
                .tz_convert("America/La_Paz")
                .tz_localize(None)
            )
        except (
            ValueError,
            TypeError,
            OverflowError,
        ):
            return None

    normalized = (
        text.lower()
        .replace(",", " ")
        .replace(" de ", " ")
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    # 10 julio 2026
    match = re.search(
        r"\b(\d{1,2})\s+"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|"
        r"agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s+(\d{4})\b",
        normalized,
    )

    if match:
        try:
            return pd.Timestamp(
                year=int(match.group(3)),
                month=MONTHS_ES[match.group(2)],
                day=int(match.group(1)),
            )
        except ValueError:
            return None

    # julio 10 2026
    match = re.search(
        r"\b"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|"
        r"agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s+(\d{1,2})\s+(\d{4})\b",
        normalized,
    )

    if match:
        try:
            return pd.Timestamp(
                year=int(match.group(3)),
                month=MONTHS_ES[match.group(1)],
                day=int(match.group(2)),
            )
        except ValueError:
            return None

    # Formatos numéricos no ISO: 10/07/2026, 10-07-2026.
    try:
        parsed = pd.to_datetime(
            text,
            errors="raise",
            dayfirst=True,
        )

        timestamp = pd.Timestamp(parsed)

        if timestamp.tzinfo is not None:
            timestamp = (
                timestamp
                .tz_convert("America/La_Paz")
                .tz_localize(None)
            )

        return timestamp

    except (
        ValueError,
        TypeError,
        OverflowError,
    ):
        return None


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    if not denominator:
        return 0.0

    return round(
        numerator / denominator * 100,
        2,
    )


def build_period_record(
    period: str,
    group: pd.DataFrame,
) -> dict[str, Any]:
    total = len(group)

    unfavorable = int(
        (
            group["tono_consolidado"]
            == "desfavorable"
        ).sum()
    )

    high_stress = int(
        group["stress_nivel"].isin(
            ["alto", "crítico"]
        ).sum()
    )

    disagreement = int(
        (
            group["coincidencia_modelos"]
            .astype(str)
            .str.lower()
            .isin(["false", "0"])
        ).sum()
    )

    stress_average = float(
        group["stress_score"].mean()
    )

    unfavorable_percentage = percentage(
        unfavorable,
        total,
    )

    high_stress_percentage = percentage(
        high_stress,
        total,
    )

    disagreement_percentage = percentage(
        disagreement,
        total,
    )

    pulsobo_score = (
        stress_average * 0.50
        + unfavorable_percentage * 0.30
        + high_stress_percentage * 0.10
        + disagreement_percentage * 0.10
    )

    pulsobo_score = round(
        max(0.0, min(100.0, pulsobo_score)),
        2,
    )

    return {
        "period": period,
        "documents": total,
        "stress_average": round(
            stress_average,
            2,
        ),
        "unfavorable_percentage": (
            unfavorable_percentage
        ),
        "high_stress_percentage": (
            high_stress_percentage
        ),
        "model_disagreement_percentage": (
            disagreement_percentage
        ),
        "pulsobo_index": pulsobo_score,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {INPUT_PATH}"
        )

    if not DASHBOARD_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {DASHBOARD_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "fecha",
        "stress_score",
        "stress_nivel",
        "tono_consolidado",
        "coincidencia_modelos",
        "indicadores",
        "fuente",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas: {sorted(missing)}"
        )

    df["fecha_normalizada"] = (
        df["fecha"]
        .apply(parse_spanish_date)
    )

    parsed_dates = df[
        df["fecha_normalizada"].notna()
    ].copy()

    invalid_range = parsed_dates[
        (
            parsed_dates["fecha_normalizada"]
            < MIN_DATE
        )
        |
        (
            parsed_dates["fecha_normalizada"]
            > MAX_DATE
        )
    ].copy()

    valid_dates = parsed_dates[
        (
            parsed_dates["fecha_normalizada"]
            >= MIN_DATE
        )
        &
        (
            parsed_dates["fecha_normalizada"]
            <= MAX_DATE
        )
    ].copy()

    if not valid_dates.empty:
        valid_dates["periodo"] = (
            valid_dates["fecha_normalizada"]
            .dt.to_period("M")
            .astype(str)
        )

        timeseries = []

        for period, group in valid_dates.groupby(
            "periodo"
        ):
            if (
                len(group)
                < MIN_DOCUMENTS_PER_PERIOD
            ):
                continue

            timeseries.append(
                build_period_record(
                    period=period,
                    group=group,
                )
            )

        timeseries.sort(
            key=lambda item: item["period"]
        )
    else:
        timeseries = []

    indicator_counts: dict[str, int] = {}

    for raw_value in df["indicadores"]:
        try:
            values = json.loads(
                str(raw_value)
            )

            if not isinstance(values, list):
                values = []

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            values = []

        for indicator in values:
            name = str(indicator).strip()

            if not name:
                continue

            indicator_counts[name] = (
                indicator_counts.get(name, 0)
                + 1
            )

    total_documents = len(df)

    indicator_map = [
        {
            "name": name,
            "count": count,
            "percentage": round(
                count / total_documents * 100,
                2,
            )
            if total_documents
            else 0.0,
        }
        for name, count in sorted(
            indicator_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    source_analysis = {}

    for source, group in df.groupby(
        "fuente"
    ):
        source_analysis[str(source)] = {
            "documents": len(group),
            "favorable": int(
                (
                    group["tono_consolidado"]
                    == "favorable"
                ).sum()
            ),
            "neutral": int(
                (
                    group["tono_consolidado"]
                    == "neutral"
                ).sum()
            ),
            "desfavorable": int(
                (
                    group["tono_consolidado"]
                    == "desfavorable"
                ).sum()
            ),
            "stress_average": round(
                float(
                    group["stress_score"].mean()
                ),
                2,
            ),
        }

    with DASHBOARD_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dashboard = json.load(file)

    dashboard["temporal_analysis"] = {
        "available": bool(timeseries),
        "frequency": "monthly",
        "minimum_documents_per_period": (
            MIN_DOCUMENTS_PER_PERIOD
        ),
        "minimum_date": str(
            MIN_DATE.date()
        ),
        "maximum_date": str(
            MAX_DATE.date()
        ),
        "parsed_documents": int(
            len(parsed_dates)
        ),
        "dated_documents": int(
            len(valid_dates)
        ),
        "out_of_range_documents": int(
            len(invalid_range)
        ),
        "undated_documents": int(
            len(df) - len(parsed_dates)
        ),
        "series": timeseries,
    }

    dashboard["indicator_map"] = (
        indicator_map
    )

    dashboard["source_analysis"] = (
        source_analysis
    )

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

    print("Documentos totales:", len(df))
    print("Fechas interpretadas:", len(parsed_dates))
    print("Fechas válidas:", len(valid_dates))
    print(
        "Fuera del rango:",
        len(invalid_range),
    )
    print(
        "Sin fecha interpretable:",
        len(df) - len(parsed_dates),
    )
    print(
        "Meses representados:",
        len(timeseries),
    )

    print("\nSerie mensual:")

    for item in timeseries:
        print(
            f"  {item['period']}: "
            f"{item['pulsobo_index']} "
            f"({item['documents']} noticias)"
        )

    if not invalid_range.empty:
        print("\nFechas descartadas:")

        for _, row in invalid_range[
            ["fecha", "fecha_normalizada"]
        ].head(20).iterrows():
            print(
                f"  {row['fecha']} -> "
                f"{row['fecha_normalizada']}"
            )

    print(
        "\nDashboard actualizado:",
        DASHBOARD_PATH,
    )


if __name__ == "__main__":
    main()

