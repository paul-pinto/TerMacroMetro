from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CURRENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_current.csv"
)

CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_clean.csv"
)

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

REQUIRED_METADATA = {
    "source_type",
    "department",
    "scope",
    "source_weight",
    "collected_at",
}

GENERIC_TITLES = {
    "sitio oficial del estado plurinacional de bolivia",
    "leyes | banco central de bolivia",
    "directorio | banco central de bolivia",
    "ejecutivos | banco central de bolivia",
    "normativa | mefp",
    "valores y principios",
    "oficinas departamentales",
}


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    if not CURRENT_PATH.exists():
        fail(
            f"No existe la recolección diaria: {CURRENT_PATH}"
        )

    current = pd.read_csv(CURRENT_PATH)

    if current.empty:
        fail("La recolección diaria está vacía.")

    missing = REQUIRED_METADATA - set(
        current.columns
    )

    if missing:
        fail(
            f"Falta metadata en la recolección: {sorted(missing)}"
        )

    source_count = current["fuente"].nunique()

    if source_count < 3:
        fail(
            f"Solo aportaron {source_count} fuentes; se requieren al menos 3."
        )

    normalized_titles = (
        current["titulo"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    generic_count = int(
        normalized_titles.isin(
            GENERIC_TITLES
        ).sum()
    )

    generic_ratio = (
        generic_count
        / len(current)
    )

    if generic_ratio > 0.05:
        fail(
            "Más del 5% de la recolección tiene títulos genéricos."
        )

    if not CLEAN_PATH.exists():
        fail(
            f"No existe el corpus limpio: {CLEAN_PATH}"
        )

    clean = pd.read_csv(CLEAN_PATH)

    if len(clean) < 30:
        fail(
            f"Corpus limpio insuficiente: {len(clean)} documentos."
        )

    if not DASHBOARD_PATH.exists():
        fail(
            f"No existe el dashboard: {DASHBOARD_PATH}"
        )

    with DASHBOARD_PATH.open(
        encoding="utf-8",
    ) as file:
        dashboard = json.load(file)

    required_dashboard = {
        "total_documents",
        "pulsobo_index",
        "optimism_index",
        "topics",
        "stress",
    }

    missing_dashboard = (
        required_dashboard
        - set(dashboard)
    )

    if missing_dashboard:
        fail(
            "Faltan secciones del dashboard: "
            f"{sorted(missing_dashboard)}"
        )

    if dashboard["total_documents"] != len(clean):
        fail(
            "El total del dashboard no coincide con el corpus limpio: "
            f"{dashboard['total_documents']} != {len(clean)}"
        )

    print("[OK] Control de calidad aprobado.")
    print(f"Recolección diaria: {len(current)}")
    print(f"Fuentes activas con datos: {source_count}")
    print(f"Corpus limpio: {len(clean)}")
    print(
        "MacroScore:",
        dashboard["pulsobo_index"]["score"],
    )
    print(
        "Optimismo:",
        dashboard["optimism_index"]["score"],
    )


if __name__ == "__main__":
    main()

