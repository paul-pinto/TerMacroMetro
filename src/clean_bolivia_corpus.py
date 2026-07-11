from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_clean.csv"
)


EXCLUDED_TITLES = {
    "leyes",
    "normativa",
    "valores y principios",
    "oficinas departamentales",
    "introducción",
    "introduccion",
    "manual de procesos y procedimientos",
    "programa fiscal financiero",
    "gráficos ipc",
    "graficos ipc",
    "sistema integral de pensiones inversiones",
    "modelos referenciales eeff",
    "sitio oficial del estado plurinacional de bolivia",
}

EXCLUDED_PATTERNS = (
    r"\bmanual\b",
    r"\bnormativa\b",
    r"\bley(?:es)?\b",
    r"\breglamento\b",
    r"\bprocedimiento(?:s)?\b",
    r"\borganigrama\b",
    r"\bmisión\b",
    r"\bmision\b",
    r"\bvisión\b",
    r"\bvision\b",
    r"\bvalores institucionales\b",
    r"\btransparencia\b",
    r"\brendición de cuentas\b",
    r"\brendicion de cuentas\b",
    r"\bconvocatoria\b",
    r"\bcontrataciones\b",
    r"\boficinas departamentales\b",
    r"\bdirectorio institucional\b",
)

ECONOMIC_TERMS = {
    "inflación",
    "inflacion",
    "dólar",
    "dolar",
    "divisas",
    "tipo de cambio",
    "reservas",
    "combustible",
    "combustibles",
    "diésel",
    "diesel",
    "gasolina",
    "petróleo",
    "petroleo",
    "gas natural",
    "litio",
    "exportaciones",
    "importaciones",
    "comercio exterior",
    "balanza comercial",
    "déficit",
    "deficit",
    "superávit",
    "superavit",
    "pib",
    "crecimiento",
    "economía",
    "economia",
    "empleo",
    "desempleo",
    "salario",
    "crédito",
    "credito",
    "banca",
    "financiero",
    "deuda",
    "presupuesto",
    "inversión",
    "inversion",
    "producción",
    "produccion",
    "ypfb",
    "bcb",
    "ine",
    "asfi",
    "mefp",
}


def normalize(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def economic_score(text: str) -> int:
    normalized = normalize(text)

    return sum(
        1
        for term in ECONOMIC_TERMS
        if term in normalized
    )


def is_excluded_title(title: str) -> bool:
    normalized = normalize(title)

    if normalized in EXCLUDED_TITLES:
        return True

    return any(
        re.search(pattern, normalized)
        for pattern in EXCLUDED_PATTERNS
    )


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
        "url",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Faltan columnas: {sorted(required - set(df.columns))}"
        )

    original_count = len(df)

    df["titulo"] = (
        df["titulo"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["texto"] = (
        df["texto"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["titulo"].str.len() >= 20
    ].copy()

    df = df[
        df["texto"].str.len() >= 250
    ].copy()

    df = df[
        ~df["titulo"].apply(is_excluded_title)
    ].copy()

    df["score_recalculado"] = (
        df["titulo"] + " " + df["texto"]
    ).apply(economic_score)

    df = df[
        df["score_recalculado"] >= 2
    ].copy()

    df["titulo_normalizado"] = (
        df["titulo"]
        .str.lower()
        .str.replace(
            r"[^a-záéíóúüñ0-9]+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    df = df.drop_duplicates(
        subset=["url"],
        keep="first",
    )

    df = df.drop_duplicates(
        subset=["titulo_normalizado"],
        keep="first",
    )

    df = df.drop(
        columns=["titulo_normalizado"]
    )

    df = df.sort_values(
        by=[
            "score_recalculado",
            "fuente",
            "titulo",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Corpus original: {original_count}")
    print(f"Corpus limpio:   {len(df)}")
    print(f"Eliminadas:      {original_count - len(df)}")

    print("\nDistribución por fuente:")
    print(df["fuente"].value_counts())

    print("\nEjemplos conservados:")
    print(
        df[
            ["fuente", "titulo"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nGuardado en:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
