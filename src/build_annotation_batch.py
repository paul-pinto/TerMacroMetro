from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT
    / "data"
    / "annotation"
    / "source_bolivia_canonical.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "annotation"
    / "termacrometro_annotation_batch_001.csv"
)

BATCH_SIZE = 200
RANDOM_STATE = 20260727


def annotation_id(news_id: object) -> str:
    digest = hashlib.sha256(
        str(news_id).encode("utf-8")
    ).hexdigest()[:12]

    return f"ann-{digest}"


def calculate_quotas(
    counts: pd.Series,
    target_size: int,
) -> dict[str, int]:
    total = int(counts.sum())

    exact = {
        str(group): target_size * int(count) / total
        for group, count in counts.items()
    }

    quotas = {
        group: min(
            int(counts[group]),
            int(value),
        )
        for group, value in exact.items()
    }

    assigned = sum(quotas.values())

    remainder_order = sorted(
        exact,
        key=lambda group: exact[group] - int(exact[group]),
        reverse=True,
    )

    for group in remainder_order:
        if assigned >= target_size:
            break

        available = int(counts[group])

        if quotas[group] < available:
            quotas[group] += 1
            assigned += 1

    return quotas


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"No existe el corpus canónico: {SOURCE_PATH}"
        )

    df = pd.read_csv(SOURCE_PATH)

    required = {
        "id",
        "titulo",
        "texto",
        "fuente",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Faltan columnas: "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(
        subset=["id", "titulo", "texto"]
    ).copy()

    df["titulo"] = df["titulo"].astype(str).str.strip()
    df["texto"] = df["texto"].astype(str).str.strip()

    df = df[
        (df["titulo"].str.len() >= 20)
        & (df["texto"].str.len() >= 150)
    ].copy()

    group_column = (
        "tema"
        if "tema" in df.columns
        else "fuente"
    )

    group_values = (
        df[group_column]
        .fillna("sin_clasificar")
        .astype(str)
    )

    counts = group_values.value_counts()
    size = min(BATCH_SIZE, len(df))
    quotas = calculate_quotas(counts, size)

    samples = []

    for index, (group, quota) in enumerate(quotas.items()):
        if quota <= 0:
            continue

        subset = df[group_values == group]

        samples.append(
            subset.sample(
                n=min(quota, len(subset)),
                random_state=RANDOM_STATE + index,
            )
        )

    batch = pd.concat(
        samples,
        ignore_index=True,
    )

    if len(batch) < size:
        remaining = df[
            ~df["id"].isin(batch["id"])
        ]

        extra = remaining.sample(
            n=min(size - len(batch), len(remaining)),
            random_state=RANDOM_STATE + 100,
        )

        batch = pd.concat(
            [batch, extra],
            ignore_index=True,
        )

    batch = batch.head(size).copy()

    batch["annotation_id"] = batch["id"].map(annotation_id)

    batch["texto_modelo"] = (
        batch["titulo"].str.rstrip(".")
        + ". "
        + batch["texto"]
    )

    batch["sentimiento"] = ""
    batch["anotador"] = ""
    batch["fecha_anotacion"] = ""
    batch["duda"] = 0
    batch["comentario"] = ""

    columns = [
        "annotation_id",
        "id",
        "titulo",
        "texto",
        "texto_modelo",
        "fuente",
        "fecha",
        "url",
        "tema",
        "indicadores",
        "sentimiento",
        "anotador",
        "fecha_anotacion",
        "duda",
        "comentario",
    ]

    columns = [
        column
        for column in columns
        if column in batch.columns
    ]

    batch = batch[columns].sample(
        frac=1,
        random_state=RANDOM_STATE + 200,
    ).reset_index(drop=True)

    batch.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("Lote creado:", OUTPUT_PATH)
    print("Filas:", len(batch))
    print("Fuentes:", batch["fuente"].nunique())

    if "tema" in batch.columns:
        print("\nDistribución temática:")
        print(
            batch["tema"]
            .fillna("sin_clasificar")
            .value_counts()
            .to_string()
        )


if __name__ == "__main__":
    main()
