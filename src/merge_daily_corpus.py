from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STANDARD_COLUMNS = [
    "id",
    "titulo",
    "texto",
    "fuente",
    "fecha",
    "url",
    "economic_score",
    "source_type",
    "department",
    "scope",
    "source_weight",
    "collected_at",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def normalize_title(value: object) -> str:
    return re.sub(
        r"[^a-záéíóúüñ0-9]+",
        " ",
        normalize_text(value).lower(),
    ).strip()


def normalize_url(value: object) -> str:
    url = normalize_text(value)

    if not url:
        return ""

    return url.split("#", 1)[0].rstrip("/")


def make_id(
    source: str,
    url: str,
    title: str,
) -> str:
    raw = (
        f"{source}|{url}|{title}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[:20]


def load_csv(path: Path) -> pd.DataFrame:
    if (
        not path.exists()
        or path.stat().st_size == 0
    ):
        return pd.DataFrame(
            columns=STANDARD_COLUMNS
        )

    df = pd.read_csv(path)

    defaults = {
        "id": "",
        "titulo": "",
        "texto": "",
        "fuente": "",
        "fecha": "",
        "url": "",
        "economic_score": 0,
        "source_type": "unknown",
        "department": "No especificado",
        "scope": "unknown",
        "source_weight": 0.5,
        "collected_at": "",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    return df[
        STANDARD_COLUMNS
    ].copy()


def quality_score(
    row: pd.Series,
) -> tuple[int, int, int, float]:
    text_length = len(
        normalize_text(
            row.get("texto", "")
        )
    )

    title_length = len(
        normalize_text(
            row.get("titulo", "")
        )
    )

    economic_score = pd.to_numeric(
        row.get(
            "economic_score",
            0,
        ),
        errors="coerce",
    )

    source_weight = pd.to_numeric(
        row.get(
            "source_weight",
            0.5,
        ),
        errors="coerce",
    )

    if pd.isna(economic_score):
        economic_score = 0

    if pd.isna(source_weight):
        source_weight = 0.5

    return (
        text_length,
        int(economic_score),
        title_length,
        float(source_weight),
    )


def merge_corpus(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    collected_at = datetime.now(
        timezone.utc
    ).isoformat()

    previous = previous.copy()
    current = current.copy()

    current["collected_at"] = (
        current["collected_at"]
        .fillna("")
        .astype(str)
        .replace("", collected_at)
    )

    combined = pd.concat(
        [
            previous,
            current,
        ],
        ignore_index=True,
    )

    text_columns = [
        "titulo",
        "texto",
        "fuente",
        "fecha",
        "source_type",
        "department",
        "scope",
        "collected_at",
    ]

    for column in text_columns:
        combined[column] = (
            combined[column]
            .map(normalize_text)
        )

    combined["url"] = (
        combined["url"]
        .map(normalize_url)
    )

    combined["economic_score"] = (
        pd.to_numeric(
            combined["economic_score"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    combined["source_weight"] = (
        pd.to_numeric(
            combined["source_weight"],
            errors="coerce",
        )
        .fillna(0.5)
        .clip(0, 1)
    )

    combined["source_type"] = (
        combined["source_type"]
        .replace("", "unknown")
    )

    combined["department"] = (
        combined["department"]
        .replace(
            "",
            "No especificado",
        )
    )

    combined["scope"] = (
        combined["scope"]
        .replace("", "unknown")
    )

    combined = combined[
        combined["titulo"].str.len() >= 10
    ].copy()

    combined = combined[
        combined["texto"].str.len() >= 100
    ].copy()

    combined["_title_key"] = (
        combined["titulo"]
        .map(normalize_title)
    )

    combined["_url_key"] = (
        combined["url"]
    )

    combined["_quality"] = (
        combined.apply(
            quality_score,
            axis=1,
        )
    )

    combined = combined.sort_values(
        "_quality",
        ascending=False,
    )

    with_url = combined[
        combined["_url_key"].str.len() > 0
    ].drop_duplicates(
        subset=["_url_key"],
        keep="first",
    )

    without_url = combined[
        combined["_url_key"].str.len() == 0
    ]

    combined = pd.concat(
        [
            with_url,
            without_url,
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=["_title_key"],
        keep="first",
    )

    combined["id"] = combined.apply(
        lambda row: (
            normalize_text(
                row["id"]
            )
            or make_id(
                source=row["fuente"],
                url=row["url"],
                title=row["titulo"],
            )
        ),
        axis=1,
    )

    combined = combined.drop(
        columns=[
            "_title_key",
            "_url_key",
            "_quality",
        ]
    )

    combined = combined.sort_values(
        by=[
            "collected_at",
            "fuente",
            "titulo",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    return combined[
        STANDARD_COLUMNS
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fusiona el histórico de TerMacroMetro "
            "con la recolección diaria."
        )
    )

    parser.add_argument(
        "--previous",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--current",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    previous = load_csv(
        args.previous
    )

    current = load_csv(
        args.current
    )

    merged = merge_corpus(
        previous=previous,
        current=current,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Histórico anterior:",
        len(previous),
    )

    print(
        "Recolección actual:",
        len(current),
    )

    print(
        "Corpus fusionado:",
        len(merged),
    )

    print(
        "Noticias nuevas netas:",
        max(
            0,
            len(merged) - len(previous),
        ),
    )

    print("\nPor fuente:")
    print(
        merged["fuente"]
        .value_counts()
    )

    print("\nPor tipo:")
    print(
        merged["source_type"]
        .value_counts()
    )

    print("\nPor departamento:")
    print(
        merged["department"]
        .value_counts()
    )

    print(
        "\nGuardado en:",
        args.output,
    )


if __name__ == "__main__":
    main()

