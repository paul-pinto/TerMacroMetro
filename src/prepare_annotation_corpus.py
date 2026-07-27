from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_analizadas.csv"
)

OUTPUT_DIR = ROOT / "data" / "annotation"
OUTPUT_PATH = OUTPUT_DIR / "source_bolivia_canonical.csv"
REPORT_PATH = OUTPUT_DIR / "encoding_repair_report.json"

TEXT_COLUMNS = [
    "titulo",
    "texto",
    "fuente",
    "department",
    "scope",
]

SUSPICIOUS = (
    "Ã",
    "Â",
    "â€",
    "â€™",
    "â€œ",
    "â€“",
    "â€”",
    "\ufffd",
)

CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def suspicious_score(text: str) -> int:
    return (
        sum(text.count(token) for token in SUSPICIOUS)
        + len(CONTROL_RE.findall(text))
    )


def repair_text(value: object) -> str:
    if pd.isna(value):
        return ""

    current = str(value)

    for _ in range(2):
        try:
            candidate = current.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break

        if suspicious_score(candidate) < suspicious_score(current):
            current = candidate
        else:
            break

    current = unicodedata.normalize("NFC", current)
    current = CONTROL_RE.sub("", current)

    return current.strip()


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\wáéíóúüñ ]+", "", value)

    return value.strip()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required = {
        "id",
        "titulo",
        "texto",
        "fuente",
        "url",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Faltan columnas: "
            + ", ".join(sorted(missing))
        )

    score_before = {}
    score_after = {}
    repaired_cells = {}

    for column in TEXT_COLUMNS:
        if column not in df.columns:
            continue

        original = df[column].fillna("").astype(str)

        score_before[column] = int(
            original.map(suspicious_score).sum()
        )

        repaired = original.map(repair_text)

        repaired_cells[column] = int(
            (original != repaired).sum()
        )

        df[column] = repaired

        score_after[column] = int(
            repaired.map(suspicious_score).sum()
        )

    df["_normalized_title"] = (
        df["titulo"]
        .fillna("")
        .astype(str)
        .map(normalize_title)
    )

    rows_before = len(df)

    sort_columns = [
        column
        for column in ["source_weight", "economic_score"]
        if column in df.columns
    ]

    if sort_columns:
        df = df.sort_values(
            by=sort_columns,
            ascending=[False] * len(sort_columns),
            na_position="last",
        )

    df = (
        df.drop_duplicates(
            subset=["_normalized_title"],
            keep="first",
        )
        .drop(columns=["_normalized_title"])
        .reset_index(drop=True)
    )

    selected_columns = [
        "id",
        "titulo",
        "texto",
        "fuente",
        "fecha",
        "url",
        "source_type",
        "department",
        "scope",
        "source_weight",
        "economic_score",
        "score_recalculado",
        "topic_id",
        "tema",
        "indicadores",
        "entidades",
        "published_at",
        "effective_at",
    ]

    selected_columns = [
        column
        for column in selected_columns
        if column in df.columns
    ]

    df = df[selected_columns]

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    report = {
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_PATH),
        "rows_before": int(rows_before),
        "rows_after": int(len(df)),
        "duplicates_removed": int(rows_before - len(df)),
        "repaired_cells": repaired_cells,
        "suspicious_score_before": score_before,
        "suspicious_score_after": score_after,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nPrimeros títulos reparados:")

    for title in df["titulo"].head(5):
        print("-", title)


if __name__ == "__main__":
    main()
