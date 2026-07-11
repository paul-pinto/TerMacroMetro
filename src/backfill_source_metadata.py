from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "bolivia_sources.json"
)

CORPUS_PATHS = [
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia.csv",

    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_clean.csv",

    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_analizadas.csv",
]


def normalize_name(value: object) -> str:
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .split()
    )


def load_source_metadata() -> dict[str, dict]:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config = json.load(file)

    metadata: dict[str, dict] = {}

    for source in config["sources"]:
        metadata[
            normalize_name(source["name"])
        ] = {
            "source_type": source["source_type"],
            "department": source["department"],
            "scope": source["scope"],
            "source_weight": float(
                source["source_weight"]
            ),
        }

    return metadata


def repair_file(
    path: Path,
    metadata: dict[str, dict],
) -> None:
    if not path.exists():
        print(f"[SKIP] No existe: {path}")
        return

    df = pd.read_csv(path)

    if "fuente" not in df.columns:
        print(f"[SKIP] Sin columna fuente: {path}")
        return

    defaults = {
        "source_type": "unknown",
        "department": "No especificado",
        "scope": "unknown",
        "source_weight": 0.5,
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    repaired = 0
    unresolved: set[str] = set()

    for index, row in df.iterrows():
        source_name = normalize_name(
            row.get("fuente", "")
        )

        source_metadata = metadata.get(
            source_name
        )

        if source_metadata is None:
            unresolved.add(
                str(row.get("fuente", ""))
            )
            continue

        needs_repair = (
            str(
                row.get(
                    "source_type",
                    "",
                )
            ).strip() in {
                "",
                "unknown",
                "nan",
            }
            or str(
                row.get(
                    "department",
                    "",
                )
            ).strip() in {
                "",
                "No especificado",
                "nan",
            }
            or str(
                row.get(
                    "scope",
                    "",
                )
            ).strip() in {
                "",
                "unknown",
                "nan",
            }
        )

        if needs_repair:
            for column, value in (
                source_metadata.items()
            ):
                df.at[index, column] = value

            repaired += 1

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nArchivo: {path.name}")
    print(f"Filas: {len(df)}")
    print(f"Reparadas: {repaired}")

    if unresolved:
        print(
            "Fuentes sin configuración:",
            ", ".join(sorted(unresolved)),
        )


def main() -> None:
    metadata = load_source_metadata()

    for path in CORPUS_PATHS:
        repair_file(
            path=path,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
