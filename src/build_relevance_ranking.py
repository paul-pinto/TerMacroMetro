from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.relevance_ranking import (
    build_relevance_rankings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_analizadas.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "relevance_ranking.json"
)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    rankings = build_relevance_rankings(
        df,
        window_hours=72,
        limit=10,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            rankings,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTOP TEMAS")

    for item in rankings["topics"]:
        print(
            f"{item['score']:6.2f} | "
            f"{item['name']} | "
            f"actual={item['current_count']} | "
            f"previo={item['previous_count']} | "
            f"fuentes={item['unique_sources']} | "
            f"{item['momentum']}"
        )

    print("\nTOP INDICADORES")

    for item in rankings["indicators"]:
        print(
            f"{item['score']:6.2f} | "
            f"{item['name']} | "
            f"actual={item['current_count']} | "
            f"previo={item['previous_count']} | "
            f"fuentes={item['unique_sources']} | "
            f"{item['momentum']}"
        )

    print("\nTOP ENTIDADES")

    for item in rankings["entities"]:
        print(
            f"{item['score']:6.2f} | "
            f"{item['name']} | "
            f"actual={item['current_count']} | "
            f"previo={item['previous_count']} | "
            f"fuentes={item['unique_sources']} | "
            f"{item['momentum']}"
        )


if __name__ == "__main__":
    main()
