from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import requests

from src.collectors.engine import (
    ListingCollector,
    RSSCollector,
)
from src.source_config import (
    load_source_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_current.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "source_collection_report.json"
)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "TerMacroMetro-Research/1.0"
    ),
    "Accept-Language": "es-BO,es;q=0.9",
}


def build_collector(
    session: requests.Session,
    source: dict[str, Any],
    defaults: dict[str, Any],
):
    method = source["collection_method"]

    if method == "rss":
        return RSSCollector(
            session,
            source,
            defaults,
        )

    if method in {
        "listing",
        "wordpress",
        "generic",
    }:
        return ListingCollector(
            session,
            source,
            defaults,
        )

    return None


def main() -> None:
    config = load_source_config()
    defaults = config["defaults"]

    session = requests.Session()
    session.headers.update(HEADERS)

    records = []
    report = {
        "sources": [],
        "total_records": 0,
    }

    for source in config["sources"]:
        if source["status"] != "active":
            continue

        collector = build_collector(
            session=session,
            source=source,
            defaults=defaults,
        )

        if collector is None:
            continue

        print(
            f"\n=== {source['name']} ==="
        )

        source_records = (
            collector.collect()
        )

        records.extend(
            item.as_dict()
            for item in source_records
        )

        report["sources"].append(
            {
                "source": source["name"],
                "method": source[
                    "collection_method"
                ],
                "records": len(
                    source_records
                ),
            }
        )

        print(
            f"{source['name']}: "
            f"{len(source_records)} registros"
        )

    unique_by_url: dict[str, dict] = {}
    seen_titles: set[str] = set()
    final_records: list[dict] = []

    for record in records:
        url_key = str(
            record.get("url", "")
        ).split("#", 1)[0].rstrip("/")

        title_key = " ".join(
            str(
                record.get("titulo", "")
            )
            .lower()
            .split()
        )

        if url_key and url_key in unique_by_url:
            continue

        if title_key in seen_titles:
            continue

        if url_key:
            unique_by_url[url_key] = record

        seen_titles.add(title_key)
        final_records.append(record)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            final_records
        )

    report["total_records"] = len(
        final_records
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nTotal recolectado:",
        len(final_records),
    )

    print(
        "CSV:",
        OUTPUT_PATH,
    )

    print(
        "Reporte:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()


