from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "bolivia_sources.json"
)

VALID_SOURCE_TYPES = {
    "official",
    "agency",
    "newspaper",
    "television",
    "regional",
    "business",
}

VALID_METHODS = {
    "rss",
    "wordpress",
    "sitemap",
    "listing",
    "generic",
    "discovery",
}

VALID_STATUSES = {
    "active",
    "pending_validation",
    "pending_discovery",
    "disabled",
}


def load_source_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la configuración: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    validate_source_config(data)

    return data


def validate_source_config(
    data: dict[str, Any],
) -> None:
    if not isinstance(data, dict):
        raise ValueError(
            "La configuración debe ser un objeto JSON."
        )

    sources = data.get("sources")

    if not isinstance(sources, list):
        raise ValueError(
            "'sources' debe ser una lista."
        )

    ids: set[str] = set()
    names: set[str] = set()

    required_fields = {
        "id",
        "name",
        "source_type",
        "department",
        "scope",
        "collection_method",
        "status",
        "source_weight",
        "max_documents",
        "tags",
    }

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(
                f"Fuente {index}: debe ser un objeto."
            )

        missing = required_fields - set(source)

        if missing:
            raise ValueError(
                f"Fuente {index}: faltan campos "
                f"{sorted(missing)}"
            )

        source_id = str(source["id"]).strip()
        source_name = str(source["name"]).strip()

        if not source_id:
            raise ValueError(
                f"Fuente {index}: id vacío."
            )

        if source_id in ids:
            raise ValueError(
                f"ID duplicado: {source_id}"
            )

        if source_name.lower() in names:
            raise ValueError(
                f"Nombre duplicado: {source_name}"
            )

        ids.add(source_id)
        names.add(source_name.lower())

        if source["source_type"] not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"{source_id}: source_type inválido."
            )

        if source["collection_method"] not in VALID_METHODS:
            raise ValueError(
                f"{source_id}: método inválido."
            )

        if source["status"] not in VALID_STATUSES:
            raise ValueError(
                f"{source_id}: estado inválido."
            )

        weight = float(source["source_weight"])

        if not 0 <= weight <= 1:
            raise ValueError(
                f"{source_id}: source_weight "
                "debe estar entre 0 y 1."
            )

        maximum = int(source["max_documents"])

        if maximum <= 0:
            raise ValueError(
                f"{source_id}: max_documents inválido."
            )

        tags = source["tags"]

        if not isinstance(tags, list):
            raise ValueError(
                f"{source_id}: tags debe ser una lista."
            )

        homepage = source.get("homepage")

        if (
            source["status"] == "active"
            and not homepage
        ):
            raise ValueError(
                f"{source_id}: una fuente activa "
                "debe tener homepage."
            )


def get_runnable_sources(
    data: dict[str, Any],
    include_pending_validation: bool = False,
) -> list[dict[str, Any]]:
    allowed_statuses = {"active"}

    if include_pending_validation:
        allowed_statuses.add(
            "pending_validation"
        )

    return [
        source
        for source in data["sources"]
        if source["status"] in allowed_statuses
        and source.get("homepage")
    ]


def main() -> None:
    data = load_source_config()

    sources = data["sources"]
    runnable = get_runnable_sources(data)

    type_counts = Counter(
        source["source_type"]
        for source in sources
    )

    status_counts = Counter(
        source["status"]
        for source in sources
    )

    departments = {
        source["department"]
        for source in sources
    }

    print("Configuración válida.")
    print("Fuentes registradas:", len(sources))
    print("Fuentes activas:", len(runnable))
    print("Departamentos:", len(departments))

    print("\nPor tipo:")

    for source_type, count in sorted(
        type_counts.items()
    ):
        print(f"  {source_type}: {count}")

    print("\nPor estado:")

    for status, count in sorted(
        status_counts.items()
    ):
        print(f"  {status}: {count}")

    print("\nFuentes activas:")

    for source in runnable:
        print(
            f"  {source['name']} "
            f"[{source['collection_method']}]"
        )


if __name__ == "__main__":
    main()
