from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import (
    cosine_similarity,
)


DEFAULT_SIMILARITY_THRESHOLD = 0.48
DEFAULT_MAX_DAYS_APART = 3


@dataclass(frozen=True)
class EventClusteringConfig:
    similarity_threshold: float = (
        DEFAULT_SIMILARITY_THRESHOLD
    )
    max_days_apart: int = DEFAULT_MAX_DAYS_APART
    min_df: int = 1
    max_features: int = 8000


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[
                self.parent[value]
            ]
            value = self.parent[value]

        return value

    def union(
        self,
        left: int,
        right: int,
    ) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        if (
            self.rank[root_left]
            < self.rank[root_right]
        ):
            self.parent[root_left] = root_right
        elif (
            self.rank[root_left]
            > self.rank[root_right]
        ):
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def normalize_text(value: object) -> str:
    text = str(value or "").lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-záéíóúüñ0-9%$\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def safe_json_list(
    value: object,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if (
        isinstance(value, float)
        and pd.isna(value)
    ):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return []

    if not isinstance(parsed, list):
        return []

    return [
        str(item).strip()
        for item in parsed
        if str(item).strip()
    ]


def build_document_text(
    row: pd.Series,
) -> str:
    title = normalize_text(
        row.get("titulo", "")
    )

    body = normalize_text(
        row.get("texto", "")
    )

    # El título se repite para darle más peso.
    return f"{title} {title} {body}".strip()


def parse_event_dates(
    df: pd.DataFrame,
) -> pd.Series:
    for column in (
        "effective_at",
        "published_at",
        "fecha",
        "collected_at",
    ):
        if column not in df.columns:
            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

        if parsed.notna().any():
            return parsed

    return pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns, UTC]",
    )


def dates_are_compatible(
    left: pd.Timestamp,
    right: pd.Timestamp,
    *,
    max_days_apart: int,
) -> bool:
    if pd.isna(left) or pd.isna(right):
        return True

    difference = abs(
        (
            left - right
        ).total_seconds()
    )

    return (
        difference
        <= max_days_apart * 86400
    )


def stable_event_id(
    rows: pd.DataFrame,
) -> str:
    values: list[str] = []

    for column in (
        "url",
        "titulo",
        "effective_at",
    ):
        if column not in rows.columns:
            continue

        values.extend(
            rows[column]
            .fillna("")
            .astype(str)
            .tolist()
        )

    material = "|".join(
        sorted(values)
    )

    digest = hashlib.sha1(
        material.encode("utf-8")
    ).hexdigest()[:12]

    return f"evt-{digest}"


def dominant_scalar(
    values: pd.Series,
) -> str:
    cleaned = [
        str(value).strip()
        for value in values
        if str(value).strip()
        and str(value).lower() != "nan"
    ]

    if not cleaned:
        return ""

    return Counter(cleaned).most_common(1)[0][0]


def dominant_json_label(
    values: pd.Series,
) -> str:
    counter: Counter[str] = Counter()

    for value in values:
        counter.update(
            safe_json_list(value)
        )

    if not counter:
        return ""

    return counter.most_common(1)[0][0]


def choose_representative(
    group: pd.DataFrame,
) -> pd.Series:
    ranked = group.copy()

    ranked["_representative_score"] = (
        pd.to_numeric(
            ranked.get(
                "stress_score",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
        * 0.35
    )

    title_length = (
        ranked["titulo"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    text_length = (
        ranked["texto"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    ranked["_representative_score"] += (
        title_length.clip(
            upper=180
        )
        / 180
        * 20
    )

    ranked["_representative_score"] += (
        text_length.clip(
            upper=3000
        )
        / 3000
        * 25
    )

    if "effective_at" in ranked.columns:
        dates = pd.to_datetime(
            ranked["effective_at"],
            errors="coerce",
            utc=True,
        )

        if dates.notna().any():
            timestamp_values = (
                dates.astype("int64")
                .where(
                    dates.notna(),
                    0,
                )
            )

            maximum = timestamp_values.max()

            if maximum > 0:
                ranked[
                    "_representative_score"
                ] += (
                    timestamp_values
                    / maximum
                    * 20
                )

    return (
        ranked
        .sort_values(
            "_representative_score",
            ascending=False,
        )
        .iloc[0]
    )


def cluster_news(
    df: pd.DataFrame,
    *,
    config: EventClusteringConfig | None = None,
) -> pd.DataFrame:
    config = (
        config
        or EventClusteringConfig()
    )

    if df.empty:
        output = df.copy()
        output["event_id"] = []
        output["event_size"] = []
        return output

    required = {
        "titulo",
        "texto",
        "fuente",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas para clustering: "
            f"{sorted(missing)}"
        )

    output = df.copy().reset_index(drop=True)

    documents = output.apply(
        build_document_text,
        axis=1,
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=config.min_df,
        max_features=config.max_features,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    similarities = cosine_similarity(
        matrix,
        dense_output=True,
    )

    dates = parse_event_dates(output)

    union_find = UnionFind(
        len(output)
    )

    for left in range(len(output)):
        for right in range(
            left + 1,
            len(output),
        ):
            similarity = float(
                similarities[left, right]
            )

            if (
                similarity
                < config.similarity_threshold
            ):
                continue

            if not dates_are_compatible(
                dates.iloc[left],
                dates.iloc[right],
                max_days_apart=(
                    config.max_days_apart
                ),
            ):
                continue

            union_find.union(
                left,
                right,
            )

    groups: dict[int, list[int]] = {}

    for index in range(len(output)):
        root = union_find.find(index)

        groups.setdefault(
            root,
            [],
        ).append(index)

    event_ids: dict[int, str] = {}
    event_sizes: dict[int, int] = {}

    for indices in groups.values():
        group = output.iloc[indices]

        event_id = stable_event_id(
            group
        )

        for index in indices:
            event_ids[index] = event_id
            event_sizes[index] = len(indices)

    output["event_id"] = [
        event_ids[index]
        for index in range(len(output))
    ]

    output["event_size"] = [
        event_sizes[index]
        for index in range(len(output))
    ]

    return output


def summarize_events(
    clustered_df: pd.DataFrame,
) -> pd.DataFrame:
    if clustered_df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_title",
                "documents",
                "unique_sources",
                "sources",
                "effective_at",
                "average_stress",
                "maximum_stress",
                "dominant_topic",
                "dominant_indicator",
                "dominant_entity",
                "representative_url",
            ]
        )

    if "event_id" not in clustered_df.columns:
        raise ValueError(
            "El DataFrame no contiene event_id."
        )

    rows: list[dict[str, Any]] = []

    for event_id, group in clustered_df.groupby(
        "event_id",
        sort=False,
    ):
        representative = (
            choose_representative(group)
        )

        sources = sorted(
            {
                str(value).strip()
                for value in group[
                    "fuente"
                ]
                if str(value).strip()
                and str(value).lower()
                != "nan"
            }
        )

        stress_values = pd.to_numeric(
            group.get(
                "stress_score",
                pd.Series(
                    [0] * len(group)
                ),
            ),
            errors="coerce",
        ).fillna(0)

        event_dates = parse_event_dates(
            group
        )

        latest_date = (
            event_dates.max()
            if event_dates.notna().any()
            else pd.NaT
        )

        rows.append(
            {
                "event_id": event_id,
                "event_title": str(
                    representative.get(
                        "titulo",
                        "",
                    )
                ).strip(),
                "documents": int(
                    len(group)
                ),
                "unique_sources": int(
                    len(sources)
                ),
                "sources": json.dumps(
                    sources,
                    ensure_ascii=False,
                ),
                "effective_at": (
                    latest_date.isoformat()
                    if pd.notna(latest_date)
                    else ""
                ),
                "average_stress": round(
                    float(
                        stress_values.mean()
                    ),
                    2,
                ),
                "maximum_stress": round(
                    float(
                        stress_values.max()
                    ),
                    2,
                ),
                "dominant_topic": (
                    dominant_scalar(
                        group.get(
                            "tema",
                            pd.Series(dtype=str),
                        )
                    )
                ),
                "dominant_indicator": (
                    dominant_json_label(
                        group.get(
                            "indicadores",
                            pd.Series(dtype=str),
                        )
                    )
                ),
                "dominant_entity": (
                    dominant_json_label(
                        group.get(
                            "entidades",
                            pd.Series(dtype=str),
                        )
                    )
                ),
                "representative_source": str(
                    representative.get(
                        "fuente",
                        "",
                    )
                ).strip(),
                "representative_url": str(
                    representative.get(
                        "url",
                        "",
                    )
                ).strip(),
                "representative_tone": str(
                    representative.get(
                        "tono_consolidado",
                        "",
                    )
                ).strip(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "unique_sources",
                "documents",
                "maximum_stress",
                "effective_at",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )
