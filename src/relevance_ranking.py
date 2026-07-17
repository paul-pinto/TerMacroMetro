from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from src.temporal_quality import (
    TemporalPolicy,
    normalize_temporal_columns,
    utc_now,
)


@dataclass(frozen=True)
class RankingWeights:
    frequency: float = 0.35
    source_diversity: float = 0.20
    recency: float = 0.15
    stress: float = 0.10
    momentum: float = 0.20


def safe_json_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, float) and pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return [text]

    if not isinstance(parsed, list):
        return []

    return [
        str(item).strip()
        for item in parsed
        if str(item).strip()
    ]


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def normalized_momentum(
    current_count: int,
    previous_count: int,
) -> float:
    ratio = (
        current_count + 1
    ) / (
        previous_count + 1
    )

    log_ratio = math.log2(ratio)

    # -2 → 0.0
    #  0 → 0.5
    # +2 → 1.0
    return clamp(
        (log_ratio + 2.0) / 4.0
    )


def momentum_label(
    current_count: int,
    previous_count: int,
) -> str:
    ratio = (
        current_count + 1
    ) / (
        previous_count + 1
    )

    if ratio >= 1.5:
        return "emergente"

    if ratio <= 0.70:
        return "en descenso"

    return "estable"


def iter_labels(
    row: pd.Series,
    column: str,
    *,
    json_list: bool,
) -> Iterable[str]:
    if json_list:
        yield from safe_json_list(
            row.get(column)
        )
        return

    value = str(
        row.get(column, "")
    ).strip()

    if value and value.lower() != "nan":
        yield value


def rank_dimension(
    historical_df: pd.DataFrame,
    *,
    column: str,
    json_list: bool = False,
    window_hours: int = 72,
    limit: int = 10,
    weights: RankingWeights | None = None,
    now: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    weights = weights or RankingWeights()
    now = now or utc_now()

    enriched = normalize_temporal_columns(
        historical_df,
        now=now,
        policy=TemporalPolicy(
            recent_window_hours=window_hours,
        ),
    )

    ages = pd.to_numeric(
        enriched["age_hours"],
        errors="coerce",
    )

    current_mask = (
        ages.notna()
        & ages.ge(0)
        & ages.le(window_hours)
    )

    previous_mask = (
        ages.gt(window_hours)
        & ages.le(window_hours * 2)
    )

    current_df = enriched[current_mask].copy()
    previous_df = enriched[previous_mask].copy()

    current_total = max(
        len(current_df),
        1,
    )

    current_stats: dict[
        str,
        dict[str, Any],
    ] = defaultdict(
        lambda: {
            "count": 0,
            "sources": set(),
            "ages": [],
            "stress": [],
        }
    )

    previous_counts: dict[str, int] = (
        defaultdict(int)
    )

    for _, row in current_df.iterrows():
        for label in iter_labels(
            row,
            column,
            json_list=json_list,
        ):
            stats = current_stats[label]

            stats["count"] += 1

            source = str(
                row.get("fuente", "")
            ).strip()

            if source and source.lower() != "nan":
                stats["sources"].add(source)

            age = pd.to_numeric(
                row.get("age_hours"),
                errors="coerce",
            )

            if pd.notna(age):
                stats["ages"].append(
                    float(age)
                )

            stress = pd.to_numeric(
                row.get("stress_score"),
                errors="coerce",
            )

            if pd.notna(stress):
                stats["stress"].append(
                    float(stress)
                )

    for _, row in previous_df.iterrows():
        for label in iter_labels(
            row,
            column,
            json_list=json_list,
        ):
            previous_counts[label] += 1

    ranking: list[dict[str, Any]] = []

    for label, stats in current_stats.items():
        current_count = int(
            stats["count"]
        )

        previous_count = int(
            previous_counts.get(label, 0)
        )

        unique_sources = len(
            stats["sources"]
        )

        frequency_component = clamp(
            current_count / current_total
        )

        diversity_denominator = max(
            min(current_count, 5),
            1,
        )

        diversity_component = clamp(
            unique_sources
            / diversity_denominator
        )

        if stats["ages"]:
            recency_component = sum(
                math.exp(
                    -age / window_hours
                )
                for age in stats["ages"]
            ) / len(stats["ages"])
        else:
            recency_component = 0.0

        if stats["stress"]:
            stress_average = sum(
                stats["stress"]
            ) / len(stats["stress"])
        else:
            stress_average = 0.0

        stress_component = clamp(
            stress_average / 100.0
        )

        momentum_component = (
            normalized_momentum(
                current_count,
                previous_count,
            )
        )

        score = (
            frequency_component
            * weights.frequency
            + diversity_component
            * weights.source_diversity
            + recency_component
            * weights.recency
            + stress_component
            * weights.stress
            + momentum_component
            * weights.momentum
        ) * 100.0

        ranking.append(
            {
                "name": label,
                "score": round(score, 2),
                "current_count": current_count,
                "previous_count": previous_count,
                "unique_sources": unique_sources,
                "average_age_hours": round(
                    sum(stats["ages"])
                    / len(stats["ages"]),
                    2,
                ) if stats["ages"] else None,
                "average_stress": round(
                    stress_average,
                    2,
                ),
                "momentum": momentum_label(
                    current_count,
                    previous_count,
                ),
                "momentum_ratio": round(
                    (
                        current_count + 1
                    ) / (
                        previous_count + 1
                    ),
                    2,
                ),
            }
        )

    return sorted(
        ranking,
        key=lambda item: (
            item["score"],
            item["current_count"],
            item["unique_sources"],
        ),
        reverse=True,
    )[:limit]


def build_relevance_rankings(
    historical_df: pd.DataFrame,
    *,
    window_hours: int = 72,
    limit: int = 10,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    topic_ranking = rank_dimension(
        historical_df,
        column="tema",
        json_list=False,
        window_hours=window_hours,
        limit=limit,
        now=now,
    )

    indicator_ranking = rank_dimension(
        historical_df,
        column="indicadores",
        json_list=True,
        window_hours=window_hours,
        limit=limit,
        now=now,
    )

    entity_ranking = rank_dimension(
        historical_df,
        column="entidades",
        json_list=True,
        window_hours=window_hours,
        limit=limit,
        now=now,
    )

    return {
        "window_hours": window_hours,
        "topics": topic_ranking,
        "indicators": indicator_ranking,
        "entities": entity_ranking,
        "top_topic": (
            topic_ranking[0]
            if topic_ranking
            else None
        ),
        "top_indicator": (
            indicator_ranking[0]
            if indicator_ranking
            else None
        ),
        "top_entity": (
            entity_ranking[0]
            if entity_ranking
            else None
        ),
    }
