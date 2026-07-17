from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EventRankingWeights:
    event_frequency: float = 0.30
    source_diversity: float = 0.25
    recency: float = 0.15
    stress: float = 0.10
    momentum: float = 0.20


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

    if isinstance(value, float) and pd.isna(value):
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


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def momentum_ratio(
    current: int,
    previous: int,
) -> float:
    return round(
        (current + 1)
        / (previous + 1),
        2,
    )


def momentum_label(
    current: int,
    previous: int,
) -> str:
    ratio = momentum_ratio(
        current,
        previous,
    )

    if ratio >= 1.5:
        return "emergente"

    if ratio <= 0.70:
        return "en descenso"

    return "estable"


def normalized_momentum(
    current: int,
    previous: int,
) -> float:
    ratio = (
        current + 1
    ) / (
        previous + 1
    )

    logarithm = math.log2(ratio)

    return clamp(
        (logarithm + 2.0)
        / 4.0
    )


def parse_event_date(
    events_df: pd.DataFrame,
) -> pd.Series:
    if "effective_at" not in events_df.columns:
        return pd.Series(
            pd.NaT,
            index=events_df.index,
            dtype="datetime64[ns, UTC]",
        )

    return pd.to_datetime(
        events_df["effective_at"],
        errors="coerce",
        utc=True,
    )


def split_event_windows(
    events_df: pd.DataFrame,
    *,
    window_hours: int,
    now: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = events_df.copy()

    dates = parse_event_date(
        output
    )

    age_hours = (
        now - dates
    ).dt.total_seconds() / 3600.0

    output["_age_hours"] = age_hours

    current = output[
        age_hours.notna()
        & age_hours.ge(0)
        & age_hours.le(window_hours)
    ].copy()

    previous = output[
        age_hours.gt(window_hours)
        & age_hours.le(window_hours * 2)
    ].copy()

    return current, previous


def dimension_value(
    row: pd.Series,
    column: str,
) -> str:
    value = str(
        row.get(column, "")
    ).strip()

    if value.lower() == "nan":
        return ""

    return value


def rank_event_dimension(
    events_df: pd.DataFrame,
    *,
    column: str,
    previous_events_df: pd.DataFrame | None = None,
    limit: int = 10,
    weights: EventRankingWeights | None = None,
    window_hours: int = 72,
) -> list[dict[str, Any]]:
    weights = (
        weights
        or EventRankingWeights()
    )

    if events_df.empty:
        return []

    previous_events_df = (
        previous_events_df
        if previous_events_df is not None
        else pd.DataFrame(
            columns=events_df.columns
        )
    )

    stats: dict[
        str,
        dict[str, Any],
    ] = defaultdict(
        lambda: {
            "events": 0,
            "documents": 0,
            "sources": set(),
            "ages": [],
            "stress": [],
        }
    )

    previous_counts: dict[str, int] = (
        defaultdict(int)
    )

    for _, row in events_df.iterrows():
        label = dimension_value(
            row,
            column,
        )

        if not label:
            continue

        entry = stats[label]

        entry["events"] += 1
        entry["documents"] += int(
            row.get("documents", 1)
        )

        for source in safe_json_list(
            row.get("sources", "[]")
        ):
            entry["sources"].add(source)

        age = pd.to_numeric(
            row.get("_age_hours"),
            errors="coerce",
        )

        if pd.notna(age):
            entry["ages"].append(
                float(age)
            )

        stress = pd.to_numeric(
            row.get("average_stress"),
            errors="coerce",
        )

        if pd.notna(stress):
            entry["stress"].append(
                float(stress)
            )

    for _, row in previous_events_df.iterrows():
        label = dimension_value(
            row,
            column,
        )

        if label:
            previous_counts[label] += 1

    total_events = max(
        len(events_df),
        1,
    )

    ranking: list[dict[str, Any]] = []

    for label, entry in stats.items():
        current_events = int(
            entry["events"]
        )

        previous_events = int(
            previous_counts.get(
                label,
                0,
            )
        )

        unique_sources = len(
            entry["sources"]
        )

        event_frequency_component = clamp(
            current_events
            / total_events
        )

        diversity_denominator = max(
            min(
                current_events * 2,
                6,
            ),
            1,
        )

        source_diversity_component = clamp(
            unique_sources
            / diversity_denominator
        )

        if entry["ages"]:
            recency_component = sum(
                math.exp(
                    -age / window_hours
                )
                for age in entry["ages"]
            ) / len(entry["ages"])
        else:
            recency_component = 0.0

        if entry["stress"]:
            average_stress = sum(
                entry["stress"]
            ) / len(entry["stress"])
        else:
            average_stress = 0.0

        stress_component = clamp(
            average_stress / 100.0
        )

        momentum_component = (
            normalized_momentum(
                current_events,
                previous_events,
            )
        )

        score = (
            event_frequency_component
            * weights.event_frequency
            + source_diversity_component
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
                "score": round(
                    score,
                    2,
                ),
                "current_events": (
                    current_events
                ),
                "previous_events": (
                    previous_events
                ),
                "documents": int(
                    entry["documents"]
                ),
                "unique_sources": (
                    unique_sources
                ),
                "average_age_hours": (
                    round(
                        sum(entry["ages"])
                        / len(entry["ages"]),
                        2,
                    )
                    if entry["ages"]
                    else None
                ),
                "average_stress": round(
                    average_stress,
                    2,
                ),
                "momentum": momentum_label(
                    current_events,
                    previous_events,
                ),
                "momentum_ratio": (
                    momentum_ratio(
                        current_events,
                        previous_events,
                    )
                ),
            }
        )

    return sorted(
        ranking,
        key=lambda item: (
            item["score"],
            item["current_events"],
            item["unique_sources"],
        ),
        reverse=True,
    )[:limit]


def build_event_rankings(
    current_events: pd.DataFrame,
    *,
    previous_events: pd.DataFrame | None = None,
    window_hours: int = 72,
    limit: int = 10,
) -> dict[str, Any]:
    topics = rank_event_dimension(
        current_events,
        column="dominant_topic",
        previous_events_df=(
            previous_events
        ),
        limit=limit,
        window_hours=window_hours,
    )

    indicators = rank_event_dimension(
        current_events,
        column="dominant_indicator",
        previous_events_df=(
            previous_events
        ),
        limit=limit,
        window_hours=window_hours,
    )

    entities = rank_event_dimension(
        current_events,
        column="dominant_entity",
        previous_events_df=(
            previous_events
        ),
        limit=limit,
        window_hours=window_hours,
    )

    return {
        "window_hours": window_hours,
        "ranking_unit": "economic_events",
        "topics": topics,
        "indicators": indicators,
        "entities": entities,
        "top_topic": (
            topics[0]
            if topics
            else None
        ),
        "top_indicator": (
            indicators[0]
            if indicators
            else None
        ),
        "top_entity": (
            entities[0]
            if entities
            else None
        ),
    }
