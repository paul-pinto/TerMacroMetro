from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


DEFAULT_RECENT_WINDOW_HOURS = 72
MAX_FUTURE_TOLERANCE_HOURS = 6


@dataclass(frozen=True)
class TemporalPolicy:
    recent_window_hours: int = DEFAULT_RECENT_WINDOW_HOURS
    future_tolerance_hours: int = MAX_FUTURE_TOLERANCE_HOURS


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(
        datetime.now(timezone.utc)
    )


def parse_datetime_series(
    values: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
    )


def normalize_temporal_columns(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    policy: TemporalPolicy | None = None,
) -> pd.DataFrame:
    """
    Enriquece un DataFrame de noticias con metadatos temporales.

    Columnas generadas:
    - published_at
    - collected_at_normalized
    - effective_at
    - age_hours
    - date_quality
    - is_recent
    - temporal_exclusion_reason
    """

    policy = policy or TemporalPolicy()
    now = now or utc_now()

    output = df.copy()

    if "fecha" in output.columns:
        published = parse_datetime_series(
            output["fecha"]
        )
    else:
        published = pd.Series(
            pd.NaT,
            index=output.index,
            dtype="datetime64[ns, UTC]",
        )

    if "collected_at" in output.columns:
        collected = parse_datetime_series(
            output["collected_at"]
        )
    else:
        collected = pd.Series(
            pd.NaT,
            index=output.index,
            dtype="datetime64[ns, UTC]",
        )

    future_limit = now + pd.Timedelta(
        hours=policy.future_tolerance_hours
    )

    published_is_future = (
        published.notna()
        & (published > future_limit)
    )

    valid_published = (
        published.notna()
        & ~published_is_future
    )

    effective = published.where(
        valid_published,
        collected,
    )

    age_hours = (
        now - effective
    ).dt.total_seconds() / 3600.0

    date_quality = pd.Series(
        "missing",
        index=output.index,
        dtype="object",
    )

    date_quality.loc[valid_published] = (
        "published"
    )

    inferred_mask = (
        ~valid_published
        & collected.notna()
    )

    date_quality.loc[inferred_mask] = (
        "inferred_collected"
    )

    date_quality.loc[published_is_future] = (
        "invalid_future"
    )

    non_negative_age = age_hours >= 0

    recent_mask = (
        effective.notna()
        & non_negative_age
        & (
            age_hours
            <= policy.recent_window_hours
        )
        & ~published_is_future
    )

    exclusion_reason = pd.Series(
        "recent",
        index=output.index,
        dtype="object",
    )

    exclusion_reason.loc[
        effective.isna()
    ] = "missing_date"

    exclusion_reason.loc[
        published_is_future
    ] = "future_date"

    exclusion_reason.loc[
        effective.notna()
        & ~published_is_future
        & (age_hours < 0)
    ] = "future_effective_date"

    exclusion_reason.loc[
        effective.notna()
        & ~published_is_future
        & (
            age_hours
            > policy.recent_window_hours
        )
    ] = "outside_operational_window"

    output["published_at"] = (
        published.dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    output["collected_at_normalized"] = (
        collected.dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    output["effective_at"] = (
        effective.dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    output["age_hours"] = (
        age_hours.round(2)
    )

    output["date_quality"] = date_quality
    output["is_recent"] = recent_mask
    output["temporal_exclusion_reason"] = (
        exclusion_reason
    )

    return output


def build_operational_corpus(
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    policy: TemporalPolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = normalize_temporal_columns(
        df,
        now=now,
        policy=policy,
    )

    operational = (
        enriched[
            enriched["is_recent"]
        ]
        .copy()
        .sort_values(
            [
                "effective_at",
                "stress_score",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return enriched, operational


def temporal_audit(
    enriched: pd.DataFrame,
    operational: pd.DataFrame,
    *,
    policy: TemporalPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or TemporalPolicy()

    exclusion_counts = (
        enriched[
            "temporal_exclusion_reason"
        ]
        .value_counts(dropna=False)
        .to_dict()
    )

    quality_counts = (
        enriched["date_quality"]
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "historical_documents": int(
            len(enriched)
        ),
        "operational_documents": int(
            len(operational)
        ),
        "excluded_documents": int(
            len(enriched) - len(operational)
        ),
        "recent_window_hours": (
            policy.recent_window_hours
        ),
        "date_quality": {
            str(key): int(value)
            for key, value
            in quality_counts.items()
        },
        "exclusion_reasons": {
            str(key): int(value)
            for key, value
            in exclusion_counts.items()
        },
    }
