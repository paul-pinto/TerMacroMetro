from __future__ import annotations

from src.build_dashboard_timeseries import (
    parse_spanish_date,
)


def test_fecha_iso_no_invierte_mes_y_dia() -> None:
    result = parse_spanish_date(
        "2026-07-10T18:00:43+00:00"
    )

    assert result is not None
    assert result.year == 2026
    assert result.month == 7
    assert result.day == 10


def test_fecha_iso_utc_se_convierte_a_bolivia() -> None:
    result = parse_spanish_date(
        "2026-07-11T00:43:00+00:00"
    )

    assert result is not None
    assert result.year == 2026
    assert result.month == 7
    assert result.day == 10


def test_fecha_espanol() -> None:
    result = parse_spanish_date(
        "10 de julio de 2026"
    )

    assert result is not None
    assert result.year == 2026
    assert result.month == 7
    assert result.day == 10


def test_fecha_numerica_dayfirst() -> None:
    result = parse_spanish_date(
        "10/07/2026"
    )

    assert result is not None
    assert result.month == 7
    assert result.day == 10
