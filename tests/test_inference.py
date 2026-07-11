from __future__ import annotations

import pytest

from src.inference import EconomicAnalyzer


@pytest.fixture(scope="session")
def analyzer() -> EconomicAnalyzer:
    return EconomicAnalyzer(
        cargar_transformer=False,
    )


def test_analizador_rechaza_texto_corto(
    analyzer: EconomicAnalyzer,
) -> None:
    with pytest.raises(ValueError):
        analyzer.analizar("Corto")


def test_analizador_clasico_devuelve_estructura(
    analyzer: EconomicAnalyzer,
) -> None:
    texto = (
        "El BCB informó que la inflación y la escasez "
        "de divisas generan presión económica."
    )

    resultado = analyzer.analizar(texto)

    assert "resultado_consolidado" in resultado
    assert "modelos" in resultado
    assert "tema" in resultado
    assert "contexto_boliviano" in resultado

    assert "naive_bayes" in resultado["modelos"]
    assert "transformer" in resultado["modelos"]


def test_tema_pertenece_a_taxonomia_boliviana(
    analyzer: EconomicAnalyzer,
) -> None:
    texto = (
        "El tipo de cambio y la falta de dólares "
        "afectan la importación de combustibles."
    )

    resultado = analyzer.analizar(texto)

    temas_validos = {
        "Sector productivo, automotor y actividad empresarial",
        "Tipo de cambio, divisas y abastecimiento de combustibles",
        "Política fiscal, gestión pública y sistema financiero",
        "Precios, alimentos y servicios básicos",
        "Producción, inversión y régimen cambiario",
        "Comercio exterior, desarrollo y empresas",
    }

    assert resultado["tema"]["nombre"] in temas_validos


def test_detecta_contexto_boliviano(
    analyzer: EconomicAnalyzer,
) -> None:
    texto = (
        "YPFB anunció una mejora en la provisión "
        "de diésel y gasolina."
    )

    resultado = analyzer.analizar(texto)
    contexto = resultado["contexto_boliviano"]

    assert "YPFB" in contexto["entidades"]
    assert "combustibles" in contexto["indicadores"]
