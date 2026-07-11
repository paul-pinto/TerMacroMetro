from __future__ import annotations

from src.entities import (
    analizar_contexto_boliviano,
    extraer_entidades,
    extraer_indicadores,
)
from src.preprocessing import (
    preparar_texto_clasico,
    preparar_texto_lda,
)


def test_preprocesamiento_clasico_conserva_negacion() -> None:
    texto = (
        "El BCB informó que las reservas no mejoraron "
        "y la inflación subió."
    )

    resultado = preparar_texto_clasico(texto)

    assert "no" in resultado
    assert "reservas" in resultado
    assert "inflación" in resultado


def test_preprocesamiento_lda_elimina_stopwords() -> None:
    texto = (
        "El Banco Central informó que la inflación "
        "subió durante el año."
    )

    resultado = preparar_texto_lda(texto)

    assert "inflación" in resultado
    assert "banco" in resultado
    assert "central" in resultado
    assert "que" not in resultado.split()


def test_sin_minuscula_no_es_entidad() -> None:
    texto = (
        "La economía creció sin aumentar el déficit fiscal."
    )

    entidades = extraer_entidades(texto)

    assert "SIN" not in entidades


def test_sin_mayuscula_es_entidad() -> None:
    texto = (
        "El SIN anunció nuevas facilidades tributarias."
    )

    entidades = extraer_entidades(texto)

    assert "SIN" in entidades


def test_nombre_completo_sin_es_entidad() -> None:
    texto = (
        "El Servicio de Impuestos Nacionales "
        "amplió el plazo de pago."
    )

    entidades = extraer_entidades(texto)

    assert "SIN" in entidades


def test_detecta_bcb_e_indicadores() -> None:
    texto = (
        "El BCB informó que las reservas internacionales "
        "y el tipo de cambio continúan bajo presión."
    )

    entidades = extraer_entidades(texto)
    indicadores = extraer_indicadores(texto)

    assert "BCB" in entidades
    assert "reservas internacionales" in indicadores
    assert "tipo de cambio" in indicadores


def test_contexto_boliviano_incluye_stress() -> None:
    texto = (
        "La escasez de divisas y la caída de las reservas "
        "aumentan la presión económica."
    )

    resultado = analizar_contexto_boliviano(texto)

    assert "stress_lexico" in resultado
    assert resultado["stress_lexico"]["score"] > 50
    assert resultado["stress_lexico"]["nivel"] in {
        "moderado",
        "alto",
        "crítico",
    }
