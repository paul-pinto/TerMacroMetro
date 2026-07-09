"""
Inferencia: carga el modelo entrenado y predice el sentimiento de un texto.

Expone una clase `Predictor` reutilizable (la usa la API) y un modo CLI para
probar rápido desde la terminal.

Uso CLI:
    python -m src.predecir "El celular es excelente, lo recomiendo"
    python -m src.predecir            # modo interactivo
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib

RUTA_MODELO_DEFECTO = "modelos/modelo_sentimiento.joblib"


class Predictor:
    """Envuelve el pipeline serializado y ofrece predicción con probabilidades."""

    def __init__(self, ruta_modelo: str = RUTA_MODELO_DEFECTO):
        ruta = Path(ruta_modelo)
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en '{ruta}'. "
                f"Ejecuta primero: python -m src.entrenar"
            )
        self.pipeline = joblib.load(ruta)
        self.clases = list(self.pipeline.named_steps["clasificador"].classes_)

    def predecir(self, texto: str) -> dict:
        """Devuelve la etiqueta predicha y la probabilidad de cada clase."""
        etiqueta = self.pipeline.predict([texto])[0]
        probas = self.pipeline.predict_proba([texto])[0]
        distribucion = {
            clase: round(float(p), 4)
            for clase, p in zip(self.clases, probas)
        }
        return {
            "texto": texto,
            "sentimiento": etiqueta,
            "confianza": round(float(max(probas)), 4),
            "probabilidades": distribucion,
        }


@lru_cache(maxsize=1)
def obtener_predictor(ruta_modelo: str = RUTA_MODELO_DEFECTO) -> Predictor:
    """Carga el predictor una sola vez (cache) para no releer el modelo."""
    return Predictor(ruta_modelo)


def _cli() -> None:
    predictor = obtener_predictor()
    if len(sys.argv) > 1:
        texto = " ".join(sys.argv[1:])
        _imprimir(predictor.predecir(texto))
        return

    print("Modo interactivo. Escribe una reseña (o 'salir'):")
    while True:
        try:
            texto = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if texto.lower() in {"salir", "exit", "quit", ""}:
            break
        _imprimir(predictor.predecir(texto))


def _imprimir(resultado: dict) -> None:
    print(f"  Sentimiento: {resultado['sentimiento'].upper()} "
          f"(confianza {resultado['confianza']:.0%})")
    for clase, p in sorted(resultado["probabilidades"].items(),
                           key=lambda x: -x[1]):
        print(f"    {clase:>9}: {p:.1%}")


if __name__ == "__main__":
    _cli()
