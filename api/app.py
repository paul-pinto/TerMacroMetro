"""
API REST del clasificador de sentimiento + servidor de la interfaz web.

Construida con FastAPI. Expone:

    GET  /            -> interfaz web (HTML)
    GET  /salud       -> estado del servicio y del modelo
    GET  /metricas    -> métricas de evaluación del modelo entrenado
    POST /predecir    -> {"texto": "..."} -> predicción de sentimiento
    POST /predecir_lote -> {"textos": ["...", "..."]} -> predicciones en lote

Ejecutar:
    uvicorn api.app:app --reload
    # o bien:  python -m api.app
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predecir import obtener_predictor

# Rutas base del proyecto (independientes del directorio desde donde se ejecute)
RAIZ = Path(__file__).resolve().parent.parent
DIR_WEB = RAIZ / "web"
RUTA_METRICAS = RAIZ / "modelos" / "metricas.json"

app = FastAPI(
    title="API de Análisis de Sentimiento",
    description="Clasifica reseñas de productos en español: positivo, negativo o neutral.",
    version="1.0.0",
)

# CORS abierto para facilitar pruebas desde cualquier origen (uso educativo).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Esquemas de entrada/salida (Pydantic) ---------------------------------

class EntradaTexto(BaseModel):
    texto: str = Field(..., min_length=1, examples=["El celular es excelente"])


class EntradaLote(BaseModel):
    textos: list[str] = Field(..., min_length=1)


# --- Carga perezosa del predictor ------------------------------------------

def _predictor():
    try:
        return obtener_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# --- Endpoints de API -------------------------------------------------------

@app.get("/salud", tags=["sistema"])
def salud():
    """Estado del servicio y si el modelo está cargado."""
    modelo_ok = (RAIZ / "modelos" / "modelo_sentimiento.joblib").exists()
    return {"estado": "ok", "modelo_cargado": modelo_ok}


@app.get("/metricas", tags=["sistema"])
def metricas():
    """Devuelve las métricas de evaluación guardadas durante el entrenamiento."""
    if not RUTA_METRICAS.exists():
        raise HTTPException(status_code=404, detail="Aún no hay métricas. Entrena el modelo.")
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


@app.post("/predecir", tags=["inferencia"])
def predecir(entrada: EntradaTexto):
    """Predice el sentimiento de un único texto."""
    return _predictor().predecir(entrada.texto)


@app.post("/predecir_lote", tags=["inferencia"])
def predecir_lote(entrada: EntradaLote):
    """Predice el sentimiento de una lista de textos."""
    predictor = _predictor()
    return {"resultados": [predictor.predecir(t) for t in entrada.textos]}


# --- Interfaz web (archivos estáticos) -------------------------------------

@app.get("/", include_in_schema=False)
def raiz():
    return FileResponse(DIR_WEB / "index.html")


# Monta el resto de archivos estáticos (CSS, JS) bajo /static
app.mount("/static", StaticFiles(directory=DIR_WEB), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
