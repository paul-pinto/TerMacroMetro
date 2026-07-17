from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.v2 import router as v2_router

from src.inference import EconomicAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NB_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "naive_bayes_metrics.json"
)

TRANSFORMER_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "transformer_metrics.json"
)

LDA_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "lda_bolivia_topics.json"
)

OBSERVATORY_INDEX_PATH = (
    PROJECT_ROOT
    / "web"
    / "observatory"
    / "index.html"
)

BOLIVIA_DASHBOARD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "bolivia_dashboard.json"
)

PULSOBO_HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "history"
    / "pulsobo_daily.json"
)

NAIVE_BAYES_CONFUSION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "naive_bayes_confusion_matrix.png"
)

TRANSFORMER_CONFUSION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "transformer_confusion_matrix.png"
)


analyzer: EconomicAnalyzer | None = None


class AnalyzeRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=10,
        max_length=20_000,
        description="Noticia o texto económico en español.",
        examples=[
            (
                "El BCB informó que las reservas internacionales "
                "continúan bajo presión por la escasez de divisas."
            )
        ],
    )


class BatchAnalyzeRequest(BaseModel):
    textos: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Lista de textos económicos para análisis por lote."
        ),
    )


def cargar_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer

    try:
        load_transformer = (
            os.environ.get(
                "PULSOBO_DISABLE_TRANSFORMER",
                "0",
            )
            != "1"
        )

        analyzer = EconomicAnalyzer(
            cargar_transformer=load_transformer,
        )

        app.state.model_ready = True
        app.state.model_error = None

    except Exception as exc:
        analyzer = None

        app.state.model_ready = False
        app.state.model_error = str(exc)

    yield

    analyzer = None


app = FastAPI(
    title="TerMacroMetro API",
    description=(
        "API del Observatorio Económico Inteligente de Bolivia. "
        "Integra TF-IDF + Naive Bayes, Transformer financiero "
        "en español, LDA, extracción de entidades e índice "
        "experimental de tensión textual."
    ),
    version="1.0.1",
    lifespan=lifespan,
)



app.mount(
    "/assets",
    StaticFiles(
        directory=str(
            Path(__file__).resolve().parents[1]
            / "web"
            / "observatory"
            / "assets"
        )
    ),
    name="assets",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def obtener_analizador() -> EconomicAnalyzer:
    if analyzer is None:
        detail = getattr(
            app.state,
            "model_error",
            "Los modelos todavía no están disponibles.",
        )

        raise HTTPException(
            status_code=503,
            detail=f"Analizador no disponible: {detail}",
        )

    return analyzer


@app.get(
    "/",
    include_in_schema=False,
)
def root() -> FileResponse:
    if not OBSERVATORY_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No se encontró la interfaz del Observatorio en: "
                f"{OBSERVATORY_INDEX_PATH}"
            ),
        )

    return FileResponse(
        path=OBSERVATORY_INDEX_PATH,
        media_type="text/html",
        filename=None,
    )


@app.get(
    "/api",
    include_in_schema=False,
)
def api_root() -> dict[str, Any]:
    return {
        "name": "TerMacroMetro API",
        "version": "1.0.1",
        "status": "online",
        "model_ready": getattr(
            app.state,
            "model_ready",
            False,
        ),
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze",
            "batch": "/api/analyze/batch",
            "models": "/api/models",
            "topics": "/api/topics",
        },
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = getattr(
        app.state,
        "model_ready",
        False,
    )

    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "error": getattr(
            app.state,
            "model_error",
            None,
        ),
    }


@app.post("/api/analyze")
def analyze(
    payload: AnalyzeRequest,
) -> dict[str, Any]:
    economic_analyzer = obtener_analizador()

    try:
        return economic_analyzer.analizar(
            payload.texto
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Error durante el análisis: "
                f"{exc}"
            ),
        ) from exc


@app.post("/api/analyze/batch")
def analyze_batch(
    payload: BatchAnalyzeRequest,
) -> dict[str, Any]:
    economic_analyzer = obtener_analizador()

    valid_texts = [
        str(text).strip()
        for text in payload.textos
        if str(text).strip()
    ]

    if not valid_texts:
        raise HTTPException(
            status_code=422,
            detail="No se recibieron textos válidos.",
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, text in enumerate(valid_texts):
        try:
            result = economic_analyzer.analizar(
                text
            )

            result["index"] = index
            results.append(result)

        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "texto": text,
                    "error": str(exc),
                }
            )

    sentiment_counts = {
        "favorable": 0,
        "neutral": 0,
        "desfavorable": 0,
    }

    topic_counts: dict[str, int] = {}
    entity_counts: dict[str, int] = {}
    indicator_counts: dict[str, int] = {}

    stress_scores: list[float] = []
    model_agreements = 0
    model_disagreements = 0

    for result in results:
        consolidated = result[
            "resultado_consolidado"
        ]

        tone = consolidated[
            "tono_informativo"
        ]

        sentiment_counts[tone] = (
            sentiment_counts.get(tone, 0) + 1
        )

        agreement = consolidated.get(
            "coincidencia_modelos"
        )

        if agreement is True:
            model_agreements += 1
        elif agreement is False:
            model_disagreements += 1

        topic_name = result["tema"]["nombre"]

        topic_counts[topic_name] = (
            topic_counts.get(topic_name, 0) + 1
        )

        context = result[
            "contexto_boliviano"
        ]

        for entity in context["entidades"]:
            entity_counts[entity] = (
                entity_counts.get(entity, 0) + 1
            )

        for indicator in context["indicadores"]:
            indicator_counts[indicator] = (
                indicator_counts.get(
                    indicator,
                    0,
                )
                + 1
            )

        stress_scores.append(
            float(
                context[
                    "stress_lexico"
                ]["score"]
            )
        )

    average_stress = (
        round(
            sum(stress_scores)
            / len(stress_scores),
            2,
        )
        if stress_scores
        else 0.0
    )

    return {
        "summary": {
            "received": len(valid_texts),
            "processed": len(results),
            "failed": len(errors),
            "sentiments": sentiment_counts,
            "topics": dict(
                sorted(
                    topic_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
            "entities": dict(
                sorted(
                    entity_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
            "indicators": dict(
                sorted(
                    indicator_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
            "average_stress": average_stress,
            "model_agreements": model_agreements,
            "model_disagreements": (
                model_disagreements
            ),
        },
        "results": results,
        "errors": errors,
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    nb_metrics = cargar_json(
        NB_METRICS_PATH
    )

    transformer_metrics = cargar_json(
        TRANSFORMER_METRICS_PATH
    )

    lda_metrics = cargar_json(
        LDA_REPORT_PATH
    )

    nb_test = nb_metrics.get(
        "test",
        {},
    )

    return {
        "classic_model": {
            "name": nb_metrics.get(
                "model",
                (
                    "TF-IDF + Multinomial "
                    "Naive Bayes"
                ),
            ),
            "accuracy": nb_test.get(
                "accuracy"
            ),
            "precision_macro": nb_test.get(
                "precision_macro"
            ),
            "recall_macro": nb_test.get(
                "recall_macro"
            ),
            "f1_macro": nb_test.get(
                "f1_macro"
            ),
            "status": (
                "ready"
                if nb_metrics
                else "missing"
            ),
        },
        "transformer_model": {
            "name": transformer_metrics.get(
                "model",
                (
                    "Transformer financiero "
                    "en español"
                ),
            ),
            "accuracy": transformer_metrics.get(
                "accuracy"
            ),
            "precision_macro": (
                transformer_metrics.get(
                    "precision_macro"
                )
            ),
            "recall_macro": (
                transformer_metrics.get(
                    "recall_macro"
                )
            ),
            "f1_macro": transformer_metrics.get(
                "f1_macro"
            ),
            "mean_confidence": (
                transformer_metrics.get(
                    "mean_confidence"
                )
            ),
            "status": (
                "ready"
                if transformer_metrics
                else "missing"
            ),
        },
        "topic_model": {
            "name": lda_metrics.get(
                "model",
                "Latent Dirichlet Allocation",
            ),
            "topics": lda_metrics.get(
                "n_topics"
            ),
            "documents": lda_metrics.get(
                "n_documents"
            ),
            "vocabulary_size": (
                lda_metrics.get(
                    "vocabulary_size"
                )
            ),
            "perplexity": lda_metrics.get(
                "perplexity"
            ),
            "status": (
                "ready"
                if lda_metrics
                else "missing"
            ),
        },
    }


@app.get("/api/topics")
def topics() -> dict[str, Any]:
    report = cargar_json(
        LDA_REPORT_PATH
    )

    return {
        "n_topics": report.get(
            "n_topics",
            0,
        ),
        "topics": report.get(
            "topics",
            [],
        ),
        "distribution": report.get(
            "topic_distribution",
            {},
        ),
        "perplexity": report.get(
            "perplexity"
        ),
    }

@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    report = cargar_json(
        BOLIVIA_DASHBOARD_PATH
    )

    if not report:
        raise HTTPException(
            status_code=404,
            detail=(
                "No existe el resumen agregado. "
                "Ejecuta: python -m src.analyze_bolivia_corpus"
            ),
        )

    return report



@app.get(
    "/api/reports/naive-bayes-confusion",
    include_in_schema=False,
)
def naive_bayes_confusion() -> FileResponse:
    if not NAIVE_BAYES_CONFUSION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No existe la matriz de confusión de Naive Bayes.",
        )

    return FileResponse(
        path=NAIVE_BAYES_CONFUSION_PATH,
        media_type="image/png",
    )


@app.get(
    "/api/reports/transformer-confusion",
    include_in_schema=False,
)
def transformer_confusion() -> FileResponse:
    if not TRANSFORMER_CONFUSION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No existe la matriz de confusión del Transformer.",
        )

    return FileResponse(
        path=TRANSFORMER_CONFUSION_PATH,
        media_type="image/png",
    )



@app.get("/api/history")
def history() -> dict[str, Any]:
    if not PULSOBO_HISTORY_PATH.exists():
        return {
            "count": 0,
            "records": [],
        }

    try:
        with PULSOBO_HISTORY_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo leer el histórico: {exc}",
        ) from exc

    if not isinstance(records, list):
        records = []

    return {
        "count": len(records),
        "records": records,
    }


app.include_router(v2_router)
