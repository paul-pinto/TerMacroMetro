from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import pandas as pd
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotation"

BATCH_PATH = (
    ANNOTATION_DIR
    / "termacrometro_annotation_batch_001.csv"
)

ALLOWED_LABELS = {
    "positivo",
    "neutral",
    "negativo",
}

CSV_LOCK = Lock()


router = APIRouter(
    prefix="/api/annotation",
    tags=["Anotación"],
)


class AnnotationUpdate(BaseModel):
    sentimiento: str = Field(
        ...,
        description="positivo, neutral o negativo",
    )
    anotador: str = Field(
        ...,
        min_length=2,
        max_length=120,
    )
    duda: int = Field(
        default=0,
        ge=0,
        le=1,
    )
    comentario: str = Field(
        default="",
        max_length=2000,
    )


def require_annotation_key(
    x_annotation_key: str | None,
) -> None:
    expected = os.environ.get(
        "TERMACROMETRO_ANNOTATION_KEY",
        "",
    ).strip()

    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "La interfaz de anotación no está configurada."
            ),
        )

    received = (x_annotation_key or "").strip()

    if received != expected:
        raise HTTPException(
            status_code=401,
            detail="Clave de anotación inválida.",
        )


def load_batch() -> pd.DataFrame:
    if not BATCH_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No existe el lote de anotación: "
                f"{BATCH_PATH.name}"
            ),
        )

    try:
        return pd.read_csv(
            BATCH_PATH,
            dtype={
                "annotation_id": "string",
                "id": "string",
                "sentimiento": "string",
                "anotador": "string",
                "fecha_anotacion": "string",
                "comentario": "string",
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo leer el lote: {exc}",
        ) from exc


def normalize_optional(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value)


def serialize_row(row: pd.Series) -> dict[str, Any]:
    return {
        column: (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for column, value in row.items()
    }


def annotation_mask(df: pd.DataFrame) -> pd.Series:
    values = (
        df["sentimiento"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return values.isin(ALLOWED_LABELS)


def save_batch_atomic(df: pd.DataFrame) -> None:
    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=ANNOTATION_DIR,
            prefix=".annotation-",
            suffix=".csv.tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

            df.to_csv(
                temporary,
                index=False,
            )

            temporary.flush()
            os.fsync(temporary.fileno())

        temporary_path.replace(BATCH_PATH)

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink(missing_ok=True)


@router.get("/status")
def get_status(
    x_annotation_key: str | None = Header(
        default=None,
    ),
) -> dict[str, Any]:
    require_annotation_key(x_annotation_key)

    with CSV_LOCK:
        df = load_batch()

    annotated = annotation_mask(df)
    doubtful = (
        pd.to_numeric(
            df.get("duda", 0),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    distribution = (
        df.loc[annotated, "sentimiento"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()
    )

    return {
        "batch": BATCH_PATH.name,
        "total": int(len(df)),
        "annotated": int(annotated.sum()),
        "pending": int((~annotated).sum()),
        "doubtful": int(doubtful.sum()),
        "progress": round(
            float(annotated.mean() * 100)
            if len(df)
            else 0.0,
            2,
        ),
        "distribution": {
            str(label): int(count)
            for label, count in distribution.items()
        },
    }


@router.get("/items")
def get_items(
    state: str = Query(
        default="pending",
        pattern="^(pending|annotated|all|doubtful)$",
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    x_annotation_key: str | None = Header(
        default=None,
    ),
) -> dict[str, Any]:
    require_annotation_key(x_annotation_key)

    with CSV_LOCK:
        df = load_batch()

    annotated = annotation_mask(df)

    if state == "pending":
        filtered = df.loc[~annotated].copy()
    elif state == "annotated":
        filtered = df.loc[annotated].copy()
    elif state == "doubtful":
        doubt = (
            pd.to_numeric(
                df.get("duda", 0),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        filtered = df.loc[doubt].copy()
    else:
        filtered = df.copy()

    page = filtered.iloc[offset:offset + limit]

    return {
        "state": state,
        "total": int(len(filtered)),
        "offset": offset,
        "limit": limit,
        "items": [
            serialize_row(row)
            for _, row in page.iterrows()
        ],
    }


@router.get("/items/{annotation_id}")
def get_item(
    annotation_id: str,
    x_annotation_key: str | None = Header(
        default=None,
    ),
) -> dict[str, Any]:
    require_annotation_key(x_annotation_key)

    with CSV_LOCK:
        df = load_batch()

    matches = df[
        df["annotation_id"].astype(str) == annotation_id
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="No existe esa anotación.",
        )

    return serialize_row(matches.iloc[0])


@router.put("/items/{annotation_id}")
def update_item(
    annotation_id: str,
    payload: AnnotationUpdate,
    x_annotation_key: str | None = Header(
        default=None,
    ),
) -> dict[str, Any]:
    require_annotation_key(x_annotation_key)

    label = payload.sentimiento.strip().lower()

    if label not in ALLOWED_LABELS:
        raise HTTPException(
            status_code=422,
            detail=(
                "El sentimiento debe ser positivo, "
                "neutral o negativo."
            ),
        )

    annotator = payload.anotador.strip()

    with CSV_LOCK:
        df = load_batch()

        matches = (
            df["annotation_id"].astype(str)
            == annotation_id
        )

        if not matches.any():
            raise HTTPException(
                status_code=404,
                detail="No existe esa anotación.",
            )

        index = df.index[matches][0]

        df.at[index, "sentimiento"] = label
        df.at[index, "anotador"] = annotator
        df.at[index, "fecha_anotacion"] = (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        df.at[index, "duda"] = int(payload.duda)
        df.at[index, "comentario"] = (
            payload.comentario.strip()
        )

        save_batch_atomic(df)

        updated = serialize_row(df.loc[index])

    return {
        "saved": True,
        "item": updated,
    }


@router.delete("/items/{annotation_id}")
def clear_item(
    annotation_id: str,
    x_annotation_key: str | None = Header(
        default=None,
    ),
) -> dict[str, Any]:
    require_annotation_key(x_annotation_key)

    with CSV_LOCK:
        df = load_batch()

        matches = (
            df["annotation_id"].astype(str)
            == annotation_id
        )

        if not matches.any():
            raise HTTPException(
                status_code=404,
                detail="No existe esa anotación.",
            )

        index = df.index[matches][0]

        df.at[index, "sentimiento"] = ""
        df.at[index, "anotador"] = ""
        df.at[index, "fecha_anotacion"] = ""
        df.at[index, "duda"] = 0
        df.at[index, "comentario"] = ""

        save_batch_atomic(df)

    return {
        "cleared": True,
        "annotation_id": annotation_id,
    }
