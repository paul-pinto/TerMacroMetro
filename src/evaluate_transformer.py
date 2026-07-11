from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "transformer_financial_es"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "bardsai/finance-sentiment-es-base"
LABEL_ORDER = ["negativo", "neutral", "positivo"]

LABEL_MAP = {
    "negative": "negativo",
    "neutral": "neutral",
    "positive": "positivo",
    "NEGATIVE": "negativo",
    "NEUTRAL": "neutral",
    "POSITIVE": "positivo",
}

BATCH_SIZE = 1
MAX_LENGTH = 128


def cargar_test() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"No existe {TEST_PATH}. Ejecuta primero: python -m src.dataset"
        )

    df = pd.read_csv(TEST_PATH)

    required = {"texto", "sentimiento"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"El archivo debe contener las columnas: {sorted(required)}"
        )

    df = df.dropna(subset=["texto", "sentimiento"]).copy()
    df["texto"] = df["texto"].astype(str)
    df["sentimiento"] = df["sentimiento"].astype(str)

    return df.reset_index(drop=True)


def resolver_etiqueta(
    raw_label: str,
    id2label: dict[int, str],
) -> str:
    label = str(raw_label).strip()

    if label in LABEL_MAP:
        return LABEL_MAP[label]

    label_lower = label.lower()

    if label_lower in LABEL_MAP:
        return LABEL_MAP[label_lower]

    if label_lower.startswith("label_"):
        label_id = int(label_lower.split("_")[-1])
        mapped = id2label.get(label_id, label)

        if mapped in LABEL_MAP:
            return LABEL_MAP[mapped]

        mapped_lower = str(mapped).lower()

        if mapped_lower in LABEL_MAP:
            return LABEL_MAP[mapped_lower]

    raise ValueError(
        f"No se pudo mapear la etiqueta del modelo: {raw_label!r}"
    )


def predecir_lote(
    textos: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
) -> tuple[list[str], list[float]]:
    predictions: list[str] = []
    confidences: list[float] = []

    id2label = {
        int(key): value
        for key, value in model.config.id2label.items()
    }

    model.eval()

    for start in range(0, len(textos), BATCH_SIZE):
        batch_texts = textos[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(**encoded)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

        batch_confidences, batch_ids = torch.max(
            probabilities,
            dim=-1,
        )

        for class_id, confidence in zip(
            batch_ids.cpu().tolist(),
            batch_confidences.cpu().tolist(),
            strict=True,
        ):
            raw_label = id2label[class_id]

            predictions.append(
                resolver_etiqueta(
                    raw_label=raw_label,
                    id2label=id2label,
                )
            )

            confidences.append(float(confidence))

        processed = min(start + BATCH_SIZE, len(textos))

        print(
            f"Procesados: {processed}/{len(textos)}",
            end="\r",
        )

    print()

    return predictions, confidences


def guardar_matriz_confusion(
    y_true: pd.Series,
    y_pred: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABEL_ORDER,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Matriz de confusión — Transformer financiero en español"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_test()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Dispositivo: {device}")
    print(f"Modelo: {MODEL_NAME}")
    print(f"Textos de prueba: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
    )

    model.to(device)

    predictions, confidences = predecir_lote(
        textos=df["texto"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    metrics = {
        "model": MODEL_NAME,
        "device": str(device),
        "samples": len(df),
        "accuracy": accuracy_score(
            df["sentimiento"],
            predictions,
        ),
        "precision_macro": precision_score(
            df["sentimiento"],
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            df["sentimiento"],
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            df["sentimiento"],
            predictions,
            average="macro",
            zero_division=0,
        ),
        "mean_confidence": sum(confidences) / len(confidences),
        "classification_report": classification_report(
            df["sentimiento"],
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            df["sentimiento"],
            predictions,
            labels=LABEL_ORDER,
        ).tolist(),
    }

    metrics_path = REPORT_DIR / "transformer_metrics.json"
    confusion_path = (
        REPORT_DIR / "transformer_confusion_matrix.png"
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    guardar_matriz_confusion(
        y_true=df["sentimiento"],
        y_pred=predictions,
        output_path=confusion_path,
    )

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("\nModelo guardado en:")
    print(MODEL_DIR)

    print("\nMétricas:")
    print(
        json.dumps(
            {
                "accuracy": round(metrics["accuracy"], 4),
                "precision_macro": round(
                    metrics["precision_macro"],
                    4,
                ),
                "recall_macro": round(
                    metrics["recall_macro"],
                    4,
                ),
                "f1_macro": round(
                    metrics["f1_macro"],
                    4,
                ),
                "mean_confidence": round(
                    metrics["mean_confidence"],
                    4,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nMatriz de confusión:")
    print(confusion_path)

    print("\nReporte:")
    print(metrics_path)


if __name__ == "__main__":
    main()

