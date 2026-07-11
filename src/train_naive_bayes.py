from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocessing import preparar_texto_clasico


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

LABEL_ORDER = ["negativo", "neutral", "positivo"]


def cargar_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python -m src.dataset"
        )

    df = pd.read_csv(path)

    required_columns = {"texto", "sentimiento"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"El archivo {path} debe contener las columnas: "
            f"{sorted(required_columns)}"
        )

    return df.dropna(subset=["texto", "sentimiento"])


def construir_pipeline(alpha: float = 0.5) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preparar_texto_clasico,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                MultinomialNB(alpha=alpha),
            ),
        ]
    )


def evaluar(
    model: Pipeline,
    x: pd.Series,
    y: pd.Series,
    split_name: str,
) -> dict:
    predictions = model.predict(x)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "classification_report": classification_report(
            y,
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=LABEL_ORDER,
        ).tolist(),
    }

    return metrics


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

    ax.set_title("Matriz de confusión — TF-IDF + Naive Bayes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = cargar_split(TRAIN_PATH)
    validation_df = cargar_split(VALIDATION_PATH)
    test_df = cargar_split(TEST_PATH)

    model = construir_pipeline(alpha=0.5)

    model.fit(
        train_df["texto"],
        train_df["sentimiento"],
    )

    validation_metrics = evaluar(
        model,
        validation_df["texto"],
        validation_df["sentimiento"],
        "validation",
    )

    test_metrics = evaluar(
        model,
        test_df["texto"],
        test_df["sentimiento"],
        "test",
    )

    model_path = MODEL_DIR / "naive_bayes_financial.joblib"
    metrics_path = REPORT_DIR / "naive_bayes_metrics.json"
    confusion_path = REPORT_DIR / "naive_bayes_confusion_matrix.png"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": "TF-IDF + Multinomial Naive Bayes",
                "validation": validation_metrics,
                "test": test_metrics,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    test_predictions = model.predict(test_df["texto"])

    guardar_matriz_confusion(
        test_df["sentimiento"],
        test_predictions,
        confusion_path,
    )

    print("\nModelo guardado en:")
    print(model_path)

    print("\nMétricas de validación:")
    print(
        json.dumps(
            {
                key: round(value, 4)
                for key, value in validation_metrics.items()
                if isinstance(value, float)
            },
            indent=2,
        )
    )

    print("\nMétricas de prueba:")
    print(
        json.dumps(
            {
                key: round(value, 4)
                for key, value in test_metrics.items()
                if isinstance(value, float)
            },
            indent=2,
        )
    )

    print("\nMatriz de confusión:")
    print(confusion_path)


if __name__ == "__main__":
    main()