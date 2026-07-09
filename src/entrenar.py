"""
Entrenamiento del pipeline de análisis de sentimiento.

Construye un `Pipeline` de scikit-learn que encadena:

    normalizar (preprocesamiento) -> TfidfVectorizer -> LogisticRegression

Ventaja de usar `Pipeline`: el objeto serializado ya contiene TODAS las etapas,
así que en inferencia basta con llamar `.predict(["texto crudo"])` y el mismo
preprocesamiento y vectorización se aplican automáticamente. Esto evita el clásico
error de "entrenar con un preprocesamiento y predecir con otro".

Uso:
    python -m src.entrenar
    python -m src.entrenar --datos data/resenas.csv --modelo modelos/modelo.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocesamiento import normalizar


def construir_pipeline() -> Pipeline:
    """Crea el pipeline completo: preprocesamiento + TF-IDF + clasificador."""
    return Pipeline(steps=[
        # 1) Vectorización TF-IDF. El `preprocessor` recibe cada documento crudo
        #    y le aplica nuestra función `normalizar` antes de tokenizar.
        ("tfidf", TfidfVectorizer(
            preprocessor=normalizar,
            ngram_range=(1, 2),      # unigramas + bigramas (capta "no recomiendo")
            min_df=2,                # ignora términos que aparecen en <2 documentos
            sublinear_tf=True,       # suaviza la frecuencia de término
        )),
        # 2) Clasificador lineal, robusto y rápido para texto disperso.
        ("clasificador", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de sentimiento.")
    parser.add_argument("--datos", default="data/resenas.csv")
    parser.add_argument("--modelo", default="modelos/modelo_sentimiento.joblib")
    parser.add_argument("--metricas", default="modelos/metricas.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    # --- 1. Cargar datos ----------------------------------------------------
    df = pd.read_csv(args.datos)
    df = df.dropna(subset=["texto", "etiqueta"])
    print(f"[1/5] Datos cargados: {len(df)} reseñas")
    print(df["etiqueta"].value_counts().to_string())

    X = df["texto"].astype(str)
    y = df["etiqueta"].astype(str)

    # --- 2. Partición train / test ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y,
    )
    print(f"\n[2/5] Partición -> train: {len(X_train)}  test: {len(X_test)}")

    # --- 3. Entrenar --------------------------------------------------------
    pipeline = construir_pipeline()
    pipeline.fit(X_train, y_train)
    print("[3/5] Modelo entrenado")

    # --- 4. Evaluar ---------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

    print("\n[4/5] Evaluación sobre el conjunto de prueba:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print(f"Etiquetas: {sorted(y.unique())}")
    print(matriz)

    # --- 5. Guardar modelo y métricas ---------------------------------------
    Path(args.modelo).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.modelo)

    metricas = {
        "exactitud": reporte["accuracy"],
        "f1_macro": reporte["macro avg"]["f1-score"],
        "reporte_por_clase": {
            k: v for k, v in reporte.items()
            if k in sorted(y.unique())
        },
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "clases": sorted(y.unique()),
    }
    with open(args.metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"\n[5/5] Modelo guardado en   : {args.modelo}")
    print(f"      Métricas guardadas en: {args.metricas}")
    print(f"      Exactitud: {metricas['exactitud']:.3f} | "
          f"F1 macro: {metricas['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
