from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocessing import preparar_texto_lda


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "financial_sentiment_es.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

N_TOPICS = 6
N_TOP_WORDS = 15


def cargar_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe {DATA_PATH}. Ejecuta primero: python -m src.dataset"
        )

    df = pd.read_csv(DATA_PATH)

    if "texto" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'texto'.")

    df = df.dropna(subset=["texto"])
    df["texto"] = df["texto"].astype(str).str.strip()
    df = df[df["texto"].str.len() >= 10]

    return df.reset_index(drop=True)


def extraer_palabras_temas(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    n_words: int = N_TOP_WORDS,
) -> list[dict]:
    feature_names = vectorizer.get_feature_names_out()
    topics: list[dict] = []

    for topic_id, component in enumerate(lda.components_):
        top_indices = component.argsort()[-n_words:][::-1]
        words = [feature_names[index] for index in top_indices]

        topics.append(
            {
                "topic_id": topic_id,
                "keywords": words,
            }
        )

    return topics


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()

    vectorizer = CountVectorizer(
        preprocessor=preparar_texto_lda,
        lowercase=False,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=8,
        max_df=0.70,
        max_features=12_000,
    )

    document_term_matrix = vectorizer.fit_transform(df["texto"])

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method="batch",
        max_iter=30,
        evaluate_every=5,
        n_jobs=-1,
    )

    document_topic_matrix = lda.fit_transform(document_term_matrix)

    topics = extraer_palabras_temas(
        lda=lda,
        vectorizer=vectorizer,
    )

    dominant_topics = document_topic_matrix.argmax(axis=1)

    topic_distribution = (
        pd.Series(dominant_topics)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    model_path = MODEL_DIR / "lda_financial.joblib"
    vectorizer_path = MODEL_DIR / "lda_vectorizer.joblib"
    report_path = REPORT_DIR / "lda_topics.json"

    joblib.dump(lda, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    report = {
        "model": "Latent Dirichlet Allocation",
        "n_topics": N_TOPICS,
        "n_documents": len(df),
        "vocabulary_size": len(vectorizer.get_feature_names_out()),
        "perplexity": float(lda.perplexity(document_term_matrix)),
        "topics": topics,
        "topic_distribution": {
            str(key): int(value)
            for key, value in topic_distribution.items()
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nModelo LDA guardado en:\n{model_path}")
    print(f"\nVectorizador guardado en:\n{vectorizer_path}")
    print(f"\nReporte guardado en:\n{report_path}")

    print("\nTemas detectados:")

    for topic in topics:
        print(
            f"\nTema {topic['topic_id']}: "
            + ", ".join(topic["keywords"])
        )

    print(f"\nPerplejidad: {report['perplexity']:.4f}")


if __name__ == "__main__":
    main()

