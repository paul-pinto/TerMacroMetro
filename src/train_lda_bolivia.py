from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocessing import preparar_texto_lda


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "bolivia"
    / "noticias_bolivia_clean.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

N_TOPICS = 6
N_TOP_WORDS = 15


def cargar_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe {DATA_PATH}. "
            "Ejecuta primero: "
            "python -m src.clean_bolivia_corpus"
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "titulo",
        "texto",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "El corpus debe contener titulo y texto."
        )

    df = df.dropna(
        subset=["titulo", "texto"]
    ).copy()

    df["documento"] = (
        df["titulo"].astype(str)
        + ". "
        + df["texto"].astype(str)
    )

    df = df[
        df["documento"].str.len() >= 250
    ]

    return df.reset_index(drop=True)


def extraer_temas(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
) -> list[dict]:
    features = vectorizer.get_feature_names_out()
    topics: list[dict] = []

    for topic_id, component in enumerate(
        lda.components_
    ):
        indices = component.argsort()[
            -N_TOP_WORDS:
        ][::-1]

        keywords = [
            features[index]
            for index in indices
        ]

        topics.append(
            {
                "topic_id": topic_id,
                "keywords": keywords,
            }
        )

    return topics


def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = cargar_dataset()

    if len(df) < 30:
        raise RuntimeError(
            "El corpus boliviano limpio quedó demasiado pequeño."
        )

    vectorizer = CountVectorizer(
        preprocessor=preparar_texto_lda,
        lowercase=False,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.80,
        max_features=8_000,
    )

    matrix = vectorizer.fit_transform(
        df["documento"]
    )

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method="batch",
        max_iter=40,
        evaluate_every=5,
        n_jobs=1,
    )

    topic_matrix = lda.fit_transform(matrix)

    dominant_topics = topic_matrix.argmax(axis=1)

    df["topic_id"] = dominant_topics
    df["topic_confidence"] = (
        topic_matrix.max(axis=1)
    )

    topics = extraer_temas(
        lda=lda,
        vectorizer=vectorizer,
    )

    distribution = (
        df["topic_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    lda_path = (
        MODEL_DIR
        / "lda_bolivia.joblib"
    )

    vectorizer_path = (
        MODEL_DIR
        / "lda_bolivia_vectorizer.joblib"
    )

    report_path = (
        REPORT_DIR
        / "lda_bolivia_topics.json"
    )

    analyzed_path = (
        PROJECT_ROOT
        / "data"
        / "bolivia"
        / "noticias_bolivia_topics.csv"
    )

    joblib.dump(
        lda,
        lda_path,
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    df.to_csv(
        analyzed_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "model": "LDA Bolivia",
        "n_topics": N_TOPICS,
        "n_documents": len(df),
        "vocabulary_size": len(
            vectorizer.get_feature_names_out()
        ),
        "perplexity": float(
            lda.perplexity(matrix)
        ),
        "topics": topics,
        "topic_distribution": {
            str(key): int(value)
            for key, value in distribution.items()
        },
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nLDA Bolivia entrenado.")
    print(f"Documentos: {len(df)}")
    print(
        "Vocabulario:",
        report["vocabulary_size"],
    )

    print("\nTemas detectados:")

    for topic in topics:
        print(
            f"\nTema {topic['topic_id']}: "
            + ", ".join(topic["keywords"])
        )

    print(
        f"\nPerplejidad: "
        f"{report['perplexity']:.4f}"
    )

    print(f"\nModelo:\n{lda_path}")
    print(f"\nReporte:\n{report_path}")
    print(f"\nCorpus analizado:\n{analyzed_path}")


if __name__ == "__main__":
    main()
