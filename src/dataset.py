from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

LABEL_MAP = {
    "negative": "negativo",
    "neutral": "neutral",
    "positive": "positivo",
    "NEGATIVE": "negativo",
    "NEUTRAL": "neutral",
    "POSITIVE": "positivo",
    "neg": "negativo",
    "neu": "neutral",
    "pos": "positivo",
}


def normalizar_etiqueta(value: object) -> str:
    label = str(value).strip()

    if label in LABEL_MAP:
        return LABEL_MAP[label]

    label_lower = label.lower()

    if label_lower in LABEL_MAP:
        return LABEL_MAP[label_lower]

    raise ValueError(f"Etiqueta desconocida: {value!r}")


def detectar_columna(
    columnas: list[str],
    candidatas: tuple[str, ...],
) -> str:
    columnas_lower = {col.lower(): col for col in columnas}

    for candidata in candidatas:
        if candidata.lower() in columnas_lower:
            return columnas_lower[candidata.lower()]

    raise KeyError(
        f"No se encontró ninguna de las columnas esperadas: {candidatas}. "
        f"Columnas disponibles: {columnas}"
    )


def cargar_dataset_huggingface() -> pd.DataFrame:
    """
    Descarga el dataset multilingüe de sentimiento financiero
    y conserva únicamente los textos en español.
    """
    dataset = load_dataset(
        "Kenpache/multilingual-financial-sentiment",
        split="train",
    )

    df = dataset.to_pandas()

    print("Columnas descargadas:", list(df.columns))
    print("Filas originales:", len(df))

    language_col = detectar_columna(
        list(df.columns),
        ("language", "lang", "locale"),
    )

    text_col = detectar_columna(
        list(df.columns),
        ("text", "sentence", "headline", "news", "content"),
    )

    label_col = detectar_columna(
        list(df.columns),
        ("label", "sentiment", "target"),
    )

    df = df[df[language_col].astype(str).str.lower().isin({"es", "spanish"})]

    df = df[[text_col, label_col]].rename(
        columns={
            text_col: "texto",
            label_col: "sentimiento",
        }
    )

    df["texto"] = (
        df["texto"]
        .astype(str)
        .str.strip()
    )

    df["sentimiento"] = df["sentimiento"].apply(normalizar_etiqueta)

    df = df[df["texto"].str.len() >= 10]
    df = df.drop_duplicates(subset=["texto"])
    df = df.dropna(subset=["texto", "sentimiento"])
    df = df.reset_index(drop=True)

    return df


def dividir_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    random_state: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temporal_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["sentimiento"],
    )

    validation_df, test_df = train_test_split(
        temporal_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temporal_df["sentimiento"],
    )

    train_path = output_dir / "train.csv"
    validation_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"
    full_path = output_dir / "financial_sentiment_es.csv"

    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print("\nDataset guardado:")
    print(f"  Completo:   {full_path} ({len(df)} filas)")
    print(f"  Train:      {train_path} ({len(train_df)} filas)")
    print(f"  Validation: {validation_path} ({len(validation_df)} filas)")
    print(f"  Test:       {test_path} ({len(test_df)} filas)")

    print("\nDistribución total:")
    print(df["sentimiento"].value_counts())
    print("\nDistribución porcentual:")
    print(
        df["sentimiento"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga y prepara el dataset financiero en español."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida para los CSV procesados.",
    )

    args = parser.parse_args()

    df = cargar_dataset_huggingface()

    if df.empty:
        raise RuntimeError(
            "No se encontraron textos en español en el dataset."
        )

    dividir_dataset(
        df=df,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()