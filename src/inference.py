from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.entities import analizar_contexto_boliviano


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NB_MODEL_PATH = PROJECT_ROOT / "models" / "naive_bayes_financial.joblib"
LDA_MODEL_PATH = PROJECT_ROOT / "models" / "lda_bolivia.joblib"
LDA_VECTORIZER_PATH = PROJECT_ROOT / "models" / "lda_bolivia_vectorizer.joblib"
LDA_REPORT_PATH = PROJECT_ROOT / "reports" / "lda_bolivia_topics.json"
TRANSFORMER_MODEL_PATH = PROJECT_ROOT / "models" / "transformer_financial_es"

TOPIC_NAMES = {
    0: "Sector productivo, automotor y actividad empresarial",
    1: "Tipo de cambio, divisas y abastecimiento de combustibles",
    2: "Política fiscal, gestión pública y sistema financiero",
    3: "Precios, alimentos y servicios básicos",
    4: "Producción, inversión y régimen cambiario",
    5: "Comercio exterior, desarrollo y empresas",
}

LABEL_MAP = {
    "negative": "negativo",
    "neutral": "neutral",
    "positive": "positivo",
    "NEGATIVE": "negativo",
    "NEUTRAL": "neutral",
    "POSITIVE": "positivo",
}

DISPLAY_MAP = {
    "positivo": "favorable",
    "neutral": "neutral",
    "negativo": "desfavorable",
}


def validar_archivo(path: Path, descripcion: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe {descripcion}: {path}")


class EconomicAnalyzer:
    def __init__(self, cargar_transformer: bool = True) -> None:
        validar_archivo(NB_MODEL_PATH, "el modelo Naive Bayes")
        validar_archivo(LDA_MODEL_PATH, "el modelo LDA")
        validar_archivo(LDA_VECTORIZER_PATH, "el vectorizador LDA")

        self.nb_model = joblib.load(NB_MODEL_PATH)
        self.lda_model = joblib.load(LDA_MODEL_PATH)

        # Evita que LDA cree múltiples workers durante la inferencia.
        # El modelo fue entrenado con n_jobs=-1, pero en producción
        # comparte memoria con el Transformer.
        if hasattr(self.lda_model, "n_jobs"):
            self.lda_model.n_jobs = 1

        self.lda_vectorizer = joblib.load(LDA_VECTORIZER_PATH)
        self.lda_keywords = self._cargar_keywords_lda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_tokenizer = None
        self.transformer_model = None

        if cargar_transformer:
            self._cargar_transformer()

    def _cargar_transformer(self) -> None:
        validar_archivo(
            TRANSFORMER_MODEL_PATH,
            "el modelo Transformer",
        )

        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            TRANSFORMER_MODEL_PATH,
            local_files_only=True,
        )

        self.transformer_model = (
            AutoModelForSequenceClassification.from_pretrained(
                TRANSFORMER_MODEL_PATH,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        )

        self.transformer_model.to(self.device)
        self.transformer_model.eval()

    def _cargar_keywords_lda(self) -> dict[int, list[str]]:
        if not LDA_REPORT_PATH.exists():
            return {}

        with LDA_REPORT_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            int(topic["topic_id"]): topic["keywords"]
            for topic in data.get("topics", [])
        }

    def analizar_naive_bayes(self, texto: str) -> dict[str, Any]:
        etiqueta = str(self.nb_model.predict([texto])[0])
        probabilities = self.nb_model.predict_proba([texto])[0]
        classes = self.nb_model.classes_

        probability_map = {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, probabilities, strict=True)
        }

        return {
            "modelo": "TF-IDF + Multinomial Naive Bayes",
            "etiqueta": etiqueta,
            "tono_informativo": DISPLAY_MAP.get(etiqueta, etiqueta),
            "confianza": round(float(np.max(probabilities)), 4),
            "probabilidades": probability_map,
        }

    def _resolver_etiqueta_transformer(self, class_id: int) -> str:
        if self.transformer_model is None:
            raise RuntimeError("El Transformer no está cargado.")

        raw_label = str(
            self.transformer_model.config.id2label.get(
                class_id,
                f"LABEL_{class_id}",
            )
        )

        if raw_label in LABEL_MAP:
            return LABEL_MAP[raw_label]

        lower = raw_label.lower()

        if lower in LABEL_MAP:
            return LABEL_MAP[lower]

        raise ValueError(
            f"No se pudo interpretar la etiqueta Transformer: {raw_label}"
        )

    def analizar_transformer(self, texto: str) -> dict[str, Any]:
        if self.transformer_model is None or self.transformer_tokenizer is None:
            return {
                "modelo": "Transformer financiero en español",
                "estado": "no disponible",
            }

        encoded = self.transformer_tokenizer(
            texto,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = self.transformer_model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        class_id = int(torch.argmax(probabilities).item())
        etiqueta = self._resolver_etiqueta_transformer(class_id)

        probability_map: dict[str, float] = {}

        for index, probability in enumerate(probabilities.cpu().tolist()):
            mapped_label = self._resolver_etiqueta_transformer(index)
            probability_map[mapped_label] = round(float(probability), 4)

        return {
            "modelo": "Transformer financiero en español",
            "etiqueta": etiqueta,
            "tono_informativo": DISPLAY_MAP.get(etiqueta, etiqueta),
            "confianza": round(float(probabilities[class_id].item()), 4),
            "probabilidades": probability_map,
        }

    def analizar_tema(self, texto: str) -> dict[str, Any]:
        matrix = self.lda_vectorizer.transform([texto])
        topic_distribution = self.lda_model.transform(matrix)[0]

        dominant_topic = int(topic_distribution.argmax())
        confidence = float(topic_distribution[dominant_topic])

        return {
            "topic_id": dominant_topic,
            "nombre": TOPIC_NAMES.get(
                dominant_topic,
                f"Tema {dominant_topic}",
            ),
            "confianza": round(confidence, 4),
            "keywords": self.lda_keywords.get(dominant_topic, []),
            "distribucion": {
                str(index): round(float(value), 4)
                for index, value in enumerate(topic_distribution)
            },
        }

    def analizar(self, texto: str) -> dict[str, Any]:
        texto = str(texto).strip()

        if len(texto) < 10:
            raise ValueError(
                "El texto debe contener al menos 10 caracteres."
            )

        naive_bayes = self.analizar_naive_bayes(texto)
        transformer = self.analizar_transformer(texto)

        transformer_etiqueta = transformer.get("etiqueta")
        coincidencia = (
            naive_bayes["etiqueta"] == transformer_etiqueta
            if transformer_etiqueta
            else None
        )

        modelo_recomendado = (
            transformer
            if transformer_etiqueta
            else naive_bayes
        )

        return {
            "texto": texto,
            "resultado_consolidado": {
                "etiqueta": modelo_recomendado["etiqueta"],
                "tono_informativo": modelo_recomendado[
                    "tono_informativo"
                ],
                "confianza": modelo_recomendado["confianza"],
                "modelo_seleccionado": modelo_recomendado["modelo"],
                "coincidencia_modelos": coincidencia,
            },
            "modelos": {
                "naive_bayes": naive_bayes,
                "transformer": transformer,
            },
            "tema": self.analizar_tema(texto),
            "contexto_boliviano": analizar_contexto_boliviano(texto),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analizador económico integral del Observatorio."
    )

    parser.add_argument(
        "texto",
        type=str,
        help="Texto económico que se desea analizar.",
    )

    parser.add_argument(
        "--sin-transformer",
        action="store_true",
        help="Ejecuta solamente el modelo clásico.",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Imprime el JSON en una sola línea.",
    )

    args = parser.parse_args()

    analyzer = EconomicAnalyzer(
        cargar_transformer=not args.sin_transformer
    )

    result = analyzer.analizar(args.texto)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
        )
    )


if __name__ == "__main__":
    main()



