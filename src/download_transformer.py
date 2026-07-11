from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPO_ID = "bardsai/finance-sentiment-es-base"

DEFAULT_TARGET = (
    PROJECT_ROOT
    / "models"
    / "transformer_financial_es"
)


def validate_model(path: Path) -> None:
    required = [
        path / "config.json",
        path / "model.safetensors",
    ]

    missing = [
        file.name
        for file in required
        if not file.exists()
        or file.stat().st_size == 0
    ]

    tokenizer_candidates = [
        path / "tokenizer.json",
        path / "vocab.txt",
    ]

    if not any(
        file.exists()
        and file.stat().st_size > 0
        for file in tokenizer_candidates
    ):
        missing.append(
            "tokenizer.json o vocab.txt"
        )

    if missing:
        raise RuntimeError(
            "El Transformer quedó incompleto. "
            f"Faltan: {', '.join(missing)}"
        )

    model_size_mb = (
        (path / "model.safetensors").stat().st_size
        / 1024
        / 1024
    )

    if model_size_mb < 100:
        raise RuntimeError(
            "El archivo model.safetensors parece "
            f"incompleto: {model_size_mb:.2f} MB"
        )

    print("Transformer validado.")
    print("Directorio:", path)
    print(
        "Modelo:",
        f"{model_size_mb:.2f} MB",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Descarga y valida el Transformer "
            "financiero de TerMacroMetro."
        )
    )

    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
    )

    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
    )

    parser.add_argument(
        "--revision",
        default=os.environ.get(
            "TERMACROMETRO_MODEL_REVISION",
            "main",
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    target = args.target.resolve()

    if not args.force:
        try:
            validate_model(target)
            print(
                "El modelo ya existe; "
                "no es necesario descargarlo."
            )
            return
        except RuntimeError:
            pass

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    token = (
        os.environ.get("HF_TOKEN")
        or None
    )

    print("Repositorio:", args.repo_id)
    print("Revisión:", args.revision)
    print("Destino:", target)

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=target,
        token=token,
        resume_download=True,
        ignore_patterns=[
            "*.h5",
            "*.msgpack",
            "*.ot",
            "*.onnx",
        ],
    )

    validate_model(target)


if __name__ == "__main__":
    main()
