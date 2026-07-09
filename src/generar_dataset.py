"""
Generador de dataset sintético de reseñas de productos en español.

Produce un archivo CSV balanceado con reseñas etiquetadas en tres clases:
    - positivo
    - negativo
    - neutral

La estrategia es combinar plantillas (estructuras de oración) con listas de
productos, adjetivos y conectores. Esto genera variedad léxica suficiente para
que un modelo TF-IDF + Regresión Logística aprenda patrones reales, y a la vez
mantiene el dataset reproducible (usamos una semilla fija).

Uso:
    python -m src.generar_dataset            # genera data/resenas.csv
    python -m src.generar_dataset --n 1200   # genera 1200 reseñas
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# --- Vocabulario base -------------------------------------------------------

PRODUCTOS = [
    "el celular", "la laptop", "los audífonos", "la cámara", "el teclado",
    "el mouse", "la mochila", "las zapatillas", "la cafetera", "el reloj",
    "la aspiradora", "el monitor", "la impresora", "el parlante", "la tablet",
    "la licuadora", "el ventilador", "la silla", "el cargador", "la batería",
]

ADJ_POSITIVOS = [
    "excelente", "increíble", "buenísimo", "maravilloso", "fantástico",
    "estupendo", "genial", "cómodo", "resistente", "rápido", "elegante",
    "práctico", "duradero", "impecable", "perfecto", "recomendable",
]

ADJ_NEGATIVOS = [
    "pésimo", "horrible", "malísimo", "defectuoso", "frágil", "lento",
    "incómodo", "decepcionante", "deficiente", "inservible", "barato",
    "ruidoso", "complicado", "sobrevalorado", "terrible", "flojo",
]

ADJ_NEUTRALES = [
    "normal", "aceptable", "correcto", "estándar", "regular", "común",
    "sencillo", "básico", "promedio", "funcional",
]

# --- Plantillas por clase ---------------------------------------------------
# {p} = producto, {adj} = adjetivo de la clase correspondiente

PLANTILLAS_POSITIVAS = [
    "{p} es {adj}, lo recomiendo totalmente.",
    "Me encantó {p}, funciona {adj} y llegó rápido.",
    "Muy contento con la compra, {p} resultó {adj}.",
    "La verdad {p} superó mis expectativas, es {adj}.",
    "Excelente relación calidad-precio, {p} es {adj}.",
    "Volvería a comprarlo, {p} es {adj} y de buena calidad.",
    "{p} llegó en perfecto estado y es {adj}.",
    "Cinco estrellas, {p} es {adj} y muy fácil de usar.",
    "Estoy feliz con {p}, es {adj} y vale cada centavo.",
    "Lo uso todos los días, {p} es {adj} y no falla.",
]

PLANTILLAS_NEGATIVAS = [
    "{p} es {adj}, no lo recomiendo para nada.",
    "Muy decepcionado, {p} resultó {adj} y se dañó pronto.",
    "No compren {p}, es {adj} y una pérdida de dinero.",
    "{p} llegó {adj} y con fallas desde el primer día.",
    "Pésima experiencia, {p} es {adj} y el soporte no responde.",
    "Esperaba más, {p} es {adj} y no cumple lo prometido.",
    "Una estrella, {p} es {adj} y dejó de funcionar rápido.",
    "Me arrepiento de la compra, {p} es {adj}.",
    "{p} es {adj}, pedí la devolución de inmediato.",
    "No vale lo que cuesta, {p} es {adj} y de mala calidad.",
]

PLANTILLAS_NEUTRALES = [
    "{p} es {adj}, cumple con lo básico.",
    "{p} está bien, nada especial pero {adj}.",
    "Es un producto {adj}, ni bueno ni malo.",
    "{p} funciona, es {adj} y hace lo que promete.",
    "Compré {p} y es {adj}, sin sorpresas.",
    "{p} es {adj}, sirve para el uso diario.",
    "Producto {adj}, esperaba algo más pero está correcto.",
    "{p} es {adj}, hay opciones mejores y peores.",
    "En general {p} es {adj}, cumple su función.",
    "{p} resultó {adj}, ni me encanta ni me molesta.",
]

CONFIG_CLASES = {
    "positivo": (PLANTILLAS_POSITIVAS, ADJ_POSITIVOS),
    "negativo": (PLANTILLAS_NEGATIVAS, ADJ_NEGATIVOS),
    "neutral": (PLANTILLAS_NEUTRALES, ADJ_NEUTRALES),
}


def generar(n_por_clase: int, semilla: int = 42) -> list[tuple[str, str]]:
    """Genera una lista de (texto, etiqueta) balanceada entre las tres clases."""
    random.seed(semilla)
    filas: list[tuple[str, str]] = []
    vistos: set[str] = set()

    for etiqueta, (plantillas, adjetivos) in CONFIG_CLASES.items():
        generadas = 0
        intentos = 0
        # Intentamos generar textos únicos para evitar duplicados exactos.
        while generadas < n_por_clase and intentos < n_por_clase * 50:
            intentos += 1
            plantilla = random.choice(plantillas)
            texto = plantilla.format(
                p=random.choice(PRODUCTOS),
                adj=random.choice(adjetivos),
            )
            # Capitalizamos la primera letra de la oración.
            texto = texto[0].upper() + texto[1:]
            if texto in vistos:
                continue
            vistos.add(texto)
            filas.append((texto, etiqueta))
            generadas += 1

    random.shuffle(filas)
    return filas


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el dataset de reseñas.")
    parser.add_argument("--n", type=int, default=900,
                        help="Total de reseñas (se reparte entre 3 clases).")
    parser.add_argument("--salida", type=str, default="data/resenas.csv",
                        help="Ruta del CSV de salida.")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    n_por_clase = args.n // 3
    filas = generar(n_por_clase, args.semilla)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    with salida.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["texto", "etiqueta"])
        writer.writerows(filas)

    print(f"[OK] {len(filas)} reseñas escritas en {salida}")
    # Resumen por clase
    from collections import Counter
    conteo = Counter(etq for _, etq in filas)
    for clase, cant in sorted(conteo.items()):
        print(f"     {clase:>9}: {cant}")


if __name__ == "__main__":
    main()
