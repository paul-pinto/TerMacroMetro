"""
Etapa de preprocesamiento del pipeline de PLN.

Convierte texto crudo en texto normalizado, listo para vectorizar. Cada función
es pequeña y con una sola responsabilidad para que sea fácil de explicar en clase
y de reutilizar (tanto en entrenamiento como en inferencia se usa la MISMA
función `normalizar`, garantizando consistencia).

Pasos aplicados:
    1. Pasar a minúsculas.
    2. Quitar URLs, menciones y números.
    3. Eliminar signos de puntuación.
    4. Normalizar espacios en blanco.
    5. (Opcional) Quitar tildes.
    6. (Opcional) Eliminar stopwords en español.

Nota didáctica: NO quitamos las tildes por defecto porque en español pueden
cambiar el significado, y tampoco eliminamos negaciones ("no", "nunca") de las
stopwords, ya que son fundamentales para el análisis de sentimiento.
"""
from __future__ import annotations

import re
import unicodedata

# Stopwords en español (lista reducida y curada). Se han EXCLUIDO a propósito
# las negaciones ("no", "ni", "nunca", "nada", "tampoco") porque invierten el
# sentimiento y queremos que el modelo las conserve.
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "ante", "con", "en", "para", "por", "sin", "sobre", "tras", "y", "o",
    "u", "e", "que", "se", "su", "sus", "mi", "mis", "me", "lo", "le", "les",
    "es", "son", "fue", "ser", "está", "están", "este", "esta", "estos",
    "estas", "ese", "esa", "eso", "muy", "más", "menos", "ya", "también",
    "pero", "como", "cuando", "porque", "si", "yo", "tú", "él", "ella",
}

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENCION = re.compile(r"@\w+")
_RE_NUMERO = re.compile(r"\d+")
_RE_PUNTUACION = re.compile(r"[^\w\sáéíóúñü]", flags=re.UNICODE)
_RE_ESPACIOS = re.compile(r"\s+")


def quitar_tildes(texto: str) -> str:
    """Reemplaza vocales acentuadas por su versión sin tilde (á -> a)."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar(
    texto: str,
    quitar_stopwords: bool = True,
    remover_tildes: bool = False,
) -> str:
    """Aplica toda la cadena de limpieza y devuelve el texto normalizado.

    Args:
        texto: cadena de entrada (reseña cruda).
        quitar_stopwords: si True, elimina palabras vacías (menos negaciones).
        remover_tildes: si True, quita las tildes al final del proceso.

    Returns:
        Texto limpio, en minúsculas y con espacios normalizados.
    """
    if not isinstance(texto, str):
        texto = str(texto)

    texto = texto.lower()
    texto = _RE_URL.sub(" ", texto)
    texto = _RE_MENCION.sub(" ", texto)
    texto = _RE_NUMERO.sub(" ", texto)
    texto = _RE_PUNTUACION.sub(" ", texto)
    texto = _RE_ESPACIOS.sub(" ", texto).strip()

    if quitar_stopwords:
        palabras = [p for p in texto.split() if p not in STOPWORDS_ES]
        texto = " ".join(palabras)

    if remover_tildes:
        texto = quitar_tildes(texto)

    return texto


if __name__ == "__main__":
    # Pequeña demostración ejecutable: python -m src.preprocesamiento
    ejemplos = [
        "¡El celular es EXCELENTE!! Lo compré en www.tienda.com por 500 dólares.",
        "No me gustó nada, la laptop es pésima :(",
        "El producto está bien, cumple su función.",
    ]
    for e in ejemplos:
        print(f"ORIGINAL : {e}")
        print(f"NORMALIZADO: {normalizar(e)}")
        print("-" * 60)
