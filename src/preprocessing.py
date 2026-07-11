from __future__ import annotations

import re
import unicodedata


NEGADORES = {
    "no",
    "nunca",
    "jamás",
    "tampoco",
    "ni",
    "sin",
}

STOPWORDS_ECONOMICAS = {
    "dijo",
    "señaló",
    "indicó",
    "informó",
    "según",
    "también",
    "tras",
    "durante",
}

STOPWORDS_LDA = STOPWORDS_ECONOMICAS | {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes",
    "aquel", "aquella", "aquellas", "aquello", "aquellos",
    "aquí", "así", "aun", "aunque", "bajo", "bien", "cada",
    "casi", "como", "con", "contra", "cual", "cuando", "de",
    "del", "desde", "donde", "dos", "durante", "e", "el",
    "ella", "ellas", "ello", "ellos", "en", "entre", "era",
    "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "están", "estar", "estas", "este", "esto", "estos",
    "fue", "fueron", "ha", "han", "hasta", "hay", "la", "las",
    "le", "les", "lo", "los", "más", "me", "menos", "mi",
    "mientras", "muy", "ni", "no", "nos", "o", "otra", "otras",
    "otro", "otros", "para", "pero", "poco", "por", "porque",
    "que", "qué", "quien", "quienes", "se", "sea", "según",
    "ser", "si", "sin", "sobre", "son", "su", "sus", "también",
    "tan", "tanto", "te", "tiene", "tienen", "todo", "todos",
    "tras", "tu", "un", "una", "uno", "unos", "usted", "ya",
    "y",

    # Boilerplate corporativo y metadatos
    "web", "teléfono", "telefono", "sede", "social", "empleados",
    "facturación", "facturacion", "domicilio", "contacto",
    "correo", "página", "pagina",

    # Años poco útiles para el modelado de temas
    "2023", "2024", "2025", "2026",
}


def normalizar_unicode(texto: str) -> str:
    return unicodedata.normalize("NFKC", texto)


def limpiar_texto(
    texto: str,
    conservar_puntuacion: bool = False,
) -> str:
    if texto is None:
        return ""

    texto = normalizar_unicode(str(texto))
    texto = texto.lower().strip()

    texto = re.sub(r"https?://\S+|www\.\S+", " URL ", texto)
    texto = re.sub(r"\S+@\S+\.\S+", " EMAIL ", texto)
    texto = re.sub(r"@\w+", " USUARIO ", texto)
    texto = re.sub(r"#(\w+)", r"\1", texto)

    texto = re.sub(r"\bus\$\b", " usd ", texto)
    texto = re.sub(r"\bu\$s\b", " usd ", texto)
    texto = re.sub(r"\bbs\.?\b", " bob ", texto)

    if conservar_puntuacion:
        texto = re.sub(
            r"[^a-záéíóúüñ0-9.,;:!?%$+\-\s]",
            " ",
            texto,
        )
    else:
        texto = re.sub(
            r"[^a-záéíóúüñ0-9%$\s]",
            " ",
            texto,
        )

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def tokenizar(texto: str) -> list[str]:
    texto_limpio = limpiar_texto(texto)

    return re.findall(
        r"[a-záéíóúüñ]+|\d+(?:[.,]\d+)?%?",
        texto_limpio,
    )


def remover_stopwords(
    tokens: list[str],
    stopwords: set[str] | None = None,
) -> list[str]:
    stopwords = stopwords or STOPWORDS_ECONOMICAS

    return [
        token
        for token in tokens
        if token not in stopwords or token in NEGADORES
    ]


def preparar_texto_clasico(texto: str) -> str:
    tokens = tokenizar(texto)
    tokens = remover_stopwords(tokens)

    return " ".join(tokens)


def preparar_texto_transformer(texto: str) -> str:
    return limpiar_texto(
        texto,
        conservar_puntuacion=True,
    )


def preparar_texto_lda(texto: str) -> str:
    tokens = tokenizar(texto)
    tokens_limpios: list[str] = []

    for token in tokens:
        if token in STOPWORDS_LDA:
            continue

        if token.isdigit():
            continue

        if re.fullmatch(r"\d+(?:[.,]\d+)?%?", token):
            continue

        if len(token) < 3:
            continue

        tokens_limpios.append(token)

    return " ".join(tokens_limpios)
