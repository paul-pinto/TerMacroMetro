from __future__ import annotations

import re
import unicodedata
from collections import defaultdict


ENTIDADES_BOLIVIA = {
    "BCB": [
        "bcb",
        "banco central de bolivia",
        "banco central",
    ],
    "ASFI": [
        "asfi",
        "autoridad de supervisión del sistema financiero",
    ],
    "INE": [
        "ine",
        "instituto nacional de estadística",
        "instituto nacional de estadistica",
    ],
    "YPFB": [
        "ypfb",
        "yacimientos petrolíferos fiscales bolivianos",
        "yacimientos petroliferos fiscales bolivianos",
    ],
    "MEFP": [
        "ministerio de economía",
        "ministerio de economia",
        "ministerio de economía y finanzas públicas",
        "ministerio de economia y finanzas publicas",
        "mefp",
    ],
    "SIN": [
        "sin",
        "servicio de impuestos nacionales",
    ],
    "Aduana Nacional": [
        "aduana nacional",
        "aduana",
    ],
    "IBCE": [
        "ibce",
        "instituto boliviano de comercio exterior",
    ],
    "FMI": [
        "fmi",
        "fondo monetario internacional",
    ],
    "BID": [
        "bid",
        "banco interamericano de desarrollo",
    ],
    "CAF": [
        "caf",
        "banco de desarrollo de américa latina",
        "banco de desarrollo de america latina",
    ],
    "Banco Mundial": [
        "banco mundial",
    ],
}

INDICADORES_ECONOMICOS = {
    "inflación": [
        "inflación",
        "inflacion",
        "ipc",
        "índice de precios al consumidor",
        "indice de precios al consumidor",
    ],
    "tipo de cambio": [
        "tipo de cambio",
        "dólar",
        "dolar",
        "divisas",
        "mercado paralelo",
        "cotización del dólar",
        "cotizacion del dolar",
    ],
    "reservas internacionales": [
        "reservas internacionales",
        "reservas netas",
        "rin",
        "reservas",
    ],
    "combustibles": [
        "combustible",
        "combustibles",
        "diésel",
        "diesel",
        "gasolina",
        "gas licuado",
        "glp",
    ],
    "deuda pública": [
        "deuda pública",
        "deuda publica",
        "endeudamiento",
        "deuda externa",
        "deuda interna",
    ],
    "déficit fiscal": [
        "déficit fiscal",
        "deficit fiscal",
        "déficit",
        "deficit",
        "gasto público",
        "gasto publico",
    ],
    "producto interno bruto": [
        "producto interno bruto",
        "pib",
        "crecimiento económico",
        "crecimiento economico",
    ],
    "exportaciones": [
        "exportaciones",
        "exportación",
        "exportacion",
        "ventas externas",
    ],
    "importaciones": [
        "importaciones",
        "importación",
        "importacion",
        "compras externas",
    ],
    "balanza comercial": [
        "balanza comercial",
        "superávit comercial",
        "superavit comercial",
        "déficit comercial",
        "deficit comercial",
    ],
    "litio": [
        "litio",
        "carbonato de litio",
        "ylb",
        "yacimientos de litio bolivianos",
    ],
    "hidrocarburos": [
        "hidrocarburos",
        "gas natural",
        "petróleo",
        "petroleo",
        "producción de gas",
        "produccion de gas",
    ],
    "empleo": [
        "empleo",
        "desempleo",
        "ocupación",
        "ocupacion",
        "mercado laboral",
    ],
    "salarios": [
        "salario",
        "salarios",
        "salario mínimo",
        "salario minimo",
        "incremento salarial",
    ],
    "sistema financiero": [
        "sistema financiero",
        "banca",
        "bancos",
        "crédito",
        "credito",
        "cartera",
        "depósitos",
        "depositos",
        "liquidez",
    ],
}

TERMINOS_RIESGO = {
    "crisis": 8,
    "escasez": 8,
    "desabastecimiento": 9,
    "déficit": 6,
    "deficit": 6,
    "caída": 5,
    "caida": 5,
    "inflación": 5,
    "inflacion": 5,
    "devaluación": 8,
    "devaluacion": 8,
    "incertidumbre": 5,
    "bloqueo": 6,
    "conflicto": 5,
    "pérdida": 5,
    "perdida": 5,
    "recesión": 8,
    "recesion": 8,
    "contracción": 7,
    "contraccion": 7,
    "quiebra": 9,
    "riesgo": 4,
    "presión": 4,
    "presion": 4,
    "deterioro": 6,
    "emergencia": 7,
    "insuficiente": 4,
    "falta": 4,
    "demora": 3,
}

TERMINOS_ALIVIO = {
    "crecimiento": 4,
    "superávit": 5,
    "superavit": 5,
    "mejora": 4,
    "recuperación": 5,
    "recuperacion": 5,
    "aumento de exportaciones": 5,
    "estabilidad": 4,
    "fortalecimiento": 4,
    "récord": 5,
    "record": 5,
    "expansión": 5,
    "expansion": 5,
}


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", str(texto)).lower()
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def _contiene_patron(texto: str, patron: str) -> bool:
    patron = normalizar(patron)

    if len(patron) <= 3 and patron.isalpha():
        return re.search(rf"\b{re.escape(patron)}\b", texto) is not None

    return patron in texto


def extraer_entidades(texto: str) -> list[str]:
    texto_original = str(texto)
    texto_normalizado = normalizar(texto_original)
    encontradas: list[str] = []

    for entidad, variantes in ENTIDADES_BOLIVIA.items():
        detectada = False

        for variante in variantes:
            variante_normalizada = normalizar(variante)

            if entidad == "SIN":
                # El acrónimo SIN solo se reconoce si aparece escrito
                # explícitamente en mayúsculas o mediante su nombre completo.
                if variante_normalizada == "sin":
                    if re.search(r"(?<!\w)SIN(?!\w)", texto_original):
                        detectada = True
                        break

                    continue

                if _contiene_patron(
                    texto_normalizado,
                    variante_normalizada,
                ):
                    detectada = True
                    break

                continue

            if _contiene_patron(
                texto_normalizado,
                variante_normalizada,
            ):
                detectada = True
                break

        if detectada:
            encontradas.append(entidad)

    return encontradas

def extraer_indicadores(texto: str) -> list[str]:
    texto_normalizado = normalizar(texto)
    encontrados = []

    for indicador, variantes in INDICADORES_ECONOMICOS.items():
        if any(_contiene_patron(texto_normalizado, variante) for variante in variantes):
            encontrados.append(indicador)

    return encontrados


def contar_terminos_ponderados(
    texto: str,
    diccionario: dict[str, int],
) -> tuple[int, dict[str, int]]:
    texto_normalizado = normalizar(texto)
    total = 0
    detalles: dict[str, int] = defaultdict(int)

    for termino, peso in diccionario.items():
        patron = normalizar(termino)

        if " " in patron:
            ocurrencias = texto_normalizado.count(patron)
        else:
            ocurrencias = len(
                re.findall(
                    rf"\b{re.escape(patron)}\b",
                    texto_normalizado,
                )
            )

        if ocurrencias:
            detalles[termino] += ocurrencias
            total += ocurrencias * peso

    return total, dict(detalles)


def calcular_stress_lexico(texto: str) -> dict:
    riesgo, riesgo_detectado = contar_terminos_ponderados(
        texto,
        TERMINOS_RIESGO,
    )

    alivio, alivio_detectado = contar_terminos_ponderados(
        texto,
        TERMINOS_ALIVIO,
    )

    puntaje_base = 50
    puntaje = puntaje_base + riesgo - alivio
    puntaje = max(0, min(100, puntaje))

    if puntaje >= 75:
        nivel = "crítico"
    elif puntaje >= 60:
        nivel = "alto"
    elif puntaje >= 40:
        nivel = "moderado"
    else:
        nivel = "bajo"

    return {
        "score": puntaje,
        "nivel": nivel,
        "riesgo_detectado": riesgo_detectado,
        "alivio_detectado": alivio_detectado,
    }


def analizar_contexto_boliviano(texto: str) -> dict:
    return {
        "entidades": extraer_entidades(texto),
        "indicadores": extraer_indicadores(texto),
        "stress_lexico": calcular_stress_lexico(texto),
    }

