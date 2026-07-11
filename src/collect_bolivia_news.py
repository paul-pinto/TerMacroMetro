from __future__ import annotations

import csv
import hashlib
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
import requests
import urllib3
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "bolivia"
OUTPUT_PATH = OUTPUT_DIR / "noticias_bolivia.csv"

TIMEOUT = 25
DELAY = 0.20
MAX_ITEMS_PER_SOURCE = 120
MIN_TEXT_LENGTH = 120

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/126 Safari/537.36 "
        "TerMacroMetro-Research/1.0"
    ),
    "Accept-Language": "es-BO,es;q=0.9",
}

ECONOMIC_TERMS = {
    "economía", "economia", "económico", "economico",
    "inflación", "inflacion", "ipc", "precio", "precios",
    "dólar", "dolar", "divisas", "tipo de cambio",
    "reservas", "bcb", "banco central",
    "exportación", "exportacion", "exportaciones",
    "importación", "importacion", "importaciones",
    "balanza comercial", "déficit", "deficit",
    "superávit", "superavit", "pib",
    "producto interno bruto", "crecimiento",
    "empleo", "desempleo", "salario", "crédito", "credito",
    "banca", "financiero", "financiera",
    "presupuesto", "deuda", "inversión", "inversion",
    "hidrocarburos", "gas", "petróleo", "petroleo",
    "combustible", "combustibles", "diésel", "diesel",
    "gasolina", "litio", "producción", "produccion",
    "industria", "agricultura", "tributario",
    "impuestos", "recaudación", "recaudacion",
    "aduana", "asfi", "ine", "ypfb", "mefp",
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


@dataclass
class Record:
    titulo: str
    texto: str
    fuente: str
    fecha: str
    url: str
    economic_score: int


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: str) -> str:
    value = normalize_space(value)
    value = re.sub(
        r"\s*[-|–]\s*(INE|ABI|EL DEBER|BCB|Opinión.*|Economy.*)$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


def score_economic(text: str) -> int:
    lower = text.lower()
    return sum(1 for term in ECONOMIC_TERMS if term in lower)


def make_id(record: Record) -> str:
    raw = (
        f"{record.fuente}|{record.url}|{record.titulo}"
        .encode("utf-8")
    )
    return hashlib.sha256(raw).hexdigest()[:20]


def session_get(
    session: requests.Session,
    url: str,
    verify: bool = True,
) -> requests.Response | None:
    try:
        response = session.get(
            url,
            timeout=TIMEOUT,
            verify=verify,
            allow_redirects=True,
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        print(f"[WARN] {url}: {exc}")
        return None


def clean_soup(soup: BeautifulSoup) -> None:
    selectors = (
        "script", "style", "noscript", "svg", "nav",
        "footer", "header", "aside", "form",
        ".sidebar", ".related", ".social", ".share",
        ".ads", ".advertisement", ".cookie",
        ".breadcrumb", ".menu",
    )

    for selector in selectors:
        for node in soup.select(selector):
            node.decompose()


def extract_title(soup: BeautifulSoup) -> str:
    selectors = (
        "h1",
        "meta[property='og:title']",
        "meta[name='twitter:title']",
        "title",
    )

    for selector in selectors:
        node = soup.select_one(selector)

        if not node:
            continue

        value = (
            node.get("content")
            if node.name == "meta"
            else node.get_text(" ", strip=True)
        )

        value = normalize_title(value or "")

        generic_titles = {
            "sitio oficial del estado plurinacional de bolivia",
            "banco central de bolivia",
            "agencia boliviana de información",
            "el deber",
        }

        if (
            len(value) >= 12
            and value.lower() not in generic_titles
        ):
            return value

    return ""


def extract_text(soup: BeautifulSoup) -> str:
    selectors = (
        "article",
        ".entry-content",
        ".post-content",
        ".td-post-content",
        ".field--name-body",
        ".node__content",
        ".article-body",
        ".article-content",
        ".contenido-noticia",
        ".nota-contenido",
        ".content",
        "main",
    )

    best = ""

    for selector in selectors:
        for node in soup.select(selector):
            paragraphs = [
                normalize_space(p.get_text(" ", strip=True))
                for p in node.select("p")
            ]

            paragraphs = [
                p for p in paragraphs
                if len(p) >= 30
            ]

            candidate = normalize_space(" ".join(paragraphs))

            if len(candidate) > len(best):
                best = candidate

    if len(best) < MIN_TEXT_LENGTH:
        paragraphs = [
            normalize_space(p.get_text(" ", strip=True))
            for p in soup.select("p")
        ]

        paragraphs = [
            p for p in paragraphs
            if len(p) >= 30
        ]

        best = normalize_space(" ".join(paragraphs))

    return best


def extract_date(soup: BeautifulSoup) -> str:
    selectors = (
        "time",
        "meta[property='article:published_time']",
        ".fecha",
        ".date",
        ".entry-date",
        ".post-date",
    )

    for selector in selectors:
        node = soup.select_one(selector)

        if not node:
            continue

        value = (
            node.get("datetime")
            or node.get("content")
            or node.get_text(" ", strip=True)
        )

        value = normalize_space(value or "")

        if value:
            return value[:100]

    return ""


def parse_article(
    session: requests.Session,
    url: str,
    source: str,
    verify: bool = True,
) -> Record | None:
    response = session_get(
        session=session,
        url=url,
        verify=verify,
    )

    if response is None:
        return None

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if "html" not in content_type:
        return None

    soup = BeautifulSoup(response.text, "lxml")
    clean_soup(soup)

    title = extract_title(soup)
    text = extract_text(soup)
    date = extract_date(soup)

    if not title or len(text) < MIN_TEXT_LENGTH:
        return None

    score = score_economic(f"{title} {text}")

    if score < 2:
        return None

    return Record(
        titulo=title,
        texto=text,
        fuente=source,
        fecha=date,
        url=response.url,
        economic_score=score,
    )


def collect_rss(
    session: requests.Session,
    feed_urls: list[str],
    source: str,
    verify: bool = True,
) -> list[Record]:
    records: list[Record] = []
    links_seen: set[str] = set()

    for feed_url in feed_urls:
        response = session_get(
            session=session,
            url=feed_url,
            verify=verify,
        )

        if response is None:
            continue

        feed = feedparser.parse(response.content)

        for entry in feed.entries[:MAX_ITEMS_PER_SOURCE]:
            link = normalize_space(
                entry.get("link", "")
            )

            if not link or link in links_seen:
                continue

            links_seen.add(link)

            record = parse_article(
                session=session,
                url=link,
                source=source,
                verify=verify,
            )

            if record:
                records.append(record)
                print(
                    f"[OK] {source}: {len(records):03d} | "
                    f"{record.titulo[:90]}"
                )

            time.sleep(DELAY)

    return records


def collect_listing(
    session: requests.Session,
    listing_urls: list[str],
    source: str,
    domain: str,
    link_patterns: tuple[str, ...],
    verify: bool = True,
) -> list[Record]:
    links: list[str] = []
    seen: set[str] = set()

    for listing_url in listing_urls:
        response = session_get(
            session=session,
            url=listing_url,
            verify=verify,
        )

        if response is None:
            continue

        soup = BeautifulSoup(response.text, "lxml")

        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "")
            absolute = urljoin(response.url, href)
            parsed = urlparse(absolute)

            if domain not in parsed.netloc:
                continue

            lower = absolute.lower()

            if not any(pattern in lower for pattern in link_patterns):
                continue

            if absolute in seen:
                continue

            seen.add(absolute)
            links.append(absolute)

    records: list[Record] = []

    for link in links[:MAX_ITEMS_PER_SOURCE]:
        record = parse_article(
            session=session,
            url=link,
            source=source,
            verify=verify,
        )

        if record:
            records.append(record)
            print(
                f"[OK] {source}: {len(records):03d} | "
                f"{record.titulo[:90]}"
            )

        time.sleep(DELAY)

    return records


def deduplicate(records: list[Record]) -> list[Record]:
    unique: list[Record] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for record in records:
        title_key = re.sub(
            r"[^a-záéíóúüñ0-9]+",
            " ",
            record.titulo.lower(),
        ).strip()

        if record.url in seen_urls:
            continue

        if title_key in seen_titles:
            continue

        seen_urls.add(record.url)
        seen_titles.add(title_key)
        unique.append(record)

    return unique


def save(records: list[Record]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "titulo",
                "texto",
                "fuente",
                "fecha",
                "url",
                "economic_score",
            ],
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "id": make_id(record),
                    "titulo": record.titulo,
                    "texto": record.texto,
                    "fuente": record.fuente,
                    "fecha": record.fecha,
                    "url": record.url,
                    "economic_score": record.economic_score,
                }
            )


def main() -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    records: list[Record] = []

    print("\n=== ABI RSS ===")
    records.extend(
        collect_rss(
            session=session,
            feed_urls=[
                "https://abi.bo/feed/",
                "https://abi.bo/category/economia/feed/",
                "https://abi.bo/rss-abi/",
            ],
            source="ABI",
            verify=False,
        )
    )

    print("\n=== BCB ===")
    records.extend(
        collect_listing(
            session=session,
            listing_urls=[
                "https://www.bcb.gob.bo/?q=notas-prensa",
                "https://www.bcb.gob.bo/?q=sala-prensa/notas-comunicados-prensa",
                "https://www.bcb.gob.bo/?q=comunicados",
                "https://www.bcb.gob.bo/?q=noticias-anteriores",
            ],
            source="BCB",
            domain="bcb.gob.bo",
            link_patterns=(
                "?q=content/",
                "/?q=content/",
                ".pdf",
            ),
            verify=True,
        )
    )

    print("\n=== EL DEBER ===")
    records.extend(
        collect_listing(
            session=session,
            listing_urls=[
                "https://eldeber.com.bo/economia",
                "https://eldeber.com.bo/economia/2",
                "https://eldeber.com.bo/economia/3",
                "https://eldeber.com.bo/economia/4",
                "https://eldeber.com.bo/tag/economia-boliviana",
                "https://eldeber.com.bo/tag/crisis-economica-en-bolivia",
            ],
            source="El Deber",
            domain="eldeber.com.bo",
            link_patterns=(
                "/economia/",
                "/dinero/",
            ),
            verify=True,
        )
    )

    print("\n=== INE ===")
    records.extend(
        collect_listing(
            session=session,
            listing_urls=[
                "https://www.ine.gob.bo/index.php/comunicacion/notas-de-prensa/",
                "https://www.ine.gob.bo/",
            ],
            source="INE",
            domain="ine.gob.bo",
            link_patterns=(
                "/index.php/",
            ),
            verify=True,
        )
    )

    print("\n=== OPINIÓN RSS ===")
    records.extend(
        collect_rss(
            session=session,
            feed_urls=[
                "https://www.opinion.com.bo/rss/section/7/",
                "https://www.opinion.com.bo/rss/listado/",
            ],
            source="Opinión",
            verify=True,
        )
    )

    print("\n=== ECONOMY RSS ===")
    records.extend(
        collect_rss(
            session=session,
            feed_urls=[
                "https://www.economy.com.bo/rss/listado/",
                "https://www.economy.com.bo/rss/",
            ],
            source="Economy",
            verify=True,
        )
    )

    print("\n=== MEFP ===")
    records.extend(
        collect_listing(
            session=session,
            listing_urls=[
                "https://www.economiayfinanzas.gob.bo/notas-de-prensa",
                "https://www.economiayfinanzas.gob.bo/",
            ],
            source="MEFP",
            domain="economiayfinanzas.gob.bo",
            link_patterns=(
                "nota",
                "prensa",
                "econom",
            ),
            verify=False,
        )
    )

    records = deduplicate(records)

    records.sort(
        key=lambda r: (
            r.fuente,
            -r.economic_score,
            r.titulo,
        )
    )

    save(records)

    print("\n====================================")
    print(f"Corpus guardado en: {OUTPUT_PATH}")
    print(f"Noticias únicas: {len(records)}")

    counts: dict[str, int] = {}

    for record in records:
        counts[record.fuente] = (
            counts.get(record.fuente, 0) + 1
        )

    print("Distribución por fuente:")

    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")

    if len(records) < 50:
        print(
            "\n[WARN] El corpus sigue siendo pequeño, "
            "pero ya puede usarse para un LDA preliminar."
        )


if __name__ == "__main__":
    main()

