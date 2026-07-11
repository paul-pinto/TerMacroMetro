from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup


GENERIC_TITLES = {
    "leyes",
    "directorio",
    "ejecutivos",
    "presentaciones del presidente",
    "encaje legal",
    "posición de cambios",
    "posicion de cambios",
    "sector externo",
    "sector monetario",
    "sector precios",
    "sitio oficial del estado plurinacional de bolivia",
    "normativa",
    "manual de procesos y procedimientos",
    "valores y principios",
    "oficinas departamentales",
    "introducción",
    "introduccion",
}

GENERIC_TITLE_PATTERNS = (
    r"^leyes(?:\s*\|.*)?$",
    r"^directorio(?:\s*\|.*)?$",
    r"^ejecutivos(?:\s*\|.*)?$",
    r"^sector\s+(externo|monetario|precios)$",
    r"^sitio oficial",
    r"^normativa(?:\s*\|.*)?$",
    r"^manual\b",
    r"^oficinas departamentales$",
)


def normalize_title_key(value: str) -> str:
    value = normalize_space(value).lower()

    value = re.sub(
        r"\s*[|–-]\s*(banco central de bolivia|bcb|ine).*$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def is_generic_title(value: str) -> bool:
    title = normalize_title_key(value)

    if title in GENERIC_TITLES:
        return True

    return any(
        re.search(pattern, title)
        for pattern in GENERIC_TITLE_PATTERNS
    )

ECONOMIC_TERMS = {
    "economía", "economia", "económico", "economico",
    "inflación", "inflacion", "ipc", "precio", "precios",
    "dólar", "dolar", "divisas", "tipo de cambio",
    "reservas", "exportación", "exportacion", "exportaciones",
    "importación", "importacion", "importaciones",
    "balanza comercial", "déficit", "deficit",
    "superávit", "superavit", "pib", "crecimiento",
    "empleo", "desempleo", "salario", "crédito", "credito",
    "banca", "financiero", "financiera", "presupuesto",
    "deuda", "inversión", "inversion", "hidrocarburos",
    "gas", "petróleo", "petroleo", "combustible",
    "combustibles", "diésel", "diesel", "gasolina",
    "litio", "producción", "produccion", "industria",
    "agricultura", "tributario", "impuestos",
    "recaudación", "recaudacion", "aduana", "asfi",
    "bcb", "ine", "ypfb", "mefp",
}


@dataclass
class CollectedItem:
    titulo: str
    texto: str
    fuente: str
    fecha: str
    url: str
    economic_score: int
    source_type: str
    department: str
    scope: str
    source_weight: float
    collected_at: str

    def as_dict(self) -> dict[str, Any]:
        raw_id = (
            f"{self.fuente}|{self.url}|{self.titulo}"
            .encode("utf-8")
        )

        return {
            "id": hashlib.sha256(raw_id).hexdigest()[:20],
            "titulo": self.titulo,
            "texto": self.texto,
            "fuente": self.fuente,
            "fecha": self.fecha,
            "url": self.url,
            "economic_score": self.economic_score,
            "source_type": self.source_type,
            "department": self.department,
            "scope": self.scope,
            "source_weight": self.source_weight,
            "collected_at": self.collected_at,
        }


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def economic_score(text: str) -> int:
    lower = text.lower()

    return sum(
        1
        for term in ECONOMIC_TERMS
        if term in lower
    )


def clean_soup(soup: BeautifulSoup) -> None:
    for selector in (
        "script", "style", "noscript", "svg",
        "nav", "footer", "header", "aside",
        "form", ".sidebar", ".related",
        ".social", ".share", ".ads",
        ".advertisement", ".cookie",
        ".breadcrumb", ".menu",
    ):
        for node in soup.select(selector):
            node.decompose()


def extract_title(soup: BeautifulSoup) -> str:
    for selector in (
        "h1",
        "meta[property='og:title']",
        "meta[name='twitter:title']",
        "title",
    ):
        node = soup.select_one(selector)

        if not node:
            continue

        value = (
            node.get("content")
            if node.name == "meta"
            else node.get_text(" ", strip=True)
        )

        value = normalize_space(value or "")

        if len(value) >= 12:
            return value

    return ""


def extract_text(soup: BeautifulSoup) -> str:
    best = ""

    for selector in (
        "article",
        ".entry-content",
        ".post-content",
        ".td-post-content",
        ".article-body",
        ".article-content",
        ".contenido-noticia",
        ".nota-contenido",
        ".content",
        "main",
    ):
        for node in soup.select(selector):
            paragraphs = [
                normalize_space(
                    p.get_text(" ", strip=True)
                )
                for p in node.select("p")
            ]

            paragraphs = [
                p
                for p in paragraphs
                if len(p) >= 30
            ]

            candidate = normalize_space(
                " ".join(paragraphs)
            )

            if len(candidate) > len(best):
                best = candidate

    return best


def extract_date(soup: BeautifulSoup) -> str:
    for selector in (
        "time",
        "meta[property='article:published_time']",
        ".fecha",
        ".date",
        ".entry-date",
        ".post-date",
    ):
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


class BaseCollector:
    def __init__(
        self,
        session: requests.Session,
        source: dict[str, Any],
        defaults: dict[str, Any],
    ) -> None:
        self.session = session
        self.source = source
        self.defaults = defaults

    @property
    def verify_ssl(self) -> bool:
        return bool(
            self.source.get(
                "verify_ssl",
                self.defaults.get(
                    "verify_ssl",
                    True,
                ),
            )
        )

    @property
    def timeout(self) -> int:
        return int(
            self.source.get(
                "request_timeout",
                self.defaults.get(
                    "request_timeout",
                    25,
                ),
            )
        )

    @property
    def minimum_text_length(self) -> int:
        return int(
            self.source.get(
                "minimum_text_length",
                self.defaults.get(
                    "minimum_text_length",
                    180,
                ),
            )
        )

    @property
    def minimum_score(self) -> int:
        return int(
            self.source.get(
                "minimum_economic_score",
                self.defaults.get(
                    "minimum_economic_score",
                    2,
                ),
            )
        )

    def fetch(self, url: str) -> requests.Response | None:
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                verify=self.verify_ssl,
                allow_redirects=True,
            )

            response.raise_for_status()

            return response

        except requests.RequestException as exc:
            print(
                f"[WARN] {self.source['name']} | "
                f"{url} | {exc}"
            )

            return None

    def parse_article(
        self,
        url: str,
    ) -> CollectedItem | None:
        response = self.fetch(url)

        if response is None:
            return None

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if "html" not in content_type:
            return None

        soup = BeautifulSoup(
            response.text,
            "lxml",
        )

        clean_soup(soup)

        title = extract_title(soup)
        text = extract_text(soup)
        date = extract_date(soup)

        if (
            not title
            or is_generic_title(title)
            or len(text) < self.minimum_text_length
        ):
            return None

        score = economic_score(
            f"{title} {text}"
        )

        if score < self.minimum_score:
            return None

        return CollectedItem(
            titulo=title,
            texto=text,
            fuente=self.source["name"],
            fecha=date,
            url=response.url,
            economic_score=score,
            source_type=self.source["source_type"],
            department=self.source["department"],
            scope=self.source["scope"],
            source_weight=float(
                self.source["source_weight"]
            ),
            collected_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )


class RSSCollector(BaseCollector):
    def collect(self) -> list[CollectedItem]:
        records: list[CollectedItem] = []
        seen: set[str] = set()

        feeds = self.source.get(
            "feeds",
            [],
        )

        for feed_url in feeds:
            response = self.fetch(feed_url)

            if response is None:
                continue

            feed = feedparser.parse(
                response.content
            )

            for entry in feed.entries:
                link = normalize_space(
                    entry.get("link", "")
                )

                if not link or link in seen:
                    continue

                seen.add(link)

                record = self.parse_article(
                    link
                )

                if record:
                    records.append(record)

                if len(records) >= int(
                    self.source["max_documents"]
                ):
                    return records

        return records


class ListingCollector(BaseCollector):
    def collect(self) -> list[CollectedItem]:
        records: list[CollectedItem] = []
        links: list[str] = []
        seen: set[str] = set()

        listing_urls = self.source.get(
            "listing_urls",
            [],
        )

        patterns = tuple(
            str(pattern).lower()
            for pattern in self.source.get(
                "link_patterns",
                [],
            )
        )

        homepage = str(
            self.source.get("homepage", "")
        )

        expected_host = (
            urlparse(homepage).hostname or ""
        ).lower()

        for listing_url in listing_urls:
            response = self.fetch(listing_url)

            if response is None:
                continue

            soup = BeautifulSoup(
                response.text,
                "lxml",
            )

            for anchor in soup.select("a[href]"):
                href = normalize_space(
                    anchor.get("href", "")
                )

                if not href:
                    continue

                absolute = urljoin(
                    response.url,
                    href,
                )

                parsed = urlparse(absolute)
                hostname = (
                    parsed.hostname or ""
                ).lower()

                if (
                    expected_host
                    and hostname != expected_host
                    and not hostname.endswith(
                        "." + expected_host
                    )
                ):
                    continue

                lower_url = absolute.lower()

                if patterns and not any(
                    pattern in lower_url
                    for pattern in patterns
                ):
                    continue

                clean_url = absolute.split(
                    "#",
                    1,
                )[0].rstrip("/")

                if clean_url in seen:
                    continue

                seen.add(clean_url)
                links.append(clean_url)

        for link in links:
            record = self.parse_article(link)

            if record:
                records.append(record)

                print(
                    f"[OK] {self.source['name']}: "
                    f"{len(records):03d} | "
                    f"{record.titulo[:85]}"
                )

            if len(records) >= int(
                self.source["max_documents"]
            ):
                break

        return records


