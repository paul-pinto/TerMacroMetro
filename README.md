# TerMacroMetro

<p align="center">

**Observatorio Inteligente de la Economía Boliviana**

Procesamiento de Lenguaje Natural • Machine Learning • Deep Learning • Inteligencia Económica

</p>

---

## Descripción

TerMacroMetro es un observatorio inteligente que recopila, procesa y analiza automáticamente noticias económicas relacionadas con Bolivia mediante técnicas modernas de Procesamiento de Lenguaje Natural (NLP), Machine Learning y Deep Learning.

El objetivo del proyecto consiste en transformar información pública dispersa en indicadores económicos experimentales capaces de describir el comportamiento del ecosistema informativo nacional.

El sistema incorpora un pipeline completamente automatizado que diariamente:

- recopila noticias;
- actualiza el corpus histórico;
- limpia y normaliza los documentos;
- identifica entidades económicas;
- descubre temas mediante LDA;
- clasifica sentimiento utilizando dos enfoques diferentes;
- construye indicadores agregados;
- publica un dashboard actualizado.

---

# Características

- Recolección automática desde múltiples fuentes.
- Pipeline completamente automatizado mediante GitHub Actions.
- Corpus histórico incremental.
- Limpieza y deduplicación automática.
- Extracción de entidades económicas bolivianas.
- Modelado de temas mediante LDA.
- Clasificación clásica mediante TF-IDF + Multinomial Naive Bayes.
- Clasificación moderna mediante Transformer financiero.
- API REST desarrollada con FastAPI.
- Dashboard interactivo.
- Serie temporal diaria.
- Quality Gate.
- Pruebas automatizadas.

---

# Arquitectura

```
Fuentes públicas
        │
        ▼
 Collectors
        │
        ▼
 Corpus diario
        │
        ▼
 Corpus histórico
        │
        ▼
 Preprocesamiento
        │
        ▼
 Extracción de entidades
        │
        ▼
 LDA
        │
        ▼
 Sentimiento
        │
        ▼
 Indicadores
        │
        ▼
 Dashboard
        │
        ▼
 API
```

---

# Tecnologías utilizadas

## Lenguaje

- Python 3.11

## Machine Learning

- Scikit-Learn
- Pandas
- Joblib

## Deep Learning

- Transformers
- Hugging Face
- PyTorch

## NLP

- TF-IDF
- LDA
- Tokenización
- Extracción de entidades

## Backend

- FastAPI
- Uvicorn

## Automatización

- GitHub Actions

---

# Modelos utilizados

## Modelo clásico

Representación

- TF-IDF

Clasificador

- Multinomial Naive Bayes

---

## Modelo Transformer

Modelo

```
bardsai/finance-sentiment-es-base
```

Distribución

- Hugging Face

---

## Modelado de temas

Algoritmo

- Latent Dirichlet Allocation (LDA)

---

# Estructura del proyecto

```text
api/
config/
data/
docs/
models/
reports/
src/
tests/
web/
```

---

# Instalación

Clonar el repositorio

```bash
git clone https://github.com/paul-pinto/TerMacroMetro.git

cd TerMacroMetro
```

Crear entorno virtual

```bash
python -m venv .venv
```

Activar entorno

Windows

```powershell
.venv\Scripts\Activate.ps1
```

Linux

```bash
source .venv/bin/activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecutar la API

```bash
python -m uvicorn api.observatory:app \
    --host 127.0.0.1 \
    --port 8000
```

---

# Dashboard

```
http://127.0.0.1:8000
```

---

# Swagger

```
http://127.0.0.1:8000/docs
```

---

# Pipeline diario

El workflow ejecuta automáticamente:

1. Recolección
2. Limpieza
3. Actualización del corpus
4. Entrenamiento LDA
5. Clasificación de sentimiento
6. Construcción de indicadores
7. Actualización del dashboard
8. Control de calidad
9. Publicación de resultados

---

# Documentación

- [Metodología](docs/METHODOLOGY.md)
- [Arquitectura](docs/ARCHITECTURE.md)
- [Referencia de la API](docs/API.md)
- [Fuentes de datos](docs/DATA_SOURCES.md)
- [Indicadores](docs/INDICATORS.md)
- [Limitaciones](docs/LIMITATIONS.md)
- [Filosofía de diseño](docs/DESIGN_PHILOSOPHY.md)
- [Misión y visión](docs/MISSION_AND_VISION.md)
- [Hoja de ruta](docs/ROADMAP.md)
- [Licencia MIT](LICENSE)

---

# Roadmap

## v1.0

- Dashboard
- API
- LDA
- Transformer
- GitHub Actions

## v2.0

- Docker
- Despliegue en DigitalOcean
- Nginx
- Dominio propio
- Series temporales avanzadas
- Nuevos indicadores

## v3.0

- API pública
- Autenticación
- Panel administrativo
- Base de datos
- Dashboard analítico

---

# Licencia

MIT License.

---

# Autor

**Jhonny Paul Pinto Phillips**

Desarrollador principal de TerMacroMetro.

---

# Citar este proyecto

```text
Pinto Phillips, J. P. (2026).

TerMacroMetro:
Observatorio Inteligente de la Economía Boliviana.

https://github.com/paul-pinto/TerMacroMetro
```

---

<p align="center">

**TerMacroMetro**

Observatorio Inteligente de la Economía Boliviana

Versión 1.0.0

</p>
