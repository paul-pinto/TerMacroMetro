# TerMacroMetro

> **Observatorio Inteligente de la Economía Boliviana**

TerMacroMetro es una plataforma de inteligencia económica basada en técnicas modernas de Procesamiento de Lenguaje Natural (NLP), Machine Learning e Inteligencia Artificial que recopila diariamente información económica de Bolivia para generar indicadores, modelado de temas, análisis de sentimiento y métricas propias sobre la situación económica nacional.

---

# Características

- Recolección automática de noticias económicas
- Integración de fuentes oficiales y medios nacionales
- Clasificación automática de sentimiento
- Modelado de temas mediante LDA
- Detección de entidades económicas bolivianas
- Índice de tensión económica
- MacroScore (índice agregado)
- Dashboard interactivo
- API REST con FastAPI
- Pipeline diario automatizado mediante GitHub Actions
- Suite de pruebas automatizadas

---

# Arquitectura

```
                ┌──────────────────────┐
                │  Portales Económicos │
                └──────────┬───────────┘
                           │
                    Recolección diaria
                           │
                           ▼
                ┌──────────────────────┐
                │    Corpus Histórico  │
                └──────────┬───────────┘
                           │
                   Limpieza y normalización
                           │
                           ▼
                ┌──────────────────────┐
                │      NLP Engine      │
                │──────────────────────│
                │ Naive Bayes          │
                │ Transformer          │
                │ LDA                  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Indicadores propios  │
                │──────────────────────│
                │ MacroScore           │
                │ Optimismo            │
                │ Tensión Económica    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Dashboard + API      │
                └──────────────────────┘
```

---

# Stack tecnológico

- Python 3.11
- FastAPI
- Scikit-Learn
- Transformers
- PyTorch
- Pandas
- NumPy
- BeautifulSoup
- Feedparser
- GitHub Actions

---

# Modelos

## Clasificador clásico

TF-IDF + Multinomial Naive Bayes

Utilizado para clasificación rápida de sentimiento.

---

## Transformer

Modelo financiero en español basado en BERT.

---

## Modelado de temas

Latent Dirichlet Allocation (LDA)

Permite detectar automáticamente los principales temas económicos presentes en el corpus.

---

# Indicadores

## MacroScore

Indicador agregado construido a partir de:

- sentimiento
- tensión
- optimismo
- distribución temática

---

## Optimismo Económico

Mide la proporción de noticias favorables respecto al total.

---

## Índice de Tensión Económica

Construido utilizando un léxico especializado para detectar presión económica.

---

# API

| Método | Endpoint |
|---------|----------|
| GET | /api/health |
| GET | /api/models |
| GET | /api/dashboard |
| GET | /api/history |
| GET | /api/topics |
| POST | /api/analyze |
| POST | /api/analyze/batch |

---

# Instalación

```bash
git clone https://github.com/paul-pinto/termacrometro.git

cd termacrometro

python -m venv .venv

source .venv/bin/activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

Ejecutar

```bash
uvicorn api.observatory:app --reload
```

---

# Pipeline diario

Todos los días GitHub Actions ejecuta automáticamente:

1. Recolección de noticias
2. Limpieza
3. Actualización del corpus
4. Entrenamiento LDA
5. Inteligencia diaria
6. Dashboard
7. Validación
8. Tests

---

# Estado del proyecto

| Componente | Estado |
|------------|--------|
| API | ✅ |
| Dashboard | ✅ |
| Naive Bayes | ✅ |
| Transformer | ✅ |
| LDA | ✅ |
| Automatización | ✅ |
| Tests | ✅ |

---

# Roadmap

## v1.1

- Más fuentes nacionales
- Series temporales avanzadas
- Radar económico

## v1.2

- Alertas inteligentes
- Comparación entre medios
- Tendencias por institución

## v2.0

- Modelo entrenado exclusivamente con noticias bolivianas
- Sistema de consulta histórica
- Reportes automáticos

---

# Licencia

MIT License

---

**TerMacroMetro**

Observatorio Inteligente de la Economía Boliviana
