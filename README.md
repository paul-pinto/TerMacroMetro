# 🧠 Análisis de Sentimiento de Reseñas en Español

Proyecto de ejemplo **completo y funcional** para la asignatura de **Procesamiento
de Lenguaje Natural** (Maestría en Ciencia de Datos e IA). Implementa un pipeline
clásico de PLN que clasifica reseñas de productos en tres categorías —
**positivo**, **negativo** y **neutral** — y lo expone mediante una **API REST**
y una **interfaz web**.

```
Preprocesamiento  →  TF-IDF (1-2 gramas)  →  Regresión Logística
```

> Ver el diagrama y las decisiones de diseño en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

---

## 🚀 Puesta en marcha (4 pasos)

Desde la carpeta `proyecto-analisis-sentimiento/`:

### 1. Crear el entorno e instalar dependencias

```bash
python3.11 -m venv venv
source venv/bin/activate          # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Requiere Python 3.10–3.13 (scikit-learn aún no publica ruedas para 3.14).

### 2. Generar el dataset de ejemplo

```bash
python -m src.generar_dataset            # crea data/resenas.csv (900 reseñas)
```

### 3. Entrenar el modelo

```bash
python -m src.entrenar                    # crea modelos/modelo_sentimiento.joblib
```

### 4. Levantar la aplicación web + API

```bash
uvicorn api.app:app --reload
```

Abre 👉 **http://127.0.0.1:8000** en el navegador.
La documentación interactiva de la API está en **http://127.0.0.1:8000/docs**.

---

## 🗂️ Estructura

```
proyecto-analisis-sentimiento/
├── data/resenas.csv          Dataset de reseñas etiquetadas (generado)
├── src/
│   ├── generar_dataset.py    Genera el dataset sintético reproducible
│   ├── preprocesamiento.py   Limpieza y normalización del texto
│   ├── entrenar.py           Entrena, evalúa y serializa el pipeline
│   └── predecir.py           Carga el modelo y predice
├── modelos/                  Modelo y métricas (.joblib / .json, generados)
├── api/app.py                API REST con FastAPI + servidor de la web
├── web/                      Interfaz web (index.html, estilos.css, app.js)
├── docs/ARQUITECTURA.md      Diagrama y justificación de diseño
├── requirements.txt
└── README.md
```

---

## 🧪 Probar sin la web

**Desde la terminal (CLI):**

```bash
python -m src.predecir "El celular es excelente, lo recomiendo"
python -m src.predecir            # modo interactivo
```

**Con la API (curl):**

```bash
curl -X POST http://127.0.0.1:8000/predecir \
     -H "Content-Type: application/json" \
     -d '{"texto":"La laptop es pésima, se dañó al segundo día"}'
```

Respuesta:

```json
{
  "texto": "La laptop es pésima, se dañó al segundo día",
  "sentimiento": "negativo",
  "confianza": 0.87,
  "probabilidades": { "negativo": 0.87, "neutral": 0.08, "positivo": 0.05 }
}
```

---

## 📊 Sobre el dataset

- **900 reseñas** balanceadas (300 por clase), generadas por plantillas + vocabulario.
- Reproducible: usa una semilla fija (`--semilla 42`).
- Ampliable: `python -m src.generar_dataset --n 1800` genera más ejemplos.

> ⚠️ **Nota didáctica:** al ser un dataset sintético con plantillas, el modelo
> alcanza ~100% de exactitud en la prueba. Esto es ideal para verificar que el
> pipeline funciona, pero **no representa un problema real**. Para una práctica
> más realista, sustituye `data/resenas.csv` por reseñas reales y observa cómo
> baja la exactitud: es la mejor manera de discutir **sobreajuste** y
> **generalización** en clase.

---

## 🔌 Endpoints de la API

| Método | Ruta               | Descripción                                  |
|--------|--------------------|----------------------------------------------|
| GET    | `/`                | Interfaz web                                 |
| GET    | `/salud`           | Estado del servicio y del modelo             |
| GET    | `/metricas`        | Métricas de evaluación del modelo            |
| POST   | `/predecir`        | Predice el sentimiento de un texto           |
| POST   | `/predecir_lote`   | Predice el sentimiento de una lista de textos|
| GET    | `/docs`            | Documentación interactiva (Swagger UI)       |

---

## 🎓 Ideas para ampliar (prácticas)

1. Cambiar `LogisticRegression` por `LinearSVC` o `MultinomialNB` y comparar.
2. Añadir lematización con spaCy (`es_core_news_sm`).
3. Sustituir TF-IDF por embeddings (Word2Vec, FastText o transformers).
4. Ajustar hiperparámetros con `GridSearchCV`.
5. Entrenar con un corpus real en español y analizar los resultados.
