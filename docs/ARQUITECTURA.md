# Arquitectura del proyecto

## 1. Visión general

El proyecto implementa un **pipeline clásico de PLN** para clasificación de texto
(análisis de sentimiento) y lo expone a través de una **API REST** con una
**interfaz web**. La misma cadena de transformaciones se usa en entrenamiento e
inferencia, garantizando consistencia.

```
                        ┌─────────────────────────────────────────────┐
                        │                NAVEGADOR                     │
                        │   web/index.html + estilos.css + app.js      │
                        └───────────────┬─────────────────────────────┘
                                        │  HTTP (fetch)
                                        │  POST /predecir  {"texto": ...}
                                        ▼
                        ┌─────────────────────────────────────────────┐
                        │            API REST  (FastAPI)               │
                        │              api/app.py                      │
                        │   /predecir  /predecir_lote  /metricas       │
                        └───────────────┬─────────────────────────────┘
                                        │  Predictor.predecir(texto)
                                        ▼
                        ┌─────────────────────────────────────────────┐
                        │        MODELO SERIALIZADO (.joblib)          │
                        │        modelos/modelo_sentimiento.joblib     │
                        │                                              │
                        │   Pipeline de scikit-learn:                  │
                        │   ┌───────────┐ ┌────────┐ ┌──────────────┐  │
                        │   │normalizar │→│ TF-IDF │→│  Regresión   │  │
                        │   │(preproc.) │ │(1,2)gr │ │  Logística   │  │
                        │   └───────────┘ └────────┘ └──────────────┘  │
                        └─────────────────────────────────────────────┘
```

## 2. Etapas del pipeline (flujo de datos)

### Fase de entrenamiento (`src/entrenar.py`)

```
data/resenas.csv
      │
      ▼
[1] Carga y limpieza de nulos          (pandas)
      │
      ▼
[2] Partición train/test estratificada (train_test_split, 80/20)
      │
      ▼
[3] Ajuste del Pipeline:
        normalizar → TfidfVectorizer.fit → LogisticRegression.fit
      │
      ▼
[4] Evaluación (classification_report + matriz de confusión)
      │
      ▼
[5] Serialización → modelos/modelo_sentimiento.joblib
                    modelos/metricas.json
```

### Fase de inferencia (`src/predecir.py` → `api/app.py`)

```
"texto crudo del usuario"
      │
      ▼
Pipeline.predict / predict_proba
   (aplica AUTOMÁTICAMENTE normalizar + TF-IDF + clasificador)
      │
      ▼
{ sentimiento, confianza, probabilidades{...} }
```

## 3. Decisiones de diseño y su justificación

| Decisión | Por qué |
|----------|---------|
| **`Pipeline` de scikit-learn** | Empaqueta preprocesamiento + vectorización + modelo en un solo objeto. Evita el desajuste train/inference y simplifica el despliegue. |
| **TF-IDF con unigramas + bigramas** | Los bigramas capturan expresiones como *"no recomiendo"* o *"mala calidad"*, clave para el sentimiento. |
| **Regresión Logística** | Rápida, interpretable y muy efectiva sobre vectores dispersos de texto. Buen modelo base para enseñar. |
| **No eliminar negaciones de las stopwords** | *"no"*, *"nunca"*, *"nada"* invierten el sentimiento; quitarlas degradaría el modelo. |
| **`class_weight="balanced"`** | Protege ante desbalance de clases si el dataset cambia. |
| **FastAPI sirve API + web** | Un solo proceso y un solo comando (`uvicorn`) para toda la demo. |
| **Dataset sintético reproducible** | Semilla fija → resultados repetibles en clase; fácil de regenerar y ampliar. |

## 4. Estructura de carpetas

```
proyecto-analisis-sentimiento/
├── data/                  Dataset de reseñas etiquetadas (CSV)
├── src/                   Código del pipeline de PLN
│   ├── generar_dataset.py   Genera el dataset sintético
│   ├── preprocesamiento.py  Limpieza y normalización de texto
│   ├── entrenar.py          Entrena y evalúa el pipeline
│   └── predecir.py          Carga el modelo y predice
├── modelos/               Modelo y métricas serializados (generados)
├── api/                   API REST (FastAPI)
│   └── app.py
├── web/                   Interfaz web (HTML + CSS + JS)
├── docs/                  Documentación y diagramas
├── requirements.txt
└── README.md
```

## 5. Extensiones sugeridas (para prácticas de clase)

1. **Reemplazar el clasificador** por `LinearSVC` o `MultinomialNB` y comparar.
2. **Añadir lematización** con spaCy (`es_core_news_sm`) en el preprocesamiento.
3. **Usar embeddings** (Word2Vec / FastText / transformers) en lugar de TF-IDF.
4. **Validación cruzada** con `GridSearchCV` para ajustar `C` y `ngram_range`.
5. **Dataset real**: sustituir el sintético por reseñas reales (p. ej. corpus de
   MercadoLibre / Amazon en español) y observar la caída de exactitud respecto
   al 100% sintético — excelente discusión sobre *sobreajuste* y *fuga de datos*.
```
