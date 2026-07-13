# Arquitectura de TerMacroMetro

**Versión:** 1.0.1

**Última actualización:** Julio 2026

---

# 1. Visión general

TerMacroMetro es una plataforma modular de inteligencia económica diseñada para recopilar, procesar, analizar y visualizar información pública relacionada con la economía boliviana.

La arquitectura fue diseñada siguiendo un enfoque por etapas (pipeline architecture), donde cada módulo tiene una responsabilidad específica y puede evolucionar independientemente del resto del sistema.

Esta separación facilita:

- mantenimiento;
- escalabilidad;
- incorporación de nuevas fuentes;
- incorporación de nuevos modelos;
- automatización completa del pipeline.

---

# 2. Arquitectura lógica

El flujo principal del sistema puede resumirse como:

```

Internet
│
├── Organismos oficiales
├── Medios nacionales
├── Medios regionales
├── Agencias
└── Portales económicos

↓

Collectors

↓

Corpus diario

↓

Merge histórico

↓

Preprocesamiento

↓

Extracción de entidades

↓

Modelado de temas (LDA)

↓

Clasificación de sentimiento

↓

Construcción de indicadores

↓

Dashboard

↓

API FastAPI

↓

Frontend HTML

```

Cada etapa consume la salida de la anterior.

---

# 3. Organización del proyecto

La estructura principal del repositorio es:

```text
.
├── api/
├── config/
├── data/
├── docs/
├── models/
├── reports/
├── src/
├── tests/
├── web/
└── .github/
```

Cada carpeta tiene una responsabilidad claramente definida.

---

# 4. Configuración

Toda la configuración de fuentes se centraliza en:

```text
config/
```

Actualmente contiene:

```text
bolivia_sources.json
```

Este archivo describe:

- nombre de la fuente;
- URL;
- tipo;
- departamento;
- método de extracción;
- peso metodológico;
- límite máximo de documentos;
- estado.

Agregar una nueva fuente normalmente sólo requiere modificar este archivo.

---

# 5. Recolección

Los recolectores se encuentran en:

```text
src/collectors/
```

y son invocados mediante:

```text
src.collect_configured_sources
```

Actualmente existen tres mecanismos principales:

- RSS;
- listados HTML;
- extracción personalizada.

Cada recolector devuelve una estructura homogénea que posteriormente es integrada al corpus.

---

# 6. Gestión del corpus

El proyecto mantiene distintos estados del corpus.

## Corpus diario

```text
noticias_bolivia_current.csv
```

Contiene únicamente la recolección más reciente.

---

## Corpus histórico

```text
noticias_bolivia.csv
```

Contiene todas las observaciones acumuladas.

---

## Corpus limpio

```text
noticias_bolivia_clean.csv
```

Es la versión utilizada por los modelos.

---

## Corpus analizado

```text
noticias_bolivia_analizadas.csv
```

Incluye:

- sentimiento;
- entidades;
- temas;
- tensión;
- optimismo;
- indicadores.

---

# 7. Preprocesamiento

El módulo:

```text
src/preprocessing.py
```

realiza:

- limpieza textual;
- tokenización;
- normalización;
- eliminación de ruido;
- preservación de negaciones;
- preparación para modelos clásicos y Transformers.

El proyecto utiliza diferentes estrategias de preprocesamiento según el modelo que vaya a utilizarse.

---

# 8. Extracción de entidades

La detección de entidades se implementa en:

```text
src/entities.py
```

El objetivo consiste en reconocer:

- organismos;
- indicadores;
- instituciones;
- actores económicos.

Se utilizan:

- expresiones regulares;
- diccionarios;
- reglas específicas para Bolivia;
- desambiguación contextual.

---

# 9. Modelado de temas

El entrenamiento LDA se encuentra en:

```text
src/train_lda_bolivia.py
```

Los modelos entrenados se almacenan en:

```text
models/
```

Incluyendo:

- lda_bolivia.joblib
- lda_bolivia_vectorizer.joblib

Los temas detectados son utilizados posteriormente por el dashboard.

---

# 10. Modelos de sentimiento

TerMacroMetro utiliza dos enfoques completamente independientes.

## Modelo clásico

Pipeline:

```
Texto
↓

TF-IDF

↓

Multinomial Naive Bayes

↓

Sentimiento
```

Este modelo es ligero y extremadamente rápido.

---

## Modelo Transformer

Pipeline:

```
Texto

↓

Tokenizer

↓

Transformer financiero

↓

Softmax

↓

Sentimiento
```

El Transformer comprende relaciones semánticas que el modelo clásico no puede representar.

---

# 11. Inferencia

Toda la lógica de inferencia se concentra en:

```text
src/inference.py
```

Este módulo:

- carga modelos;
- valida artefactos;
- prepara entradas;
- ejecuta ambos clasificadores;
- calcula desacuerdo;
- genera resultados estructurados.

Es el núcleo del observatorio.

---

# 12. Construcción de indicadores

Los indicadores se generan mediante:

```text
src/build_daily_intelligence.py
```

Aquí se calculan:

- MacroScore;
- optimismo;
- tensión;
- indicadores dominantes;
- instituciones predominantes;
- estadísticas generales.

---

# 13. Dashboard

El dashboard consume principalmente:

```text
reports/bolivia_dashboard.json
```

El frontend no realiza cálculos.

Únicamente visualiza los resultados previamente calculados por el pipeline.

Esto mantiene una clara separación entre:

- procesamiento;
- presentación.

---

# 14. API

La API está implementada mediante FastAPI.

Archivo principal:

```text
api/observatory.py
```

Actualmente expone endpoints como:

- `/`
- `/api/health`
- `/api/models`
- `/api/dashboard`
- `/api/history`
- `/api/analyze`

La API únicamente consulta artefactos previamente generados.

No ejecuta entrenamiento durante las peticiones.

---

# 15. Frontend

El frontend se encuentra en:

```text
web/observatory/
```

Está desarrollado utilizando:

- HTML5;
- CSS3;
- JavaScript Vanilla.

No depende de frameworks como React, Vue o Angular.

Esta decisión reduce la complejidad y facilita el despliegue.

---

# 16. Automatización

Toda la arquitectura se ejecuta diariamente mediante:

```text
.github/workflows/daily-termacrometro.yml
```

El workflow realiza automáticamente:

1. instalación de dependencias;
2. descarga y validación del Transformer;
3. recolección;
4. validación;
5. actualización del corpus;
6. entrenamiento LDA;
7. análisis de sentimiento;
8. construcción del dashboard;
9. pruebas;
10. publicación de resultados.

---

# 17. Control de calidad

El proyecto incorpora un conjunto de pruebas automatizadas ubicadas en:

```text
tests/
```

Actualmente verifican:

- API;
- inferencia;
- entidades;
- indicadores;
- fechas;
- preprocesamiento;
- consistencia del pipeline.

El control de calidad se ejecuta antes de publicar nuevos resultados.

---

# 18. Escalabilidad

La arquitectura fue diseñada para facilitar futuras extensiones.

Entre las mejoras previstas se encuentran:

- incorporación de nuevos modelos Transformer;
- análisis multilingüe;
- series temporales avanzadas;
- predicción económica;
- análisis regional;
- API pública;
- autenticación;
- panel administrativo;
- despliegue mediante Docker;
- infraestructura en DigitalOcean.

---

# 19. Principios de diseño

Durante el desarrollo se siguieron los siguientes principios:

- separación de responsabilidades;
- modularidad;
- reproducibilidad;
- automatización;
- escalabilidad;
- simplicidad;
- trazabilidad;
- bajo acoplamiento.

Cada módulo tiene una única responsabilidad claramente definida.

---

# 20. Conclusión

La arquitectura de TerMacroMetro combina técnicas modernas de NLP, Machine Learning y Deep Learning dentro de un pipeline completamente automatizado.

Su diseño modular permite evolucionar progresivamente sin necesidad de rediseñar el sistema completo, facilitando la incorporación de nuevas fuentes, nuevos modelos y nuevos indicadores conforme el observatorio continúe creciendo.

---

**TerMacroMetro v1.0.1**

**Observatorio Inteligente de la Economía Boliviana**

Desarrollado por **Jhonny Paul Pinto Phillips**