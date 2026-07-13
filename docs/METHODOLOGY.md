# Metodología de TerMacroMetro

**Versión:** 1.0.1

**Última actualización:** Julio 2026

---

# 1. Introducción

TerMacroMetro es un observatorio inteligente de la economía boliviana basado en técnicas modernas de Procesamiento de Lenguaje Natural (Natural Language Processing, NLP), Machine Learning y Deep Learning.

El objetivo del proyecto consiste en transformar información económica pública dispersa en indicadores cuantificables que permitan describir el panorama económico nacional mediante el análisis automatizado de noticias financieras.

TerMacroMetro no pretende reemplazar indicadores oficiales como el PIB, la inflación o las estadísticas del Banco Central de Bolivia. Su propósito es construir indicadores experimentales basados en el lenguaje utilizado diariamente por medios de comunicación, organismos públicos y entidades financieras.

---

# 2. Objetivo metodológico

El sistema busca responder una pregunta sencilla:

> **¿Qué está diciendo el ecosistema informativo sobre la economía boliviana hoy?**

Para responder esta pregunta se desarrolla un pipeline completamente automatizado capaz de:

- recolectar información;
- limpiarla;
- eliminar duplicados;
- detectar entidades;
- identificar temas;
- estimar sentimiento;
- calcular indicadores agregados;
- publicar un dashboard actualizado diariamente.

---

# 3. Arquitectura general

El flujo metodológico puede resumirse como:

```

Fuentes públicas
↓
Recolección automática
↓
Validación
↓
Corpus histórico
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

```

Cada una de estas etapas se describe en las siguientes secciones.

---

# 4. Recolección de información

TerMacroMetro obtiene información exclusivamente de fuentes públicas.

Actualmente las fuentes se clasifican en:

- organismos oficiales;
- entidades reguladoras;
- agencias de noticias;
- prensa nacional;
- prensa regional;
- televisión;
- portales económicos.

Las fuentes se administran mediante el archivo:

```

config/bolivia_sources.json

```

Cada fuente define:

- nombre;
- URL;
- método de extracción;
- departamento;
- alcance;
- tipo de fuente;
- peso metodológico;
- estado.

La arquitectura permite incorporar nuevas fuentes sin modificar el código principal.

---

# 5. Corpus histórico

Cada ejecución diaria produce un conjunto de noticias nuevas.

Antes de almacenarlas se realiza un proceso de deduplicación considerando:

- URL;
- título;
- contenido;
- longitud;
- calidad.

El resultado se integra al corpus histórico:

```

data/bolivia/noticias_bolivia.csv

```

Esto permite construir una serie temporal creciente.

---

# 6. Preprocesamiento

Antes de cualquier modelo de Machine Learning se aplica un proceso de limpieza.

Entre las principales operaciones se encuentran:

- normalización Unicode;
- eliminación de espacios duplicados;
- eliminación de URLs;
- eliminación de correos electrónicos;
- normalización monetaria;
- tokenización;
- eliminación de stopwords;
- preservación de negaciones;
- limpieza HTML.

Se mantienen expresiones como:

- no cayó
- no aumentó
- sin déficit

porque modifican completamente el significado económico.

---

# 7. Extracción de entidades

TerMacroMetro incorpora un extractor específico para contexto boliviano.

Actualmente identifica organismos como:

- Banco Central de Bolivia
- Ministerio de Economía
- ASFI
- INE
- YPFB
- SIN
- CAF
- BID
- FMI
- Banco Mundial

También detecta indicadores económicos como:

- inflación;
- reservas internacionales;
- tipo de cambio;
- combustibles;
- exportaciones;
- empleo;
- salarios;
- deuda pública;
- déficit fiscal.

Estas entidades alimentan posteriormente los indicadores del dashboard.

---

# 8. Modelado de temas

Para descubrir automáticamente los principales temas del corpus se utiliza:

**Latent Dirichlet Allocation (LDA)**

LDA asume que:

- un documento puede contener múltiples temas;
- un tema corresponde a una distribución probabilística de palabras.

El modelo permite identificar automáticamente agrupaciones como:

- combustibles;
- sistema financiero;
- comercio exterior;
- política fiscal;
- hidrocarburos;
- producción.

Los temas son completamente no supervisados.

---

# 9. Representación clásica

El primer enfoque de clasificación utiliza:

TF-IDF

(Term Frequency – Inverse Document Frequency)

TF-IDF transforma cada documento en un vector numérico ponderando las palabras más representativas.

Posteriormente se utiliza:

Multinomial Naive Bayes

para clasificar cada noticia en:

- positiva;
- neutral;
- negativa.

Este modelo constituye la línea base del sistema.

---

# 10. Modelo Transformer

Como segundo enfoque se utiliza un modelo moderno basado en Transformers.

Modelo utilizado:

```

bardsai/finance-sentiment-es-base

```

Distribuido mediante Hugging Face.

El modelo fue entrenado específicamente para sentimiento financiero en español.

A diferencia de TF-IDF, el Transformer comprende:

- contexto;
- orden de palabras;
- negaciones;
- relaciones semánticas.

Esto mejora considerablemente la calidad del análisis.

---

# 11. Evaluación de modelos

Ambos modelos se evaluaron sobre exactamente el mismo conjunto de prueba.

## TF-IDF + Naive Bayes

Accuracy

≈ 66 %

F1 macro

≈ 0.58

---

## Transformer financiero

Accuracy

≈ 66 %

F1 macro

≈ 0.64

Aunque ambos modelos obtienen accuracies similares, el Transformer presenta una capacidad significativamente mayor para equilibrar las tres clases.

Por esta razón ambos modelos permanecen dentro del observatorio.

---

# 12. Indicadores experimentales

A partir de las noticias procesadas se generan distintos indicadores.

## Optimismo económico

Mide la proporción de lenguaje favorable presente en el corpus.

Valores altos indican predominio de expresiones asociadas a:

- crecimiento;
- inversión;
- recuperación;
- producción;
- estabilidad.

---

## Tensión textual

Mide la intensidad del lenguaje económico.

Incrementa cuando aparecen términos relacionados con:

- crisis;
- déficit;
- escasez;
- presión;
- incertidumbre;
- caída;
- bloqueo.

No representa un indicador económico oficial.

---

## MacroScore

Es el indicador principal del observatorio.

Combina:

- tensión promedio;
- noticias desfavorables;
- tensión alta;
- desacuerdo entre modelos.

Su interpretación es:

| Rango | Nivel |
|--------|--------|
| 0–39 | Bajo |
| 40–59 | Moderado |
| 60–74 | Alto |
| 75–100 | Crítico |

---

# 13. Histórico diario

Cada ejecución diaria genera una observación compuesta por:

- fecha;
- noticias analizadas;
- MacroScore;
- optimismo;
- tensión;
- indicador dominante.

Estas observaciones permiten construir series temporales y detectar tendencias conforme aumenta el corpus.

---

# 14. Automatización

Toda la metodología se ejecuta automáticamente mediante GitHub Actions.

Cada día el workflow realiza:

1. instalación de dependencias;
2. descarga del Transformer;
3. recolección;
4. validación;
5. limpieza;
6. entrenamiento LDA;
7. análisis de sentimiento;
8. cálculo de indicadores;
9. actualización histórica;
10. pruebas automatizadas;
11. publicación de resultados.

---

# 15. Control de calidad

El pipeline incorpora un conjunto de pruebas automáticas que validan:

- API;
- inferencia;
- entidades;
- indicadores;
- fechas;
- preprocesamiento;
- calidad del corpus.

Actualmente la versión 1.0 supera la totalidad de las pruebas automatizadas.

---

# 16. Limitaciones

TerMacroMetro presenta las siguientes limitaciones:

- depende de la disponibilidad de las fuentes;
- algunos medios modifican periódicamente su estructura;
- el corpus continúa creciendo;
- los indicadores son experimentales;
- el Transformer no fue entrenado exclusivamente sobre noticias bolivianas;
- el lenguaje periodístico puede contener ambigüedad.

---

# 17. Interpretación responsable

Los indicadores generados por TerMacroMetro deben interpretarse como medidas experimentales derivadas del lenguaje presente en el corpus.

No constituyen:

- estadísticas oficiales;
- predicciones económicas;
- recomendaciones financieras;
- asesoramiento de inversión.

---

# 18. Reproducibilidad

Toda la metodología puede ejecutarse localmente mediante:

```

python -m src.collect_configured_sources
python -m src.merge_daily_corpus
python -m src.clean_bolivia_corpus
python -m src.train_lda_bolivia
python -m src.analyze_bolivia_corpus
python -m src.build_daily_intelligence
python -m src.build_dashboard_timeseries
python -m src.pipeline_quality_gate

```

o automáticamente mediante GitHub Actions.

---

# 19. Conclusión

TerMacroMetro integra técnicas clásicas de Machine Learning y modelos modernos basados en Transformers para construir un observatorio económico reproducible, automatizado y extensible.

La arquitectura fue diseñada para evolucionar progresivamente incorporando nuevas fuentes, modelos, indicadores y visualizaciones conforme aumenta el corpus histórico.

---

**TerMacroMetro v1.0.1**

**Observatorio Inteligente de la Economía Boliviana**

Desarrollado por **Jhonny Paul Pinto Phillips**
