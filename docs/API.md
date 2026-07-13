# API de TerMacroMetro

**Versión:** 1.0.0

**Última actualización:** Julio 2026

---

# 1. Introducción

TerMacroMetro expone una API REST desarrollada con FastAPI para consultar los principales indicadores generados por el observatorio.

La API está diseñada bajo principios RESTful y entrega respuestas en formato JSON.

Todos los cálculos se realizan previamente durante el pipeline diario. La API únicamente consulta los artefactos ya generados, lo que permite respuestas rápidas y una baja carga computacional.

---

# 2. Base URL

## Desarrollo local

```
http://127.0.0.1:8000
```

## Producción

```
https://paulpinto.ia.bo/termacrometro/
```

---

# 3. Documentación interactiva

FastAPI genera automáticamente la documentación OpenAPI.

Swagger UI:

```
/docs
```

ReDoc:

```
/redoc
```

---

# 4. Endpoints disponibles

| Método | Endpoint | Descripción |
|---------|----------|-------------|
| GET | `/` | Información general del observatorio |
| GET | `/api/health` | Estado del servicio |
| GET | `/api/models` | Modelos disponibles |
| GET | `/api/dashboard` | Dashboard completo |
| GET | `/api/history` | Serie temporal histórica |
| POST | `/api/analyze` | Analizar un texto individual |

---

# 5. GET /

Devuelve información general sobre el servicio.

## Ejemplo

```http
GET /
```

Respuesta

```json
{
  "name": "TerMacroMetro",
  "version": "1.0.0",
  "status": "online"
}
```

---

# 6. GET /api/health

Permite verificar el estado del servicio.

## Ejemplo

```http
GET /api/health
```

Respuesta

```json
{
  "status": "ok",
  "models_loaded": true,
  "dashboard_available": true
}
```

Este endpoint es utilizado por:

- balanceadores de carga;
- monitoreo;
- GitHub Actions;
- pruebas automáticas.

---

# 7. GET /api/models

Lista los modelos disponibles.

Ejemplo:

```http
GET /api/models
```

Respuesta

```json
{
  "classical": {
    "name": "TF-IDF + Multinomial Naive Bayes",
    "enabled": true
  },
  "transformer": {
    "name": "finance-sentiment-es-base",
    "enabled": true
  },
  "topic_model": {
    "name": "LDA Bolivia",
    "enabled": true
  }
}
```

---

# 8. GET /api/dashboard

Es el endpoint principal del observatorio.

Devuelve todos los indicadores agregados calculados durante la ejecución diaria.

Ejemplo

```http
GET /api/dashboard
```

Respuesta simplificada

```json
{
  "total_documents": 208,
  "sources": 8,
  "macro_score": 39.88,
  "optimism": 59.3,
  "average_stress": 57.4,
  "dominant_topic": "Tipo de cambio"
}
```

El dashboard también incluye:

- indicadores;
- instituciones;
- temas;
- entidades;
- distribución de sentimiento;
- evaluación de modelos.

---

# 9. GET /api/history

Devuelve la serie temporal del observatorio.

Ejemplo

```http
GET /api/history
```

Respuesta

```json
[
  {
    "date": "2026-07-11",
    "macro_score": 39.9,
    "optimism": 59.4,
    "stress": 57.4
  },
  {
    "date": "2026-07-12",
    "macro_score": 39.6,
    "optimism": 58.8,
    "stress": 57.3
  }
]
```

Conforme aumenta el histórico, este endpoint permite construir gráficos temporales.

---

# 10. POST /api/analyze

Analiza un texto individual.

Solicitud

```http
POST /api/analyze
Content-Type: application/json
```

Cuerpo

```json
{
  "text": "El Banco Central anunció nuevas medidas para fortalecer las reservas internacionales."
}
```

Respuesta

```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "entities": [
    "BCB",
    "reservas internacionales"
  ],
  "topic": "Política monetaria"
}
```

Este endpoint resulta útil para:

- pruebas;
- integración;
- investigación;
- validación de modelos.

---

# 11. Códigos de estado

| Código | Significado |
|---------|-------------|
| 200 | Solicitud exitosa |
| 400 | Solicitud inválida |
| 404 | Recurso no encontrado |
| 422 | Error de validación |
| 500 | Error interno |

---

# 12. Formato de errores

Las respuestas de error utilizan JSON.

Ejemplo

```json
{
  "detail": "Text must contain at least ten words."
}
```

---

# 13. Rendimiento

La API no realiza entrenamiento durante las solicitudes.

Los modelos ya se encuentran cargados en memoria o los resultados fueron previamente serializados.

Esto permite tiempos de respuesta del orden de milisegundos para la mayoría de los endpoints.

---

# 14. Seguridad

La versión 1.0 no implementa autenticación.

La API está pensada inicialmente para uso público y demostración.

Versiones futuras podrán incorporar:

- API Keys;
- OAuth2;
- JWT;
- Rate Limiting.

---

# 15. Versionado

La API utiliza versionado semántico del proyecto.

Versión actual

```
1.0.0
```

Cambios incompatibles producirán una nueva versión mayor.

---

# 16. Compatibilidad

La API devuelve exclusivamente JSON codificado en UTF-8.

Puede ser consumida desde:

- JavaScript;
- Python;
- Java;
- Go;
- C#;
- aplicaciones móviles;
- herramientas BI.

---

# 17. Ejemplo utilizando Python

```python
import requests

response = requests.get(
    "http://127.0.0.1:8000/api/dashboard"
)

dashboard = response.json()

print(dashboard["macro_score"])
```

---

# 18. Ejemplo utilizando JavaScript

```javascript
const response = await fetch("/api/dashboard");

const dashboard = await response.json();

console.log(dashboard.macro_score);
```

---

# 19. Roadmap

La API evolucionará incorporando:

- filtros por fecha;
- filtros por fuente;
- consultas por departamento;
- búsqueda de entidades;
- consulta por tema;
- autenticación;
- documentación OpenAPI extendida.

---

# 20. Conclusión

La API de TerMacroMetro proporciona una interfaz sencilla y estable para consultar los indicadores generados por el observatorio.

Su diseño desacoplado respecto al pipeline de procesamiento facilita el despliegue, mejora el rendimiento y permite integrar el observatorio con aplicaciones externas de forma simple.

---

**TerMacroMetro v1.0.0**

**Observatorio Inteligente de la Economía Boliviana**

Desarrollado por **Jhonny Paul Pinto Phillips**