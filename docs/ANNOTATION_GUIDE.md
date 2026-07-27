# Guía de anotación de sentimiento macroeconómico — TerMacroMetro

## Objetivo

Clasificar textos económicos según el impacto macroeconómico que describen
para Bolivia, no según la emoción, el tono periodístico ni la opinión personal
del anotador.

## Etiquetas

### positivo

El texto describe una mejora, alivio o resultado favorable relevante para la
economía, los hogares, las empresas, el empleo, la inversión, la producción,
la estabilidad financiera o la disponibilidad de bienes y servicios.

Ejemplos:

- aumentan las exportaciones;
- disminuye la inflación;
- mejora el empleo;
- crecen la producción o las reservas;
- se normaliza el abastecimiento de combustibles;
- se reduce el déficit;
- aumenta la inversión;
- bajan los costos relevantes.

### negativo

El texto describe un deterioro, riesgo o resultado desfavorable relevante para
la economía, los hogares, las empresas, el empleo, la inversión, la producción,
la estabilidad financiera o la disponibilidad de bienes y servicios.

Ejemplos:

- aumenta la inflación;
- escasean combustibles o divisas;
- disminuyen reservas, producción o exportaciones;
- aumentan desempleo, déficit o deuda;
- bloqueos afectan actividad económica;
- cierran empresas;
- suben costos esenciales;
- empeoran expectativas o condiciones financieras.

### neutral

El texto es principalmente descriptivo, procedimental o informativo, y no
presenta un impacto económico inequívocamente favorable o desfavorable.

Ejemplos:

- anuncio de una reunión;
- publicación de una norma sin efectos todavía identificables;
- explicación técnica de un indicador;
- declaración institucional sin un resultado económico concreto;
- cifras mixtas sin dirección dominante;
- texto insuficiente o ambiguo.

## Reglas de decisión

1. Etiquetar el hecho económico, no las palabras emocionales.
2. Evaluar el impacto desde la perspectiva macroeconómica boliviana.
3. No inferir consecuencias que el texto no permite sostener.
4. Ante resultados mixtos, usar la consecuencia económica dominante.
5. Si no existe una consecuencia dominante, usar neutral.
6. No clasificar automáticamente críticas políticas como negativas.
7. No clasificar automáticamente anuncios gubernamentales como positivos.
8. Conservar el contexto temporal del texto.
9. Separar sentimiento de relevancia e impacto; son variables distintas.
10. Los casos dudosos deben marcarse para revisión.

## Campos mínimos de anotación

- id
- texto
- sentimiento
- anotador
- fecha_anotacion
- duda
- comentario
- fuente
- fecha_publicacion

## Control de calidad

Una muestra debe ser anotada independientemente por dos personas.

Se calculará:

- acuerdo porcentual;
- Cohen's kappa;
- desacuerdos por clase;
- matriz de confusión entre anotadores.

Los desacuerdos deben resolverse antes de usar la muestra para entrenamiento.
