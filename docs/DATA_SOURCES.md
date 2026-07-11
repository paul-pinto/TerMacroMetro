# Fuentes de datos

TerMacroMetro integra organismos oficiales, reguladores, agencias, medios nacionales, regionales y portales especializados.

La configuración está centralizada en `config/bolivia_sources.json`.

Cada fuente registra su nombre, tipo, departamento, alcance, URL, método de recolección, estado, peso metodológico y límite de documentos.

## Tipos

- `official`: organismos y reguladores.
- `agency`: agencias de noticias.
- `newspaper`: prensa nacional.
- `television`: canales de televisión.
- `regional`: medios departamentales.
- `business`: portales empresariales.

Los límites por fuente buscan reducir la dominancia editorial. El peso de fuente no modifica el sentimiento individual de una noticia.

Estados disponibles: `active`, `pending_validation`, `pending_discovery` y `disabled`.

Cada documento conserva título, cuerpo, URL, fuente, fecha, metadata territorial y fecha de recolección.
