# Reporte de evaluación — RAG híbrido con Pinecone

Generado: 2026-10-04T16:40:48-03:00 por `python evaluate.py`.

## Configuración

- **index_name**: `techcorp-rag-e5`
- **namespace**: `techcorp-dev`
- **cloud**: `aws`
- **region**: `us-east-1`
- **metric**: `cosine`
- **embedding_provider**: `huggingface`
- **embedding_model**: `intfloat/multilingual-e5-small`
- **chunk_size**: `500`
- **chunk_overlap**: `75`
- **top_k**: `5`
- **candidate_k**: `10`
- **weights**: `[0.3, 0.7]`
- **rrf_c**: `60`
- **k**: `5`
- **golden_set**: `eval\golden_set.json`

## Consola

```text
==================== BENCHMARK CONSIGNA (5 preguntas, k=5, hibrido) ====================
✅ q01 ¿Cuánto descanso anual me corresponde si llevo seis años trabajando en la emp...
   Esperado: politica_vacaciones.txt | Top-5: politica_vacaciones.txt, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md, politica_gastos_y_viajes.md, manual_beneficios_y_licencias.md
   Recall@5: 100.0% | Precision@5: 20.0% (techo 20.0%)
✅ q08 Se cayó el servicio principal y se filtraron datos, ¿en cuánto tiempo tienen ...
   Esperado: procedimiento_gestion_incidentes.md | Top-5: procedimiento_gestion_incidentes.md, sdk_errores_y_reintentos.md, procedimiento_gestion_incidentes.md, guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md
   Recall@5: 100.0% | Precision@5: 40.0% (techo 60.0%)
✅ q10 ¿Qué significa el error VPN-809?
   Esperado: guia_acceso_remoto_vpn.md | Top-5: guia_acceso_remoto_vpn.md, sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md, sdk_errores_y_reintentos.md
   Recall@5: 100.0% | Precision@5: 20.0% (techo 60.0%)
✅ q23 ¿Qué excepción lanza techcorp-sdk ante un HTTP 429?
   Esperado: sdk_errores_y_reintentos.md | Top-5: sdk_errores_y_reintentos.md, sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md, sdk_referencia_cliente.md, sdk_instalacion_configuracion.md
   Recall@5: 100.0% | Precision@5: 40.0% (techo 60.0%)
✅ q24 ¿Cómo instalo la librería interna de Python y qué versión del intérprete nece...
   Esperado: sdk_instalacion_configuracion.md | Top-5: sdk_instalacion_configuracion.md, sdk_referencia_cliente.md, sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md, sdk_errores_y_reintentos.md
   Recall@5: 100.0% | Precision@5: 40.0% (techo 60.0%)
📊 RECALL@5 PROMEDIO:    100.0%
📊 PRECISION@5 PROMEDIO: 32.0%  (techo promedio 52.0% → precisión normalizada 66.7%)

================================ RESUMEN (k=5, 26 casos) ================================
            P@5        P@5 norm   R@5        F1@5       Hit@5      MRR        nDCG@5     Evid.      
bm25        0.34       0.64       0.92       0.47       0.96       0.83       0.63       0.92       
vectorial   0.50       0.86       1.00       0.64       1.00       1.00       0.88       1.00       
hibrido     0.45       0.79       1.00       0.60       1.00       1.00       0.82       1.00       

Por tipo de pregunta (hibrido):
            P@5        P@5 norm   R@5        F1@5       Hit@5      MRR        nDCG@5     Evid.      
filtrada    0.60       1.00       1.00       0.75       1.00       1.00       0.98       1.00       
lexica      0.36       0.74       1.00       0.51       1.00       1.00       0.77       1.00       
multi_doc   0.64       0.86       1.00       0.77       1.00       1.00       0.87       1.00       
semantica   0.42       0.77       1.00       0.56       1.00       1.00       0.81       1.00       

Casos filtrados evaluados SIN filtro (valor del hard filter, híbrido):
  q21: R@5 con filtro 1.00 → sin filtro 0.00 | top-1 sin filtro: politica_vacaciones.txt
  q22: R@5 con filtro 1.00 → sin filtro 0.00 | top-1 sin filtro: manual_beneficios_y_licencias.md

============================== BARRIDO DE PESOS (bm25, vectorial) ==============================
            P@5        P@5 norm   R@5        F1@5       Hit@5      MRR        nDCG@5     Evid.      
0.3/0.7     0.45       0.79       1.00       0.60       1.00       1.00       0.82       1.00       
0.4/0.6     0.45       0.79       1.00       0.60       1.00       0.98       0.81       1.00       
0.5/0.5     0.44       0.77       1.00       0.58       1.00       0.97       0.79       1.00       
0.6/0.4     0.42       0.74       0.98       0.56       1.00       0.97       0.77       0.98       
0.7/0.3     0.42       0.73       0.98       0.55       1.00       0.91       0.73       0.98       
Mejor combinación por (MRR, Recall@5, Precision@5): bm25=0.3 vectorial=0.7

Conclusión breve:
- El híbrido iguala o supera al mejor retriever individual (vectorial) en Recall@5 (1.00 vs 1.00); MRR 1.00 vs 1.00.
- Preguntas léxicas: empate (MRR bm25 1.00 vs vectorial 1.00).
- Preguntas semánticas: gana el vectorial (MRR bm25 0.67 vs vectorial 1.00).
- El híbrido recupera todos los documentos esperados en todos los casos (recall=1).
```

## Comparación de modos (k=5, 26 casos)

| modo | P@5 | P@5 norm | R@5 | F1@5 | Hit@5 | MRR | nDCG@5 | Evid. |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.34 | 0.64 | 0.92 | 0.47 | 0.96 | 0.83 | 0.63 | 0.92 |
| vectorial | 0.50 | 0.86 | 1.00 | 0.64 | 1.00 | 1.00 | 0.88 | 1.00 |
| hibrido | 0.45 | 0.79 | 1.00 | 0.60 | 1.00 | 1.00 | 0.82 | 1.00 |

## Híbrido por tipo de pregunta

| tipo | P@5 | P@5 norm | R@5 | F1@5 | Hit@5 | MRR | nDCG@5 | Evid. |
|---|---|---|---|---|---|---|---|---|
| filtrada | 0.60 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 0.98 | 1.00 |
| lexica | 0.36 | 0.74 | 1.00 | 0.51 | 1.00 | 1.00 | 0.77 | 1.00 |
| multi_doc | 0.64 | 0.86 | 1.00 | 0.77 | 1.00 | 1.00 | 0.87 | 1.00 |
| semantica | 0.42 | 0.77 | 1.00 | 0.56 | 1.00 | 1.00 | 0.81 | 1.00 |

## Detalle por caso (híbrido)

| id | tipo | núcleo | esperados | top-5 | P@5 | techo | R@5 | RR | Evid. |
|---|---|---|---|---|---|---|---|---|---|
| q01 | semantica | sí | politica_vacaciones.txt | politica_vacaciones.txt, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md, politica_gastos_y_viajes.md, manual_beneficios_y_licencias.md | 0.20 | 0.20 | 1.00 | 1.00 | 1.00 |
| q02 | semantica |  | politica_teletrabajo.txt | politica_teletrabajo.txt, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md, guia_acceso_remoto_vpn.md, manual_beneficios_y_licencias.md | 0.20 | 0.20 | 1.00 | 1.00 | 1.00 |
| q03 | semantica |  | onboarding_nuevos_empleados.txt | onboarding_nuevos_empleados.txt, politica_vacaciones.txt, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md | 0.20 | 0.20 | 1.00 | 1.00 | 1.00 |
| q04 | semantica |  | manual_beneficios_y_licencias.md | manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md, politica_vacaciones.txt, manual_beneficios_y_licencias.md | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 |
| q05 | semantica |  | manual_beneficios_y_licencias.md | manual_beneficios_y_licencias.md, politica_vacaciones.txt, manual_beneficios_y_licencias.md, procedimiento_gestion_incidentes.md, politica_gastos_y_viajes.md | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 |
| q06 | semantica |  | manual_beneficios_y_licencias.md | manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md, politica_vacaciones.txt, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 |
| q07 | semantica |  | politica_gastos_y_viajes.md | politica_gastos_y_viajes.md, politica_gastos_y_viajes.md, guia_acceso_remoto_vpn.md, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q08 | semantica | sí | procedimiento_gestion_incidentes.md | procedimiento_gestion_incidentes.md, sdk_errores_y_reintentos.md, procedimiento_gestion_incidentes.md, guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q09 | semantica |  | politica_gastos_y_viajes.md | politica_gastos_y_viajes.md, politica_gastos_y_viajes.md, sdk_errores_y_reintentos.md, manual_beneficios_y_licencias.md, manual_beneficios_y_licencias.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q10 | lexica | sí | guia_acceso_remoto_vpn.md | guia_acceso_remoto_vpn.md, sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md, sdk_errores_y_reintentos.md | 0.20 | 0.60 | 1.00 | 1.00 | 1.00 |
| q11 | lexica |  | guia_acceso_remoto_vpn.md | guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md, sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md | 0.60 | 0.60 | 1.00 | 1.00 | 1.00 |
| q12 | lexica |  | politica_seguridad_informatica.txt | politica_seguridad_informatica.txt, politica_gastos_y_viajes.md, sdk_errores_y_reintentos.md, manual_beneficios_y_licencias.md, politica_teletrabajo.txt | 0.20 | 0.20 | 1.00 | 1.00 | 1.00 |
| q13 | lexica |  | procedimiento_gestion_incidentes.md | procedimiento_gestion_incidentes.md, sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md, procedimiento_gestion_incidentes.md, politica_teletrabajo.txt | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q14 | lexica |  | politica_gastos_y_viajes.md | politica_gastos_y_viajes.md, sdk_errores_y_reintentos.md, sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md, politica_gastos_y_viajes.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q15 | lexica |  | guia_acceso_remoto_vpn.md | guia_acceso_remoto_vpn.md, sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md, procedimiento_gestion_incidentes.md, guia_acceso_remoto_vpn.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q16 | lexica |  | politica_seguridad_informatica.txt | politica_seguridad_informatica.txt, guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md, sdk_errores_y_reintentos.md, politica_gastos_y_viajes.md | 0.20 | 0.20 | 1.00 | 1.00 | 1.00 |
| q17 | multi_doc |  | politica_seguridad_informatica.txt, guia_acceso_remoto_vpn.md | politica_seguridad_informatica.txt, guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md, manual_beneficios_y_licencias.md, procedimiento_gestion_incidentes.md | 0.60 | 0.80 | 1.00 | 1.00 | 1.00 |
| q18 | multi_doc |  | politica_seguridad_informatica.txt, procedimiento_gestion_incidentes.md | procedimiento_gestion_incidentes.md, procedimiento_gestion_incidentes.md, procedimiento_gestion_incidentes.md, politica_seguridad_informatica.txt, sdk_errores_y_reintentos.md | 0.80 | 0.80 | 1.00 | 1.00 | 1.00 |
| q19 | multi_doc |  | politica_teletrabajo.txt, guia_acceso_remoto_vpn.md | politica_teletrabajo.txt, guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md, manual_beneficios_y_licencias.md, politica_seguridad_informatica.txt | 0.60 | 0.80 | 1.00 | 1.00 | 1.00 |
| q20 | multi_doc |  | onboarding_nuevos_empleados.txt, politica_seguridad_informatica.txt | onboarding_nuevos_empleados.txt, guia_acceso_remoto_vpn.md, guia_acceso_remoto_vpn.md, manual_beneficios_y_licencias.md, politica_seguridad_informatica.txt | 0.40 | 0.40 | 1.00 | 1.00 | 1.00 |
| q21 | filtrada |  | politica_gastos_y_viajes.md | politica_gastos_y_viajes.md, politica_gastos_y_viajes.md, politica_gastos_y_viajes.md | 0.60 | 0.60 | 1.00 | 1.00 | 1.00 |
| q22 | filtrada |  | procedimiento_gestion_incidentes.md | procedimiento_gestion_incidentes.md, procedimiento_gestion_incidentes.md, politica_seguridad_informatica.txt, procedimiento_gestion_incidentes.md | 0.60 | 0.60 | 1.00 | 1.00 | 1.00 |
| q23 | lexica | sí | sdk_errores_y_reintentos.md | sdk_errores_y_reintentos.md, sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md, sdk_referencia_cliente.md, sdk_instalacion_configuracion.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q24 | semantica | sí | sdk_instalacion_configuracion.md | sdk_instalacion_configuracion.md, sdk_referencia_cliente.md, sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md, sdk_errores_y_reintentos.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q25 | lexica |  | sdk_referencia_cliente.md | sdk_referencia_cliente.md, sdk_errores_y_reintentos.md, sdk_referencia_cliente.md, sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md | 0.40 | 0.60 | 1.00 | 1.00 | 1.00 |
| q26 | multi_doc |  | sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md | sdk_errores_y_reintentos.md, sdk_instalacion_configuracion.md, sdk_errores_y_reintentos.md, sdk_referencia_cliente.md, sdk_instalacion_configuracion.md | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 |

## Barrido de pesos (bm25/vectorial)

| pesos | P@5 | P@5 norm | R@5 | F1@5 | Hit@5 | MRR | nDCG@5 | Evid. |
|---|---|---|---|---|---|---|---|---|
| 0.3/0.7 | 0.45 | 0.79 | 1.00 | 0.60 | 1.00 | 1.00 | 0.82 | 1.00 |
| 0.4/0.6 | 0.45 | 0.79 | 1.00 | 0.60 | 1.00 | 0.98 | 0.81 | 1.00 |
| 0.5/0.5 | 0.44 | 0.77 | 1.00 | 0.58 | 1.00 | 0.97 | 0.79 | 1.00 |
| 0.6/0.4 | 0.42 | 0.74 | 0.98 | 0.56 | 1.00 | 0.97 | 0.77 | 0.98 |
| 0.7/0.3 | 0.42 | 0.73 | 0.98 | 0.55 | 1.00 | 0.91 | 0.73 | 0.98 |

## Lectura de resultados

- **Benchmark de la consigna (5 preguntas):** el híbrido trae el documento correcto en el top-1 de las 5 (Recall@5 = 100%). Precision@5 = 32% contra un techo promedio de 52% (q01 apunta a un documento de 1 solo chunk, cuyo techo es 20%): precisión normalizada 66,7%.
- **BM25 solo es el retriever más débil** (Recall@5 0,92, MRR 0,83): falla en paráfrasis sin palabras compartidas. El caso extremo es q05 ("no soy quien gesta" ↔ "licencia parental"), con Recall 0, y en multi-documento pierde el segundo documento de q17 y q20.
- **E5 multilingüe es muy fuerte en este corpus:** el vectorial solo logra Recall@5 = 1,00 y MRR = 1,00, y empata con BM25 incluso en las preguntas léxicas (VPN-809, SEV3, GastosTC, page_size). Con 28 chunks, los códigos exactos son lo bastante raros como para que el embedding los distinga.
- **Qué aporta la fusión:** con pesos 0,5/0,5 la señal de BM25 en q05 empujaba el documento correcto al 4.º lugar (MRR del híbrido 0,97). El barrido muestra que 0,3/0,7 recupera MRR 1,00 y mantiene Recall 1,00, mientras que dar más peso a BM25 (0,6–0,7) empieza a perder documentos (Recall 0,98). El híbrido queda como red de seguridad léxica sin degradar el ranking semántico, a cambio de una Precision@5 algo menor que el vectorial puro (0,45 vs 0,50), porque BM25 suma chunks de otros documentos que comparten términos genéricos.
- **El hard filter es imprescindible en preguntas ambiguas:** sin filtro, q21 ("días de anticipación") devuelve vacaciones y mudanza (Recall 0) y q22 ("5 días hábiles") devuelve el manual de beneficios (Recall 0); con `department` = finanzas / seguridad ambas llegan a Recall 1.
- **Limitación honesta:** con un corpus chico y un embedder multilingüe moderno, la ventaja del híbrido sobre el vectorial es de robustez, no de métricas. Se espera que crezca con corpus grandes con muchos identificadores casi duplicados (códigos de error, versiones, SKUs), que es justamente el escenario de escalar el índice.

### Notas sobre las métricas

- **Precision@5** divide siempre por k. Un documento de 1 chunk como única respuesta tiene un techo de 0.20; por eso se informa el techo y la precisión normalizada (precisión / techo).
- **Recall@5** vale 0 o 1 en preguntas de un documento (benchmark núcleo) y es fraccionario en las multi-documento.
- **Evid.** mide si los chunks recuperados de documentos relevantes contienen el dato esperado.
