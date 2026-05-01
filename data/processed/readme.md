# Processed data

Datasets limpiados y validados para exploracion y modelado.

## Resumen

### churn_target
- Filas raw: 321987
- Filas processed: 321987
- Filas eliminadas: 0
- Duplicados exactos raw: 0
- Duplicados por clave raw: 0
- Duplicados por clave processed: 0
- Nota: El target ya venia bastante limpio; se normalizaron fechas y se garantizo unicidad cliente-mes.

### clientes
- Filas raw: 10150
- Filas processed: 10000
- Filas eliminadas: 150
- Duplicados exactos raw: 150
- Duplicados por clave raw: 150
- Duplicados por clave processed: 0
- Nota: Se eliminaron duplicados, se corrigieron antiguedades imposibles y se imputaron faltantes estructurales por tipo_plan.

### facturacion_mensual
- Filas raw: 326816
- Filas processed: 320987
- Filas eliminadas: 5829
- Duplicados exactos raw: 4829
- Duplicados por clave raw: 4829
- Duplicados por clave processed: 0
- Nota: Se quitaron duplicados cliente-mes, se completaron tipo_plan/zona/num_lineas desde clientes y se reconstruyo importe_total cuando faltaba.

### calidad_senal_zona_mensual
- Filas raw: 1096
- Filas processed: 1076
- Filas eliminadas: 20
- Duplicados exactos raw: 16
- Duplicados por clave raw: 16
- Duplicados por clave processed: 0
- Nota: Se deduplico zona-mes y se imputaron faltantes temporales por interpolacion dentro de cada zona.

### encuestas_texto
- Filas raw: 1015
- Filas processed: 1000
- Filas eliminadas: 15
- Duplicados exactos raw: 15
- Duplicados por clave raw: 15
- Duplicados por clave processed: 0
- Nota: Se eliminaron encuestas repetidas y se invalidaron respuestas fuera de rango para que no sesguen agregados zonales.

### interacciones_soporte
- Filas raw: 308487
- Filas processed: 303929
- Filas eliminadas: 4558
- Duplicados exactos raw: 4558
- Duplicados por clave raw: 4558
- Duplicados por clave processed: 0
- Nota: Se consolidaron eventos unicos, se alineo el mes con la fecha del evento y se imputaron duraciones/satisfaccion por canal.

## Regeneracion

Ejecuta:

```bash
python src/data/make_processed_data.py
```

Tambien se genera `data_quality_report.csv` con el detalle tabular.