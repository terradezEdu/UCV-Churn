# Unión Central Voz — Customer Churn Prediction

## 📌 Contexto

Unión Central Voz es una operadora de telecomunicaciones que busca entender y predecir la fuga de clientes (churn).

El objetivo de este proyecto es analizar múltiples fuentes de datos para identificar los principales drivers del churn y construir un modelo predictivo.

---

## Shiny App En Python

Se agrego una app interactiva en [app.py](/c:/Users/Eduar/OneDrive/Documentos/Personal/clases%20UCV/proy/UCV-Churn/app.py) para explorar rapidamente hipotesis de churn con una interfaz visual, limpia y orientada a negocio.

### Que hace

* Toma `churn_target.csv` como base analitica a nivel `cliente_id + fecha`
* Permite seleccionar un dataset de `data/raw`
* Ajusta automaticamente la granularidad antes del merge cuando hace falta
* Genera graficos y comparativas para analizar diferencias entre churn y no churn
* Muestra una tabla interactiva del dataset resultante

### Reglas De Merge

* `facturacion_mensual.csv`: join directo por `cliente_id + fecha`
* `clientes.csv`: join por `cliente_id`
* `interacciones_soporte.csv`: agregacion previa a `cliente_id + mes`
* `calidad_senal_zona_mensual.csv`: join por `zona_id + fecha` usando mapa cliente-zona mensual
* `encuestas_texto.csv`: agregacion previa a `zona_id + fecha`

### Como Ejecutarla

0. Regenerar la capa limpia si quieres refrescar `data/processed` desde `data/raw`:

```bash
python src/data/make_processed_data.py
```

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Levantar la app:

```bash
shiny run --reload app.py
```

3. Abrir la URL local que devuelva Shiny en consola.

### Nota

La app prioriza automaticamente los CSV de `data/processed` y solo cae a `data/raw` si no encuentra la version limpiada. Dentro de `data/processed` tambien se genera `data_quality_report.csv` con un resumen de las correcciones aplicadas.

## 🎯 Objetivos

* Predecir churn a nivel cliente-mes
* Identificar factores que influyen en la baja de clientes
* Detectar señales tempranas de riesgo
* Proponer posibles acciones de retención

---

## 📂 Datos disponibles

* Calidad de red por zona y mes
* Información estructural de clientes
* Facturación mensual
* Interacciones con soporte
* Encuestas con texto libre
* Variable objetivo de churn

---

## 🧩 Estructura del proyecto

* `data/` → datos originales y procesados
* `notebooks/` → análisis exploratorio y modelado
* `src/` → código reutilizable
* `models/` → modelos entrenados
* `reports/` → resultados y visualizaciones

---

## 🚀 Cómo empezar

1. Crear entorno virtual
2. Instalar dependencias
3. Explorar notebooks en orden

---

## 📊 Entregables esperados

* Análisis exploratorio
* Feature engineering
* Modelo predictivo
* Evaluación del modelo
* Conclusiones de negocio

---

## ⚠️ Importante

* Evitar fuga de información (data leakage)
* Respetar la estructura temporal
* Justificar decisiones de modelado

---
