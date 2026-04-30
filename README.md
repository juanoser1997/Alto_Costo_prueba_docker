# MLOps Final Project - Prediccion de Reingreso Hospitalario en Pacientes con Enfermedades de Alto Costo

Universidad de Medellin  
Curso MLOps - Proyecto Final Integrador

---

## Descripcion del Problema

Las enfermedades de alto costo (diabetes, insuficiencia cardiaca, enfermedad renal cronica) generan
un alto volumen de reingresos hospitalarios evitables. En Colombia, el sistema de salud penaliza
a las IPS por reingresos dentro de los 30 dias posteriores al alta, lo que genera un fuerte
incentivo para la prevencion.

Este proyecto implementa un pipeline MLOps completo para predecir si un paciente diabetico
sera reingresado al hospital en menos de 30 dias, permitiendo a los equipos clinicos
intervenir de forma preventiva antes del alta.

**Dataset:** Diabetes 130-US Hospitals (1999-2008)
**Fuente:** UCI Machine Learning Repository
**URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
**Tarea:** Clasificacion binaria (reingresado en < 30 dias: Si / No)
**Metrica principal:** ROC-AUC (prioridad en reducir falsos negativos)
**Filas:** 101,766 encuentros hospitalarios
**Columnas:** 50 variables clinicas y administrativas

---

## Contexto de Negocio

Una IPS hipotetica en Colombia necesita reducir su tasa de reingreso en 30 dias del 11%
actual al 7%. El modelo permite:

- Identificar pacientes de alto riesgo antes del alta
- Priorizar visitas de seguimiento del equipo de enfermeria
- Optimizar la asignacion de recursos de gestion de casos cronicos
- Cumplir con indicadores de calidad de la Resolucion 256 de 2016 del MinSalud

**Objetivo de negocio:** Reducir reingresos evitables en un 30% en los primeros 6 meses
de implementacion del modelo.

---

## Stack Tecnologico

| Componente              | Tecnologia               | Version  |
|-------------------------|--------------------------|----------|
| Lenguaje                | Python                   | 3.11+    |
| Dependencias            | uv                       | latest   |
| Experiment Tracking     | MLflow                   | 3.x      |
| Optimizacion            | Optuna                   | 4.x      |
| Orquestacion            | Prefect                  | 3.x      |
| API                     | FastAPI + Uvicorn        | latest   |
| Contenerizacion         | Docker + Compose         | latest   |
| CI/CD                   | GitHub Actions           | -        |
| Calidad de codigo       | ruff                     | latest   |
| Testing                 | pytest                   | 8.x      |
| Monitoreo               | Evidently                | latest   |

---

## Estructura del Repositorio

```
mlops-alto-costo/
├── README.md
├── pyproject.toml                   # Dependencias con uv
├── .python-version                  # Version de Python fijada
├── .pre-commit-config.yaml          # Hooks: ruff, tests basicos
├── .env.example                     # Variables de entorno de referencia
├── CONTRIBUTING.md                  # Guia de contribucion
├── COURSE_OVERVIEW.md               # Mapeo del proyecto a modulos del curso
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + tests en cada push
│       └── deploy.yml               # Build y push Docker en merge a main
├── .githooks/
│   └── pre-push                     # Hook adicional para tests antes de push
├── src/
│   ├── data/
│   │   ├── download.py              # Descarga desde UCI
│   │   └── preprocess.py            # Limpieza, encoding, split
│   ├── features/
│   │   └── engineering.py           # Features clinicas derivadas
│   ├── models/
│   │   ├── train.py                 # Entrenamiento + MLflow + Optuna
│   │   └── predict.py               # Inferencia desde MLflow Registry
│   ├── api/
│   │   ├── main.py                  # FastAPI app
│   │   └── schemas.py               # Pydantic schemas
│   └── monitoring/
│       └── drift.py                 # Deteccion de drift con Evidently
├── flows/
│   └── training_pipeline.py         # Prefect flow orquestado
├── notebooks/
│   ├── 01_eda.ipynb                 # Analisis exploratorio
│   ├── 02_baseline.ipynb            # Modelo baseline
│   └── 03_experiments.ipynb         # Optuna + MLflow
├── tests/
│   ├── unit/
│   │   ├── test_preprocess.py
│   │   ├── test_features.py
│   │   └── test_api.py
│   └── integration/
│       └── test_pipeline.py
├── configs/
│   ├── model_config.yaml
│   └── mlflow_config.yaml
├── data/
│   ├── raw/                         # No versionado en git
│   ├── processed/                   # No versionado en git
│   └── external/
├── models/
│   ├── registry/
│   └── artifacts/
├── logs/
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── deployment.md
│   └── monitoring.md
├── Dockerfile
└── docker-compose.yml
```

---

## Inicio Rapido

### Requisitos

- Python 3.11+
- uv: `pip install uv` o `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker Desktop (para el stack completo)

### 1. Clonar e instalar

```bash
git clone https://github.com/<tu-usuario>/mlops-alto-costo.git
cd mlops-alto-costo

uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Configurar pre-commit

```bash
uv run pre-commit install
```

### 3. Descargar el dataset

```bash
uv run python src/data/download.py
```

Guarda el archivo en `data/raw/diabetic_data.csv` (aprox. 23 MB).

### 4. Iniciar MLflow

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./models/artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Abrir http://localhost:5000

### 5. Ejecutar el pipeline de entrenamiento con Prefect

```bash
# En una terminal, levantar el servidor de Prefect
uv run prefect server start

# En otra terminal, ejecutar el flow completo
uv run python flows/training_pipeline.py
```

Dashboard de Prefect: http://localhost:4200

### 6. Promover el modelo a Production en MLflow

```python
import mlflow

client = mlflow.tracking.MlflowClient("sqlite:///mlflow.db")
client.transition_model_version_stage(
    name="diabetes-readmission-xgboost",
    version=1,
    stage="Production",
)
```

### 7. Levantar la API

```bash
# Modo desarrollo
uv run uvicorn src.api.main:app --reload --port 8000

# Stack completo con Docker
docker-compose up --build
```

Documentacion: http://localhost:8000/docs

### 8. Ejecutar los tests

```bash
uv run pytest tests/unit/ -v
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Dataset: Diabetes 130-US Hospitals

**URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
**Periodo:** 1999-2008
**Encuentros:** 101,766
**Caracteristicas:** 50 (demograficas, clinicas, farmacologicas)

Variables clave:

| Variable          | Descripcion                                          |
|-------------------|------------------------------------------------------|
| age               | Grupo de edad en intervalos de 10 anos               |
| time_in_hospital  | Dias de hospitalizacion (1-14)                       |
| num_medications   | Numero de medicamentos distintos administrados       |
| num_lab_procedures| Numero de procedimientos de laboratorio              |
| number_diagnoses  | Numero de diagnosticos ingresados al sistema         |
| A1Cresult         | Resultado de HbA1c (>8, >7, normal, no realizado)    |
| insulin           | Cambio en dosis de insulina                          |
| diabetesMed       | Se prescribio medicamento para diabetes              |
| readmitted        | Reingreso: <30 dias, >30 dias, o no reingresado      |

**Target binario:** `readmitted_lt30` = 1 si reingreso en menos de 30 dias, 0 en caso contrario.

**Balance de clases:** ~11% positivo (reingreso < 30 dias), clase desbalanceada.

---

## Metricas de Exito del Modelo

| Metrica     | Objetivo  | Justificacion                                              |
|-------------|-----------|-------------------------------------------------------------|
| ROC-AUC     | > 0.72    | Metrica principal: evalua discriminacion sin umbral fijo    |
| Recall      | > 0.65    | Prioridad en no dejar pasar reingresos (falsos negativos)   |
| Precision   | > 0.25    | Aceptable dado el desbalance de clases                      |
| F1-Score    | > 0.35    | Balance entre precision y recall                            |

---

## Flujo MLOps Completo

```
UCI Repository
      |
      v
src/data/download.py          --> data/raw/diabetic_data.csv
      |
      v
src/data/preprocess.py        --> data/processed/
      |
      v
src/features/engineering.py   --> features clinicas derivadas
      |
      v
src/models/train.py           --> MLflow Tracking + Optuna
      |
      v
MLflow Model Registry         --> Staging -> Production
      |
      v
src/api/main.py (FastAPI)     --> /predict endpoint
      |
      v
Docker Container              --> despliegue reproducible
      |
      v
src/monitoring/drift.py       --> deteccion de drift con Evidently
```

---

## Estrategia de Commits

Ver `docs/deployment.md` para la guia completa de 12 commits progresivos
usando Conventional Commits.

---

## Relacion con los Modulos del Curso

Ver `COURSE_OVERVIEW.md` para el mapeo detallado de cada componente del
proyecto con los modulos del curso MLOps - Universidad de Medellin.

---

## Docker corregido / arranque autonomo

Este paquete incluye un `docker-compose.yml` corregido con tres servicios:

1. `mlflow`: tracking server y registry persistente.
2. `trainer`: descarga el dataset UCI, preprocesa, entrena, registra y promueve el modelo.
3. `api`: arranca solo cuando el modelo y `data/processed/X_train.csv` ya existen.

Ejecucion recomendada:

```bash
docker compose up --build
```

Por defecto `N_TRIALS=2` para que el primer arranque sea mas rapido en Docker. Para entrenamiento completo:

```bash
N_TRIALS=20 docker compose up --build
```

Ver tambien `README_DOCKER_CORREGIDO.md`.
