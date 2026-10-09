# Experimentación reproducible con MLflow: Regresión Logística sobre Iris

**Alumno:** Jesus Alvizo Morneo · **Matrícula:** A01411061
**Materia:** TC5061 Operaciones de aprendizaje automático (MLOps), Maestría en Inteligencia Artificial Aplicada
**Actividad:** Experimentación reproducible con MLflow

Proyecto que convierte el modelo base del notebook en un script de experimentación ejecutable desde la línea de comandos, con cada corrida registrada en MLflow Tracking.

## Resumen de resultados

Se ejecutaron 9 corridas en el experimento `iris-logreg-ajuste-hiperparametros`. La mejor configuración que convergió fue **C = 10 con lbfgs** (accuracy de prueba 0.956, F1 macro 0.955, validación cruzada 0.962). El hiperparámetro más influyente fue `C`. El análisis completo y las capturas de MLflow están en [`reports/analisis.md`](reports/analisis.md).

## Ver las corridas ya registradas (sin volver a entrenar)

El repositorio incluye `mlflow.db` (parámetros y métricas) y `mlruns/` (artefactos y modelos) con las 9 corridas. Para verlas:

```powershell
git clone <URL de este repositorio>
cd mna-exp-mlflow
conda create -n mna-exp-mlflow python=3.13 -y
conda activate mna-exp-mlflow
pip install -r requirements.txt
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir http://localhost:5000 y entrar al experimento `iris-logreg-ajuste-hiperparametros`.

> **Nota sobre los artefactos:** MLflow guarda la ruta de los artefactos como ruta absoluta de la computadora donde se ejecutaron, así que en otra computadora la pestaña *Artifacts* puede no abrirlos. Los parámetros y métricas sí se ven completos, y los archivos están en el repositorio dentro de `mlruns/`. Si se prefiere, basta con borrar `mlflow.db` y `mlruns/` y volver a ejecutar las 9 corridas de la sección *Corridas del experimento* para regenerar todo; gracias a la semilla fija, las métricas salen iguales.

## Estructura

El proyecto sigue la estructura de [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) (CCDS v2).

```
mna-exp-mlflow/
├── data/                     <- Carpetas estándar de CCDS (no se usan: Iris se carga desde scikit-learn)
├── docs/                     <- Carpeta estándar de CCDS
├── models/                   <- Carpeta estándar de CCDS (los modelos se guardan en MLflow, dentro de mlruns/)
├── notebooks/
│   └── 0.01-ja-eda-iris.ipynb  <- Exploración de datos y decisiones de limpieza (solo EDA)
├── references/               <- Carpeta estándar de CCDS
├── reports/
│   ├── analisis.md           <- Análisis de resultados (máximo una cuartilla)
│   └── figures/              <- Capturas de la tabla y la vista de comparación de MLflow
├── mna_exp_mlflow/           <- Código fuente del proyecto
│   ├── __init__.py
│   └── modeling/
│       ├── __init__.py
│       └── train.py          <- Carga, limpieza, entrenamiento, evaluación y registro en MLflow
├── mlflow.db                 <- Base de datos de MLflow con los parámetros y métricas de las corridas
├── mlruns/                   <- Artefactos y modelos registrados de cada corrida
├── LICENSE                   <- Generado por CCDS
├── Makefile                  <- Generado por CCDS (no se usa en Windows)
├── pyproject.toml            <- Configuración del proyecto generada por CCDS
├── requirements.txt          <- Dependencias con las versiones exactas usadas
└── README.md
```

`mna_exp_mlflow/modeling/train.py` está organizado en funciones con una sola responsabilidad cada una:

| Función | Responsabilidad |
|---|---|
| `leer_argumentos()` | Lee los hiperparámetros desde la línea de comandos |
| `cargar_datos()` | Carga Iris y elimina el registro duplicado (decisión del EDA) |
| `dividir_datos()` | Divide en 70 % entrenamiento y 30 % prueba (estratificado) |
| `entrenar_modelo()` | Crea y entrena la Regresión Logística |
| `evaluar_modelo()` | Calcula las métricas |
| `guardar_matriz_confusion()` | Guarda la matriz de confusión como imagen |
| `main()` | Une todos los pasos y registra todo en MLflow dentro de `mlflow.start_run()` |

## Instalación

Se usó un ambiente de [Miniconda](https://docs.anaconda.com/miniconda/) con Python 3.13. En Windows (PowerShell o Anaconda Prompt, desde la raíz del proyecto):

```powershell
conda create -n mna-exp-mlflow python=3.13 -y
conda activate mna-exp-mlflow
pip install -r requirements.txt
```

Con el ambiente activado (`(mna-exp-mlflow)` al inicio de la línea), el comando `python` ya apunta al Python del ambiente. Las librerías se instalan con `pip` dentro del ambiente de conda para que `requirements.txt` registre sus versiones exactas.

## Uso

Con valores por defecto:

```powershell
python mna_exp_mlflow/modeling/train.py
```

Cambiando hiperparámetros, sin editar el código:

```powershell
python mna_exp_mlflow/modeling/train.py --C 10 --solver newton-cg --max-iter 300 --seed 7
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la regularización (más pequeño = más regularización) |
| `--solver` | `lbfgs` | Optimizador: `lbfgs`, `newton-cg`, `sag`, `saga` |
| `--max-iter` | `200` | Iteraciones máximas del optimizador |
| `--seed` | `42` | Semilla aleatoria |
| `--run-name` | automático | Nombre de la corrida (si no se da, se arma con los hiperparámetros) |

`python mna_exp_mlflow/modeling/train.py --help` muestra esta misma lista.

## Corridas del experimento

Las 9 corridas se ejecutaron una por una desde la terminal, en este orden:

| # | Comando | Propósito |
|---|---|---|
| 1 | `python mna_exp_mlflow/modeling/train.py` | Baseline: C=1.0, lbfgs, max_iter=200, seed=42 |
| 2 | `python mna_exp_mlflow/modeling/train.py --C 0.01` | Regularización muy fuerte |
| 3 | `python mna_exp_mlflow/modeling/train.py --C 0.1` | Regularización fuerte |
| 4 | `python mna_exp_mlflow/modeling/train.py --C 10` | Regularización débil |
| 5 | `python mna_exp_mlflow/modeling/train.py --C 100` | Regularización muy débil |
| 6 | `python mna_exp_mlflow/modeling/train.py --max-iter 5` | Efecto de no dar suficientes iteraciones |
| 7 | `python mna_exp_mlflow/modeling/train.py --solver saga` | Efecto del optimizador |
| 8 | `python mna_exp_mlflow/modeling/train.py --seed 0` | Efecto de la semilla (otra división de datos) |
| 9 | `python mna_exp_mlflow/modeling/train.py --run-name baseline-repeticion` | Verificar reproducibilidad (mismos argumentos que la 1) |

En las corridas 6 y 7 se imprime un aviso de que el modelo no convergió; es esperado y se analiza en `reports/analisis.md`.

## Qué se registra en MLflow en cada corrida

- **Parámetros** (`log_param`): `C`, `solver`, `max_iter`, `seed`.
- **Métricas** (`log_metric`): `accuracy_train`, `accuracy_test`, `f1_macro_test`, `cv_accuracy` (validación cruzada de 5 partes sobre el entrenamiento) e `iteraciones` (si es igual a `max_iter`, el modelo no convergió).
- **Artefacto** (`log_artifact`): `matriz_confusion.png`.
- **Modelo** (`mlflow.sklearn.log_model`, carpeta `modelo/`): modelo entrenado con un ejemplo de entrada y sus archivos de entorno.

## Ver y comparar las corridas

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir http://localhost:5000, entrar al experimento `iris-logreg-ajuste-hiperparametros`, seleccionar las corridas con las casillas y presionar **Compare**. La UI se detiene con `Ctrl + C`.

## Reproducibilidad

**Semilla.** El argumento `--seed` se usa como `random_state` en dos lugares:

1. `train_test_split`: decide qué flores quedan en entrenamiento y cuáles en prueba.
2. `LogisticRegression`: relevante para los solvers `sag` y `saga`.

La validación cruzada (`cross_val_score` con `cv=5`) divide los datos siempre de la misma forma, así que no necesita semilla. Dos ejecuciones con los mismos argumentos producen exactamente las mismas métricas. Esto se verifica en el propio experimento: la corrida `baseline-repeticion` repite el baseline y obtiene valores idénticos.

**Dependencias.** `requirements.txt` fija las versiones exactas de las librerías principales (incluye `seaborn` e `ipykernel`, que usa el notebook de EDA). Se generó desde el ambiente de conda con:

```powershell
pip freeze | Select-String -Pattern "^(mlflow|scikit-learn|pandas|numpy|matplotlib|seaborn|ipykernel)==" | ForEach-Object { $_.Line } | Set-Content requirements.txt
```

Además, el modelo registrado en MLflow incluye su propio `requirements.txt` y `conda.yaml`.
