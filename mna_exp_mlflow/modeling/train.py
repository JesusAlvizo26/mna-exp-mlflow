"""
mna_exp_mlflow/modeling/train.py
Entrena una Regresión Logística sobre el dataset Iris y registra
la corrida en MLflow (parámetros, métricas, artefacto y modelo).

Uso (desde la raíz del proyecto):
    python mna_exp_mlflow/modeling/train.py           # valores por defecto
    python mna_exp_mlflow/modeling/train.py --C 10    # cambia un hiperparámetro
    python mna_exp_mlflow/modeling/train.py --help    # muestra todas las opciones
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

COLUMNAS = ["largo_sepalo", "ancho_sepalo", "largo_petalo", "ancho_petalo"]
NOMBRE_EXPERIMENTO = "iris-logreg-ajuste-hiperparametros"

# Oculta los avisos largos de scikit-learn cuando el modelo no converge.
# En su lugar, al final se imprime un aviso corto.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def leer_argumentos():
    """Lee los hiperparámetros desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena una Regresión Logística sobre Iris.")
    parser.add_argument("--C", type=float, default=1.0, help="Regularización (más pequeño = más regularización)")
    parser.add_argument("--solver", type=str, default="lbfgs", help="Algoritmo de optimización: lbfgs, newton-cg, sag o saga")
    parser.add_argument("--max-iter", type=int, default=200, help="Número máximo de iteraciones")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--run-name", type=str, default=None, help="Nombre de la corrida en MLflow")
    return parser.parse_args()


def cargar_datos():
    """Carga Iris y elimina el registro duplicado (decisión tomada en el EDA)."""
    df = load_iris(as_frame=True).frame
    df.columns = COLUMNAS + ["target"]
    df = df.drop_duplicates()
    return df


def dividir_datos(df, seed):
    """Divide en 70 % entrenamiento y 30 % prueba, manteniendo la proporción de especies."""
    X = df[COLUMNAS]
    y = df["target"]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def entrenar_modelo(X_train, y_train, C, solver, max_iter, seed):
    """Crea y entrena el modelo."""
    modelo = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=seed)
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    """Calcula las métricas del modelo y las regresa en un diccionario."""
    y_pred = modelo.predict(X_test)
    metricas = {
        "accuracy_train": accuracy_score(y_train, modelo.predict(X_train)),
        "accuracy_test": accuracy_score(y_test, y_pred),
        "f1_macro_test": f1_score(y_test, y_pred, average="macro", zero_division=0),
        # Validación cruzada de 5 partes sobre el entrenamiento: estimación más estable
        "cv_accuracy": cross_val_score(modelo, X_train, y_train, cv=5).mean(),
        # Si es igual a max_iter, el modelo no alcanzó a converger
        "iteraciones": int(modelo.n_iter_[0]),
    }
    return metricas, y_pred


def guardar_matriz_confusion(y_test, y_pred, ruta):
    """Dibuja la matriz de confusión y la guarda como imagen."""
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["setosa", "versicolor", "virginica"], cmap="Blues"
    )
    plt.title("Matriz de confusión")
    plt.savefig(ruta)
    plt.close()


def main():
    args = leer_argumentos()

    # 1. Datos
    df = cargar_datos()
    X_train, X_test, y_train, y_test = dividir_datos(df, args.seed)

    # 2. Configurar MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)

    # Si no se da un nombre, se arma uno con los hiperparámetros
    nombre_corrida = args.run_name
    if nombre_corrida is None:
        nombre_corrida = f"C={args.C}_{args.solver}_iter={args.max_iter}_seed={args.seed}"

    with mlflow.start_run(run_name=nombre_corrida):
        # 3. Registrar hiperparámetros
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("seed", args.seed)

        # 4. Entrenar y evaluar
        modelo = entrenar_modelo(X_train, y_train, args.C, args.solver, args.max_iter, args.seed)
        metricas, y_pred = evaluar_modelo(modelo, X_train, y_train, X_test, y_test)

        # 5. Registrar métricas
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        # 6. Registrar artefacto (matriz de confusión)
        os.makedirs("reports/evaluacion", exist_ok=True)
        ruta_imagen = "reports/evaluacion/matriz_confusion.png"
        guardar_matriz_confusion(y_test, y_pred, ruta_imagen)
        mlflow.log_artifact(ruta_imagen)

        # 7. Registrar el modelo entrenado
        mlflow.sklearn.log_model(modelo, name="modelo", input_example=X_train.head(3))

    # 8. Mostrar resultados en la terminal
    print("Resultados:")
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {round(valor, 3)}")
    if metricas["iteraciones"] == args.max_iter:
        print("  AVISO: el modelo no convergió (llegó al máximo de iteraciones).")


if __name__ == "__main__":
    main()
