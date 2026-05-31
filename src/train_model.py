"""
Entrenamiento y evaluación del modelo de predicción de churn.

Compara tres algoritmos de clasificación, selecciona el mejor por AUC-ROC,
evalúa con validación cruzada y guarda todos los artefactos.
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    average_precision_score, precision_recall_curve, fbeta_score,
)
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

CAT_FEATURES = [
    "genero", "tiene_pareja", "tiene_dependientes", "servicio_telefono",
    "servicio_internet", "soporte_tecnico", "tipo_contrato",
    "metodo_pago", "factura_electronica",
]
NUM_FEATURES = [
    "edad", "es_senior", "meses_antiguedad", "cargo_mensual",
    "cargo_total", "tickets_soporte",
]
TARGET = "churn"

# GradientBoosting no acepta class_weight; se balancea con sample_weight en fit().
MODELOS_SIN_CLASS_WEIGHT = {"Gradient Boosting"}

MODELOS_CANDIDATOS = {
    "Regresión Logística": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ),
}


def cargar_datos(ruta=None):
    if ruta is None:
        ruta = PROJECT_ROOT / "data" / "clientes.csv"
    return pd.read_csv(ruta)


def construir_pipeline(clasificador):
    """Devuelve un Pipeline de sklearn: preprocesamiento + clasificador dado."""
    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            # StandardScaler normaliza las numéricas: necesario para LR,
            # inocuo para modelos basados en árboles.
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )
    return Pipeline(steps=[
        ("preprocesador", preprocesador),
        ("clasificador", clasificador),
    ])


def comparar_modelos(X_train, X_test, y_train, y_test):
    """Entrena cada candidato y devuelve tabla de métricas + pipelines entrenados."""
    resultados = []
    pipelines_entrenados = {}
    sample_weight = compute_sample_weight("balanced", y_train)

    for nombre, clf in MODELOS_CANDIDATOS.items():
        logger.info("Entrenando: %s...", nombre)
        pipeline = construir_pipeline(clf)
        if nombre in MODELOS_SIN_CLASS_WEIGHT:
            pipeline.fit(X_train, y_train, clasificador__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        reporte = classification_report(y_test, y_pred, output_dict=True)

        fila = {
            "modelo": nombre,
            "auc_roc": round(roc_auc_score(y_test, y_proba), 3),
            "avg_precision": round(average_precision_score(y_test, y_proba), 3),
            "precision_churn": round(reporte["1"]["precision"], 3),
            "recall_churn": round(reporte["1"]["recall"], 3),
            "f1_churn": round(reporte["1"]["f1-score"], 3),
            # F2-score: beta=2 da el doble de peso al Recall que a la Precisión.
            # En churn, no detectar a un cliente que se va (falso negativo) tiene
            # mayor costo que activar retención en alguien que no la necesitaba
            # (falso positivo). F2 formaliza ese trade-off de negocio.
            "f2_churn": round(fbeta_score(y_test, y_pred, beta=2), 3),
        }
        resultados.append(fila)
        pipelines_entrenados[nombre] = pipeline
        logger.info(
            "  AUC-ROC=%.3f  PR-AUC=%.3f  Recall=%.3f  F2=%.3f",
            fila["auc_roc"], fila["avg_precision"], fila["recall_churn"], fila["f2_churn"],
        )

    # Ordenar por F2-score: criterio principal de selección para este problema.
    # AUC-ROC es útil para comparar capacidad discriminante general, pero F2
    # refleja directamente el objetivo de negocio: maximizar detección de churn
    # sin ignorar completamente la precisión.
    tabla = (
        pd.DataFrame(resultados)
        .sort_values("f2_churn", ascending=False)
        .reset_index(drop=True)
    )
    return tabla, pipelines_entrenados


def entrenar_y_evaluar():
    df = cargar_datos()
    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("=" * 55)
    logger.info("COMPARACIÓN DE MODELOS")
    logger.info("=" * 55)
    tabla, pipelines = comparar_modelos(X_train, X_test, y_train, y_test)
    logger.info("\n%s", tabla.to_string(index=False))

    mejor_nombre = tabla.iloc[0]["modelo"]
    mejor_modelo = pipelines[mejor_nombre]
    logger.info("Mejor modelo: %s (AUC-ROC: %.3f)", mejor_nombre, tabla.iloc[0]["auc_roc"])

    # Validación cruzada del mejor modelo — 5 folds sobre el dataset completo
    logger.info("Validación cruzada 5-fold...")
    cv_scores = cross_val_score(mejor_modelo, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    logger.info(
        "CV AUC-ROC: %.3f ± %.3f  (folds: %s)",
        cv_scores.mean(), cv_scores.std(), np.round(cv_scores, 3),
    )

    # Reporte detallado del ganador
    y_pred = mejor_modelo.predict(X_test)
    y_proba = mejor_modelo.predict_proba(X_test)[:, 1]
    logger.info("\n%s\nREPORTE DETALLADO — %s\n%s", "=" * 55, mejor_nombre, "=" * 55)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))
    logger.info("AUC-ROC:         %.3f", roc_auc_score(y_test, y_proba))
    logger.info("Average Precision (PR-AUC): %.3f", average_precision_score(y_test, y_proba))
    logger.info("Matriz de confusión:\n%s", confusion_matrix(y_test, y_pred))

    importancias = obtener_importancias(mejor_modelo)
    if importancias is not None:
        logger.info("Top 10 factores:\n%s", importancias.head(10).to_string(index=False))

    # Artefactos adicionales para el dashboard
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_df = pd.DataFrame({"precision": prec_vals, "recall": rec_vals})
    cv_df = pd.DataFrame({"fold": range(1, len(cv_scores) + 1), "auc_roc": np.round(cv_scores, 4)})

    # Guardar todo
    models_dir = PROJECT_ROOT / "models"
    joblib.dump(mejor_modelo, models_dir / "modelo_churn.pkl")
    tabla.to_csv(models_dir / "comparacion_modelos.csv", index=False)
    pr_df.to_csv(models_dir / "pr_curve.csv", index=False)
    cv_df.to_csv(models_dir / "cv_scores.csv", index=False)
    if importancias is not None:
        importancias.to_csv(models_dir / "importancias.csv", index=False)

    logger.info("Artefactos guardados en %s", models_dir)
    return mejor_modelo, tabla, importancias


def obtener_importancias(modelo):
    """Extrae importancia de variables; retorna None para modelos sin feature_importances_."""
    clf = modelo.named_steps["clasificador"]
    if not hasattr(clf, "feature_importances_"):
        return None
    encoder = modelo.named_steps["preprocesador"].named_transformers_["cat"]
    nombres_cat = encoder.get_feature_names_out(CAT_FEATURES)
    nombres = list(nombres_cat) + NUM_FEATURES
    valores = clf.feature_importances_
    return (
        pd.DataFrame({"variable": nombres, "importancia": valores})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    entrenar_y_evaluar()
