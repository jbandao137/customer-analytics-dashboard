"""
Tests para src/train_model.py.

Cubren tres niveles:
  1. Estructura del pipeline (construir_pipeline)
  2. Comportamiento de la comparación (comparar_modelos)
  3. Utilidades auxiliares (obtener_importancias, serialización del modelo)
"""
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.train_model import (
    CAT_FEATURES,
    MODELOS_CANDIDATOS,
    NUM_FEATURES,
    comparar_modelos,
    construir_pipeline,
    obtener_importancias,
)


# ---------------------------------------------------------------------------
# construir_pipeline
# ---------------------------------------------------------------------------

def test_construir_pipeline_devuelve_pipeline():
    pipe = construir_pipeline(LogisticRegression(max_iter=100))
    assert isinstance(pipe, Pipeline)


def test_pipeline_tiene_pasos_correctos():
    pipe = construir_pipeline(LogisticRegression(max_iter=100))
    assert list(pipe.named_steps.keys()) == ["preprocesador", "clasificador"]


def test_pipeline_predice_solo_0_y_1(datos_split):
    X_train, X_test, y_train, y_test = datos_split
    pipe = construir_pipeline(LogisticRegression(max_iter=300))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_pipeline_produce_probabilidades_validas(datos_split):
    X_train, X_test, y_train, y_test = datos_split
    pipe = construir_pipeline(LogisticRegression(max_iter=300))
    pipe.fit(X_train, y_train)
    probas = pipe.predict_proba(X_test)[:, 1]
    assert ((probas >= 0) & (probas <= 1)).all()


# ---------------------------------------------------------------------------
# comparar_modelos
# ---------------------------------------------------------------------------

def test_comparacion_incluye_todos_los_candidatos(resultado_comparacion):
    tabla, _ = resultado_comparacion
    assert set(tabla["modelo"]) == set(MODELOS_CANDIDATOS.keys())


def test_comparacion_tiene_columnas_esperadas(resultado_comparacion):
    tabla, _ = resultado_comparacion
    assert set(tabla.columns) == {
        "modelo", "auc_roc", "avg_precision",
        "precision_churn", "recall_churn", "f1_churn", "f2_churn",
    }


def test_todas_las_metricas_entre_0_y_1(resultado_comparacion):
    tabla, _ = resultado_comparacion
    for col in ["auc_roc", "precision_churn", "recall_churn", "f1_churn", "f2_churn"]:
        assert tabla[col].between(0.0, 1.0).all(), f"Métrica fuera de rango [0,1]: {col}"


def test_tabla_ordenada_por_f2_descendente(resultado_comparacion):
    """La tabla se ordena por F2-score: criterio de negocio para churn."""
    tabla, _ = resultado_comparacion
    f2s = tabla["f2_churn"].tolist()
    assert f2s == sorted(f2s, reverse=True)


def test_mejor_modelo_supera_umbral_minimo(resultado_comparacion):
    """Un AUC-ROC < 0.60 indicaría que el modelo no aprende nada útil."""
    tabla, _ = resultado_comparacion
    assert tabla["auc_roc"].max() > 0.60, (
        f"Ningún modelo supera AUC-ROC 0.60. Mejor: {tabla['auc_roc'].max():.3f}"
    )


# ---------------------------------------------------------------------------
# obtener_importancias
# ---------------------------------------------------------------------------

def test_importancias_none_para_logistic_regression(datos_split):
    """LR tiene coef_, no feature_importances_; la función debe retornar None."""
    X_train, X_test, y_train, y_test = datos_split
    pipe = construir_pipeline(LogisticRegression(max_iter=300))
    pipe.fit(X_train, y_train)
    assert obtener_importancias(pipe) is None


def test_importancias_dataframe_para_random_forest(datos_split):
    X_train, X_test, y_train, y_test = datos_split
    pipe = construir_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe.fit(X_train, y_train)
    imp = obtener_importancias(pipe)
    assert imp is not None
    assert set(imp.columns) == {"variable", "importancia"}


def test_importancias_suman_uno(datos_split):
    """feature_importances_ de sklearn siempre suma exactamente 1.0."""
    X_train, X_test, y_train, y_test = datos_split
    pipe = construir_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe.fit(X_train, y_train)
    imp = obtener_importancias(pipe)
    assert abs(imp["importancia"].sum() - 1.0) < 1e-6


def test_importancias_no_negativas(datos_split):
    X_train, X_test, y_train, y_test = datos_split
    pipe = construir_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe.fit(X_train, y_train)
    imp = obtener_importancias(pipe)
    assert (imp["importancia"] >= 0).all()


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------

def test_modelo_persiste_y_predice_igual(datos_split, tmp_path):
    """El modelo guardado con joblib debe reproducir exactamente las mismas predicciones."""
    X_train, X_test, y_train, y_test = datos_split
    pipe = construir_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe.fit(X_train, y_train)

    ruta = tmp_path / "modelo_temp.pkl"
    joblib.dump(pipe, ruta)
    cargado = joblib.load(ruta)

    import numpy as np
    probas_original = pipe.predict_proba(X_test)[:, 1]
    probas_cargado = cargado.predict_proba(X_test)[:, 1]
    assert np.allclose(probas_original, probas_cargado)
