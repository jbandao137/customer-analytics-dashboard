"""
Fixtures compartidas entre todos los módulos de tests.

scope="session" significa que cada fixture se computa UNA sola vez por
ejecución de pytest, sin importar cuántos tests la usen. Esto evita
entrenar modelos repetidamente y mantiene la suite rápida.
"""
import pytest
from sklearn.model_selection import train_test_split

from src.generate_data import generar_datos
from src.train_model import comparar_modelos, CAT_FEATURES, NUM_FEATURES


@pytest.fixture(scope="session")
def dataset():
    """Dataset sintético pequeño (600 filas) para toda la sesión de tests."""
    return generar_datos(n=600)


@pytest.fixture(scope="session")
def datos_split(dataset):
    """Split estratificado 80/20 listo para entrenar y evaluar."""
    X = dataset[CAT_FEATURES + NUM_FEATURES]
    y = dataset["churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@pytest.fixture(scope="session")
def resultado_comparacion(datos_split):
    """Resultado de comparar_modelos(), calculado una vez para todos los tests."""
    X_train, X_test, y_train, y_test = datos_split
    return comparar_modelos(X_train, X_test, y_train, y_test)
