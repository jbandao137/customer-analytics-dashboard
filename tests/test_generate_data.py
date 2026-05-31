"""
Tests para src/generate_data.py.

Verifican que el generador produzca datos con la forma, calidad y
distribuciones esperadas. Si alguien cambia la lógica de generación,
estos tests lo detectan.
"""
import pandas as pd
import pytest
from src.generate_data import generar_datos

COLUMNAS_ESPERADAS = {
    "id_cliente", "genero", "edad", "es_senior", "tiene_pareja",
    "tiene_dependientes", "meses_antiguedad", "servicio_telefono",
    "servicio_internet", "soporte_tecnico", "tipo_contrato",
    "metodo_pago", "factura_electronica", "cargo_mensual",
    "cargo_total", "tickets_soporte", "churn", "churn_label",
}


def test_devuelve_n_filas():
    """El parámetro n debe controlar exactamente el número de registros."""
    df = generar_datos(n=300)
    assert len(df) == 300


def test_columnas_completas():
    """El dataset debe tener exactamente las columnas del contrato."""
    df = generar_datos(n=100)
    assert set(df.columns) == COLUMNAS_ESPERADAS


def test_sin_nulos():
    """No debe haber valores faltantes; el modelo no los toleraría en producción."""
    df = generar_datos(n=500)
    assert df.isnull().sum().sum() == 0


def test_churn_es_binario():
    """La variable objetivo solo puede ser 0 o 1."""
    df = generar_datos(n=500)
    assert set(df["churn"].unique()).issubset({0, 1})


def test_tasa_churn_en_rango_realista():
    """La tasa de churn debe estar entre 10% y 60%; fuera de ese rango el modelo no tiene sentido."""
    df = generar_datos(n=2000)
    tasa = df["churn"].mean()
    assert 0.10 <= tasa <= 0.60, f"Tasa de churn inesperada: {tasa:.2%}"


def test_ids_son_unicos():
    """Cada cliente debe tener un identificador irrepetible."""
    df = generar_datos(n=500)
    assert df["id_cliente"].nunique() == len(df)


def test_es_senior_derivado_de_edad():
    """es_senior es una variable derivada: debe ser 1 si y solo si edad >= 65."""
    df = generar_datos(n=500)
    esperado = (df["edad"] >= 65).astype(int)
    pd.testing.assert_series_equal(df["es_senior"], esperado, check_names=False)


def test_cargo_total_positivo():
    """Los cargos totales deben ser siempre positivos."""
    df = generar_datos(n=500)
    assert (df["cargo_total"] > 0).all()
