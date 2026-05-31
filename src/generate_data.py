"""
Generador de dataset realista de clientes para análisis de churn.
Simula una empresa de telecomunicaciones/servicios con patrones de abandono realistas.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

np.random.seed(42)
N = 5000

def generar_datos(n=N):
    # Variables demográficas
    genero = np.random.choice(["Masculino", "Femenino"], n)
    edad = np.random.randint(18, 80, n)
    es_senior = (edad >= 65).astype(int)
    tiene_pareja = np.random.choice(["Si", "No"], n, p=[0.48, 0.52])
    tiene_dependientes = np.random.choice(["Si", "No"], n, p=[0.30, 0.70])

    # Variables de servicio
    meses_antiguedad = np.random.randint(1, 73, n)
    servicio_telefono = np.random.choice(["Si", "No"], n, p=[0.90, 0.10])
    servicio_internet = np.random.choice(["DSL", "Fibra optica", "No"], n, p=[0.34, 0.44, 0.22])
    soporte_tecnico = np.random.choice(["Si", "No"], n, p=[0.29, 0.71])
    tipo_contrato = np.random.choice(
        ["Mensual", "Anual", "Bianual"], n, p=[0.55, 0.24, 0.21]
    )
    metodo_pago = np.random.choice(
        ["Cheque electronico", "Cheque por correo", "Transferencia bancaria", "Tarjeta credito"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    factura_electronica = np.random.choice(["Si", "No"], n, p=[0.59, 0.41])

    # Cargos
    cargo_mensual = np.round(np.random.uniform(18.0, 120.0, n), 2)
    cargo_total = np.round(cargo_mensual * meses_antiguedad * np.random.uniform(0.9, 1.1, n), 2)

    # Tickets de soporte (indicador de fricción)
    tickets_soporte = np.random.poisson(1.5, n)

    df = pd.DataFrame({
        "id_cliente": [f"CLI-{i:05d}" for i in range(n)],
        "genero": genero,
        "edad": edad,
        "es_senior": es_senior,
        "tiene_pareja": tiene_pareja,
        "tiene_dependientes": tiene_dependientes,
        "meses_antiguedad": meses_antiguedad,
        "servicio_telefono": servicio_telefono,
        "servicio_internet": servicio_internet,
        "soporte_tecnico": soporte_tecnico,
        "tipo_contrato": tipo_contrato,
        "metodo_pago": metodo_pago,
        "factura_electronica": factura_electronica,
        "cargo_mensual": cargo_mensual,
        "cargo_total": cargo_total,
        "tickets_soporte": tickets_soporte,
    })

    # Generar churn con lógica realista (probabilidad basada en factores de riesgo)
    prob_churn = np.full(n, -0.15)
    prob_churn += (df["tipo_contrato"] == "Mensual") * 0.28
    prob_churn += (df["tipo_contrato"] == "Anual") * 0.05
    prob_churn += (df["meses_antiguedad"] < 12) * 0.18
    prob_churn += (df["servicio_internet"] == "Fibra optica") * 0.10
    prob_churn += (df["soporte_tecnico"] == "No") * 0.08
    prob_churn += (df["metodo_pago"] == "Cheque electronico") * 0.10
    prob_churn += (df["cargo_mensual"] > 80) * 0.08
    prob_churn += (df["tickets_soporte"] > 3) * 0.12
    prob_churn += (df["es_senior"] == 1) * 0.06
    prob_churn += np.random.uniform(-0.04, 0.04, n)
    prob_churn = np.clip(prob_churn, 0, 0.95)

    df["churn"] = (np.random.uniform(0, 1, n) < prob_churn).astype(int)
    df["churn_label"] = df["churn"].map({1: "Si", 0: "No"})

    return df

if __name__ == "__main__":
    df = generar_datos()
    df.to_csv(PROJECT_ROOT / "data" / "clientes.csv", index=False)
    print(f"Dataset generado: {len(df)} clientes")
    print(f"Tasa de churn: {df['churn'].mean():.1%}")
    print(df.head())
