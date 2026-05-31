"""
Dashboard de Analítica de Clientes y Predicción de Churn.

Tablero de gestión interactivo que muestra:
- KPIs principales del negocio
- Análisis de fricciones que afectan la experiencia del cliente
- Predictor de churn en tiempo real

Ejecutar con:  streamlit run src/dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

st.set_page_config(
    page_title="Analítica de Clientes | Churn",
    page_icon="📊",
    layout="wide",
)

# ----------------------------- Carga de datos ------------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv(PROJECT_ROOT / "data" / "clientes.csv")

@st.cache_resource
def cargar_modelo():
    ruta = PROJECT_ROOT / "models" / "modelo_churn.pkl"
    if ruta.exists():
        return joblib.load(ruta)
    return None

@st.cache_data
def cargar_importancias():
    ruta = PROJECT_ROOT / "models" / "importancias.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    return None

@st.cache_data
def cargar_comparacion():
    ruta = PROJECT_ROOT / "models" / "comparacion_modelos.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    return None

@st.cache_data
def cargar_pr_curve():
    ruta = PROJECT_ROOT / "models" / "pr_curve.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    return None

@st.cache_data
def cargar_cv_scores():
    ruta = PROJECT_ROOT / "models" / "cv_scores.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    return None

df = cargar_datos()
modelo = cargar_modelo()
importancias = cargar_importancias()
comparacion = cargar_comparacion()
pr_curve = cargar_pr_curve()
cv_scores_df = cargar_cv_scores()

# ----------------------------- Encabezado ----------------------------------
st.title("📊 Dashboard de Analítica de Clientes")
st.markdown(
    "Tablero de gestión para **monitoreo de indicadores**, "
    "**detección de fricciones** y **predicción de abandono (churn)**."
)

# ----------------------------- Filtros (sidebar) ---------------------------
st.sidebar.header("Filtros")
contratos = st.sidebar.multiselect(
    "Tipo de contrato",
    options=df["tipo_contrato"].unique(),
    default=list(df["tipo_contrato"].unique()),
)
internet = st.sidebar.multiselect(
    "Servicio de internet",
    options=df["servicio_internet"].unique(),
    default=list(df["servicio_internet"].unique()),
)
rango_antiguedad = st.sidebar.slider(
    "Antigüedad (meses)",
    int(df["meses_antiguedad"].min()),
    int(df["meses_antiguedad"].max()),
    (int(df["meses_antiguedad"].min()), int(df["meses_antiguedad"].max())),
)

df_f = df[
    (df["tipo_contrato"].isin(contratos))
    & (df["servicio_internet"].isin(internet))
    & (df["meses_antiguedad"].between(*rango_antiguedad))
]

# ----------------------------- KPIs ----------------------------------------
st.subheader("Indicadores principales")
c1, c2, c3, c4 = st.columns(4)
total = len(df_f)
tasa_churn = df_f["churn"].mean() if total else 0
ingreso_mensual = df_f["cargo_mensual"].sum()
antiguedad_prom = df_f["meses_antiguedad"].mean() if total else 0

c1.metric("Clientes", f"{total:,}")
c2.metric("Tasa de churn", f"{tasa_churn:.1%}")
c3.metric("Ingreso mensual", f"${ingreso_mensual:,.0f}")
c4.metric("Antigüedad prom.", f"{antiguedad_prom:.0f} meses")

st.divider()

# ----------------------------- Gráficos ------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Churn por tipo de contrato**")
    churn_contrato = (
        df_f.groupby("tipo_contrato")["churn"].mean().reset_index()
    )
    fig = px.bar(
        churn_contrato, x="tipo_contrato", y="churn",
        labels={"churn": "Tasa de churn", "tipo_contrato": "Contrato"},
        color="churn", color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("**Distribución de antigüedad por estado de churn**")
    fig = px.histogram(
        df_f, x="meses_antiguedad", color="churn_label",
        barmode="overlay", nbins=30,
        labels={"meses_antiguedad": "Meses de antigüedad", "churn_label": "Churn"},
        color_discrete_map={"Si": "#EF553B", "No": "#636EFA"},
    )
    st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("**Churn por método de pago**")
    churn_pago = df_f.groupby("metodo_pago")["churn"].mean().reset_index()
    fig = px.bar(
        churn_pago, x="churn", y="metodo_pago", orientation="h",
        labels={"churn": "Tasa de churn", "metodo_pago": ""},
        color="churn", color_continuous_scale="Oranges",
    )
    fig.update_layout(xaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    st.markdown("**Fricciones: factores que más predicen el churn**")
    if importancias is not None:
        top = importancias.head(8).sort_values("importancia")
        fig = px.bar(
            top, x="importancia", y="variable", orientation="h",
            color="importancia", color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Entrena el modelo primero: python src/train_model.py")

st.divider()

# ----------------------------- Comparación de modelos ----------------------
st.subheader("🤖 Comparación de modelos")

if comparacion is not None:
    st.markdown(
        "Tres algoritmos entrenados sobre el mismo split 80/20 (estratificado). "
        "El modelo con mayor AUC-ROC se selecciona automáticamente para el predictor."
    )
    col_tbl, col_chart = st.columns([1, 1.4])

    with col_tbl:
        rename_cols = {
            "modelo": "Modelo",
            "auc_roc": "AUC-ROC ↑",
            "precision_churn": "Precisión (churn)",
            "recall_churn": "Recall (churn)",
            "f1_churn": "F1 (churn)",
        }
        if "avg_precision" in comparacion.columns:
            rename_cols["avg_precision"] = "PR-AUC ↑"
        st.dataframe(
            comparacion.rename(columns=rename_cols),
            use_container_width=True,
            hide_index=True,
        )
        mejor = comparacion.iloc[0]["modelo"]
        st.success(f"Modelo seleccionado: **{mejor}**")
        if cv_scores_df is not None:
            media = cv_scores_df["auc_roc"].mean()
            std = cv_scores_df["auc_roc"].std()
            st.info(f"Validación cruzada (5-fold): AUC-ROC = **{media:.3f} ± {std:.3f}**")

    with col_chart:
        if pr_curve is not None:
            baseline = df["churn"].mean()
            fig = px.line(
                pr_curve, x="recall", y="precision",
                title="Curva Precision-Recall (mejor modelo)",
                labels={"recall": "Recall", "precision": "Precisión"},
            )
            fig.add_hline(
                y=baseline, line_dash="dash", line_color="gray",
                annotation_text=f"Baseline aleatorio ({baseline:.2f})",
            )
            fig.update_layout(yaxis_range=[0, 1.05], xaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)
        else:
            df_melted = comparacion.melt(
                id_vars="modelo",
                value_vars=["auc_roc", "recall_churn", "f1_churn"],
                var_name="métrica", value_name="valor",
            )
            df_melted["métrica"] = df_melted["métrica"].map({
                "auc_roc": "AUC-ROC",
                "recall_churn": "Recall (churn)",
                "f1_churn": "F1 (churn)",
            })
            fig = px.bar(
                df_melted, x="valor", y="modelo", color="métrica",
                barmode="group", orientation="h",
                labels={"valor": "", "modelo": ""},
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
            )
            fig.update_layout(
                xaxis_range=[0, 1], legend_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Entrena los modelos primero: `python src/train_model.py`")

st.divider()

# ----------------------------- Predictor en vivo ---------------------------
st.subheader("🔮 Predictor de churn en tiempo real")
st.markdown("Simula un cliente y predice su probabilidad de abandono.")

if modelo is not None:
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**Contrato y servicio**")
        in_contrato = st.selectbox("Tipo de contrato", df["tipo_contrato"].unique())
        in_internet = st.selectbox("Servicio internet", df["servicio_internet"].unique())
        in_soporte = st.selectbox("Soporte técnico", df["soporte_tecnico"].unique())
        in_telefono = st.selectbox("Servicio telefónico", df["servicio_telefono"].unique())
    with p2:
        st.markdown("**Comportamiento**")
        in_antiguedad = st.slider("Antigüedad (meses)", 1, 72, 12)
        in_cargo = st.slider("Cargo mensual ($)", 18, 120, 70)
        in_tickets = st.slider("Tickets de soporte", 0, 10, 2)
    with p3:
        st.markdown("**Pago y facturación**")
        in_pago = st.selectbox("Método de pago", df["metodo_pago"].unique())
        in_factura = st.selectbox("Factura electrónica", df["factura_electronica"].unique())
    with p4:
        st.markdown("**Perfil personal**")
        in_genero = st.selectbox("Género", df["genero"].unique())
        in_edad = st.slider("Edad", 18, 80, 40)
        in_pareja = st.selectbox("Tiene pareja", df["tiene_pareja"].unique())
        in_dependientes = st.selectbox("Tiene dependientes", df["tiene_dependientes"].unique())

    if st.button("Predecir riesgo de churn", type="primary"):
        cliente = pd.DataFrame([{
            "genero": in_genero,
            "tiene_pareja": in_pareja,
            "tiene_dependientes": in_dependientes,
            "servicio_telefono": in_telefono,
            "servicio_internet": in_internet,
            "soporte_tecnico": in_soporte,
            "tipo_contrato": in_contrato,
            "metodo_pago": in_pago,
            "factura_electronica": in_factura,
            "edad": in_edad,
            "es_senior": int(in_edad >= 65),
            "meses_antiguedad": in_antiguedad,
            "cargo_mensual": in_cargo,
            "cargo_total": in_cargo * in_antiguedad,
            "tickets_soporte": in_tickets,
        }])
        proba = modelo.predict_proba(cliente)[0, 1]
        st.markdown(f"### Probabilidad de churn: **{proba:.1%}**")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#EF553B" if proba > 0.5 else "#00CC96"},
                "steps": [
                    {"range": [0, 33], "color": "#d4f7dc"},
                    {"range": [33, 66], "color": "#fff3cd"},
                    {"range": [66, 100], "color": "#f8d7da"},
                ],
            },
            number={"suffix": "%"},
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        if proba > 0.5:
            st.error("⚠️ Cliente de ALTO riesgo. Se recomienda acción de retención.")
        elif proba > 0.3:
            st.warning("Cliente de riesgo medio. Monitorear.")
        else:
            st.success("✅ Cliente de bajo riesgo.")
else:
    st.info("Entrena el modelo primero ejecutando: python src/train_model.py")

st.divider()
st.caption("Proyecto de portafolio · Analítica de datos + Machine Learning · Python, scikit-learn, Streamlit, Plotly")
