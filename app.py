"""MotoTaller IA: análisis de opiniones de clientes con una red recurrente.

Uso:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src import config
from src.predict import load_info, load_model, predict_texts

st.set_page_config(page_title="MotoTaller IA - Opiniones", page_icon="🏍️")
st.title("🏍️ MotoTaller IA: opiniones de clientes")
st.caption("Una red recurrente (GRU/LSTM) clasifica una opinión como positiva o negativa.")


@st.cache_resource
def get_model():
    return load_model()


if not config.MODEL_PATH.exists():
    st.error("No se encontró el modelo entrenado. Ejecuta `python -m src.train` "
             "o copia `models/opiniones_rnn.keras` desde Colab.")
    st.stop()

get_model()
info = load_info()

EJEMPLOS = {
    "Positiva": "Excelente servicio, me arreglaron la moto rápido y quedó funcionando perfecto.",
    "Negativa": "Pésimo, el cambio de aceite lo hicieron mal y todavía gotea.",
}
if "texto" not in st.session_state:
    st.session_state.texto = ""

c1, c2 = st.columns(2)
if c1.button("Ejemplo positivo"):
    st.session_state.texto = EJEMPLOS["Positiva"]
if c2.button("Ejemplo negativo"):
    st.session_state.texto = EJEMPLOS["Negativa"]

texto = st.text_area("Escribe la opinión del cliente (en español):", key="texto", height=140)

if st.button("Analizar", type="primary") and texto.strip():
    label, conf, p_pos = predict_texts([texto])[0]
    (st.success if label == "Positivo" else st.error)(f"Sentimiento: {label}")
    st.metric("Confianza", f"{conf:.1%}")
    if conf < 0.65:
        st.warning("Confianza baja: la opinión es ambigua o tiene vocabulario poco habitual.")
    if label == "Negativo":
        st.info("🔧 Sugerencia: contactar al cliente para entender el problema y ofrecer una solución.")
    st.bar_chart(pd.Series({"Negativo": 1 - p_pos, "Positivo": p_pos}, name="Probabilidad"))
elif texto.strip() == "":
    st.caption("Escribe una opinión o usa uno de los ejemplos.")

if info:
    st.sidebar.header("Modelo")
    st.sidebar.write(f"Arquitectura: **{info['tipo'].upper()} bidireccional**")
    st.sidebar.write(f"Accuracy en test: **{info['test_accuracy']:.1%}**")
    if "f1_macro" in info:
        st.sidebar.write(f"F1 macro: **{info['f1_macro']:.2f}** | umbral: **{info['umbral']:.2f}**")
    st.sidebar.caption("Entrenado con opiniones en español de otro dominio (ver README). "
                       "Es una ayuda de apoyo, no una evaluación definitiva.")
