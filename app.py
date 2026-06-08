from __future__ import annotations

import streamlit as st

from modules.neuron_explorer import render_neuron_explorer


st.set_page_config(
    page_title="Rulkov Interactive Lab",
    page_icon="🧠",
    layout="wide",
)

st.title("🐢Rulkov Interactive Lab")
st.caption("Simulador educativo para Modelo de Rulkov")

page = st.sidebar.radio(
    "Selecciona un módulo",
    [
        "Módulo 1 · Neurona individual",
        "Módulo 2 · Regímenes dinámicos (próximamente)",
        "Módulo 3 · Red de 3 neuronas (próximamente)",
        "Módulo 4 · WLC y secuencias (próximamente)",
        "Módulo 5 · Codificación simbólica (próximamente)",
    ],
)

if page == "Módulo 1 · Neurona individual":
    render_neuron_explorer()
else:
    st.info("Este módulo será añadido después. Primero se validará el modelo de una neurona.")
