import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURACION GENERAL
# ==========================================================
st.set_page_config(
    page_title="Rulkov Interactive Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("Rulkov Interactive Lab")
st.caption("Simulador educativo de una neurona discreta de Rulkov")

# ==========================================================
# MODELO DE RULKOV: UNA NEURONA
# ==========================================================
def rulkov_map(x, y, alpha, sigma, mu):
    """
    Modelo discreto tipo Rulkov.

    x: variable rapida
    y: variable lenta
    alpha: parametro de excitabilidad
    sigma: parametro de control lento
    mu: escala lenta
    """
    x_next = alpha / (1 + x**2) + y
    y_next = y - mu * (x - sigma)
    return x_next, y_next


def simulate_single_neuron(alpha, sigma, mu, x0, y0, n_iter):
    x = np.zeros(n_iter)
    y = np.zeros(n_iter)
    x[0] = x0
    y[0] = y0

    for n in range(n_iter - 1):
        x[n + 1], y[n + 1] = rulkov_map(
            x[n], y[n], alpha, sigma, mu
        )

    return pd.DataFrame({
        "n": np.arange(n_iter),
        "x": x,
        "y": y
    })

# ==========================================================
# BARRA LATERAL
# ==========================================================
st.sidebar.header("Parámetros de simulación")

alpha = st.sidebar.slider("α - excitabilidad", 1.0, 8.0, 4.5, 0.01)
sigma = st.sidebar.slider("σ - control lento", -1.0, 1.0, -0.2, 0.01)
mu = st.sidebar.number_input("μ - escala lenta", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")

st.sidebar.divider()
st.sidebar.subheader("Condiciones iniciales")
x0 = st.sidebar.number_input("x₀", value=-1.0, step=0.1)
y0 = st.sidebar.number_input("y₀", value=-2.5, step=0.1)

st.sidebar.divider()
n_iter = st.sidebar.slider("Número de iteraciones", 500, 20000, 5000, 500)
transient = st.sidebar.slider("Transitorio a descartar", 0, n_iter - 100, 1000, 100)

# ==========================================================
# SIMULACION
# ==========================================================
df = simulate_single_neuron(alpha, sigma, mu, x0, y0, n_iter)
df_plot = df[df["n"] >= transient]

# ==========================================================
# INTERFAZ PRINCIPAL
# ==========================================================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Serie temporal de la variable rápida x")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df_plot["n"], df_plot["x"], linewidth=0.9)
    ax.set_xlabel("Iteración n")
    ax.set_ylabel("x")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("Plano de fase")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.plot(df_plot["x"], df_plot["y"], linewidth=0.7)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

st.divider()

col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Vista de datos")
    st.dataframe(df_plot.tail(100), use_container_width=True)

with col4:
    st.subheader("Guía de interpretación")
    st.markdown(
        """
        En este primer módulo el estudiante explora cómo los parámetros modifican
        la actividad de una neurona discreta:

        - **x** representa la variable rápida asociada a la activación.
        - **y** representa la variable lenta de recuperación.
        - **α** controla la excitabilidad del sistema.
        - **σ** desplaza la dinámica lenta.
        - **μ** determina qué tan lenta es la evolución de y.

        Actividad sugerida: cambiar α y σ para identificar reposo, oscilaciones,
        spikes o bursting.
        """
    )

st.divider()
st.info("Próximo módulo sugerido: red de tres neuronas acopladas y detección de secuencias WLC.")
