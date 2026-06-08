from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt

from utils.rulkov import simulate_single_neuron


def render_neuron_explorer() -> None:
    st.header("Módulo 1 · Explorador interactivo de una neurona de Rulkov")

    st.markdown(
        """
        Este módulo permite explorar la dinámica de una neurona discreta tipo Rulkov no caótica. 
        La idea educativa es que el estudiante observe cómo los parámetros modifican la actividad temporal,
        el retrato de fase y la detección simple de ráfagas.
        """
    )

    with st.sidebar:
        st.subheader("Parámetros del modelo")
        alpha = st.slider("α · parámetro no lineal", min_value=2.0, max_value=8.0, value=6.0, step=0.01)
        sigma = st.slider("σ · corriente lenta externa", min_value=-1.0, max_value=1.0, value=-0.2, step=0.01)
        mu = st.number_input("μ · escala lenta", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
        n_iter = st.slider("Iteraciones", min_value=500, max_value=20000, value=5000, step=500)

        st.subheader("Detección de burst")
        upper = st.slider("Umbral superior", min_value=0.50, max_value=0.95, value=0.80, step=0.01)
        lower = st.slider("Umbral inferior", min_value=0.01, max_value=0.40, value=0.10, step=0.01)

    result = simulate_single_neuron(
        alpha=alpha,
        sigma=sigma,
        mu=mu,
        n_iter=n_iter,
        upper_threshold=upper,
        lower_threshold=lower,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("α", f"{alpha:.2f}")
    col2.metric("σ", f"{sigma:.2f}")
    col3.metric("μ", f"{mu:.4f}")

    st.subheader("1. Potencial rápido x(n)")
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(result.n, result.x, linewidth=0.8)
    ax1.set_xlabel("n")
    ax1.set_ylabel("x(n)")
    ax1.grid(True, alpha=0.25)
    st.pyplot(fig1, clear_figure=True)

    st.subheader("2. Variable lenta y(n)")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(result.n, result.y, linewidth=0.8)
    ax2.set_xlabel("n")
    ax2.set_ylabel("y(n)")
    ax2.grid(True, alpha=0.25)
    st.pyplot(fig2, clear_figure=True)

    st.subheader("3. Retrato de fase x(n) vs y(n)")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(result.x, result.y, linewidth=0.6)
    ax3.set_xlabel("x(n)")
    ax3.set_ylabel("y(n)")
    ax3.grid(True, alpha=0.25)
    st.pyplot(fig3, clear_figure=True)

    st.subheader("4. Señal normalizada y detección de burst")
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.plot(result.n, result.x_norm, linewidth=0.8, label="x normalizada")
    ax4.plot(result.n, result.burst_square, linewidth=0.8, label="burst detectado")
    ax4.set_xlabel("n")
    ax4.set_ylabel("valor normalizado")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="upper right")
    st.pyplot(fig4, clear_figure=True)

    with st.expander("Ecuaciones usadas en este módulo"):
        st.latex(r"""
        x_{n+1}=
        \begin{cases}
        \dfrac{\alpha}{1-x_n}+y_n, & x_n\le 0,\\
        \alpha+y_n, & 0<x_n<\alpha+y_n \; \text{and}\; x_{n-1}\le0,\\
        -1, & \text{otherwise.}
        \end{cases}
        """)
        st.latex(r"""
        y_{n+1}=y_n-\mu(x_n+1)+\mu\sigma
        """)

    with st.expander("Guía para estudiantes"):
        st.markdown(
            """
            **Actividad sugerida:**

            1. Fije μ = 0.001 y σ = -0.2. Varíe α entre 2 y 8. Describa qué cambia en x(n).  
            2. Fije α = 6.0 y μ = 0.001. Varíe σ entre -1 y 1. Observe si cambia la frecuencia de activación.  
            3. Compare la serie temporal con el retrato de fase. Explique qué información aporta cada gráfico.  
            4. Modifique los umbrales de burst y observe cómo cambia la señal cuadrada detectada.
            """
        )
