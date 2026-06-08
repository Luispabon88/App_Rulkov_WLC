import streamlit as st
import matplotlib.pyplot as plt

from utils.rulkov import simulate_single_neuron


def render_neuron_explorer():
    st.header("Module 1: Single Rulkov Neuron Explorer")

    st.markdown(
        """
        Este módulo permite a los estudiates explorar la dinámica de una
        neurona aislada no caótica de Rulkov mediante la variación de sus parámetros
        """
    )

    # -------------------------------------------------
    # Sidebar / parameter controls
    # -------------------------------------------------
    st.subheader("Parámetros del Modelo")

    col1, col2, col3 = st.columns(3)

    with col1:
        alpha = st.slider(
            "Alpha (α)",
            min_value=2.0,
            max_value=8.0,
            value=6.0,
            step=0.01
        )

    with col2:
        sigma = st.slider(
            "Sigma (σ)",
            min_value=-1.0,
            max_value=1.0,
            value=-0.2,
            step=0.01
        )

    with col3:
        mu = st.slider(
            "Mu (μ)",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )

    n_iter = st.slider(
        "Número de iteraciones (n)",
        min_value=500,
        max_value=20000,
        value=5000,
        step=500
    )
    st.subheader("Detección de burst")
        upper = st.slider("Umbral superior", min_value=0.50, max_value=0.95, value=0.80, step=0.01)
        lower = st.slider("Umbral inferior", min_value=0.01, max_value=0.40, value=0.10, step=0.01)

    # -------------------------------------------------
    # Simulation
    # -------------------------------------------------
    result = simulate_single_neuron(
    alpha=alpha,
    sigma=sigma,
    mu=mu,
    n_iter=n_iter
    )

    t = result.n
    x = result.x
    y = result.y
    x_norm = result.x_norm
    burst_square = result.burst_square

    # -------------------------------------------------
    # Temporal zoom
    # -------------------------------------------------
    st.subheader("Ventana Temporal / Zoom")

    n_start, n_end = st.slider(
        "Seleccione el rango de n",
        min_value=0,
        max_value=len(t) - 1,
        value=(0, len(t) - 1),
        step=1
    )

    t_zoom = t[n_start:n_end + 1]
    x_zoom = x[n_start:n_end + 1]
    y_zoom = y[n_start:n_end + 1]

    st.info(
        f"Mostrar desde n = {n_start} hasta n = {n_end}. "
        "Aplicado también a la imagen de fase."
    )

    # -------------------------------------------------
    # Time series plots
    # -------------------------------------------------
    st.subheader("Dinámica Temporal")

    fig_x, ax_x = plt.subplots(figsize=(8, 3))
    ax_x.plot(t_zoom, x_zoom)
    ax_x.set_xlabel("n")
    ax_x.set_ylabel("x(n)")
    ax_x.set_title("Potencial de membrana x(n)")
    ax_x.grid(True, alpha=0.3)
    st.pyplot(fig_x)

    fig_y, ax_y = plt.subplots(figsize=(8, 3))
    ax_y.plot(t_zoom, y_zoom)
    ax_y.set_xlabel("n")
    ax_y.set_ylabel("y(n)")
    ax_y.set_title("Variable lenta y(n)")
    ax_y.grid(True, alpha=0.3)
    st.pyplot(fig_y)

    # -------------------------------------------------
    # Phase portrait
    # -------------------------------------------------
    st.subheader("Diagrama de fase")

    fig_phase, ax_phase = plt.subplots(figsize=(5, 5))
    ax_phase.plot(x_zoom, y_zoom, linewidth=0.8)
    ax_phase.set_xlabel("x(n)")
    ax_phase.set_ylabel("y(n)")
    ax_phase.set_title("Diagrama de fase: x(n) vs y(n)")
    ax_phase.grid(True, alpha=0.3)
    st.pyplot(fig_phase)

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
    
    with st.expander("Actividades para interpretación"):
        st.markdown(
            """
            **Actividad sugerida:**
         - Varíe **α**, para observar sus acciones sobre las señales.
         - Varíe **σ** para observar sus acciones sobre las señales.
         - Varíe **μ** para observar sus acciones sobre las señales.
         - ¿En qué ayuda a afinar las ventanas temporales de n al análisis de las señales?.
            """
        )

   
