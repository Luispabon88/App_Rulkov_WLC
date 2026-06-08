from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st

from utils.rulkov import (
    count_pulses,
    detect_burst_square_from_normalized,
    normalize_signal,
    simulate_single_neuron,
)


AIP_DOUBLE_COLUMN = (6.69, 2.35)
AIP_PHASE = (3.35, 3.35)


COLOR_OPTIONS = {
    "Azul": "tab:blue",
    "Rojo": "tab:red",
    "Verde": "tab:green",
    "Morado": "tab:purple",
    "Gris": "tab:gray",
    "Negro": "black",
}


def _fig_to_png(fig) -> bytes:
    """Return a matplotlib figure as PNG bytes."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    buffer.seek(0)
    return buffer.getvalue()


def _show_plot_with_download(fig, filename: str, key: str) -> None:
    """Show a compact figure and add a download button below it."""
    st.pyplot(fig, clear_figure=False, use_container_width=False)
    st.download_button(
        label="Descargar PNG",
        data=_fig_to_png(fig),
        file_name=filename,
        mime="image/png",
        key=key,
    )
    plt.close(fig)


def _configure_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def render_neuron_explorer():
    st.header("Module 1: Single Rulkov Neuron Explorer")

    st.markdown(
        """
        Este módulo permite explorar la dinámica de una neurona aislada no caótica
        de Rulkov. La detección de pulsos se aplica sobre la ventana temporal
        seleccionada, lo cual evita que el transitorio inicial distorsione la
        normalización y prepara la señal para el futuro módulo WLC.
        """
    )

    st.subheader("Parámetros del modelo")

    col1, col2, col3 = st.columns(3)

    with col1:
        alpha = st.slider("Alpha (α)", 2.0, 8.0, 6.0, 0.01)

    with col2:
        sigma = st.slider("Sigma (σ)", -1.0, 1.0, -0.2, 0.01)

    with col3:
        mu = st.slider("Mu (μ)", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")

    n_iter = st.slider("Número de iteraciones (n)", 500, 20000, 5000, 500)

    result = simulate_single_neuron(alpha=alpha, sigma=sigma, mu=mu, n_iter=n_iter)

    t = result.n
    x = result.x
    y = result.y

    st.subheader("Ventana temporal / zoom")

    default_start = min(1000, len(t) - 2)
    n_start, n_end = st.slider(
        "Seleccione el rango de n usado para graficar y detectar pulsos",
        min_value=0,
        max_value=len(t) - 1,
        value=(default_start, len(t) - 1),
        step=1,
    )

    t_zoom = t[n_start:n_end + 1]
    x_zoom = x[n_start:n_end + 1]
    y_zoom = y[n_start:n_end + 1]

    st.info(
        f"Mostrando desde n = {n_start} hasta n = {n_end}. "
        "La normalización, el detector de pulsos y el retrato de fase usan esta misma ventana."
    )

    st.subheader("Estilo de las figuras")
    col_style1, col_style2, col_style3 = st.columns(3)

    with col_style1:
        x_color_name = st.selectbox("Color de x(n)", list(COLOR_OPTIONS), index=0)
    with col_style2:
        y_color_name = st.selectbox("Color de y(n)", list(COLOR_OPTIONS), index=1)
    with col_style3:
        pulse_color_name = st.selectbox("Color del pulso detectado", list(COLOR_OPTIONS), index=2)

    x_color = COLOR_OPTIONS[x_color_name]
    y_color = COLOR_OPTIONS[y_color_name]
    pulse_color = COLOR_OPTIONS[pulse_color_name]

    st.subheader("Dinámica temporal")

    fig_x, ax_x = plt.subplots(figsize=AIP_DOUBLE_COLUMN)
    ax_x.plot(t_zoom, x_zoom, linewidth=0.85, color=x_color)
    _configure_axes(ax_x, "n", "x(n)", "Potencial de membrana x(n)")
    _show_plot_with_download(fig_x, "rulkov_xn.png", "download_xn")

    fig_y, ax_y = plt.subplots(figsize=AIP_DOUBLE_COLUMN)
    ax_y.plot(t_zoom, y_zoom, linewidth=0.85, color=y_color)
    _configure_axes(ax_y, "n", "y(n)", "Variable lenta y(n)")
    _show_plot_with_download(fig_y, "rulkov_yn.png", "download_yn")

    st.subheader("Diagrama de fase")

    fig_phase, ax_phase = plt.subplots(figsize=AIP_PHASE)
    ax_phase.plot(x_zoom, y_zoom, linewidth=0.65, color=x_color)
    _configure_axes(ax_phase, "x(n)", "y(n)", "Diagrama de fase: x(n) vs y(n)")
    _show_plot_with_download(fig_phase, "rulkov_phase_portrait.png", "download_phase")

    st.subheader("Señal normalizada y detección de pulsos")

    st.markdown(
        """
        En esta sección, la señal se normaliza **dentro de la ventana temporal seleccionada**.
        Esto corrige el problema donde el transitorio inicial dominaba la normalización
        y hacía que los umbrales no detectaran correctamente los pulsos posteriores.
        """
    )

    col_thr1, col_thr2, col_thr3 = st.columns(3)

    with col_thr1:
        norm_method_label = st.selectbox(
            "Método de normalización",
            ["Min-Max en ventana", "Percentil 1-99 en ventana"],
            index=0,
        )

    with col_thr2:
        upper = st.slider("Upper threshold", 0.50, 0.95, 0.80, 0.01)

    with col_thr3:
        lower_max = max(0.01, upper - 0.05)
        lower_default = min(0.10, lower_max)
        lower = st.slider("Lower threshold", 0.01, lower_max, lower_default, 0.01)

    normalize_method = "percentile" if norm_method_label.startswith("Percentil") else "minmax"
    x_norm_zoom = normalize_signal(x_zoom, method=normalize_method)
    pulse_zoom = detect_burst_square_from_normalized(
        x_norm=x_norm_zoom,
        upper_threshold=upper,
        lower_threshold=lower,
    )
    pulse_count = count_pulses(pulse_zoom)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Pulsos detectados", pulse_count)
    metric_col2.metric("Upper threshold", f"{upper:.2f}")
    metric_col3.metric("Lower threshold", f"{lower:.2f}")

    fig_pulse, ax_pulse = plt.subplots(figsize=AIP_DOUBLE_COLUMN)
    ax_pulse.plot(t_zoom, x_norm_zoom, linewidth=0.85, color=x_color, label="x(n) normalizada")
    ax_pulse.step(
        t_zoom,
        pulse_zoom,
        where="post",
        linewidth=0.95,
        color=pulse_color,
        label="pulso detectado",
    )
    ax_pulse.axhline(upper, linestyle="--", linewidth=0.8, color="black", label="upper")
    ax_pulse.axhline(lower, linestyle=":", linewidth=0.9, color="black", label="lower")
    _configure_axes(ax_pulse, "n", "valor normalizado", "Señal normalizada y detector de pulsos")
    ax_pulse.set_ylim(-0.05, 1.05)
    ax_pulse.legend(loc="upper right", fontsize=8)
    _show_plot_with_download(fig_pulse, "rulkov_pulse_detector.png", "download_pulse")

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
            4. Use la ventana temporal para retirar el transitorio inicial y ajuste los umbrales hasta obtener una señal cuadrada estable.
            """
        )

    with st.expander("Actividades para interpretación"):
        st.markdown(
            """
            - Varíe **α** y observe sus efectos sobre la amplitud y forma de x(n).  
            - Varíe **σ** y observe sus efectos sobre la frecuencia de activación.  
            - Varíe **μ** y observe la evolución de la variable lenta.  
            - Explique por qué el detector mejora cuando la normalización se aplica solo a la ventana temporal de análisis.
            """
        )
