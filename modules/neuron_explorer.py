from __future__ import annotations
import streamlit as st
import matplotlib.pyplot as plt

from utils.rulkov import simulate_single_neuron


def run_neuron_explorer():
    st.header("Module 1: Single Rulkov Neuron Explorer")

    st.markdown(
        """
        This module allows students to explore the dynamics of a single
        non-chaotic Rulkov neuron by changing its main parameters.
        """
    )

    # -------------------------------------------------
    # Sidebar / parameter controls
    # -------------------------------------------------
    st.subheader("Model parameters")

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
        "Number of iterations",
        min_value=500,
        max_value=20000,
        value=5000,
        step=500
    )

    # -------------------------------------------------
    # Simulation
    # -------------------------------------------------
    t, x, y = simulate_single_neuron(
        alpha=alpha,
        sigma=sigma,
        mu=mu,
        n_iter=n_iter
    )

    # -------------------------------------------------
    # Temporal zoom
    # -------------------------------------------------
    st.subheader("Temporal window / Zoom")

    n_start, n_end = st.slider(
        "Select temporal range n",
        min_value=0,
        max_value=len(t) - 1,
        value=(0, len(t) - 1),
        step=1
    )

    t_zoom = t[n_start:n_end + 1]
    x_zoom = x[n_start:n_end + 1]
    y_zoom = y[n_start:n_end + 1]

    st.info(
        f"Showing data from n = {n_start} to n = {n_end}. "
        "This filter is also applied to the phase portrait."
    )

    # -------------------------------------------------
    # Time series plots
    # -------------------------------------------------
    st.subheader("Temporal dynamics")

    fig_x, ax_x = plt.subplots(figsize=(8, 3))
    ax_x.plot(t_zoom, x_zoom)
    ax_x.set_xlabel("n")
    ax_x.set_ylabel("x(n)")
    ax_x.set_title("Fast variable x(n)")
    ax_x.grid(True, alpha=0.3)
    st.pyplot(fig_x)

    fig_y, ax_y = plt.subplots(figsize=(8, 3))
    ax_y.plot(t_zoom, y_zoom)
    ax_y.set_xlabel("n")
    ax_y.set_ylabel("y(n)")
    ax_y.set_title("Slow variable y(n)")
    ax_y.grid(True, alpha=0.3)
    st.pyplot(fig_y)

    # -------------------------------------------------
    # Phase portrait
    # -------------------------------------------------
    st.subheader("Phase portrait")

    fig_phase, ax_phase = plt.subplots(figsize=(5, 5))
    ax_phase.plot(x_zoom, y_zoom, linewidth=0.8)
    ax_phase.set_xlabel("x(n)")
    ax_phase.set_ylabel("y(n)")
    ax_phase.set_title("Phase portrait: x(n) vs y(n)")
    ax_phase.grid(True, alpha=0.3)
    st.pyplot(fig_phase)

    # -------------------------------------------------
    # Educational interpretation
    # -------------------------------------------------
    st.subheader("Educational interpretation")

    st.markdown(
        """
        - Increasing **α** mainly modifies the fast activation amplitude.
        - Changing **σ** affects the slow recovery variable and may modify the
          activation frequency.
        - Reducing **μ** makes the slow variable evolve more slowly.
        - The temporal window allows students to remove transient behavior or
          focus on specific bursting/spiking regions.
        """
    )
