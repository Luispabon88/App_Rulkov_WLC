"""
Utilities for the simplified non-chaotic Rulkov map used in the WLC project.

This file intentionally separates the mathematical model from the Streamlit UI.
The first educational module uses only one uncoupled neuron.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RulkovResult:
    """Container for one-neuron simulation results."""
    n: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_norm: np.ndarray
    burst_square: np.ndarray


def _rulkov_x_next(x_prev: float, x_prev2: float, y_prev: float, alpha: float, coupling_beta: float = 0.0) -> float:
    """Piecewise simplified Rulkov fast variable update."""
    threshold = alpha + y_prev + coupling_beta

    if x_prev <= 0:
        return alpha / (1.0 - x_prev) + y_prev + coupling_beta

    if (0 < x_prev < threshold) and (x_prev2 <= 0):
        return threshold

    return -1.0


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Normalize a signal between 0 and 1, avoiding division by zero."""
    xmin = np.min(x)
    xmax = np.max(x)
    if np.isclose(xmax, xmin):
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def detect_burst_square(
    x: np.ndarray,
    upper_threshold: float = 0.80,
    lower_threshold: float = 0.10,
) -> np.ndarray:
    """
    Convert the normalized membrane potential into a square burst indicator.

    1 means the signal is inside a burst interval; 0 means inactive/silent segment.
    """
    x_norm = normalize_signal(x)
    burst = np.zeros(len(x_norm), dtype=float)
    starts: list[int] = []
    ends: list[int] = []
    in_burst = False

    for i, value in enumerate(x_norm):
        if value > upper_threshold and not in_burst:
            starts.append(i)
            in_burst = True
        elif value < lower_threshold and in_burst:
            ends.append(i)
            in_burst = False

    if in_burst:
        ends.append(len(x_norm) - 1)

    for start, end in zip(starts, ends):
        if end > start:
            burst[start:end] = 1.0

    return burst


def simulate_single_neuron(
    alpha: float = 6.0,
    sigma: float = -0.2,
    mu: float = 0.001,
    n_iter: int = 5000,
    x0: float = 0.0,
    y0: float = 0.0,
    upper_threshold: float = 0.80,
    lower_threshold: float = 0.10,
) -> RulkovResult:
    """
    Simulate one uncoupled simplified Rulkov neuron.

    Equations:
        x[n+1] = alpha/(1 - x[n]) + y[n], if x[n] <= 0
        x[n+1] = alpha + y[n], if 0 < x[n] < alpha + y[n] and x[n-1] <= 0
        x[n+1] = -1, otherwise

        y[n+1] = y[n] - mu*(x[n] + 1) + mu*sigma

    Notes:
        This matches the single-neuron limit of the RulkovMapV4 notebook model,
        with coupling terms set to zero.
    """
    n_iter = int(n_iter)
    if n_iter < 3:
        raise ValueError("n_iter must be at least 3.")

    n = np.arange(n_iter)
    x = np.zeros(n_iter, dtype=float)
    y = np.zeros(n_iter, dtype=float)

    # Initial values. x[0] and x[1] are both initialized to avoid ambiguity in x[n-1].
    x[0] = x0
    y[0] = y0
    x[1] = _rulkov_x_next(x_prev=x[0], x_prev2=x0, y_prev=y[0], alpha=alpha)
    y[1] = y[0] - mu * (x[0] + 1.0) + mu * sigma

    for k in range(2, n_iter):
        x[k] = _rulkov_x_next(x_prev=x[k - 1], x_prev2=x[k - 2], y_prev=y[k - 1], alpha=alpha)
        y[k] = y[k - 1] - mu * (x[k - 1] + 1.0) + mu * sigma

    x_norm = normalize_signal(x)
    burst_square = detect_burst_square(x, upper_threshold, lower_threshold)

    return RulkovResult(n=n, x=x, y=y, x_norm=x_norm, burst_square=burst_square)
