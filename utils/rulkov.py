"""
Utilities for the simplified non-chaotic Rulkov map used in the WLC project.

This file separates the mathematical model from the Streamlit UI.
The first educational module uses one uncoupled neuron and includes a
robust pulse/burst detector intended to be reused before the WLC module.
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


def _rulkov_x_next(
    x_prev: float,
    x_prev2: float,
    y_prev: float,
    alpha: float,
    coupling_beta: float = 0.0,
) -> float:
    """Piecewise simplified Rulkov fast variable update."""
    threshold = alpha + y_prev + coupling_beta

    if x_prev <= 0:
        return alpha / (1.0 - x_prev) + y_prev + coupling_beta

    if (0 < x_prev < threshold) and (x_prev2 <= 0):
        return threshold

    return -1.0


def normalize_signal(
    x: np.ndarray,
    method: str = "minmax",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Normalize a signal between 0 and 1.

    Parameters
    ----------
    x:
        Input signal.
    method:
        "minmax" uses the absolute minimum and maximum.
        "percentile" uses percentile limits, which reduces the effect of
        isolated transient peaks.
    p_low, p_high:
        Percentiles used when method="percentile".
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        return x.copy()

    if method == "percentile":
        xmin = np.percentile(x, p_low)
        xmax = np.percentile(x, p_high)
    else:
        xmin = np.min(x)
        xmax = np.max(x)

    if np.isclose(xmax, xmin):
        return np.zeros_like(x, dtype=float)

    x_norm = (x - xmin) / (xmax - xmin)
    return np.clip(x_norm, 0.0, 1.0)


def detect_burst_square_from_normalized(
    x_norm: np.ndarray,
    upper_threshold: float = 0.80,
    lower_threshold: float = 0.10,
    min_width: int = 1,
) -> np.ndarray:
    """
    Detect active pulse/burst intervals from an already-normalized signal.

    A pulse starts when x_norm crosses the upper threshold and remains active
    until x_norm falls below the lower threshold. This hysteresis avoids rapid
    on/off switching around a single threshold.
    """
    if lower_threshold >= upper_threshold:
        raise ValueError("lower_threshold must be smaller than upper_threshold.")

    x_norm = np.asarray(x_norm, dtype=float)
    burst = np.zeros(len(x_norm), dtype=float)
    in_burst = False
    start = 0

    for i, value in enumerate(x_norm):
        if (not in_burst) and (value >= upper_threshold):
            in_burst = True
            start = i
        elif in_burst and (value <= lower_threshold):
            end = i
            if end - start >= min_width:
                burst[start:end + 1] = 1.0
            in_burst = False

    if in_burst:
        end = len(x_norm) - 1
        if end - start >= min_width:
            burst[start:end + 1] = 1.0

    return burst


def detect_burst_square(
    x: np.ndarray,
    upper_threshold: float = 0.80,
    lower_threshold: float = 0.10,
    normalize_method: str = "minmax",
    min_width: int = 1,
) -> np.ndarray:
    """Normalize x and return a square pulse/burst indicator."""
    x_norm = normalize_signal(x, method=normalize_method)
    return detect_burst_square_from_normalized(
        x_norm=x_norm,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        min_width=min_width,
    )


def count_pulses(square_signal: np.ndarray) -> int:
    """Count rising edges in a square pulse signal."""
    s = np.asarray(square_signal, dtype=float)
    if s.size == 0:
        return 0
    previous = np.r_[0.0, s[:-1]]
    return int(np.sum((previous <= 0.0) & (s > 0.0)))


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
    """
    n_iter = int(n_iter)
    if n_iter < 3:
        raise ValueError("n_iter must be at least 3.")

    n = np.arange(n_iter)
    x = np.zeros(n_iter, dtype=float)
    y = np.zeros(n_iter, dtype=float)

    x[0] = x0
    y[0] = y0
    x[1] = _rulkov_x_next(x_prev=x[0], x_prev2=x0, y_prev=y[0], alpha=alpha)
    y[1] = y[0] - mu * (x[0] + 1.0) + mu * sigma

    for k in range(2, n_iter):
        x[k] = _rulkov_x_next(
            x_prev=x[k - 1],
            x_prev2=x[k - 2],
            y_prev=y[k - 1],
            alpha=alpha,
        )
        y[k] = y[k - 1] - mu * (x[k - 1] + 1.0) + mu * sigma

    x_norm = normalize_signal(x, method="minmax")
    burst_square = detect_burst_square_from_normalized(
        x_norm,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )

    return RulkovResult(n=n, x=x, y=y, x_norm=x_norm, burst_square=burst_square)
