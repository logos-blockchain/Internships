"""
Stake model for the leader election: relative stakes alpha_i are sampled once from a
Pareto(k, x_m) and normalised to sum to 1 (quenched). The scale x_m cancels in the
ratio; the shape k controls inequality (small k -> a few large stakeholders dominate).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma
from scipy.stats import pareto

DEFAULT_SHAPE = 4.0 / 3.0  # = shape_from_gini(0.60)


def sample_relative_stakes(N, shape=DEFAULT_SHAPE, scale=1.0, rng=None):
    """
    Sample N relative stakes alpha_i = w_i / sum_j w_j with w_i ~ Pareto(shape, scale).

    Returns a float array of length N summing to 1.
    """
    rng = np.random.default_rng(rng)
    w = pareto.rvs(b=shape, scale=scale, size=N, random_state=rng)
    return w / w.sum()


def gini(weights):
    """
    Gini coefficient of a weight vector (0 = perfectly equal, 1 = concentrated).

    G = (2 * sum_i i * x_(i)) / (n * sum_i x_i) - (n + 1) / n with x sorted ascending
    and i 1-indexed.
    """
    x = np.sort(np.asarray(weights, dtype=float))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * total) - (n + 1.0) / n)


def shape_from_gini(G):
    """
    Invert the Pareto Gini relation G = 1 / (2k - 1) to k = (1 + G) / (2G).

    Used to pick the shape k matching a published stake Gini of a real chain.
    """
    return (1.0 + G) / (2.0 * G)


def gini_from_shape(shape):
    """
    Population Gini of a Pareto(shape k): G = 1 / (2k - 1), valid for k > 1.

    The inverse of shape_from_gini; independent of the scale x_m.
    """
    return 1.0 / (2.0 * np.asarray(shape, dtype=float) - 1.0)


def expected_max_relative_stake(shape, N):
    """
    Large-N extreme-value approximation of the expected maximum relative stake E[M_N].

    E[M_N] ~= (k - 1)/k * N^(1/k - 1) * Gamma(1 - 1/k), from the Frechet law of the
    top raw stake over the LLN total. Valid for k > 1 (nan otherwise); overestimates
    at finite N when k is near 1, so use simulate_max_relative_stake for numbers.
    """
    k = np.asarray(shape, dtype=float)
    with np.errstate(invalid="ignore"):
        out = (k - 1.0) / k * N ** (1.0 / k - 1.0) * gamma(1.0 - 1.0 / k)
    return np.where(k > 1.0, out, np.nan)


def simulate_max_relative_stake(N, shape=DEFAULT_SHAPE, reps=50_000, scale=1.0,
                                rng=None, chunk_size=10_000):
    """
    Monte-Carlo estimate of the maximum relative stake M_N = max_i S_i over `reps` realisations.

    Returns a dict with mean (E[M_N]), sd (between-realisation spread, the quenched-disorder
    error bar), se (sd / sqrt(reps)), ci95, median and reps.
    """
    rng = np.random.default_rng(rng)
    total = 0.0
    total_sq = 0.0
    vals = np.empty(reps)
    done = 0
    while done < reps:
        c = min(chunk_size, reps - done)
        w = pareto.rvs(b=shape, scale=scale, size=(c, N), random_state=rng)
        m = w.max(axis=1) / w.sum(axis=1)
        total += float(m.sum())
        total_sq += float((m * m).sum())
        vals[done:done + c] = m
        done += c

    mean = total / reps
    var = (total_sq - reps * mean * mean) / (reps - 1)
    sd = float(np.sqrt(max(var, 0.0)))
    se = sd / np.sqrt(reps)
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95": (mean - 1.96 * se, mean + 1.96 * se),
        "median": float(np.median(vals)),
        "reps": reps,
    }
