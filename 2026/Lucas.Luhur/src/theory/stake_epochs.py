"""
Multi-epoch extension of stake_top1: P(the whale is ever named within E epochs).

The stake vector alpha is quenched and epochs are independent replicas, so per-epoch hits
are iid Bernoulli(p(alpha)) given alpha and C_quenched(E) = E_alpha[1 - (1 - p(alpha))^E].
The naive C_naive(E) = 1 - (1 - E_alpha[p])^E is a Jensen upper bound (equal only at E = 1).
The empirical leg for E <= M uses per-chain hit counts: a chain with h hits out of M misses
E specific epochs with probability C(M-h, E)/C(M, E), an exactly unbiased estimator.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

try:
    from consensus.election import DEFAULT_F, DEFAULT_T
    from consensus.stake import DEFAULT_SHAPE, sample_relative_stakes
    from theory.stake_top1 import top1_probability, top1_probability_normal
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from consensus.election import DEFAULT_F, DEFAULT_T
    from consensus.stake import DEFAULT_SHAPE, sample_relative_stakes
    from theory.stake_top1 import top1_probability, top1_probability_normal

SECONDS_PER_SLOT = 1.0


def epoch_days(T=DEFAULT_T):
    """One epoch in days: T slots x 1 s/slot (4.5 days at the full epoch)."""
    return T * SECONDS_PER_SLOT / 86_400.0


def epochs_to_years(E, T=DEFAULT_T):
    """Epoch count -> years of constant observation (365.25-day year)."""
    return np.asarray(E, dtype=float) * epoch_days(T) / 365.25


def cumulative_top1(p_draws, epochs):
    """
    Return C_quenched(E) = E_alpha[1 - (1 - p)^E] over `epochs` from per-chain probabilities.

    Uses expm1/log1p so small p compounded over many epochs keeps full precision.
    """
    p = np.asarray(p_draws, dtype=float)
    E = np.asarray(epochs, dtype=float)
    lq = np.log1p(-np.clip(p, 0.0, 1.0 - 1e-15))
    return -np.expm1(E[..., None] * lq).mean(axis=-1)


def naive_cumulative(p_mean, epochs):
    """Return C_naive(E) = 1 - (1 - p_bar)^E, iid at the mean rate (the Jensen upper bound)."""
    E = np.asarray(epochs, dtype=float)
    return -np.expm1(E * np.log1p(-float(p_mean)))


def subsampled_cumulative(hits, m_epochs, epochs):
    """
    Estimate E[1 - (1-p)^E] from per-chain hit counts for E <= m_epochs (the empirical leg).

    `hits` is the whale-named count per chain out of `m_epochs` epochs (stake_top1_hit *
    cover_runs). P(E specific epochs all miss) = C(m-h, E)/C(m, E) by exchangeability, so the
    estimator is exactly unbiased. A fractional count marks a tie epoch (credit 1/w); it is
    handled by averaging the two adjacent integer counts with weights (1 - frac, frac).
    Returns the mean over chains at each E; raises on E > m_epochs.
    """
    h = np.asarray(hits, dtype=float)
    m = int(m_epochs)
    if np.any(h < -1e-9) or np.any(h > m + 1e-9):
        raise ValueError(f"hits must lie in [0, {m}]")
    E = np.asarray(epochs)
    if np.any(E < 1) or np.any(E > m):
        raise ValueError(f"subsampling is defined for 1 <= E <= {m} observed epochs only")

    def _no_hit(h_int):
        misses = m - h_int
        lo = np.where(E[:, None] <= misses[None, :],
                      gammaln(misses + 1.0)[None, :]
                      - gammaln(np.maximum(misses[None, :] - E[:, None], 0) + 1.0)
                      - (gammaln(m + 1.0) - gammaln(m - E[:, None] + 1.0)),
                      -np.inf)
        return np.exp(lo)

    h_floor = np.floor(h + 1e-9).astype(int)
    frac = np.clip(h - h_floor, 0.0, 1.0)
    no_hit = (1.0 - frac) * _no_hit(h_floor) + frac * _no_hit(np.minimum(h_floor + 1, m))
    return (1.0 - no_hit).mean(axis=1)


def epochs_to_level(epochs, curve, level):
    """
    Return the first E at which the cumulative curve crosses `level` (inf if never).

    Interpolation is linear in (log E, C); the curves are strictly increasing in E.
    """
    E = np.asarray(epochs, dtype=float)
    C = np.asarray(curve, dtype=float)
    i = int(np.searchsorted(C, float(level)))
    if i >= C.size:
        return float("inf")
    if i == 0:
        return float(E[0])
    return float(np.exp(np.log(E[i - 1]) + (level - C[i - 1]) / (C[i] - C[i - 1])
                        * (np.log(E[i]) - np.log(E[i - 1]))))


def expected_cumulative_top1(N, p_s, epochs, *, shape=DEFAULT_SHAPE, f=DEFAULT_F,
                             T=DEFAULT_T, draws=32, rng=None, method="exact",
                             progress=None):
    """
    Draw stake vectors and return {epochs, quenched, naive, p_draws, p_mean, p_sem}.

    Evaluates the exact per-chain per-epoch probability on each of `draws` quenched stake
    vectors and compounds both curves. `p_sem` is the Monte-Carlo uncertainty of the disorder
    average. method: "exact" (binomial closed form) or "normal" (Gaussian cross-check).
    progress: optional callable (done, total) invoked after each draw.
    """
    fn = top1_probability if method == "exact" else top1_probability_normal
    rng = np.random.default_rng(rng)
    vals = []
    for i in range(int(draws)):
        vals.append(fn(sample_relative_stakes(N, shape, rng=rng), p_s, f=f, T=T))
        if progress is not None:
            progress(i + 1, int(draws))
    p = np.array(vals, dtype=float)
    E = np.asarray(epochs, dtype=float)
    return {
        "epochs": E,
        "quenched": cumulative_top1(p, E),
        "naive": naive_cumulative(p.mean(), E),
        "p_draws": p,
        "p_mean": float(p.mean()),
        "p_sem": float(p.std(ddof=1) / np.sqrt(p.size)) if p.size > 1 else 0.0,
    }
