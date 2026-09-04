"""
Closed form for stake_top1_hit, P(the GPA names the largest stakeholder); the analytic twin
of metrics.stake_privacy.stake_top1_hit.

The stake estimate is an increasing transform of the participation count n_i, so the measure
is an order statistic on counts: P_Top1 = P(argmax_i n_i = argmax_i alpha_i) with uniform
tie-split, where n_i ~ Binomial(T, q_i), q_i = p_s + (1 - p_s) phi(alpha_i), independent across
nodes. With w the whale, b_i / F_i the Binomial pmf / cdf and 1/(K+1) = integral_0^1 u^K du,
P_Top1(alpha) = sum_m b_w(m) integral_0^1 prod_{i != w} [F_i(m - 1) + u b_i(m)] du.
expected_top1 takes the quenched average over stake draws.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom, norm

try:
    from consensus.election import DEFAULT_F, DEFAULT_T, lottery
    from consensus.stake import DEFAULT_SHAPE, sample_relative_stakes
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from consensus.election import DEFAULT_F, DEFAULT_T, lottery
    from consensus.stake import DEFAULT_SHAPE, sample_relative_stakes


def participation_prob(alpha, p_s, f=DEFAULT_F):
    """
    Return q_i = Pr(i in S_t) = p_s + (1 - p_s) phi(alpha_i), the per-slot send probability.

    p_s -> 0 gives q = phi (stake fully exposed); p_s -> 1 gives q -> 1 for every node.
    """
    return p_s + (1.0 - p_s) * lottery(np.asarray(alpha, dtype=float), f)


MAX_QUAD = 256


def top1_probability(alpha, p_s, *, f=DEFAULT_F, T=DEFAULT_T, n_quad=None,
                     log_tol=40.0, n_sd=12.0, grid_points=200):
    """
    Evaluate exact P_Top1 for one quenched stake vector; returns a float in [0, 1].

    Computed in log space with logsumexp. n_quad: Gauss-Legendre order for the u-integral;
    None picks Q = clip(ceil((K_eff + 3)/2), 4, MAX_QUAD) from the expected tie degree
    K_eff = sum_i b_i(m)/(F_i(m-1) + b_i(m)), so the p_s -> 1 limit still returns 1/N.
    n_sd/log_tol: the m-window (whale mean +/- n_sd sd, trimmed to summands within
    exp(-log_tol) of the peak). grid_points: the sum over m runs on a strided grid and is
    multiplied by the stride. The null is 1/N.
    """
    alpha = np.asarray(alpha, dtype=float)
    q = participation_prob(alpha, p_s, f)
    w = int(np.argmax(alpha))
    q_o = np.delete(q, w)
    if q_o.size == 0:
        return 1.0

    mu = T * q[w]
    sd = np.sqrt(T * q[w] * (1.0 - q[w]))
    lo = max(0, int(np.floor(mu - n_sd * sd)))
    hi = min(T, int(np.ceil(mu + n_sd * sd)))
    coarse = max(1, (hi - lo) // 200)
    mc = np.arange(lo, hi + 1, coarse)
    F_c = binom.cdf(mc[:, None] - 1, T, q_o[None, :])
    b_c = binom.pmf(mc[:, None], T, q_o[None, :])
    prof = binom.logpmf(mc, T, q[w]) + np.log(np.maximum(F_c, 1e-300)).sum(axis=1)
    keep = prof > prof.max() - log_tol
    k_eff = float((b_c / np.maximum(F_c + b_c, 1e-300)).sum(axis=1)[keep].max())
    lo = max(0, int(mc[keep][0]) - coarse)
    hi = min(T, int(mc[keep][-1]) + coarse)

    stride = max(1, (hi - lo) // max(grid_points, 1))
    m_grid = np.arange(lo, hi + 1, stride)

    Q = int(n_quad) if n_quad else int(np.clip(np.ceil((k_eff + 3.0) / 2.0), 4, MAX_QUAD))
    x, wt = np.polynomial.legendre.leggauss(Q)
    u = 0.5 * (x + 1.0)
    wt = 0.5 * wt

    parts = []
    for s in range(0, m_grid.size, 512):
        m = m_grid[s:s + 512]
        b_o = binom.pmf(m[:, None], T, q_o[None, :])
        F_o = binom.cdf(m[:, None] - 1, T, q_o[None, :])
        inner = F_o[:, None, :] + u[None, :, None] * b_o[:, None, :]
        lp = np.log(np.maximum(inner, 1e-300)).sum(axis=2)
        lp += binom.logpmf(m, T, q[w])[:, None]
        parts.append(logsumexp(lp, b=np.broadcast_to(wt, lp.shape)))
    return float(np.clip(np.exp(logsumexp(parts)) * stride, 0.0, 1.0))


def top1_probability_normal(alpha, p_s, *, f=DEFAULT_F, T=DEFAULT_T, n_grid=2001, n_sd=10.0):
    """
    Return the same quantity under a Gaussian approximation to the counts (cross-check route).

    n_i ~= Normal(T q_i, T q_i (1 - q_i)) is continuous, so there are no ties and
    P_Top1 ~= int phi_w(x) prod_{i != w} Phi_i(x) dx is a 1-D quadrature. Omitting the tie
    mass makes it sit ~0.6% low at p_s = 0.9.
    """
    alpha = np.asarray(alpha, dtype=float)
    q = participation_prob(alpha, p_s, f)
    w = int(np.argmax(alpha))
    mu = T * q
    sd = np.sqrt(T * q * (1.0 - q))
    if alpha.size < 2:
        return 1.0
    xs = np.linspace(mu[w] - n_sd * sd[w], mu[w] + n_sd * sd[w], int(n_grid))
    mu_o, sd_o = np.delete(mu, w), np.delete(sd, w)
    log_cdf = norm.logcdf((xs[:, None] - mu_o[None, :]) / sd_o[None, :]).sum(axis=1)
    return float(np.clip(np.trapezoid(norm.pdf(xs, mu[w], sd[w]) * np.exp(log_cdf), xs), 0.0, 1.0))


def expected_top1(N, p_s, *, shape=DEFAULT_SHAPE, f=DEFAULT_F, T=DEFAULT_T,
                  draws=32, rng=None, method="exact"):
    """
    Return the quenched average E_alpha[P_Top1(alpha)] as {mean, sem, sd, draws}.

    One stake draw from the Pareto(shape) law = one chain; `sd` is the chain-to-chain spread
    and `sem` = sd/sqrt(draws) the Monte-Carlo uncertainty of the curve. method: "exact"
    (binomial closed form) or "normal" (Gaussian cross-check).
    """
    fn = top1_probability if method == "exact" else top1_probability_normal
    rng = np.random.default_rng(rng)
    vals = np.array([fn(sample_relative_stakes(N, shape, rng=rng), p_s, f=f, T=T)
                     for _ in range(int(draws))], dtype=float)
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return {"mean": float(vals.mean()), "sd": sd,
            "sem": sd / np.sqrt(vals.size) if vals.size > 1 else 0.0, "draws": int(vals.size)}
