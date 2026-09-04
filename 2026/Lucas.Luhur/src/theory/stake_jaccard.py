"""
Closed form for stake_top_jaccard, E[J_x], the Jaccard overlap of the inferred and true top-x%
sets; the analytic twin of metrics.stake_privacy.stake_top_jaccard.

The inferred top-m set (m = round(x N)) is the m largest counts c_i ~ Binomial(T, q_i),
q_i = p_s + (1 - p_s) phi(alpha_i); A is the true top-m set, B the rest, K = |A n Top_hat| and
J = K/(2m - K). Conditioning on the threshold c* = c (the m-th largest count) with n_>(c) / n_=(c)
the number of nodes above / at c, c* = c iff n_>(c) < m <= n_>(c) + n_=(c), and
E[J] = sum_c sum_{a>, a=, b>, b=} P_A(a>, a=; c) P_B(b>, b=; c) 1[a> + b> < m <= a> + b> + a= + b=]
E[(a> + H)/(2m - a> - H)], H ~ Hypergeom(a= + b=, a=, m - a> - b>). P_A, P_B are
Poisson-multinomial laws from a dynamic programme over each group's nodes; the hypergeometric
term is the uniform tie credit. The sum over c runs on a strided grid within a trimmed window
(grid_points, n_sd, log_tol) and the DP caps the tied B-count at tie_cap (mass beyond it is
returned as `overflow`).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom, hypergeom

try:
    from consensus.election import DEFAULT_F, DEFAULT_T
    from consensus.stake import DEFAULT_SHAPE, sample_relative_stakes
    from theory.stake_top1 import participation_prob
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from consensus.election import DEFAULT_F, DEFAULT_T
    from consensus.stake import DEFAULT_SHAPE, sample_relative_stakes
    from theory.stake_top1 import participation_prob

DEFAULT_TOP_FRAC = 0.01
_TABLE_CACHE = {}


def top_set_size(N, top_frac=DEFAULT_TOP_FRAC):
    """Return m = round(x N), at least 1, as in metrics.stake_privacy.stake_top_jaccard."""
    return max(1, int(round(float(top_frac) * int(N))))


def random_set_jaccard(N, m):
    """
    Return E[J] of a uniformly random m-set against the true top-m (no-information reference).

    K ~ Hypergeom(N, m, m), so E[J] = sum_h P(K = h) h/(2m - h); 0.00532 at N = 1000, m = 10.
    """
    h = np.arange(m + 1)
    return float(np.sum(hypergeom.pmf(h, N, m, m) * h / (2.0 * m - h)))


def _config_tables(m, tie_cap):
    """
    Build cond[a>, a=, b>, b=] = 1[c is the threshold] and g = cond x E[J | configuration].

    Stake-independent, so cached per (m, tie_cap). E[J | config] averages
    J = (a> + H)/(2m - a> - H) over H ~ Hypergeom(a= + b=, a=, m - a> - b>).
    """
    key = (int(m), int(tie_cap))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    L = int(tie_cap)
    cond = np.zeros((m + 1, m + 1, m + 1, L + 1))
    g = np.zeros_like(cond)
    a_eq = np.arange(m + 1)[:, None]
    b_eq = np.arange(L + 1)[None, :]
    pop = a_eq + b_eq
    for ag in range(m + 1):
        for bg in range(m + 1):
            d = m - ag - bg
            if d < 1:
                continue
            ok = pop >= d
            h = np.arange(d + 1)[:, None, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                pmf = hypergeom.pmf(h, np.maximum(pop, d)[None], a_eq[None], d)
            pmf = np.where(np.isfinite(pmf), pmf, 0.0)
            ej = (pmf * (ag + h) / (2.0 * m - ag - h)).sum(axis=0)
            cond[ag, :, bg, :] = ok
            g[ag, :, bg, :] = np.where(ok, ej, 0.0)
    _TABLE_CACHE[key] = (cond, g)
    return cond, g


def _group_dp(pmf, sf, j_cap, l_cap):
    """
    Return the joint law S[c, j, l] of (# nodes above c, # nodes at c) over one group.

    pmf, sf: [n_grid, n_nodes] Binomial pmf at c and survival beyond c. The top indices
    j_cap / l_cap are absorbing; nodes that cannot reach the window are skipped exactly.
    """
    n_grid, n_nodes = pmf.shape
    S = np.zeros((n_grid, j_cap + 1, l_cap + 1))
    S[:, 0, 0] = 1.0
    for i in range(n_nodes):
        pg, pe = sf[:, i], pmf[:, i]
        if pg.max() + pe.max() < 1e-15:
            continue
        pl = np.clip(1.0 - pg - pe, 0.0, 1.0)
        new = S * pl[:, None, None]
        new[:, 1:, :] += S[:, :-1, :] * pg[:, None, None]
        new[:, j_cap, :] += S[:, j_cap, :] * pg[:, None]
        new[:, :, 1:] += S[:, :, :-1] * pe[:, None, None]
        new[:, :, l_cap] += S[:, :, l_cap] * pe[:, None]
        S = new
    return S


def jaccard_probability(alpha, p_s, *, top_frac=DEFAULT_TOP_FRAC, f=DEFAULT_F, T=DEFAULT_T,
                        grid_points=200, n_sd=10.0, log_tol=40.0, tie_cap=16, details=False):
    """
    Evaluate exact E[J_x] for one quenched stake vector; returns a float in [0, 1].

    With details=True returns {value, mass, overflow, stride, window}, where `mass` is the
    threshold law's total probability over the window (~1) and `overflow` the probability that
    more than tie_cap B-nodes tie at the threshold (~0). The null is random_set_jaccard(N, m).
    """
    alpha = np.asarray(alpha, dtype=float)
    N = alpha.size
    m = top_set_size(N, top_frac)
    if m >= N:
        return (1.0 if not details else {"value": 1.0, "mass": 1.0, "overflow": 0.0,
                                         "stride": 1, "window": (0, int(T))})
    q = participation_prob(alpha, p_s, f)
    order = np.argsort(alpha)
    A, B = order[-m:], order[:-m]
    mu = T * q
    sd = np.sqrt(T * q * (1.0 - q))
    lo = max(0, int(np.floor(np.sort(mu - n_sd * sd)[-m])))
    hi = min(int(T), int(np.ceil((mu + n_sd * sd).max())))
    cond, g = _config_tables(m, tie_cap)

    def _evaluate(c):
        pmf = binom.pmf(c[:, None], T, q[None, :])
        sf = binom.sf(c[:, None], T, q[None, :])
        SA = _group_dp(pmf[:, A], sf[:, A], m, m)
        SB = _group_dp(pmf[:, B], sf[:, B], m, tie_cap)
        mass = np.einsum("cij,ckl,ijkl->c", SA, SB, cond)
        val = np.einsum("cij,ckl,ijkl->c", SA, SB, g)
        over = np.einsum("cij,ck,ijk->c", SA, SB[:, :, tie_cap], cond[:, :, :, tie_cap])
        return mass, val, over

    coarse = max(1, (hi - lo) // 200)
    c_c = np.arange(lo, hi + 1, coarse)
    mass_c, _, _ = _evaluate(c_c)
    peak = mass_c.max()
    if peak <= 0.0:
        raise RuntimeError("stake_jaccard: the threshold law has no mass in the window -- "
                           "widen n_sd")
    keep = mass_c > peak * np.exp(-log_tol)
    lo = max(0, int(c_c[keep][0]) - coarse)
    hi = min(int(T), int(c_c[keep][-1]) + coarse)

    stride = max(1, (hi - lo) // max(int(grid_points), 1))
    c = np.arange(lo, hi + 1, stride)
    mass, val, over = _evaluate(c)
    value = float(np.clip(val.sum() * stride, 0.0, 1.0))
    if not details:
        return value
    return {"value": value, "mass": float(mass.sum() * stride),
            "overflow": float(over.sum() * stride), "stride": int(stride),
            "window": (int(lo), int(hi))}


def expected_jaccard(N, p_s, *, shape=DEFAULT_SHAPE, top_frac=DEFAULT_TOP_FRAC, f=DEFAULT_F,
                     T=DEFAULT_T, draws=32, rng=None):
    """
    Return the quenched average E_alpha[J_x] as {mean, sem, sd, draws}.

    Same convention as expected_top1: one stake draw from the Pareto(shape) law = one chain,
    `sd` the chain-to-chain spread and `sem` the Monte-Carlo uncertainty of the curve.
    """
    rng = np.random.default_rng(rng)
    vals = np.array([jaccard_probability(sample_relative_stakes(N, shape, rng=rng), p_s,
                                         top_frac=top_frac, f=f, T=T)
                     for _ in range(int(draws))], dtype=float)
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return {"mean": float(vals.mean()), "sd": sd,
            "sem": sd / np.sqrt(vals.size) if vals.size > 1 else 0.0, "draws": int(vals.size)}
