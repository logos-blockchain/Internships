"""
Closed form for stake_confidence, the fraction of nodes whose stake estimate lands within
+/-gamma; the analytic twin of metrics.stake_privacy.stake_confidence.

alpha_hat_i = log((1 - n_i/T) / (1 - p_s)) / log(1 - f) is increasing in the count n_i, so
the band [alpha_i(1-gamma), alpha_i(1+gamma)] is exactly an integer interval
[ceil(n(alpha_i(1-gamma))), floor(n(alpha_i(1+gamma)))] with n(a) = T[1 - (1 - p_s)(1 - f)^a].
With n_i ~ Binomial(T, q_i), q_i = p_s + (1 - p_s) phi(alpha_i), independent across nodes,
E[stake_confidence | alpha] = (1/N) sum_i [F_i(hi_i) - F_i(lo_i - 1)]. The alpha_hat floor at
0 never falls inside a band. This is the cover generalisation of the no-cover Gaussian
metrics.stake_privacy.inference_confidence, which is its p_s = 0 limit.
"""

from __future__ import annotations

import numpy as np
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

DEFAULT_GAMMA = 0.1


def participation_prob(alpha, p_s, f=DEFAULT_F):
    """
    Return q_i = Pr(i in S_t) = p_s + (1 - p_s) phi(alpha_i), the per-slot send probability.

    Identical to theory.stake_top1.participation_prob.
    """
    return p_s + (1.0 - p_s) * lottery(np.asarray(alpha, dtype=float), f)


def count_at_estimate(a, p_s, f=DEFAULT_F, T=DEFAULT_T):
    """
    Return n(a) = T[1 - (1 - p_s)(1 - f)^a], the count at which alpha_hat equals a.

    The exact inverse of the sender-set estimator; increasing in a, with n(0) = T p_s.
    """
    return T * (1.0 - (1.0 - p_s) * (1.0 - f) ** np.asarray(a, dtype=float))


def per_node_confidence(alpha, p_s, *, gamma=DEFAULT_GAMMA, f=DEFAULT_F, T=DEFAULT_T):
    """
    Return P(alpha_hat_i in [alpha_i(1-gamma), alpha_i(1+gamma)]) for each node as an array.
    """
    alpha = np.asarray(alpha, dtype=float)
    q = participation_prob(alpha, p_s, f)
    lo = np.ceil(count_at_estimate(alpha * (1.0 - gamma), p_s, f, T))
    hi = np.floor(count_at_estimate(alpha * (1.0 + gamma), p_s, f, T))
    lo = np.clip(lo, 0.0, T)
    hi = np.clip(hi, -1.0, T)
    p_in = binom.cdf(hi, T, q) - binom.cdf(lo - 1.0, T, q)
    return np.clip(p_in, 0.0, 1.0)


def confidence_probability(alpha, p_s, *, gamma=DEFAULT_GAMMA, f=DEFAULT_F, T=DEFAULT_T):
    """
    Return exact E[stake_confidence] for one quenched stake vector (mean of per_node_confidence).

    Returns a float in [0, 1]. Its null is 0, not a positive random-guess floor: a blinded
    adversary's alpha_hat concentrates at 0, below every band.
    """
    return float(per_node_confidence(alpha, p_s, gamma=gamma, f=f, T=T).mean())


def confidence_probability_normal(alpha, p_s, *, gamma=DEFAULT_GAMMA, f=DEFAULT_F, T=DEFAULT_T):
    """
    Return the same quantity under a Gaussian approximation to the counts (cross-check route).

    n_i ~= Normal(T q_i, T q_i (1 - q_i)) over the same integer interval as the exact form,
    with a +/-0.5 continuity correction. The correction is essential: the band is only
    2 gamma alpha T (1 - p_s)(1 - f)^alpha |ln(1 - f)| counts wide, a small fraction of one
    count sd, so the uncorrected version errs by up to ~26% at reduced T.
    """
    alpha = np.asarray(alpha, dtype=float)
    q = participation_prob(alpha, p_s, f)
    sd = np.sqrt(T * q * (1.0 - q))
    lo = np.clip(np.ceil(count_at_estimate(alpha * (1.0 - gamma), p_s, f, T)), 0.0, T)
    hi = np.clip(np.floor(count_at_estimate(alpha * (1.0 + gamma), p_s, f, T)), -1.0, T)
    p_in = norm.cdf((hi + 0.5 - T * q) / sd) - norm.cdf((lo - 0.5 - T * q) / sd)
    return float(np.clip(p_in, 0.0, 1.0).mean())


def expected_confidence(N, p_s, *, shape=DEFAULT_SHAPE, gamma=DEFAULT_GAMMA, f=DEFAULT_F,
                        T=DEFAULT_T, draws=32, rng=None, method="exact"):
    """
    Return the quenched average E_alpha[stake_confidence] as {mean, sem, sd, draws}.

    One stake draw from the Pareto(shape) law = one chain; `sd` is the chain-to-chain spread
    and `sem` the Monte-Carlo uncertainty of the curve. method: "exact" (binomial closed
    form) or "normal" (Gaussian cross-check).
    """
    fn = confidence_probability if method == "exact" else confidence_probability_normal
    rng = np.random.default_rng(rng)
    vals = np.array([fn(sample_relative_stakes(N, shape, rng=rng), p_s, gamma=gamma, f=f, T=T)
                     for _ in range(int(draws))], dtype=float)
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return {"mean": float(vals.mean()), "sd": sd,
            "sem": sd / np.sqrt(vals.size) if vals.size > 1 else 0.0, "draws": int(vals.size)}
