"""
Closed forms for the reported signal spread sigma_hat_d and the population sd sigma_d(W) it
estimates. sigma_hat_d = sqrt((1/m) sum_{i in S_t} (d_i^S - dbar)^2) over the m = |S_t| candidates
of one sender set; sigma_d(W) is sigma_link under `split` and sigma_link/sqrt(W) under `uniform`.
The bias b(W, m) = E[sigma_hat_d]/sigma_d(W) < 1 combines the exact 1/m-divisor factor
sqrt((m-1)/m) with a Jensen gap that shrinks as averaging W links Gaussianises the leg. Scope:
shifted log-normal link law and fixed-`count` cover (|S_t| law from attribution.candidate_set_law).
"""

from __future__ import annotations

import numpy as np

try:
    from consensus.election import DEFAULT_F
    from network.lognormal_latency import link_moments, lognormal_draw
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from consensus.election import DEFAULT_F
    from network.lognormal_latency import link_moments, lognormal_draw

from .attribution import candidate_set_law

ASSIGNMENTS = ("split", "uniform")

_SIGMA_HAT_CACHE = {}


def population_sigma_d(link, width=1, assignment="split"):
    """
    Return sigma_d(W), the population sd of the per-candidate sender leg, in seconds.

    `split`: the leg is one fixed entry link, so sigma_d = sigma_link at every W.
    `uniform`: the leg is the mean of W i.i.d. entry links, so sigma_d = sigma_link/sqrt(W).
    """
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"unknown sender->entry assignment {assignment!r}; expected one of "
                         f"{ASSIGNMENTS} (see network/mixnet_latency.py)")
    sigma_link = float(np.sqrt(link_moments(link)[1]))
    W = int(width)
    if W < 1:
        raise ValueError(f"width must be >= 1, got {W}")
    return sigma_link / np.sqrt(W) if assignment == "uniform" else sigma_link


def expected_sigma_d_hat(*, link, count, width=1, assignment="split", f=DEFAULT_F,
                         sets=200_000, seed=0):
    """
    Return E[sigma_hat_d], the expectation of the reported per-set spread, in seconds.

    `link` is a LogNormalParams; |S_t| = count + 1 + Poisson(-ln(1-f)) is taken from
    attribution.candidate_set_law. E[sigma_hat_d^2 | m] = sigma_d(W)^2 (m-1)/m is exact; the
    sqrt is evaluated by Monte Carlo conditioned on each m and recombined with the exact
    probabilities. Deterministic given the arguments and memoised.
    """
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"unknown sender->entry assignment {assignment!r}; expected one of "
                         f"{ASSIGNMENTS} (see network/mixnet_latency.py)")
    key = (link, int(count), int(width), assignment, float(f), int(sets), int(seed))
    if key in _SIGMA_HAT_CACHE:
        return _SIGMA_HAT_CACHE[key]

    W = int(width)
    n_avg = W if assignment == "uniform" else 1
    sizes, probs = candidate_set_law(int(count), f)
    draw = lognormal_draw(link)
    rng = np.random.default_rng(seed)

    total = 0.0
    for m, p in zip(sizes, probs):
        m, p = int(m), float(p)
        if p < 1e-12:
            continue
        n_sets = max(1_000, int(round(int(sets) * p)))
        legs = np.zeros((n_sets, m), dtype=np.float64)
        for _ in range(n_avg):
            legs += draw(rng, (n_sets, m))
        legs /= n_avg
        var = legs.var(axis=1)
        total += p * float(np.sqrt(var).mean())

    _SIGMA_HAT_CACHE[key] = total
    return total


def sigma_d_bias(*, link, count, width=1, assignment="split", f=DEFAULT_F, **kw):
    """
    Return b(W, m) = E[sigma_hat_d] / sigma_d(W), the finite-m bias of the reported spread.

    Strictly < 1 and bounded above by sqrt((m-1)/m); b -> 1 as m -> infinity.
    """
    return expected_sigma_d_hat(link=link, count=count, width=width, assignment=assignment,
                                f=f, **kw) / population_sigma_d(link, width, assignment)
