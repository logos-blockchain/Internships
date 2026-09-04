"""
Per-message link jitter: the annealed congestion term eps_m ~ Exp(lambda_eps) added to every
link on top of the quenched structural latency, d_link(m) = L + X + eps_m. Over the k+1 AC-path
links it sums to E ~ Gamma(k+1, scale); the adversary knows the parameter, not the draw, so its
likelihood is the density of Z + E and the SNR is eta = sigma_d / sqrt(sigma_Z^2 + sigma_eps^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JitterParams:
    """
    Per-message jitter parameters.

    scale -- 1/lambda_eps, the mean (and sd) per-link jitter in slot units. 0.0 (default)
             turns the jitter off and consumes no randomness. No magnitude is calibrated
             here; a config states it or a sweep varies it.
    """

    scale: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError(f"jitter scale must be finite and >= 0, got {self.scale!r}")


def jitter_moments(n_links, scale):
    """
    Return (mean, variance) of the AC path's total jitter E ~ Gamma(n_links, scale).

    E[E] = n_links*scale and Var(E) = n_links*scale^2, so sigma_eps = sqrt(n_links)*scale.
    n_links is the AC path's link count k+1, not the hold's stage count.
    """
    n = int(n_links)
    s = float(scale)
    if n < 0:
        raise ValueError(f"jitter_moments needs n_links >= 0, got {n}")
    if s < 0.0:
        raise ValueError(f"jitter_moments needs scale >= 0, got {s}")
    return float(n * s), float(n * s * s)


def ac_path_links(hops):
    """
    Return the AC path's link count k+1 (S->M_1, the k-1 inter-mix links, M_k->R).

    Shared by the layer, the attack and the SNR bookkeeping. Independent of
    `receiver_delays`, which adds a hold at R, not a link.
    """
    k = int(hops)
    if k < 1:
        raise ValueError(f"ac_path_links needs hops >= 1, got {k}")
    return k + 1
