"""
Shifted log-normal link-latency law, d = L + LogNormal(mu, sigma), with L a hard floor.

The law is parameterised by its physical (mean, sd) and moment-matched to the WonderNetwork
ping population (ping_data.ping_link_moments): floor = 7.2 ms, mean = 62.9 ms, sd = 33.3 ms.
It is continuous, so candidate senders never tie exactly, and sd is a direct dial on sigma_d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogNormalParams:
    """
    Shifted-log-normal parameters, given as physical moments rather than log-space (mu, sigma).

    floor -- L, the hard minimum link latency (seconds); a pure shift that never moves the variance.
    mean  -- E[d], the mean of the whole link latency, floor included; must exceed `floor`.
    sd    -- sd(d); the signal dial (sigma_d scales with it). sd = 0 makes every link exactly `mean`.

    One law drives the sender link d_i^S, the k-1 inter-mix links and the receiver link d_r^R.
    """

    floor: float = 0.0
    mean: float = 1.0
    sd: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.floor) or self.floor < 0.0:
            raise ValueError(f"lognormal floor must be finite and >= 0, got {self.floor!r}")
        if not np.isfinite(self.mean) or self.mean <= self.floor:
            raise ValueError(
                f"lognormal mean must exceed the floor (the log-normal part carries mean - floor "
                f"> 0), got mean={self.mean!r} floor={self.floor!r}")
        if not np.isfinite(self.sd) or self.sd < 0.0:
            raise ValueError(f"lognormal sd must be finite and >= 0, got {self.sd!r}")


def lognormal_from_moments(mean, sd):
    """
    Map a log-normal's (mean, sd) to its log-space (mu, sigma).

    With cv = sd/mean: sigma^2 = log(1 + cv^2), mu = log(mean) - sigma^2/2. `mean` is the
    log-normal part only (link mean minus the floor). sd = 0 gives the point mass at `mean`.
    """
    mean = float(mean)
    sd = float(sd)
    if mean <= 0.0:
        raise ValueError(f"a log-normal needs mean > 0, got {mean!r}")
    if sd < 0.0:
        raise ValueError(f"a log-normal needs sd >= 0, got {sd!r}")
    var_log = float(np.log1p((sd / mean) ** 2))          # log1p: accurate for small cv
    return float(np.log(mean) - var_log / 2.0), float(np.sqrt(var_log))


def lognormal_moments(mu, sigma):
    """
    Map log-space (mu, sigma) to the log-normal's (mean, variance).

    The inverse of lognormal_from_moments; the round trip is checked by the validation harness.
    """
    mu = float(mu)
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError(f"a log-normal needs sigma >= 0, got {sigma!r}")
    s2 = sigma * sigma
    mean = float(np.exp(mu + s2 / 2.0))
    var = float(np.expm1(s2) * np.exp(2.0 * mu + s2))    # expm1: exact for small sigma
    return mean, var


def link_moments(params=None):
    """
    Return the (mean, variance) of the shifted law d = floor + LogNormal, in seconds.

    Recomputed from the derived (mu, sigma) with the shift added back. sqrt(var) is the
    population sigma_d; see per_set_sd_moments for the per-sender-set reduction.
    """
    params = params or LogNormalParams()
    mu, sigma = lognormal_from_moments(params.mean - params.floor, params.sd)
    mean, var = lognormal_moments(mu, sigma)
    return float(mean + params.floor), float(var)


def per_set_sd_moments(set_size, params=None):
    """
    Return (rms_anchor, population_sd) for the per-sender-set sigma_d at |S_t| = m candidates.

    With sigma_d = E[sqrt((1/m) sum_{i in S_t} (d_i^S - dbar)^2)], the anchor
    sqrt(sigma^2 (m-1)/m) is exact and distribution-free, and the realised sigma_d sits below
    it by Jensen, with a gap that grows with the law's skew.
    """
    params = params or LogNormalParams()
    m = int(set_size)
    if m < 2:
        raise ValueError(f"per_set_sd_moments needs set_size >= 2, got {m}")
    _, var = link_moments(params)
    sigma = float(np.sqrt(var))
    return float(sigma * np.sqrt((m - 1) / m)), sigma


def lognormal_draw(params=None):
    """
    Return the per-link sampler draw(rng, size) -> [size] latencies for the shifted law.

    Shared by the single-path AC legs, the mix-net grid legs and the P2P gossip edges, so
    d = floor + LogNormal(mu, sigma) lives in one place. sigma = 0 returns the point mass
    at `mean` and consumes no randomness.
    """
    params = params or LogNormalParams()
    mu, sigma = lognormal_from_moments(params.mean - params.floor, params.sd)
    floor = float(params.floor)

    def draw(rng, size):
        if sigma == 0.0:
            return np.full(size, floor + float(np.exp(mu)), dtype=np.float64)
        return floor + rng.lognormal(mu, sigma, size=size).astype(np.float64)

    return draw


def sample_lognormal_links(N, k, params=None, rng=None):
    """
    Draw the quenched AC path S -> M_1 -> ... -> M_k -> R (k+1 links) from the shifted law.

        d_sender[i]   = floor + LogNormal        (sender -> first mix, the signal)
        d_receiver[i] = floor + LogNormal        (last mix -> receiver; cancels in the residual)
        d_mix         = sum_{j=1}^{k-1} (floor + LogNormal_j)   (cancels in the residual)

    The sender vector is drawn first, then the receiver, then the mix chain, so changing k
    leaves d_sender bit-identical. Returns (d_sender[N], d_receiver[N], d_mix scalar).
    """
    rng = np.random.default_rng(rng)
    params = params or LogNormalParams()
    if int(k) < 1:
        raise ValueError(f"sample_lognormal_links needs k >= 1, got {k}")
    draw = lognormal_draw(params)

    d_sender = draw(rng, int(N))
    d_receiver = draw(rng, int(N))
    d_mix = float(draw(rng, max(int(k) - 1, 0)).sum())
    return d_sender, d_receiver, d_mix
