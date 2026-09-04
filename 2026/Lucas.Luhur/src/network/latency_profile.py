"""
Quenched AC-path latency profile: the deterministic per-node link latencies the GPA knows.

On the single path S -> M_1 -> ... -> M_k -> R a message from sender i to receiver r accrues
mu(i, r) = d_i^S + D_M + d_r^R (sender link, mix-chain total, receiver link), sampled once per
experiment and frozen. D_M + d_r^R cancel in the attack's residual, so only the spread of
{d_i^S} discriminates candidates. Two laws share one schema: uniform U(low, high) per link
kind, or the shifted log-normal (lognormal_latency.py); the homogeneous default has sigma_d = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .jitter import JitterParams
    from .latency import (DEFAULT_C, DEFAULT_D, DEFAULT_LAM, broadcast_latency_theory,
                          weighted_broadcast_latency)
    from .lognormal_latency import LogNormalParams, lognormal_draw, sample_lognormal_links
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from network.jitter import JitterParams
    from network.latency import (DEFAULT_C, DEFAULT_D, DEFAULT_LAM, broadcast_latency_theory,
                                 weighted_broadcast_latency)
    from network.lognormal_latency import (LogNormalParams, lognormal_draw,
                                           sample_lognormal_links)


@dataclass(frozen=True)
class LatencyProfileParams:
    """
    Quenched-latency parameters: one law for the AC-path links, plus the optional jitter.

    sender_low/high   -- d_i^S ~ U(low, high) in slot units; None -> d (homogeneous). The
                         attribution signal: sigma_d = (high - low)/sqrt(12), eta = sigma_d/sigma_Z.
    lognormal         -- a LogNormalParams: draw every AC-path link from d = floor + LogNormal.
                         Mutually exclusive with the uniform knobs.
    receiver_low/high -- d_r^R ~ U(low, high); None -> d. Cancels in attribution.
    mix_total         -- D_M as one fixed number; None -> (k-1)*d. Cancels in attribution.
    mix_low/high      -- D_M as a sum of k-1 per-link U(low, high) draws (Irwin-Hall:
                         E[D_M] = (k-1)(low+high)/2, Var(D_M) = (k-1)(high-low)^2/12).
                         Mutually exclusive with mix_total.
    jitter            -- a JitterParams: add the annealed per-message term eps_m ~ Exp(lambda_eps)
                         to every link on top of the quenched law. None or scale = 0 -> off.
    """

    sender_low: float | None = None
    sender_high: float | None = None
    receiver_low: float | None = None
    receiver_high: float | None = None
    mix_total: float | None = None
    mix_low: float | None = None
    mix_high: float | None = None
    lognormal: LogNormalParams | None = None
    jitter: JitterParams | None = None


@dataclass(frozen=True)
class LatencyProfile:
    """
    A frozen realisation of the deterministic AC-path latencies (quenched disorder).

    d_sender / d_receiver are length-N per-node vectors; d_mix is the scalar mix-chain total;
    jitter_scale is the annealed per-message jitter's 1/lambda_eps (0.0 = off). The layer
    generates and the attack reads through this one object.
    """

    d_sender: np.ndarray
    d_receiver: np.ndarray
    d_mix: float
    jitter_scale: float = 0.0

    def mu(self, sender, receiver):
        """
        Deterministic latency mu(i, r) = d_i^S + D_M + d_r^R, vectorised over index arrays.

        The GPA subtracts it from the observed broadcast time y to form the residual z = y - mu.
        """
        i = np.asarray(sender, dtype=np.int64)
        r = np.asarray(receiver, dtype=np.int64)
        return self.d_sender[i] + self.d_mix + self.d_receiver[r]


def _draw_links(N, low, high, d, rng):
    """One quenched vector of N link latencies: U(low, high), or homogeneous d if unset."""
    if low is None and high is None:
        return np.full(N, float(d), dtype=np.float64)
    lo = float(d if low is None else low)
    hi = float(d if high is None else high)
    if hi < lo:
        raise ValueError(f"latency-profile range needs low <= high, got ({lo}, {hi})")
    if lo < 0.0:
        raise ValueError(f"latency-profile range must be non-negative, got low={lo}")
    return rng.uniform(lo, hi, size=N).astype(np.float64)


def _draw_mix_total(k, params, d, rng):
    """
    D_M, the total inter-mix latency over the k-1 middle links.

    mix_low/high set -> the sum of k-1 per-link U(low, high) draws; mix_total set -> that
    scalar; neither -> (k-1)*d. Called after d_sender / d_receiver so the sender signal is
    unchanged by the mix-chain choice.
    """
    n_links = max(int(k) - 1, 0)
    drawn = params.mix_low is not None or params.mix_high is not None
    if drawn and params.mix_total is not None:
        raise ValueError(
            "latency params set BOTH `mix_total` and `mix_low`/`mix_high` -- a fixed D_M and a "
            "per-link draw are two competing models of the same inter-mix chain. Keep one.")
    if not drawn:
        return float(params.mix_total) if params.mix_total is not None else n_links * float(d)
    return float(_draw_links(n_links, params.mix_low, params.mix_high, d, rng).sum())


def sample_latency_profile(N, k, d, params=None, rng=None):
    """
    Draw a quenched LatencyProfile for N nodes on a k-hop single path.

    N -- node count; k -- hop count (default D_M = (k-1)*d); d -- the homogeneous link latency
    each unset bound collapses to; params -- LatencyProfileParams (None -> homogeneous);
    rng -- seeded per realisation. The homogeneous default gives mu(i, r) = (k+1)d for every
    candidate. Links follow either the uniform law per link kind or, if params.lognormal is
    set, the shifted log-normal for every leg; the result has the same three fields either way.
    """
    rng = np.random.default_rng(rng)
    params = params or LatencyProfileParams()

    if params.lognormal is not None:
        clash = [name for name, v in (
            ("sender_low", params.sender_low), ("sender_high", params.sender_high),
            ("receiver_low", params.receiver_low), ("receiver_high", params.receiver_high),
            ("mix_total", params.mix_total),
            ("mix_low", params.mix_low), ("mix_high", params.mix_high)) if v is not None]
        if clash:
            raise ValueError(
                f"latency params set BOTH `lognormal` and {clash} -- `lognormal` drives every "
                "AC-path link (sender, inter-mix, receiver) from its own law, so the "
                "uniform/scalar knobs are redundant. Drop them, or drop `lognormal` for the "
                "uniform eta dial.")

    eps = 0.0 if params.jitter is None else float(params.jitter.scale)

    if params.lognormal is not None:
        d_sender, d_receiver, d_mix = sample_lognormal_links(N, k, params.lognormal, rng=rng)
        return LatencyProfile(d_sender=d_sender, d_receiver=d_receiver, d_mix=d_mix,
                              jitter_scale=eps)

    d_sender = _draw_links(N, params.sender_low, params.sender_high, d, rng)
    d_receiver = _draw_links(N, params.receiver_low, params.receiver_high, d, rng)
    d_mix = _draw_mix_total(k, params, d, rng)
    return LatencyProfile(d_sender=d_sender, d_receiver=d_receiver, d_mix=d_mix, jitter_scale=eps)


def is_homogeneous(params=None):
    """
    Return True when no link law is set, so every link is the flat system value d.

    The homogeneous case is the only one with a closed-form broadcast latency.
    """
    if params is None:
        return True
    return (params.lognormal is None
            and params.sender_low is None and params.sender_high is None)


def profile_broadcast_latency(N, C=DEFAULT_C, d=None, lam=DEFAULT_LAM, params=None,
                              profile=None, n_sources=64, rng=None):
    """
    E[D_br], the block-broadcast latency over the P2P gossip graph under the link law in force.

    homogeneous -> the exact closed form d * E[ecc] (broadcast_latency_theory at rho_net = 0);
    lognormal   -> edge = floor + LogNormal, i.i.d. per edge, from the AC path's params;
    uniform     -> edge = U(sender_low, sender_high), i.i.d. per edge.
    The heterogeneous branches share weighted_broadcast_latency (quenched weights, Dijkstra).
    `profile` is accepted for interface compatibility and currently unused. E[D_br] is
    downstream of the observed broadcast time, so it moves the latency cost ell, never the posterior.
    """
    params = params or LatencyProfileParams()
    d = DEFAULT_D if d is None else float(d)

    eps = 0.0 if params.jitter is None else float(params.jitter.scale)

    if eps > 0.0 and np.isfinite(lam):
        raise ValueError(
            f"both jitter models are on: latency.jitter.scale = {eps} (seconds) AND the "
            f"gossip jitter rho_net (lam = {lam:.4g} < inf). They are the SAME congestion term -- "
            f"jitter.scale = rho_net * d -- so enabling both double-counts it. Keep "
            f"latency.jitter and leave the system `rho` at 0.")

    if is_homogeneous(params):
        return broadcast_latency_theory(N, C, d + eps, lam)

    if params.lognormal is not None:
        draw = lognormal_draw(params.lognormal)

        def _edges(u, v, r):
            return draw(r, u.size) + eps
    else:
        lo = float(d if params.sender_low is None else params.sender_low)
        hi = float(d if params.sender_high is None else params.sender_high)

        def _edges(u, v, r):
            return r.uniform(lo, hi, size=u.size) + eps

    return weighted_broadcast_latency(N, C, _edges, n_sources=n_sources, rng=rng)
