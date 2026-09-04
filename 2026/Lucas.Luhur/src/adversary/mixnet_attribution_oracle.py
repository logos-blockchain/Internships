"""
Mix-net route-oracle sender attribution: the same Bayesian attack as mixnet_attribution
but granted the true route p* of each broadcast (trace.true_route), giving an upper bound
on leakage that pairs with the route-latent lower bound.

Pr(L = i | y, r, p*, S_t) is proportional to Pr(p* | i) f_Z(y - mu(i, p*, r)). Under the
"split" assignment Pr(p* | i) excludes candidates whose entry differs from entry(p*)
(|S_t| -> ~|S_t|/W); under "uniform" it is a common constant. W = 1 reduces exactly to
mixnet_attribution. Reuses MixnetAttributionParams.
"""

from __future__ import annotations

import numpy as np

from pipeline_contract import PosteriorGuess

try:
    from anonymity.single_path_mix import residual_delay_pdf
    from network.jitter import ac_path_links
    from .gpa import observe_broadcasts
    from .mixnet_attribution import MixnetAttributionParams
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anonymity.single_path_mix import residual_delay_pdf
    from network.jitter import ac_path_links
    from adversary.gpa import observe_broadcasts
    from adversary.mixnet_attribution import MixnetAttributionParams


def run(trace, *, params=None, rng=None):
    """
    Run the route-oracle attribution attack; returns a PosteriorGuess (one posterior per broadcast).

    Uniform attack signature run(trace, *, params, rng); rng is unused (deterministic).
    One f_Z evaluation per candidate row ([K], against the latent arm's [K, P']).
    """
    p = params or MixnetAttributionParams()
    if (p.latency_profile is None or p.hops is None or p.mix_scale is None
            or p.sender_scale is None):
        raise ValueError(
            "mixnet_attribution_oracle called unwired: hops / mix_scale / sender_scale / "
            "latency_profile must be filled (run_once fills them from the system config via "
            "AttackSpec.knows).")
    n_stages = int(p.hops) + (1 if p.receiver_delays else 0)
    prof = p.latency_profile

    obs = observe_broadcasts(trace)
    B = len(obs)
    if B == 0:
        return PosteriorGuess(broadcast_row=obs.broadcast_row, start=obs.start,
                              candidate=obs.candidate, posterior=np.zeros(0))

    p_star = trace.true_route[obs.broadcast_row]
    if np.any(p_star < 0):
        raise ValueError(
            "mixnet_attribution_oracle needs trace.true_route (the mixnet layer records it when "
            "run with a latency oracle); this trace has broadcasts with no recorded route. The "
            "oracle arm is defined only for the routed mix-net experiment.")

    counts = np.diff(obs.start)
    r_row = np.repeat(obs.receiver, counts)
    y_row = np.repeat(obs.y, counts)
    p_row = np.repeat(p_star, counts)

    eps = float(getattr(prof, "jitter_scale", 0.0) or 0.0)
    mu = prof.mu_on_route(obs.candidate, r_row, p_row)
    like = residual_delay_pdf(y_row - mu, n_stages, float(p.sender_scale), float(p.mix_scale),
                              eps, ac_path_links(int(p.hops)) if eps > 0.0 else 0)

    if prof.assignment == "split":
        allowed = prof.entry_of[obs.candidate] == prof.route_entry[p_row]
        like = np.where(allowed, like, 0.0)
    else:
        allowed = np.ones(like.size, dtype=bool)

    seg_sum = np.add.reduceat(like, obs.start[:-1])
    denom = np.repeat(seg_sum, counts)
    n_allowed = np.add.reduceat(allowed.astype(float), obs.start[:-1])
    n_allowed_row = np.repeat(n_allowed, counts)
    posterior = np.where(denom > 0, like / np.where(denom > 0, denom, 1.0),
                         allowed / np.maximum(n_allowed_row, 1.0))

    return PosteriorGuess(broadcast_row=obs.broadcast_row, start=obs.start,
                          candidate=obs.candidate, posterior=posterior.astype(np.float64))
