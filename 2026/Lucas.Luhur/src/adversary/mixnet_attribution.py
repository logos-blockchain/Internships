"""
Mix-net sender-attribution attack: bayes_attribution generalised to a W x k grid whose
route is unobserved. The likelihood marginalises over the W^k routes,
L_i = mean_p f_Z(y - mu(i, p, r)) with mu(i, p, r) = d_sender[i, entry_p] + D_int_p +
d_receiver[r, exit_p], and the posterior is L_i / sum_j L_j under a uniform prior.
W = 1 reduces exactly to bayes_attribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline_contract import PosteriorGuess

try:
    from anonymity.single_path_mix import residual_delay_pdf
    from network.jitter import ac_path_links
    from .gpa import observe_broadcasts
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anonymity.single_path_mix import residual_delay_pdf
    from network.jitter import ac_path_links
    from adversary.gpa import observe_broadcasts


CHUNK_ELEMENTS = 33_554_432


@dataclass(frozen=True)
class MixnetAttributionParams:
    """
    Public AC knowledge of the mix-net attribution attack, filled by run_once from the config.

    hops / mix_scale / sender_scale / receiver_delays -- AC protocol parameters (k, 1/lambda_M,
        1/lambda_S), from which the attack reconstructs f_Z.
    latency_profile -- the quenched MixnetLatencyProfile giving mu(i, p, r) per candidate x route.

    All default None; run() raises if any is still None.
    """

    hops: int | None = None
    mix_scale: float | None = None
    sender_scale: float | None = None
    receiver_delays: bool | None = None
    latency_profile: object = None


def run(trace, *, params=None, rng=None):
    """
    Run the mix-net attribution attack; returns a PosteriorGuess (one posterior per broadcast).

    Uniform attack signature run(trace, *, params, rng); rng is unused (deterministic).
    Evaluates f_Z over every candidate x route, averages over routes and normalises per
    broadcast: O(total_candidates * P'), streamed in row chunks of <= CHUNK_ELEMENTS.
    """
    p = params or MixnetAttributionParams()
    if (p.latency_profile is None or p.hops is None or p.mix_scale is None
            or p.sender_scale is None):
        raise ValueError(
            "mixnet_attribution called unwired: hops / mix_scale / sender_scale / latency_profile "
            "must be filled (run_once fills them from the system config via AttackSpec.knows).")
    n_stages = int(p.hops) + (1 if p.receiver_delays else 0)

    obs = observe_broadcasts(trace)
    B = len(obs)
    if B == 0:
        return PosteriorGuess(broadcast_row=obs.broadcast_row, start=obs.start,
                              candidate=obs.candidate, posterior=np.zeros(0))

    counts = np.diff(obs.start)
    r_row = np.repeat(obs.receiver, counts)
    y_row = np.repeat(obs.y, counts)

    eps = float(getattr(p.latency_profile, "jitter_scale", 0.0) or 0.0)
    n_links = ac_path_links(int(p.hops)) if eps > 0.0 else 0
    K = obs.candidate.size
    P = int(p.latency_profile.n_routes_per_sender)
    rows_per_chunk = max(1, CHUNK_ELEMENTS // max(P, 1))
    like = np.empty(K)
    for lo in range(0, K, rows_per_chunk):
        hi = min(lo + rows_per_chunk, K)
        mu = p.latency_profile.route_mu(obs.candidate[lo:hi], r_row[lo:hi])
        z = y_row[lo:hi, None] - mu
        like_routes = residual_delay_pdf(z.ravel(), n_stages, float(p.sender_scale),
                                         float(p.mix_scale), eps, n_links).reshape(z.shape)
        like[lo:hi] = like_routes.mean(axis=1)

    seg_sum = np.add.reduceat(like, obs.start[:-1]) if like.size else np.zeros(B)
    denom = np.repeat(seg_sum, counts)
    count_row = np.repeat(counts, counts).astype(float)
    posterior = np.where(denom > 0, like / np.where(denom > 0, denom, 1.0), 1.0 / count_row)

    return PosteriorGuess(broadcast_row=obs.broadcast_row, start=obs.start,
                          candidate=obs.candidate, posterior=posterior.astype(np.float64))
