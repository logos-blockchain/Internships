"""
Bayesian sender-attribution attack: which node in S_t sent an observed broadcast.

For each broadcast (t, S_t, r, y) the residual z_i = y - mu(i, r) with
mu(i, r) = d_i^S + D_M + d_r^R gives the likelihood L_i = f_Z(z_i) (0 for z_i < 0), and
Pr(L=i | y, r, S_t) = L_i / sum_j L_j under a uniform prior. Only the sender-link
differences {d_i^S} separate candidates; the shared D_M + d_r^R cancel.
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


@dataclass(frozen=True)
class BayesAttributionParams:
    """
    Public AC knowledge of the Bayesian attribution attack, filled by run_once from the config.

    hops / mix_scale / sender_scale / receiver_delays -- AC protocol parameters (k, 1/lambda_M,
        1/lambda_S), from which the attack reconstructs the delay density f_Z.
    latency_profile -- the quenched deterministic latencies (network.LatencyProfile) giving mu(i, r).

    All default None; run() raises if any is still None.
    """

    hops: int | None = None
    mix_scale: float | None = None
    sender_scale: float | None = None
    receiver_delays: bool | None = None
    latency_profile: object = None


def run(trace, *, params=None, rng=None):
    """
    Run the Bayesian attribution attack; returns a PosteriorGuess (one posterior per broadcast).

    Uniform attack signature run(trace, *, params, rng); rng is unused (deterministic).
    Residuals and f_Z are evaluated over all candidate rows at once and normalised per
    broadcast with segment sums, so the epoch costs O(total candidates).
    """
    p = params or BayesAttributionParams()
    if p.latency_profile is None or p.hops is None or p.mix_scale is None or p.sender_scale is None:
        raise ValueError(
            "bayes_attribution called unwired: hops / mix_scale / sender_scale / latency_profile "
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

    mu = p.latency_profile.mu(obs.candidate, r_row)
    z = y_row - mu
    eps = float(getattr(p.latency_profile, "jitter_scale", 0.0) or 0.0)
    like = residual_delay_pdf(z, n_stages, float(p.sender_scale), float(p.mix_scale),
                              eps, ac_path_links(int(p.hops)) if eps > 0.0 else 0)

    seg_sum = np.add.reduceat(like, obs.start[:-1]) if like.size else np.zeros(B)
    denom = np.repeat(seg_sum, counts)
    count_row = np.repeat(counts, counts).astype(float)
    posterior = np.where(denom > 0, like / np.where(denom > 0, denom, 1.0), 1.0 / count_row)

    return PosteriorGuess(broadcast_row=obs.broadcast_row, start=obs.start,
                          candidate=obs.candidate, posterior=posterior.astype(np.float64))
