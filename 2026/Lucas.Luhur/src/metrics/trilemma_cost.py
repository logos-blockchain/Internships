"""
Trilemma cost axes: bandwidth overhead beta = |S|/L (driven by the cover p_s) and
latency (driven by the k hops, the per-hop delays and the link latencies). Both are
properties of the Trace and config rather than of the adversary's guess, so run_once
computes them directly; they are not in the MEASURES registry.
"""

from __future__ import annotations

import numpy as np


def bandwidth_overhead(trace):
    """
    Bandwidth overhead beta = (AC messages) / (genuine block proposals).

    beta = n_emissions / n_genuine -> 1 + p_s (N/Phi - 1) as T -> inf, with Phi = E[L_t]
    leaders per slot. Independent of the hop count k (counts messages entering the AC
    system). Reads is_dummy (scoring side). nan if no genuine proposal is present.
    """
    entry = trace.is_entry
    n_emissions = int(entry.sum())
    n_genuine = int((entry & ~trace.is_dummy).sum())
    return n_emissions / n_genuine if n_genuine else float("nan")


def mean_latency(trace):
    """
    Mean AC-path latency = mean(exit_time) - mean(entry_time) over emissions.

    This is the AC-path leg only (mu + Z, or mu + Z + E with jitter), not E[D^AC], which
    also includes the broadcast term E[D_br(N)]. With entry = t and exit = t + mu + Z,
    mean(exit - entry) = mean(mu) + 1/lambda_S + (k or k+1)/lambda_M, where
    mu = d_i^S + D_M + d_r^R from the latency profile (mu = 0 with latency_oracle=None).
    """
    ex = trace.obs_time[trace.is_exit]
    en = trace.obs_time[trace.is_entry]
    if ex.size == 0:
        return 0.0
    return float(ex.mean() - en.mean())


def latency_overhead(trace, *, broadcast_mean):
    """
    Latency overhead ell = E[D^AC] / E[D^base] = 1 + mean_latency(trace) / E[D_br(N)].

    The numerator is the measured AC-path delay (links + sender hold + k-stage mixing);
    the denominator broadcast_mean is the baseline block-broadcast latency from the network
    model. Dimensionless; nan if broadcast_mean is not positive.
    """
    if broadcast_mean <= 0:
        return float("nan")
    return 1.0 + mean_latency(trace) / broadcast_mean
