"""
Stake-inference attack: the GPA estimates a node's relative stake from its
sender-set participation frequency. Pr(i in S_t) = p_s + (1 - p_s) phi(alpha_i),
so inverting the lottery phi(a) = 1 - (1-f)^a gives
alpha_hat_i = log((1 - q_hat_i) / (1 - p_s)) / log(1 - f).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from consensus.election import DEFAULT_F, DEFAULT_T
    from .gpa import observe_sender_sets
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from consensus.election import DEFAULT_F, DEFAULT_T
    from adversary.gpa import observe_sender_sets


def estimate_stake_from_sets(entry_counts, T, p_s, f=DEFAULT_F):
    """
    Estimate relative stake from per-node sender-set counts over T slots.

    q_hat_i = entry_counts[i] / T estimates Pr(i in S_t) = p_s + (1 - p_s) phi(alpha_i),
    so alpha_hat_i = log((1 - q_hat_i) / (1 - p_s)) / log(1 - f). q_hat is clipped to
    [0, 1) and alpha_hat is floored at 0 (below-baseline participation is noise).
    """
    entry_counts = np.asarray(entry_counts, dtype=float)
    if T <= 0:
        return np.zeros_like(entry_counts)
    q = np.clip(entry_counts / T, 0.0, 1.0 - 1e-12)
    alpha_hat = np.log((1.0 - q) / (1.0 - p_s)) / np.log(1.0 - f)
    return np.maximum(alpha_hat, 0.0)


@dataclass(frozen=True)
class SetStakeInferenceParams:
    """
    Parameters of the sender-set stake-inference attack.

    f    -- active-slots coefficient (the lottery constant).
    p_s  -- cover probability; must match the cover model the trace was generated with.
    T    -- observation window in slots (the denominator of q_hat).
    N    -- node universe; None -> inferred from the trace.
    """

    f: float = DEFAULT_F
    p_s: float = 0.01
    T: int = DEFAULT_T
    N: int | None = None


def run_set_stake_inference(trace, *, params=None, rng=None):
    """
    Run the sender-set stake-inference attack; returns per-node alpha_hat (a scalar guess).

    Uniform attack signature run(trace, *, params, rng). Reads the sender set through
    observe_sender_sets and applies estimate_stake_from_sets. A node emits at most once
    per slot, so the entry count equals sum_t 1[i in S_t] exactly.
    """
    p = params or SetStakeInferenceParams()
    if p.N is not None:
        N = p.N
    else:
        ent_nodes = trace.obs_node[trace.is_entry]
        N = int(ent_nodes.max()) + 1 if ent_nodes.size else 0
    counts = observe_sender_sets(trace, N)
    return estimate_stake_from_sets(counts, p.T, p_s=p.p_s, f=p.f)
