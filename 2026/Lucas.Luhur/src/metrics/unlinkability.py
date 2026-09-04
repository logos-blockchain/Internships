"""
Unlinkability measures (family-A): grade a sender-attribution PosteriorGuess against the
Trace ground truth. deanon_top1 (P_Top1, null 1/|S|), mean_true_posterior (P_true) and
posterior_entropy (H, max log2|S|) each average over the B observed broadcasts, using
segment reductions over the flat CSR posterior.
"""

from __future__ import annotations

import numpy as np


def _true_sender_rows(guess, trace):
    """
    Return (mask of the true sender's candidate row [K], per-broadcast candidate counts [B]).

    Exactly one True per broadcast, since the true sender is always in its own S_t.
    """
    counts = np.diff(guess.start)
    true_sender = trace.true_source[guess.broadcast_row]
    true_row = np.repeat(true_sender, counts)
    return guess.candidate == true_row, counts


def deanon_top1(guess, trace):
    """
    Top-1 de-anonymisation probability P_Top1: fraction of broadcasts where MAP == true sender.

    Ties are scored at their expected value under a uniform tie-break (1/M if the true sender
    is among the M tied-max candidates, else 0), so the null is exactly 1/|S| at zero signal.
    """
    if len(guess) == 0:
        return float("nan")
    is_true, counts = _true_sender_rows(guess, trace)
    seg_max = np.maximum.reduceat(guess.posterior, guess.start[:-1])
    is_max = guess.posterior >= np.repeat(seg_max, counts) - 1e-12
    n_ties = np.add.reduceat(is_max.astype(float), guess.start[:-1])
    true_is_max = np.add.reduceat((is_true & is_max).astype(float), guess.start[:-1]) > 0
    return float(np.where(true_is_max, 1.0 / n_ties, 0.0).mean())


def mean_true_posterior(guess, trace):
    """
    Mean posterior mass on the true sender, P_true = (1/B) sum_b Pr(L_b | y_b, r_b, S_t_b).

    ~ 1/|S| means no information; -> 1 means the attack is confident and correct.
    """
    if len(guess) == 0:
        return float("nan")
    is_true, _ = _true_sender_rows(guess, trace)
    true_post = np.add.reduceat(guess.posterior * is_true, guess.start[:-1])
    return float(true_post.mean())


def posterior_entropy(guess, trace):
    """
    Expected posterior entropy H = (1/B) sum_b [ -sum_i P_i log2 P_i ] in bits.

    Max log2|S| (uniform posterior); -> 0 means de-anonymised. Probabilities are floored at
    1e-300 before the log (0 log 0 = 0). Does not use the ground truth.
    """
    if len(guess) == 0:
        return float("nan")
    p = guess.posterior
    plogp = -p * np.log2(np.maximum(p, 1e-300))
    return float(np.add.reduceat(plogp, guess.start[:-1]).mean())
