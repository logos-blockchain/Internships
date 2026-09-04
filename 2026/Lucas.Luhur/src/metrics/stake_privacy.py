"""
Stake-privacy measures: grade the stake-inference estimate alpha_hat. Provides the
analytic confidence and time-to-link closed forms (functions of the lottery phi alone)
and the empirical family-B measures scored against the true stake.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

try:
    from consensus.election import DEFAULT_F, DEFAULT_T, lottery
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from consensus.election import DEFAULT_F, DEFAULT_T, lottery


def inference_confidence(alpha, gamma, T=DEFAULT_T, f=DEFAULT_F):
    """
    Closed-form P(alpha_hat in [alpha(1 - gamma), alpha(1 + gamma)]) after T observed slots.

    P = 2 erf(eps / sqrt(2 sigma^2)) / [erf(phi / sqrt(2 sigma^2)) + erf((1 - phi) / sqrt(2 sigma^2))]
    with phi = phi(alpha), sigma^2 = phi(1 - phi) / T and eps = gamma alpha phi'(alpha),
    phi'(alpha) = -(1 - f)^alpha log(1 - f). Monotone increasing in alpha and T.
    alpha may be a scalar or array; gamma and T are scalars.
    """
    alpha = np.asarray(alpha, dtype=float)
    phi = lottery(alpha, f)
    dphi = -((1.0 - f) ** alpha) * np.log(1.0 - f)
    eps = gamma * alpha * dphi
    scale = np.sqrt(2.0 * phi * (1.0 - phi) / T)
    num = 2.0 * erf(eps / scale)
    den = erf(phi / scale) + erf((1.0 - phi) / scale)
    return num / den


def time_to_confidence(alpha, gamma, delta, f=DEFAULT_F, t_max=None):
    """
    Minimum number of slots T such that inference_confidence >= delta (time-to-link).

    Confidence is monotone in T, so the crossing is found by bracketing + brentq.
    Returns ceil(T), or +inf if delta is unreachable within t_max (default
    730 * DEFAULT_T). alpha is a scalar.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if t_max is None:
        t_max = 730.0 * DEFAULT_T

    def g(T):
        return float(inference_confidence(alpha, gamma, T, f)) - delta

    if g(1.0) >= 0.0:
        return 1.0
    if g(t_max) < 0.0:
        return float("inf")
    return float(np.ceil(brentq(g, 1.0, t_max)))


def stake_confidence(guess, ctx):
    """
    Empirical confidence: fraction of nodes whose alpha_hat lies in [alpha (1 +/- gamma)].

    The measured counterpart of inference_confidence; guess is the per-node alpha_hat and
    ctx the ScoreContext carrying the true alpha and gamma.
    """
    alpha = np.asarray(ctx.alpha, dtype=float)
    ahat = np.asarray(guess, dtype=float)
    in_band = (ahat >= alpha * (1.0 - ctx.gamma)) & (ahat <= alpha * (1.0 + ctx.gamma))
    return float(in_band.mean())


def stake_top1_hit(guess, ctx):
    """
    Top-staker identification: 1 if argmax alpha_hat is the largest true stakeholder, else 0.

    Reps average this into P(top staker identified), whose null is 1/N. Ties in alpha_hat
    (common: counts are integers and heavy cover floors many alpha_hat at 0) are scored at
    their expected value under a uniform tie-break, 1/|ties| if the whale is among them.
    """
    alpha = np.asarray(ctx.alpha, dtype=float)
    ahat = np.asarray(guess, dtype=float)
    tied = np.flatnonzero(ahat == ahat.max())
    return float(int(np.argmax(alpha) in tied) / tied.size)


def stake_top_jaccard(guess, ctx):
    """
    Jaccard overlap J_x = |Top_hat & Top| / |Top_hat | Top| of the inferred vs true top-x% sets.

    x = ctx.top_frac; J_x = 1 means the whales are perfectly identified, 0 means no overlap.
    """
    alpha = np.asarray(ctx.alpha, dtype=float)
    ahat = np.asarray(guess, dtype=float)
    k = max(1, int(round(ctx.top_frac * alpha.size)))
    true_top = set(np.argsort(alpha)[-k:].tolist())
    hat_top = set(np.argsort(ahat)[-k:].tolist())
    return float(len(true_top & hat_top) / len(true_top | hat_top))


def plot_inference_confidence(out_path=None, gamma=0.1, T=DEFAULT_T,
                              f=DEFAULT_F, alpha_max=0.02, theta=0.5):
    """
    Plot the analytic adversarial confidence against true relative stake alpha.

    The curve rises monotonically in alpha (high-stake nodes are pinned with higher
    confidence); the dashed line marks the confidence threshold theta.
    """
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle

    if out_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        out_path = repo_root / "results" / "figures" / "stage_2_figures" / "metrics_inference_confidence.png"

    alpha = np.linspace(1e-4, alpha_max, 400)

    fig, ax = plt.subplots()
    ax.plot(alpha, inference_confidence(alpha, gamma, T, f),
            color=plt.cm.viridis(0.0), linestyle="-", linewidth=2, label="GPA")

    ax.axhline(theta, color="0.4", linestyle=(0, (1, 1)), linewidth=1.5,
               label=fr"threshold $\theta={theta:g}$")

    ax.set_xlim(0.0, alpha_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Relative stake  $\alpha$")
    ax.set_ylabel(r"Adversarial confidence")
    ax.legend(fontsize=18, title=fr"$\gamma={gamma:g}$", title_fontsize=18)

    return plotstyle.save(fig, out_path)


if __name__ == "__main__":
    saved = plot_inference_confidence()
    print(f"Figure written to {saved}")
