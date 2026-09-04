"""
Closed form for deanon_top1: P(the GPA's MAP names the true sender) on the single-path mix
with a continuous link law, the analytic twin of metrics.unlinkability.deanon_top1.

With y = mu(true, r) + Z and mu(i, r) = d_i^S + D_M + d_r^R, the residuals are
z_i = Z + (d_true^S - d_i^S) (D_M and d_r^R cancel). f_Z is unimodal, so its super-level set
{w : f_Z(w) >= f_Z(z)} = [a(z), b(z)] is one interval and rival i beats or ties the truth iff
d_i lands in [z + s - b(z), z + s - a(z)]. With B = P(strictly inside), T = P(at an endpoint)
and G = 1 - B - T, P_Top1 = E_m E_s E_Z[integral_0^1 (G + u T)^(m-1) du]; the tie term makes
the sigma_d -> 0 limit return 1/m. Scope: single-path mix only (the mix-net breaks the
cancellation and the iid product form).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, poisson

try:
    from anonymity.single_path_mix import delay_moments, random_delay_pdf
    from consensus.election import DEFAULT_F
    from network.lognormal_latency import lognormal_from_moments
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anonymity.single_path_mix import delay_moments, random_delay_pdf
    from consensus.election import DEFAULT_F
    from network.lognormal_latency import lognormal_from_moments


def lognormal_link_cdf(params):
    """
    Return (cdf, quantile, atom, is_degenerate) closures for the shifted log-normal link law.

    d = floor + LogNormal(mu, sigma) with (mu, sigma) moment-matched as in
    network.lognormal_latency.lognormal_draw; sd = 0 is a point mass at `mean`.
    """
    floor = float(params.floor)
    mu, sigma = lognormal_from_moments(params.mean - params.floor, params.sd)
    point = floor + float(np.exp(mu))

    def cdf(x):
        x = np.asarray(x, dtype=float)
        if sigma == 0.0:
            return (x >= point).astype(float)
        out = np.zeros_like(x)
        ok = x > floor
        out[ok] = norm.cdf((np.log(x[ok] - floor) - mu) / sigma)
        return out

    def quantile(q):
        q = np.asarray(q, dtype=float)
        if sigma == 0.0:
            return np.full_like(q, point)
        return floor + np.exp(mu + sigma * norm.ppf(q))

    return cdf, quantile, point, sigma == 0.0


def _level_set(z, n_stages, sender_scale, mix_scale, w_max, n_fine=200_001):
    """
    Return [a(z), b(z)] = {w >= 0 : f_Z(w) >= f_Z(z)}, vectorised over z.

    f_Z is unimodal, so each branch is monotone and inverted by interpolation on a fine grid.
    """
    w = np.linspace(0.0, w_max, int(n_fine))
    fw = random_delay_pdf(w, n_stages, sender_scale, mix_scale)
    i_mode = int(np.argmax(fw))
    v = random_delay_pdf(np.asarray(z, dtype=float), n_stages, sender_scale, mix_scale)

    # right branch is decreasing: reverse it so np.interp sees increasing xp
    a = np.interp(v, fw[:i_mode + 1], w[:i_mode + 1], left=0.0, right=w[i_mode])
    rev_f = fw[i_mode:][::-1]
    rev_w = w[i_mode:][::-1]
    b = np.interp(v, rev_f, rev_w, left=w_max, right=w[i_mode])
    return a, b


def candidate_set_law(count, f=DEFAULT_F, tol=1e-15):
    """
    Return the per-broadcast law of |S_t| for the fixed-`count` cover as (sizes, probs).

    Scoring is per broadcast, which size-biases the winner count, so
    |S_t| = count + 1 + Poisson(lambda) with lambda = -ln(1-f).
    """
    lam = -np.log1p(-float(f))
    j = np.arange(int(poisson.isf(tol, lam)) + 1)
    pj = poisson.pmf(j, lam)
    return (int(count) + 1 + j).astype(int), pj / pj.sum()


def deanon_top1(*, hops, mix_scale, sender_scale, receiver_delays, link, count,
                f=DEFAULT_F, n_z=600, n_s=400, n_sd=14.0):
    """
    Evaluate P_Top1 for one configuration by 2-D quadrature; returns a float in [0, 1].

    `link` is a LogNormalParams; hops/mix_scale/sender_scale/receiver_delays are the AC
    parameters, `count` the fixed cover size and f the lottery constant. n_z points over Z
    weighted by f_Z (out to n_sd sd), n_s points over the sender's own link at CDF quantiles.
    The null is E[1/|S_t|].
    """
    n_stages = int(hops) + (1 if receiver_delays else 0)
    cdf, quantile, point, degenerate = lognormal_link_cdf(link)

    e_z, var_z = delay_moments(n_stages, float(sender_scale), float(mix_scale))
    z_hi = e_z + n_sd * np.sqrt(var_z)
    z = np.linspace(z_hi / n_z, z_hi, int(n_z))
    f_z = random_delay_pdf(z, n_stages, float(sender_scale), float(mix_scale))
    w_z = f_z / f_z.sum()

    a, b = _level_set(z, n_stages, float(sender_scale), float(mix_scale), z_hi)
    s = quantile((np.arange(int(n_s)) + 0.5) / int(n_s))

    lo = z[:, None] + s[None, :] - b[:, None]
    hi = z[:, None] + s[None, :] - a[:, None]
    if degenerate:
        tie = (np.isclose(lo, point) | np.isclose(hi, point)).astype(float)
        beat = np.zeros_like(tie)
    else:
        beat = cdf(hi) - cdf(lo)
        tie = np.zeros_like(beat)
    g = np.clip(1.0 - beat - tie, 0.0, 1.0)

    sizes, probs = candidate_set_law(count, f)
    total = 0.0
    for m, p_m in zip(sizes, probs):
        # int_0^1 (G + uT)^(m-1) du = [(G+T)^m - G^m]/(mT), or G^(m-1) if T = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            win = np.where(tie > 0,
                           ((g + tie) ** m - g ** m) / (m * np.where(tie > 0, tie, 1.0)),
                           g ** (m - 1))
        total += float(p_m) * float(w_z @ win.mean(axis=1))
    return float(np.clip(total, 0.0, 1.0))
