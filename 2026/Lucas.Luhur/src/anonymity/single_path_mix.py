"""
The single_path_mix layer: a candidate batch is routed down one shared k-hop mix path.
The GPA sees an ENTRY (source, slot t) and an exit (random destination, time
y = t + mu(i, r) + Z) per message, with Z = X_S + sum_{j=1}^{k} X_{M,j} the intentional
delay (X_S ~ Exp(lambda_S), X_{M,j} ~ Exp(lambda_M)) and mu the deterministic latency
from the latency oracle. Also provides the closed-form delay densities the attack evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, lgamma

import numpy as np

from .trace import BROADCAST, ENTRY, EXIT, make_trace

try:
    from network.jitter import ac_path_links
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from network.jitter import ac_path_links


@dataclass(frozen=True)
class SinglePathMixParams:
    """
    Parameters of the single-path mix layer.

    hops            -- k, relay hops on the shared path; the k mixer holds sum to Gamma(k, mix_scale)
    mix_scale       -- mean per-hop mixing delay 1/lambda_M (slot units), the swept delay dial
    n_nodes         -- size of the node universe destinations are drawn from; None -> from the data
    sender_scale    -- mean sender hold 1/lambda_S (slot units), or "auto" = mix_scale / rho so the
                       ratio holds while mix_scale is swept
    rho             -- target ratio lambda_S/lambda_M when sender_scale = "auto" (default 4,
                       Das et al., PoPETs 2024); ignored when sender_scale is a number
    receiver_delays -- if True the receiver also mixes (k+1 stages); default False (k stages)
    sender_auto     -- derived flag: True once sender_scale was given as "auto"
    """

    hops: int = 3
    mix_scale: float = 1.0
    n_nodes: int | None = None
    sender_scale: float | str = 0.25
    receiver_delays: bool = False
    rho: float = 4.0
    sender_auto: bool = False

    def __post_init__(self):
        """
        Resolve sender_scale = "auto" to mix_scale / rho.

        A fixed sender hold would floor sigma_Z = sqrt(sender^2 + k*mix^2) when mix_scale
        is swept down; with "auto", sigma_Z = mix_scale*sqrt(k + 1/rho^2) with no floor.
        """
        auto = self.sender_auto or isinstance(self.sender_scale, str)
        if isinstance(self.sender_scale, str) and self.sender_scale != "auto":
            raise ValueError(
                f"sender_scale must be a number or the string 'auto' (= mix_scale/rho), "
                f"got {self.sender_scale!r}")
        if auto:
            if not (self.rho > 0.0):
                raise ValueError(f"sender_scale='auto' needs rho > 0, got {self.rho!r}")
            object.__setattr__(self, "sender_auto", True)
            object.__setattr__(self, "sender_scale", float(self.mix_scale) / float(self.rho))


def apply(slots, nodes, is_dummy=None, group=None, *, params=None, latency_oracle=None, rng=None):
    """
    Route each candidate batch down a shared path with k-hop mixing delay.

    Inputs are the emission stream from inject_dummies: (slots, nodes) plus the
    per-emission is_dummy tag and group label. latency_oracle is a LatencyProfile
    supplying mu(i, r) on the exit (None -> mu = 0). Returns a Trace with 2*M rows.
    """
    rng = np.random.default_rng(rng)
    params = params or SinglePathMixParams()
    slots = np.asarray(slots, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)
    m = slots.size
    dummy = np.zeros(m, dtype=bool) if is_dummy is None else np.asarray(is_dummy, dtype=bool)
    grp = np.arange(m, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64)

    k = int(params.hops)
    if k < 1:
        raise ValueError(f"single_path_mix needs hops >= 1, got {k}")

    entry_time = slots.astype(np.float64)

    n_stages = k + 1 if params.receiver_delays else k
    Z = rng.gamma(shape=n_stages, scale=params.mix_scale, size=m)
    if params.sender_scale > 0.0:
        Z = Z + rng.exponential(params.sender_scale, size=m)

    N = params.n_nodes if params.n_nodes is not None else (int(nodes.max()) + 1 if m else 1)
    exit_node = rng.integers(0, N, size=m) if m else np.empty(0, dtype=np.int64)

    mu = latency_oracle.mu(nodes, exit_node) if latency_oracle is not None else 0.0
    exit_time = entry_time + mu + Z

    jitter_scale = float(getattr(latency_oracle, "jitter_scale", 0.0) or 0.0)
    if jitter_scale > 0.0 and m:
        exit_time = exit_time + rng.gamma(shape=ac_path_links(k), scale=jitter_scale, size=m)

    exit_kind = np.where(dummy, EXIT, BROADCAST)
    return make_trace(
        broadcast_id=np.concatenate([grp, grp]),
        true_source=np.concatenate([nodes, nodes]),
        obs_node=np.concatenate([nodes, exit_node]),
        obs_time=np.concatenate([entry_time, exit_time]),
        kind=np.concatenate([np.full(m, ENTRY), exit_kind]),
        is_dummy=np.concatenate([dummy, dummy]),
    )


def delay_moments(n_stages, sender_scale, mix_scale):
    """
    Return (mean, variance) of the intentional delay Z = X_S + sum_{j=1}^{n_stages} X_{M,j}.

    E[Z] = sender_scale + n_stages * mix_scale,
    Var(Z) = sender_scale^2 + n_stages * mix_scale^2.
    """
    mean = sender_scale + n_stages * mix_scale
    var = sender_scale ** 2 + n_stages * mix_scale ** 2
    return float(mean), float(var)


def random_delay_pdf(z, n_stages, sender_scale, mix_scale, rate_tol=1e-8):
    """
    Closed-form density f_Z of the intentional delay Z = X_S + Gamma(n_stages, mix_scale).

    With m := n_stages, lambda_S = 1/sender_scale, lambda_M = 1/mix_scale:
        f_Z(z) = A_S [ e^{-lambda_S z} - e^{-lambda_M z} sum_{q=0}^{m-1} ((lambda_M-lambda_S) z)^q / q! ],
        A_S = lambda_S lambda_M^m / (lambda_M - lambda_S)^m,  z >= 0,
    collapsing to Gamma(m+1, lambda) at equal rates and to Gamma(m, lambda_M) when
    sender_scale = 0. Vectorised over z; 0 for z < 0.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)
    pos = z >= 0.0
    if not np.any(pos):
        return out
    zz = z[pos]
    m = int(n_stages)
    lam_m = 1.0 / mix_scale

    if sender_scale <= 0.0:
        out[pos] = lam_m ** m * zz ** (m - 1) * np.exp(-lam_m * zz) / factorial(m - 1)
        return out

    lam_s = 1.0 / sender_scale
    if abs(lam_s - lam_m) <= rate_tol * max(lam_s, lam_m):
        out[pos] = lam_m ** (m + 1) * zz ** m * np.exp(-lam_m * zz) / factorial(m)
        return out

    a_s = lam_s * lam_m ** m / (lam_m - lam_s) ** m
    poly = np.zeros_like(zz)
    for q in range(m):
        poly = poly + ((lam_m - lam_s) * zz) ** q / factorial(q)
    out[pos] = np.maximum(a_s * (np.exp(-lam_s * zz) - np.exp(-lam_m * zz) * poly), 0.0)
    return out


_PF_MIN_GAP = 0.2
_SERIES_TOL = 1e-12
_SERIES_MAX_TERMS = 8192


def _merge_rates(blocks, rate_tol):
    """[(n, scale)] -> [(n, rate)] sorted by rate, with rates equal to `rate_tol` merged."""
    live = [(int(n), float(s)) for n, s in blocks if int(n) > 0 and float(s) > 0.0]
    if not live:
        raise ValueError(
            "gamma_sum_pdf got no non-degenerate blocks -- every scale is 0, so the sum is the "
            "point mass at 0 and has no density. The caller must handle that case.")
    rates = sorted(((1.0 / s, n) for n, s in live), key=lambda t: t[0])
    merged = [[rates[0][0], rates[0][1]]]
    for lam, n in rates[1:]:
        if abs(lam - merged[-1][0]) <= rate_tol * max(lam, merged[-1][0]):
            merged[-1][1] += n
        else:
            merged.append([lam, n])
    return merged


def _gamma_sum_pdf_partial_fractions(zz, merged):
    """
    Generalised hypoexponential density by partial fractions (well separated rates only).

    With lambda_j = 1/scale_j:
        f(z) = sum_j sum_{i=1}^{n_j} A_{j,i} z^{i-1} e^{-lambda_j z} / (i-1)!,
        A_{j,i} = C * G_j^{(n_j-i)}(-lambda_j) / (n_j-i)!,
        C = prod_j lambda_j^{n_j},   G_j(s) = prod_{l != j} (lambda_l + s)^{-n_l},
    the derivatives of G_j obtained from those of its log by the recursion
    G^{(p)} = sum_{q<p} C(p-1,q) h^{(p-q)} G^{(q)}.
    """
    lams = np.array([m[0] for m in merged], dtype=np.float64)
    mult = [int(m[1]) for m in merged]
    log_c = float(np.sum([n * np.log(lam) for lam, n in zip(lams, mult)]))
    acc = np.zeros_like(zz)
    for j in range(len(merged)):
        lam_j, n_j = lams[j], mult[j]
        gaps = np.array([lams[l] - lam_j for l in range(len(merged)) if l != j])
        n_l = np.array([mult[l] for l in range(len(merged)) if l != j], dtype=np.float64)

        g = np.empty(n_j)
        g[0] = float(np.exp(-np.sum(n_l * np.log(np.abs(gaps)))) * np.prod(np.sign(gaps) ** n_l))
        for p in range(1, n_j):
            val = 0.0
            for q in range(1, p + 1):
                h_q = -float(np.sum(n_l * (-1.0) ** (q - 1) * factorial(q - 1) / gaps ** q))
                val += (factorial(p - 1) / (factorial(q - 1) * factorial(p - q))) * h_q * g[p - q]
            g[p] = val

        for i in range(1, n_j + 1):
            a = np.exp(log_c) * g[n_j - i] / factorial(n_j - i)
            acc = acc + a * zz ** (i - 1) * np.exp(-lam_j * zz) / factorial(i - 1)
    return acc


def _gamma_sum_pdf_series(zz, merged, tol=_SERIES_TOL, max_terms=_SERIES_MAX_TERMS):
    """
    Generalised hypoexponential density as a positive Gamma mixture (Moschopoulos 1985).

    With beta = min_j scale_j, rho = sum_j n_j, u_j = 1 - beta/scale_j:
        f(z) = C sum_{k>=0} delta_k * GammaPdf(z; shape = rho + k, scale = beta),
        C = prod_j (beta/scale_j)^{n_j},  delta_0 = 1,
        delta_{k+1} = (1/(k+1)) sum_{i=1}^{k+1} [sum_j n_j u_j^i] delta_{k+1-i}.
    All terms are positive; truncation stops once C*sum_{k<=K} delta_k exceeds 1 - tol.
    """
    rho = float(sum(int(m[1]) for m in merged))
    scales = np.array([1.0 / m[0] for m in merged], dtype=np.float64)
    mult = np.array([int(m[1]) for m in merged], dtype=np.float64)
    beta = float(scales.min())
    u = 1.0 - beta / scales
    log_c = float(np.sum(mult * np.log(beta / scales)))

    # g_0 = Gamma(rho, beta) density in logs: rho can be large, so lgamma not factorial
    with np.errstate(divide="ignore"):
        log_g0 = ((rho - 1.0) * np.log(np.where(zz > 0, zz, 1.0)) - zz / beta
                  - lgamma(rho) - rho * np.log(beta))
    g = np.exp(log_g0)
    g = np.where(zz > 0, g, 0.0) if rho > 1.0 else g         # z = 0 is a zero of z^{rho-1}

    delta = [1.0]
    acc = np.exp(log_c) * g
    mass = np.exp(log_c)
    k = 0
    while mass < 1.0 - tol and k < max_terms:
        s = 0.0
        for i in range(1, k + 2):
            s += float(np.sum(mult * u ** i)) * delta[k + 1 - i]
        delta.append(s / (k + 1))
        g = g * zz / (beta * (rho + k))   # Gamma pdf ratio: shape rho+k+1 over rho+k
        acc = acc + np.exp(log_c) * delta[-1] * g
        mass += np.exp(log_c) * delta[-1]
        k += 1
    return acc


def gamma_sum_pdf(z, blocks, rate_tol=1e-9):
    """
    Density of sum_j Gamma(n_j, scale_j) for blocks = [(n_1, scale_1), ...] with integer shapes.

    Blocks with n_j = 0 or scale_j <= 0 are dropped; rates equal to `rate_tol` (relative)
    are merged. Partial fractions are used when every pairwise rate gap is >= _PF_MIN_GAP
    (they cancel catastrophically near ties), the Moschopoulos series otherwise; raises if
    neither is accurate. Returns f(z), 0 for z < 0, vectorised over z.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)
    merged = _merge_rates(blocks, rate_tol)

    pos = z >= 0.0
    if not np.any(pos):
        return out
    zz = z[pos]

    if len(merged) == 1:
        lam, n = merged[0][0], int(merged[0][1])
        out[pos] = lam ** n * zz ** (n - 1) * np.exp(-lam * zz) / factorial(n - 1)
        return out

    lams = np.array([m[0] for m in merged], dtype=np.float64)
    min_gap = float(np.min(np.diff(lams) / lams[1:]))
    scales = 1.0 / lams
    u_max = 1.0 - float(scales.min() / scales.max())
    terms_needed = (np.inf if u_max >= 1.0
                    else np.log(_SERIES_TOL) / np.log(max(u_max, 1e-300)))

    if min_gap >= _PF_MIN_GAP:
        acc = _gamma_sum_pdf_partial_fractions(zz, merged)
    elif terms_needed <= _SERIES_MAX_TERMS:
        acc = _gamma_sum_pdf_series(zz, merged)
    else:
        raise ValueError(
            f"gamma_sum_pdf cannot evaluate this rate set accurately: smallest relative rate gap "
            f"{min_gap:.2e} (< {_PF_MIN_GAP} -> partial fractions cancel) while the series would "
            f"need ~{terms_needed:.0f} terms (> {_SERIES_MAX_TERMS}). That needs a near-tie AND a "
            f"wide scale range at once, which the shipped parameterisation cannot produce; "
            f"rates = {lams.tolist()}.")

    out[pos] = np.maximum(acc, 0.0)
    return out


def residual_delay_pdf(z, n_stages, sender_scale, mix_scale, jitter_scale=0.0, n_links=0,
                       rate_tol=1e-6):
    """
    Density f_{Z+E} of the residual: intentional delay Z plus per-message link jitter E.

    Z = X_S + sum_{j<=n_stages} X_{M,j}, E ~ Gamma(n_links, jitter_scale). With jitter_scale = 0
    or n_links = 0 this delegates to random_delay_pdf. Vectorised over z; 0 for z < 0.
    """
    if jitter_scale <= 0.0 or int(n_links) <= 0:
        return random_delay_pdf(z, n_stages, sender_scale, mix_scale)

    blocks = [(int(n_stages), float(mix_scale)), (int(n_links), float(jitter_scale))]
    if sender_scale > 0.0:
        blocks.append((1, float(sender_scale)))
    return gamma_sum_pdf(z, blocks, rate_tol=rate_tol)


def residual_moments(n_stages, sender_scale, mix_scale, jitter_scale=0.0, n_links=0):
    """
    Return (mean, variance) of the residual Z + E.

    E[Z+E] = E[Z] + n_links*jitter_scale, Var = Var(Z) + n_links*jitter_scale^2.
    """
    mean_z, var_z = delay_moments(n_stages, sender_scale, mix_scale)
    n = max(int(n_links), 0)
    s = max(float(jitter_scale), 0.0)
    return float(mean_z + n * s), float(var_z + n * s * s)
