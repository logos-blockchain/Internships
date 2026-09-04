"""
Broadcast-network latency oracle: the annealed link disorder on the quenched graph.

Each edge carries T_e = d + X_e, X_e ~ Exp(lambda), resampled on every broadcast, and the
pairwise delay Delta_ij is the weighted shortest path (Dijkstra). The jitter ratio
rho = E[X_e]/d = 1/(lambda d) is the single dimensionless parameter; rho = 0 is the noise-free
channel T_e = d. Headline scaling: E[Delta] = log N / alpha + O_P(1), with alpha solving
(C - 1) e^{-alpha d} lambda / (lambda + alpha) = 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.special import lambertw

try:
    from .graph import (DEFAULT_C, edge_endpoints, mean_eccentricity_theory,
                        sample_regular_graph)
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from network.graph import (DEFAULT_C, edge_endpoints, mean_eccentricity_theory,
                               sample_regular_graph)

# slot units, 1 slot = 1 s
DEFAULT_D = 0.3
DEFAULT_RHO = 0.0
DEFAULT_LAM = float("inf")


def lam_from_rho(rho, d=DEFAULT_D):
    """
    Exponential rate lambda from the jitter ratio rho = 1/(lambda d)

    rho = 0 (the noise-free channel) -> lam = inf, i.e. T_e = d exactly. Callers that
    branch on the deterministic limit test np.isfinite(lam)
    """
    if rho <= 0.0:
        return float("inf")
    return 1.0 / (rho * d)


def rho_from_lam(lam, d=DEFAULT_D):
    """Jitter ratio rho = 1/(lambda d) from the exponential rate lambda (lam = inf -> rho = 0)"""
    if not np.isfinite(lam):
        return 0.0
    return 1.0 / (lam * d)


def sample_edge_latencies(n_edges, d=DEFAULT_D, lam=DEFAULT_LAM, rng=None):
    """
    One fresh draw of all edge weights T_e = d + Exp(lambda) (annealed)

    Mean-1/lambda jitter on top of the baseline d; resampled per broadcast.
    lam = inf (rho_net = 0) -> zero jitter, so every edge weighs exactly d
    """
    rng = np.random.default_rng(rng)
    scale = 0.0 if not np.isfinite(lam) else 1.0 / lam
    return d + rng.exponential(scale, size=n_edges)


def _weighted_csr(u, v, w, N):
    """Symmetric sparse weighted adjacency from edge endpoints + weights"""
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.concatenate([w, w])
    return csr_matrix((data, (rows, cols)), shape=(N, N))


def broadcast_delays(u, v, N, source, d=DEFAULT_D, lam=DEFAULT_LAM, rng=None):
    """
    One broadcast from source: fresh link latencies, then the weighted
    shortest-path delay to every node (single-source Dijkstra)

    Returns a length-N array; the source entry is 0. This is the core oracle
    call -- the infection process is I_s(t) = {v : delay[v] <= t}
    """
    rng = np.random.default_rng(rng)
    w = sample_edge_latencies(u.size, d, lam, rng)
    csr = _weighted_csr(u, v, w, N)
    return dijkstra(csr, directed=False, indices=source)


def weighted_broadcast_latency(N, C=DEFAULT_C, weight_fn=None, n_sources=64, rng=None):
    """
    Mean global-broadcast latency E[D_br] on a quenched gossip graph with heterogeneous edges.

    The block gossips over a random C-regular graph; each edge carries a weight from the law in
    force, D_br(s) = max_j dist(s, j) is the weighted eccentricity, and E[D_br] is the mean over
    sources. With unequal edges no closed form survives, so Dijkstra is used.
    weight_fn -- callable (u, v, rng) -> [E] per-edge latencies, drawn once per realisation.
    n_sources -- broadcasters sampled for the mean.
    Returns E[D_br] in seconds. Draw order is graph -> weights -> sources.
    """
    rng = np.random.default_rng(rng)
    if weight_fn is None:
        raise ValueError("weighted_broadcast_latency needs a weight_fn (u, v, rng) -> weights")
    G = sample_regular_graph(int(N), int(C), rng=rng)
    u, v = edge_endpoints(G)
    w = np.asarray(weight_fn(u, v, rng), dtype=np.float64)
    if w.shape != u.shape:
        raise ValueError(f"weight_fn returned {w.shape} weights for {u.shape} edges")
    csr = _weighted_csr(u, v, w, int(N))

    n_src = min(int(n_sources), int(N))
    sources = rng.choice(int(N), size=n_src, replace=False)
    dist = dijkstra(csr, directed=False, indices=sources)
    dist[~np.isfinite(dist)] = np.nan
    ecc = np.nanmax(dist, axis=1)
    return float(np.nanmean(ecc))


def sample_pairwise_delays(u, v, N, d=DEFAULT_D, lam=DEFAULT_LAM,
                           n_broadcasts=300, rng=None):
    """
    Pool delays from n_broadcasts independent broadcasts (fresh latencies each,
    random source) -> an empirical sample of the pairwise delay Delta

    The source-to-self zero is dropped, as are any non-finite entries (a stray
    disconnection). Used for the delay distribution F_Delta and its statistics
    """
    rng = np.random.default_rng(rng)
    sources = rng.integers(0, N, size=n_broadcasts)
    out = np.empty((n_broadcasts, N - 1), dtype=float)
    for k, s in enumerate(sources):
        dist = broadcast_delays(u, v, N, int(s), d, lam, rng)
        out[k] = np.delete(dist, s)
    out = out.ravel()
    return out[np.isfinite(out)]


def mean_pairwise_delay(u, v, N, d=DEFAULT_D, lam=DEFAULT_LAM,
                        n_broadcasts=300, rng=None):
    """Empirical mean of the pairwise delay Delta (the quantity that scales as log N / alpha)"""
    return float(np.mean(
        sample_pairwise_delays(u, v, N, d, lam, n_broadcasts, rng)
    ))


def broadcast_completion_quantile(delays, N):
    """
    Near-completion broadcast latency t^b_{1 - 1/N} ~ F_Delta^{-1}(1 - 1/N)

    Time for all but one of N nodes to receive the message -- a concentrated,
    conservative proxy for the full broadcast latency that needs only the
    pairwise delay distribution
    """
    return float(np.quantile(delays, 1.0 - 1.0 / N))


def broadcast_latency(u, v, N, source, d=DEFAULT_D, lam=DEFAULT_LAM, rng=None):
    """
    Global-broadcast latency of one broadcast from source: the source's weighted eccentricity
    L_broadcast(s) = max_{j != s} Delta_sj.

    Non-finite entries (a stray disconnection) are dropped before the max.
    """
    rng = np.random.default_rng(rng)
    dist = broadcast_delays(u, v, N, source, d, lam, rng)
    return float(np.max(dist[np.isfinite(dist)]))


def sample_broadcast_latencies(u, v, N, d=DEFAULT_D, lam=DEFAULT_LAM,
                               n_broadcasts=300, rng=None):
    """
    Pool global-broadcast latencies from n_broadcasts independent broadcasts
    (fresh link latencies each, random source) -> an empirical sample of the
    completion time B_s = max_j Delta_sj

    Its mean E[B_s] sits a straggler-gap above the (1 - 1/N) quantile and is
    bracketed analytically by broadcast_latency_theory
    """
    rng = np.random.default_rng(rng)
    sources = rng.integers(0, N, size=n_broadcasts)
    out = np.empty(n_broadcasts, dtype=float)
    for k, s in enumerate(sources):
        dist = broadcast_delays(u, v, N, int(s), d, lam, rng)
        out[k] = np.max(dist[np.isfinite(dist)])
    return out


def propagation_exponent(C=DEFAULT_C, d=DEFAULT_D, lam=DEFAULT_LAM):
    """
    Solve the branching-process balance for the propagation exponent alpha:

        (C - 1) e^{-alpha d} lambda / (lambda + alpha) = 1.

    Balances the number of non-backtracking length-r paths, (C-1)^r, against the
    chance of an unusually fast path. In the deterministic limit (lambda -> inf)
    it reduces to alpha = log(C - 1) / d. Requires C >= 3 for a positive root
    """
    if C < 3:
        raise ValueError("propagation exponent needs connectivity C >= 3")
    if not np.isfinite(lam):
        return float(np.log(C - 1.0) / d)

    def balance(a):
        return (C - 1.0) * np.exp(-a * d) * lam / (lam + a) - 1.0

    lo, hi = 1e-12, 1.0
    while balance(hi) > 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("failed to bracket the propagation exponent")
    return float(brentq(balance, lo, hi))


def mean_delay_theory(N, C=DEFAULT_C, d=DEFAULT_D, lam=DEFAULT_LAM):
    """
    Headline closed form E[Delta] ~ log N / alpha (leading order)

    A few percent accurate for C >= 7; for low connectivity (C = 4, 5) it
    overestimates the mean by ~5-10%
    """
    return np.log(N) / propagation_exponent(C, d, lam)


def _broadcast_mgf_constant(s, C):
    """
    Finite-N constant H_C(s) collecting the O(1) (N-independent) part of the
    bounded graph-distance MGF in the closed-form broadcast bound:

        log M_L^ub(s) ~ (s / log(C-1)) log N + H_C(s),

    with A(s, q) a geometric tail bound on the MGF sum (q = C - 1)
    """
    q = C - 1.0
    logq = np.log(q)
    A = (1.0
         + 1.0 / (1.0 - np.exp(-s * (q - 1.0)))
         + 1.0 / (1.0 - np.exp(-s * (q - 1.0) / q)))
    log_em1 = s + np.log1p(-np.exp(-s))           # log(e^s - 1), stable for large s
    return log_em1 + np.log(A) + (s / logq) * (np.log(s * (C - 2.0) / (C * logq)) - 1.0)


def broadcast_latency_theory(N, C=DEFAULT_C, d=DEFAULT_D, lam=DEFAULT_LAM):
    """
    Closed-form conservative approximation t_N^cf for the (1 - 1/N) quantile of the pairwise delay.

    Derived in the low-jitter regime rho << 1 from a Chernoff/MGF bound under the
    Gompertz-distance/Gamma law, minimised via the -1 branch of the Lambert-W function:

        u*     = 1 + 1 / W_{-1}(-1 / (e (C-1))),    theta* = lam u*,
        s*     = lam d u* - log(1 - u*),
        t_N^cf = [ (1 + s*/log(C-1)) log N + H_C(s*) ] / theta*.

    Overestimates the simulated quantile by ~5-20% at moderate N; needs C >= 3. At rho = 0
    (lam = inf) the form diverges, so the exact d * E[ecc] (graph.mean_eccentricity_theory) is
    returned instead.
    """
    if C < 3:
        raise ValueError("broadcast latency theory needs connectivity C >= 3")
    if not np.isfinite(lam):
        return float(d * mean_eccentricity_theory(N, C))

    q = C - 1.0
    w = float(np.real(lambertw(-1.0 / (np.e * q), k=-1)))
    u_star = 1.0 + 1.0 / w
    theta_star = lam * u_star
    s_star = lam * d * u_star - np.log(1.0 - u_star)
    return ((1.0 + s_star / np.log(q)) * np.log(N)
            + _broadcast_mgf_constant(s_star, C)) / theta_star


def plot_propagation_exponent(out_path=None, C_values=range(4, 11),
                              rho_values=(0.1, 1.0, 10.0)):
    """
    Plot the delay coefficient 1/alpha (mean delay per unit log N) against
    connectivity C, for a family of jitter ratios rho

    Illustrates the headline scaling E[Delta] ~ (1/alpha) log N: the coefficient
    falls as C grows (more, shorter paths -> faster broadcast) and rises with rho
    (heavier jitter slows propagation). The dashed reference is the deterministic
    limit 1/alpha = d / log(C - 1) (rho -> 0)
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
        out_path = repo_root / "results" / "figures" / "stage_2_figures" / "network_propagation_exponent.png"

    C_values = np.asarray(list(C_values))
    styles = ["-", "--", "-.", ":"]
    cmap = plt.cm.viridis

    fig, ax = plt.subplots()
    for i, rho in enumerate(rho_values):
        lam = lam_from_rho(rho)
        coeff = [1.0 / propagation_exponent(int(C), lam=lam) for C in C_values]
        ax.plot(C_values, coeff, color=cmap(i / max(len(rho_values) - 1, 1)),
                linestyle=styles[i % len(styles)], linewidth=2, marker="o",
                label=fr"$\rho={rho:g}$")

    det = DEFAULT_D / np.log(C_values - 1.0)
    ax.plot(C_values, det, color="0.4", linestyle=(0, (1, 1)), linewidth=1.5,
            label=r"det. limit $d/\log(C-1)$")

    ax.set_xlabel(r"Connectivity  $C$")
    ax.set_ylabel(r"Delay coefficient  $1/\alpha$")
    ax.legend(fontsize=18)

    return plotstyle.save(fig, out_path)


if __name__ == "__main__":
    saved = plot_propagation_exponent()
    print(f"Figure written to {saved}")
