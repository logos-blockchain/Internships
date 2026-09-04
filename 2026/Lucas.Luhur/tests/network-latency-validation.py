"""
Network-latency validation: simulated broadcast delays (src/network) against closed forms.

Checks: the deterministic-limit propagation exponent alpha = log(C-1)/d; the Gompertz
graph-distance mean (Tishby et al. 2022); logarithmic delay scaling E[Delta] ~ log N / alpha;
the global-broadcast (1 - 1/N) quantile against the Lambert-W bound t_N^cf; the noise-free
channel rho_net = 0 with E[D_br] = d * E[ecc]; and the AC-path inter-mix chain D_M as an
Irwin-Hall sum of k-1 uniform links. Also saves the delay-distribution, scaling,
graph-distance, broadcast-latency and topology (BFS shell) figures.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import networkx as nx             # noqa: E402
import numpy as np
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402
from scipy.stats import kstest, triang  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotstyle  # noqa: E402
from network import (  # noqa: E402
    DEFAULT_C,
    DEFAULT_D,
    broadcast_completion_quantile,
    broadcast_latency_theory,
    edge_endpoints,
    graph_distance_survival,
    lam_from_rho,
    mean_delay_theory,
    mean_eccentricity_theory,
    mean_graph_distance_theory,
    mean_pairwise_delay,
    propagation_exponent,
    sample_broadcast_latencies,
    sample_edge_latencies,
    sample_pairwise_delays,
    sample_regular_graph,
)
from network.latency import _weighted_csr  # noqa: E402
from network.latency_profile import (  # noqa: E402
    LatencyProfileParams,
    sample_latency_profile,
)

REL_TOL_EXP = 1e-3
REL_TOL_DIST = 0.03
REL_TOL_SLOPE = 0.05
R2_MIN = 0.99
REL_TOL_BCAST_LO = 0.10
REL_TOL_BCAST_HI = 0.40
R2_MIN_BCAST = 0.95
REL_TOL_ECC = 0.05


def _fmt_row(name, theo, emp, extra=""):
    """Format one theory-vs-simulation result row."""
    return f"  {name:<36} theory={theo:<12.6g} sim={emp:<12.6g} {extra}"


def _print_ecc_tail_sum(N, C):
    """
    Print the E[ecc] tail sum term by term.

    E[ecc] = sum_{l>=0} Pr(ecc > l) = sum_l [1 - (1 - Pr(L>l))^(N-1)], with Pr(L>l) the
    Gompertz hop-distance survival.
    """
    print(f"  E[ecc] tail-sum  sum_l [1 - (1 - Pr(L>l))^(N-1)]   (N-1 = {N - 1} other nodes):")
    print(f"    {'l':>2} | {'Pr(L>l)':>9} | {'(1-Pr)^(N-1)':>12} | {'Pr(ecc>l)':>10}")
    print(f"    {'-' * 2}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 10}")
    total, l = 0.0, 0
    while True:
        S = float(graph_distance_survival(l, N, C))
        E = 1.0 - (1.0 - S) ** (N - 1)
        total += E
        print(f"    {l:>2} | {S:>9.5f} | {(1.0 - S) ** (N - 1):>12.5f} | {E:>10.6f}")
        if E < 1e-6 and l >= 1:
            break
        l += 1
    print(f"    {'-' * 2}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 10}")
    print(f"    E[ecc] = {total:.6f} hops  (Pr(ecc>l) = 1 up to l=4, 0 after -> a.s. exactly 5)")
    return total


def check_propagation_exponent(C, d):
    """Check alpha(C, lambda -> inf) == log(C - 1) / d, the deterministic limit."""
    print("\n[1. Propagation exponent -- deterministic limit]")
    lam_huge = lam_from_rho(1e-7, d)
    alpha = propagation_exponent(C, d=d, lam=lam_huge)
    alpha_det = np.log(C - 1.0) / d
    rel = abs(alpha - alpha_det) / alpha_det
    passed = rel < REL_TOL_EXP
    print(_fmt_row("alpha  [== log(C-1)/d]", alpha_det, alpha,
                   f"rel={rel:.2e} {'PASS' if passed else 'FAIL'}"))
    return passed


def check_graph_distance(u, v, N, C, rng):
    """Check the mean unweighted shortest path on the quenched RRG against the Gompertz mean."""
    print("\n[2. Graph-distance mean -- Gompertz law (Tishby et al.)]")
    adj = _weighted_csr(u, v, np.ones(u.size), N)
    D = shortest_path(adj, method="D", directed=False, unweighted=True)
    iu = np.triu_indices(N, k=1)
    mean_emp = float(np.mean(D[iu]))
    mean_theo = mean_graph_distance_theory(N, C)
    rel = abs(mean_emp - mean_theo) / mean_theo
    passed = rel < REL_TOL_DIST
    print(_fmt_row("E[L]  [Tishby]", mean_theo, mean_emp,
                   f"rel={rel:.2%} {'PASS' if passed else 'FAIL'}"))
    return passed, D[iu]


def check_noise_free_limit(C, d, N, n_broadcasts, rng):
    """
    Check the noise-free channel rho_net = 0 (T_e = d exactly), the AC-base operating point.

    Every edge weighs exactly d, the broadcast latency is d * eccentricity, and
    broadcast_latency_theory takes the exact branch d * E[ecc] rather than the
    Lambert-W bound, which diverges as lam -> inf.
    """
    print("\n[5. Noise-free channel rho_net = 0 -- T_e = d, E[D_br] = d * E[ecc]]")
    lam = lam_from_rho(0.0, d)
    if np.isfinite(lam):
        print(f"  lam_from_rho(0) = {lam} -- expected inf   FAIL")
        return False

    w = sample_edge_latencies(1000, d=d, lam=lam, rng=rng)
    edges_ok = bool(np.all(w == d))

    G = sample_regular_graph(N, C, rng=rng)
    u, v = edge_endpoints(G)
    B = sample_broadcast_latencies(u, v, N, d=d, lam=lam, n_broadcasts=n_broadcasts, rng=rng)
    hops = B / d
    integral_ok = bool(np.all(np.abs(hops - np.round(hops)) < 1e-9))
    emp = float(B.mean())
    theo = broadcast_latency_theory(N, C, d, lam)
    _print_ecc_tail_sum(N, C)
    ecc_theo = mean_eccentricity_theory(N, C)
    rel = abs(emp - theo) / theo
    limit_ok = rel < REL_TOL_ECC

    print(f"  edge weights all == d ({d:g})                     {'PASS' if edges_ok else 'FAIL'}")
    print(f"  broadcast latency is an exact multiple of d      {'PASS' if integral_ok else 'FAIL'}")
    print(_fmt_row("E[D_br]  [== d * E[ecc]]", theo, emp,
                   f"rel={rel:.2%} (E[ecc]={ecc_theo:.3f} hops) {'PASS' if limit_ok else 'FAIL'}"))
    return edges_ok and integral_ok and limit_ok


def check_ac_mix_chain(d, reps, seed, low=0.0, high=6.22):
    """
    Check the AC-path inter-mix chain D_M under the per-link uniform law.

    D_M = sum_{j=1}^{k-1} U_j with U_j ~ U(low, high) i.i.d., so it is Irwin-Hall:
    E[D_M] = (k-1)(low+high)/2, Var = (k-1)(high-low)^2/12, and the k = 3 sum is
    triangular on [2*low, 2*high]. Also checks that d_sender/d_receiver stay bit-identical
    across the scalar and per-link forms (draw order) and that competing knobs are rejected.
    """
    print("\n[6. AC-path inter-mix chain D_M -- per-link uniform draws (Irwin-Hall)]")
    ok = True

    def totals(k):
        lp = LatencyProfileParams(mix_low=low, mix_high=high)
        return np.array([sample_latency_profile(4, k, d, lp, rng=np.random.default_rng(seed + r)).d_mix
                         for r in range(reps)])

    for k in (2, 3, 4, 5):
        n_links, dm = k - 1, totals(k)
        mean_t, sd_t = n_links * (low + high) / 2, (high - low) * np.sqrt(n_links / 12)
        sem = sd_t / np.sqrt(reps)
        row_ok = abs(dm.mean() - mean_t) < 4 * sem and abs(dm.std(ddof=1) / sd_t - 1) < 0.05
        ok &= row_ok
        print(_fmt_row(f"E[D_M]  k={k} ({n_links} links)", mean_t, dm.mean(),
                       f"{abs(dm.mean() - mean_t) / sem:.2f} sem, sd={dm.std(ddof=1):.4f} "
                       f"(theory {sd_t:.4f}) {'PASS' if row_ok else 'FAIL'}"))

    dm3 = totals(3)
    tri = triang(c=0.5, loc=2 * low, scale=2 * (high - low))
    p_link = float(kstest(dm3, tri.cdf).pvalue)
    one = np.random.default_rng(seed).uniform(low, high, size=reps)
    p_one = float(kstest(one, tri.cdf).pvalue)
    shape_ok, ctrl_ok = p_link > 0.01, p_one < 0.01
    ok &= shape_ok and ctrl_ok
    print(f"  k=3 D_M ~ Triangular(0, 2*high) (KS p={p_link:.3f})       {'PASS' if shape_ok else 'FAIL'}")
    print(f"  negative control: ONE whole-chain draw rejected (p={p_one:.1e})  {'PASS' if ctrl_ok else 'FAIL'}")

    N, k = 500, 3
    common = dict(sender_low=low, sender_high=high, receiver_low=low, receiver_high=high)
    scalar = sample_latency_profile(N, k, d, LatencyProfileParams(**common),
                                    rng=np.random.default_rng(seed))
    drawn = sample_latency_profile(N, k, d, LatencyProfileParams(**common, mix_low=low, mix_high=high),
                                   rng=np.random.default_rng(seed))
    order_ok = (np.array_equal(scalar.d_sender, drawn.d_sender)
                and np.array_equal(scalar.d_receiver, drawn.d_receiver)
                and scalar.d_mix != drawn.d_mix and abs(scalar.d_mix - (k - 1) * d) < 1e-12)
    ok &= order_ok
    print(f"  d_sender/d_receiver BIT-identical, only D_M moves        {'PASS' if order_ok else 'FAIL'}")
    print(f"    sigma_d = {scalar.d_sender.std(ddof=1):.4f}s both;  "
          f"D_M {scalar.d_mix:.4f}s (scalar) -> {drawn.d_mix:.4f}s (per-link)")

    def rejects(lp):
        try:
            sample_latency_profile(10, 3, d, lp, rng=np.random.default_rng(0))
            return False
        except ValueError:
            return True

    wiring_ok = (rejects(LatencyProfileParams(mix_total=1.0, mix_low=low, mix_high=high))
                 and abs(sample_latency_profile(50, 3, d, LatencyProfileParams(),
                                                rng=np.random.default_rng(4)).d_mix - 2 * d) < 1e-12
                 and abs(sample_latency_profile(50, 3, d, LatencyProfileParams(mix_total=1.25),
                                                rng=np.random.default_rng(4)).d_mix - 1.25) < 1e-12
                 and abs(sample_latency_profile(10, 1, d, LatencyProfileParams(mix_low=low, mix_high=high),
                                                rng=np.random.default_rng(0)).d_mix) < 1e-12)
    ok &= wiring_ok
    print(f"  wiring: mix_total|mix_low/high exclusive,")
    print(f"          unset -> (k-1)d, mix_total verbatim, k=1 -> 0     {'PASS' if wiring_ok else 'FAIL'}")
    return ok


def check_logn_scaling(C, d, rho, Ns, n_broadcasts, base_seed):
    """
    Check E[Delta] = a + b log N with slope b ~ 1/alpha and a near-perfect linear fit.

    Returns (passed, Ns, means, alpha) for the figure.
    """
    print(f"\n[3. Logarithmic delay scaling -- C={C}, rho={rho:g}]")
    lam = lam_from_rho(rho, d)
    alpha = propagation_exponent(C, d=d, lam=lam)
    means = []
    for j, N in enumerate(Ns):
        rng = np.random.default_rng(base_seed + 200 + j)
        G = sample_regular_graph(N, C, rng=rng)
        u, v = edge_endpoints(G)
        m = mean_pairwise_delay(u, v, N, d=d, lam=lam,
                                n_broadcasts=n_broadcasts, rng=rng)
        means.append(m)
        print(f"  N={N:<6} E[Delta]_sim={m:<10.5g} "
              f"logN/alpha={np.log(N) / alpha:<10.5g} "
              f"ratio={m / (np.log(N) / alpha):.4f}")

    means = np.asarray(means)
    logN = np.log(Ns)
    slope, intercept = np.polyfit(logN, means, 1)
    fit = slope * logN + intercept
    ss_res = np.sum((means - fit) ** 2)
    ss_tot = np.sum((means - means.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    slope_theo = 1.0 / alpha
    rel = abs(slope - slope_theo) / slope_theo
    passed = (rel < REL_TOL_SLOPE) and (r2 > R2_MIN)
    print(_fmt_row("slope d E[Delta]/d log N  [== 1/alpha]", slope_theo, slope,
                   f"rel={rel:.2%}  R^2={r2:.4f}  {'PASS' if passed else 'FAIL'}"))
    return passed, np.asarray(Ns), means, alpha


def check_broadcast_latency(C, d, rho, Ns, n_broadcasts, base_seed):
    """
    Check the global-broadcast latency (source eccentricity B_s = max_j Delta_sj) vs t_N^cf.

    The simulated (1 - 1/N) quantile is compared with the closed-form Lambert-W bound,
    which sits ~5-20% above it; the exact max E[B_s] is reported alongside.
    Returns (passed, Ns, e_max, e_quantile, theory) for the figure.
    """
    print(f"\n[4. Global-broadcast latency -- C={C}, rho={rho:g}]")
    lam = lam_from_rho(rho, d)
    e_max, e_quant, theory = [], [], []
    for j, N in enumerate(Ns):
        rng = np.random.default_rng(base_seed + 400 + j)
        G = sample_regular_graph(N, C, rng=rng)
        u, v = edge_endpoints(G)
        delays = sample_pairwise_delays(u, v, N, d=d, lam=lam,
                                        n_broadcasts=n_broadcasts, rng=rng)
        bmax = float(np.mean(
            sample_broadcast_latencies(u, v, N, d=d, lam=lam,
                                       n_broadcasts=n_broadcasts, rng=rng)))
        q = broadcast_completion_quantile(delays, N)
        t_cf = broadcast_latency_theory(N, C, d, lam)
        e_quant.append(q)
        e_max.append(bmax)
        theory.append(t_cf)
        print(f"  N={N:<6} (1-1/N) quantile={q:<10.5g} E[B_s]={bmax:<10.5g} "
              f"t_N^cf={t_cf:<10.5g} cf/quantile={t_cf / q:.3f}")

    e_quant = np.asarray(e_quant)
    e_max = np.asarray(e_max)
    theory = np.asarray(theory)
    ratio = theory / e_quant
    bounded = bool(np.all(ratio > 1.0 - REL_TOL_BCAST_LO)
                   and np.all(ratio < 1.0 + REL_TOL_BCAST_HI))
    logN = np.log(Ns)
    slope, intercept = np.polyfit(logN, e_quant, 1)
    fit = slope * logN + intercept
    r2 = 1.0 - np.sum((e_quant - fit) ** 2) / np.sum((e_quant - e_quant.mean()) ** 2)
    passed = bounded and (slope > 0.0) and (r2 > R2_MIN_BCAST)
    print(_fmt_row("t_N^cf vs (1-1/N) quantile  [conservative upper bound]",
                   float(theory.mean()), float(e_quant.mean()),
                   f"cf/quantile in [{ratio.min():.3f}, {ratio.max():.3f}]  "
                   f"R^2(logN)={r2:.4f}  {'PASS' if passed else 'FAIL'}"))
    return passed, np.asarray(Ns), e_max, e_quant, theory


def save_distribution_figure(C, d, N, rhos, n_broadcasts, base_seed, out_path):
    """
    Plot the pairwise delay distribution across three jitter regimes.

    Multimodal at small rho (inheriting the discrete graph distance), washing out to a
    smooth right-skewed law as rho grows.
    """
    styles =["-", "--", "-."]
    cmap = plt.cm.viridis
    fig, ax = plt.subplots()
    for i, rho in enumerate(rhos):
        rng = np.random.default_rng(base_seed + 300 + i)
        G = sample_regular_graph(N, C, rng=rng)
        u, v = edge_endpoints(G)
        delays = sample_pairwise_delays(u, v, N, d=d, lam=lam_from_rho(rho, d),
                                        n_broadcasts=n_broadcasts, rng=rng)
        ax.hist(delays, bins=120, density=True, histtype="step", linewidth=2,
                color=cmap(i / max(len(rhos) - 1, 1)),
                linestyle=styles[i % len(styles)], label=fr"$\rho={rho:g}$")

    ax.set_xlabel(r"Pairwise broadcast delay  $\Delta$")
    ax.set_ylabel(r"Density")
    ax.legend(fontsize=18)
    saved = plotstyle.save(fig, out_path)
    print(f"\nFigure written to {saved.relative_to(REPO_ROOT)}")


def save_scaling_figure(Ns, means, alpha, C, rho, out_path):
    """Plot E[Delta] vs N (log-x) against the closed-form log N / alpha."""
    fig, ax = plt.subplots()
    grid = np.linspace(Ns.min(), Ns.max(), 200)
    ax.plot(grid, np.log(grid) / alpha, "--", color="C3",
            label=r"Theory $\log N/\alpha$")
    ax.plot(Ns, means, "o", color="C0", markersize=9,
            label=r"Simulated $E[\Delta]$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Number of nodes  $N$")
    ax.set_ylabel(r"Mean pairwise delay  $E[\Delta]$")
    ax.legend(fontsize=18, title=fr"$C={C},\ \rho={rho:g}$", title_fontsize=18)
    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def save_graph_distance_figure(dists, N, C, out_path):
    """Plot the empirical unweighted-distance survival against the discrete Gompertz law."""
    ells = np.arange(1, int(dists.max()) + 1)
    surv_emp = np.array([np.mean(dists > ell) for ell in ells])
    surv_theo = graph_distance_survival(ells, N, C)

    fig, ax = plt.subplots()
    ax.plot(ells, surv_emp, "s", color="C0", markersize=9,
            label=r"Simulated $\Pr(L>\ell)$")
    ax.plot(ells, surv_theo, "--o", color="C3",
            label=r"Gompertz (Tishby et al.)")

    ax.set_yscale("log")
    ax.set_xticks(ells)
    ax.set_xlabel(r"Graph distance  $\ell$")
    ax.set_ylabel(r"$\Pr(L>\ell)$")
    ax.legend(fontsize=18, title=fr"$N={N},\ C={C}$", title_fontsize=18)
    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


_SHELL_CMAP = plt.cm.plasma


def _bfs_shells(G, source):
    """
    Return (dist, shells) for one source: hop distance per node and node lists per shell.

    max(shells) is the source's eccentricity, the hop count behind d * ecc.
    """
    dist = nx.single_source_shortest_path_length(G, source)
    shells = {}
    for v, r in dist.items():
        shells.setdefault(r, []).append(v)
    return dist, shells


def _shell_layout(G, source):
    """
    Lay out the graph as a sunburst: radius = hop distance, angle inherited from the BFS parent.

    Each node takes an angular slot inside its parent's, sized by subtree weight; a small
    radial jitter spreads the crowded outer shells into bands. Returns (pos, dist, shells).
    """
    dist, shells = _bfs_shells(G, source)
    R = max(shells)

    parent, order = {source: None}, [source]
    for r in range(1, R + 1):
        for v in shells[r]:
            parent[v] = next(w for w in G.neighbors(v) if dist[w] == r - 1)
            order.append(v)
    weight = {v: 1 for v in G}
    for v in reversed(order):
        if parent[v] is not None:
            weight[parent[v]] += weight[v]

    span, ang = {source: (0.0, 2.0 * np.pi)}, {source: 0.0}
    for v in order:
        kids = [w for w in G.neighbors(v) if parent.get(w) == v]
        a0, a1 = span[v]
        tot = sum(weight[k] for k in kids) or 1
        a = a0
        for k in sorted(kids, key=lambda k: -weight[k]):
            width = (a1 - a0) * weight[k] / tot
            span[k], ang[k] = (a, a + width), a + width / 2.0
            a += width

    jit = np.random.default_rng(7)
    pos = {}
    for v in G:
        rad = dist[v] + (0.0 if dist[v] == 0 else jit.uniform(-0.16, 0.16))
        pos[v] = (rad * np.cos(ang[v]), rad * np.sin(ang[v]))
    return pos, dist, shells


def save_topology_figure(G, N, C, source, out_path):
    """
    Plot the quenched RRG by hop distance from one source (its BFS shells).

    The outermost occupied ring is ecc(source); it is drawn with a larger marker since
    the sparse straggler set alone sets the broadcast latency.
    """
    pos, _dist, shells = _shell_layout(G, source)
    R = max(shells)

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    ax.set_axis_off()
    ax.grid(False)
    ax.set_aspect("equal")
    ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in G.edges()],
                                     colors="0.55", linewidths=0.25, alpha=0.18, zorder=1))
    for r in range(1, R + 1):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="0.75", ls="--", lw=0.8, zorder=0))
    for r in range(R + 1):
        xy = np.array([pos[v] for v in shells[r]])
        ax.scatter(xy[:, 0], xy[:, 1],
                   s=200 if r == 0 else (26 if r == R else 11),
                   color="0.1" if r == 0 else _SHELL_CMAP(0.10 + 0.75 * (r - 1) / max(R - 1, 1)),
                   edgecolors="white", linewidths=0.5 if r == 0 else 0.15,
                   zorder=4 if r == 0 else 3,
                   label=("source" if r == 0 else
                          fr"$r={r}$  ($n={len(shells[r])}$)"))
    ax.set_xlim(-R - 0.7, R + 0.7)
    ax.set_ylim(-R - 0.7, R + 0.7)
    ax.legend(fontsize=13, loc="lower left", bbox_to_anchor=(-0.02, -0.02), framealpha=0.92,
              labelspacing=0.35, title=fr"hops $r$   ($N={N},\ C={C}$)", title_fontsize=13)

    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def save_shell_growth_figure(u, v, N, C, out_path):
    """
    Plot shell size vs hop distance over all sources against the tree law C(C-1)^(r-1).

    Points are the mean shell size over all N sources; whiskers span [min, max].
    """
    adj = _weighted_csr(u, v, np.ones(u.size), N)
    D = shortest_path(adj, method="D", directed=False, unweighted=True)
    R = int(D.max())
    rs = np.arange(1, R + 1)
    per_source = np.array([[np.sum(row == r) for r in rs] for row in D], dtype=float)
    mean, lo, hi = per_source.mean(0), per_source.min(0), per_source.max(0)
    tree = C * (C - 1.0) ** (rs - 1)

    fig, ax = plt.subplots()
    ax.plot(rs, tree, "--^", color="C3", markersize=9, linewidth=2,
            label=r"Tree growth $C(C-1)^{r-1}$")
    ax.errorbar(rs, mean, yerr=np.vstack([mean - lo, hi - mean]), marker="o", markersize=9,
                linewidth=2, capsize=5, capthick=1.5, color="C0",
                label=r"Simulated shell size (min/max over sources)")
    ax.axhline(N, color="0.4", linestyle=":", linewidth=1.5, label=fr"$N={N}$")

    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    ax.set_xticks(rs)
    ax.set_xlabel(r"Hops from source  $r$")
    ax.set_ylabel(r"Nodes at distance $r$")
    ax.legend(fontsize=15, title=fr"$N={N},\ C={C}$", title_fontsize=15, loc="lower right")
    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def save_broadcast_latency_figure(Ns, e_max, e_quant, C, d, rho, out_path):
    """Plot the simulated max E[B_s] and (1 - 1/N) quantile vs N against t_N^cf."""
    lam = lam_from_rho(rho, d)
    fig, ax = plt.subplots()
    grid = np.linspace(Ns.min(), Ns.max(), 200)
    ax.plot(grid, [broadcast_latency_theory(n, C, d, lam) for n in grid], "--",
            color="C3", label=r"Theory $t_N^{\mathrm{cf}}$ (Lambert-$W$)")
    ax.plot(Ns, e_max, "o", color="C0", markersize=9,
            label=r"Simulated max $E[B_s]$")
    ax.plot(Ns, e_quant, "s", color="C2", markersize=8,
            label=r"Simulated $(1-1/N)$ quantile")

    ax.set_xscale("log")
    ax.set_xlabel(r"Number of nodes  $N$")
    ax.set_ylabel(r"Global-broadcast latency")
    ax.legend(fontsize=16, title=fr"$C={C},\ \rho={rho:g}$", title_fontsize=16)
    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def main():
    """Run all checks, save the figures and return a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--C", type=int, default=DEFAULT_C, help="connectivity (degree)")
    ap.add_argument("--d", type=float, default=DEFAULT_D, help="baseline link delay")
    ap.add_argument("--rho", type=float, default=0.1, help="jitter ratio for scaling")
    ap.add_argument("--N", type=int, default=1000, help="N for the distance/dist checks")
    ap.add_argument("--n-broadcasts", type=int, default=300)
    ap.add_argument("--mix-reps", type=int, default=4000,
                    help="quenched realisations for check 6's D_M moments (cheap: no graph)")
    ap.add_argument("--source", type=int, default=0,
                    help="source node laid out in the topology figure (any node: ecc is a.s. flat)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 72)
    print("Network-latency validation -- simulated vs analytical")
    print("=" * 72)
    t0 = time.perf_counter()

    rng = np.random.default_rng(args.seed)
    G = sample_regular_graph(args.N, args.C, rng=rng)
    u, v = edge_endpoints(G)
    print(f"  quenched RRG: N={args.N}  C={args.C}  edges={u.size}  d={args.d:g}")

    ok_exp = check_propagation_exponent(args.C, args.d)
    ok_dist, dists = check_graph_distance(u, v, args.N, args.C, rng)

    Ns = [200, 500, 1000, 2000, 4000]
    ok_scale, Ns_arr, means, alpha = check_logn_scaling(
        args.C, args.d, args.rho, Ns, args.n_broadcasts, args.seed
    )
    ok_bcast, Ns_b, e_max, e_quant, _t_cf = check_broadcast_latency(
        args.C, args.d, args.rho, Ns, args.n_broadcasts, args.seed
    )
    ok_det =check_noise_free_limit(args.C, args.d, args.N, args.n_broadcasts,
                                    np.random.default_rng(args.seed + 900))
    ok_mix = check_ac_mix_chain(args.d, args.mix_reps, args.seed + 1300)

    fig_dir = REPO_ROOT / "results" / "figures" / "stage_2_figures"
    save_distribution_figure(args.C, args.d, args.N, (0.1, 1.0, 10.0),
                             args.n_broadcasts, args.seed,
                             fig_dir / "network_delay_distribution.png")
    save_scaling_figure(Ns_arr, means, alpha, args.C, args.rho,
                        fig_dir / "network_logn_scaling.png")
    save_graph_distance_figure(dists, args.N, args.C,
                               fig_dir / "network_graph_distance.png")
    save_topology_figure(G, args.N, args.C, args.source,
                         fig_dir / "network_topology_shells.png")
    save_shell_growth_figure(u, v, args.N, args.C,
                             fig_dir / "network_shell_growth.png")
    save_broadcast_latency_figure(Ns_b, e_max, e_quant, args.C, args.d, args.rho,
                                  fig_dir / "network_broadcast_latency.png")

    all_ok = ok_exp and ok_dist and ok_scale and ok_bcast and ok_det and ok_mix
    print("\n" + "=" * 72)
    print(f"RESULT: {'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}"
          f"   (total {time.perf_counter() - t0:.1f}s)")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
