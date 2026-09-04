"""
Leader-election validation: runs the consensus message generator and checks the
simulated winner statistics n(t) against the closed forms.

Checks: P(n >= 1) == f; filled slots == T*f; E[n] and Var[n] (Poisson-binomial);
P(n >= 2) (fork rate); total messages == T * sum phi (eq 7); and P(n >= 1) == f
across several Pareto stake shapes k.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np
from scipy.stats import poisson

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotstyle  # noqa: E402
from consensus import (  # noqa: E402
    DEFAULT_F,
    DEFAULT_T,
    expected_winners_per_slot,
    gini,
    lottery,
    prob_at_least_one,
    prob_at_least_two,
    sample_relative_stakes,
    shape_from_gini,
    simulate_events,
    var_winners_per_slot,
    winner_counts_from_events,
    wins_per_node_from_events,
)

Z_TOL = 4.0
REL_TOL = 0.02


def _fmt_row(name, theo, emp, extra=""):
    """Format one theory-vs-simulation result row."""
    return f"  {name:<34} theory={theo:<14.6g} sim={emp:<14.6g} {extra}"


def run_checks(alpha, f, T, rng):
    """
    Simulate one horizon and compare empirical vs theoretical winner statistics.

    Returns (all_ok, n_t, mu_theo, nodes).
    """
    t0 = time.perf_counter()
    slots, nodes = simulate_events(alpha, f=f, T=T, rng=rng)
    n_t = winner_counts_from_events(slots, T)
    dt = time.perf_counter() - t0

    N = alpha.size
    print(f"  N={N}  T={T:,}  f={f:.6g}  Gini(stake)={gini(alpha):.3f}  "
          f"(sim {dt:.1f}s)\n")

    ok = True

    p1_theo = prob_at_least_one(f)
    p1_emp = float(np.mean(n_t >= 1))
    se = np.sqrt(p1_theo * (1 - p1_theo) / T)
    z = (p1_emp - p1_theo) / se
    passed = abs(z) < Z_TOL
    ok &= passed
    print(_fmt_row("P(n>=1)  [== f]", p1_theo, p1_emp,
                   f"z={z:+.2f} {'PASS' if passed else 'FAIL'}"))

    filled_theo = T * f
    filled_emp = int(np.sum(n_t >= 1))
    se_f = np.sqrt(p1_theo * (1 - p1_theo) * T)
    z = (filled_emp - filled_theo) / se_f
    passed = abs(z) < Z_TOL
    ok &= passed
    print(_fmt_row("filled slots  [== T*f]", filled_theo, filled_emp,
                   f"z={z:+.2f} {'PASS' if passed else 'FAIL'}"))

    mu_theo = expected_winners_per_slot(alpha, f)
    mu_emp = float(np.mean(n_t))
    var_theo = var_winners_per_slot(alpha, f)
    se_mu = np.sqrt(var_theo / T)
    z = (mu_emp - mu_theo) / se_mu
    passed = abs(z) < Z_TOL
    ok &= passed
    print(_fmt_row("E[n]", mu_theo, mu_emp,
                   f"z={z:+.2f} {'PASS' if passed else 'FAIL'}"))

    var_emp = float(np.var(n_t))
    rel = abs(var_emp - var_theo) / var_theo
    passed = rel < REL_TOL
    ok &= passed
    print(_fmt_row("Var[n]", var_theo, var_emp,
                   f"rel={rel:.3%} {'PASS' if passed else 'FAIL'}"))

    p2_theo = prob_at_least_two(alpha, f)
    p2_emp = float(np.mean(n_t >= 2))
    se2 = np.sqrt(max(p2_theo, 1e-12) * (1 - p2_theo) / T)
    z = (p2_emp - p2_theo) / se2
    passed = abs(z) < Z_TOL
    ok &= passed
    print(_fmt_row("P(n>=2)  [fork rate]", p2_theo, p2_emp,
                   f"z={z:+.2f} {'PASS' if passed else 'FAIL'}"))

    msgs_theo = T * mu_theo
    msgs_emp = int(np.sum(n_t))
    se_w = np.sqrt(var_theo * T)
    z = (msgs_emp - msgs_theo) / se_w
    passed = abs(z) < Z_TOL
    ok &= passed
    print(_fmt_row("total messages  [== T*sum phi]", msgs_theo, msgs_emp,
                   f"z={z:+.2f} {'PASS' if passed else 'FAIL'}"))

    return ok, n_t, mu_theo, nodes


def independence_demo(N, f, T, shapes, base_seed):
    """
    Sweep Pareto shapes k and record P(n>=1) (distribution-free, == f) and P(n>=2).

    Returns (all_ok, ginis, p1s, p2_emp, p2_theo) with per-shape arrays.
    """
    print("\n[Winner-count statistics across stake distributions]")
    se = np.sqrt(f * (1 - f) / T)
    all_ok = True
    ginis, p1s, p2_emp, p2_theo = [], [], [], []
    for j, k in enumerate(shapes):
        rng = np.random.default_rng(base_seed + 100 + j)
        alpha = sample_relative_stakes(N, shape=k, rng=rng)
        slots, _ = simulate_events(alpha, f=f, T=T, rng=rng)
        n_t = winner_counts_from_events(slots, T)
        p1 = float(np.mean(n_t >= 1))
        p2e = float(np.mean(n_t >= 2))
        p2t = prob_at_least_two(alpha, f)
        g = gini(alpha)
        z = (p1 - f) / se
        z2 = (p2e - p2t) / np.sqrt(max(p2t, 1e-12) * (1 - p2t) / T)
        passed = abs(z) < Z_TOL and abs(z2) < Z_TOL
        all_ok &= passed
        ginis.append(g)
        p1s.append(p1)
        p2_emp.append(p2e)
        p2_theo.append(p2t)
        print(f"  k={k:<4} Gini={g:.3f}  P(n>=1)={p1:.6f} z={z:+.2f}  "
              f"P(n>=2)={p2e:.6f} z={z2:+.2f}  {'PASS' if passed else 'FAIL'}")
    return (all_ok, np.array(ginis), np.array(p1s),
            np.array(p2_emp), np.array(p2_theo))


def save_figure(n_t, mu_theo, out_path):
    """Save a histogram of simulated winner counts n_t against the Poisson reference."""
    kmax = int(n_t.max())
    ks = np.arange(0, kmax + 1)
    emp = np.array([np.mean(n_t == k) for k in ks])

    fig, ax = plt.subplots()
    ax.bar(ks, emp, width=0.6, alpha=0.7, color="C0", label=r"Simulated $n_t$")
    ax.plot(ks, poisson.pmf(ks, mu_theo), "o--", color="C3",
            label=fr"Poisson($\lambda={mu_theo:.4f}$)")

    for k, v in zip(ks, emp):
        if v <= 0:
            continue
        p = 100.0 * v
        lab = f"{p:.1f}" if p >= 10 else (f"{p:.2f}" if p >= 0.1 else f"{p:.2g}")
        ax.annotate(lab + r"\%", (k, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=16)

    ax.set_yscale("log")
    ax.set_ylim(top=5.0)
    ax.set_xticks(ks)
    ax.set_xlabel(r"Leaders per slot  $n_t$")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=16)

    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def save_independence_figure(ginis, p1s, f, T, out_path):
    """Plot empirical P(n>=1) against stake Gini with a +/- Z_TOL SE band around f."""
    order = np.argsort(ginis)
    g = np.asarray(ginis)[order]
    p = np.asarray(p1s)[order]
    se = np.sqrt(f * (1 - f) / T)

    fig, ax = plt.subplots()
    ax.axhspan(f - Z_TOL * se, f + Z_TOL * se, color="C3", alpha=0.15,
               label=fr"Theory $f \pm {Z_TOL:.0f}\,$SE")
    ax.axhline(f, color="C3", linestyle="--",
               label=fr"Theory $P(n\geq 1)=f={f:.4g}$")
    ax.plot(g, p, "s", color="C0", markersize=9,
            label=r"Simulated $P(n\geq 1)$")

    lo = min(p.min(), f - Z_TOL * se)
    hi = max(p.max(), f + Z_TOL * se)
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"Stake inequality  Gini($\alpha$)")
    ax.set_ylabel(r"$P(n\geq 1)$")
    ax.legend()

    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def save_p2_figure(ginis, p2_emp, p2_theo, T, out_path):
    """Plot the fork rate P(n>=2) against stake Gini, simulated vs theory."""
    order = np.argsort(ginis)
    g = np.asarray(ginis)[order]
    pe = np.asarray(p2_emp)[order]
    pt = np.asarray(p2_theo)[order]
    se = np.sqrt(np.clip(pt, 1e-12, None) * (1.0 - pt) / T)

    fig, ax = plt.subplots()
    ax.fill_between(g, pt - Z_TOL * se, pt + Z_TOL * se, color="C3", alpha=0.15,
                    label=fr"Theory $\pm {Z_TOL:.0f}\,$SE")
    ax.plot(g, pt, "--", color="C3", marker="o", markersize=5,
            label=r"Theory $P(n\geq 2)=0.000561$")
    ax.plot(g, pe, "s", color="C0", markersize=9,
            label=r"Simulated $P(n\geq 2)$")

    ax.set_xlabel(r"Stake inequality  Gini($\alpha$)")
    ax.set_ylabel(r"$P(n\geq 2)$")
    ax.legend(loc="upper left")

    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def save_wins_vs_stake_figure(alpha, wins, f, T, out_path):
    """
    Scatter per-node wins against relative stake with the closed-form mean
    E[wins_i] = T * phi(alpha_i) = T * [1 - (1 - f)^{alpha_i}].
    """
    fig, ax = plt.subplots()
    ax.scatter(alpha, wins, s=12, color="C0", alpha=0.4,
               label=r"Node $(\alpha_i,\,\mathrm{wins}_i)$")
    a = np.linspace(0.0, float(alpha.max()), 200)
    ax.plot(a, T * lottery(a, f), "--", color="C3",
            label=r"Theory $T\,\phi(\alpha)=T\,[1-(1-f)^{\alpha}]$")

    ax.set_xlabel(r"Relative stake  $\alpha_i$")
    ax.set_ylabel(r"Wins over horizon  $\mathrm{wins}_i$")
    ax.legend()

    saved = plotstyle.save(fig, out_path)
    print(f"Figure written to {saved.relative_to(REPO_ROOT)}")


def main():
    """Run the checks and the shape sweep, then save the figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000, help="number of nodes")
    ap.add_argument("--T", type=int, default=DEFAULT_T, help="slots (horizon)")
    ap.add_argument("--f", type=float, default=DEFAULT_F, help="active-slots coeff")
    ap.add_argument("--shape", type=float, default=2.0, help="Pareto shape k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T = args.T
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("Leader-election validation -- simulated vs analytical")
    print("=" * 72)

    alpha = sample_relative_stakes(args.N, shape=args.shape, rng=rng)
    ok, n_t, mu_theo, nodes = run_checks(alpha, args.f, T, rng)

    shapes = [float(shape_from_gini(g)) for g in np.linspace(0.08, 0.50, 9)]
    indep_ok, ginis, p1s, p2_emp, p2_theo = independence_demo(
        args.N, args.f, T, shapes=shapes, base_seed=args.seed
    )

    fig_dir = REPO_ROOT / "results" / "figures" / "stage_2_figures"
    save_figure(n_t, mu_theo, fig_dir / "leader_election_validation.png")
    save_independence_figure(ginis, p1s, args.f, T,
                             fig_dir / "leader_election_invariant.png")
    save_p2_figure(ginis, p2_emp, p2_theo, T,
                   fig_dir / "leader_election_p2.png")
    wins = wins_per_node_from_events(nodes, alpha.size)
    save_wins_vs_stake_figure(alpha, wins, args.f, T,
                              fig_dir / "wins_vs_stake.png")

    all_ok = ok and indep_ok
    print("\n" + "=" * 72)
    print("RESULT:", "ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
