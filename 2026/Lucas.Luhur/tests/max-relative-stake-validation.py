"""
Validation of the maximum relative stake M_N = max_i X_i / sum_j X_j (src/consensus/stake.py).

Checks:
  1. Bounds -- every realisation has 1/N <= M_N <= 1.
  2. Scale-invariance -- M_N is independent of the Pareto scale x_m.
  3. Gini closed form -- sampled mean Gini ~= G = 1/(2k - 1).
  4. Reference table -- simulated E[M_N] matches the reference values within Monte-Carlo CI.
  5. Asymptotic vs finite-N -- the large-N formula overestimates near k = 1, converges for large k.
Also archives the E[M_N] table and saves the E[M_N] vs Pareto shape figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.stats import pareto     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotstyle  # noqa: E402
from consensus import (  # noqa: E402
    expected_max_relative_stake,
    gini,
    gini_from_shape,
    sample_relative_stakes,
    simulate_max_relative_stake,
)

Z_TOL = 4.0
GINI_TOL = 0.03

# reference E[M_N] per Pareto shape k at N = 1000 (50k draws; k=4/3 used 1M)
REF_TABLE = {
    1.1: 0.1853,
    1.2: 0.1429,
    4 / 3: 0.1027,
    1.5: 0.0690,
    2.0: 0.0268,
    3.0: 0.00901,
    5.0: 0.00370,
}
REF_CI95 = {
    1.1: (0.1838, 0.1868),
    1.2: (0.1417, 0.1442),
    4 / 3: (0.1017, 0.1037),
    1.5: (0.0683, 0.0697),
    2.0: (0.0265, 0.0270),
    3.0: (0.00896, 0.00906),
    5.0: (0.00369, 0.00371),
}
ALPHAS = sorted(REF_TABLE)


def _fmt_row(name, theo, emp, extra=""):
    """Format one theory-vs-simulation result line."""
    return f"  {name:<40} theory={theo:<12.6g} sim={emp:<12.6g} {extra}"


def check_bounds(N, shape, reps, rng):
    """1/N <= M_N <= 1 for every realisation (distribution-free)."""
    w = pareto.rvs(b=shape, scale=1.0, size=(reps, N), random_state=rng)
    m = w.max(axis=1) / w.sum(axis=1)
    lo, hi = float(m.min()), float(m.max())
    passed = (lo >= 1.0 / N - 1e-12) and (hi <= 1.0 + 1e-12)
    print(_fmt_row("M_N bounds [1/N, 1]", 1.0 / N, lo,
                   f"max={hi:.4g} {'PASS' if passed else 'FAIL'}"))
    return passed


def check_scale_invariance(N, shape, reps, seed):
    """
    M_N is independent of the Pareto scale x_m. Two draws with identical RNG
    state but scale 1 vs 1000: every stake is exactly rescaled, so the ratio
    max/sum is bit-identical up to floating-point rounding.
    """
    w1 = pareto.rvs(b=shape, scale=1.0, size=(reps, N),
                    random_state=np.random.default_rng(seed))
    w2 = pareto.rvs(b=shape, scale=1000.0, size=(reps, N),
                    random_state=np.random.default_rng(seed))
    m1 = w1.max(axis=1) / w1.sum(axis=1)
    m2 = w2.max(axis=1) / w2.sum(axis=1)
    dmax = float(np.max(np.abs(m1 - m2)))
    passed = dmax < 1e-12
    print(_fmt_row("scale-invariance  max|M(x_m=1)-M(x_m=1e3)|", 0.0, dmax,
                   f"{'PASS' if passed else 'FAIL'}"))
    return passed


def _mean_gini(N, shape, reps, rng):
    """Mean plug-in Gini over `reps` sampled stake vectors."""
    gs = np.array([gini(sample_relative_stakes(N, shape=shape, rng=rng))
                   for _ in range(reps)])
    return float(gs.mean())


def check_gini(N, reps, rng):
    """
    Sampled mean Gini ~= closed-form G = 1/(2k - 1).

    Gated at a light tail (k=3), where the finite-N plug-in Gini is nearly unbiased;
    k=4/3 and k=2 are printed for context only, since the plug-in under-estimates heavy tails.
    """
    reps = min(reps, 5_000)
    gate_k = 3.0
    g_theo = float(gini_from_shape(gate_k))
    g_emp = _mean_gini(N, gate_k, reps, rng)
    passed = abs(g_emp - g_theo) < GINI_TOL
    print(_fmt_row(f"Gini(k={gate_k:.3g})  [== 1/(2k-1)]", g_theo, g_emp,
                   f"|d|={abs(g_emp - g_theo):.4f} {'PASS' if passed else 'FAIL'}"))
    for k in (2.0, 4 / 3):
        gt = float(gini_from_shape(k))
        ge = _mean_gini(N, k, reps, rng)
        print(_fmt_row(f"  (context) Gini(k={k:.3g})", gt, ge,
                       f"|d|={abs(ge - gt):.4f}  plug-in bias (heavy tail)"))
    return passed


def check_table(N, reps, rng):
    """Reproduce the reference E[M_N] table; each simulated mean within Z_TOL SE."""
    print("\n[Reproduce the reference max-stake table -- E[M_N] vs shape k]")
    print(f"  {'k':>6} | {'ref':>8} | {'sim':>8} {'CI95':>16} | {'z':>6} | "
          f"{'SD(quenched)':>12}")
    print("  " + "-" * 68)
    all_ok = True
    means, sds, ci_lo, ci_hi, medians = [], [], [], [], []
    for a in ALPHAS:
        stats = simulate_max_relative_stake(N, shape=a, reps=reps, rng=rng)
        ref_se = (REF_CI95[a][1] - REF_CI95[a][0]) / (2.0 * 1.96)
        z = (stats["mean"] - REF_TABLE[a]) / np.hypot(stats["se"], ref_se)
        passed = abs(z) < Z_TOL
        all_ok &= passed
        means.append(stats["mean"])
        sds.append(stats["sd"])
        ci_lo.append(stats["ci95"][0])
        ci_hi.append(stats["ci95"][1])
        medians.append(stats["median"])
        print(f"  {a:>6.3f} | {100*REF_TABLE[a]:>7.2f}% | {100*stats['mean']:>7.2f}% "
              f"[{100*stats['ci95'][0]:>5.2f},{100*stats['ci95'][1]:>5.2f}] | "
              f"{z:>+6.2f} | {100*stats['sd']:>11.1f} "
              f"{'PASS' if passed else 'FAIL'}")
    med = float(medians[ALPHAS.index(4 / 3)])
    med_ok = abs(med - 0.065) < 0.003
    all_ok &= med_ok
    print(f"  median M_N at k=4/3: {100*med:.2f}%  (reference 6.5%; {reps:,} draws) "
          f"{'PASS' if med_ok else 'FAIL'}")
    return all_ok, (np.array(means), np.array(sds),
                    np.array(ci_lo), np.array(ci_hi), np.array(medians))


def save_table(N, reps, seed, means, sds, ci_lo, ci_hi, medians, out_csv):
    """
    Archive the finite-N E[M_N] table to results/tables/max_relative_stake.csv.

    This file is the single source of every E[M_N] number (figure annotations and legend
    labels). An existing table drawn at more reps is never overwritten; the committed table
    is `--reps 2000000 --seed 0`.
    """
    import pandas as pd
    out_csv = Path(out_csv)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if int(old["reps"].max()) > reps:
            print(f"\n(table {out_csv.relative_to(REPO_ROOT)} kept: it holds {int(old['reps'].max()):,} "
                  f"draws per shape, this run only {reps:,})")
            return
    se = np.asarray(sds) / np.sqrt(reps)
    df = pd.DataFrame({"shape": ALPHAS, "N": N, "mean": means, "sd": sds, "se": se,
                       "ci95_lo": ci_lo, "ci95_hi": ci_hi, "median": medians,
                       "reps": reps, "seed": seed})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nTable written to {out_csv.relative_to(REPO_ROOT)} ({reps:,} draws per shape, seed {seed})")


def check_asymptotic(N, means):
    """
    Directional check on the extreme-value approximation.

    Near k=1 (heavy tail) it overestimates (asymp > sim); for large k it converges.
    """
    print("\n[Asymptotic (large-N) vs finite-N simulation]")
    ok = True
    a_low = 4 / 3
    asymp_low = float(expected_max_relative_stake(a_low, N))
    sim_low = float(means[ALPHAS.index(a_low)])
    over = asymp_low > sim_low
    ok &= over
    print(_fmt_row(f"k=4/3 overestimate (asymp>sim)", asymp_low, sim_low,
                   f"ratio={asymp_low/sim_low:.2f}x {'PASS' if over else 'FAIL'}"))

    a_hi = 5.0
    asymp_hi = float(expected_max_relative_stake(a_hi, N))
    sim_hi = float(means[ALPHAS.index(a_hi)])
    rel = abs(asymp_hi - sim_hi) / sim_hi
    conv = rel < 0.05
    ok &= conv
    print(_fmt_row(f"k=5 convergence (|asymp-sim|/sim<5%)", asymp_hi, sim_hi,
                   f"rel={rel:.2%} {'PASS' if conv else 'FAIL'}"))
    return ok


def save_figure(N, means, ci_lo, ci_hi, out_path):
    """Plot E[M_N] vs Pareto shape (log-y): the large-N curve, the simulation and reference lines."""
    a_curve = np.linspace(1.01, 6.0, 600)
    mean_curve = expected_max_relative_stake(a_curve, N)
    a_sim = np.array(ALPHAS)
    yerr = np.vstack([means - ci_lo, ci_hi - means])

    fig, ax = plt.subplots()
    ax.plot(a_curve, mean_curve, "--", color="C3", lw=2.2,
            label=r"Large-$N$ approximation")
    ax.errorbar(a_sim, means, yerr=yerr, fmt="o", color="C0", ms=6,
                capsize=4, elinewidth=1.3,
                label=r"Finite-$N$ simulation (mean $\pm$ 95\% CI)")
    ax.axhline(1.0 / N, color="0.4", ls="--", lw=1.3,
               label=fr"Equal shares  $1/N = {100/N:.1f}\%$")
    ax.axvline(4 / 3, color="0.55", ls=":", lw=1.5, label=r"$k = 4/3$")

    i43 = ALPHAS.index(4 / 3)
    ax.annotate(fr"Asymptotic: {100*float(expected_max_relative_stake(4/3, N)):.1f}\%",
                xy=(4 / 3, float(expected_max_relative_stake(4 / 3, N))),
                xytext=(1.7, 0.30), arrowprops={"arrowstyle": "->"}, fontsize=14)
    ax.annotate(fr"Finite-$N$: {100*means[i43]:.1f}\%",
                xy=(4 / 3, means[i43]), xytext=(1.7, 0.055),
                arrowprops={"arrowstyle": "->"}, fontsize=14)

    ax.set_yscale("log")
    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(9e-4, 1.0)
    ax.set_xlabel(r"Pareto shape  $k$")
    ax.set_ylabel(r"Expected maximum relative stake  $\mathbb{E}[M_N]$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=15)

    saved = plotstyle.save(fig, out_path)
    print(f"\nFigure written to {saved.relative_to(REPO_ROOT)}")


def main():
    """Run the maximum-relative-stake checks, archive the table and save the figure."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000, help="number of nodes")
    ap.add_argument("--reps", type=int, default=50_000,
                    help="stake realisations per shape (the reference table used 50k)")
    ap.add_argument("--seed", type=int, default=162, help="the reference seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    print("=" * 72)
    print("Maximum-relative-stake validation -- simulated vs closed form")
    print(f"  N={args.N:,}  reps={args.reps:,}  seed={args.seed}")
    print("=" * 72 + "\n")

    print("[Closed-form anchors]")
    ok_bounds = check_bounds(args.N, 4 / 3, reps=min(args.reps, 20_000), rng=rng)
    ok_scale = check_scale_invariance(args.N, 4 / 3, reps=2_000, seed=args.seed)
    ok_gini = check_gini(args.N, args.reps, rng=rng)

    ok_table, (means, sds, ci_lo, ci_hi, medians) = check_table(args.N, args.reps, rng)
    ok_asymp = check_asymptotic(args.N, means)
    save_table(args.N, args.reps, args.seed, means, sds, ci_lo, ci_hi, medians,
               REPO_ROOT / "results" / "tables" / "max_relative_stake.csv")

    fig_dir = REPO_ROOT / "results" / "figures" / "stage_2_figures"
    save_figure(args.N, means, ci_lo, ci_hi, fig_dir / "max_relative_stake.png")

    all_ok = ok_bounds and ok_scale and ok_gini and ok_table and ok_asymp
    print("\n" + "=" * 72)
    print(f"RESULT: {'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}"
          f"   ({time.perf_counter() - t0:.1f}s)")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
