"""
Comparative analysis of the one-epoch results across the size ladder N = 100/1000/3000.

No new simulation: the six committed tables (`recommendation_stake`,
`recommendation_attribution` and their `_N100` / `_N3000` variants) are joined on matched
cells, with nulls resolved through run.py's `_measure_law`. Also draws the relative-stake
histograms per rung. Writes results/figures/size_bracket/*.png and
results/tables/size_bracket_summary.csv, plus a console read of the headline comparisons.
"""

from __future__ import annotations

import sys
from dataclasses import replace as dc_replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import matplotlib.ticker as mticker   # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import plotstyle                      # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

from consensus.stake import gini, sample_relative_stakes   # noqa: E402
from experiments import load_experiment, make_apply_cell     # noqa: E402
import run as runpy                   # noqa: E402

CONFIG_DIR = REPO / "experiments" / "configs"
TABLES = REPO / "results" / "tables"
FIGDIR = REPO / "results" / "figures" / "size_bracket"

P_S = 0.1
LADDER = {
    100:  ("recommendation_stake_N100",  "recommendation_attribution_N100"),
    1000: ("recommendation_stake",       "recommendation_attribution"),
    3000: ("recommendation_stake_N3000", "recommendation_attribution_N3000"),
}
_VIRIDIS = plt.get_cmap("viridis")
K_SERIES = [(4.0 / 3.0, _VIRIDIS(0.5), r"$k = 4/3$"),   # viridis 0.5 / 1.0: one colour per
            (3.0, _VIRIDIS(1.0), r"$k = 3$")]           # Pareto shape across the stake figures
LATENT = "mixnet_attribution"


def _save(fig, name):
    """Save the figure under FIGDIR."""
    return plotstyle.save(fig, FIGDIR / name)


STAKE_SEED = 20260817
STAKE_REPS = 200
HIST_BINS = 40
LOG_XLIM = (5e-6, 1.0)
HIST_YMAX = {100: 25.0, 1000: 250.0, 3000: 850.0}

ATTR_CELLS = [
    ("100",    100,  {"p_s": 0.1, "width": 4}, None),
    ("1000",   1000, {"width": 4, "mix_scale": 0.5}, None),
    ("3000",   3000, {}, None),
]


def _load(name):
    """Load a named experiment config and its committed results table."""
    exp = load_experiment(CONFIG_DIR / f"{name}.yaml")
    df = pd.read_csv(TABLES / f"{exp.name}.csv")
    return exp, df


def _null_for(exp, cell, attack, ps_override=None):
    """
    Return the random-guess null E[1/|S|] for this cell via run.py's `_measure_law`.

    `ps_override` pins cfg.dummy.p_s if a table's rate differs from its config.
    """
    apply_cell = make_apply_cell(exp.path_map)
    cfg = apply_cell(exp.base_cfg, {**cell, "attack": attack})
    if ps_override is not None:
        cfg = dc_replace(cfg, dummy=dc_replace(cfg.dummy, p_s=ps_override))
    sizes, probs = runpy._measure_law(cfg)
    return float(np.sum(probs / sizes))


def _mean_sem(v):
    """Return (mean, standard error of the mean) of a sample."""
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def _n_axis(ax):
    """Format a log-scale network-size axis with the ladder's N values as ticks."""
    ax.set_xscale("log")
    ax.set_xticks(list(LADDER))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(80, 3800)
    ax.set_xlabel(r"Network size  $N$")


def stake_figures(rows):
    """Draw the stake-face figures (top-1, Jaccard, bandwidth vs N) at p_s = P_S; append rows."""
    data = {N: _load(LADDER[N][0]) for N in LADDER}
    Ns = np.array(list(LADDER))
    xpos = np.arange(len(Ns))

    def _cat_n_axis(ax):
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(N) for N in Ns])
        ax.set_xlim(-0.5, len(Ns) - 0.5)
        ax.set_xlabel(r"Network size  $N$")

    fig, ax = plt.subplots()
    width = 0.32
    for j, (k, color, lab) in enumerate(K_SERIES):
        m, s = zip(*(_mean_sem(
            data[N][1][np.isclose(data[N][1]["shape"], k)
                       & np.isclose(data[N][1]["p_s"], P_S)]["stake_top1_hit"]) for N in Ns))
        ax.bar(xpos + (j - 0.5) * width, m, width, yerr=s, capsize=3, color=color,
               edgecolor="0.2", lw=0.8, label=lab)
        for N, mm, ss in zip(Ns, m, s):
            rows.append(dict(face="stake", N=int(N), shape=k, p_s=P_S,
                             measure="stake_top1_hit", mean=mm, sem=ss, null=1.0 / N,
                             excess=mm - 1.0 / N))
    for i, N in enumerate(Ns):
        ax.hlines(1.0 / N, i - 0.45, i + 0.45, color="0.4", ls="--", lw=1.3, zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", ls="--", lw=1.3))
    labels.append(r"$1/N$ (random guess)")
    _cat_n_axis(ax)
    ax.set_ylabel(r"Largest stakeholder identified  $P_{\mathrm{max}}$")
    ax.legend(handles, labels, fontsize=15, loc="upper right")
    _save(fig, "size_bracket_stake_top1_vs_N.png")

    fig, ax = plt.subplots()
    width = 0.32
    for j, (k, color, lab) in enumerate(K_SERIES):
        m, s = zip(*(_mean_sem(
            data[N][1][np.isclose(data[N][1]["shape"], k)
                       & np.isclose(data[N][1]["p_s"], P_S)]["stake_top_jaccard"]) for N in Ns))
        ax.bar(xpos + (j - 0.5) * width, m, width, yerr=s, capsize=3, color=color,
               edgecolor="0.2", lw=0.8, label=lab)
        for N, mm, ss in zip(Ns, m, s):
            rows.append(dict(face="stake", N=int(N), shape=k, p_s=P_S,
                             measure="stake_top_jaccard", mean=mm, sem=ss))
    _cat_n_axis(ax)
    ax.set_ylabel(r"Top 1\% stakeholder set identified  $J_x$")
    ax.legend(fontsize=15, loc="upper right")
    _save(fig, "size_bracket_stake_jaccard_vs_N.png")

    fig, ax = plt.subplots()
    b, bs = zip(*(_mean_sem(
        data[N][1][np.isclose(data[N][1]["p_s"], P_S)]["bandwidth_overhead"]) for N in Ns))
    ax.bar(xpos, b, 0.5, yerr=bs, capsize=3, color="C0", edgecolor="0.2", lw=0.8)
    for N, bm, ss in zip(Ns, b, bs):
        rows.append(dict(face="stake", N=int(N), p_s=P_S, measure="bandwidth_overhead",
                         mean=bm, sem=ss))
    _cat_n_axis(ax)
    ax.set_ylabel(r"Bandwidth overhead  $\beta$")
    _save(fig, "size_bracket_stake_bandwidth_vs_N.png")


def attribution_figures(rows):
    """Draw the attribution-face figures (top-1, excess, latency vs N) and print the transfer test."""
    exps = {N: _load(LADDER[N][1]) for N in LADDER}
    xpos = np.arange(len(ATTR_CELLS))
    stats = []
    for lab, N, cell, ps_over in ATTR_CELLS:
        exp, df = exps[N]
        sel = df[df["attack"] == LATENT]
        for key, val in cell.items():
            sel = sel[np.isclose(sel[key], val)]
        m, s = _mean_sem(sel["deanon_top1"])
        null = _null_for(exp, cell, LATENT, ps_over)
        stats.append((lab, m, s, null))
        rows.append(dict(face="attribution", N=N, cell=lab.replace("\n", " "), arm="latent",
                         measure="deanon_top1", mean=m, sem=s, null=null,
                         excess=m - null, z_vs_null=(m - null) / s if s > 0 else np.nan,
                         times_chance=m / null))

    def _cat_axis(ax):
        ax.set_xticks(xpos)
        ax.set_xticklabels([lab for lab, _, _, _ in ATTR_CELLS])
        ax.set_xlabel(r"Network size  $N$")
        ax.set_xlim(-0.5, len(ATTR_CELLS) - 0.5)

    fig, ax = plt.subplots()
    ax.bar(xpos, [m for _, m, _, _ in stats], 0.5, yerr=[s for _, _, s, _ in stats],
           capsize=3, color="C0", edgecolor="0.2", lw=0.8)
    for i, (_, _, _, null) in enumerate(stats):
        ax.hlines(null, i - 0.33, i + 0.33, color="0.4", ls="--", lw=1.4, zorder=3)
    ax.set_ylabel(r"Top-1 de-anonymisation  $P_\mathrm{Top1}$")
    _cat_axis(ax)
    ax.legend([Line2D([], [], color="0.4", ls="--", lw=1.4)],
              [r"$\mathbb{E}[1/|S_t|]$ (random guess)"], fontsize=15, loc="upper right")
    plotstyle.save(fig, FIGDIR / "size_bracket_attribution_top1.png")

    fig, ax = plt.subplots()
    ax.bar(xpos, [m - null for _, m, _, null in stats], 0.5,
           yerr=[s for _, _, s, _ in stats], capsize=3, color="C0", edgecolor="0.2", lw=0.8)
    ax.axhline(0.0, color="0.4", ls="--", lw=1.0, zorder=3)
    ax.set_ylabel(r"Excess of $P_\mathrm{Top1}$ over the random guess")
    _cat_axis(ax)
    _save(fig, "size_bracket_attribution_top1_excess.png")

    def _cell_mean(N, cell, col):
        """Mean and sem of one column over the latent-arm rows of a cell."""
        exp, df = exps[N]
        sel = df[df["attack"] == LATENT]
        for key, val in cell.items():
            sel = sel[np.isclose(sel[key], val)]
        return _mean_sem(sel[col])

    for lab, N, cell, _ in ATTR_CELLS:
        bm, ss = _cell_mean(N, cell, "bandwidth_overhead")
        rows.append(dict(face="attribution", N=N, cell=lab.replace("\n", " "),
                         measure="bandwidth_overhead", mean=bm, sem=ss))

    fig, ax = plt.subplots()
    leg, leg_s = zip(*(_cell_mean(N, cell, "latency") for _, N, cell, _ in ATTR_CELLS))
    ell = [_cell_mean(N, cell, "latency_overhead")[0] for _, N, cell, _ in ATTR_CELLS]
    dbr = np.array(leg) / (np.array(ell) - 1.0)            # E[D_br] = leg/(ell - 1)
    total = np.array(leg) + dbr
    width = 0.32
    ax.bar(xpos - 0.5 * width, leg, width, yerr=leg_s, capsize=3, color="C0",
           edgecolor="0.2", lw=0.8, label=r"AC path  $\mathbb{E}[Y]$")
    ax.bar(xpos + 0.5 * width, total, width, color="C2", edgecolor="0.2", lw=0.8,
           label=r"total propagation  $\mathbb{E}[\Delta]$")
    ax.set_ylim(0, max(total) * 1.6)
    ax.set_ylabel("Mean delay  (s)")
    _cat_axis(ax)
    ax.legend(fontsize=15, loc="upper left")
    for (lab, N, _, _), lm, ls_, tm in zip(ATTR_CELLS, leg, leg_s, total):
        rows.append(dict(face="attribution", N=N, cell=lab.replace("\n", " "),
                         measure="latency_leg", mean=lm, sem=ls_))
        rows.append(dict(face="attribution", N=N, cell=lab.replace("\n", " "),
                         measure="total_propagation", mean=tm, sem=np.nan))
    _save(fig, "size_bracket_attribution_latency_vs_N.png")

    by_lab = {lab: (m, s, null) for lab, m, s, null in stats}
    m1, s1, n1 = by_lab["1000"]
    m3, s3, n3 = by_lab[ATTR_CELLS[-1][0]]
    z = ((m3 - n3) - (m1 - n1)) / np.hypot(s1, s3)
    print("\nTRANSFER TEST (same design p_s = 0.1: N = 3000 vs N = 1000, "
          "latent arm, each against its own null -- E|S| ~ 301 vs ~101):")
    print(f"  additive excess:  N=1000 {m1 - n1:+.5f} +/- {s1:.5f}  |  "
          f"N=3000 {m3 - n3:+.5f} +/- {s3:.5f}  |  diff z = {z:+.2f}")
    rz = (m3 / n3 - m1 / n1) / np.hypot(s1 / n1, s3 / n3)
    print(f"  x-chance (the invariant): N=1000 {m1 / n1:.3f} +/- {s1 / n1:.3f}  |  "
          f"N=3000 {m3 / n3:.3f} +/- {s3 / n3:.3f}  |  diff z = {rz:+.2f}")
    print("  (additive excess is EXPECTED to shrink with the crowd -- the leak scales with the"
          " null; the multiplicative form is what transfers)")
    print("\nLEAK AS MULTIPLE OF CHANCE (mean / own null) per cell:")
    for lab, m, s, null in stats:
        print(f"  {lab.replace(chr(10), ' '):>22}:  {m / null:.3f}x +/- {s / null:.3f}")


def _stake_draws(N, shape, reps=STAKE_REPS):
    """Draw `reps` quenched stake vectors as a (reps, N) array (each row sums to 1)."""
    rng = np.random.default_rng([STAKE_SEED, int(N), int(round(shape * 1000))])
    return np.array([sample_relative_stakes(N, shape=shape, rng=rng) for _ in range(reps)])


def _hist_figure(N, draws):
    """
    Draw one rung's stake histogram: alpha on a log x, node count on a linear y, one curve per k.

    x is shared across rungs (LOG_XLIM); y is hand-limited per rung (HIST_YMAX), which clips
    the k = 3 peak but keeps heights comparable as fractions of N. The tail is not visible on
    a linear y and is reported numerically by the caller instead.
    """
    series =[(k, color, ls, draws[k]) for (k, color, _), ls in zip(K_SERIES, ("-", "-"))]
    edges = np.logspace(np.log10(LOG_XLIM[0]), np.log10(LOG_XLIM[1]), HIST_BINS + 1)

    fig, ax = plt.subplots()
    for k, color, ls, d in series:
        counts = np.mean([np.histogram(row, bins=edges)[0] for row in d], axis=0)
        lab = r"$k = 4/3$" if k < 2 else r"$k = 3$"
        ax.stairs(counts, edges, color=color, ls=ls, lw=2.0, label=lab)
        ax.stairs(counts, edges, color=color, fill=True, alpha=0.12, lw=0)
        if counts.max() > HIST_YMAX[N]:
            print(f"    (N = {N}, k = {k:.3f}: peak bin {counts.max():.1f} nodes is CLIPPED "
                  f"by the y-limit {HIST_YMAX[N]:.0f})")

    ax.set_xscale("log")
    ax.set_xlim(*LOG_XLIM)
    ax.set_ylim(0.0, HIST_YMAX[N])
    ax.set_xlabel(r"Relative stake  $\alpha_i$")
    ax.set_ylabel("Number of validators")
    ax.legend(fontsize=13, loc="upper right", title=rf"$N = {N}$", title_fontsize=13)
    plotstyle.save(fig, FIGDIR / f"size_bracket_stake_hist_N{N}_logx.png")


def stake_distribution_figures():
    """
    Draw one stake histogram per rung and print the tail statistics the linear y hides.

    Reports E[max alpha], E[Gini] against its 1/(2k-1) anchor, and the expected number of
    nodes richer than 10x / 100x an equal share.
    """
    print("\nSTAKE DISTRIBUTION per rung (mean over "
          f"{STAKE_REPS} quenched draws; Gini anchor G = 1/(2k-1)):")
    for N in LADDER:
        draws = {k: _stake_draws(N, k) for k, _, _ in K_SERIES}
        _hist_figure(N, draws)
        for k, d in draws.items():
            fair = d * N
            print(f"  N = {N:>4d}, k = {k:.3f}:  E[max alpha] = {d.max(axis=1).mean():.4f}   "
                  f"E[Gini] = {np.mean([gini(row) for row in d]):.3f}  "
                  f"(theory {1.0 / (2 * k - 1):.3f})   "
                  f"E[#nodes > 10x fair] = {(fair > 10).sum(axis=1).mean():.2f}   "
                  f"E[#nodes > 100x fair] = {(fair > 100).sum(axis=1).mean():.2f}")


def main():
    """Draw every figure and write the summary table."""
    rows = []
    stake_figures(rows)
    stake_distribution_figures()
    attribution_figures(rows)
    out = TABLES / "size_bracket_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"wrote figures -> {FIGDIR}")


if __name__ == "__main__":
    main()
