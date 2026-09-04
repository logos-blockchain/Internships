"""
Extend a committed stake experiment to many epochs (quenched compounding): figure + table.

No new simulation: N, T, f, M = cover_runs and the swept values are read from the named
experiment's YAML and committed table. Adds the time axis
C_quenched(E) = E_alpha[1 - (1 - p(alpha))^E], the naive curve 1 - (1 - p_bar)^E, and the
table's per-chain hit counts as the simulated leg for E <= M (theory.stake_epochs).
Usage: python experiments/stake_epochs.py [config] [--p-s X] [--e-max N] [--quick] [--plot-only].
Writes results/figures/<name>/<name>_epochs_{10yr,1yr}.png and results/tables/<name>_epochs_theory.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import plotstyle                      # noqa: E402
from tqdm import tqdm                 # noqa: E402

from experiments import load_experiment                       # noqa: E402
from theory.stake_epochs import (                             # noqa: E402
    epochs_to_level, epochs_to_years, expected_cumulative_top1, subsampled_cumulative,
)

CONFIG_DIR = REPO / "experiments" / "configs"

CONFIG = "recommendation_stake"
P_S = None
E_MAX = 812                      # ~10 years at 81.2 epochs/year
DRAWS = 1_000
LEVELS = (0.25, 0.5, 0.9)
SEED = None
PLOT_ONLY = False
TIME_MARKS = (
    ("1 day", 1.0),
    ("1 month", 30.44),
    ("1 year", 365.25),
    ("5 years", 5 * 365.25),
    ("10 years", 10 * 365.25),
)
LEGEND_FONTSIZE = 15

QUICK_DRAWS = 50


def _k_label(shape):
    """Series label for a Pareto shape: '4/3' for the Cardano anchor, else compact."""
    return "4/3" if abs(shape - 4.0 / 3.0) < 1e-9 else f"{shape:g}"


K_COLORS = ((1.1, "#440154"), (4.0 / 3.0, "#21918c"), (3.0, "#fde725"))   # viridis 0 / 0.5 / 1


def _k_color(shape, i):
    """One colour per Pareto shape across the stake figures (k = 1.1 / 4/3 / 3); else C{i}."""
    for k, c in K_COLORS:
        if abs(shape - k) < 1e-9:
            return c
    return f"C{i}"


def _compute_arms(cfg, shapes, p_s, draws, seed, e_max):
    """Evaluate the closed forms per shape (the only expensive step) -> list of arm dicts."""
    E_grid = np.unique(np.round(np.logspace(0, np.log10(e_max), 200)).astype(int))
    arms = []
    for shape in shapes:
        t0 = time.time()
        with tqdm(total=draws, desc=f"k = {_k_label(shape)}", unit="draw", leave=False) as bar:
            th = expected_cumulative_top1(
                cfg.N, p_s, E_grid, shape=shape, f=cfg.f, T=cfg.T, draws=draws,
                rng=np.random.default_rng([seed, int(shape * 1000)]),
                progress=lambda done, total: bar.update(done - bar.n))
        arms.append({"shape": shape, "epochs": E_grid, "quenched": th["quenched"],
                     "naive": th["naive"], "p_mean": th["p_mean"], "p_sem": th["p_sem"],
                     "draws": draws, "secs": time.time() - t0})
    return arms


def _load_arms(out_csv, p_s, e_max):
    """--plot-only: rebuild the arm dicts from the saved theory table (no draws)."""
    if not out_csv.exists():
        raise SystemExit(f"--plot-only: {out_csv} not found -- run once without it first")
    saved = pd.read_csv(out_csv)
    if "p_s" in saved.columns and not np.isclose(float(saved["p_s"].iloc[0]), p_s):
        raise SystemExit(f"--plot-only: saved table is at p_s = {saved['p_s'].iloc[0]}, "
                         f"requested {p_s} -- re-run without --plot-only")
    arms = []
    for shape in saved["shape"].unique():
        s = saved[saved["shape"] == shape]
        if e_max > int(s["epochs"].max()):
            raise SystemExit(f"--plot-only: saved grid ends at E = {int(s['epochs'].max())} "
                             f"< E_MAX = {e_max} -- re-run without --plot-only to extend")
        s = s[s["epochs"] <= e_max]
        arms.append({"shape": float(shape), "epochs": s["epochs"].to_numpy(),
                     "quenched": s["cum_top1_quenched"].to_numpy(),
                     "naive": s["cum_top1_naive"].to_numpy(),
                     "p_mean": float(s["p_mean"].iloc[0]), "p_sem": float(s["p_sem"].iloc[0]),
                     "draws": int(s["theory_draws"].iloc[0]), "secs": None})
    return arms


def draw_figure_10yr(arms, op, M, time_marks, yr, table_name):
    """
    Draw the decade frame (E out to 10 years), resolving both quenched crossings in-frame.

    A standalone plotting function, deliberately not shared with the 1-year one.
    """
    e_max = int(round(10.0 / yr))
    fig, ax = plt.subplots()
    for i, arm in enumerate(arms):
        color, marker = _k_color(arm["shape"], i), "os^v"[i % 4]
        emp_rates = op[np.isclose(op["shape"], arm["shape"])]["stake_top1_hit"].to_numpy()
        if emp_rates.size == 0:
            raise SystemExit(f"no rows at shape = {arm['shape']} in {table_name}")
        E_emp = np.arange(1, M + 1)
        leg = subsampled_cumulative(emp_rates * M, M, E_emp)
        ax.plot(arm["epochs"], arm["quenched"], "-", color=color)
        ax.plot(arm["epochs"], arm["naive"], ":", color=color)
        ax.plot(E_emp, leg, marker, color=color, mfc="none", ms=6, ls="none")

    ax.axhline(0.5, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(1, e_max)
    ax.set_ylim(0, 1)
    for label, days in time_marks:
        if label == "1 month":
            continue
        e_mark = days / (yr * 365.25)
        if 1.0 <= e_mark <= e_max:
            ax.axvline(e_mark, color="0.55", lw=0.9, ls="-.", zorder=0)
            ax.text(e_mark, 0.02, label, rotation=90, fontsize=LEGEND_FONTSIZE, color="0.35",
                    ha="right", va="bottom")
        else:
            print(f"  (10yr figure: time mark '{label}' = {e_mark:.2f} epochs is outside "
                  f"the frame [1, {e_max}] -- skipped)")
    ax.set_xlabel(rf"epochs observed $E$  (1 epoch $= {yr * 365.25:.1f}$ days)")
    ax.set_ylabel(r"Largest stakeholder identified  $C(E)$")
    top = ax.secondary_xaxis("top", functions=(lambda e: e * yr, lambda y: y / yr))
    top.set_xlabel("years of constant observation")
    n = len(arms)
    leg_handles = [
        Line2D([], [], ls="none"),
        tuple(Line2D([], [], color=_k_color(arms[i]["shape"], i), ls="-") for i in range(n)),
        tuple(Line2D([], [], color=_k_color(arms[i]["shape"], i), ls=":") for i in range(n)),
        tuple(Line2D([], [], color=_k_color(arms[i]["shape"], i), ls="none", marker="os^v"[i % 4], mfc="none",
                     ms=6) for i in range(n)),
    ]
    leg_labels = ["", "closed form (quenched)", "annealed bound", "simulated"]
    main = ax.legend(leg_handles, leg_labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                     fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax.add_artist(main)
    k_handles = [Patch(facecolor=_k_color(arm["shape"], i), label=rf"$k = {_k_label(arm['shape'])}$")
                 for i, arm in enumerate(arms)]
    ax.legend(handles=k_handles, ncol=n, fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=False,
              handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(0.99, 0.995))
    return fig


def draw_figure_1yr(arms, op, M, time_marks, yr, table_name):
    """
    Draw the one-year frame (E out to ~81 epochs), where the simulated leg E <= M is visible.

    A standalone plotting function, deliberately not shared with the 10-year one.
    """
    e_max = int(round(1.01 / yr))
    fig, ax = plt.subplots()
    for i, arm in enumerate(arms):
        color, marker = _k_color(arm["shape"], i), "os^v"[i % 4]
        emp_rates = op[np.isclose(op["shape"], arm["shape"])]["stake_top1_hit"].to_numpy()
        if emp_rates.size == 0:
            raise SystemExit(f"no rows at shape = {arm['shape']} in {table_name}")
        E_emp = np.arange(1, M + 1)
        leg = subsampled_cumulative(emp_rates * M, M, E_emp)
        ax.plot(arm["epochs"], arm["quenched"], "-", color=color)
        ax.plot(arm["epochs"], arm["naive"], ":", color=color)
        ax.plot(E_emp, leg, marker, color=color, mfc="none", ms=6, ls="none")

    ax.axhline(0.5, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(1, e_max)
    ax.set_ylim(0, 1)
    for label, days in time_marks:
        e_mark = days / (yr * 365.25)
        if 1.0 <= e_mark <= e_max:
            ax.axvline(e_mark, color="0.55", lw=0.9, ls="-.", zorder=0)
            ax.text(e_mark, 0.02, label, rotation=90, fontsize=LEGEND_FONTSIZE, color="0.35",
                    ha="right", va="bottom")
        else:
            print(f"  (1yr figure: time mark '{label}' = {e_mark:.2f} epochs is outside "
                  f"the frame [1, {e_max}] -- skipped)")
    ax.set_xlabel(rf"epochs observed $E$  (1 epoch $= {yr * 365.25:.1f}$ days)")
    ax.set_ylabel(r"Largest stakeholder identified  $C(E)$")
    top = ax.secondary_xaxis("top", functions=(lambda e: e * yr, lambda y: y / yr))
    top.set_xlabel("years of constant observation")
    n = len(arms)
    leg_handles = [
        Line2D([], [], ls="none"),
        tuple(Line2D([], [], color=_k_color(arms[i]["shape"], i), ls="-") for i in range(n)),
        tuple(Line2D([], [], color=_k_color(arms[i]["shape"], i), ls=":") for i in range(n)),
        tuple(Line2D([], [], color=_k_color(arms[i]["shape"], i), ls="none", marker="os^v"[i % 4], mfc="none",
                     ms=6) for i in range(n)),
    ]
    leg_labels = ["", "closed form (quenched)", "annealed bound", "simulated"]
    main = ax.legend(leg_handles, leg_labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                     fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax.add_artist(main)
    k_handles = [Patch(facecolor=_k_color(arm["shape"], i), label=rf"$k = {_k_label(arm['shape'])}$")
                 for i, arm in enumerate(arms)]
    ax.legend(handles=k_handles, ncol=n, fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=False,
              handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(0.99, 0.995))
    return fig


def main(argv=None):
    """Parse arguments, evaluate (or reload) the arms, print the crossings and draw both frames."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("config", nargs="?", default=CONFIG,
                    help="experiment name in experiments/configs/ (default: CONFIG above)")
    ap.add_argument("--p-s", type=float, default=P_S, dest="p_s",
                    help="operating-point cover rate; must be a swept value of the table")
    ap.add_argument("--e-max", type=int, default=E_MAX,
                    help="compute-grid horizon in epochs (must cover the 10-year figure)")
    ap.add_argument("--draws", type=int, default=DRAWS, help="quenched stake draws per shape")
    ap.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                    help="cumulative levels whose crossing epochs to report")
    ap.add_argument("--seed", type=int, default=SEED, help="override the experiment's seed")
    ap.add_argument("--quick", action="store_true", help=f"smoke run ({QUICK_DRAWS} draws/shape)")
    ap.add_argument("--plot-only", action="store_true", default=PLOT_ONLY,
                    help="redraw figure from the saved theory table instead of re-drawing stakes")
    args = ap.parse_args(argv)

    exp = load_experiment(CONFIG_DIR / f"{Path(args.config).stem}.yaml")
    cfg = exp.base_cfg
    seed = exp.seed if args.seed is None else args.seed
    draws = QUICK_DRAWS if args.quick else args.draws
    p_s = cfg.dummy.p_s if args.p_s is None else args.p_s
    shapes = list(exp.axes.get("shape", [cfg.shape]))
    M = int(cfg.cover_runs)
    if M <= 1:
        raise SystemExit(f"{exp.name}: cover_runs = {M} -- the empirical leg needs the "
                         "R x M quenched structure (per-chain hit counts over M epochs)")
    if "p_s" in exp.axes and not any(np.isclose(p_s, v) for v in exp.axes["p_s"]):
        raise SystemExit(f"p_s = {p_s} is not a swept value of {exp.name}: {exp.axes['p_s']}")

    table = REPO / "results" / "tables" / f"{exp.name}.csv"
    if not table.exists():
        raise SystemExit(f"{table} not found -- run `python experiments/run.py {exp.name}` "
                         "first (this script extends a committed result, it does not create one)")
    df = pd.read_csv(table)
    op = df[np.isclose(df["p_s"], p_s)] if "p_s" in df.columns else df

    out_csv = REPO / "results" / "tables" / f"{exp.name}_epochs_theory.csv"
    yr = float(epochs_to_years(1, cfg.T))                    # years per epoch
    print(f"{exp.name} @ p_s = {p_s}: N = {cfg.N}, T = {cfg.T} "
          f"(1 epoch = {yr * 365.25:.1f} d), M = {M}, shapes = {shapes}, "
          + (f"plot-only from {out_csv.name}" if args.plot_only
             else f"{draws} draws/shape, seed = {seed}"))

    if args.plot_only:
        arms = _load_arms(out_csv, p_s, args.e_max)
    else:
        arms = _compute_arms(cfg, shapes, p_s, draws, seed, args.e_max)

    for arm in arms:
        cross = {lev: (epochs_to_level(arm["epochs"], arm["quenched"], lev),
                       epochs_to_level(arm["epochs"], arm["naive"], lev))
                 for lev in args.levels}
        msg = " | ".join(f"{lev:.0%}: quenched {q:.1f} ep ({q * yr:.2f} yr) vs naive {n:.1f} ep"
                         for lev, (q, n) in cross.items())
        timing = "" if arm["secs"] is None else f" ({arm['secs']:.0f}s)"
        print(f"  k = {_k_label(arm['shape'])}{timing}: E[p] = {arm['p_mean']:.5f} "
              f"+/- {arm['p_sem']:.5f} | {msg}")

    if not args.plot_only:
        rows = [{"shape": arm["shape"], "epochs": int(e), "years": float(e * yr),
                 "cum_top1_quenched": cq, "cum_top1_naive": cn,
                 "p_mean": arm["p_mean"], "p_sem": arm["p_sem"],
                 "p_s": p_s, "theory_draws": arm["draws"]}
                for arm in arms
                for e, cq, cn in zip(arm["epochs"], arm["quenched"], arm["naive"])]
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")
    for fn, tag in ((draw_figure_10yr, "10yr"), (draw_figure_1yr, "1yr")):
        fig = fn(arms, op, M, TIME_MARKS, yr, table.name)
        out_png = REPO / "results" / "figures" / exp.name / f"{exp.name}_epochs_{tag}.png"
        plotstyle.save(fig, out_png)
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
