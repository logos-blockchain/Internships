"""
Run an experiment declared in a YAML config: sweep -> results table + figures.

Set CONFIG below and run the file, or pass a config name on the command line
(`python experiments/run.py mixnet`; `--list` shows the available configs, `--quick`
runs a short smoke sweep). The sweep is written to results/tables/<name>.csv and the
figures are drawn from the config's plot spec; all matplotlib code lives here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
import plotstyle  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binom, poisson  # noqa: E402
from experiments import load_experiment, make_apply_cell, sweep  # noqa: E402
from consensus import (  # noqa: E402
    expected_winners_per_slot, sample_relative_stakes, simulate_max_relative_stake,
)
from anonymity import delay_moments  # noqa: E402
from network.latency import broadcast_latency_theory, lam_from_rho  # noqa: E402
from network.lognormal_latency import link_moments  # noqa: E402
from network.latency_profile import (  # noqa: E402
    is_homogeneous, profile_broadcast_latency, sample_latency_profile)

CONFIG_DIR = REPO / "experiments" / "configs"
QUICK_T = 20_000

# Config menu for running this file without arguments (e.g. the VS Code Run button).
# Every experiment in experiments/configs/ is listed; uncomment exactly one CONFIG line.
# A config name given on the command line overrides it.

# --- Your own experiment: copy a template to experiments/configs/<my_experiment>.yaml ---
# CONFIG = "my_experiment"
# CONFIG = "template_stake_inference"          # documented template, stake-inference (scalar) attack
# CONFIG = "template_bayesian_inference"       # documented template, Bayesian attribution (posterior) attack

# --- Stake inference (count-based fast path) ---
# CONFIG = "single_path_mix_stake"             # stake privacy vs cover rate p_s, one curve per Pareto shape k
# CONFIG = "recommendation_stake"              # recommended operating point, stake arm: p_s ladder at k = 3 and 4/3
# CONFIG = "recommendation_stake_N100"         # size bracket, N = 100: same p_s x shape grid
# CONFIG = "recommendation_stake_N3000"        # size bracket, N = 3000: p_s ladder plus the held-|S| cell

# --- Sender attribution, single-path mix (W = 1) ---
# CONFIG = "single_path_mix_lognormal_attribution"  # attribution vs mean delay per mix, log-normal links
# CONFIG = "single_path_mix_depth_attribution"      # same, one curve per depth L = 1..4
# CONFIG = "single_path_mix_cover_attribution"      # attribution vs cover rate p_s, one curve per mix delay
# CONFIG = "single_path_mix_jitter_sweep"           # per-message link jitter x mixing budget

# --- Sender attribution, stratified mix-net (W x k grid) ---
# CONFIG = "mixnet_attribution_split"          # 2 x 3 grid, split entry assignment (the shipped model), vs mix delay
# CONFIG = "mixnet_attribution"                # 2 x 3 grid, uniform entry assignment (control arm), vs mix delay
# CONFIG = "mixnet_attribution_bracket"        # route-latent vs route-oracle adversary: lower/upper band
# CONFIG = "mixnet_width_sweep"                # width W = 1..40 at k = 3, vs mix delay
# CONFIG = "mixnet_depth_sweep"                # depth k = 1..6 at W = 2, vs mix delay
# CONFIG = "mixnet_assignment_vs_width"        # split vs uniform assignment as W = 1..10 grows, pinned budget
# CONFIG = "mixnet_geometry_sweep"             # (W, k) surface at one frozen mixing budget
# CONFIG = "mixnet_geometry_vs_budget"         # W x k x mix delay, uniform assignment (three axes)
# CONFIG = "mixnet_jitter_sweep"               # per-message link jitter x mixing budget on the 2 x 3 split grid

# --- Recommended operating point, attribution arm, and its size bracket ---
CONFIG = "recommendation_attribution"          # W x mix delay grid at p_s = 0.1 under both limiting adversaries
# CONFIG = "recommendation_attribution_N100"   # size bracket, N = 100: p_s x width x attack
# CONFIG = "recommendation_attribution_N3000"  # size bracket, N = 3000: one cell under both adversaries, no figures

# Run-mode defaults, used when the file is run without flags (e.g. an editor's "Run File").
# Each has a command-line override: --quick, --plot-only, --jobs N, --mem-fraction F.
QUICK = False          # True: smoke run with T = QUICK_T slots and reps = 1; outputs get a _quick suffix
PLOT_ONLY = False      # True: skip the sweep and redraw the figures from results/tables/<name>.csv
JOBS = -1              # worker processes per sweep cell: 1 = serial; <= 0 = auto (CPU cores - 2,
                       # capped at 16). The reps of one cell run in parallel, one process each,
                       # so a cell never uses more workers than it has reps.
MEM_FRACTION = 0.70    # fraction of usable RAM (min of total and currently free) the pool may
                       # use. Workers per cell = min(JOBS, MEM_FRACTION * RAM / estimated peak
                       # memory of one run, reps), so heavy cells (large N, wide mix-nets) get
                       # fewer workers. Lower it if the machine swaps; ~0.8 is safe on 16 GB.
                       # Neither knob changes a result: every (cell, rep) has its own seed.


def _resolve(name):
    """Bare config name -> experiments/configs/<name>.yaml; an explicit path -> itself."""
    p = Path(name)
    if p.suffix != ".yaml":
        p = p.with_suffix(".yaml")
    return p if p.parent != Path(".") else CONFIG_DIR / p.name


DEFAULT_CONFIG = _resolve(CONFIG)


def list_configs():
    """Print every config in experiments/configs/ with its name (marks the default)."""
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    if not files:
        print(f"no configs in {CONFIG_DIR.relative_to(REPO)}")
        return
    print(f"configs in {CONFIG_DIR.relative_to(REPO)} (set CONFIG in run.py, or pass one as an arg):")
    for p in files:
        name = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("name", p.stem)
        marker = "  <- CONFIG (Run button)" if p == DEFAULT_CONFIG else ""
        print(f"  {p.name:<22} {name}{marker}")


def _broadcast_mean(cfg):
    """
    Return E[D_br(N)], the broadcast-latency denominator used by the ell values.

    Homogeneous law: the closed form d * E[ecc]. Heterogeneous law: the mean weighted
    eccentricity over a few fixed-seed realisations (no closed form).
    """
    lat = cfg.latency
    if is_homogeneous(lat):
        return broadcast_latency_theory(cfg.N, cfg.C, cfg.d, lam_from_rho(cfg.rho, cfg.d))
    rng = np.random.default_rng(0)
    lam = lam_from_rho(cfg.rho, cfg.d)
    vals = []
    for _ in range(8):
        prof = sample_latency_profile(cfg.N, 1, cfg.d, lat, rng=rng)
        vals.append(profile_broadcast_latency(cfg.N, C=cfg.C, d=cfg.d, lam=lam, params=lat,
                                              profile=prof, rng=rng))
    return float(np.mean(vals))


def _latency_feasibility(cfg):
    """
    Return the E[Delta] < 1/f feasibility ceiling in ell units: ell < (1/f) / E[D_br(N)].

    E[D^AC] = ell * E[D_br(N)] must stay below the slot length 1/f.
    """
    return {"y": (1.0 / cfg.f) / _broadcast_mean(cfg),
            "label": r"$\mathbb{E}[\Delta]=1/f$ ceiling (feasible below)"}


def _top1_chance(cfg):
    """Return the random-guess baseline 1/N for stake_top1_hit."""
    return {"y": 1.0 / cfg.N, "label": r"$1/N$ (random guess)"}


def _winners_per_slot_rate(cfg):
    """
    Return lambda = E[winners per slot] in the small-stake limit, -ln(1-f).

    sum_i phi(alpha_i) <= -ln(1-f) with phi(alpha) = 1 - (1-f)^alpha; the gap is
    O(sum_i phi_i^2), and the limit keeps the anchor a config constant.
    """
    return -math.log1p(-cfg.f)


def _candidate_set_law(cfg):
    """
    Return the per-broadcast law of the candidate set size |S_t| as (sizes, probs).

    Scoring is per broadcast, so the winner count W is size-biased: W = 1 + Poisson(lambda).
      count cover -> |S_t| = count + 1 + Poisson(lambda)
      p_s cover   -> |S_t| = W + Binomial(N - W, p_s)   (a slot's winners never cover for it)
    """
    lam = _winners_per_slot_rate(cfg)
    j = np.arange(int(poisson.isf(1e-15, lam)) + 1)
    pw = poisson.pmf(j, lam)
    pw /= pw.sum()
    w = j + 1
    if cfg.dummy.p_s is None:
        return (cfg.dummy.count + w).astype(float), pw
    sizes, probs = [], []
    for w_i, p_i in zip(w, pw):
        n = cfg.N - int(w_i)
        m = np.arange(n + 1)
        sizes.append(w_i + m)
        probs.append(p_i * binom.pmf(m, n, float(cfg.dummy.p_s)))
    return np.concatenate(sizes).astype(float), np.concatenate(probs)


ORACLE_ATTACK = "mixnet_attribution_oracle"


def _entry_filtered_law(cfg):
    """
    Return the candidate-set law faced by the route-oracle arm: |S_t ∩ e| = 1 + Binomial(|S_t| - 1, 1/W).

    Under `split` each rival shares the observed entry independently w.p. 1/W, so this is
    `_candidate_set_law` thinned. Its anchors depend on W where the latent ones do not.
    Returns the latent law unchanged when W = 1 or the assignment is not `split`.
    """
    sizes, probs = _candidate_set_law(cfg)
    lp = getattr(cfg, "layer_params", None)
    W = int(getattr(lp, "width", 1) or 1)
    if W <= 1 or getattr(lp, "entry_assignment", None) != "split":
        return sizes, probs
    keep = probs > 1e-15
    sizes, probs = sizes[keep], probs[keep]
    out_s, out_p = [], []
    for s, p in zip(sizes, probs):
        rivals = int(round(float(s))) - 1
        m = np.arange(rivals + 1)
        out_s.append(1.0 + m)
        out_p.append(p * binom.pmf(m, rivals, 1.0 / W))
    return np.concatenate(out_s), np.concatenate(out_p)


def _measure_law(cfg):
    """Return the candidate-set law the cell's attack faces (entry-filtered for the oracle arm)."""
    return _entry_filtered_law(cfg) if getattr(cfg, "attack", None) == ORACLE_ATTACK \
        else _candidate_set_law(cfg)


def _deanon_chance(cfg):
    """
    Return the random-guess baseline E[1/|S_t|] for deanon_top1.

    A uniform posterior scores 1/|S_t| on each broadcast, so the anchor is the mean reciprocal.
    """
    sizes, probs = _measure_law(cfg)
    return {"y": float((probs / sizes).sum()), "label": r"$\mathbb{E}[1/|S_t|]$ (random guess)"}


def _true_post_floor(cfg):
    """Return the no-information floor E[1/|S_t|] for mean_true_posterior."""
    sizes, probs = _measure_law(cfg)
    return {"y": float((probs / sizes).sum()), "label": r"$\mathbb{E}[1/|S_t|]$ (uniform posterior)"}


def _entropy_ceiling(cfg):
    """Return the uniform-posterior ceiling E[log2|S_t|] for posterior_entropy."""
    sizes, probs = _measure_law(cfg)
    return {"y": float((probs * np.log2(sizes)).sum()), "label": r"$\mathbb{E}[\log_2|S_t|]$ (uniform posterior)"}


def _link_law_sigma(cfg):
    """
    Return the population sigma_d of the link law for the sigma_hat_d panel.

    This is the `split` estimand at every W; `uniform`'s estimand is sigma_d / sqrt(W) and is
    drawn as a theory curve. The plotted estimates sit below it by the finite-m bias.
    """
    lat = getattr(cfg, "latency", None)
    link = getattr(lat, "lognormal", None) if lat is not None else None
    if link is None:
        return None
    return {"y": float(np.sqrt(link_moments(link)[1])),
            "label": r"standard deviation of the link law"}


def _jaccard_chance(cfg):
    """
    Return the random-set reference for stake_top_jaccard (theory.random_set_jaccard).

    A uniformly random top-m set overlaps the true one by K ~ Hypergeom(N, m, m), so E[J] > 0.
    """
    from theory import random_set_jaccard, top_set_size

    m = top_set_size(cfg.N, cfg.top_frac)
    return {"y": float(random_set_jaccard(cfg.N, m)),
            "label": r"random set of $m$ (random guess)".replace("$m$", f"${m}$")}


def _ac_path_limit(cfg):
    """
    Return the feasibility ceiling on the AC-path delay E[Y] (the `latency` column, in seconds).

    E[Delta] = E[Y] + E[D_br] < 1/f  <=>  E[Y] < 1/f - E[D_br(N)]; the same constraint as
    _latency_feasibility.
    """
    return {"y": 1.0 / cfg.f - _broadcast_mean(cfg),
            "label": r"latency limit  $1/f - \mathbb{E}[D_{\mathrm{br}}]$"}


REFERENCE_LINES = {
    "stake_top1_hit": _top1_chance,
    "stake_top_jaccard": _jaccard_chance,
    "latency_overhead": _latency_feasibility,
    "latency": _ac_path_limit,
    "deanon_top1": _deanon_chance,
    "mean_true_posterior": _true_post_floor,
    "posterior_entropy": _entropy_ceiling,
    "sigma_d": _link_law_sigma,
}


def _perfect_attribution(cfg, *, value, label):
    """
    Return the sigma_Z -> 0 anchor where no delay noise means certain attribution.

    Under a continuous link law the true sender's latency is a.s. unique, so top-1 = 1 and
    entropy = 0. Under a homogeneous law the limit is the random-guess floor: returns None.
    """
    lat = cfg.latency
    if lat is None or is_homogeneous(lat):
        return None
    return {"y": float(value), "label": label}


LIMIT_LINES = {
    "deanon_top1": lambda cfg: _perfect_attribution(cfg, value=1.0, label=r"no intentional delay"),
    "mean_true_posterior": lambda cfg: _perfect_attribution(cfg, value=1.0, label=r"no intentional delay"),
    "posterior_entropy": lambda cfg: _perfect_attribution(cfg, value=0.0, label=r"no intentional delay"),
}


def _beta_from_ps(cfg, ps):
    """Return beta = 1 + p_s(N/Phi - 1), Phi = sum_i phi(alpha_i), for each p_s in ps."""
    alpha = sample_relative_stakes(cfg.N, cfg.shape, rng=np.random.default_rng(0))
    Phi = expected_winners_per_slot(alpha, cfg.f)
    return [1.0 + float(p) * (cfg.N / Phi - 1.0) for p in ps]


def _beta_theory(cfg, xs, **_):
    """Return the bandwidth closed form beta = 1 + p_s(N/Phi - 1) as a single curve."""
    return {"x": list(xs), "y": _beta_from_ps(cfg, xs),
            "label": r"theory $\beta = 1 + p_s(N/\Phi - 1)$"}


_TOP1_THEORY_CACHE = {}
_CONF_THEORY_CACHE = {}
_JAC_THEORY_CACHE = {}
_SWEEP_STAKES_CACHE = {}


def _sweep_stake_vectors(exp, base_cfg, df, seed, reps):
    """
    Regenerate the sweep's own per-(cell, rep) stake vectors -> [(cell_dict, [alpha_0 .. alpha_R])].

    Evaluating a closed form on the simulation's own stake vectors makes the comparison paired:
    the quenched stake term cancels. The sweep seeds (cell, rep) by index
    (`SeedSequence(seed).spawn(n_cells * reps)`, cell-major) and the stake vector is the first
    draw off each child, so the vectors are reproducible without re-running the sweep. One
    (cell, rep) is re-run through run_once and must reproduce its table row exactly, else raises.
    """
    key = (exp.name, int(seed), int(reps))
    if key in _SWEEP_STAKES_CACHE:
        return _SWEEP_STAKES_CACHE[key]

    from experiments import run_once

    names = list(exp.axes)
    cells = list(itertools.product(*(list(exp.axes[n]) for n in names)))
    child = np.random.SeedSequence(seed).spawn(len(cells) * reps)
    apply_cell = make_apply_cell(exp.path_map)

    cell0 = dict(zip(names, cells[0]))
    cfg0 = apply_cell(base_cfg, cell0)
    t0 = time.perf_counter()
    got = run_once(cfg0, rng=np.random.default_rng(child[0]))
    mask = df["rep"].to_numpy() == 0
    for n, v in cell0.items():
        mask &= np.isclose(df[n].to_numpy(dtype=float), float(v))
    if not mask.any():
        raise SystemExit(f"theory_match_stakes: no table row for cell {cell0} rep 0 -- the table "
                         "was not produced by this config, so its stake draws cannot be matched.")
    row = df[mask].iloc[0]
    bad = {m: (float(row[m]), float(got[m])) for m in got
           if m in df.columns and np.isfinite(float(row[m]))
           and not np.isclose(float(row[m]), float(got[m]), rtol=1e-9, atol=1e-12)}
    if bad:
        raise SystemExit(
            "theory_match_stakes: the sweep's RNG could NOT be reproduced -- re-running cell "
            f"{cell0} rep 0 gave {bad} (table value, re-run value). The regenerated stake vectors "
            "would not be the ones the table was produced with, so the 'paired' comparison would "
            "silently be unpaired. Drop `theory_match_stakes` from the plot spec (the figure falls "
            "back to independent draws) or re-run the sweep.")
    print(f"        matched stakes: sweep RNG reproduced at cell {cell0} rep 0 "
          f"({time.perf_counter() - t0:.1f}s)")

    out = []
    for c, values in enumerate(cells):
        cell = dict(zip(names, values))
        cfg = apply_cell(base_cfg, cell)
        out.append((cell, [sample_relative_stakes(cfg.N, cfg.shape,
                                                  rng=np.random.default_rng(child[c * reps + r]))
                           for r in range(reps)]))
    _SWEEP_STAKES_CACHE[key] = out
    return out


def _table_fingerprint(csv_path):
    """
    Return a content hash of the results CSV, the paired theory archive's validity key.

    The file is hashed rather than the DataFrame so a sweep run and a later --plot-only run agree.
    """
    # the prefix keeps an all-digit hash from being read back from the CSV as an int
    return "sha256:" + hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:16]


def _theory_from_cache(cache, plot, cfg, exp, ser, svals, xs, xcol, x_name, table_hash):
    """
    Rebuild a measure's theory curves from an archived `<name>_theory.csv`, or return None.

    Reuse requires the archived provenance (measure, N, T, f, paired mode, draw count) to match
    this figure and the rows to cover every (series, x) point; otherwise the caller recomputes.
    """
    def _reject(why):
        """Print why the archive was not reused and return None (the caller recomputes)."""
        print(f"        theory: archive NOT reused -- {why}; recomputing")
        return None

    if cache is None or cache.empty or "measure" not in cache.columns:
        return None
    if not (cache["measure"] == plot["y"]).any():
        return None

    want = {"measure": plot["y"], "N": int(cfg.N), "T": int(cfg.T),
            "paired": bool(plot.get("theory_match_stakes"))}
    m = np.ones(len(cache), dtype=bool)
    for col, v in want.items():
        if col not in cache.columns:
            return _reject(f"archive has no {col!r} column (written by an older version)")
        m &= (cache[col].astype(type(v)) == v) if col != "measure" else (cache[col] == v)
    m &= np.isclose(cache["f"].to_numpy(float), float(cfg.f))
    if want["paired"]:
        if "table_hash" not in cache.columns:
            return _reject("archive predates the table-hash guard")
        hit = cache["table_hash"].astype(str).to_numpy() == str(table_hash)
        if m.any() and not (m & hit).any():
            return _reject(f"the results table CHANGED since the archive was written "
                           f"(now {table_hash}, archived "
                           f"{sorted(set(cache.loc[m, 'table_hash'].astype(str)))[:2]}) -- a new "
                           "seed / reps / sweep axis moves the stake draws, so the archived "
                           "pairing no longer applies")
        m &= hit
    else:
        m &= cache["stake_draws"].to_numpy(int) == int(plot.get("theory_draws", 32))
    sub = cache[m]
    if sub.empty:
        return _reject(f"no archived rows match this figure's parameters "
                       f"(measure={plot['y']}, N={cfg.N}, T={cfg.T}, paired={want['paired']})")

    ks = [float(v) for v in svals] if (ser and svals) else [float(cfg.shape)]
    xps = [float(x) for x in xs]
    xvals = _beta_from_ps(cfg, xps) if xcol == "bandwidth_overhead" else xps
    out = []
    for k in ks:
        ys = []
        for p_s in xps:
            cell = sub[np.isclose(sub["series_val"].to_numpy(float), k)
                       & np.isclose(sub["x_val"].to_numpy(float), p_s)]
            if cell.empty:
                return _reject(f"archive does not cover {ser}={k:g}, {x_name}={p_s:g} "
                               "(a new series value or sweep point)")
            ys.append(float(cell["theory"].mean()))
        out.append({"x": list(xvals), "y": ys, "series_value": k,
                    "label": None if len(ks) > 1 else r"theory (closed form)"})
    if want["paired"] and sub["sim"].notna().any():
        d = (sub["theory"] - sub["sim"]).dropna().to_numpy(float)
        print(f"        PAIRED theory - simulation over {d.size} matched (cell, rep) runs: "
              f"mean {d.mean():+.5f} +/- {d.std(ddof=1) / np.sqrt(d.size):.5f} (sem), "
              f"max |diff| {np.abs(d).max():.5f}")
    return out


def _stake_top1_theory(cfg, xs, *, series=None, series_vals=None, xcol=None, draws=32,
                       stakes=None, x_name=None, df=None, measure=None, table_hash=None, **_):
    """
    Return the closed-form stake_top1_hit curve, one per Pareto k (theory.expected_top1).

    P_Top1 is an order statistic on the participation counts n_i ~ Binomial(T, q_i),
    q_i = p_s + (1 - p_s) phi(alpha_i), averaged over stake realisations. Unpaired mode draws
    `draws` fresh stake vectors; paired mode (`stakes` given) evaluates the form on the sweep's
    own per-(cell, rep) vectors and prints the paired theory - simulation difference.
    """
    from theory import expected_top1, top1_probability

    ks = [float(v) for v in series_vals] if (series == "shape" and series_vals) else [cfg.shape]
    xps = [float(x) for x in xs]
    xvals = _beta_from_ps(cfg, xps) if xcol == "bandwidth_overhead" else xps
    xn = x_name or "p_s"

    def _matched(k, p_s):
        """Evaluate the closed form on the sweep's own stake vectors -> (mean, per-rep list)."""
        for cell, alphas in stakes:
            if (np.isclose(float(cell.get(series, k)), k)
                    and np.isclose(float(cell.get(xn, p_s)), p_s)):
                vals = [top1_probability(a, p_s, f=cfg.f, T=cfg.T) for a in alphas]
                return float(np.mean(vals)), vals
        raise SystemExit(f"theory_match_stakes: no swept cell at {series}={k}, {xn}={p_s}")

    n_stakes = len(stakes[0][1]) if stakes else int(draws)
    prov = {"measure": measure or "stake_top1_hit", "series_col": series, "x_col": xn,
            "N": int(cfg.N), "T": int(cfg.T), "f": float(cfg.f),
            "paired": bool(stakes is not None), "stake_draws": int(n_stakes),
            "table_hash": str(table_hash) if (stakes is not None and table_hash) else ""}

    out, pairs = [], []
    t0 = time.perf_counter()
    for k in ks:
        ys, sems, recs = [], [], []
        for p_s in xps:
            if stakes is not None:
                mean, vals = _matched(k, p_s)
                ys.append(mean)
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                sim = np.full(len(vals), np.nan)
                if df is not None:
                    m = np.isclose(df[series].to_numpy(float), k) & np.isclose(
                        df[xn].to_numpy(float), p_s)
                    s = df[m].sort_values("rep")[prov["measure"]].to_numpy(float)
                    if s.size == len(vals):
                        sim = s
                        pairs.append(np.asarray(vals) - s)
                recs += [{**prov, "series_val": k, "x_val": p_s, "rep": r,
                          "theory": v, "sim": sim[r], "theory_sem": float("nan")}
                         for r, v in enumerate(vals)]
            else:
                key = (cfg.N, k, p_s, cfg.f, cfg.T, int(draws))
                if key not in _TOP1_THEORY_CACHE:
                    _TOP1_THEORY_CACHE[key] = expected_top1(
                        cfg.N, p_s, shape=k, f=cfg.f, T=cfg.T, draws=int(draws),
                        rng=np.random.default_rng(0))
                r = _TOP1_THEORY_CACHE[key]
                ys.append(r["mean"])
                sems.append(r["sem"])
                recs.append({**prov, "series_val": k, "x_val": p_s, "rep": -1,
                             "theory": r["mean"], "sim": float("nan"), "theory_sem": r["sem"]})
        out.append({"x": list(xvals), "y": ys, "series_value": k, "records": recs,
                    "label": None if len(ks) > 1 else r"theory (closed form)"})
        print(f"        theory k={k:.4g}: max sem {max(sems):.4f}")
    mode = "PAIRED on the sweep's own stakes" if stakes is not None else f"{int(draws)} fresh draws"
    print(f"        theory curve(s) in {time.perf_counter() - t0:.1f}s "
          f"({len(ks) * len(xps)} points, {mode})")
    if pairs:
        d = np.concatenate(pairs)
        print(f"        PAIRED theory - simulation over {d.size} matched (cell, rep) runs: "
              f"mean {d.mean():+.5f} +/- {d.std(ddof=1) / np.sqrt(d.size):.5f} (sem), "
              f"max |diff| {np.abs(d).max():.5f}")
    return out


def _stake_confidence_theory(cfg, xs, *, series=None, series_vals=None, xcol=None, draws=32,
                             stakes=None, x_name=None, df=None, measure=None, table_hash=None,
                             **_):
    """
    Return the closed-form stake_confidence curve, one per Pareto k (theory.expected_confidence).

    The +/-gamma band is an interval in n_i ~ Binomial(T, q_i), so the measure is a mean of N
    Binomial tails (not metrics.stake_privacy.inference_confidence, the no-cover form).
    Paired/unpaired modes and the archived record schema follow _stake_top1_theory.
    """
    from theory import confidence_probability, expected_confidence

    ks = [float(v) for v in series_vals] if (series == "shape" and series_vals) else [cfg.shape]
    xps = [float(x) for x in xs]
    xvals = _beta_from_ps(cfg, xps) if xcol == "bandwidth_overhead" else xps
    xn = x_name or "p_s"
    gamma = float(cfg.gamma)

    def _matched(k, p_s):
        """Evaluate the closed form on the sweep's own stake vectors -> (mean, per-rep list)."""
        for cell, alphas in stakes:
            if (np.isclose(float(cell.get(series, k)), k)
                    and np.isclose(float(cell.get(xn, p_s)), p_s)):
                vals = [confidence_probability(a, p_s, gamma=gamma, f=cfg.f, T=cfg.T)
                        for a in alphas]
                return float(np.mean(vals)), vals
        raise SystemExit(f"theory_match_stakes: no swept cell at {series}={k}, {xn}={p_s}")

    n_stakes = len(stakes[0][1]) if stakes else int(draws)
    prov = {"measure": measure or "stake_confidence", "series_col": series, "x_col": xn,
            "N": int(cfg.N), "T": int(cfg.T), "f": float(cfg.f),
            "paired": bool(stakes is not None), "stake_draws": int(n_stakes),
            "table_hash": str(table_hash) if (stakes is not None and table_hash) else ""}

    out, pairs = [], []
    t0 = time.perf_counter()
    for k in ks:
        ys, sems, recs = [], [], []
        for p_s in xps:
            if stakes is not None:
                mean, vals = _matched(k, p_s)
                ys.append(mean)
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                sim = np.full(len(vals), np.nan)
                if df is not None:
                    m = np.isclose(df[series].to_numpy(float), k) & np.isclose(
                        df[xn].to_numpy(float), p_s)
                    s = df[m].sort_values("rep")[prov["measure"]].to_numpy(float)
                    if s.size == len(vals):
                        sim = s
                        pairs.append(np.asarray(vals) - s)
                recs += [{**prov, "series_val": k, "x_val": p_s, "rep": r,
                          "theory": v, "sim": sim[r], "theory_sem": float("nan")}
                         for r, v in enumerate(vals)]
            else:
                key = (cfg.N, k, p_s, cfg.f, cfg.T, gamma, int(draws))
                if key not in _CONF_THEORY_CACHE:
                    _CONF_THEORY_CACHE[key] = expected_confidence(
                        cfg.N, p_s, shape=k, gamma=gamma, f=cfg.f, T=cfg.T, draws=int(draws),
                        rng=np.random.default_rng(0))
                r = _CONF_THEORY_CACHE[key]
                ys.append(r["mean"])
                sems.append(r["sem"])
                recs.append({**prov, "series_val": k, "x_val": p_s, "rep": -1,
                             "theory": r["mean"], "sim": float("nan"), "theory_sem": r["sem"]})
        out.append({"x": list(xvals), "y": ys, "series_value": k, "records": recs,
                    "label": None if len(ks) > 1 else r"theory (closed form)"})
        print(f"        theory k={k:.4g}: max sem {max(sems):.6f}")
    mode = "PAIRED on the sweep's own stakes" if stakes is not None else f"{int(draws)} fresh draws"
    print(f"        theory curve(s) in {time.perf_counter() - t0:.1f}s "
          f"({len(ks) * len(xps)} points, {mode})")
    if pairs:
        d = np.concatenate(pairs)
        print(f"        PAIRED theory - simulation over {d.size} matched (cell, rep) runs: "
              f"mean {d.mean():+.6f} +/- {d.std(ddof=1) / np.sqrt(d.size):.6f} (sem), "
              f"max |diff| {np.abs(d).max():.6f}")
    return out


def _stake_jaccard_theory(cfg, xs, *, series=None, series_vals=None, xcol=None, draws=32,
                          stakes=None, x_name=None, df=None, measure=None, table_hash=None,
                          **_):
    """
    Return the closed-form stake_top_jaccard curve, one per Pareto k (theory.expected_jaccard).

    J = K/(2m - K) with K the overlap of the m largest estimated and true stakes; the law of K
    conditions on the m-th largest count (src/theory/stake_jaccard.py). Paired/unpaired modes
    and the archived record schema follow _stake_top1_theory.
    """
    from theory import expected_jaccard, jaccard_probability

    ks = [float(v) for v in series_vals] if (series == "shape" and series_vals) else [cfg.shape]
    xps = [float(x) for x in xs]
    xvals = _beta_from_ps(cfg, xps) if xcol == "bandwidth_overhead" else xps
    xn = x_name or "p_s"
    top_frac = float(cfg.top_frac)

    def _matched(k, p_s):
        """Evaluate the closed form on the sweep's own stake vectors -> (mean, per-rep list)."""
        for cell, alphas in stakes:
            if (np.isclose(float(cell.get(series, k)), k)
                    and np.isclose(float(cell.get(xn, p_s)), p_s)):
                vals = [jaccard_probability(a, p_s, top_frac=top_frac, f=cfg.f, T=cfg.T)
                        for a in alphas]
                return float(np.mean(vals)), vals
        raise SystemExit(f"theory_match_stakes: no swept cell at {series}={k}, {xn}={p_s}")

    n_stakes = len(stakes[0][1]) if stakes else int(draws)
    prov = {"measure": measure or "stake_top_jaccard", "series_col": series, "x_col": xn,
            "N": int(cfg.N), "T": int(cfg.T), "f": float(cfg.f),
            "paired": bool(stakes is not None), "stake_draws": int(n_stakes),
            "table_hash": str(table_hash) if (stakes is not None and table_hash) else ""}

    out, pairs = [], []
    t0 = time.perf_counter()
    for k in ks:
        ys, sems, recs = [], [], []
        for p_s in xps:
            if stakes is not None:
                mean, vals = _matched(k, p_s)
                ys.append(mean)
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
                sim = np.full(len(vals), np.nan)
                if df is not None:
                    m = np.isclose(df[series].to_numpy(float), k) & np.isclose(
                        df[xn].to_numpy(float), p_s)
                    s = df[m].sort_values("rep")[prov["measure"]].to_numpy(float)
                    if s.size == len(vals):
                        sim = s
                        pairs.append(np.asarray(vals) - s)
                recs += [{**prov, "series_val": k, "x_val": p_s, "rep": r,
                          "theory": v, "sim": sim[r], "theory_sem": float("nan")}
                         for r, v in enumerate(vals)]
            else:
                key = (cfg.N, k, p_s, cfg.f, cfg.T, top_frac, int(draws))
                if key not in _JAC_THEORY_CACHE:
                    _JAC_THEORY_CACHE[key] = expected_jaccard(
                        cfg.N, p_s, shape=k, top_frac=top_frac, f=cfg.f, T=cfg.T,
                        draws=int(draws), rng=np.random.default_rng(0))
                r = _JAC_THEORY_CACHE[key]
                ys.append(r["mean"])
                sems.append(r["sem"])
                recs.append({**prov, "series_val": k, "x_val": p_s, "rep": -1,
                             "theory": r["mean"], "sim": float("nan"), "theory_sem": r["sem"]})
        out.append({"x": list(xvals), "y": ys, "series_value": k, "records": recs,
                    "label": None if len(ks) > 1 else r"theory (closed form)"})
        print(f"        theory k={k:.4g}: max sem {max(sems):.6f}")
    mode = "PAIRED on the sweep's own stakes" if stakes is not None else f"{int(draws)} fresh draws"
    print(f"        theory curve(s) in {time.perf_counter() - t0:.1f}s "
          f"({len(ks) * len(xps)} points, {mode})")
    if pairs:
        d = np.concatenate(pairs)
        print(f"        PAIRED theory - simulation over {d.size} matched (cell, rep) runs: "
              f"mean {d.mean():+.6f} +/- {d.std(ddof=1) / np.sqrt(d.size):.6f} (sem), "
              f"max |diff| {np.abs(d).max():.6f}")
    return out


def _latency_theory(cfg, xs, *, exp=None, xcol=None, df=None, x_name=None, **_):
    """
    Return the closed-form latency cost ell(mix_scale) for the log-normal link law.

        ell = 1 + [(k+1)(E[d] + E[eps]) + E[Z]] / E[D_br(N)],  E[Z] = 1/lambda_S + n_stages/lambda_M

    E[d] is the link law's mean, E[eps] the per-message jitter scale (also inside E[D_br]),
    E[Z] comes from delay_moments on the per-cell config. Every route through a W x k mix-net
    crosses k+1 links, so the same formula covers both layers. Returns None for other laws.
    """
    if cfg.layer not in ("single_path_mix", "mixnet"):
        return None
    if getattr(cfg.latency, "lognormal", None) is None:
        return None
    if cfg.layer_params is None or exp is None:
        return None

    link_mean = float(cfg.latency.lognormal.mean)
    apply_cell = make_apply_cell(exp.path_map)
    xn = x_name or "mix_scale"

    ys, d_brs = [], []
    for x in xs:
        cell = apply_cell(cfg, {xn: x})
        lp = cell.layer_params
        n_stages = int(lp.hops) + (1 if lp.receiver_delays else 0)
        e_z, _ = delay_moments(n_stages, float(lp.sender_scale), float(lp.mix_scale))
        eps = 0.0 if cell.latency is None or cell.latency.jitter is None else float(
            cell.latency.jitter.scale)
        d_br = _broadcast_mean(cell)
        d_brs.append(d_br)
        ys.append(1.0 + ((int(lp.hops) + 1) * (link_mean + eps) + e_z) / d_br)

    xvals = list(xs)
    if xcol and df is not None and xcol != xn and xcol in df.columns:
        means = df.groupby(xn)[xcol].mean()
        xvals = [float(means.loc[x]) for x in xs]
    return {"x": xvals, "y": ys, "d_br": d_brs,
            "label": r"theory $\ell = 1 + [(L{+}1)\,\mathbb{E}[d] + \mathbb{E}[Z]]/\mathbb{E}[D_{\mathrm{br}}]$"}


def _latency_seconds_theory(cfg, xs, **kw):
    """
    Return the closed-form AC-path delay in seconds, E[Y] = (L+1)(E[d] + E[eps]) + E[Z].

    Recovered per cell as (ell - 1) E[D_br] from _latency_theory; keyed to the `latency` column.
    """
    t = _latency_theory(cfg, xs, **kw)
    if t is None:
        return None
    return {"x": t["x"], "y": [(l - 1.0) * d for l, d in zip(t["y"], t["d_br"])],
            "label": r"theory $\mathbb{E}[Y] = (L{+}1)\,\mathbb{E}[d] + \mathbb{E}[Z]$"}


_DEANON_THEORY_CACHE = {}


def _deanon_theory(cfg, xs, *, exp=None, xcol=None, df=None, x_name=None, **_):
    """
    Return the closed-form deanon_top1 curve P_Top1 vs the swept mixing budget.

    src/theory/attribution.py: single-path mix with a continuous link law and fixed-count cover;
    one point is a 2-D quadrature, memoised across figures. Returns None for other layers/laws.
    """
    from theory.attribution import deanon_top1 as _deanon

    if cfg.layer != "single_path_mix" or getattr(cfg.latency, "lognormal", None) is None:
        return None
    if cfg.layer_params is None or exp is None or cfg.dummy.count is None:
        return None
    if getattr(cfg.dummy, "p_s", None) is not None:
        return None

    apply_cell = make_apply_cell(exp.path_map)
    xn = x_name or "mix_scale"
    ys = []
    t0 = time.perf_counter()
    for x in xs:
        lp = apply_cell(cfg, {xn: x}).layer_params
        key = (int(lp.hops), float(lp.mix_scale), float(lp.sender_scale), bool(lp.receiver_delays),
               int(cfg.dummy.count), float(cfg.f), cfg.latency.lognormal)
        if key not in _DEANON_THEORY_CACHE:
            _DEANON_THEORY_CACHE[key] = _deanon(
                hops=int(lp.hops), mix_scale=float(lp.mix_scale),
                sender_scale=float(lp.sender_scale), receiver_delays=bool(lp.receiver_delays),
                link=cfg.latency.lognormal, count=int(cfg.dummy.count), f=cfg.f)
        ys.append(_DEANON_THEORY_CACHE[key])
    print(f"        theory deanon_top1: {len(xs)} points in {time.perf_counter() - t0:.1f}s")

    xvals = list(xs)
    if xcol and df is not None and xcol != xn and xcol in df.columns:
        means = df.groupby(xn)[xcol].mean()
        xvals = [float(means.loc[x]) for x in xs]
    return {"x": xvals, "y": ys, "label": r"theory (closed form)"}


def _sigma_d_theory(cfg, xs, *, exp=None, series=None, series_vals=None, x_name=None, **_):
    """
    Return E[sigma_hat_d] vs W, one curve per sender -> entry assignment (src/theory/sigma_hat.py).

    This is the estimator's expectation (with its finite-m bias), not the population sigma_d.
    Returns None outside the mix-net layer with a log-normal law and fixed-count cover.
    """
    from theory.sigma_hat import expected_sigma_d_hat

    if cfg.layer != "mixnet" or getattr(cfg.latency, "lognormal", None) is None:
        return None
    if cfg.layer_params is None or exp is None or cfg.dummy.count is None:
        return None

    apply_cell = make_apply_cell(exp.path_map)
    xn = x_name or "width"
    arms = list(series_vals) if series == "entry_assignment" and series_vals else [None]
    t0 = time.perf_counter()
    out = []
    for arm in arms:
        ys = []
        for x in xs:
            axes = {xn: x} if arm is None else {xn: x, "entry_assignment": arm}
            lp = apply_cell(cfg, axes).layer_params
            ys.append(expected_sigma_d_hat(
                link=cfg.latency.lognormal, count=int(cfg.dummy.count), width=int(lp.width),
                assignment=str(lp.entry_assignment), f=cfg.f))
        out.append({"x": list(xs), "y": ys, "series_value": arm,
                    "label": r"expected $\sigma_d$ (closed form)"})

    if "uniform" in [str(a) for a in arms if a is not None]:
        sigma_link = float(np.sqrt(link_moments(cfg.latency.lognormal)[1]))
        widths = [int(apply_cell(cfg, {xn: x}).layer_params.width) for x in xs]
        out.append({"x": list(xs), "y": [sigma_link / np.sqrt(w) for w in widths],
                    "series_value": None, "color": "0.4", "linewidth": 1.5,
                    "label": r"standard deviation of the link law$\,/\sqrt{W}$"})

    print(f"        theory sigma_d: {len(arms)}x{len(xs)} points in {time.perf_counter() - t0:.1f}s")
    return out[0] if arms == [None] else out


def _make_anchor_theory(stat, label):
    """
    Build a theory-curve fn drawing a measure's |S_t| anchor per cell, for sweeps that move it.

    `stat(sizes, probs)` is evaluated on `_measure_law` of each cell's config. Returns None when
    the anchor is constant (the hline's job), one curve when it varies with x only, and one
    curve per series (in the series' colour) when it varies by series.
    """
    def fn(cfg, xs, *, exp=None, series=None, series_vals=None, xcol=None, df=None,
           x_name=None, **_):
        """Evaluate the anchor over the drawn (series, x) cells."""
        if exp is None or x_name is None:
            return None
        apply_cell = make_apply_cell(exp.path_map)
        svals = (list(series_vals) if series and series_vals and series in exp.path_map
                 else [None])
        rows = []
        for sv in svals:
            ys = []
            for x in xs:
                axes = {x_name: x} if sv is None else {x_name: x, series: sv}
                sizes, probs = _measure_law(apply_cell(cfg, axes))
                ys.append(float(stat(sizes, probs)))
            rows.append(ys)
        arr = np.asarray(rows)
        if np.allclose(arr, arr.flat[0], rtol=1e-12, atol=0.0):
            return None
        xvals = list(xs)
        if xcol and df is not None and xcol != x_name and xcol in df.columns:
            means = df.groupby(x_name)[xcol].mean()
            xvals = [float(means.loc[x]) for x in xs]
        if len(svals) > 1 and not np.allclose(arr, arr[0][None, :], rtol=1e-12, atol=0.0):
            return [{"x": xvals, "y": ys, "series_value": sv, "label": label}
                    for sv, ys in zip(svals, rows)]
        return {"x": xvals, "y": rows[0], "label": label}
    return fn


_deanon_null_theory = _make_anchor_theory(
    lambda sizes, probs: (probs / sizes).sum(),
    r"$\mathbb{E}[1/|S_t|]$ (random guess)")
_true_post_floor_theory = _make_anchor_theory(
    lambda sizes, probs: (probs / sizes).sum(),
    r"$\mathbb{E}[1/|S_t|]$ (uniform posterior)")
_entropy_ceiling_theory = _make_anchor_theory(
    lambda sizes, probs: (probs * np.log2(sizes)).sum(),
    r"$\mathbb{E}[\log_2|S_t|]$ (uniform posterior)")


def _deanon_top1_theory(cfg, xs, **kw):
    """Return the deanon_top1 closed form where derived, else the moving null anchor."""
    out = _deanon_theory(cfg, xs, **kw)
    return out if out is not None else _deanon_null_theory(cfg, xs, **kw)


THEORY_CURVES = {
    "bandwidth_overhead": _beta_theory,
    "latency_overhead": _latency_theory,
    "latency": _latency_seconds_theory,
    "stake_top1_hit": _stake_top1_theory,
    "stake_confidence": _stake_confidence_theory,
    "stake_top_jaccard": _stake_jaccard_theory,
    "deanon_top1": _deanon_top1_theory,
    "mean_true_posterior": _true_post_floor_theory,
    "posterior_entropy": _entropy_ceiling_theory,
    "sigma_d": _sigma_d_theory,
}


def _latency_seconds(cfg):
    """
    Return the twin-axis map ell -> total block-propagation delay E[Delta] = ell * E[D_br(N)] (s).

    E[D^AC] includes the broadcast, so ell * E[D_br] is the total; the AC leg on its own
    (mu + Z) is the `latency` column. The 1/f ceiling reads directly off this axis.
    """
    return {"scale": _broadcast_mean(cfg),
            "label": r"Total propagation  $\mathbb{E}[\Delta]$  (s)"}


SECONDARY_AXES = {
    "latency_overhead": _latency_seconds,
}


_EMN_LABEL_CACHE = {}


def _emn_series_labels(shape_vals, cfg):
    """
    Return a legend label per Pareto shape k: its max expected stake E[M_N] and k, e.g. "10.2%  (k=1.33)".

    E[M_N] is the finite-N Monte Carlo value read from results/tables/max_relative_stake.csv
    (written by tests/max-relative-stake-validation.py), so legends and the max-stake figure
    agree. A (k, N) row that is missing falls back to a seeded 50k-draw MC.
    """
    table = REPO / "results" / "tables" / "max_relative_stake.csv"
    if not _EMN_LABEL_CACHE and table.exists():
        for r in pd.read_csv(table).itertuples():
            _EMN_LABEL_CACHE[(float(r.shape), int(r.N))] = float(r.mean)
    out = []
    for k in shape_vals:
        key = (float(k), int(cfg.N))
        if key not in _EMN_LABEL_CACHE:
            print(f"(E[M_N] label for k = {float(k):g}, N = {cfg.N}: not in {table.name}, "
                  f"50k-draw MC fallback -- run tests/max-relative-stake-validation.py to archive)")
            _EMN_LABEL_CACHE[key] = simulate_max_relative_stake(
                cfg.N, float(k), reps=50_000, rng=np.random.default_rng(0))["mean"]
        out.append(f"{100 * _EMN_LABEL_CACHE[key]:.1f}\\%  ($k$ = {float(k):.3g})")
    return out


SERIES_LABELS = {
    "shape": _emn_series_labels,
}


def _plot_heatmap(df, spec, out_path):
    """
    Draw a 2-D sweep as a colour surface of measure `y` over the (x, yaxis) grid.

    Spec keys: kind: heatmap; x, yaxis (the swept columns), y (the measure). Optional: cmap
    (default viridis), annotate, cbar_label, xlabel/ylabel, out. Axes are categorical: one
    equal-width cell per swept value.
    """
    x, yax, z = spec["x"], spec["yaxis"], spec["y"]
    grid = df.groupby([yax, x])[z].mean().unstack(x)
    xs, ys = list(grid.columns), list(grid.index)

    fig, ax = plt.subplots()
    ax.grid(False)
    im = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower",
                   cmap=spec.get("cmap", "viridis"), interpolation="nearest")
    ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs], rotation=90)
    ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])

    if spec.get("annotate"):
        norm, cmap = im.norm, im.cmap
        for i in range(len(ys)):
            for j in range(len(xs)):
                v = grid.to_numpy()[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = cmap(norm(v))
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(spec.get("cbar_label", z))
    ax.set_xlabel(spec.get("xlabel", x))
    ax.set_ylabel(spec.get("ylabel", yax))
    return _save_figure(fig, out_path, spec)


def _akey(v):
    """Return a plain Python scalar for an anchor-grid key (unwraps numpy scalars)."""
    return v.item() if hasattr(v, "item") else v


def _series_key(v):
    """Return a hashable key for a series value: float where numeric, else its string form."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _save_figure(fig, out_path, spec):
    """Save a results figure."""
    return plotstyle.save(fig, out_path)


def plot_sweep(df, spec, out_path):
    """
    Draw a figure for a sweep DataFrame from the config's plot spec (mean +/- error of y vs x).

    Spec keys: x, y, series, series_values, fix, x_values, xcol, xscale/yscale, ylim, xticks,
    xlabel/ylabel, hline/limit/vline {y, label}, theory ({x, y, label} or a list with
    "series_value"), errorbar ("sem"|"std"|"minmax"), band, show_median, whiskers, marker,
    color/label, colors, cmap, series_label, series_labels, legend_title, ytransform,
    secondary_y/secondary_x, overlay, panels, fix_each, panel_label, legend, out.
    main() resolves the `auto` anchors before calling this; see _draw_panel for the key semantics.
    """
    df = _apply_fix(df, spec)
    if spec.get("panels"):
        return _plot_panels(df, spec, out_path)
    if spec.get("kind") == "heatmap":
        return _plot_heatmap(df, spec, out_path)
    fig, ax = plt.subplots()
    _draw_panel(df, spec, ax)
    return _save_figure(fig, out_path, spec)


def _apply_fix(df, spec):
    """
    Filter the table to the spec's `fix: {column: value}` cells, then apply `x_values`.

    Filtering before the groupby keeps unplotted swept axes from being pooled into the mean;
    a column or value that selects nothing raises.
    """
    for col, val in (spec.get("fix") or {}).items():
        if col not in df.columns:
            raise ValueError(f"plot fix: column {col!r} not in the results table "
                             f"(have {sorted(df.columns)})")
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = np.isclose(df[col].to_numpy(dtype=float), float(val), rtol=1e-9, atol=0.0)
        else:
            mask = df[col].to_numpy() == val
        if not mask.any():
            raise ValueError(f"plot fix: {col}={val!r} matches no cell "
                             f"(have {sorted(df[col].unique())})")
        df = df[mask]
    return _apply_x_values(df, spec)


def _apply_x_values(df, spec):
    """
    Filter the table to the spec's `x_values` (default: all values of the swept x column).

    Applied before aggregation, anchors and theory alike, so a hidden cell cannot widen the
    axis; tolerance-matched, and a value matching no cell raises.
    """
    want = spec.get("x_values")
    x = spec.get("x")
    if want is None or not x:
        return df
    if x not in df.columns:
        raise ValueError(f"plot x_values: column {x!r} not in the results table")
    col = df[x].to_numpy()
    numeric = pd.api.types.is_numeric_dtype(df[x])
    mask = np.zeros(len(df), dtype=bool)
    for v in want:
        hit = (np.isclose(col.astype(float), float(v), rtol=1e-9, atol=0.0) if numeric
               else col == v)
        if not hit.any():
            raise ValueError(f"plot x_values: {x}={v!r} matches no cell "
                             f"(have {sorted(df[x].unique())})")
        mask |= hit
    return df[mask]


def _plot_panels(df, spec, out_path):
    """
    Draw one figure with one subplot per entry of `panels:`, each inheriting the shared keys.

    Layout keys: ncols (default: one row), panel_size [w, h] inches per panel (default 7.5 x 6.0).
    """
    panels = list(spec["panels"])
    shared = {k: v for k, v in spec.items()
              if k not in ("panels", "out", "ncols", "panel_size")}
    n = len(panels)
    ncols = max(1, int(spec.get("ncols", n)))
    nrows = math.ceil(n / ncols)
    pw, ph = spec.get("panel_size", (7.5, 6.0))
    fig, axes = plt.subplots(nrows, ncols, figsize=(float(pw) * ncols, float(ph) * nrows),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, panel in zip(flat, panels):
        _draw_panel(df, {**shared, **panel}, ax)
    for ax in flat[n:]:
        ax.set_visible(False)
    return _save_figure(fig, out_path, spec)


def _draw_panel(df, spec, ax):
    """Draw one spec onto one axes (the body of plot_sweep)."""
    df = _apply_fix(df, spec)
    x, series, y = spec["x"], spec.get("series"), spec["y"]
    xcol = spec.get("xcol", x)
    err = spec.get("errorbar", "sem")
    if err not in ("sem", "std", "minmax"):
        err = "sem"
    group = [series, x] if series else [x]
    stats = ["mean", "min", "max", "median"] if err == "minmax" else ["mean", err]
    agg = df.groupby(group)[y].agg(stats).reset_index()
    if xcol != x:
        agg = agg.merge(df.groupby(group)[xcol].mean().reset_index(), on=group)

    want = spec.get("series_values")
    if series and want is not None:
        have = np.asarray(sorted(agg[series].unique()), dtype=float)
        keep = []
        for v in want:
            hit = have[np.isclose(have, float(v), rtol=1e-9, atol=0.0)]
            if not hit.size:
                raise ValueError(f"plot series_values: {series}={v} is not in the sweep "
                                 f"(have {have.tolist()})")
            keep.append(hit[0])
        agg = agg[np.isclose(agg[series].to_numpy(dtype=float)[:, None],
                             np.asarray(keep)[None, :], rtol=1e-9, atol=0.0).any(axis=1)]

    if series and xcol != x and spec.get("xcol_fit") == "affine":
        parts = []
        for v in agg[series].unique():
            sub = agg[agg[series] == v].copy()
            xv, yv = sub[x].to_numpy(dtype=float), sub[xcol].to_numpy(dtype=float)
            slope, icpt = np.polyfit(xv, yv, 1)
            span = float(np.ptp(yv)) or 1.0
            resid = float(np.max(np.abs(yv - (icpt + slope * xv)))) / span
            print(f"        xcol_fit [{series}={v}]: {xcol} = {icpt:.4g} + {slope:.4g}*{x} "
                  f"(max residual {resid:.2%} of range)")
            sub[xcol] = icpt + slope * xv
            parts.append(sub)
        agg = pd.concat(parts)

    if spec.get("ytransform") == "total_propagation":
        off = float(spec["y_offset"])
        for c in ("mean", "min", "max", "median"):
            if c in agg.columns:
                agg[c] = agg[c] + off
        th = spec.get("theory")
        if isinstance(th, dict):
            spec = {**spec, "theory": {**th, "y": [v + off for v in th["y"]]}}
    if spec.get("ytransform") in ("excess", "deficit"):
        kind = spec["ytransform"]
        ref = spec.get("hline")
        grid = spec.get("anchor_grid")
        if not ref and not grid:
            raise ValueError(f"plot ytransform: {kind} needs a reference line, but {y!r} has no "
                             f"hline (set hline: auto, or drop the transform)")
        if grid:
            base = np.array([grid[(_akey(sv), _akey(xv))] for sv, xv in
                             zip(agg[series] if series else [None] * len(agg), agg[x])])
            if spec.get("limit"):
                print(f"figure -> NOTE: {y!r} limit line dropped ({kind} is per-cell, so the "
                      "defence-off anchor is not one y-value)")
                spec = {k: v for k, v in spec.items() if k != "limit"}
        else:
            base = float(ref["y"])
        if kind == "excess":
            for c in ("mean", "min", "max", "median"):
                if c in agg.columns:
                    agg[c] = agg[c] - base
        else:
            for c in ("mean", "median"):
                if c in agg.columns:
                    agg[c] = base - agg[c]
            if "min" in agg.columns and "max" in agg.columns:
                agg["min"], agg["max"] = base - agg["max"], base - agg["min"]
        spec = {k: v for k, v in spec.items() if k != "hline"}
        if spec.get("limit"):
            lim_y = spec["limit"]["y"]
            spec["limit"] = {**spec["limit"],
                             "y": lim_y - ref["y"] if kind == "excess" else ref["y"] - lim_y}
        if spec.get("yscale") == "log":
            bad = agg["mean"] <= 0
            if bad.any():
                side = "below" if kind == "excess" else "above"
                lost = agg.loc[bad, [c for c in (series, x) if c]].to_dict("records")
                print(f"figure -> NOTE: {int(bad.sum())} cell(s) dropped from the log axis "
                      f"({kind} <= 0, i.e. at or {side} the reference line): {lost}")
                agg = agg[~bad]

    if spec.get("xscale") == "log":
        bad_x = agg[xcol] <= 0
        if bad_x.any():
            lost = agg.loc[bad_x, [c for c in (series, x) if c]].to_dict("records")
            print(f"figure -> NOTE: {int(bad_x.sum())} cell(s) dropped from the LOG X-AXIS "
                  f"({xcol} <= 0, which a log scale cannot show): {lost}")
            agg = agg[~bad_x]

    show_median = spec.get("show_median", False)
    cap = 5 if err == "minmax" else 3
    def _yerr(sub):
        """Return the error-bar extent for the chosen statistic."""
        if err == "minmax":
            return np.vstack([(sub["mean"] - sub["min"]).to_numpy(),
                              (sub["max"] - sub["mean"]).to_numpy()])
        return sub[err].to_numpy()

    theory = spec.get("theory")
    theory_by_series = {}
    if isinstance(theory, dict):
        ax.plot(theory["x"], theory["y"], color="0.35", linestyle="--", linewidth=3,
                zorder=1, label=spec.get("theory_label", theory.get("label")))
    elif theory:
        for t in theory:
            if t.get("series_value") is None:
                ax.plot(t["x"], t["y"], color=t.get("color", "0.35"), linestyle="--",
                        linewidth=t.get("linewidth", 3), zorder=1, label=t.get("label"))
            else:
                theory_by_series[_series_key(t["series_value"])] = t
    points_only = err == "minmax" and not spec.get("whiskers", True)
    banded = bool(spec.get("band", False))

    def _bounds(sub):
        """Return the (lower, upper) envelope for the chosen statistic."""
        if err == "minmax":
            return sub["min"].to_numpy(), sub["max"].to_numpy()
        return (sub["mean"] - sub[err]).to_numpy(), (sub["mean"] + sub[err]).to_numpy()

    mk = spec.get("marker", "o")
    if mk in (None, False, "", "none", "None"):
        mk = "None"

    def _draw(sub, color, label=None, idx=0, ntot=1):
        """Draw one aggregated curve in the configured error style."""
        col = color if color is not None else "C0"
        if banded:
            lo, hi = _bounds(sub)
            ax.fill_between(sub[xcol], lo, hi, color=col, alpha=0.18, linewidth=0, zorder=0)
            ax.plot(sub[xcol], sub["mean"], marker=mk, markersize=7, linewidth=2,
                    color=col, label=label)
            if err == "minmax" and show_median:
                ax.plot(sub[xcol], sub["median"], marker="o", markersize=7, markerfacecolor="white",
                        markeredgecolor=col, markeredgewidth=1.5, linestyle="none", zorder=6)
        elif points_only:
            ax.plot(sub[xcol], sub["mean"], marker=mk, markersize=7, linewidth=2, color=col, label=label)
            tsize = 24 - (24 - 8) * (idx / max(ntot - 1, 1))
            for stat in ("min", "max"):
                ax.plot(sub[xcol], sub[stat], marker="_", markersize=tsize, markeredgewidth=2.0,
                        linestyle="none", color=col, alpha=0.9, zorder=5 + idx)
        else:
            ax.errorbar(sub[xcol], sub["mean"], yerr=_yerr(sub), marker=mk, markersize=7,
                        linewidth=2, capsize=cap, capthick=1.5, color=color, label=label)
            if err == "minmax" and show_median:
                ax.plot(sub[xcol], sub["median"], marker="o", markersize=7, markerfacecolor="white",
                        markeredgecolor=col, markeredgewidth=1.5, linestyle="none", zorder=6)

    theory_proxy, theory_colors = None, []
    if series:
        vals = sorted(agg[series].unique())
        labels = spec.get("series_labels")
        palette = spec.get("colors")
        cmap = plt.get_cmap(spec.get("cmap", "viridis"))
        for i, v in enumerate(vals):
            sub = agg[agg[series] == v].sort_values(xcol)
            label = (str(labels[i]) if labels and i < len(labels)
                     else str(spec.get("series_label", "{v}")).format(v=v))
            color = (palette[i] if palette and i < len(palette)
                     else cmap(i / max(len(vals) - 1, 1)))
            _draw(sub, color, label, idx=i, ntot=len(vals))
            vk = _series_key(v)
            match = [t for k, t in theory_by_series.items()
                     if (np.isclose(k, vk) if isinstance(k, float) and isinstance(vk, float)
                         else k == vk)]
            if match:
                ax.plot(match[0]["x"], match[0]["y"], color=color, linestyle="--",
                        linewidth=2, zorder=1)
                theory_colors.append(color)
        if theory_by_series:
            theory_proxy = ax.plot([], [], color="0.35", linestyle="--", linewidth=2,
                                   label=spec.get("theory_label", "closed form (theory)"))[0]
    else:
        _draw(agg.sort_values(xcol), spec.get("color"), spec.get("label"))

    for ov in spec.get("overlay") or []:
        opath = REPO / "results" / "tables" / f"{ov['table']}.csv"
        odf = pd.read_csv(opath)
        if ov.get("fix"):
            odf = _apply_fix(odf, {"fix": ov["fix"]})
        oagg = odf.groupby([x])[y].agg(stats).reset_index()
        if xcol != x:
            oagg = oagg.merge(odf.groupby([x])[xcol].mean().reset_index(), on=[x])
        if spec.get("xscale") == "log":
            oagg = oagg[oagg[xcol] > 0]
        _draw(oagg.sort_values(xcol), ov.get("color", "0.4"), ov.get("label", ov["table"]))

    hline = spec.get("hline")
    if hline:
        # grey by default (the random-guess / reference token); `hline_color:` overrides per panel
        ax.axhline(hline["y"], color=spec.get("hline_color", "0.4"), linestyle="--",
                   linewidth=1.5, label=hline.get("label"))
    limit = spec.get("limit")
    if limit:
        if spec.get("yscale") == "log" and limit["y"] <= 0:
            print(f"figure -> NOTE: {y!r} limit line y={limit['y']:g} omitted (not on a log y-axis)")
            limit = None
        else:
            ax.axhline(limit["y"], color="0.35", linestyle="-.", linewidth=1.5,
                       label=limit.get("label"))
    vline = spec.get("vline")
    if vline:
        ax.axvline(vline["y"], color="0.4", linestyle=":", linewidth=1.5,
                   label=vline.get("label"))
    if spec.get("xscale") == "log":
        ax.set_xscale("log")
    elif spec.get("xscale") == "symlog":
        lt = spec.get("linthresh")
        if lt is None:
            pos = agg.loc[agg[xcol] > 0, xcol]
            lt = float(pos.min()) if len(pos) else 1.0
        ax.set_xscale("symlog", linthresh=lt)
        if float(agg[xcol].min()) >= 0.0:
            ax.set_xlim(left=-0.05 * lt)
    if spec.get("yscale") == "log":
        ax.set_yscale("log")
    xticks = spec.get("xticks")
    if xticks:
        xs = agg[xcol].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
        tol = 1e-9 * max(abs(lo), abs(hi), 1.0)
        kept = [t for t in xticks if lo - tol <= float(t) <= hi + tol]
        if len(kept) < len(xticks):
            dropped = [t for t in xticks if t not in kept]
            print(f"figure -> NOTE: xticks {dropped} lie outside the drawn range "
                  f"[{lo:g}, {hi:g}] and are omitted")
        ax.set_xticks(kept)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ylim = spec.get("ylim")
    if ylim:
        bottom, top = ylim
        if bottom == "auto":
            bottom = (hline["y"] - 0.05) if hline else None
        ax.set_ylim(bottom, top)
    ax.set_xlabel(spec.get("xlabel", xcol))
    ax.set_ylabel(spec.get("ylabel", y))
    sec = spec.get("secondary_y")
    if sec:
        s, off = float(sec["scale"]), float(sec.get("offset", 0.0))
        secax = ax.secondary_yaxis(
            "right", functions=(lambda v, s=s, o=off: (v - o) * s,
                                lambda v, s=s, o=off: v / s + o))
        secax.set_ylabel(sec["label"])
    sx = spec.get("secondary_x")
    if sx:
        col = sx.get("column")
        if col:
            if col not in df.columns:
                raise ValueError(f"plot secondary_x: column {col!r} not in the results table "
                                 f"(have {sorted(df.columns)})")
            cm = df.groupby(x)[col].mean()
            xv = (df.groupby(x)[xcol].mean().to_numpy(dtype=float) if xcol != x
                  else cm.index.to_numpy(dtype=float))
            yv = cm.to_numpy(dtype=float)
            if xv.size < 2:
                raise ValueError(f"plot secondary_x: need >= 2 x cells to fit {col!r}, got {xv.size}")
            slope, intercept = (float(v) for v in np.polyfit(xv, yv, 1))
            span = float(np.ptp(yv))
            resid = float(np.max(np.abs(yv - (intercept + slope * xv)))) / (span if span else 1.0)
            if not np.isfinite(slope) or slope == 0.0 or resid > float(sx.get("tol", 0.01)):
                raise ValueError(
                    f"plot secondary_x: {col!r} is not affine in {x!r} (max residual "
                    f"{resid:.3%} of range, slope {slope:.4g}) -- a twin axis can only carry an "
                    f"affine map, so this would mis-place the ticks between cells. Plot it as its "
                    f"own panel (or with xcol) instead.")
            print(f"        secondary x-axis: {col} = {intercept:.6g} + {slope:.6g}*{x} "
                  f"(max residual {resid:.3%} of range)")
        else:
            s, off = float(sx["scale"]), float(sx.get("offset", 0.0))
            slope, intercept = s, -off * s
        secx = ax.secondary_xaxis(
            "top", functions=(lambda v, m=slope, c=intercept: c + m * v,
                              lambda v, m=slope, c=intercept: (v - c) / m))
        secx.set_xlabel(sx["label"], labelpad=float(sx.get("labelpad", 12)))
        ticks = sx.get("ticks")
        if ticks == "aligned":
            lo, hi = ax.get_xlim()
            base = [t for t in ax.get_xticks() if lo <= t <= hi]
            secx.set_xticks([intercept + slope * t for t in base])
            secx.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:.2g}"))
            secx.xaxis.set_minor_locator(mticker.NullLocator())
        elif ticks is not None:
            secx.set_xticks(list(ticks))
            secx.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
            secx.xaxis.set_minor_locator(mticker.NullLocator())
        if sx.get("tickfmt") == "thousands":
            secx.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda v, _pos: f"{v / 1000:.3g}k" if abs(v) >= 1000 else f"{v:.3g}"))
    if spec.get("legend", True) and (series or hline or limit or theory or vline
                                     or spec.get("overlay") or spec.get("label")):
        handles, labels_ = ax.get_legend_handles_labels()
        handler_map = None
        if theory_proxy is not None and theory_colors:
            handles[handles.index(theory_proxy)] = tuple(
                Line2D([], [], color=c, linestyle="--", linewidth=2) for c in theory_colors)
            handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.4)}
        ax.legend(handles, labels_, fontsize=spec.get("legend_fontsize", 15),
                  title=spec.get("legend_title"),
                  title_fontsize=spec.get("legend_fontsize", 15), handler_map=handler_map,
                  loc=spec.get("legend_loc"))

    if spec.get("panel_label"):
        ax.text(0.025, 0.975, str(spec["panel_label"]), transform=ax.transAxes,
                ha="left", va="top", fontsize=20, zorder=10)


def _expand_fix_each(specs):
    """
    Expand `fix_each: {col: [v1, v2, ...]}` into one otherwise-identical spec per value.

    `{v}` in `out` is replaced by the value; an existing `fix:` is extended.
    """
    out = []
    for spec in specs:
        each = spec.get("fix_each")
        if not each:
            out.append(spec)
            continue
        if len(each) != 1:
            raise ValueError(f"plot fix_each: expected exactly one column, got {sorted(each)}")
        (col, values), = each.items()
        for v in values:
            s = {k: val for k, val in spec.items() if k != "fix_each"}
            s["fix"] = {**(spec.get("fix") or {}), col: v}
            token = f"{v:g}" if isinstance(v, (int, float)) else str(v)
            if "{v}" not in spec.get("out", ""):
                raise ValueError(f"plot fix_each: `out` must contain '{{v}}' so each figure gets "
                                 f"its own filename, got {spec.get('out')!r}")
            s["out"] = spec["out"].replace("{v}", token)
            out.append(s)
    return out


def _fmt_axis(values):
    """Render a sweep axis's values compactly: [a, b, ..., z] (n pts)."""
    def g(x):
        """Format one axis value."""
        return f"{x:g}" if isinstance(x, float) else str(x)
    vals = list(values)
    n = len(vals)
    body = ", ".join(g(v) for v in vals) if n <= 4 else f"{g(vals[0])}, {g(vals[1])}, ..., {g(vals[-1])}"
    return f"[{body}] ({n} pts)"


def _print_run_header(exp, base_cfg, *, reps, seed, jobs, mem_fraction):
    """Print the pre-sweep summary: pipeline, system params, grid and run budget."""
    n_cells = math.prod(len(v) for v in exp.axes.values()) if exp.axes else 1
    total = n_cells * reps
    print(f"experiment: {exp.name}")
    print(f"  pipeline  {base_cfg.layer} -> {base_cfg.attack} -> [{', '.join(base_cfg.measures)}]")
    sysparts = [f"N={base_cfg.N}", f"f={base_cfg.f:.4g}"]
    if "shape" not in exp.axes:
        sysparts.append(f"shape={base_cfg.shape:.4g}")
    sysparts.append(f"T={base_cfg.T}")
    print(f"  system    {'  '.join(sysparts)}")
    grid = "  x  ".join(f"{name}={_fmt_axis(vals)}" for name, vals in exp.axes.items())
    print(f"  sweep     {grid}  x  reps={reps}  =>  {total:,} runs  (seed={seed})")
    execparts = [f"jobs={jobs}", f"mem_fraction={mem_fraction}"]
    if base_cfg.fast_counts:
        execparts.append("fast_counts")
    if base_cfg.cover_runs > 1:
        execparts.append(f"cover_runs(M)={base_cfg.cover_runs}  [=> {total * base_cfg.cover_runs:,} inner runs]")
    print(f"  exec      {'  '.join(execparts)}")
    try:
        from pipeline import _cell_workers, _est_peak_bytes, _resolve_cpu_cap, _total_ram_bytes
        avail = _total_ram_bytes()
        budget = mem_fraction * avail
        w = _cell_workers(base_cfg, _resolve_cpu_cap(jobs), budget, reps)
        note = "  <- MEMORY-bound" if w < min(_resolve_cpu_cap(jobs), reps) else ""
        print(f"  memory    {avail/2**30:.1f} GiB usable x {mem_fraction} = {budget/2**30:.1f} GiB "
              f"budget;  ~{_est_peak_bytes(base_cfg)/2**30:.2f} GiB/run  =>  {w} workers{note}")
    except Exception:
        pass


def main():
    """Parse the command line, run (or reload) the sweep, and draw the configured figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="path to a YAML config")
    ap.add_argument("--list", action="store_true", help="list available configs and exit")
    ap.add_argument("--quick", action="store_true", help="fast smoke run (short T, reps=1)")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    ap.add_argument("--jobs", type=int, default=None,
                    help="sweep worker processes (1 = serial; <=0 = auto). Overrides JOBS.")
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="fraction of RAM the pool may use (caps heavy-cell workers). "
                         "Overrides MEM_FRACTION. ~0.8 is the safe ceiling on 16 GB.")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip the sweep; redraw the figures from the existing results table")
    ap.add_argument("--refresh-theory", action="store_true",
                    help="recompute the closed-form theory curves instead of reusing "
                         "results/tables/<name>_theory.csv (needed after changing src/theory)")
    args = ap.parse_args()

    if args.list:
        list_configs()
        return

    cfg_path = _resolve(args.config).resolve()
    try:
        shown = cfg_path.relative_to(REPO)
    except ValueError:
        shown = cfg_path
    print(f"config: {shown}" + ("  (from CONFIG)" if cfg_path == DEFAULT_CONFIG.resolve() else ""))
    exp = load_experiment(cfg_path)
    quick = args.quick or QUICK
    plot_only = args.plot_only or PLOT_ONLY
    base_cfg = replace(exp.base_cfg, T=QUICK_T) if quick else exp.base_cfg
    reps = 1 if quick else exp.reps
    seed = exp.seed if args.seed is None else args.seed

    tag = "_quick" if quick else ""
    tables = REPO / "results" / "tables"
    csv = tables / f"{exp.name}{tag}.csv"

    if plot_only:
        if not csv.exists():
            raise SystemExit(f"plot-only: no results table at {csv.relative_to(REPO)} "
                             "-- run the sweep first")
        df = pd.read_csv(csv)
        print(f"plot-only: {len(df)} rows from {csv.relative_to(REPO)}")
    else:
        apply_cell = make_apply_cell(exp.path_map)
        jobs = args.jobs if args.jobs is not None else JOBS
        mem_fraction = args.mem_fraction if args.mem_fraction is not None else MEM_FRACTION
        _print_run_header(exp, base_cfg, reps=reps, seed=seed, jobs=jobs, mem_fraction=mem_fraction)
        t0 = time.perf_counter()
        df = sweep(base_cfg, exp.axes, apply_cell, seed=seed, reps=reps, progress=True,
                   n_jobs=jobs, mem_fraction=mem_fraction)
        elapsed = time.perf_counter() - t0
        tables.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        print(f"\n{len(df)} runs in {elapsed:.1f}s ({elapsed / len(df):.2f}s/run).")
        print(f"table  -> {csv.relative_to(REPO)}")

    theory_csv = tables / f"{exp.name}{tag}_theory.csv"
    theory_cache = None
    if theory_csv.exists() and not args.refresh_theory:
        theory_cache = pd.read_csv(theory_csv)
    theory_rows = []
    table_hash = _table_fingerprint(csv)

    specs = exp.plots if exp.plots else ([exp.plot] if exp.plot else [])
    specs = _expand_fix_each(specs)

    def resolve(plot):
        """Resolve one spec's `auto` keys against the config its cells describe.

        Returns None to skip (measure absent from this table); appends to theory_rows.
        """
        if plot["y"] not in df.columns:
            print(f"figure -> skipped: no column {plot['y']!r} in the table")
            return None
        fixed = {k: v for k, v in (plot.get("fix") or {}).items() if k in exp.path_map}
        cfg_p = make_apply_cell(exp.path_map)(base_cfg, fixed) if fixed else base_cfg
        dfp = _apply_x_values(df, plot)
        if plot.get("hline") == "auto":
            ref = REFERENCE_LINES.get(plot["y"])
            plot["hline"] = ref(cfg_p) if ref else None
        if plot.get("ytransform") == "total_propagation":
            off = _broadcast_mean(cfg_p)
            plot["y_offset"] = off
            if plot.get("hline"):
                plot["hline"] = {**plot["hline"], "y": plot["hline"]["y"] + off,
                                 "label": r"latency limit  $\mathbb{E}[\Delta] = 1/f$"}
        if plot.get("hline") and plot.get("hline_label"):
            plot["hline"] = dict(plot["hline"],
                                 label=plot["hline_label"])
        if plot.get("limit") == "auto":
            lim = LIMIT_LINES.get(plot["y"])
            plot["limit"] = lim(cfg_p) if lim else None
        if plot.get("vline") == "auto":
            ref = REFERENCE_LINES.get(plot.get("xcol", plot["x"]))
            if ref is not None:
                v = dict(ref(cfg_p))
                v["label"] = v["label"].replace("below", "left")
                plot["vline"] = v
            else:
                plot["vline"] = None
        vl = plot.get("vline")
        if isinstance(vl, dict) and "at" in vl and "y" not in vl:
            vl = dict(vl)
            at = float(vl.pop("at"))
            if plot.get("xcol") == "bandwidth_overhead":
                at = _beta_from_ps(cfg_p, [at])[0]
            vl["y"] = at
            plot["vline"] = vl
        ser = plot.get("series")
        want = plot.get("series_values")
        if want is not None:
            svals = sorted(float(v) for v in want) if all(
                isinstance(v, (int, float)) for v in want) else sorted(str(v) for v in want)
        else:
            svals = sorted(df[ser].unique()) if ser and ser in df.columns else None
        if plot.get("ytransform") in ("excess", "deficit"):
            ref_fn = REFERENCE_LINES.get(plot["y"])
            if ref_fn is not None:
                svs = svals if (ser and ser in exp.path_map) else [None]
                grid = {}
                for sv in svs:
                    for xv in sorted(dfp[plot["x"]].unique()):
                        axes_ = {plot["x"]: xv} if sv is None else {plot["x"]: xv, ser: sv}
                        axes_ = {k: v for k, v in axes_.items() if k in exp.path_map}
                        grid[(_akey(sv), _akey(xv))] = float(
                            ref_fn(make_apply_cell(exp.path_map)(cfg_p, axes_))["y"])
                if len(set(grid.values())) > 1:
                    plot["anchor_grid"] = grid
                    print(f"        anchor: per-cell {plot['y']} baseline over "
                          f"{len(grid)} cells ({min(grid.values()):.4f}..{max(grid.values()):.4f})")
        if plot.get("theory") == "auto":
            tc = THEORY_CURVES.get(plot["y"])
            xs_t = sorted(dfp[plot["x"]].unique())
            reused = _theory_from_cache(theory_cache, plot, cfg_p, exp, ser, svals, xs_t,
                                        plot.get("xcol"), plot["x"], table_hash) if tc else None
            if reused is not None:
                plot["theory"] = reused
                print(f"        theory: reused {theory_csv.relative_to(REPO)} "
                      "(--refresh-theory to recompute)")
            elif tc:
                stakes = (_sweep_stake_vectors(exp, cfg_p, df, seed, reps)
                          if plot.get("theory_match_stakes") else None)
                plot["theory"] = tc(
                    cfg_p, xs_t, series=ser, series_vals=svals, xcol=plot.get("xcol"),
                    draws=int(plot.get("theory_draws", 32)), stakes=stakes,
                    x_name=plot["x"], df=dfp, measure=plot["y"], table_hash=table_hash, exp=exp,
                )
                for t in plot["theory"] if isinstance(plot["theory"], list) else []:
                    theory_rows.extend(t.get("records", []))
            else:
                plot["theory"] = None
        if plot.get("secondary_y") == "auto":
            sa = SECONDARY_AXES.get(plot["y"])
            plot["secondary_y"] = sa(cfg_p) if sa else None
        if plot.get("series_labels") == "auto":
            fn = SERIES_LABELS.get(ser)
            plot["series_labels"] = fn(svals, cfg_p) if fn and svals else None
        return plot

    for spec in specs:
        plot = dict(spec)
        if plot.get("panels"):
            shared = {k: v for k, v in plot.items()
                      if k not in ("panels", "out", "ncols", "panel_size")}
            resolved = [r for r in (resolve({**shared, **dict(p)}) for p in plot["panels"])
                        if r is not None]
            if not resolved:
                print(f"figure -> skipped: no plottable panel for {plot['out']!r}")
                continue
            plot["panels"] = resolved
        else:
            plot = resolve(plot)
            if plot is None:
                continue
        out_name = plot["out"]
        if tag:
            out_name = out_name.replace(".png", f"{tag}.png")
        saved = plot_sweep(df, plot, REPO / "results" / "figures" / exp.name / out_name)
        print(f"figure -> {saved.relative_to(REPO)}")

    if theory_rows:
        new = pd.DataFrame(theory_rows)
        keys = ["measure", "series_col", "x_col", "series_val", "x_val", "rep", "paired",
                "stake_draws"]
        if theory_cache is not None and not theory_cache.empty:
            old = theory_cache[[c for c in theory_cache.columns]]
            merged = pd.concat([old, new], ignore_index=True)
            merged = merged.drop_duplicates(subset=[k for k in keys if k in merged.columns],
                                            keep="last")
        else:
            merged = new
        tables.mkdir(parents=True, exist_ok=True)
        merged.to_csv(theory_csv, index=False)
        print(f"theory -> {theory_csv.relative_to(REPO)}  ({len(new)} new rows, "
              f"{len(merged)} total)")


if __name__ == "__main__":
    main()
