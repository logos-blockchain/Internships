"""
Complexity validation of the mix-net attribution attack: the |S| W^(k-1) work law, the
time model t = t_0 + K P' tau(k), the memory model peak = c_mem K P', the closed-form epoch
scale K = T lambda (cover + 1 + lambda), a machine-by-machine (W, k) capability ladder,
Monte-Carlo route sub-sampling (shown not to recover the posterior), and the polynomial
transfer-matrix DP in depth and width. Writes results/tables and the stage_2 figures.
Usage: python tests/mixnet-complexity-validation.py [--plot-only] [--T N --seed S --skip-epoch]
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotstyle  # noqa: E402

PLOT_ONLY = False

from consensus import sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, inject_dummies  # noqa: E402
from anonymity.mixnet import MixnetParams, apply as mixnet_apply  # noqa: E402
from anonymity.single_path_mix import delay_moments, random_delay_pdf  # noqa: E402
from network.lognormal_latency import LogNormalParams, lognormal_draw  # noqa: E402
from network.mixnet_latency import MAX_ROUTES, sample_mixnet_lognormal_profile  # noqa: E402
from adversary.gpa import observe_broadcasts  # noqa: E402
import adversary.mixnet_attribution as mixnet_attack  # noqa: E402
from adversary.mixnet_attribution import MixnetAttributionParams  # noqa: E402
from adversary.mixnet_attribution import run as run_mixnet_attribution  # noqa: E402
from metrics import deanon_top1  # noqa: E402
from pipeline_contract import PosteriorGuess  # noqa: E402

N_NODES = 1000
F_LOTTERY = 1 / 30
SHAPE = 4 / 3
COVER = 19
RHO = 4.0
MIX_DEFAULT = 0.003
T_EPOCH = 388_800
LOGN = LogNormalParams(floor=0.0072, mean=0.0629, sd=0.0333)

LAPTOP_NAME = "Intel i7-12800H (14 cores / 20 threads, 16 GB)"
LAPTOP_CORES = 14
LAPTOP_PEAK_CORE = 6.4e10       # flop/s, one P-core
LAPTOP_RAM = 16_847_892_480     # bytes, measured installed RAM (15.7 GiB)

MACHINES = [
    ("laptop, 1 core", LAPTOP_PEAK_CORE, "this machine, single-threaded (the shipped attack)"),
    ("laptop, 14 cores", LAPTOP_PEAK_CORE * LAPTOP_CORES, "this machine, fully parallel"),
    ("GPU (A100 80GB)", 9.7e12, "NVIDIA A100, fp64 vector peak"),
    ("Top500 #1 (El Capitan)", 1.742e18, "HPL Rmax, Top500 Nov-2024/Jun-2025"),
    ("Lloyd's ultimate laptop", 5.4e50, "1 kg, 1 litre at the Margolus-Levitin limit (Lloyd 2000)"),
]
UNIVERSE_OPS = 1e120
UNIVERSE_BITS = 1e90

CALIB_CSV = "mixnet_complexity_calibration.csv"

BUDGETS = [("1 hour", 3.6e3), ("1 day", 8.64e4), ("1 year", 3.156e7)]

FLOPS_PER_EVAL = lambda k: 30.0 + 15.0 * k    # noqa: E731


MEM_BUDGET_FRACTION = 0.5
C_MEM_WORST = 65.0


def write_calibration(path, **fields):
    """
    Cache the fitted constants (one CSV row, one column each) so figures can be redrawn later.

    The row also records the horizon, seed and machine, so a reduced-T calibration is not
    silently plotted as a full-epoch result.
    """
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields.keys()))
        w.writeheader()
        w.writerow(fields)
    print(f"   wrote {path.relative_to(REPO_ROOT)}  (redraw with --plot-only)")


def read_calibration(path):
    """Read the cached constants into a dict (floats parsed); exit loudly if no cache exists."""
    import csv
    if not path.exists():
        raise SystemExit(
            f"--plot-only needs {path.relative_to(REPO_ROOT)}, which a full run writes. "
            f"Run `python tests/mixnet-complexity-validation.py` once first.")
    with open(path, encoding="utf-8") as fh:
        row = next(iter(csv.DictReader(fh)))
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def _epoch_K(T):
    """Return E[K] = T lambda (cover + 1 + lambda), the candidate-row count at horizon T."""
    lam = -np.log(1 - F_LOTTERY)
    return T * lam * (COVER + 1 + lam)


def _fits(K, Pp, budget_frac=MEM_BUDGET_FRACTION):
    """Return whether a monolithic [K, P'] block fits in the budgeted share of RAM."""
    return C_MEM_WORST * K * Pp <= budget_frac * LAPTOP_RAM


def _runnable(grids, K, label):
    """Filter (W, k) grids to those that fit in memory, printing what was dropped and why."""
    keep, drop = [], []
    for W, k in grids:
        (keep if _fits(K, W ** (k - 1)) else drop).append((W, k))
    if drop:
        print(f"   !! skipped at the full epoch ({label}): "
              + ", ".join(f"{W}x{k} (P'={W**(k-1)}, needs "
                          f"{C_MEM_WORST * K * W**(k-1) / 2**30:,.0f} GiB)" for W, k in drop))
    return keep


def _fmt_seconds(secs):
    """Format a wall-clock in the largest readable unit (the ladder spans 1e-1 to 1e30 s)."""
    if secs < 3600:
        return f"{secs:,.1f} s"
    if secs < 8.64e4:
        return f"{secs/3600:,.1f} h"
    if secs < 3.156e7:
        return f"{secs/8.64e4:,.1f} days"
    return f"{secs/3.156e7:,.3g} CPU-years"


def _check(label, ok, detail=""):
    """Print a PASS/FAIL line for one check and return the boolean."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   -- {detail}" if detail else ""))
    return bool(ok)


def build_run(W, k, T, mix=MIX_DEFAULT, seed=0, N=N_NODES):
    """
    Build one mix-net realisation -> (trace, attack params, profile, sender_scale).

    Mirrors the pipeline's mixnet path (same stages, same order, profile drawn from a spawned
    child rng), so the timings below are of the shipped attack on shipped traffic.
    """
    send = mix / RHO
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, SHAPE, rng=rng)
    slots, nodes = simulate_events(alpha, f=F_LOTTERY, T=T, rng=rng)
    slots, nodes, is_dummy, group = inject_dummies(
        slots, nodes, N, params=DummyParams(count=COVER), T=T, rng=rng)
    prof = sample_mixnet_lognormal_profile(N, W, k, LOGN, rng=rng.spawn(1)[0], assignment="split")
    lp = MixnetParams(width=W, hops=k, mix_scale=mix, sender_scale=send,
                      receiver_delays=False, entry_assignment="split")
    trace = mixnet_apply(slots, nodes, is_dummy, group, params=lp, latency_oracle=prof, rng=rng)
    ap = MixnetAttributionParams(hops=k, mix_scale=mix, sender_scale=send,
                                 receiver_delays=False, latency_profile=prof)
    return trace, ap, prof, send


def _rows(obs):
    """Broadcast (receiver, y) to per-candidate rows, as the attack does; returns (counts, r, y)."""
    counts = np.diff(obs.start)
    return counts, np.repeat(obs.receiver, counts), np.repeat(obs.y, counts)


def _posterior_from_like(like, obs, counts):
    """Normalise likelihoods per broadcast (segment sums) with the attack's all-zero fallback."""
    seg = np.add.reduceat(like, obs.start[:-1]) if like.size else np.zeros(len(counts))
    denom = np.repeat(seg, counts)
    crow = np.repeat(counts, counts).astype(float)
    return np.where(denom > 0, like / np.where(denom > 0, denom, 1.0), 1.0 / crow)


def _argmax_rows(post, obs, counts):
    """Return the MAP candidate row of every broadcast (what deanon_top1 grades)."""
    return np.array([obs.start[b] + int(np.argmax(post[obs.start[b]:obs.start[b + 1]]))
                     for b in range(len(counts))])


def check_work_law(seed):
    """Check P = W^k routes and P' = W^(k-1) routes per candidate under split (W^k uniform)."""
    print("\n1. THE WORK LAW -- routes per candidate = W^(k-1) under `split`, W^k under `uniform`")
    print("   (the exponent is the depth, the base is the width)")
    ok = True
    print(f"   {'W':>3} {'k':>3} {'W^k':>8} {'split P/cand':>13} {'uniform P/cand':>15}")
    for W, k in [(2, 3), (2, 5), (3, 3), (4, 3), (3, 6), (4, 5), (2, 11)]:
        rng = np.random.default_rng(seed)
        ps = sample_mixnet_lognormal_profile(50, W, k, LOGN, rng=rng, assignment="split")
        pu = sample_mixnet_lognormal_profile(50, W, k, LOGN, rng=rng, assignment="uniform")
        ok &= (ps.n_routes == W ** k and ps.n_routes_per_sender == W ** (k - 1)
               and pu.n_routes_per_sender == W ** k)
        print(f"   {W:>3} {k:>3} {W**k:>8} {ps.n_routes_per_sender:>13} {pu.n_routes_per_sender:>15}")
    return _check("P = W^k routes, P' = W^(k-1) per candidate (split) / W^k (uniform)", ok,
                  "exact -- one independent node choice per layer, entry fixed under split")


def check_time_model(T, seed):
    """Calibrate t = t_0 + K P' tau(k); returns (ok, (a, b) of tau = a + b k in ns, t_0)."""
    print("\n2. THE TIME MODEL -- t = t_0 + K * P' * tau(k)   (K = candidate rows, P' = routes/candidate)")

    n = 2_000_000
    z = np.abs(np.random.default_rng(seed).normal(1.0, 0.5, n))
    ks, taus = [], []
    for k in (1, 2, 3, 4, 6, 8, 12, 16):
        random_delay_pdf(z[:1000], k, 0.25, 1.0)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            random_delay_pdf(z, k, 0.25, 1.0)
            reps.append(time.perf_counter() - t0)
        ks.append(k)
        taus.append(min(reps) / n * 1e9)
    b, a = np.polyfit(ks, taus, 1)
    resid = np.max(np.abs(np.polyval([b, a], ks) - np.array(taus)))
    print(f"   microbenchmark  tau(k) = {a:.1f} + {b:.2f} k  ns/eval   "
          f"(max residual {resid:.1f} ns over k = 1..16)")
    ok = _check("tau(k) is linear in k", b > 0 and resid < 0.35 * np.mean(taus),
                f"slope {b:.2f} ns per layer, intercept {a:.1f} ns")

    print(f"\n   the shipped attack at T = {T:,} (K held fixed; only P' varies)")
    grids = _runnable([(2, 3), (2, 5), (3, 3), (4, 3), (4, 4), (3, 6), (4, 5), (2, 9), (2, 11)],
                      _epoch_K(T), "timing grids")
    print(f"   {'W':>3} {'k':>3} {'P/cand':>8} {'K':>8} {'evals':>12} {'time_s':>9} {'ns/eval':>9}")
    rows = []
    for W, k in grids:
        trace, ap, prof, _ = build_run(W, k, T, seed=seed)
        g = run_mixnet_attribution(trace, params=ap)
        reps = []
        for _ in range(2):
            t0 = time.perf_counter()
            g = run_mixnet_attribution(trace, params=ap)
            reps.append(time.perf_counter() - t0)
        dt = min(reps)
        K, Pp = int(g.candidate.size), prof.n_routes_per_sender
        rows.append((W, k, Pp, K, dt))
        print(f"   {W:>3} {k:>3} {Pp:>8} {K:>8} {K*Pp:>12,} {dt:>9.3f} {dt/(K*Pp)*1e9:>9.1f}")

    evals = np.array([K * Pp for _, _, Pp, K, _ in rows], dtype=float)
    depth = np.array([k for _, k, *_ in rows], dtype=float)
    times = np.array([dt for *_, dt in rows])
    A = np.column_stack([np.ones_like(evals), evals * 1e-9, evals * depth * 1e-9])
    (t0, a_att, b_att), *_ = np.linalg.lstsq(A, times, rcond=None)
    pred = A @ np.array([t0, a_att, b_att])
    rel = float(np.max(np.abs(pred - times) / times))
    print(f"   fit: t = {t0*1e3:.1f} ms + K P' x ({a_att:.1f} + {b_att:.2f} k) ns,  "
          f"max relative residual {rel:.1%}")
    ok &= _check("attack time = fixed overhead + K P' x tau(k), tau linear in the depth",
                 rel < 0.25 and b_att > 0,
                 f"in-attack tau = {a_att:.0f} + {b_att:.1f}k ns vs the standalone "
                 f"{a:.0f} + {b:.1f}k ns -- the attack is cheaper per evaluation because it "
                 f"prunes z < 0 before the k-term polynomial")

    print("\n   scaling in the horizon (4x4 grid, P' = 64): is the cost linear in K?")
    print(f"   {'T':>10} {'K':>9} {'time_s':>9} {'ns/eval':>9}")
    ks_, ts_ = [], []
    for TT in (T // 16, T // 4, T):
        trace, ap, prof, _ = build_run(4, 4, TT, seed=seed)
        g = run_mixnet_attribution(trace, params=ap)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            g = run_mixnet_attribution(trace, params=ap)
            reps.append(time.perf_counter() - t0)
        dt = min(reps)
        Kx = int(g.candidate.size)
        ks_.append(Kx * prof.n_routes_per_sender)
        ts_.append(dt)
        print(f"   {TT:>10,} {Kx:>9,} {dt:>9.3f} {dt/ks_[-1]*1e9:>9.1f}")
    per_eval = np.array(ts_) / np.array(ks_)
    spread = float((per_eval.max() - per_eval.min()) / per_eval.mean())
    ok &= _check("cost per evaluation is flat in K, so the model extrapolates in the horizon",
                 spread < 0.30,
                 f"ns/eval varies {spread:.1%} over a {ks_[-1]/ks_[0]:.0f}x range in K -- so a "
                 f"different T, or the sweep's 1000 runs, is a MULTIPLIER on the ladder, not a "
                 f"different regime")
    return ok, (float(a_att), float(b_att)), float(t0)


def check_memory_model(T, seed):
    """Measure c_mem (bytes per [row x route] element) with row-chunking disabled; (ok, c_mem)."""
    print("\n3. THE MEMORY MODEL -- the attack's route block costs c BYTES PER ELEMENT")
    print("   (c/8 = the number of live float64 [rows, P'] blocks held at once)")
    print("   Three blocks are FULL-SIZE (mu, z, the pdf's output) plus a byte mask; the pdf's")
    print("   working temporaries (zz, the polynomial, the two exponentials) live on the")
    print("   z >= 0 SUBSET only, so c rises with the un-pruned fraction -- measured, not assumed.")
    print("   !! MEASURED WITH ROW-CHUNKING DISABLED, deliberately. c is DEFINED as the")
    print("      per-element cost of the block; the shipped attack streams that block in bounded")
    print("      row chunks (mixnet_attribution.CHUNK_ELEMENTS), so its peak is c * min(CHUNK,")
    print("      K*P') -- a CAP, not a per-element cost. Measuring the shipped path here would")
    print("      read the cap back as a (too small) c and corrupt every figure that uses it.")
    print(f"   {'W':>3} {'k':>3} {'P/cand':>8} {'mix':>7} {'frac z>=0':>10} {'peak_MB':>9} "
          f"{'bytes/elem':>11} {'blocks':>8}")
    cs, fracs = [], []
    cells = [((2, 8), MIX_DEFAULT), ((3, 6), MIX_DEFAULT), ((4, 5), MIX_DEFAULT),
             ((2, 9), MIX_DEFAULT), ((3, 6), 1.0), ((3, 6), 0.0001)]
    cells = [(g, m) for g, m in cells if _fits(_epoch_K(T), g[0] ** (g[1] - 1))]
    chunk_shipped = mixnet_attack.CHUNK_ELEMENTS
    mixnet_attack.CHUNK_ELEMENTS = 1 << 62
    try:
        for (W, k), mix in cells:
            trace, ap, prof, _ = build_run(W, k, T, mix=mix, seed=seed)
            run_mixnet_attribution(trace, params=ap)
            tracemalloc.start()
            tracemalloc.reset_peak()
            g = run_mixnet_attribution(trace, params=ap)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            K, Pp = int(g.candidate.size), prof.n_routes_per_sender
            obs = observe_broadcasts(trace)
            _, r_row, y_row = _rows(obs)
            frac = float(np.mean(y_row[:, None] - prof.route_mu(obs.candidate, r_row) >= 0.0))
            c = peak / (K * Pp)
            cs.append(c)
            fracs.append(frac)
            print(f"   {W:>3} {k:>3} {Pp:>8} {mix:>7g} {frac:>10.3f} {peak/1e6:>9.1f} "
                  f"{c:>11.2f} {c/8:>8.2f}")
    finally:
        mixnet_attack.CHUNK_ELEMENTS = chunk_shipped
    blocks = np.array(cs) / 8
    fr = np.array(fracs)
    c1, c0 = np.polyfit(fr, blocks, 1)
    resid = float(np.max(np.abs(np.polyval([c1, c0], fr) - blocks)))
    print(f"   fit: blocks = {c0:.2f} + {c1:.2f} x frac(z >= 0)   (max residual {resid:.2f} blocks)")
    c_mem = float(np.max(cs))
    ok = _check("peak memory is c * K * P' with c set by the pruned fraction",
                resid < 0.35 and 2.5 <= c0 <= 4.0 and 3.0 <= c1 <= 8.0,
                f"the frac -> 0 intercept is {c0:.2f} full [K, P'] blocks (mu, z, out + the byte "
                f"mask = 3.1 expected); the slope {c1:.1f} is the pdf's subset temporaries. "
                f"c spans {min(cs):.0f}-{max(cs):.0f} B/elem across mix budgets; the ladder uses "
                f"the worst case {c_mem:.0f}")
    return ok, c_mem


def check_epoch_scale(seed, run_full):
    """Check the full-epoch candidate-row count K against its closed form; returns (ok, K)."""
    print("\n4. THE EPOCH SCALE -- K = T * lambda * (cover + 1 + lambda),  lambda = -ln(1-f)")
    lam = -np.log(1 - F_LOTTERY)
    B_hat = T_EPOCH * lam
    K_hat = B_hat * (COVER + 1 + lam)
    print(f"   lambda = {lam:.6f}   E[B] = {B_hat:,.0f} broadcasts   "
          f"E[|S_t|] = {COVER + 1 + lam:.4f}   E[K] = {K_hat:,.0f} candidate rows")
    if not run_full:
        print("   (--skip-epoch: the full-epoch measurement is skipped; the ladder uses the closed form)")
        return True, K_hat
    trace, ap, prof, _ = build_run(2, 3, T_EPOCH, seed=seed)
    obs = observe_broadcasts(trace)
    B, K = len(obs), int(obs.candidate.size)
    t0 = time.perf_counter()
    run_mixnet_attribution(trace, params=ap)
    dt = time.perf_counter() - t0
    z = (K - K_hat) / np.sqrt(K_hat)
    print(f"   measured at T = {T_EPOCH:,} (W=2, k=3): B = {B:,}  K = {K:,}  "
          f"(closed form {K_hat:,.0f}, {z:+.2f} sqrt-K)   attack {dt:.2f} s")
    ok = _check("the epoch's candidate-row count matches the candidate-set law", abs(z) < 3.0,
                f"{K:,} vs {K_hat:,.0f} -- the ladder's K is derived, not fitted")
    return ok, float(K)


def log10_enum_seconds(W, k, K, tau_fit):
    """
    Return log10 of the exact enumeration's wall-clock for one epoch on one laptop core.

    t = K W^(k-1) tau(k) with tau(k) = a + b k ns; in logs because the far end overflows float64.
    """
    a, b = tau_fit
    return (np.log10(K) + (k - 1) * np.log10(W) + np.log10(a + b * k) - 9.0)


def log10_enum_ops(W, k, K):
    """Return log10 of the enumeration's total operation count at the ideal flop count."""
    return np.log10(K) + (k - 1) * np.log10(W) + np.log10(FLOPS_PER_EVAL(k))


def k_max(W, log10_budget, K, tau_fit, ops=False, k_hi=2000):
    """
    Return the deepest width-W grid a budget can enumerate (fractional, linearly interpolated).

    Cost is strictly increasing in k, so the crossing is unique. `ops=True` prices in total
    operations instead of laptop-core seconds.
    """
    if W <= 1:
        return float("inf")
    cost = (lambda kk: log10_enum_ops(W, kk, K)) if ops else \
           (lambda kk: log10_enum_seconds(W, kk, K, tau_fit))
    prev = cost(1.0)
    if prev > log10_budget:
        return 1.0
    for k in range(2, k_hi):
        cur = cost(float(k))
        if cur > log10_budget:
            return (k - 1) + (log10_budget - prev) / (cur - prev)
        prev = cur
    return float(k_hi)


def check_ladder(K, tau_fit, c_mem, out_csv):
    """Price the largest enumerable (W, k) per machine and budget; returns (ok, rows, P_mem)."""
    print("\n5. THE LADDER -- the largest ENUMERABLE mix-net, machine by machine")
    print(f"   (K = {K:,.0f} candidate rows per full epoch, ONE rep; the shipped `split` attack.")
    print("    Real machines are the MEASURED single-core rate scaled by fp64 peak -- equal")
    print("    efficiency assumed on every tier, so the flops-per-evaluation constant cancels.")
    print("    The two physical rows are budgets in TOTAL operations at the IDEAL flop count,")
    print("    which is the charitable reading for the adversary -- the standing policy.)")
    widths = (2, 4, 16, 100)
    rows = []
    print(f"   {'machine':>26} {'budget':>11} {'log10 budget':>13} "
          + " ".join(f"{'k@W=' + str(w):>9}" for w in widths))
    for name, peak, note in MACHINES:
        speed = peak / LAPTOP_PEAK_CORE
        for bname, secs in BUDGETS:
            lb = np.log10(secs * speed)
            r = dict(machine=name, note=note, budget=bname, seconds=secs,
                     speedup_vs_laptop_core=speed, log10_core_seconds=lb, priced_in="seconds")
            for w in widths:
                r[f"k_max_W{w}"] = k_max(w, lb, K, tau_fit)
            rows.append(r)
            print(f"   {name:>26} {bname:>11} {lb:>13.2f} "
                  + " ".join(f"{r['k_max_W' + str(w)]:>9.1f}" for w in widths))
    r = dict(machine="the observable universe", budget="all of time", seconds=float("inf"),
             note=f"Lloyd 2002: <=1e120 operations on <=1e90 bits over its whole history",
             speedup_vs_laptop_core=float("inf"), log10_core_seconds=float("inf"),
             priced_in="total operations")
    for w in widths:
        r[f"k_max_W{w}"] = k_max(w, np.log10(UNIVERSE_OPS), K, tau_fit, ops=True)
    rows.append(r)
    print(f"   {'the observable universe':>26} {'all of time':>11} {'1e120 ops':>13} "
          + " ".join(f"{r['k_max_W' + str(w)]:>9.1f}" for w in widths))

    P_mem = LAPTOP_RAM / (c_mem * K)
    chunk_gib = c_mem * mixnet_attack.CHUNK_ELEMENTS / 2 ** 30
    print(f"\n   memory on {LAPTOP_NAME}: NO LONGER A RUNG (the attack is row-chunked)")
    print(f"     shipped peak = c x min(CHUNK, K P') <= {chunk_gib:.2f} GiB at EVERY (W, k) "
          f"(CHUNK = {mixnet_attack.CHUNK_ELEMENTS:,} elements)")
    print(f"     the OLD monolithic ceiling was P' <= {P_mem:,.0f} routes/candidate "
          f"({c_mem:.0f} B/elem x {K:,.0f} rows against {LAPTOP_RAM/2**30:.0f} GiB)")
    kfit = lambda W: int(np.floor(1 + np.log(P_mem) / np.log(W)))          # noqa: E731
    print(f"       i.e. it used to stop at W=2 -> k <= {kfit(2)}, W=4 -> k <= {kfit(4)}, "
          f"W=16 -> k <= {kfit(16)}; those caps are GONE.")
    print("     !! that rung was an ARTEFACT of vectorising the whole epoch at once, and chunking "
          "removed it\n        at BIT-IDENTICAL output (anonymity-mixnet-validation.py --section "
          "chunking).\n        The COMPUTE rungs above did not move. Memory was this CODE's wall; "
          "compute is the PROBLEM's.")

    print("\n   where real grids sit (full epoch, one rep, one measured laptop core):")
    k_guard = int(np.floor(np.log(MAX_ROUTES) / np.log(4)))
    for label, W, k in [("shipped 2x3", 2, 3), (f"MAX_ROUTES guard ({MAX_ROUTES:,})", 4, k_guard),
                        ("NYM as deployed (40x3)", 40, 3),
                        ("Nym + one layer (40x4)", 40, 4), ("a deep grid", 8, 12)]:
        unit = _fmt_seconds(10.0 ** log10_enum_seconds(W, k, K, tau_fit))
        gb = c_mem * K * float(W) ** (k - 1) / 2 ** 30
        peak = min(gb, c_mem * mixnet_attack.CHUNK_ELEMENTS / 2 ** 30)
        print(f"     {label:>27} (W={W:>3}, k={k:>2}): P' = {W**(k-1):>18,}  "
              f"enumeration {unit:>16}  block {gb:>14,.1f} GiB  peak {peak:>6.2f} GiB")

    print("\n   the 'computer the size of the universe' boundary (1e120 operations, Lloyd 2002):")
    for W in (2, 4, 16, 100, 1000):
        kk = k_max(W, np.log10(UNIVERSE_OPS), K, tau_fit, ops=True)
        print(f"     W = {W:>4}: enumeration exhausts it at k = {kk:>5.0f} layers  "
              f"({W*kk:>8,.0f} mix nodes;  E[mu] = {(kk+1)*LOGN.mean:>7.2f} s of link latency alone)")
    print("     (memory too: 1e90 bits is ~1e89 float64s, so even STORING the route list of a")
    print("      W=2, k=300 grid exceeds the universe's storage -- the two walls arrive together.)")

    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n   wrote {out_csv.relative_to(REPO_ROOT)}")

    by_machine = {n: [r for r in rows if r["machine"] == n] for n, *_ in MACHINES}
    mono = all(all(g[i]["k_max_W2"] <= g[i + 1]["k_max_W2"] + 1e-9 for i in range(len(g) - 1))
               for g in by_machine.values())
    for bname, _ in BUDGETS:
        seq = [r["k_max_W2"] for r in rows if r["budget"] == bname]
        mono &= all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
    ok = _check("the ladder is monotone in capability, and depth is the dial that matters",
                mono and rows[-1]["k_max_W2"] > rows[0]["k_max_W2"],
                f"40 orders of magnitude of compute, from 1 core-hour to the whole universe, buy "
                f"k = {rows[0]['k_max_W2']:.0f} -> {rows[-1]['k_max_W2']:.0f} layers at W = 2 "
                f"(the cost is exponential in k, so capability enters only through its logarithm)")
    return ok, rows, P_mem


def check_mc_sampling(T, seed):
    """Check that sub-sampling M of the P' routes does not reproduce the exact MAP; returns ok."""
    print("\n6. MONTE-CARLO ROUTE SUB-SAMPLING -- unbiased, and still not usable")
    W, k = 3, 6
    trace, ap, prof, send = build_run(W, k, T, seed=seed)
    exact = run_mixnet_attribution(trace, params=ap)
    obs = observe_broadcasts(trace)
    counts, r_row, y_row = _rows(obs)
    cand, Pp = obs.candidate, prof.n_routes_per_sender
    am_exact = _argmax_rows(exact.posterior, obs, counts)
    top1_exact = deanon_top1(exact, trace)
    rng = np.random.default_rng(seed + 1)
    print(f"   grid W={W}, k={k}: P' = {Pp} routes per candidate; exact Top-1 = {top1_exact:.4f}")
    print(f"   {'M':>6} {'Top-1':>8} {'MAP agreement':>15} {'mean|dP|':>10} {'time_s':>8}")
    agreements = {}
    for M in (1, 4, 16, 64, Pp):
        t0 = time.perf_counter()
        pick = rng.integers(0, Pp, size=(cand.size, M))
        rts = prof.routes_by_entry[prof.entry_of[cand][:, None], pick]
        mu = (prof.d_sender[cand, prof.entry_of[cand]][:, None]
              + prof.route_internal[rts]
              + prof.d_receiver[r_row[:, None], prof.route_exit[rts]])
        like = random_delay_pdf((y_row[:, None] - mu).ravel(), k, send, ap.mix_scale
                                ).reshape(cand.size, M).mean(axis=1)
        post = _posterior_from_like(like, obs, counts)
        dt = time.perf_counter() - t0
        g = PosteriorGuess(broadcast_row=obs.broadcast_row, start=obs.start,
                           candidate=cand, posterior=post)
        agree = float((_argmax_rows(post, obs, counts) == am_exact).mean())
        agreements[M] = agree
        print(f"   {M:>6} {deanon_top1(g, trace):>8.4f} {agree:>15.4f} "
              f"{np.mean(np.abs(post - exact.posterior)):>10.5f} {dt:>8.3f}")
    ok = _check("MC route sampling does NOT recover the exact posterior", agreements[Pp] < 0.9,
                f"even M = P' = {Pp} draws (with replacement) agrees with the exact MAP only "
                f"{agreements[Pp]:.1%} of the time -- f_Z is sharply peaked, so a few routes carry "
                f"the mixture and the estimator's variance dominates")
    return ok


def dp_likelihood(prof, cand, r_row, y_row, k, send, mix, points_per_sigma=40):
    """
    Compute the exact route marginalisation as a transfer-matrix DP -> (likelihood, timings, G).

    The route latency D_p is a sum of one link per layer, so the layered grid factorises:
      rho^(1)_{w,u} = delta(internal[1, w, u]),
      rho^(j+1)_{w,v} = sum_u rho^(j)_{w,u} * delta(internal[j+1, u, v]),
      S_{w,v} = rho^(k-1)_{w,v} * f_Z,
      L_i = (1/P') sum_v S_{w(i), v}(y - a_i - b_{r,v}).
    Cost O(k W^3 G) + O(W^2 G log G) per profile, then O(K W) per epoch. The only approximation
    is the grid step h = sigma_Z / points_per_sigma. Reads the same public knowledge as the
    enumeration (grid links and delay params).
    """
    W = prof.width
    sigma_Z = float(np.sqrt(delay_moments(k, send, mix)[1]))       # (mean, var)[1]
    internal = prof.grid_internal
    h = sigma_Z / points_per_sigma
    D_max = float(sum(internal[j].max() for j in range(k - 1))) if k > 1 else 0.0
    G_D = int(np.ceil(D_max / h)) + 2

    t0 = time.perf_counter()
    rho = np.zeros((W, W, G_D))
    for w in range(W):
        cur = np.zeros((W, G_D))
        if k == 1:
            cur[w, 0] = 1.0
        else:
            d = internal[0, w] / h
            i0 = np.floor(d).astype(int)
            fr = d - i0
            for u in range(W):
                cur[u, i0[u]] += 1 - fr[u]
                cur[u, i0[u] + 1] += fr[u]
            for j in range(1, k - 1):
                nxt = np.zeros_like(cur)
                for u in range(W):
                    for v in range(W):
                        d = internal[j, u, v] / h
                        s = int(np.floor(d))
                        fr = d - s
                        nxt[v, s:] += (1 - fr) * cur[u, :G_D - s]
                        nxt[v, s + 1:] += fr * cur[u, :G_D - s - 1]
                cur = nxt
        rho[w] = cur
    t_paths = time.perf_counter() - t0

    t0 = time.perf_counter()
    G_Z = int(np.ceil((send + k * mix + 40 * sigma_Z) / h))
    fz = random_delay_pdf(np.arange(G_Z) * h, k, send, mix)
    G_S = G_D + G_Z
    nfft = 1 << int(np.ceil(np.log2(G_S)))
    FZ = np.fft.rfft(fz, nfft)
    S = np.fft.irfft(np.fft.rfft(rho.reshape(W * W, G_D), nfft, axis=1) * FZ,
                     nfft, axis=1)[:, :G_S].reshape(W, W, G_S)
    t_conv = time.perf_counter() - t0

    t0 = time.perf_counter()
    e = prof.entry_of[cand]
    base = y_row - prof.d_sender[cand, e]
    like = np.zeros(cand.size)
    for v in range(W):
        x = (base - prof.d_receiver[r_row, v]) / h
        i0 = np.floor(x).astype(np.int64)
        fr = x - i0
        ok = (i0 >= 0) & (i0 + 1 < G_S)
        idx = np.clip(i0, 0, G_S - 2)
        like += np.where(ok, (1 - fr) * S[e, v, idx] + fr * S[e, v, idx + 1], 0.0)
    like /= prof.n_routes_per_sender
    t_eval = time.perf_counter() - t0
    return like, (t_paths, t_conv, t_eval), G_S


class _RawProfile:
    """Duck-typed latency profile for grids past MAX_ROUTES; the DP reads only these fields."""

    def __init__(self, N, W, k, rng):
        """Draw the sender, receiver and internal grid latencies for a W x k grid."""
        draw = lognormal_draw(LOGN)
        self.width, self.hops = W, k
        self.d_sender = draw(rng, (N, W))
        self.d_receiver = draw(rng, (N, W))
        self.grid_internal = draw(rng, (k - 1, W, W)) if k > 1 else None
        self.entry_of = np.arange(N, dtype=np.int64) % W
        self.n_routes_per_sender = W ** (k - 1)


def check_transfer_matrix_dp(T, seed, tau_fit):
    """Validate the DP against the enumeration and measure its k-scaling; (ok, (icept, slope))."""
    print("\n7. THE TRANSFER-MATRIX DP -- the same exact marginalisation in POLYNOMIAL time")
    ok = True

    print(f"   (a) against the enumeration, across grids and mix budgets (T = {T:,})")
    print(f"   {'W':>3} {'k':>3} {'P/cand':>7} {'mix':>7} {'Top-1 enum':>11} {'Top-1 DP':>9} "
          f"{'max|dP|':>9} {'MAP agree':>10} {'t_enum':>8} {'t_DP':>7}")
    worst_top1, worst_agree = 0.0, 1.0
    for (W, k) in _runnable([(2, 3), (3, 6), (4, 5), (2, 9), (2, 11)], _epoch_K(T), "DP-vs-enum"):
        for mix in (0.0001, MIX_DEFAULT, 1.0):
            trace, ap, prof, send = build_run(W, k, T, mix=mix, seed=seed)
            t0 = time.perf_counter()
            ex = run_mixnet_attribution(trace, params=ap)
            t_enum = time.perf_counter() - t0
            obs = observe_broadcasts(trace)
            counts, r_row, y_row = _rows(obs)
            like, ts, _ = dp_likelihood(prof, obs.candidate, r_row, y_row, k, send, mix)
            post = _posterior_from_like(like, obs, counts)
            g = PosteriorGuess(broadcast_row=obs.broadcast_row, start=obs.start,
                               candidate=obs.candidate, posterior=post)
            t1e, t1d = deanon_top1(ex, trace), deanon_top1(g, trace)
            agree = float((_argmax_rows(post, obs, counts)
                           == _argmax_rows(ex.posterior, obs, counts)).mean())
            worst_top1 = max(worst_top1, abs(t1e - t1d))
            worst_agree = min(worst_agree, agree)
            print(f"   {W:>3} {k:>3} {prof.n_routes_per_sender:>7} {mix:>7g} {t1e:>11.4f} "
                  f"{t1d:>9.4f} {np.max(np.abs(post - ex.posterior)):>9.2e} {agree:>10.4f} "
                  f"{t_enum:>8.3f} {sum(ts):>7.3f}")
    ok &= _check("the DP reproduces the enumerated attack's privacy number", worst_top1 < 0.005,
                 f"worst |Top-1 difference| = {worst_top1:.4f} over 12 (grid, mix) cells; "
                 f"worst MAP agreement {worst_agree:.3f} (the disagreements are at mix = 1.0, "
                 f"where every posterior is on the 1/|S| null and the argmax is a coin flip)")

    print("\n   (b) the error is a GRID parameter, not a limitation -- refine and it falls")
    W, k, mix = 3, 6, MIX_DEFAULT
    trace, ap, prof, send = build_run(W, k, T, mix=mix, seed=seed)
    ex = run_mixnet_attribution(trace, params=ap)
    obs = observe_broadcasts(trace)
    counts, r_row, y_row = _rows(obs)
    errs = []
    print(f"   {'points/sigma_Z':>15} {'grid G':>10} {'max|dP|':>10} {'mean|dP|':>10} {'t_DP':>7}")
    for pps in (5, 20, 80, 320):
        like, ts, G = dp_likelihood(prof, obs.candidate, r_row, y_row, k, send, mix,
                                    points_per_sigma=pps)
        post = _posterior_from_like(like, obs, counts)
        e = float(np.max(np.abs(post - ex.posterior)))
        errs.append(e)
        print(f"   {pps:>15} {G:>10,} {e:>10.2e} "
              f"{np.mean(np.abs(post - ex.posterior)):>10.2e} {sum(ts):>7.3f}")
    ok &= _check("refining the grid drives the DP onto the enumeration", errs[-1] < errs[0] / 5,
                 f"max|dP| {errs[0]:.1e} -> {errs[-1]:.1e} over a 64x refinement "
                 f"(the only approximation is placing each edge's shift on the grid)")

    print("\n   (c) beyond the enumeration wall -- grids no adversary could ever list")
    print(f"   {'W':>4} {'k':>3} {'routes/candidate':>20} {'enum @ full epoch':>20} "
          f"{'t_DP':>7} {'DP mem MB':>10}")
    lam = -np.log(1 - F_LOTTERY)
    K_epoch = T_EPOCH * lam * (COVER + 1 + lam)
    rng = np.random.default_rng(seed)
    for (W, k) in [(4, 5), (8, 8), (16, 10), (32, 6)]:
        prof = _RawProfile(N_NODES, W, k, rng)
        Kc = int(K_epoch)
        cand = rng.integers(0, N_NODES, Kc)
        r_row = rng.integers(0, N_NODES, Kc)
        y_row = (prof.d_sender[cand, prof.entry_of[cand]] + prof.d_receiver[r_row, 0]
                 + (k - 1) * LOGN.mean + rng.exponential(MIX_DEFAULT, Kc))
        like, ts, G = dp_likelihood(prof, cand, r_row, y_row, k, MIX_DEFAULT / RHO,
                                    MIX_DEFAULT, points_per_sigma=40)
        unit = _fmt_seconds(10.0 ** log10_enum_seconds(W, k, K_epoch, tau_fit))
        print(f"   {W:>4} {k:>3} {W**(k-1):>20,} {unit:>20} {sum(ts):>7.3f} "
              f"{W*W*G*8/1e6:>10.1f}")
        ok &= bool(np.all(np.isfinite(like)))

    print("\n   (d) the DP's own scaling in the depth, at W = 4 (the figure's blue curve)")
    print("   Split into its two halves, because only ONE of them can depend on k: the transfer")
    print("   steps (one per layer) and the per-candidate table lookup (O(K W), k-INDEPENDENT by")
    print("   construction). At full-epoch K the lookup is the larger and noisier term, so the")
    print("   linearity claim is tested where it lives -- on the propagation.")
    W, Kc = 4, int(K_epoch)
    ks, t_prop, t_look = [], [], []
    print(f"   {'k':>4} {'propagate_s':>12} {'lookup_s':>10} {'total_s':>9}")
    for k in (3, 6, 10, 16, 24, 32):
        prof = _RawProfile(N_NODES, W, k, rng)
        cand = rng.integers(0, N_NODES, Kc)
        r_row = rng.integers(0, N_NODES, Kc)
        y_row = (prof.d_sender[cand, prof.entry_of[cand]] + prof.d_receiver[r_row, 0]
                 + (k - 1) * LOGN.mean + rng.exponential(MIX_DEFAULT, Kc))
        _, ts, _ = dp_likelihood(prof, cand, r_row, y_row, k, MIX_DEFAULT / RHO, MIX_DEFAULT)
        ks.append(k)
        t_prop.append(ts[0] + ts[1])
        t_look.append(ts[2])
        print(f"   {k:>4} {t_prop[-1]:>12.4f} {t_look[-1]:>10.4f} {sum(ts):>9.4f}")
    slope, icept = np.polyfit(ks, t_prop, 1)
    lookup = float(np.mean(t_look))
    corr = float(np.corrcoef(ks, t_prop)[0, 1])
    print(f"   fit: propagation = {icept:.4f} + {slope:.4f} k  s   (r = {corr:.3f});  "
          f"lookup = {lookup:.3f} s, flat in k")
    ok &= _check("the DP's cost is linear in k where the enumeration's is exponential",
                 slope > 0 and corr > 0.9,
                 f"{slope*1e3:.1f} ms per extra layer -- the layer count multiplies the transfer "
                 f"steps, it does not multiply the state space. The rest of the DP ({lookup:.3f} s) "
                 f"is the k-independent per-candidate lookup.")
    return (_check("the DP runs grids the enumeration cannot reach", ok,
                   "W=16, k=10 is 6.9e10 routes per candidate -- decades of CPU time to enumerate, "
                   "sub-second as a transfer-matrix product"),
            (float(icept + lookup), float(slope)))


def check_dp_width_scaling(seed):
    """
    Measure the DP's wall-clock against W at fixed k = 3 -> (ok, (c0, c1, c2, c3), W_max).

    Fits t_DP(W) = c0 + c1 W + c2 W^2 + c3 W^3 (non-negative least squares): the lookup is
    O(K W) and the propagation under `split` is O(k W^3 G), one pass per entry. Full-epoch
    candidate rows, so the curve is comparable to the enumeration's.
    """
    print("\n8. THE DP ACROSS THE WIDTH AT k = 3 -- the fixed-depth cut of the cost curve")
    print("   (Fix the depth and vary W: the enumeration's exponent is gone -- K W^2 is")
    print("    polynomial -- so NEITHER algorithm ever walls out in width. Measured for the DP,")
    print("    exact work law for the enumeration; the width figure draws both.)")
    k = 3
    send, mix = MIX_DEFAULT / RHO, MIX_DEFAULT
    Kc = int(_epoch_K(T_EPOCH))
    rng = np.random.default_rng(seed)
    sigma_Z = float(np.sqrt(delay_moments(k, send, mix)[1]))
    h = sigma_Z / 40.0
    Ws, props, looks, totals = [], [], [], []
    print(f"   {'W':>5} {'P/cand':>8} {'propagate_s':>12} {'lookup_s':>10} {'total_s':>9}")
    for W in (2, 4, 8, 16, 32, 64, 96):
        prof = _RawProfile(N_NODES, W, k, rng)
        D_max = float(sum(prof.grid_internal[j].max() for j in range(k - 1)))
        G_D = int(np.ceil(D_max / h)) + 2
        G_S = G_D + int(np.ceil((send + k * mix + 40 * sigma_Z) / h))
        nfft = 1 << int(np.ceil(np.log2(G_S)))
        need = 8.0 * W * W * (G_D + G_S) + 32.0 * W * W * (nfft // 2 + 1)
        if need > MEM_BUDGET_FRACTION * LAPTOP_RAM:
            print(f"   !! skipped W = {W}: the DP tables want {need / 2**30:.1f} GiB")
            continue
        cand = rng.integers(0, N_NODES, Kc)
        r_row = rng.integers(0, N_NODES, Kc)
        y_row = (prof.d_sender[cand, prof.entry_of[cand]] + prof.d_receiver[r_row, 0]
                 + (k - 1) * LOGN.mean + rng.exponential(mix, Kc))
        _, ts, _ = dp_likelihood(prof, cand, r_row, y_row, k, send, mix)
        Ws.append(float(W))
        props.append(ts[0] + ts[1])
        looks.append(ts[2])
        totals.append(sum(ts))
        print(f"   {W:>5} {W**(k-1):>8,} {props[-1]:>12.4f} {looks[-1]:>10.4f} {totals[-1]:>9.4f}")
    from scipy.optimize import nnls
    Wa, t = np.array(Ws), np.array(totals)
    A = np.stack([np.ones_like(Wa), Wa, Wa * Wa, Wa ** 3], axis=1)
    coef = nnls(A / t[:, None], np.ones_like(t))[0]
    rel = float(np.max(np.abs(A @ coef - t) / t))
    print(f"   fit: t_DP(W) = {coef[0]:.4f} + {coef[1]:.5f} W + {coef[2]:.6f} W^2 "
          f"+ {coef[3]:.2e} W^3  s   (max rel. residual {rel:.0%})")
    ok = _check("the DP's width cost is the KW + kW^3G polynomial (cubic: one pass PER ENTRY)",
                rel < 0.35 and t[-1] > t[0],
                f"lookup ~ {coef[1] * 1e3:.1f} ms per unit of W, propagation ~ "
                f"{coef[3] * 1e6:.1f} us per W^3; extrapolated to W = 1000 that is "
                f"{_fmt_seconds(float(coef @ np.array([1.0, 1e3, 1e6, 1e9])))} "
                f"against the enumeration's ~4 h -- polynomial both, width is a wall for NOBODY")
    return ok, tuple(float(c) for c in coef), float(Wa.max())


def save_ladder_figure(out_path, K, tau_fit):
    """
    Draw the (W, k) capability ladder: what each machine tier can still enumerate.

    Only compute rungs are drawn; the chunked attack has no memory ceiling in this range. The
    ceiling is Lloyd's 1 kg physical limit, not the observable universe (table row only).
    """
    fig, ax = plt.subplots()
    Wg = np.unique(np.round(np.logspace(np.log10(2), 3, 120)).astype(int))
    tiers = [
        ("laptop core, 1 day", np.log10(8.64e4), False, "#0072B2"),
        ("8-GPU cloud node, 1 month",
         np.log10(2.63e6 * 8 * 9.7e12 / LAPTOP_PEAK_CORE), False, "#D55E00"),
        ("fastest supercomputer, 1 year",
         np.log10(3.156e7 * 1.742e18 / LAPTOP_PEAK_CORE), False, "#009E73"),
        ("1 kg of matter, physical limit, 1 year",
         np.log10(3.156e7 * 5.4e50 / LAPTOP_PEAK_CORE), False, "#000000"),
    ]
    kg = np.linspace(0.0, 60.0, 300)
    Wm, km = np.meshgrid(Wg.astype(float), kg)
    terms_log = (km - 1) * np.log10(Wm)
    shade = LinearSegmentedColormap.from_list("routes", ["#EFEAF7", "#BBA9DC", "#7C68B4"])
    mesh = ax.pcolormesh(Wm, km, np.clip(terms_log, 0.0, 50.0), cmap=shade,
                         shading="gouraud", vmin=0.0, vmax=50.0, zorder=0)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(r"Routes per sender  $W^{L-1}$", fontsize=15)
    cb.set_ticks([0, 10, 20, 30, 40, 50])
    cb.ax.set_yticklabels([r"$10^{%d}$" % t for t in (0, 10, 20, 30, 40, 50)], fontsize=13)

    for label, lb, ops, colour in tiers:
        curve = np.array([k_max(int(w), lb, K, tau_fit, ops=ops) for w in Wg], dtype=float)
        ax.plot(Wg, curve, lw=2.6, color=colour, zorder=3, label=label)
    ax.plot([40], [3], "o", ms=10, color="#000000", zorder=6)
    ax.annotate("Nym as deployed\n$W = 40$, $L = 3$", (40, 3), textcoords="offset points",
                xytext=(12, 10), fontsize=15, zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(2, 1e3)
    ax.set_ylim(0, 60)
    ax.set_xlabel(r"Mix nodes per layer  $W$")
    ax.set_ylabel(r"Mix layers  $L$")
    leg = ax.legend(fontsize=15, loc="upper right", framealpha=0.92, facecolor="white",
                    edgecolor="none")
    leg.set_zorder(5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    saved = plotstyle.save(fig, out_path)
    print(f"\n   wrote {Path(saved).relative_to(REPO_ROOT)}")
    return saved


def save_memory_figure(out_path, K, c_mem):
    """
    Draw the memory the enumeration needs against depth at W = 4.

    Two implementations of the same arithmetic: the monolithic [K, P'] block, and the shipped
    row-chunked attack whose peak is capped at c * CHUNK_ELEMENTS.
    """
    fig, ax = plt.subplots()
    W = 4
    kk = np.arange(2, 17, dtype=float)
    Pp = float(W) ** (kk - 1)
    mono = c_mem * K * Pp
    ax.semilogy(kk, mono, lw=2.4, color="#D55E00", ls="--",
                label=r"monolithic: whole epoch in one $[K, P']$ block")
    ax.semilogy(kk, np.minimum(mono, c_mem * mixnet_attack.CHUNK_ELEMENTS), lw=2.4, color="#0072B2",
                label=r"shipped: bounded row chunks (capped)")
    for by, txt in [(LAPTOP_RAM, "this laptop, 16 GB"), (1e12, "a 1 TB server")]:
        ax.axhline(by, color="grey", lw=1.0, ls=":")
        ax.annotate(txt, (2.2, by * 2.2), fontsize=13, color="grey")
    ax.set_xlabel(r"Layers $k$   (width $W = 4$)")
    ax.set_ylabel(r"Memory the enumeration needs (bytes)")
    ax.set_ylim(1e3, 1e22)
    ax.legend(fontsize=13, loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    saved = plotstyle.save(fig, out_path)
    print(f"   wrote {Path(saved).relative_to(REPO_ROOT)}")
    return saved


def save_cost_curve_figure(out_path, K, tau_fit, dp_fit):
    """Draw enumeration vs the transfer-matrix DP against depth at W = 4."""
    fig, ax = plt.subplots()
    W = 4
    kk = np.arange(2, 41, dtype=float)
    enum_s = 10.0 ** np.array([log10_enum_seconds(W, k, K, tau_fit) for k in kk])
    dp_s = dp_fit[0] + dp_fit[1] * kk
    ax.semilogy(kk, enum_s, lw=2.4, color="#D55E00",
                label=r"exact enumeration  $\propto K\,W^{L-1}$")
    ax.semilogy(kk, dp_s, lw=2.4, color="#0072B2",
                label=r"transfer-matrix DP  $\propto L\,W^{3} G + K\,W$")
    for secs, txt in [(3.6e3, "1 hour"), (3.156e7, "1 year"), (3.156e9, "100 years")]:
        ax.axhline(secs, color="grey", lw=1.0, ls=":")
        ax.annotate(txt, (2.4, secs * 2.2), fontsize=15, color="grey")
    ax.set_xlabel(r"Mix layers  $L$   (at $W = 4$)")
    ax.set_ylabel(r"Running time per epoch  (s)")
    ax.set_ylim(1e-3, 1e26)
    ax.legend(fontsize=15, loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    saved = plotstyle.save(fig, out_path)
    print(f"   wrote {Path(saved).relative_to(REPO_ROOT)}")
    return saved


def save_width_curve_figure(out_path, K, tau_fit, dpw_fit, w_meas_max):
    """
    Draw enumeration vs the transfer-matrix DP against width at fixed k = 3 (Nym's depth).

    Enumeration is priced by the exact work law (K W^2) and the measured tau; the DP by the
    cubic width fit, drawn solid over the measured widths and dashed beyond.
    """
    fig, ax = plt.subplots()
    k = 3
    Wg = np.logspace(np.log10(2), 3, 200)
    enum_s = 10.0 ** log10_enum_seconds(Wg, float(k), K, tau_fit)
    c0, c1, c2, c3 = dpw_fit
    dp_s = c0 + c1 * Wg + c2 * Wg ** 2 + c3 * Wg ** 3
    ax.loglog(Wg, enum_s, lw=2.4, color="#D55E00",
              label=r"exact enumeration  $\propto K\,W^{2}$")
    meas = Wg <= w_meas_max
    ax.loglog(Wg[meas], dp_s[meas], lw=2.4, color="#0072B2",
              label=r"transfer-matrix DP  $\propto L\,W^{3} G + K\,W$")
    ax.loglog(Wg[~meas], dp_s[~meas], lw=2.4, ls="--", color="#0072B2")
    ax.annotate(f"measured to $W = {w_meas_max:.0f}$,\nfitted model beyond",
                (w_meas_max * 1.6, dp_s[meas][-1] * 0.04), fontsize=15, color="#0072B2")
    for secs, txt in [(60.0, "1 minute"), (3.6e3, "1 hour"), (8.64e4, "1 day")]:
        ax.axhline(secs, color="grey", lw=1.0, ls=":")
        ax.annotate(txt, (2.2, secs * 1.7), fontsize=15, color="grey")
    t_nym = float(10.0 ** log10_enum_seconds(40.0, float(k), K, tau_fit))
    ax.plot([40.0], [t_nym], "o", ms=10, color="#000000", zorder=6)
    ax.annotate("Nym as deployed\n$W = 40$", (40.0, t_nym), textcoords="offset points",
                xytext=(-150, 14), fontsize=15, zorder=6,
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8, shrinkB=7))
    ax.set_xlabel(r"Mix nodes per layer  $W$   (at $L = 3$)")
    ax.set_ylabel(r"Running time per epoch  (s)")
    ax.set_xlim(2, 1e3)
    ax.set_ylim(1e-2, 1e6)
    ax.legend(fontsize=15, loc="upper right", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    saved = plotstyle.save(fig, out_path)
    print(f"   wrote {Path(saved).relative_to(REPO_ROOT)}")
    return saved


def main():
    """Run the validation (or redraw the figures with --plot-only); returns the exit status."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, default=T_EPOCH,
                    help="horizon for EVERY run (default = the full epoch 388,800, which is "
                         "also what the ladder is quoted at, so nothing is extrapolated in T; "
                         "lower it only to iterate quickly)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-epoch", action="store_true",
                    help="skip the one full-epoch measurement (check 4 falls back to the closed form)")
    ap.add_argument("--plot-only", action="store_true", default=PLOT_ONLY,
                    help="redraw both figures from the cached constants in results/tables/"
                         "mixnet_complexity_calibration.csv, running NO measurement (seconds "
                         "instead of ~7 min -- for iterating on the plot design). Also settable "
                         "via the PLOT_ONLY toggle at the top, for VS Code's Run button.")
    ap.add_argument("--full", dest="plot_only", action="store_false",
                    help="force the full measuring run even when PLOT_ONLY is set at the top")
    args = ap.parse_args()

    figs = REPO_ROOT / "results" / "figures" / "stage_2_figures"
    calib_path = REPO_ROOT / "results" / "tables" / CALIB_CSV

    if args.plot_only:
        cal = read_calibration(calib_path)
        print("=" * 94)
        print("MIX-NET ENUMERATION COMPLEXITY -- REDRAW ONLY (no measurement)")
        print(f"cached calibration: T = {cal['T']:,.0f}, seed {cal['seed']:.0f}, "
              f"K = {cal['K']:,.0f} rows, tau(k) = {cal['tau_a_ns']:.1f} + {cal['tau_b_ns']:.2f}k ns, "
              f"c = {cal['c_mem_bytes']:.0f} B/elem")
        print(f"measured on: {cal['machine']}")
        if cal["T"] != T_EPOCH:
            print(f"!! the cache is from a REDUCED horizon ({cal['T']:,.0f} vs the full epoch "
                  f"{T_EPOCH:,}) -- rerun without --plot-only before quoting these figures")
        print("=" * 94)
        save_ladder_figure(figs / "mixnet_complexity_ladder.png", cal["K"],
                           (cal["tau_a_ns"], cal["tau_b_ns"]))
        save_cost_curve_figure(figs / "mixnet_complexity_enum_vs_dp.png", cal["K"],
                               (cal["tau_a_ns"], cal["tau_b_ns"]),
                               (cal["dp_intercept_s"], cal["dp_slope_s"]))
        save_memory_figure(figs / "mixnet_complexity_memory.png", cal["K"], cal["c_mem_bytes"])
        if "dpw_c0_s" in cal:
            save_width_curve_figure(figs / "mixnet_complexity_enum_vs_dp_width.png", cal["K"],
                                    (cal["tau_a_ns"], cal["tau_b_ns"]),
                                    (cal["dpw_c0_s"], cal["dpw_c1_s"], cal["dpw_c2_s"],
                                     cal["dpw_c3_s"]),
                                    cal["dpw_w_max"])
        else:
            print("!! no width-scaling constants in the cache (check 8) -- rerun the full "
                  "validation once to add them; skipping the width figure")
        return 0

    print("=" * 94)
    print("MIX-NET ENUMERATION COMPLEXITY -- validation")
    print(f"T = {args.T:,}" + ("  (THE FULL EPOCH -- every run below is at it)"
                               if args.T == T_EPOCH else
                               f"  !! REDUCED (the full epoch is {T_EPOCH:,})")
          + f"   |S| = {COVER + 1}   seed = {args.seed}")
    print(f"machine: {LAPTOP_NAME}")
    print("=" * 94)

    results = [check_work_law(args.seed)]
    ok_t, tau_fit, _ = check_time_model(args.T, args.seed)
    results.append(ok_t)
    ok_m, c_mem = check_memory_model(args.T, args.seed)
    results.append(ok_m)
    ok_k, K = check_epoch_scale(args.seed, run_full=not args.skip_epoch)
    results.append(ok_k)
    ok_l, _, P_mem = check_ladder(K, tau_fit, c_mem,
                                  REPO_ROOT / "results" / "tables" / "mixnet_complexity.csv")
    results.append(ok_l)
    results.append(check_mc_sampling(args.T, args.seed))
    ok_dp, dp_fit = check_transfer_matrix_dp(args.T, args.seed, tau_fit)
    results.append(ok_dp)
    ok_wid, dpw_fit, dpw_wmax = check_dp_width_scaling(args.seed)
    results.append(ok_wid)

    write_calibration(calib_path, T=args.T, seed=args.seed, machine=LAPTOP_NAME, K=K,
                      tau_a_ns=tau_fit[0], tau_b_ns=tau_fit[1], c_mem_bytes=c_mem, P_mem=P_mem,
                      dp_intercept_s=dp_fit[0], dp_slope_s=dp_fit[1],
                      dpw_c0_s=dpw_fit[0], dpw_c1_s=dpw_fit[1], dpw_c2_s=dpw_fit[2],
                      dpw_c3_s=dpw_fit[3], dpw_w_max=dpw_wmax)
    save_ladder_figure(figs / "mixnet_complexity_ladder.png", K, tau_fit)
    save_cost_curve_figure(figs / "mixnet_complexity_enum_vs_dp.png", K, tau_fit, dp_fit)
    save_memory_figure(figs / "mixnet_complexity_memory.png", K, c_mem)
    save_width_curve_figure(figs / "mixnet_complexity_enum_vs_dp_width.png", K, tau_fit,
                            dpw_fit, dpw_wmax)

    print("\n" + "=" * 94)
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    print("=" * 94)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
