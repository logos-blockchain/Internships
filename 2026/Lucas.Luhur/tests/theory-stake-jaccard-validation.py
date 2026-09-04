"""
Validation of theory.stake_jaccard against the simulation pipeline and closed-form anchors.

The measure stake_top_jaccard is the Jaccard overlap between the m largest estimated and true
stakes; since alpha_hat is monotone in the count c_i ~ Binomial(T, q_i), E[J] follows from the
threshold-count law (see src/theory/stake_jaccard.py). Checks:
  1. Threshold law -- total mass 1, strided sum equals stride 1, tie-cap overflow ~0.
  2. Count-law Monte Carlo -- exact counts through the real estimator and measure vs theory.
  3. Engine -- theory vs the full pipeline, paired on the same stake vectors.
  4. Limits -- monotone in p_s and T, above the random-set reference, bounded in [0, 1].
  5. Committed grid -- the closed form at N = 1000, T = 388,800 over the swept cover rates.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import DEFAULT_F, DEFAULT_T, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, inject_dummies, passthrough            # noqa: E402
from adversary import SetStakeInferenceParams, run_set_stake_inference    # noqa: E402
from metrics.stake_privacy import stake_top_jaccard                       # noqa: E402
from theory.stake_jaccard import (                                        # noqa: E402
    jaccard_probability, random_set_jaccard, top_set_size,
)
from theory.stake_top1 import participation_prob                          # noqa: E402


class _Ctx:
    """Minimal stand-in for the pipeline ScoreContext read by the stake-privacy measures."""

    def __init__(self, alpha, top_frac):
        self.alpha = alpha
        self.top_frac = top_frac


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _estimate(counts, p_s, f, T):
    """Apply the sender-set stake estimator to sampled participation counts."""
    q_hat = np.clip(counts / T, 0.0, 1.0 - 1e-12)
    return np.maximum(np.log((1.0 - q_hat) / (1.0 - p_s)) / np.log(1.0 - f), 0.0)


def check_threshold_law(N, T, f, top_frac, shape, seed):
    """1. The threshold event has total mass 1, the strided sum is exact, and overflow is ~0."""
    print("\n1. THRESHOLD LAW -- total mass 1, stride-1 agreement, tie-cap overflow ~0")
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, shape, rng=rng)
    ok = True
    for p_s in (0.01, 0.1, 0.5):
        d = jaccard_probability(alpha, p_s, top_frac=top_frac, f=f, T=T, details=True)
        ok &= _check(f"threshold mass = 1 at p_s={p_s}", abs(d["mass"] - 1.0) < 2e-3,
                     f"mass {d['mass']:.6f}, stride {d['stride']}, window {d['window']}")
        ok &= _check(f"tie-cap overflow ~0 at p_s={p_s}", d["overflow"] < 1e-6,
                     f"overflow {d['overflow']:.2e}")
        fine = jaccard_probability(alpha, p_s, top_frac=top_frac, f=f, T=T, grid_points=10**9)
        ok &= _check(f"strided == stride-1 at p_s={p_s}", abs(fine - d["value"]) < 2e-4,
                     f"{d['value']:.6f} vs {fine:.6f}")
    return ok


def check_count_law_mc(N, T, f, top_frac, shape, seed, reps=4000):
    """2. Exactly distributed counts through the real estimator and measure vs the closed form."""
    print("\n2. COUNT-LAW MONTE CARLO -- the real measure on exactly-distributed counts")
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, shape, rng=rng)
    ctx = _Ctx(alpha, top_frac)
    ok = True
    for p_s in (0.02, 0.1, 0.5):
        q = participation_prob(alpha, p_s, f)
        counts = rng.binomial(T, q, size=(reps, N))
        vals = np.array([stake_top_jaccard(_estimate(c, p_s, f, T), ctx) for c in counts])
        theo = jaccard_probability(alpha, p_s, top_frac=top_frac, f=f, T=T)
        sem = vals.std(ddof=1) / np.sqrt(reps)
        z = (vals.mean() - theo) / sem
        ok &= _check(f"|z| <= 3 at p_s={p_s}", abs(z) <= 3.0,
                     f"MC {vals.mean():.5f} +/- {sem:.5f} vs theory {theo:.5f}, z = {z:+.2f}")
    return ok


def check_vs_engine(N, T, f, top_frac, shape, reps, seed):
    """3. Theory vs the full pipeline, paired on the same stake vectors."""
    print("\n3. ENGINE -- closed form vs the real pipeline (paired on the same chains)")
    ok = True
    for p_s in (0.02, 0.1):
        diffs, sims, theos = [], [], []
        for r in range(reps):
            rng = np.random.default_rng([seed, r])
            alpha = sample_relative_stakes(N, shape, rng=rng)
            slots, nodes = simulate_events(alpha, f=f, T=T, rng=rng)
            s, n, d, g = inject_dummies(slots, nodes, N, params=DummyParams(p_s=p_s), T=T, rng=rng)
            trace = passthrough(s, n, d, g)
            ahat = run_set_stake_inference(
                trace, params=SetStakeInferenceParams(f=f, p_s=p_s, T=T, N=N))
            sim = stake_top_jaccard(ahat, _Ctx(alpha, top_frac))
            theo = jaccard_probability(alpha, p_s, top_frac=top_frac, f=f, T=T)
            sims.append(sim)
            theos.append(theo)
            diffs.append(sim - theo)
        dd = np.asarray(diffs)
        sem = float(dd.std(ddof=1) / np.sqrt(dd.size)) if dd.size > 1 else 0.0
        z = abs(float(dd.mean()) / sem) if sem > 0 else 0.0
        ok &= _check(
            f"paired |z| <= 3 at p_s={p_s}", z <= 3.0,
            f"sim {np.mean(sims):.5f} vs theory {np.mean(theos):.5f}, "
            f"diff {dd.mean():+.5f} +/- {sem:.5f}, z = {z:.2f}")
    return ok


def check_limits(N, T, f, top_frac, shape, seed):
    """4. The closed form's own monotonicities and bounds."""
    print("\n4. LIMITS -- monotone in p_s and T, above the random-set reference, bounded")
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, shape, rng=rng)
    m = top_set_size(N, top_frac)
    ok = True
    mono = [jaccard_probability(alpha, p, top_frac=top_frac, f=f, T=T)
            for p in (0.01, 0.05, 0.1, 0.3, 0.7)]
    ok &= _check("monotone DECREASING in p_s (cover protects)", all(np.diff(mono) < 0),
                 " -> ".join(f"{v:.5f}" for v in mono))
    grow = [jaccard_probability(alpha, 0.1, top_frac=top_frac, f=f, T=t)
            for t in (T, 10 * T, 100 * T)]
    ok &= _check("monotone INCREASING in T (alpha_hat is consistent)", all(np.diff(grow) > 0),
                 " -> ".join(f"{v:.5f}" for v in grow))
    ref = random_set_jaccard(N, m)
    heavy = jaccard_probability(alpha, 0.9, top_frac=top_frac, f=f, T=T)
    ok &= _check("heavy cover stays ABOVE the random-set reference (never below chance)",
                 heavy >= ref * 0.98, f"J(p_s=0.9) = {heavy:.5f} vs random-set {ref:.5f}")
    vals = [jaccard_probability(sample_relative_stakes(N, k, rng=rng), 0.05,
                                top_frac=top_frac, f=f, T=T) for k in (1.2, 2.0, 5.0)]
    ok &= _check("bounded in [0, 1] across shapes", all(0.0 <= v <= 1.0 for v in vals),
                 ", ".join(f"{v:.5f}" for v in vals))
    return ok


def check_committed_grid(seed):
    """5. The committed regime: N = 1000, the full epoch, the swept cover rates."""
    print("\n5. THE COMMITTED GRID -- N = 1000, T = 388,800, k = 3 and 4/3, p_s in the sweep")
    ok = True
    for k in (3.0, 4.0 / 3.0):
        alpha = sample_relative_stakes(1000, k, rng=np.random.default_rng(seed))
        for p_s in (0.02, 0.1, 0.5, 0.9):
            t0 = time.perf_counter()
            d = jaccard_probability(alpha, p_s, f=DEFAULT_F, T=DEFAULT_T, details=True)
            ok &= _check(f"k={k:.3g} p_s={p_s}: mass 1, overflow < 1e-9",
                         abs(d["mass"] - 1.0) < 2e-3 and d["overflow"] < 1e-9,
                         f"J = {d['value']:.5f}, mass {d['mass']:.6f}, overflow "
                         f"{d['overflow']:.1e}, {time.perf_counter() - t0:.2f}s")
    return ok


def main():
    """Run the stake_jaccard validation checks; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=500, help="nodes")
    ap.add_argument("--T", type=int, default=40_000,
                    help="slots (overridden DOWN from the epoch for speed -- tests may, "
                         "experiments may not)")
    ap.add_argument("--f", type=float, default=DEFAULT_F)
    ap.add_argument("--top-frac", type=float, default=0.01, dest="top_frac",
                    help="the top fraction x of the measure")
    ap.add_argument("--shape", type=float, default=2.0,
                    help="Pareto k (2.0 = the tamer validation default, per the repo convention)")
    ap.add_argument("--reps", type=int, default=12, help="paired chains for the engine check")
    ap.add_argument("--mc-reps", type=int, default=4000, help="count-law Monte Carlo draws")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"theory.stake_jaccard validation -- N={a.N} T={a.T} f={a.f:.5f} "
          f"top_frac={a.top_frac} shape={a.shape} reps={a.reps} seed={a.seed}")

    ok = True
    ok &= check_threshold_law(a.N, a.T, a.f, a.top_frac, a.shape, a.seed)
    ok &= check_count_law_mc(a.N, a.T, a.f, a.top_frac, a.shape, a.seed, a.mc_reps)
    ok &= check_vs_engine(a.N, a.T, a.f, a.top_frac, a.shape, a.reps, a.seed)
    ok &= check_limits(a.N, a.T, a.f, a.top_frac, a.shape, a.seed)
    ok &= check_committed_grid(a.seed)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
