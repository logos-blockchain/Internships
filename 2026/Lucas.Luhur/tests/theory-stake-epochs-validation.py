"""
Validation of theory.stake_epochs against the simulation pipeline and closed-form anchors.

The module compounds the per-epoch top-1 probability p(alpha) into
C_quenched(E) = E_alpha[1 - (1 - p)^E], the probability the whale is named within E epochs;
conditional on alpha the per-epoch hits are iid. Checks:
  1. Closed form -- a two-point mixture is reproduced exactly; E = 1 returns E[p].
  2. Jensen -- C_quenched <= C_naive at every E, with equality at E = 1 and degenerate draws.
  3. Subsampling estimator -- kernel equals hypergeom.pmf(0, m, h, E) and is unbiased for (1-p)^E.
  4. Engine -- chains x real epochs through the full pipeline, paired with the closed form.
  5. Limits -- monotone in E, -> 1 as E -> inf, and epochs_to_level inverts the single-p form.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import DEFAULT_F, sample_relative_stakes, simulate_events   # noqa: E402
from anonymity import DummyParams, inject_dummies, passthrough             # noqa: E402
from adversary import SetStakeInferenceParams, run_set_stake_inference     # noqa: E402
from metrics.stake_privacy import stake_top1_hit                           # noqa: E402
from theory.stake_top1 import top1_probability                             # noqa: E402
from theory.stake_epochs import (                                          # noqa: E402
    cumulative_top1, epochs_to_level, naive_cumulative, subsampled_cumulative,
)


class _Ctx:
    """Minimal stand-in for the pipeline ScoreContext read by the stake-privacy measures."""

    def __init__(self, alpha):
        self.alpha = alpha


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def check_closed_form():
    """1. Two-point mixture, hand-computed; E = 1 returns the mean."""
    print("\n1. CLOSED FORM -- two-point mixture and the E = 1 identity")
    p = np.array([0.3, 0.01])
    E = np.array([1, 2, 7, 50])
    want = 1.0 - 0.5 * ((1 - 0.3) ** E + (1 - 0.01) ** E)
    got = cumulative_top1(p, E)
    err = float(np.abs(got - want).max())
    ok = _check("mixture exact", err < 1e-14, f"max err = {err:.2e}")
    e1 = float(cumulative_top1(p, np.array([1]))[0])
    ok &= _check("E = 1 returns E[p]", abs(e1 - p.mean()) < 1e-15, f"{e1} vs {p.mean()}")
    return ok


def check_jensen(seed):
    """2. Quenched <= naive everywhere; equality at E = 1 and for degenerate draws."""
    print("\n2. JENSEN -- the quenched curve sits below the naive one")
    rng = np.random.default_rng(seed)
    p = rng.beta(0.2, 20.0, size=500)
    E = np.unique(np.round(np.logspace(0, 5, 60)).astype(int))
    cq, cn = cumulative_top1(p, E), naive_cumulative(p.mean(), E)
    gap = cn - cq
    ok = _check("C_quenched <= C_naive at every E", bool(np.all(gap >= -1e-12)),
                f"min gap = {gap.min():.2e}, max gap = {gap.max():.4f}")
    ok &= _check("equality at E = 1", abs(gap[0]) < 1e-14, f"gap(1) = {gap[0]:.2e}")
    flat = np.full(50, 0.037)
    gd = np.abs(naive_cumulative(0.037, E) - cumulative_top1(flat, E)).max()
    ok &= _check("degenerate draws collapse the gap", gd < 1e-12, f"max |gap| = {gd:.2e}")
    return ok


def check_subsampling(seed):
    """3. The subsampling estimator: exact vs hypergeom, unbiased for (1-p)^E."""
    print("\n3. SUBSAMPLING -- two routes agree exactly, and the estimator is unbiased")
    m = 25
    ok = True
    worst = 0.0
    for h in range(m + 1):
        E = np.arange(1, m + 1)
        got = 1.0 - subsampled_cumulative(np.array([h]), m, E)
        want = hypergeom.pmf(0, m, h, E)
        worst = max(worst, float(np.abs(got - want).max()))
    ok &= _check("kernel == hypergeom.pmf(0) on the full (h, E) grid", worst < 1e-12,
                 f"max err = {worst:.2e}")
    from scipy.stats import binom
    for p in (0.02, 0.3):
        hs = np.arange(m + 1)
        w = binom.pmf(hs, m, p)
        E = np.array([1, 5, 12, 25])
        K = np.array([1.0 - subsampled_cumulative(np.array([h]), m, E) for h in hs])
        est = w @ K
        want = (1.0 - p) ** E
        err = float(np.abs(est - want).max())
        ok &= _check(f"exactly unbiased at p = {p}", err < 1e-12, f"max err = {err:.2e}")
    Ef = np.arange(1, m + 1)
    mix = 0.5 * (subsampled_cumulative(np.array([13.0]), m, Ef)
                 + subsampled_cumulative(np.array([14.0]), m, Ef))
    got = subsampled_cumulative(np.array([13.5]), m, Ef)
    err = float(np.abs(got - mix).max())
    ok &= _check("tie credit = mixture of adjacent integer kernels", err < 1e-12,
                 f"max err = {err:.2e}")
    return ok


def check_vs_engine(N, T, f, p_s, shape, chains, epochs, seed):
    """4. M real epochs per quenched chain vs the compounded closed form."""
    print(f"\n4. ENGINE -- {chains} chains x {epochs} epochs at N={N}, T={T}, p_s={p_s} "
          f"(paired on the same stake vectors)")
    rng_tie = np.random.default_rng([seed, 999])
    hit_mat = np.zeros((chains, epochs))
    p_theory = np.zeros(chains)
    for r in range(chains):
        rng = np.random.default_rng([seed, r])
        alpha = sample_relative_stakes(N, shape, rng=rng)
        p_theory[r] = top1_probability(alpha, p_s, f=f, T=T)
        for m in range(epochs):
            erng = np.random.default_rng([seed, r, m])
            slots, nodes = simulate_events(alpha, f=f, T=T, rng=erng)
            s, n, d, g = inject_dummies(slots, nodes, N, params=DummyParams(p_s=p_s),
                                        T=T, rng=erng)
            trace = passthrough(s, n, d, g)
            ahat = run_set_stake_inference(
                trace, params=SetStakeInferenceParams(f=f, p_s=p_s, T=T, N=N))
            score = stake_top1_hit(ahat, _Ctx(alpha))
            hit_mat[r, m] = float(rng_tie.random() < score)

    E = np.arange(1, epochs + 1)
    c_theory = cumulative_top1(p_theory, E)
    ok = True
    ever = np.maximum.accumulate(hit_mat, axis=1)
    for e in (1, max(1, epochs // 2), epochs):
        emp = float(ever[:, e - 1].mean())
        th = float(c_theory[e - 1])
        c_r = -np.expm1(e * np.log1p(-p_theory))
        sem = float(np.sqrt((c_r * (1 - c_r)).sum()) / chains)
        z = abs(emp - th) / sem if sem > 0 else 0.0
        ok &= _check(f"any-hit-yet |z| <= 3.5 at E = {e}", z <= 3.5,
                     f"engine {emp:.4f} vs theory {th:.4f}, z = {z:.2f}")
    est = subsampled_cumulative(hit_mat.sum(axis=1), epochs, E)
    d = est - c_theory
    ok &= _check("subsampled leg tracks the closed form",
                 bool(np.all(np.abs(d) <= 4.0 * np.maximum(
                     np.sqrt(c_theory * (1 - c_theory) / chains), 1e-9))),
                 f"max |diff| = {np.abs(d).max():.4f}")
    print(f"       E[p]: theory {p_theory.mean():.5f}, engine per-epoch {hit_mat.mean():.5f}")
    return ok


def check_limits(seed):
    """5. Monotonicity, the E -> inf limit, and the level-crossing inverse."""
    print("\n5. LIMITS -- monotone, saturating, and epochs_to_level inverts")
    rng = np.random.default_rng(seed)
    p = rng.beta(0.2, 20.0, size=200) + 1e-6
    E = np.unique(np.round(np.logspace(0, 7, 120)).astype(int))
    C = cumulative_top1(p, E)
    ok = _check("monotone non-decreasing in E", bool(np.all(np.diff(C) >= -1e-15)))
    ok &= _check("-> 1 as E -> inf (all p > 0)", C[-1] > 0.999, f"C(1e7) = {C[-1]:.5f}")
    p0, level = 0.0098, 0.5
    Cs = cumulative_top1(np.array([p0]), E)
    want = np.log1p(-level) / np.log1p(-p0)
    got = epochs_to_level(E, Cs, level)
    ok &= _check("epochs_to_level inverts the single-p form", abs(got / want - 1) < 0.02,
                 f"{got:.1f} vs {want:.1f} epochs")
    ok &= _check("never-crossing returns inf",
                 np.isinf(epochs_to_level(E[:3], Cs[:3], 0.99)))
    return ok


def main():
    """Run the stake_epochs validation checks; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--T", type=int, default=40_000, help="reduced horizon (tests override T down)")
    ap.add_argument("--p-s", type=float, default=0.05, dest="p_s")
    ap.add_argument("--shape", type=float, default=2.0)
    ap.add_argument("--chains", type=int, default=30, help="quenched chains for the engine check")
    ap.add_argument("--epochs", type=int, default=10, help="real epochs per chain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("theory.stake_epochs validation -- multi-epoch compounding of the stake clause")
    print("=" * 78)
    ok = check_closed_form()
    ok &= check_jensen(args.seed)
    ok &= check_subsampling(args.seed)
    ok &= check_vs_engine(args.N, args.T, DEFAULT_F, args.p_s, args.shape,
                          args.chains, args.epochs, args.seed)
    ok &= check_limits(args.seed)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
