"""
Validation of slot-synchronous cover traffic (inject_dummies) on real consensus traffic.

Checks:
  1. Count and batching -- M real + T*count cover emissions, slot-sorted, `count` cover per slot.
  2. Self-exclusion -- a cover emitter is never its slot's winner; all ids lie in [0, N).
  3. Timing -- window=0 emits at the slot; window=w keeps every emission inside [0, T).
  4. Per-node p_s rule -- cover count per slot ~ Binomial(N, p_s), uniform over nodes.
  5. Trace propagation -- the tag survives the none layer and links stay certain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import (  # noqa: E402
    DEFAULT_F,
    sample_relative_stakes,
    simulate_events,
)
from anonymity import DummyParams, inject_dummies, passthrough  # noqa: E402


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _rank_corr(x, y):
    """Spearman rank correlation without scipy; ties are broken arbitrarily."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _winner_of_slot(slots, nodes, T):
    """Per-slot winner (last winner in the rare multi-winner slot), -1 where empty."""
    w = np.full(T, -1, dtype=np.int64)
    w[slots] = nodes
    return w


def main():
    """Run the inject_dummies validation checks; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--T", type=int, default=40_000)
    ap.add_argument("--shape", type=float, default=1.33, help="Pareto stake shape k")
    ap.add_argument("--f", type=float, default=DEFAULT_F)
    ap.add_argument("--count", type=int, default=3, help="cover senders per slot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    alpha = sample_relative_stakes(args.N, args.shape, rng=rng)
    slots, nodes = simulate_events(alpha, f=args.f, T=args.T, rng=rng)
    M = slots.size
    C = args.count
    T = args.T
    print(f"consensus: N={args.N}, T={T}, k={args.shape} -> {M} real winners; "
          f"cover/slot={C}\n")

    ok = True

    params = DummyParams(count=C, window=0)
    s, n, is_dummy, grp = inject_dummies(slots, nodes, args.N, params=params, T=T, rng=rng)
    n_real = int((~is_dummy).sum())
    n_cover = int(is_dummy.sum())
    sorted_ok = bool(np.all(np.diff(s) >= 0))
    cover_per_slot = np.bincount(grp[is_dummy], minlength=T)
    every_slot_covered = bool(cover_per_slot.size == T and np.all(cover_per_slot == C))
    real_by_slot = np.array_equal(np.sort(grp[~is_dummy]), np.sort(slots))
    ok &= _check("count & batching (count cover/slot, real joins its slot)",
                 s.size == M + T * C and n_real == M and n_cover == T * C
                 and sorted_ok and every_slot_covered and real_by_slot,
                 f"{M} real + {n_cover} cover = {s.size}; {T} slots x {C} cover")

    wos = _winner_of_slot(slots, nodes, T)
    excl = not np.any(n[is_dummy] == wos[grp[is_dummy]])
    ids_ok = bool(n.min() >= 0 and n.max() < args.N)
    ok &= _check("self-exclusion + valid ids (cover never its slot's winner)",
                 excl and ids_ok)

    w0_ok = np.array_equal(s[is_dummy], grp[is_dummy])
    w = 5
    sp = DummyParams(count=C, window=w)
    sw, _, _, _ = inject_dummies(slots, nodes, args.N, params=sp, T=T, rng=rng)
    band_ok = bool(sw.min() >= 0 and sw.max() < T)
    ok &= _check("timing (window=0 at slot; window=w within [0,T))",
                 w0_ok and band_ok, f"window={w}")

    p_s = 5.0 / args.N
    _, npn, dpn, gpn = inject_dummies(slots, nodes, args.N,
                                      params=DummyParams(p_s=p_s), T=T, rng=rng)
    sizes = np.bincount(gpn[dpn], minlength=T)
    mean_ok = abs(sizes.mean() - p_s * args.N) < 0.2
    var_ok = abs(sizes.var() - args.N * p_s * (1 - p_s)) < 0.3
    random_ok = bool(sizes.min() < sizes.max())
    freq = np.bincount(npn[dpn], minlength=args.N).astype(float)
    corr = _rank_corr(alpha, freq)
    ok &= _check("per-node p_s rule (Binomial(N,p_s): mean p_s*N, random size, uniform)",
                 mean_ok and var_ok and random_ok and abs(corr) < 0.1,
                 f"cover/slot mean={sizes.mean():.3f} (p_s*N={p_s * args.N:.1f}), "
                 f"var={sizes.var():.3f} (np(1-p)={args.N * p_s * (1 - p_s):.3f}), "
                 f"corr(alpha,freq)={corr:.3f}")

    trace = passthrough(s, n, is_dummy)
    n_cover_rows = int(trace.is_dummy.sum())
    n_real_rows = int((~trace.is_dummy).sum())
    none_certain = np.array_equal(trace.obs_node[trace.is_exit],
                                  trace.true_source[trace.is_exit])
    ok &= _check("Trace propagation (tag rides through none; links still certain)",
                 len(trace) == 2 * (M + T * C) and n_cover_rows == 2 * T * C
                 and n_real_rows == 2 * M and none_certain,
                 f"{n_cover_rows} cover + {n_real_rows} real rows")

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
