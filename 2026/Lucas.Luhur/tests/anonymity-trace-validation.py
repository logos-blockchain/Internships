"""
Validation of the Trace schema and the `none` passthrough layer on real consensus traffic.

Checks:
  1. Schema well-formedness -- M broadcasts -> 2M rows, M ENTRY + M EXIT, correct dtypes.
  2. Passthrough fidelity -- entry = exit = true_source, obs_time = slot, no dummies.
  3. Baseline equivalence -- every link is certain, so all M anonymity sets equal 1.
  4. Registry contract -- LAYERS["none"] is the passthrough with the uniform signature.
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
from anonymity import ENTRY, EXIT, LAYERS, Trace, passthrough  # noqa: E402


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def main():
    """Run the Trace / passthrough validation checks; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000, help="number of nodes")
    ap.add_argument("--T", type=int, default=20_000, help="observation window (slots)")
    ap.add_argument("--shape", type=float, default=1.33, help="Pareto stake shape k (matches the other anonymity-* tests)")
    ap.add_argument("--f", type=float, default=DEFAULT_F, help="active-slots coefficient")
    ap.add_argument("--seed", type=int, default=0, help="master seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    alpha = sample_relative_stakes(args.N, args.shape, rng=rng)
    slots, nodes = simulate_events(alpha, f=args.f, T=args.T, rng=rng)
    m = slots.size
    print(f"consensus: N={args.N}, T={args.T}, k={args.shape} -> {m} broadcast events\n")

    trace = passthrough(slots, nodes)
    ok = True

    cols = (trace.broadcast_id, trace.true_source, trace.obs_node,
            trace.obs_time, trace.kind, trace.is_dummy)
    equal_len = all(c.size == 2 * m for c in cols)
    dtypes_ok = (
        trace.broadcast_id.dtype == np.int64 and trace.true_source.dtype == np.int64
        and trace.obs_node.dtype == np.int64 and trace.obs_time.dtype == np.float64
        and trace.kind.dtype == np.int8 and trace.is_dummy.dtype == bool
    )
    n_entry = int(trace.is_entry.sum())
    n_exit = int(trace.is_exit.sum())
    bid_entry = np.sort(trace.broadcast_id[trace.is_entry])
    bid_exit = np.sort(trace.broadcast_id[trace.is_exit])
    pairing = (np.array_equal(bid_entry, np.arange(m))
               and np.array_equal(bid_exit, np.arange(m)))
    ok &= _check("schema well-formedness",
                 equal_len and dtypes_ok and n_entry == m and n_exit == m and pairing,
                 f"2M={2 * m} rows, {n_entry} ENTRY / {n_exit} EXIT")

    en, ex = trace.is_entry, trace.is_exit
    en_order = np.argsort(trace.broadcast_id[en])
    ex_order = np.argsort(trace.broadcast_id[ex])
    entry_node = trace.obs_node[en][en_order]
    exit_node = trace.obs_node[ex][ex_order]
    src = trace.true_source[ex][ex_order]
    entry_time = trace.obs_time[en][en_order]
    exit_time = trace.obs_time[ex][ex_order]
    fidelity = (
        np.array_equal(entry_node, src) and np.array_equal(exit_node, src)
        and np.array_equal(entry_time, slots.astype(float))
        and np.array_equal(exit_time, slots.astype(float))
        and not trace.is_dummy.any()
    )
    ok &= _check("passthrough fidelity (entry = exit = source, no delay, no dummies)",
                 fidelity)

    set_sizes_from_trace = np.where(exit_node == src, 1, 0)
    baseline = np.ones(m, dtype=np.int64)
    equiv = (np.array_equal(set_sizes_from_trace, np.ones(m, dtype=np.int64))
             and np.array_equal(set_sizes_from_trace, baseline))
    ok &= _check("baseline equivalence (none reproduces all-ones through the Trace)",
                 equiv, f"all {m} anonymity sets = 1")

    layer = LAYERS.get("none")
    reg_trace = layer(slots, nodes, params=None, latency_oracle=None, rng=rng)
    registry_ok = (
        layer is passthrough and callable(layer)
        and isinstance(reg_trace, Trace) and len(reg_trace) == 2 * m
    )
    ok &= _check("registry contract (LAYERS['none'] -> uniform signature -> Trace)",
                 registry_ok)

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
