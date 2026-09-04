"""
Validation of the sender-set stake-inference attack (run_set_stake_inference).

Pr(i in S_t) = p_s + (1 - p_s) phi(alpha_i), so inverting the participation frequency gives
alpha_hat_i = log((1 - q_hat_i) / (1 - p_s)) / log(1 - f) -> alpha_i as T -> inf. Checks:
  1. Consistency -- alpha_hat tracks alpha on the largest stakeholders.
  2. Layer-invariance -- alpha_hat is identical under `none` and `single_path_mix`.
  3. Top-k identification -- the top staker and most of the true top-5 are recovered.
  4. Privacy dial -- small p_s leaks the whale, large p_s protects it.
  5. Registry contract -- ATTACKS['set_stake_inference'] produces SCALAR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import DEFAULT_F, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import LAYERS, DummyParams, SinglePathMixParams, inject_dummies, passthrough  # noqa: E402
from adversary import SetStakeInferenceParams, run_set_stake_inference, ATTACKS  # noqa: E402
from pipeline_contract import SCALAR  # noqa: E402


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _rank_corr(x, y):
    """Spearman rank correlation without scipy; ties are broken arbitrarily."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _infer(alpha, N, T, f, p_s, seed, layer="single_path_mix"):
    """Build the p_s sender set, route it through a layer and return the estimated alpha_hat."""
    rng = np.random.default_rng(seed)
    slots, nodes = simulate_events(alpha, f=f, T=T, rng=rng)
    s, n, d, g = inject_dummies(slots, nodes, N, params=DummyParams(p_s=p_s), T=T, rng=rng)
    if layer == "none":
        trace = passthrough(s, n, d, g)
    else:
        trace = LAYERS["single_path_mix"](s, n, d, g, params=SinglePathMixParams(hops=3, mix_scale=1.0, n_nodes=N),
                                   latency_oracle=None, rng=rng)
    sp = SetStakeInferenceParams(f=f, p_s=p_s, T=T, N=N)
    return run_set_stake_inference(trace, params=sp)


def main():
    """Run the set stake-inference validation checks; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--T", type=int, default=100_000)
    ap.add_argument("--shape", type=float, default=1.33, help="Pareto stake shape k")
    ap.add_argument("--f", type=float, default=DEFAULT_F)
    ap.add_argument("--p_s", type=float, default=0.002,
                    help="per-node cover probability (small -> win signal clears the cover noise)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N, T, f, p_s = args.N, args.T, args.f, args.p_s
    alpha = sample_relative_stakes(N, args.shape, rng=np.random.default_rng(args.seed))
    true_top = int(np.argmax(alpha))
    print(f"consensus: N={N}, T={T}, k={args.shape}, p_s={p_s}; "
          f"top staker = node {true_top} (alpha={alpha[true_top]:.4f})\n")

    ok = True

    ah = _infer(alpha, N, T, f, p_s, args.seed, layer="single_path_mix")
    top10 = np.argsort(alpha)[-10:]
    corr_top = _rank_corr(alpha[top10], ah[top10])
    whale_relerr = abs(ah[true_top] - alpha[true_top]) / alpha[true_top]
    ok &= _check("consistency (alpha_hat -> alpha on the whales; top-staker estimate close)",
                 corr_top > 0.5 and whale_relerr < 0.15,
                 f"rank-corr(top10)={corr_top:.2f}; top alpha_hat={ah[true_top]:.4f} "
                 f"vs alpha={alpha[true_top]:.4f} (rel err {whale_relerr:.1%})")

    ah_none = _infer(alpha, N, T, f, p_s, args.seed, layer="none")
    ah_tor = _infer(alpha, N, T, f, p_s, args.seed, layer="single_path_mix")
    ok &= _check("layer-invariance (alpha_hat identical: none == single_path_mix; mixing buys no stake privacy)",
                 np.array_equal(ah_none, ah_tor),
                 f"max|diff|={float(np.max(np.abs(ah_none - ah_tor))):.2e}")

    guessed_top = int(np.argmax(ah))
    top5 = np.argsort(alpha)[-5:]
    top5_overlap = len(set(np.argsort(ah)[-5:].tolist()) & set(top5.tolist()))
    ok &= _check("whale / top-k identification (top staker + top-5 overlap)",
                 guessed_top == true_top and top5_overlap >= 3,
                 f"top guess node {guessed_top} (true {true_top}); top-5 overlap {top5_overlap}/5")

    ah_hi = _infer(alpha, N, T, f, p_s=0.5, seed=args.seed)
    corr_lo = _rank_corr(alpha[top10], ah[top10])
    corr_hi = _rank_corr(alpha[top10], ah_hi[top10])
    leak = guessed_top == true_top and corr_lo > 0.5
    protect = corr_hi < corr_lo - 0.3
    ok &= _check("privacy dial (p_s small leaks the whale, p_s large protects it)",
                 leak and protect,
                 f"whale rank-corr(top10): p_s={p_s} -> {corr_lo:.2f} (leak), p_s=0.5 -> {corr_hi:.2f} (protect)")

    spec = ATTACKS.get("set_stake_inference")
    reg_ok = spec is not None and spec.produces == SCALAR and spec.run is run_set_stake_inference
    ok &= _check("registry contract (ATTACKS['set_stake_inference'] -> SCALAR)", reg_ok)

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
