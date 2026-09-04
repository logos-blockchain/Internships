"""
Validation of the Bayesian sender-attribution attack and its three unlinkability measures.

The attack forms a posterior over the candidate set S_t from the residuals z_i = y - mu(i, r)
scored by f_Z. Checks:
  1. Reference reproduction -- the posterior reproduces the reference implementation exactly.
  2. eta = 0 (homogeneous links) -- uniform posterior; measures sit at their null/max values.
  3. eta large -- de-anonymisation: deanon_top1 -> 1, posterior_entropy -> 0.
  4. Monotonicity -- deanon_top1 rises and posterior_entropy falls with eta.
  5. Firewall -- scrambling true_source / is_dummy leaves the posterior unchanged.
  6. Registry -- the attack produces POSTERIOR and validate_pairing enforces the pairing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import DEFAULT_F, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, SinglePathMixParams, inject_dummies, make_trace  # noqa: E402
from anonymity.single_path_mix import apply as spm, random_delay_pdf  # noqa: E402
from network import LatencyProfileParams, sample_latency_profile  # noqa: E402
from adversary import run_bayes_attribution, BayesAttributionParams, ATTACKS  # noqa: E402
from metrics import deanon_top1, mean_true_posterior, posterior_entropy, MEASURES  # noqa: E402
from pipeline_contract import POSTERIOR, SCALAR, validate_pairing  # noqa: E402


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _attack(N, T, k, count, latency_params, seed, sender_scale=0.25, mix_scale=1.0):
    """Run consensus + cover + single_path_mix with a latency profile; return (trace, guess)."""
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=count), T=T, rng=rng)
    prof = sample_latency_profile(N, k, 0.3, latency_params, rng=np.random.default_rng(seed + 1))
    tr = spm(s, n, isd, g, params=SinglePathMixParams(k, mix_scale, N, sender_scale=sender_scale),
             latency_oracle=prof, rng=rng)
    guess = run_bayes_attribution(
        tr, params=BayesAttributionParams(k, mix_scale, sender_scale, False, prof))
    return tr, guess


def check_reference():
    """1. Reproduce the reference implementation's experiment exactly with our f_Z."""
    print("== reference reproduction ==")
    mrng = np.random.default_rng(7)
    cand = np.arange(20)
    d_sender = mrng.uniform(2, 20, size=100)
    d_receiver = mrng.uniform(1, 8, size=10)
    d_mix, k, lam_s, lam_m = 12.0, 3, 0.8, 1.7
    mu = lambda c, r: d_sender[c] + d_mix + d_receiver[r]

    rng = np.random.default_rng(17)
    correct = mtp = ment = 0.0
    for _ in range(5000):
        true_sender = int(rng.choice(cand))
        receiver = int(rng.integers(len(d_receiver)))
        y = mu(true_sender, receiver) + rng.exponential(1 / lam_s) + rng.gamma(k, 1 / lam_m)
        post = (lambda L: L / L.sum())(random_delay_pdf(y - mu(cand, receiver), k, 1 / lam_s, 1 / lam_m))
        correct += int(cand[np.argmax(post)] == true_sender)
        mtp += post[true_sender]
        ment += -np.sum(post * np.log2(np.maximum(post, 1e-300)))
    top1, mtp, ment = correct / 5000, mtp / 5000, ment / 5000
    ok = (abs(top1 - 0.2852) < 1e-4 and abs(mtp - 0.22188855728580298) < 1e-6
          and abs(ment - 2.545664650912402) < 1e-6)
    return _check("attack posterior == the reference implementation (top1 / true-post / entropy)", ok,
                  f"top1={top1:.4f} (ref 0.2852); true-post={mtp:.6f}; H={ment:.6f}")


def check_regimes(args):
    """2-4. eta = 0 null/max; eta large de-anonymises; monotonicity in eta."""
    print("\n== eta regimes (the SNR sweep) ==")
    N, T, k = args.N, args.T, args.hops
    ok = True

    tr0, g0 = _attack(N, T, k, args.count, LatencyProfileParams(), args.seed)
    sizes = np.diff(g0.start).astype(float)
    null = float(np.mean(1.0 / sizes))
    maxH = float(np.mean(np.log2(sizes)))
    t1_0 = deanon_top1(g0, tr0); mp_0 = mean_true_posterior(g0, tr0); H_0 = posterior_entropy(g0, tr0)
    ok &= _check("eta=0 (homogeneous): deanon_top1 = mean(1/|S|), true-post = mean(1/|S|), H = mean(log2|S|)",
                 abs(t1_0 - null) < 1e-9 and abs(mp_0 - null) < 1e-9 and abs(H_0 - maxH) < 1e-9,
                 f"top1={t1_0:.4f}~{null:.4f}; true-post={mp_0:.4f}~{null:.4f}; H={H_0:.4f}~{maxH:.4f}")

    trL, gL = _attack(N, T, k, args.count, LatencyProfileParams(sender_low=0.0, sender_high=1000.0), args.seed)
    t1_L, H_L = deanon_top1(gL, trL), posterior_entropy(gL, trL)
    ok &= _check("eta large: deanon_top1 -> 1 (>0.9), posterior_entropy -> 0 (<0.5)",
                 t1_L > 0.9 and H_L < 0.5, f"top1={t1_L:.4f}, H={H_L:.4f} (null {null:.3f}, max {maxH:.3f})")

    highs = [0.0, 2.0, 8.0, 40.0]
    t1s, Hs = [], []
    for hi in highs:
        lp = LatencyProfileParams() if hi == 0.0 else LatencyProfileParams(sender_low=0.0, sender_high=hi)
        tr, gg = _attack(N, T, k, args.count, lp, args.seed)
        t1s.append(deanon_top1(gg, tr)); Hs.append(posterior_entropy(gg, tr))
    mono = all(t1s[i] <= t1s[i + 1] + 1e-9 for i in range(len(t1s) - 1)) and \
           all(Hs[i] >= Hs[i + 1] - 1e-9 for i in range(len(Hs) - 1))
    ok &= _check("monotone: deanon_top1 rises and posterior_entropy falls with eta",
                 mono, f"top1 {[round(x,3) for x in t1s]}; H {[round(x,3) for x in Hs]}")
    return ok


def check_firewall(args):
    """5. Scrambling true_source / is_dummy leaves the attack's posterior identical."""
    print("\n== privacy firewall ==")
    N, T, k = args.N, args.T, args.hops
    tr, g = _attack(N, T, k, args.count, LatencyProfileParams(sender_low=0.1, sender_high=2.0), args.seed)
    rng = np.random.default_rng(args.seed + 99)
    scrambled = make_trace(
        broadcast_id=tr.broadcast_id, obs_node=tr.obs_node, obs_time=tr.obs_time, kind=tr.kind,
        true_source=rng.permutation(tr.true_source), is_dummy=rng.permutation(tr.is_dummy))
    prof = sample_latency_profile(N, k, 0.3, LatencyProfileParams(sender_low=0.1, sender_high=2.0),
                                  rng=np.random.default_rng(args.seed + 1))
    g2 = run_bayes_attribution(scrambled, params=BayesAttributionParams(k, 1.0, 0.25, False, prof))
    identical = (np.array_equal(g.candidate, g2.candidate) and np.allclose(g.posterior, g2.posterior)
                 and np.array_equal(g.broadcast_row, g2.broadcast_row))
    return _check("firewall: scrambling true_source/is_dummy leaves the posterior byte-identical",
                  identical, "attack reads only obs_node/obs_time/kind/broadcast_id")


def check_registry():
    """6. A POSTERIOR attack pairs with POSTERIOR measures and rejects a SCALAR one."""
    print("\n== registry / guess-type wall ==")
    spec = ATTACKS.get("bayes_attribution")
    ok = _check("ATTACKS['bayes_attribution'] -> POSTERIOR", spec is not None and spec.produces == POSTERIOR)
    trio = ("deanon_top1", "mean_true_posterior", "posterior_entropy")
    ok &= _check("the three measures consume POSTERIOR",
                 all(MEASURES[m].consumes == POSTERIOR for m in trio))
    try:
        validate_pairing("bayes_attribution", trio, ATTACKS, MEASURES)
        accept = True
    except ValueError:
        accept = False
    try:
        validate_pairing("bayes_attribution", ("stake_top1_hit",), ATTACKS, MEASURES)
        reject = False
    except ValueError:
        reject = True
    ok &= _check("validate_pairing accepts POSTERIOR trio, rejects a SCALAR measure", accept and reject)
    return ok


def main():
    """Run the selected validation sections; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=int, default=60_000)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--count", type=int, default=19, help="fixed cover per slot -> |S| = count + 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--section", choices=["reference", "regimes", "firewall", "registry"])
    args = ap.parse_args()

    sections = {"reference": check_reference, "regimes": lambda: check_regimes(args),
                "firewall": lambda: check_firewall(args), "registry": check_registry}
    ok = True
    for name in ([args.section] if args.section else list(sections)):
        ok &= sections[name]()

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
