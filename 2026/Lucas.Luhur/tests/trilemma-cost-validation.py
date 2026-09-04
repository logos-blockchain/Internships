"""
Validation of the trilemma cost metrics bandwidth_overhead (beta) and latency_overhead (ell).

  beta = |S| / L -> 1 + p_s (N/Phi - 1)
  ell  = E[D^AC] / E[D^base] = 1 + (d(k+1) + 1/lambda_S + (k+1 or k)/lambda_M) / E[D_br(N)]

Sections (--section {bandwidth, latency}):
  bandwidth: beta = 1 without cover; the closed form across p_s; monotone in p_s;
    layer-invariant; the Jensen floor beta >= 1 + p_s(1/phi(1/N) - 1) and the small-alpha
    expansion Phi = -ln(1-f) - 1/2 ln(1-f)^2 sum(alpha^2) + O(a^3).
  latency: ell matches the closed form; ell > 1; rises with hold, mixing and hops;
    a homogeneous link d adds exactly d(k+1)/E[D_br(N)].
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
    expected_winners_per_slot,
    lottery,
    sample_relative_stakes,
    simulate_events,
)
from anonymity import LAYERS, DummyParams, SinglePathMixParams, inject_dummies, passthrough  # noqa: E402
from anonymity.single_path_mix import apply as single_path_mix_apply  # noqa: E402
from metrics import bandwidth_overhead, latency_overhead  # noqa: E402
from network.latency import DEFAULT_D, broadcast_latency_theory, lam_from_rho  # noqa: E402
from network.latency_profile import sample_latency_profile  # noqa: E402


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def check_bandwidth(alpha, slots, nodes, args, rng):
    """Check beta = |S|/L: the closed form, monotonicity, layer-invariance and the floor."""
    print("== bandwidth (beta = |S|/L) ==")
    N, T = args.N, args.T
    Phi = expected_winners_per_slot(alpha, args.f)
    print(f"  Phi = E[L_t] = {Phi:.4f}  (~ f = {args.f:.4f})")
    ok = True

    s0, n0, d0, _ = inject_dummies(slots, nodes, N, params=DummyParams(count=0), T=T, rng=rng)
    beta0 = bandwidth_overhead(passthrough(s0, n0, d0))
    ok &= _check("no cover -> beta = 1", beta0 == 1.0, f"beta = {beta0:.6f}")

    betas = []
    for p_s in [0.002, 0.005, 0.01, 0.02]:
        s, n, d, _ = inject_dummies(slots, nodes, N, params=DummyParams(p_s=p_s), T=T, rng=rng)
        beta_sim = bandwidth_overhead(passthrough(s, n, d))
        beta_th = 1.0 + p_s * (N / Phi - 1.0)
        betas.append(beta_sim)
        rel = abs(beta_sim - beta_th) / beta_th
        ok &= _check(f"beta(p_s={p_s}) -> 1 + p_s(N/Phi - 1)", rel < args.rtol,
                     f"sim = {beta_sim:.2f}   theory = {beta_th:.2f}   rel_err = {rel:.3f}")

    ok &= _check("beta increases with p_s",
                 all(betas[i] < betas[i + 1] for i in range(len(betas) - 1)),
                 f"{[round(b, 1) for b in betas]}")

    s, n, d, g = inject_dummies(slots, nodes, N, params=DummyParams(p_s=0.005), T=T, rng=rng)
    b_none = bandwidth_overhead(passthrough(s, n, d))
    b_tor = bandwidth_overhead(LAYERS["single_path_mix"](s, n, d, g, params=None, latency_oracle=None, rng=rng))
    ok &= _check("layer-invariant (none == single_path_mix; beta ignores hops)",
                 b_none == b_tor, f"none = {b_none:.4f}   single_path_mix = {b_tor:.4f}")

    phi1 = float(lottery(1.0 / N, args.f))
    cap = N * phi1
    ok &= _check("Jensen: Phi <= N*phi(1/N) (equal-stake cap)", Phi <= cap * (1 + 1e-9),
                 f"Phi = {Phi:.6f} <= cap = {cap:.6f}")
    beta_floor = lambda ps: 1.0 + ps * (1.0 / phi1 - 1.0)
    floor_ok, worst = True, 0.0
    for p_s in [0.002, 0.005, 0.01, 0.02]:
        s, n, d, _ = inject_dummies(slots, nodes, N, params=DummyParams(p_s=p_s), T=T, rng=rng)
        b_sim = bandwidth_overhead(passthrough(s, n, d))
        floor = beta_floor(p_s)
        floor_ok &= b_sim >= floor * (1.0 - args.rtol)
        worst = max(worst, (b_sim - floor) / floor)
    ok &= _check("beta >= 1 + p_s(1/phi(1/N) - 1) (the floor bounds sim)", floor_ok,
                 f"floor(0.01) = {beta_floor(0.01):.1f}; sim within +{worst*100:.1f}% of floor")
    Phis = {}
    for k in (1.0, args.shape, 10.0):
        ak = sample_relative_stakes(N, k, rng=np.random.default_rng(args.seed + 100))
        Phis[k] = expected_winners_per_slot(ak, args.f)
    spread = (max(Phis.values()) - min(Phis.values())) / cap
    ok &= _check("beta distribution-free: Phi flat across k in {1, shape, 10}", spread < 0.01,
                 f"Phi/cap in [{min(Phis.values())/cap:.4f}, {max(Phis.values())/cap:.4f}] (spread {spread*100:.2f}%)")

    L = -np.log(1.0 - args.f)
    s2 = float(np.sum(alpha ** 2))
    Phi_eq13 = L - 0.5 * L * L * s2
    rel13 = abs(Phi_eq13 - Phi) / Phi
    ok &= _check("Phi = -ln(1-f) - 1/2 ln(1-f)^2 * sum(alpha^2) (exact to O(a^3))",
                 rel13 < 1e-3,
                 f"eq13 = {Phi_eq13:.6f}  sim Phi = {Phi:.6f}  (sum a^2 = {s2:.5f}, rel {rel13*100:.3f}%)")
    return ok


def check_latency(slots, nodes, args, rng):
    """Check ell = E[D^AC]/E[D^base]: the closed form, ell > 1, monotonicity, the link-d term."""
    print("== latency (ell = E[D^AC]/E[D^base]) ==")
    N = args.N
    bmean = broadcast_latency_theory(N, args.C, DEFAULT_D, lam_from_rho(args.rho, DEFAULT_D))
    print(f"  E[D_br(N)] = {bmean:.3f}")
    ok = True

    def _ell(sender_scale, mix_scale, hops, receiver_delays, link_d, rngx=None):
        """Measure ell from a single_path_mix trace under a homogeneous link-d profile."""
        rr = rng if rngx is None else rngx
        s, n, d, g = inject_dummies(slots, nodes, N, params=DummyParams(count=1), T=args.T, rng=rr)
        profile = sample_latency_profile(N, hops, link_d, rng=rr)
        tr = single_path_mix_apply(s, n, d, g,
                            params=SinglePathMixParams(hops, mix_scale, N, sender_scale=sender_scale,
                                                 receiver_delays=receiver_delays),
                            latency_oracle=profile, rng=rr)
        return latency_overhead(tr, broadcast_mean=bmean)

    def _theo(sender_scale, mix_scale, hops, receiver_delays, link_d):
        """Closed-form ell for the same parameters."""
        n_stages = hops + (1 if receiver_delays else 0)
        return 1.0 + (link_d * (hops + 1) + sender_scale + n_stages * mix_scale) / bmean

    n_emit = inject_dummies(slots, nodes, N, params=DummyParams(count=1), T=args.T,
                            rng=np.random.default_rng(args.seed + 99))[0].size
    cases = [
        dict(sender_scale=0.0, mix_scale=1.0, hops=3, receiver_delays=True, link_d=0.0),
        dict(sender_scale=1.0, mix_scale=1.0, hops=3, receiver_delays=True, link_d=0.0),
        dict(sender_scale=2.0, mix_scale=5.0, hops=3, receiver_delays=True, link_d=0.0),
        dict(sender_scale=1.0, mix_scale=1.0, hops=1, receiver_delays=False, link_d=0.0),
        dict(sender_scale=1.0, mix_scale=2.0, hops=3, receiver_delays=True, link_d=0.3),
    ]
    ells = []
    for c in cases:
        e_sim, e_th = _ell(**c), _theo(**c)
        ells.append(e_sim)
        n_stages = c["hops"] + (1 if c["receiver_delays"] else 0)
        var_Z = c["sender_scale"] ** 2 + n_stages * c["mix_scale"] ** 2
        se = np.sqrt(var_Z / n_emit) / bmean
        z = abs(e_sim - e_th) / se
        ok &= _check(f"ell(l_S={c['sender_scale']}, mu={c['mix_scale']}, k={c['hops']}, "
                     f"R+={c['receiver_delays']}, d={c['link_d']})",
                     z < 4.0, f"sim={e_sim:.4f}  theo={e_th:.4f}  (z={z:.2f})")

    ok &= _check("ell > 1 for every case", all(e > 1.0 for e in ells),
                 f"{[round(e, 2) for e in ells]}")

    base = dict(sender_scale=1.0, mix_scale=1.0, hops=2, receiver_delays=True, link_d=0.0)
    e0 = _ell(**base)
    ok &= _check("ell rises with sender hold, mixing delay, hops",
                 _ell(**{**base, "sender_scale": 3.0}) > e0
                 and _ell(**{**base, "mix_scale": 4.0}) > e0
                 and _ell(**{**base, "hops": 4}) > e0)

    dd, k = 0.5, 3
    e_no = _ell(1.0, 1.0, k, True, 0.0, rngx=np.random.default_rng(args.seed + 7))
    e_d = _ell(1.0, 1.0, k, True, dd, rngx=np.random.default_rng(args.seed + 7))
    ok &= _check("homogeneous link d adds d(k+1)/E[D_br]",
                 abs((e_d - e_no) - dd * (k + 1) / bmean) < 1e-9,
                 f"delta_ell={e_d - e_no:.4f} vs d(k+1)/E[D_br]={dd * (k + 1) / bmean:.4f}")
    return ok


def main():
    """Run the selected cost-overhead sections; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--T", type=int, default=100_000)
    ap.add_argument("--shape", type=float, default=1.33, help="Pareto stake shape k")
    ap.add_argument("--f", type=float, default=DEFAULT_F)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.1, help="network jitter ratio")
    ap.add_argument("--rtol", type=float, default=0.05, help="relative tol for beta -> theory")
    ap.add_argument("--section", choices=["bandwidth", "latency"], help="run only one section")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = np.random.default_rng(args.seed)
    alpha = sample_relative_stakes(args.N, args.shape, rng=base)
    slots, nodes = simulate_events(alpha, f=args.f, T=args.T, rng=base)
    print(f"consensus: N={args.N}, T={args.T}, k={args.shape} -> {slots.size} winners\n")

    sections = {
        "bandwidth": lambda: check_bandwidth(alpha, slots, nodes, args, np.random.default_rng(args.seed + 1)),
        "latency": lambda: check_latency(slots, nodes, args, np.random.default_rng(args.seed + 2)),
    }
    ok = True
    for i, name in enumerate([args.section] if args.section else list(sections)):
        if i:
            print()
        ok &= sections[name]()

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
