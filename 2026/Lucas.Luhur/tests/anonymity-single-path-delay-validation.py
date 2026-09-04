"""
Validation of the single-path delay density f_Z and the observation model y = mu(i, r) + Z.

Z = X_S + sum_{j<=m} X_{M,j} is the layer's intentional delay, so the true-sender residual
z = y - mu(true sender, r) ~ f_Z. Checks:
  1-3. f_Z integrates to 1; equal rates give Gamma(m+1); sender hold off gives Gamma(m).
  4-5. f_Z matches simulated Z (KS); E[Z] = sender + m*mix, Var(Z) = sender^2 + m*mix^2.
  6. Residual hinge -- the true-sender residual from a real trace is >= 0 and ~ f_Z (KS).
  7. Discrimination -- heterogeneous d_i^S spreads candidate residuals; homogeneous collapses them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataclasses import replace  # noqa: E402
from scipy import integrate  # noqa: E402
from scipy.stats import gamma as gamma_dist, ks_1samp  # noqa: E402

from consensus import DEFAULT_F, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, SinglePathMixParams, delay_moments, inject_dummies, random_delay_pdf  # noqa: E402
from anonymity.single_path_mix import apply as single_path_mix_apply  # noqa: E402
from network import LatencyProfileParams, sample_latency_profile  # noqa: E402


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _cdf_from_pdf(m, ss, ms, zmax=80.0, npts=80000):
    """Return a callable numerical CDF of f_Z on a fine grid, for the KS tests."""
    zg = np.linspace(0.0, zmax, npts)
    pdf = random_delay_pdf(zg, m, ss, ms)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(zg))])
    return lambda x: np.interp(x, zg, cdf)


def check_density():
    """f_Z: normalisation, the two Gamma limits, KS vs simulation, moments."""
    print("== f_Z (the delay density the attack's likelihood evaluates) ==")
    ok = True

    norm_ok, worst = True, 0.0
    for m, ss, ms in [(3, 0.25, 1.0), (1, 0.25, 1.0), (3, 1.0 / 0.8, 1.0 / 1.7),
                      (3, 0.0, 1.0), (3, 1.0, 1.0)]:
        integral, _ = integrate.quad(lambda z: random_delay_pdf(np.array([z]), m, ss, ms)[0], 0, 300)
        norm_ok &= abs(integral - 1.0) < 1e-6
        worst = max(worst, abs(integral - 1.0))
    ok &= _check("f_Z integrates to 1 (5 rate pairs incl. layer default + sender-off + equal-rate)",
                 norm_ok, f"worst |integral - 1| = {worst:.2e}")

    m, s = 3, 1.0
    zg = np.linspace(0.01, 25, 800)
    d_eq = float(np.max(np.abs(random_delay_pdf(zg, m, s, s) - gamma_dist.pdf(zg, a=m + 1, scale=s))))
    ok &= _check("equal rates -> f_Z == Gamma(m+1, scale)", d_eq < 1e-9, f"max|diff| = {d_eq:.2e}")

    d_off = float(np.max(np.abs(random_delay_pdf(zg, m, 0.0, s) - gamma_dist.pdf(zg, a=m, scale=s))))
    ok &= _check("sender-off -> f_Z == Gamma(m, mix_scale)", d_off < 1e-9, f"max|diff| = {d_off:.2e}")

    m, ss, ms, n = 3, 0.25, 1.0, 300_000
    rng = np.random.default_rng(0)
    z_sim = rng.gamma(m, ms, size=n) + rng.exponential(ss, size=n)
    D = float(ks_1samp(z_sim, _cdf_from_pdf(m, ss, ms)).statistic)
    ok &= _check("general f_Z matches simulated Z (KS, layer default rates)", D < 0.01,
                 f"KS D = {D:.4f}  (n = {n})")

    mean, var = delay_moments(m, ss, ms)
    mean_ok = abs(mean - (ss + m * ms)) < 1e-12 and abs(var - (ss ** 2 + m * ms ** 2)) < 1e-12
    ok &= _check("delay_moments (E[Z] = sender + m*mix; Var = sender^2 + m*mix^2)", mean_ok,
                 f"E[Z] = {mean:.4f} (sim {z_sim.mean():.4f}); Var = {var:.4f} (sim {z_sim.var():.4f})")
    return ok


def check_observation_model(args):
    """Check y = mu(i, r) + Z: the true-sender residual z = y - mu(true, r) is ~ f_Z."""
    print("\n== observation model  y = mu(i, r) + Z  (the residual hinge) ==")
    N, T, f, k, ms, ss = args.N, args.T, args.f, args.hops, args.mix_scale, args.sender_scale
    ok = True

    rng = np.random.default_rng(args.seed)
    alpha = sample_relative_stakes(N, args.shape, rng=rng)
    slots, nodes = simulate_events(alpha, f=f, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=1), T=T, rng=rng)

    profile = sample_latency_profile(
        N, k, args.d,
        LatencyProfileParams(sender_low=0.1, sender_high=1.0, receiver_low=0.1, receiver_high=1.0),
        rng=np.random.default_rng(args.seed + 1))
    trace = single_path_mix_apply(s, n, isd, g,
                                  params=SinglePathMixParams(k, ms, N, sender_scale=ss),
                                  latency_oracle=profile, rng=rng)

    ent, ext = trace.is_entry, trace.is_exit
    y = trace.obs_time[ext] - trace.obs_time[ent]
    sender = trace.true_source[ext]
    receiver = trace.obs_node[ext]
    z = y - profile.mu(sender, receiver)
    isd_msg = trace.is_dummy[ext]

    nonneg = bool(np.all(z >= -1e-9))
    ok &= _check("true-sender residual z = y - mu(true, r) >= 0", nonneg,
                 f"min z = {float(z.min()):.4e}")

    n_stages = k + (1 if False else 0)
    D_all = float(ks_1samp(z, _cdf_from_pdf(n_stages, ss, ms)).statistic)
    z_real = z[~isd_msg]
    D_real = float(ks_1samp(z_real, _cdf_from_pdf(n_stages, ss, ms)).statistic)
    ok &= _check("HINGE: residual z ~ f_Z (KS, all messages AND real broadcasts only)",
                 D_all < 0.02 and D_real < 0.05,
                 f"KS D = {D_all:.4f} (all, n={z.size}); {D_real:.4f} (real, n={z_real.size})")

    grp_msg = trace.broadcast_id[ext]
    i0 = int(np.where(~isd_msg)[0][0])
    b0, y_b, r_b = grp_msg[i0], y[i0], receiver[i0]
    cand = sender[grp_msg == b0]
    r_vec = np.full(cand.size, r_b)
    z_het = y_b - profile.mu(cand, r_vec)
    prof_homog = sample_latency_profile(N, k, args.d, rng=np.random.default_rng(args.seed + 2))
    z_hom = y_b - prof_homog.mu(cand, r_vec)
    spread_ok = np.std(z_het) > 1e-6 and np.std(z_hom) < 1e-12
    ok &= _check("discrimination (heterogeneous d_i^S spreads residuals; homogeneous collapses them)",
                 bool(spread_ok) and cand.size >= 2,
                 f"batch |S|={cand.size}; std(z) heterogeneous={np.std(z_het):.4f}, homogeneous={np.std(z_hom):.2e}")
    return ok


def check_rho_coupling():
    """
    Check `sender_scale = "auto"` keeps rho = mix_scale/sender_scale fixed across a sweep.

    Both intentional holds are design knobs tied by rho >= 4 (Das et al.); "auto" re-derives
    the sender hold per cell instead of pinning the literal 0.25.
    """
    print("\n== rho coupling (sender_scale = 'auto') ==")
    ok = True
    k, rho = 3, 4.0

    base = SinglePathMixParams(hops=k, mix_scale=1.0, sender_scale="auto", rho=rho)
    ok &= _check("'auto' resolves to mix_scale/rho", abs(base.sender_scale - 0.25) < 1e-12,
                 f"mix 1.0, rho {rho} -> sender {base.sender_scale}")
    swept = [replace(base, mix_scale=m) for m in (0.001, 0.1, 1.0, 10.0)]
    ratios = [p.mix_scale / p.sender_scale for p in swept]
    ok &= _check("rho HOLDS across a swept mix_scale (survives dataclasses.replace)",
                 max(abs(r - rho) for r in ratios) < 1e-9,
                 f"rho = {[round(r, 6) for r in ratios]} over mix 0.001-10")

    worst = 0.0
    for p in swept:
        _, var = delay_moments(k, p.sender_scale, p.mix_scale)
        worst = max(worst, abs(np.sqrt(var) - p.mix_scale * np.sqrt(k + 1.0 / rho ** 2)))
    ok &= _check("sigma_Z == mix_scale*sqrt(k + 1/rho^2) exactly (scales to 0, no floor)",
                 worst < 1e-12, f"max|diff| = {worst:.2e}  (= 1.75*mix at k=3, rho=4)")

    fixed = replace(SinglePathMixParams(hops=k, mix_scale=1.0, sender_scale=0.25), mix_scale=0.001)
    ok &= _check("a FIXED sender_scale is left alone by the sweep (old runs reproducible)",
                 fixed.sender_scale == 0.25, f"mix 0.001 -> sender {fixed.sender_scale} (rho "
                                             f"{fixed.mix_scale/fixed.sender_scale:.3f}, the stale-hold case)")

    L = 3.25
    sig = lambda r: L * np.sqrt(1 + k * r ** 2) / (1 + k * r)
    grid = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0]
    vals = [sig(r) for r in grid]
    i_min = int(np.argmin(vals))
    ok &= _check("sigma_Z(rho) at fixed budget is minimised near rho = 1 (not at the rho=4 boundary)",
                 abs(grid[i_min] - 1.0) < 1e-9,
                 f"argmin at rho = {grid[i_min]}; sigma_Z(1) = {sig(1.0):.3f} < "
                 f"sigma_Z(4) = {sig(4.0):.3f} < sigma_Z(inf) = {L/np.sqrt(k):.3f}")
    ok &= _check("=> rho = 4 is the LEAST anonymous split allowed by rho >= 4 (adversary-favouring)",
                 all(sig(r) >= sig(4.0) - 1e-12 for r in grid if r >= 4.0),
                 f"sigma_Z rises monotonically for rho >= 4: "
                 f"{[round(sig(r), 3) for r in grid if r >= 4.0]}")

    for kw, why in (({"sender_scale": "nope"}, "an unknown string"),
                    ({"sender_scale": "auto", "rho": 0.0}, "rho = 0 with 'auto'")):
        try:
            SinglePathMixParams(hops=k, mix_scale=1.0, **kw)
            ok &= _check(f"rejects {why}", False, "accepted")
        except ValueError:
            ok &= _check(f"rejects {why}", True)
    return ok


def main():
    """Run the selected validation sections; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--T", type=int, default=60_000)
    ap.add_argument("--shape", type=float, default=1.33)
    ap.add_argument("--f", type=float, default=DEFAULT_F)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--mix-scale", dest="mix_scale", type=float, default=1.0)
    ap.add_argument("--sender-scale", dest="sender_scale", type=float, default=0.25)
    ap.add_argument("--d", type=float, default=0.3)
    ap.add_argument("--section", choices=["density", "observation", "rho"], help="run only one section")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sections = {"density": check_density, "observation": lambda: check_observation_model(args),
                "rho": check_rho_coupling}
    ok = True
    for i, name in enumerate([args.section] if args.section else list(sections)):
        ok &= sections[name]()

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
