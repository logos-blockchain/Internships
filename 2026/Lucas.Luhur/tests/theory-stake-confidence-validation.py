"""
Validation of theory.stake_confidence against the simulation pipeline and closed-form anchors.

stake_confidence is the fraction of nodes whose alpha_hat lies within +/-gamma of the true
stake; that band is an interval in n_i ~ Binomial(T, q_i) (see src/theory/stake_confidence.py).
Checks:
  1. Inversion -- count_at_estimate inverts the estimator; n(0) = T*p_s.
  2. Band == interval -- the measure's event and the count interval select identical nodes.
  3. Engine -- theory vs the full pipeline, paired on the same stake vectors.
  4. Two routes -- exact binomial vs continuity-corrected Gaussian agree; 5. limits in p_s, T, gamma.
  6. No-cover trap -- inference_confidence is the p_s = 0 limit, not this curve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import DEFAULT_F, DEFAULT_T, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, inject_dummies, passthrough            # noqa: E402
from adversary import SetStakeInferenceParams, run_set_stake_inference    # noqa: E402
from metrics.stake_privacy import inference_confidence, stake_confidence  # noqa: E402
from theory.stake_confidence import (                                     # noqa: E402
    confidence_probability, confidence_probability_normal, count_at_estimate,
    participation_prob, per_node_confidence,
)


class _Ctx:
    """Minimal stand-in for the pipeline ScoreContext read by the stake-privacy measures."""

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def check_inversion(N, T, f, gamma, seed):
    """1. count_at_estimate inverts the estimator exactly, and n(0) is the cover baseline."""
    print("\n1. INVERSION -- n(a) is the estimator's exact inverse")
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, 2.0, rng=rng)
    ok = True
    for p_s in (0.01, 0.1, 0.5):
        n = count_at_estimate(alpha, p_s, f, T)
        q = np.clip(n / T, 0.0, 1.0 - 1e-12)
        back = np.log((1.0 - q) / (1.0 - p_s)) / np.log(1.0 - f)
        err = float(np.abs(back - alpha).max())
        ok &= _check(f"round-trip at p_s={p_s}", err < 1e-9, f"max |a - a(n(a))| = {err:.2e}")
        base = float(count_at_estimate(0.0, p_s, f, T))
        ok &= _check(f"n(0) = T*p_s at p_s={p_s}", abs(base - T * p_s) < 1e-6,
                     f"{base:.4f} vs {T * p_s:.4f}")
    return ok


def check_band_is_interval(N, T, f, gamma, seed):
    """2. The measure's band event and the theory's count interval are the same event."""
    print("\n2. BAND == INTERVAL -- the two definitions select identical nodes")
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, 2.0, rng=rng)
    ok = True
    for p_s in (0.01, 0.1, 0.3):
        q = participation_prob(alpha, p_s, f)
        n = rng.binomial(T, q)
        ahat = np.maximum(
            np.log((1.0 - np.clip(n / T, 0.0, 1.0 - 1e-12)) / (1.0 - p_s)) / np.log(1.0 - f), 0.0)
        in_band = (ahat >= alpha * (1.0 - gamma)) & (ahat <= alpha * (1.0 + gamma))
        lo = np.ceil(count_at_estimate(alpha * (1.0 - gamma), p_s, f, T))
        hi = np.floor(count_at_estimate(alpha * (1.0 + gamma), p_s, f, T))
        in_interval = (n >= lo) & (n <= hi)
        bad = int((in_band != in_interval).sum())
        ok &= _check(f"identical node sets at p_s={p_s}", bad == 0,
                     f"{bad} mismatches / {N} nodes ({int(in_band.sum())} in band)")
        floored = ahat == 0.0
        ok &= _check(f"floored alpha_hat never in band at p_s={p_s}",
                     not bool((floored & in_band).any()),
                     f"{int(floored.sum())} nodes floored at 0")
    return ok


def check_vs_engine(N, T, f, gamma, shape, reps, seed):
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
            sim = stake_confidence(ahat, _Ctx(alpha, gamma))
            theo = confidence_probability(alpha, p_s, gamma=gamma, f=f, T=T)
            sims.append(sim)
            theos.append(theo)
            diffs.append(sim - theo)
        d = np.asarray(diffs)
        sem = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        z = abs(float(d.mean()) / sem) if sem > 0 else 0.0
        ok &= _check(
            f"paired |z| <= 3 at p_s={p_s}", z <= 3.0,
            f"sim {np.mean(sims):.6f} vs theory {np.mean(theos):.6f}, "
            f"diff {d.mean():+.6f} +/- {sem:.6f}, z = {z:.2f}")
    return ok


def check_two_routes(N, T, f, gamma, shape, seed):
    """
    4. The exact binomial route and the continuity-corrected Gaussian route agree.

    Run at both the reduced T and the full epoch, since the approximation's quality depends
    on the horizon; the band spans only a fraction of a count sd, so the correction matters.
    """
    print("\n4. TWO ROUTES -- exact binomial vs continuity-corrected Gaussian")
    ok = True
    for horizon in (T, DEFAULT_T):
        for p_s in (0.01, 0.1):
            rng = np.random.default_rng(seed)
            alpha = sample_relative_stakes(N, shape, rng=rng)
            q = participation_prob(alpha, p_s, f)
            width = (count_at_estimate(alpha * (1.0 + gamma), p_s, f, horizon)
                     - count_at_estimate(alpha * (1.0 - gamma), p_s, f, horizon))
            band_sd = float(np.median(width / np.sqrt(horizon * q * (1.0 - q))))
            ex = confidence_probability(alpha, p_s, gamma=gamma, f=f, T=horizon)
            ga = confidence_probability_normal(alpha, p_s, gamma=gamma, f=f, T=horizon)
            rel = abs(ga - ex) / max(ex, 1e-12)
            ok &= _check(f"agree to 2% at T={horizon}, p_s={p_s}", rel < 0.02,
                         f"exact {ex:.6f} vs normal {ga:.6f} ({rel:+.2%}); "
                         f"band = {band_sd:.3f} sd")
    return ok


def check_limits(N, T, f, gamma, shape, seed):
    """
    5. The closed form's own asymptotics.

    gamma -> 1 does not accept every node: alpha_hat is floored at 0, which lies below every
    band with alpha > 0. Consistency is in T (alpha_hat -> alpha), but band/sd grows only as
    sqrt(T), so monotone increase over decades is checked rather than convergence to 1.
    """
    print("\n5. LIMITS -- the closed form's own asymptotics")
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, shape, rng=rng)
    ok = True
    heavy = confidence_probability(alpha, 0.999, gamma=gamma, f=f, T=T)
    ok &= _check("p_s -> 1 drives confidence to ~0 (cover kills the signal)",
                 heavy < 0.02, f"{heavy:.6f}")
    mono = [confidence_probability(alpha, p, gamma=gamma, f=f, T=T)
            for p in (0.01, 0.05, 0.1, 0.2)]
    ok &= _check("monotone DECREASING in p_s (cover protects)",
                 all(np.diff(mono) < 0), " -> ".join(f"{v:.5f}" for v in mono))
    grow = [confidence_probability(alpha, 0.05, gamma=gamma, f=f, T=t)
            for t in (T, 10 * T, 100 * T, 1000 * T)]
    ok &= _check("monotone INCREASING in T (alpha_hat is consistent)",
                 all(np.diff(grow) > 0), " -> ".join(f"{v:.5f}" for v in grow))
    ratios = [grow[i + 1] / grow[i] for i in range(len(grow) - 1)]
    ok &= _check("and grows as sqrt(T) -- the narrow-band scaling law",
                 abs(ratios[-1] / np.sqrt(10.0) - 1.0) < 0.10,
                 f"per-decade ratios {', '.join(f'{r:.2f}' for r in ratios)} "
                 f"vs sqrt(10) = 3.162")
    wide = [confidence_probability(alpha, 0.05, gamma=g, f=f, T=T) for g in (0.1, 0.5, 0.999)]
    ok &= _check("monotone INCREASING in gamma (but NOT -> 1: the alpha_hat floor)",
                 all(np.diff(wide) > 0), " -> ".join(f"{v:.5f}" for v in wide))
    pn = per_node_confidence(alpha, 0.05, gamma=gamma, f=f, T=T)
    top, bottom = pn[np.argsort(alpha)[-20:]].mean(), pn[np.argsort(alpha)[:20]].mean()
    ok &= _check("whales are pinned more often than minnows", top > bottom,
                 f"top-20 {top:.5f} vs bottom-20 {bottom:.5f}")
    return ok


def check_no_cover_trap(N, T, f, gamma, shape, seed):
    """
    6. metrics.inference_confidence is the p_s = 0 form, and not this curve at p_s > 0.

    The agreement leg runs at the full epoch because the paper form uses a Gaussian with a
    linearised band and needs large T; the disagreement leg shows it is flat in p_s.
    """
    print("\n6. THE NO-COVER TRAP -- inference_confidence is the p_s = 0 limit, not this curve")
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, shape, rng=rng)
    ok = True
    old_epoch = float(np.mean(inference_confidence(alpha, gamma, DEFAULT_T, f)))
    at0 = confidence_probability(alpha, 0.0, gamma=gamma, f=f, T=DEFAULT_T)
    ok &= _check("the two AGREE at p_s = 0, full epoch (it IS the no-cover limit)",
                 abs(old_epoch - at0) / max(at0, 1e-12) < 0.01,
                 f"paper form {old_epoch:.5f} vs exact {at0:.5f} "
                 f"({old_epoch / max(at0, 1e-12):.4f}x)")
    at_op = confidence_probability(alpha, 0.1, gamma=gamma, f=f, T=DEFAULT_T)
    ok &= _check("and DISAGREE at the operating point p_s = 0.1 (>5x)",
                 old_epoch > 5.0 * at_op,
                 f"paper form {old_epoch:.5f} vs exact {at_op:.5f} "
                 f"({old_epoch / max(at_op, 1e-12):.1f}x)")
    ok &= _check("the paper form is FLAT in p_s (it has no p_s argument at all)",
                 True, "which is exactly why it cannot carry a cover-swept panel")
    return ok


def main():
    """Run the stake_confidence validation checks; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=500, help="nodes")
    ap.add_argument("--T", type=int, default=40_000,
                    help="slots (overridden DOWN from the epoch for speed -- tests may, "
                         "experiments may not)")
    ap.add_argument("--f", type=float, default=DEFAULT_F)
    ap.add_argument("--gamma", type=float, default=0.1, help="accuracy band half-width")
    ap.add_argument("--shape", type=float, default=2.0,
                    help="Pareto k (2.0 = the tamer validation default, per the repo convention)")
    ap.add_argument("--reps", type=int, default=12, help="paired chains for the engine check")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"theory.stake_confidence validation -- N={a.N} T={a.T} f={a.f:.5f} "
          f"gamma={a.gamma} shape={a.shape} reps={a.reps} seed={a.seed}")

    ok = True
    ok &= check_inversion(a.N, a.T, a.f, a.gamma, a.seed)
    ok &= check_band_is_interval(a.N, a.T, a.f, a.gamma, a.seed)
    ok &= check_vs_engine(a.N, a.T, a.f, a.gamma, a.shape, a.reps, a.seed)
    ok &= check_two_routes(a.N, a.T, a.f, a.gamma, a.shape, a.seed)
    ok &= check_limits(a.N, a.T, a.f, a.gamma, a.shape, a.seed)
    ok &= check_no_cover_trap(a.N, a.T, a.f, a.gamma, a.shape, a.seed)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
