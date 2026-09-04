"""
Route-oracle validation: the upper-bound arm (mixnet_attribution_oracle, granted the
true route) against the route-latent lower bound (mixnet_attribution).

Checks: registry and true_route column contract; W = 1 identity (oracle == latent);
split support collapse; the two nulls (exact at sd = 0, bounded at the shipped law);
paired bracket ordering oracle >= latent over mixing budgets; and the firewall in both
directions (scoring truth unread by the oracle, the route unread by the latent attack).

Run:  python tests/adversary-mixnet-oracle-validation.py [--section null --T 60000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import DEFAULT_F, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, MixnetParams, inject_dummies, make_trace  # noqa: E402
from anonymity.mixnet import apply as mixnet_apply  # noqa: E402
from anonymity.trace import passthrough  # noqa: E402
from network.lognormal_latency import LogNormalParams  # noqa: E402
from network.mixnet_latency import sample_mixnet_lognormal_profile  # noqa: E402
from adversary import (  # noqa: E402
    ATTACKS, MixnetAttributionParams, run_mixnet_attribution, run_mixnet_attribution_oracle)
from metrics import MEASURES, deanon_top1, mean_true_posterior, posterior_entropy  # noqa: E402
from pipeline_contract import POSTERIOR, validate_pairing  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "mixnet_attribution_split.yaml"


def _config_lognormal(path=CONFIG_PATH):
    """Read the shipped LogNormalParams from an experiment config."""
    import yaml
    with path.open(encoding="utf-8") as fh:
        return LogNormalParams(**yaml.safe_load(fh)["latency"]["lognormal"])


SHIPPED = _config_lognormal()


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line and return the boolean."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _build(N, T, W, k, count, seed, mix, assignment="split", sender_scale=0.25):
    """consensus + cover + routed mixnet -> (trace, profile, wired attack params)."""
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=count), T=T, rng=rng)
    prof = sample_mixnet_lognormal_profile(N, W, k, SHIPPED, rng=np.random.default_rng(seed + 1),
                                           assignment=assignment)
    tr = mixnet_apply(s, n, isd, g,
                      params=MixnetParams(W, k, mix, N, sender_scale=sender_scale,
                                          entry_assignment=assignment),
                      latency_oracle=prof, rng=rng)
    return tr, prof, MixnetAttributionParams(k, mix, sender_scale, False, prof)


def _allowed_mask(tr, prof, guess):
    """[K] bool: candidate row consistent with its broadcast's true route (split entry rule)."""
    counts = np.diff(guess.start)
    p_row = np.repeat(tr.true_route[guess.broadcast_row], counts)
    if prof.assignment == "split":
        return prof.entry_of[guess.candidate] == prof.route_entry[p_row]
    return np.ones(guess.candidate.size, dtype=bool)


def check_structure(args):
    """1. Registry + the true_route column's contract + fail-loud on a route-less trace."""
    print("== structure / registry / the true_route column ==")
    ok = True
    spec = ATTACKS.get("mixnet_attribution_oracle")
    ok &= _check("ATTACKS['mixnet_attribution_oracle'] -> POSTERIOR",
                 spec is not None and spec.produces == POSTERIOR)
    try:
        validate_pairing("mixnet_attribution_oracle",
                         ("deanon_top1", "mean_true_posterior", "posterior_entropy"),
                         ATTACKS, MEASURES)
        paired = True
    except ValueError:
        paired = False
    ok &= _check("guess-type wall: pairs with the family-A trio", paired)

    tr, prof, _ = _build(args.N, args.T, 2, args.hops, args.count, args.seed, mix=0.02)
    m = len(tr) // 2
    ok &= _check("mixnet records true_route on BOTH rows of each message",
                 np.array_equal(tr.true_route[:m], tr.true_route[m:]))
    ok &= _check(f"routes in [0, W^k) = [0, {prof.n_routes})",
                 tr.true_route.min() >= 0 and tr.true_route.max() < prof.n_routes)
    under_split = prof.route_entry[tr.true_route[:m]] == prof.entry_of[tr.true_source[:m]]
    ok &= _check("split: every recorded route starts at its sender's own entry",
                 bool(np.all(under_split)))

    ok &= _check("passthrough (none layer) defaults true_route to -1",
                 np.all(passthrough(np.array([1, 2]), np.array([3, 4])).true_route == -1))

    tr_off = mixnet_apply(np.array([1]), np.array([3]), None, None,
                          params=MixnetParams(2, args.hops, 1.0, args.N),
                          latency_oracle=None, rng=np.random.default_rng(0))
    try:
        run_mixnet_attribution_oracle(tr_off, params=MixnetAttributionParams(
            args.hops, 1.0, 0.25, False, prof))
        loud = False
    except ValueError:
        loud = True
    ok &= _check("oracle on a route-less trace fails loud (never silently degrades)", loud)
    return ok


def check_w1(args):
    """2. W = 1: one route makes the grant worthless; oracle == latent bit-identically."""
    print("\n== W = 1 identity (one route -> the oracle learns nothing) ==")
    ok = True
    for assignment in ("split", "uniform"):
        tr, prof, ap = _build(args.N, args.T, 1, args.hops, args.count, args.seed + 7,
                              mix=0.02, assignment=assignment)
        g_lat = run_mixnet_attribution(tr, params=ap)
        g_ora = run_mixnet_attribution_oracle(tr, params=ap)
        same = (np.array_equal(g_lat.candidate, g_ora.candidate)
                and np.array_equal(g_lat.posterior, g_ora.posterior)
                and np.array_equal(g_lat.broadcast_row, g_ora.broadcast_row))
        ok &= _check(f"[{assignment}] oracle == latent, bit-identical", same,
                     f"max|dpost| = {np.abs(g_lat.posterior - g_ora.posterior).max():.2e}")
    return ok


def check_collapse(args):
    """3. Split support collapse: zero off-entry mass; true sender always allowed."""
    print("\n== split support collapse (the |S_t| -> ~|S_t|/W restriction) ==")
    ok = True
    tr, prof, ap = _build(args.N, args.T, 2, args.hops, args.count, args.seed, mix=0.02)
    g = run_mixnet_attribution_oracle(tr, params=ap)
    allowed = _allowed_mask(tr, prof, g)
    counts = np.diff(g.start)

    off_mass = float(g.posterior[~allowed].max(initial=0.0))
    ok &= _check("EXACTLY zero posterior mass on off-entry candidates", off_mass == 0.0,
                 f"max off-entry mass = {off_mass:.1e}")
    true_row = np.repeat(tr.true_source[g.broadcast_row], counts) == g.candidate
    ok &= _check("the true sender is ALWAYS in the allowed set", bool(np.all(allowed[true_row])))
    seg = np.add.reduceat(g.posterior, g.start[:-1])
    ok &= _check("every broadcast's posterior normalises to 1", bool(np.allclose(seg, 1.0)),
                 f"max|sum - 1| = {np.abs(seg - 1).max():.1e}")

    n_allowed = np.add.reduceat(allowed.astype(float), g.start[:-1])
    ratio = float(np.mean(n_allowed / counts))
    ok &= _check("allowed fraction ~ 1/W (the balanced entry map seen through the cover draw)",
                 abs(ratio - 0.5) < 0.03, f"mean n_allowed/|S_t| = {ratio:.4f} vs 1/W = 0.5")
    return ok


def check_null(args):
    """
    4. The two nulls: exact closed forms at sd = 0, bounded convergence at the shipped law.

    At sd = 0 the latent posterior is uniform over S_t and the oracle's over the allowed
    set. At the shipped law with mix -> huge, deterministic per-row bounds are asserted
    rather than argmax z-tests: at eta ~ 0 Top-1 is driven by the one quenched profile.
    """
    print("\n== the nulls: exact at sd = 0; bounded at the shipped law ==")
    ok = True

    flat = LogNormalParams(floor=SHIPPED.floor, mean=SHIPPED.mean, sd=0.0)
    rng = np.random.default_rng(args.seed + 3)
    alpha = sample_relative_stakes(args.N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=args.T, rng=rng)
    s, n, isd, grp = inject_dummies(slots, nodes, args.N, params=DummyParams(count=args.count),
                                    T=args.T, rng=rng)
    prof0 = sample_mixnet_lognormal_profile(args.N, 2, args.hops, flat,
                                            rng=np.random.default_rng(args.seed + 4))
    tr0 = mixnet_apply(s, n, isd, grp,
                       params=MixnetParams(2, args.hops, 1.0, args.N, sender_scale=0.25),
                       latency_oracle=prof0, rng=rng)
    ap0 = MixnetAttributionParams(args.hops, 1.0, 0.25, False, prof0)
    g_lat = run_mixnet_attribution(tr0, params=ap0)
    g_ora = run_mixnet_attribution_oracle(tr0, params=ap0)
    counts = np.diff(g_ora.start).astype(float)
    n_allowed = np.add.reduceat(_allowed_mask(tr0, prof0, g_ora).astype(float), g_ora.start[:-1])

    null_lat, ceil_lat = float(np.mean(1 / counts)), float(np.mean(np.log2(counts)))
    null_ora, ceil_ora = float(np.mean(1 / n_allowed)), float(np.mean(np.log2(n_allowed)))
    t_l, p_l, h_l = (deanon_top1(g_lat, tr0), mean_true_posterior(g_lat, tr0),
                     posterior_entropy(g_lat, tr0))
    t_o, p_o, h_o = (deanon_top1(g_ora, tr0), mean_true_posterior(g_ora, tr0),
                     posterior_entropy(g_ora, tr0))
    ok &= _check("sd=0 latent: top1 = true-post = mean(1/|S|), H = mean(log2|S|)  (exact)",
                 abs(t_l - null_lat) < 1e-9 and abs(p_l - null_lat) < 1e-9
                 and abs(h_l - ceil_lat) < 1e-9,
                 f"top1={t_l:.5f}~{null_lat:.5f}; H={h_l:.4f}~{ceil_lat:.4f}")
    ok &= _check("sd=0 oracle: top1 = true-post = mean(1/n_allowed), H = mean(log2 n_allowed)  (exact)",
                 abs(t_o - null_ora) < 1e-9 and abs(p_o - null_ora) < 1e-9
                 and abs(h_o - ceil_ora) < 1e-9,
                 f"top1={t_o:.5f}~{null_ora:.5f}; H={h_o:.4f}~{ceil_ora:.4f}")
    ok &= _check("the oracle null never falls to the latent null (factor ~W, exact plateaus)",
                 null_ora > 1.5 * null_lat,
                 f"mean(1/n_allowed)={null_ora:.5f} vs mean(1/|S|)={null_lat:.5f} "
                 f"(ratio {null_ora / null_lat:.2f})")
    ok &= _check("route observation costs ~log2 W bits of entropy at infinite mixing",
                 abs((ceil_lat - ceil_ora) - 1.0) < 0.1,
                 f"ceiling gap = {ceil_lat - ceil_ora:.4f} bits vs log2 W = 1.0")

    mix = 30.0
    tr, prof, ap = _build(args.N, args.T, 2, args.hops, args.count, args.seed + 3, mix=mix)
    g_lat = run_mixnet_attribution(tr, params=ap)
    g_ora = run_mixnet_attribution_oracle(tr, params=ap)
    allowed = _allowed_mask(tr, prof, g_ora)
    counts = np.diff(g_ora.start)
    n_allowed = np.add.reduceat(allowed.astype(float), g_ora.start[:-1])

    uniform_allowed = allowed / np.repeat(n_allowed, counts)
    dev = float(np.abs(g_ora.posterior - uniform_allowed).max())
    ok &= _check("shipped law: per-row |oracle posterior - allowed-uniform| < 0.01",
                 dev < 0.01, f"max deviation = {dev:.2e}")
    mtp_o, tgt_o = mean_true_posterior(g_ora, tr), float(np.mean(1 / n_allowed))
    mtp_l, tgt_l = mean_true_posterior(g_lat, tr), float(np.mean(1.0 / counts))
    ok &= _check("shipped law: |oracle true-post - mean(1/n_allowed)| < 0.01",
                 abs(mtp_o - tgt_o) < 0.01, f"{mtp_o:.5f} vs {tgt_o:.5f}")
    ok &= _check("shipped law: |latent true-post - mean(1/|S|)| < 0.01",
                 abs(mtp_l - tgt_l) < 0.01, f"{mtp_l:.5f} vs {tgt_l:.5f}")
    for name, guess, ref in (("oracle", g_ora, n_allowed), ("latent", g_lat, counts)):
        H = posterior_entropy(guess, tr)
        ceil = float(np.log2(ref).mean())
        ok &= _check(f"shipped law: {name} entropy at its ceiling (0 <= ceiling - H <= 0.02)",
                     -1e-9 <= ceil - H <= 0.02, f"H = {H:.4f} vs mean(log2 n) = {ceil:.4f}")
    return ok


def check_bracket(args):
    """5. oracle >= latent at every budget -- paired over seeds, one-sided t where predicted."""
    print("\n== bracket ordering (paired over seeds) ==")
    ok = True
    cells = [1e-4, 5e-3, 2e-2, 1e-1, 1.0]
    seeds = range(args.seed, args.seed + args.seeds)
    for mix in cells:
        d_top1, d_post = [], []
        for s in seeds:
            tr, prof, ap = _build(args.N, args.T, 2, args.hops, args.count, s, mix=mix)
            g_lat = run_mixnet_attribution(tr, params=ap)
            g_ora = run_mixnet_attribution_oracle(tr, params=ap)
            d_top1.append(deanon_top1(g_ora, tr) - deanon_top1(g_lat, tr))
            d_post.append(mean_true_posterior(g_ora, tr) - mean_true_posterior(g_lat, tr))
        d_top1, d_post = np.array(d_top1), np.array(d_post)
        if mix <= 2e-4:
            ok &= _check(f"mix={mix:g}: no reversal beyond noise (both arms ~ certain)",
                         d_top1.mean() >= -0.01 and d_post.mean() >= -0.01,
                         f"d_top1 = {d_top1.mean():+.4f}, d_post = {d_post.mean():+.4f}")
        else:
            t1, p1 = stats.ttest_1samp(d_top1, 0.0, alternative="greater")
            t2, p2 = stats.ttest_1samp(d_post, 0.0, alternative="greater")
            ok &= _check(f"mix={mix:g}: oracle > latent (paired, one-sided p < 0.05)",
                         p1 < 0.05 and p2 < 0.05,
                         f"d_top1 = {d_top1.mean():+.4f} (p={p1:.1e}), "
                         f"d_post = {d_post.mean():+.4f} (p={p2:.1e})")
    return ok


def check_firewall(args):
    """6. Both directions: scoring truth unread by the oracle; the grant unread by the latent."""
    print("\n== firewall, both directions ==")
    ok = True
    tr, prof, ap = _build(args.N, args.T, 2, args.hops, args.count, args.seed, mix=0.02)
    g_lat = run_mixnet_attribution(tr, params=ap)
    g_ora = run_mixnet_attribution_oracle(tr, params=ap)
    rr = np.random.default_rng(args.seed + 99)

    scr = make_trace(broadcast_id=tr.broadcast_id, obs_node=tr.obs_node, obs_time=tr.obs_time,
                     kind=tr.kind, true_source=rr.permutation(tr.true_source),
                     is_dummy=rr.permutation(tr.is_dummy), true_route=tr.true_route)
    g2 = run_mixnet_attribution_oracle(scr, params=ap)
    ok &= _check("scrambling true_source/is_dummy leaves the ORACLE posterior byte-identical",
                 np.array_equal(g_ora.posterior, g2.posterior)
                 and np.array_equal(g_ora.candidate, g2.candidate))

    scr2 = make_trace(broadcast_id=tr.broadcast_id, obs_node=tr.obs_node, obs_time=tr.obs_time,
                      kind=tr.kind, true_source=tr.true_source, is_dummy=tr.is_dummy,
                      true_route=rr.permutation(tr.true_route))
    g3_lat = run_mixnet_attribution(scr2, params=ap)
    g3_ora = run_mixnet_attribution_oracle(scr2, params=ap)
    ok &= _check("scrambling true_route leaves the LATENT posterior byte-identical "
                 "(the standard attack really is route-latent)",
                 np.array_equal(g_lat.posterior, g3_lat.posterior))
    ok &= _check("scrambling true_route MOVES the oracle (the grant is real, not decorative)",
                 not np.array_equal(g_ora.posterior, g3_ora.posterior))
    return ok


SECTIONS = {"structure": check_structure, "w1": check_w1, "collapse": check_collapse,
            "null": check_null, "bracket": check_bracket, "firewall": check_firewall}


def main():
    """Run the selected sections and exit non-zero on any failure."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=int, default=40_000,
                    help="deliberately << the full epoch (tests override T down for speed)")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--count", type=int, default=19, help="fixed cover per slot -> |S| = count + 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=6, help="paired seeds in the bracket section")
    ap.add_argument("--section", choices=sorted(SECTIONS))
    args = ap.parse_args()

    names = [args.section] if args.section else list(SECTIONS)
    ok = all([SECTIONS[n](args) for n in names])          # list first: no short-circuit skipping
    print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
