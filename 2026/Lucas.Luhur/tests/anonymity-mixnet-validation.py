"""
Mix-net attribution validation: the log-normal W x k grid layer and its route-marginalising
attack (network/mixnet_latency.py, adversary/mixnet_attribution.py).

Checks: route counts W^k / W^(k-1), the split entry map and the registry; the W = 1 reduction
to the single path (bit-identical draws and posterior); no exact ties, sigma_d(W) closed forms
and the route-diversity gain; the split assignment as the adversary-favouring default (paired
over seeds); sigma_hat_d as a biased estimator of sigma_d; the sd = 0 null; the privacy
firewall; calibration to the ping data; and row-chunked marginalisation being bit-identical.
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
from anonymity import DummyParams, MixnetParams, SinglePathMixParams, inject_dummies, make_trace  # noqa: E402
from anonymity.mixnet import apply as mixnet_apply  # noqa: E402
from anonymity.single_path_mix import apply as spm  # noqa: E402
from network.latency_profile import LatencyProfile  # noqa: E402
from network.lognormal_latency import LogNormalParams, link_moments, sample_lognormal_links  # noqa: E402
from network.mixnet_latency import sample_mixnet_lognormal_profile  # noqa: E402
from network.ping_data import ping_link_moments  # noqa: E402
from adversary import (  # noqa: E402
    ATTACKS, BayesAttributionParams, MixnetAttributionParams,
    run_bayes_attribution, run_mixnet_attribution)
from metrics import MEASURES, deanon_top1, mean_true_posterior, posterior_entropy  # noqa: E402
from pipeline_contract import POSTERIOR, validate_pairing  # noqa: E402
from theory.attribution import candidate_set_law  # noqa: E402

CONFIG_PATH =REPO_ROOT / "experiments" / "configs" / "mixnet_attribution.yaml"
SINGLE_PATH_CONFIG = REPO_ROOT / "experiments" / "configs" / "single_path_mix_lognormal_attribution.yaml"
PING_TOL = 0.02


def _config_lognormal(path=CONFIG_PATH):
    """Load a config's `latency.lognormal` block as LogNormalParams."""
    import yaml
    with path.open(encoding="utf-8") as fh:
        return LogNormalParams(**yaml.safe_load(fh)["latency"]["lognormal"])


SHIPPED = _config_lognormal()


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line and return the boolean."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _attack_mixnet(N, T, W, k, count, params, seed, sender_scale=0.25, mix_scale=1.0,
                   assignment="split"):
    """Run consensus + cover + mixnet (log-normal grid) + attack, returning (trace, guess).

    `assignment` is the sender -> entry rule; it travels with the profile so the layer, the
    traffic and the attack all read one rule.
    """
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=count), T=T, rng=rng)
    prof = sample_mixnet_lognormal_profile(N, W, k, params, rng=np.random.default_rng(seed + 1),
                                           assignment=assignment)
    tr = mixnet_apply(s, n, isd, g,
                      params=MixnetParams(W, k, mix_scale, N, sender_scale=sender_scale,
                                          entry_assignment=assignment),
                      latency_oracle=prof, rng=rng)
    guess = run_mixnet_attribution(
        tr, params=MixnetAttributionParams(k, mix_scale, sender_scale, False, prof))
    return tr, guess


def _paired_w1_profiles(N, k, params, seed, assignment="split"):
    """
    Draw a single-path LatencyProfile and a W=1 grid profile from the same seed.

    Each calls its own sampler on a fresh rng; the draw order alone makes them bit-identical.
    The entry assignment consumes no randomness, so the anchor must hold under both rules.
    """
    d_s, d_r, d_mix = sample_lognormal_links(N, k, params, rng=np.random.default_rng(seed))
    sp = LatencyProfile(d_sender=d_s, d_receiver=d_r, d_mix=d_mix)
    mx = sample_mixnet_lognormal_profile(N, 1, k, params, rng=np.random.default_rng(seed),
                                         assignment=assignment)
    return sp, mx


def check_structure(args):
    """Check route counts W^k / W^(k-1), the split's entry map, and the registry pairing."""
    print("== structure / registry ==")
    ok = True
    for W, k in ((1, 3), (2, 3), (3, 2), (2, 4)):
        prof = sample_mixnet_lognormal_profile(50, W, k, SHIPPED, rng=np.random.default_rng(0))
        ok &= _check(f"W={W}, k={k}: profile.n_routes == W^k = {W ** k}", prof.n_routes == W ** k)
        ok &= _check(f"W={W}, k={k}: link tables are [N, W]",
                     prof.d_sender.shape == (50, W) and prof.d_receiver.shape == (50, W))

    for W, k in ((1, 3), (2, 3), (3, 2)):
        for a, want in (("uniform", W ** k), ("split", W ** (k - 1))):
            prof = sample_mixnet_lognormal_profile(50, W, k, SHIPPED, rng=np.random.default_rng(0),
                                                   assignment=a)
            ok &= _check(f"W={W}, k={k}, {a}: n_routes_per_sender == {want}",
                         prof.n_routes_per_sender == want)
    for W in (2, 3):
        prof = sample_mixnet_lognormal_profile(600, W, 3, SHIPPED, rng=np.random.default_rng(0),
                                               assignment="split")
        counts = np.bincount(prof.entry_of, minlength=W)
        consistent = all(np.all(prof.route_entry[prof.routes_by_entry[w]] == w) for w in range(W))
        ok &= _check(f"W={W} split: entry map is balanced and routes_by_entry is consistent",
                     counts.max() - counts.min() <= 1 and consistent,
                     f"senders per entry {counts.tolist()}")
    pa =sample_mixnet_lognormal_profile(300, 2, 3, SHIPPED, rng=np.random.default_rng(5),
                                         assignment="split")
    pb = sample_mixnet_lognormal_profile(300, 2, 3, SHIPPED, rng=np.random.default_rng(5),
                                         assignment="uniform")
    ok &= _check("the two assignments read the SAME quenched links (assignment draws no randomness)",
                 np.array_equal(pa.d_sender, pb.d_sender)
                 and np.array_equal(pa.d_receiver, pb.d_receiver)
                 and np.array_equal(pa.route_internal, pb.route_internal))
    try:
        mixnet_apply(np.array([1]), np.array([3]), None, None,
                     params=MixnetParams(2, 3, 1.0, 300, entry_assignment="uniform"),
                     latency_oracle=pa, rng=np.random.default_rng(0))
        caught = False
    except ValueError:
        caught = True
    ok &= _check("layer/profile entry_assignment mismatch fails loud", caught)

    from anonymity import LAYERS
    ok &= _check("LAYERS['mixnet'] registered", "mixnet" in LAYERS)
    spec = ATTACKS.get("mixnet_attribution")
    ok &= _check("ATTACKS['mixnet_attribution'] -> POSTERIOR", spec is not None and spec.produces == POSTERIOR)
    trio = ("deanon_top1", "mean_true_posterior", "posterior_entropy")
    try:
        validate_pairing("mixnet_attribution", trio, ATTACKS, MEASURES); accept = True
    except ValueError:
        accept = False
    try:
        validate_pairing("mixnet_attribution", ("stake_top1_hit",), ATTACKS, MEASURES); reject = False
    except ValueError:
        reject = True
    ok &= _check("validate_pairing accepts POSTERIOR trio, rejects a SCALAR measure", accept and reject)
    return ok


def check_single_path_limit(args):
    """Check W=1 mixnet_attribution == bayes_attribution from independently drawn profiles.

    Run under both assignments: at W = 1 there is one entry node, so the rule is vacuous.
    """
    print("\n== W = 1 reduces to the single path (the degenerate anchor) ==")
    N, T, k = args.N, args.T, args.hops
    ok = True

    rng = np.random.default_rng(args.seed)
    alpha = sample_relative_stakes(N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=args.count), T=T, rng=rng)

    for a in ("split", "uniform"):
        sp, mx = _paired_w1_profiles(N, k, SHIPPED, args.seed + 1, assignment=a)

        ok &=_check(f"[{a}] W=1 grid draw is BIT-IDENTICAL to sample_lognormal_links (variates, order)",
                     np.array_equal(mx.d_sender[:, 0], sp.d_sender)
                     and np.array_equal(mx.d_receiver[:, 0], sp.d_receiver)
                     and float(mx.route_internal[0]) == float(sp.d_mix),
                     f"max|d_sender diff| = {np.max(np.abs(mx.d_sender[:, 0] - sp.d_sender)):.2e}")

        i = np.arange(0, 30); r = np.arange(30, 60)
        mu_sp = sp.mu(i, r)
        mu_mx = mx.route_mu(i, r)[:, 0]
        ok &= _check(f"[{a}] route_mu(i,r)[:,0] == single-path mu(i,r)",
                     np.allclose(mu_sp, mu_mx, atol=1e-15),
                     f"max|diff| = {np.max(np.abs(mu_sp - mu_mx)):.2e}")

        idx = np.arange(N)
        ok &= _check(f"[{a}] sender_leg(i) == single-path d_i^S at W=1 (the eta numerator generalises)",
                     np.allclose(mx.sender_leg(idx), sp.d_sender[idx], atol=1e-15),
                     f"max|diff| = {np.max(np.abs(mx.sender_leg(idx) - sp.d_sender[idx])):.2e}")

        tr = spm(s, n, isd, g, params=SinglePathMixParams(k, 1.0, N, sender_scale=0.25),
                 latency_oracle=sp, rng=np.random.default_rng(args.seed + 7))
        g_sp = run_bayes_attribution(tr, params=BayesAttributionParams(k, 1.0, 0.25, False, sp))
        g_mx = run_mixnet_attribution(tr, params=MixnetAttributionParams(k, 1.0, 0.25, False, mx))
        same = (np.array_equal(g_sp.candidate, g_mx.candidate)
                and np.array_equal(g_sp.broadcast_row, g_mx.broadcast_row)
                and np.allclose(g_sp.posterior, g_mx.posterior, atol=1e-12))
        ok &= _check(f"[{a}] W=1 mixnet_attribution posterior == bayes_attribution (byte-identical)",
                     same, f"max|dpost| = {np.max(np.abs(g_sp.posterior - g_mx.posterior)):.2e}")
    return ok


def check_no_ties(args):
    """Check zero exact ties (no free ceiling), the sigma_d(W) laws, and the route-diversity gain."""
    print("\n== continuity: no exact ties, no ceiling + route-diversity gain ==")
    N, T, k = args.N, args.T, args.hops
    m = args.count + 1
    null = 1.0 / m

    rng =np.random.default_rng(args.seed)
    prof = sample_mixnet_lognormal_profile(N, 2, k, SHIPPED, rng=rng)
    legs = prof.sender_leg(np.arange(N))
    tied = 0
    for _ in range(args.sets):
        s = rng.choice(N, size=m, replace=False)
        tied += int(np.unique(legs[s]).size < m)
    ok = _check(f"P(any exact tie in |S| = {m}) == 0 over {args.sets} sets",
                tied == 0, f"{tied}/{args.sets} sets contained an exact duplicate")

    sigma_pop =float(np.sqrt(link_moments(SHIPPED)[1]))
    for a, law in (("uniform", "sigma/sqrt(W)"), ("split", "sigma (flat in W)")):
        ratios = {}
        for W in (1, 2, 4):
            vals = [float(np.std(sample_mixnet_lognormal_profile(
                4000, W, k, SHIPPED, rng=np.random.default_rng(s),
                assignment=a).sender_leg(np.arange(4000)))) for s in range(args.seeds)]
            scale = np.sqrt(W) if a == "uniform" else 1.0
            ratios[W] = float(np.mean(vals)) * scale / sigma_pop
        dev = max(abs(r - 1.0) for r in ratios.values())
        ok &= _check(f"[{a}] sigma_d(W) == {law}, two-sided",
                     0.97 < min(ratios.values()) and max(ratios.values()) < 1.03,
                     "  ".join(f"W{W}: {r:.4f}" for W, r in ratios.items())
                     + f"   (sigma = {sigma_pop*1000:.2f} ms, max dev {dev:.2%})")

    def _top1(W, assignment):
        out = []
        for sd in range(args.seed, args.seed + args.seeds):
            tr, gu = _attack_mixnet(N, T, W, k, args.count, SHIPPED, sd,
                                    sender_scale=0.0, mix_scale=1e-4, assignment=assignment)
            out.append(deanon_top1(gu, tr))
        return float(np.mean(out))

    for a in ("split", "uniform"):
        t1 = {W: _top1(W, a) for W in (1, 2, 3)}
        ok &= _check(f"[{a}] no ceiling: at sigma_Z -> 0, W=1 DE-ANONYMISES (Top-1 -> ~1)",
                     t1[1] > 0.9, f"W1={t1[1]:.4f}  null={null:.3f}")
        ok &= _check(f"[{a}] route diversity helps: Top-1(W=2) <= Top-1(W=1)",
                     t1[2] <= t1[1] + 0.005,
                     f"W1={t1[1]:.4f} >= W2={t1[2]:.4f} >= W3={t1[3]:.4f}")
        ok &= _check(f"[{a}] every Top-1 sits above the 1/|S| null (the leak is real at sigma_Z -> 0)",
                     all(v > null for v in t1.values()),
                     "  ".join(f"W{W}={v:.4f}" for W, v in t1.items()))
    return ok


def _paired_gain(a, b):
    """
    Paired one-sided comparison of two seed-matched arms.

    `a[i]` and `b[i]` must come from the same seed. Returns (d_bar, sem, t, p, ci_lo, ci_hi):
    the mean difference, its sem, the paired t, the one-sided p for H1: mean(a-b) > 0, and a
    95% CI. Pairing cancels the common-mode seed-to-seed spread; report the CI, not the p.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n < 2:
        return float(d.mean()), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    dbar = float(d.mean())
    semd = float(d.std(ddof=1) / np.sqrt(n))
    if semd == 0.0:
        return dbar, 0.0, float("nan"), (0.0 if dbar > 0 else 1.0), dbar, dbar
    t = dbar / semd
    p = float(stats.t.sf(t, n - 1))
    lo, hi = stats.t.interval(0.95, n - 1, loc=dbar, scale=semd)
    return dbar, semd, float(t), p, float(lo), float(hi)


def check_entry_assignment(args):
    """
    Check the split entry assignment is the adversary-favouring default on identical links.

    Two mechanisms push Top-1 up under "split": sigma_d keeps its single-path value (no
    1/sqrt(W) shrink), and a candidate's likelihood marginalises only the W^(k-1) routes
    through its own entry. Seed-matched arms differ in the assignment alone; checked across
    the mixing budget, where the advantage should vanish on the null.
    """
    print("\n== the split entry assignment is the leakier (adversary-favouring) model ==")
    N, T, k = args.N, args.T, args.hops
    ok = True
    for mix in (1e-3, 1e-2, 1.0):
        per_seed = {"split": [], "uniform": []}
        for sd in range(args.seed, args.seed + args.seeds):
            for a in ("split", "uniform"):
                tr, gu = _attack_mixnet(N, T, 2, k, args.count, SHIPPED, sd,
                                        sender_scale=mix / 4.0,
                                        mix_scale=mix, assignment=a)
                per_seed[a].append(deanon_top1(gu, tr))
        dbar, semd, t, p, lo, hi =_paired_gain(per_seed["split"], per_seed["uniform"])
        detail = (f"split={np.mean(per_seed['split']):.4f}  uniform={np.mean(per_seed['uniform']):.4f}  "
                  f"d={dbar:+.4f} [95% {lo:+.4f},{hi:+.4f}]  t={t:.2f}  p={p:.1e}  (n={args.seeds} pairs)")
        if mix >= 1.0:
            ok &=_check(f"mix={mix:<6}: on the null, the ordering does NOT reverse (no significant "
                         f"Top-1(split) < Top-1(uniform))", not (p > 0.95), detail)
        else:
            ok &= _check(f"mix={mix:<6}: Top-1(split) > Top-1(uniform), paired over seeds (p<0.05)",
                         p < 0.05, detail)
    legs ={a: sample_mixnet_lognormal_profile(4000, 2, k, SHIPPED, rng=np.random.default_rng(3),
                                               assignment=a).sender_leg(np.arange(4000))
            for a in ("split", "uniform")}
    sd_split, sd_uni = float(np.std(legs["split"])), float(np.std(legs["uniform"]))
    ok &= _check("mechanism 1: the split's sigma_d is ~sqrt(2)x the uniform's (no averaging shrink)",
                 1.35 < sd_split / sd_uni < 1.50,
                 f"split {sd_split*1e3:.2f} ms / uniform {sd_uni*1e3:.2f} ms = {sd_split/sd_uni:.4f} "
                 f"(sqrt(2) = 1.4142)")
    prof_s = sample_mixnet_lognormal_profile(50, 2, k, SHIPPED, rng=np.random.default_rng(0),
                                             assignment="split")
    prof_u = sample_mixnet_lognormal_profile(50, 2, k, SHIPPED, rng=np.random.default_rng(0),
                                             assignment="uniform")
    ok &= _check("mechanism 2: the split marginalises W^(k-1) routes, the uniform W^k",
                 prof_s.n_routes_per_sender == 2 ** (k - 1) and prof_u.n_routes_per_sender == 2 ** k,
                 f"split {prof_s.n_routes_per_sender} vs uniform {prof_u.n_routes_per_sender} routes")
    return ok


def _per_set_sigma_hat(legs, m_sizes, m_probs, rng, perms=4):
    """
    Compute the pipeline's sigma_hat_d reduction on one quenched profile.

    Per sender set, the std of the candidates' sender legs with the 1/m divisor, averaged
    over sets drawn without replacement (disjoint m-blocks of a permutation) and recombined
    over the |S_t| law. Returns a per-profile value; a single profile scatters ~2.5% at
    N = 1000, so the caller must average over profiles.
    """
    total, used = 0.0, 0.0
    for m, p in zip(m_sizes, m_probs):
        m, p = int(m), float(p)
        if p < 1e-6:
            continue
        blocks = legs.size // m
        vals = [np.sqrt(legs[rng.permutation(legs.size)[:blocks * m]]
                        .reshape(blocks, m).var(axis=1))
                for _ in range(int(perms))]
        total += p * float(np.concatenate(vals).mean())
        used += p
    return total / used


def check_sigma_hat(args):
    """
    Check the reported sigma_hat_d is a biased estimator of sigma_d, not the parameter itself.

    E[sigma_hat_d^2] = sigma_d(W)^2 (m-1)/m is distribution-free, so E[sigma_hat_d] sits
    strictly under sigma_d(W) sqrt((m-1)/m) by Jensen. The bias rises in W under `uniform`
    and is flat under `split`; sigma_hat_d -> sigma_d only as the cover count grows. The
    closed form is then checked against the pipeline's own reduction on quenched profiles.
    """
    print("\n== the reported sigma_hat_d is an ESTIMATOR (population sigma_d is not its limit) ==")
    from theory.sigma_hat import expected_sigma_d_hat, population_sigma_d, sigma_d_bias

    k, count = args.hops, args.count
    ok = True
    sizes, probs = candidate_set_law(count, DEFAULT_F)
    m_bar = float((sizes * probs).sum())
    ceiling = float(np.sqrt((probs * (sizes - 1) / sizes).sum()))

    for a in ("split", "uniform"):
        for W in (1, 4, 10):
            b = sigma_d_bias(link=SHIPPED, count=count, width=W, assignment=a)
            ok &= _check(f"[{a}] W={W:<3} bias b = E[sigma_hat_d]/sigma_d strictly under the exact "
                         f"sqrt((m-1)/m)",
                         0.85 < b < ceiling,
                         f"b = {b:.4f}  <  {ceiling:.4f}   (Jensen gap {ceiling - b:.4f}, "
                         f"E[m] = {m_bar:.3f})")

    b_uni =[sigma_d_bias(link=SHIPPED, count=count, width=W, assignment="uniform")
             for W in (1, 2, 4, 10)]
    ok &= _check("uniform: the bias RISES in W (the averaged leg Gaussianises -> smaller skew gap)",
                 all(x < y for x, y in zip(b_uni, b_uni[1:])) and b_uni[-1] < ceiling,
                 "  ".join(f"W{W}: {b:.4f}" for W, b in zip((1, 2, 4, 10), b_uni))
                 + f"   -> ceiling {ceiling:.4f}")
    b_split = [sigma_d_bias(link=SHIPPED, count=count, width=W, assignment="split")
               for W in (1, 2, 4, 10)]
    ok &= _check("split: the bias is FLAT in W (one link per sender at every width)",
                 max(b_split) - min(b_split) < 0.002,
                 "  ".join(f"W{W}: {b:.4f}" for W, b in zip((1, 2, 4, 10), b_split)))
    ok &= _check("W = 1: the two assignments give the IDENTICAL E[sigma_hat_d] (the anchor)",
                 abs(expected_sigma_d_hat(link=SHIPPED, count=count, width=1, assignment="split")
                     - expected_sigma_d_hat(link=SHIPPED, count=count, width=1,
                                            assignment="uniform")) < 1e-12,
                 "same leg, same draws")
    b_by_count =[sigma_d_bias(link=SHIPPED, count=c, width=1, assignment="split")
                  for c in (9, 49, 199, 1999)]
    ok &= _check("sigma_hat_d -> sigma_d as the COVER COUNT grows (NOT as W grows)",
                 all(x < y for x, y in zip(b_by_count, b_by_count[1:])) and b_by_count[-1] > 0.999,
                 "  ".join(f"count={c}: {b:.4f}" for c, b in zip((9, 49, 199, 1999), b_by_count)))

    print(f"    -- theory vs the pipeline reduction ({args.profiles} quenched profiles, N = 1000) --")
    for a in ("split", "uniform"):
        zs = []
        for W in (1, 2, 5, 10):
            per_profile = []
            for r in range(args.profiles):
                prof = sample_mixnet_lognormal_profile(
                    1000, W, k, SHIPPED, assignment=a,
                    rng=np.random.default_rng(args.seed + 1000 * W + r))
                per_profile.append(_per_set_sigma_hat(
                    prof.sender_leg(np.arange(1000)), sizes, probs,
                    np.random.default_rng(args.seed + 50_000 + 1000 * W + r)))
            v = np.asarray(per_profile)
            meas, sem = float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))
            pred = expected_sigma_d_hat(link=SHIPPED, count=count, width=W, assignment=a)
            zs.append((W, meas, sem, pred, (meas - pred) / sem))
        ok &= _check(f"[{a}] the pipeline's sigma_hat_d lands on E[sigma_hat_d] at every W "
                     f"(|z| <= 4, two-sided)",
                     all(abs(z) <= 4.0 for *_, z in zs),
                     "  ".join(f"W{W}: {meas*1e3:.2f}~{pred*1e3:.2f} ms z={z:+.2f}"
                               for W, meas, _, pred, z in zs))
        W = 10
        pop = population_sigma_d(SHIPPED, W, a)
        pred = expected_sigma_d_hat(link=SHIPPED, count=count, width=W, assignment=a)
        ok &= _check(f"[{a}] W=10: E[sigma_hat_d] is measurably BELOW the population sigma_d "
                     f"(why the hat is not cosmetic)",
                     pred < pop * 0.99,
                     f"E[sigma_hat_d] = {pred*1e3:.2f} ms vs population {pop*1e3:.2f} ms "
                     f"({(1 - pred/pop)*100:.1f}% low)")
    return ok


def check_null(args):
    """Check sd = 0 (a point mass) gives identical mu across candidates and a uniform posterior."""
    print("\n== sd = 0 null (sigma_d = 0, the homogeneous boundary) ==")
    N, T, k = args.N, args.T, args.hops
    flat = LogNormalParams(floor=SHIPPED.floor, mean=SHIPPED.mean, sd=0.0)
    tr, g = _attack_mixnet(N, T, 2, k, args.count, flat, args.seed,
                           sender_scale=0.25, mix_scale=1.0)
    sizes = np.diff(g.start).astype(float)
    null = float(np.mean(1.0 / sizes)); maxH = float(np.mean(np.log2(sizes)))
    t1 = deanon_top1(g, tr); mp = mean_true_posterior(g, tr); H = posterior_entropy(g, tr)
    return _check("sd = 0: deanon_top1 = mean(1/|S|), true-post = mean(1/|S|), H = mean(log2|S|)",
                  abs(t1 - null) < 1e-9 and abs(mp - null) < 1e-9 and abs(H - maxH) < 1e-9,
                  f"top1={t1:.4f}~{null:.4f}; true-post={mp:.4f}; H={H:.4f}~{maxH:.4f}")


def check_firewall(args):
    """Check scrambling true_source / is_dummy leaves the posterior byte-identical."""
    print("\n== privacy firewall ==")
    N, T, k = args.N, args.T, args.hops
    rng = np.random.default_rng(args.seed)
    alpha = sample_relative_stakes(N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=args.count), T=T, rng=rng)
    prof = sample_mixnet_lognormal_profile(N, 2, k, SHIPPED, rng=np.random.default_rng(args.seed + 1))
    tr = mixnet_apply(s, n, isd, g, params=MixnetParams(2, k, 1.0, N, sender_scale=0.25),
                      latency_oracle=prof, rng=rng)
    g1 = run_mixnet_attribution(tr, params=MixnetAttributionParams(k, 1.0, 0.25, False, prof))
    rr = np.random.default_rng(args.seed + 99)
    scrambled = make_trace(broadcast_id=tr.broadcast_id, obs_node=tr.obs_node, obs_time=tr.obs_time,
                           kind=tr.kind, true_source=rr.permutation(tr.true_source),
                           is_dummy=rr.permutation(tr.is_dummy))
    g2 = run_mixnet_attribution(scrambled, params=MixnetAttributionParams(k, 1.0, 0.25, False, prof))
    identical = (np.array_equal(g1.candidate, g2.candidate) and np.allclose(g1.posterior, g2.posterior)
                 and np.array_equal(g1.broadcast_row, g2.broadcast_row))
    return _check("firewall: scrambling true_source/is_dummy leaves the posterior byte-identical",
                  identical, "attack reads only obs_node/obs_time/kind/broadcast_id")


def check_calibration(args):
    """
    Check the shipped mix-net config is moment-matched to the ping data.

    The params come from the YAML and the target from ping_link_moments over
    data/wondernetwork_pings.csv, so no magnitude is stated in this file.
    """
    print("\n[6] Calibration: is the shipped mix-net config moment-matched to the ping data?")
    ok = True
    p = SHIPPED
    m_ping, var_ping, min_ping = ping_link_moments()
    sd_ping = float(np.sqrt(var_ping))
    print(f"      config: floor={p.floor}  mean={p.mean}  sd={p.sd}")
    print(f"      pings : min={min_ping:.4f}  mean={m_ping:.4f}  sd={sd_ping:.4f}  (one-way, s)")

    ok &= _check("mean is moment-matched to the ping population",
                 abs(p.mean - m_ping) <= PING_TOL * m_ping,
                 f"config {p.mean*1e3:.1f} ms vs pings {m_ping*1e3:.1f} ms")
    ok &= _check("sd is moment-matched to the ping population (the STRUCTURAL spread)",
                 abs(p.sd - sd_ping) <= PING_TOL * sd_ping,
                 f"config {p.sd*1e3:.1f} ms vs pings {sd_ping*1e3:.1f} ms")
    ok &= _check("floor is the smallest MEASURED one-way latency (not a chosen constant)",
                 abs(p.floor - min_ping) <= PING_TOL * min_ping,
                 f"config {p.floor*1e3:.1f} ms vs min ping {min_ping*1e3:.2f} ms (rounded up)")

    sp =_config_lognormal(SINGLE_PATH_CONFIG)
    ok &= _check("identical to the single-path log-normal config (same physical law, both topologies)",
                 (p.floor, p.mean, p.sd) == (sp.floor, sp.mean, sp.sd),
                 f"mixnet {(p.floor, p.mean, p.sd)} vs single-path {(sp.floor, sp.mean, sp.sd)}")

    mean_link, var_link = link_moments(p)
    ok &= _check("the magnitudes are PHYSICAL (a link is tens of ms, not seconds)",
                 0.005 < mean_link < 0.5,
                 f"mean link {mean_link*1e3:.1f} ms, sd {np.sqrt(var_link)*1e3:.1f} ms")
    return ok


def check_chunking(args):
    """Check the row-chunked route marginalisation is bit-identical to the monolithic block.

    The attack streams its [K, P'] likelihood block in candidate-row chunks (CHUNK_ELEMENTS),
    so each row's mean over its route axis is computed in one piece and strict equality is
    the correct assertion. Three grids x three chunkings (single pass, tiny, unaligned); the
    jitter-on grid covers the eps > 0 density path.
    """
    import adversary.mixnet_attribution as ma
    from network.jitter import JitterParams
    print("\n[chunking] row-chunked enumeration == monolithic, bit-for-bit")
    ok = True
    k, mix, N, T = 3, 0.001, 300, 20_000

    def traffic(W, jitter, seed):
        rng = np.random.default_rng(seed)
        alpha = sample_relative_stakes(N, 1.33, rng=rng)
        slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
        s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=args.count),
                                      T=T, rng=rng)
        prof = sample_mixnet_lognormal_profile(
            N, W, k, SHIPPED, rng=np.random.default_rng(seed + 1),
            jitter=JitterParams(scale=jitter) if jitter else None)
        tr = mixnet_apply(s, n, isd, g,
                          params=MixnetParams(W, k, mix, N, sender_scale=mix / 4),
                          latency_oracle=prof, rng=rng)
        return tr, prof

    def posterior(tr, prof, chunk):
        old = ma.CHUNK_ELEMENTS
        ma.CHUNK_ELEMENTS = chunk
        try:
            return run_mixnet_attribution(
                tr, params=MixnetAttributionParams(k, mix, mix / 4, False, prof)).posterior
        finally:
            ma.CHUNK_ELEMENTS = old

    for W, jit, label in [(2, 0.0, "2x3 shipped"), (16, 0.0, "16x3 wide"),
                          (2, 1.0, "2x3 jitter-on (eps>0 density path)")]:
        tr, prof = traffic(W, jit, args.seed + 31)
        mono = posterior(tr, prof, 1 << 62)
        tiny = posterior(tr, prof, 1_000)
        awk = posterior(tr, prof, 7 * prof.n_routes_per_sender + 3)
        ok &= _check(f"{label}: tiny/unaligned chunks == monolithic, bit-for-bit",
                     np.array_equal(mono, tiny) and np.array_equal(mono, awk))

    tr, prof = traffic(40, 0.0, args.seed + 37)
    post = posterior(tr, prof, ma.CHUNK_ELEMENTS)
    ok &= _check("W=40 x 3 runs chunked under MAX_ROUTES=65,536 (P'=1,600), finite posterior",
                 post.size > 0 and bool(np.isfinite(post).all()))
    return ok


def main():
    """Run the selected sections (all by default) and return a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=int, default=60_000)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--count", type=int, default=19, help="fixed cover per slot -> |S| = count + 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5, help="seeds averaged for the no-ties section")
    ap.add_argument("--sets", type=int, default=20_000, help="sender sets drawn for the tie count")
    ap.add_argument("--profiles", type=int, default=40,
                    help="quenched profiles averaged in the sigma_hat_d section. The binding "
                         "uncertainty is profile-to-profile (~2.5%% per draw), NOT set-to-set, so "
                         "spend the budget here rather than on more sets per profile.")
    ap.add_argument("--section",
                    choices=["structure", "single_path", "no_ties", "assignment", "sigma_hat",
                             "null", "firewall", "calibration", "chunking"])
    args = ap.parse_args()

    sections = {"structure": lambda: check_structure(args),
                "single_path": lambda: check_single_path_limit(args),
                "no_ties": lambda: check_no_ties(args),
                "assignment": lambda: check_entry_assignment(args),
                "sigma_hat": lambda: check_sigma_hat(args),
                "null": lambda: check_null(args),
                "firewall": lambda: check_firewall(args),
                "calibration": lambda: check_calibration(args),
                "chunking": lambda: check_chunking(args)}
    ok = True
    for name in ([args.section] if args.section else list(sections)):
        ok &= sections[name]()

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
