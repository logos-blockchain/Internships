"""
Per-message link-jitter validation: the annealed congestion term d = L + X + eps_m with
eps_m ~ Exp redrawn per message (src/network/jitter.py), end to end.

Checks: the residual density f_{Z+E} against four references, on both sides of the
partial-fraction / series switch; jitter_moments and ac_path_links = k+1; jitter off
reproduces the pre-jitter numbers bit-identically; the layer draws the assumed law;
eta = sigma_d/sqrt(sigma_Z^2 + sigma_eps^2); the mis-specified adversary is confidently
wrong; anonymity from jitter alone at sigma_Z -> 0; and the latency cost face with its
exact limit ell -> 1 + (k+1)/E[ecc].

Run:  python tests/network-jitter-validation.py [--section pdf]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import integrate, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from consensus import DEFAULT_F, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, MixnetParams, SinglePathMixParams, inject_dummies  # noqa: E402
from anonymity.mixnet import apply as mixnet_apply  # noqa: E402
from anonymity.single_path_mix import (  # noqa: E402
    apply as spm_apply, gamma_sum_pdf, random_delay_pdf, residual_delay_pdf, residual_moments)
from network import JitterParams, LatencyProfileParams, LogNormalParams  # noqa: E402
from network.graph import mean_eccentricity_theory  # noqa: E402
from network.jitter import ac_path_links, jitter_moments  # noqa: E402
from network.latency_profile import sample_latency_profile  # noqa: E402
from network.mixnet_latency import sample_mixnet_lognormal_profile  # noqa: E402
from adversary import (  # noqa: E402
    BayesAttributionParams, MixnetAttributionParams, run_bayes_attribution,
    run_mixnet_attribution)
from adversary.gpa import observe_broadcasts  # noqa: E402
from metrics import deanon_top1, posterior_entropy  # noqa: E402
from experiments import load_experiment, run_once  # noqa: E402

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


def _build(N, T, k, count, seed, mix, eps, sender=None, width=None, assignment="split"):
    """consensus + cover + layer (single path or mix-net) -> (trace, profile, guess)."""
    sender = mix / 4.0 if sender is None else sender
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=count), T=T, rng=rng)
    jit = None if eps is None else JitterParams(scale=eps)
    if width is None:
        lp = LatencyProfileParams(lognormal=SHIPPED, jitter=jit)
        prof = sample_latency_profile(N, k, 0.3, lp, rng=np.random.default_rng(seed + 1))
        tr = spm_apply(s, n, isd, g, params=SinglePathMixParams(k, mix, N, sender_scale=sender),
                       latency_oracle=prof, rng=rng)
        guess = run_bayes_attribution(tr, params=BayesAttributionParams(k, mix, sender, False, prof))
    else:
        prof = sample_mixnet_lognormal_profile(N, width, k, SHIPPED,
                                               rng=np.random.default_rng(seed + 1),
                                               assignment=assignment, jitter=jit)
        tr = mixnet_apply(s, n, isd, g,
                          params=MixnetParams(width, k, mix, N, sender_scale=sender,
                                              entry_assignment=assignment),
                          latency_oracle=prof, rng=rng)
        guess = run_mixnet_attribution(tr, params=MixnetAttributionParams(k, mix, sender, False, prof))
    return tr, prof, guess


def check_pdf(args):
    """1. f_{Z+E} against four independent references + the near-equal-rate hazard."""
    print("== [pdf] the residual density f_{Z+E} ==")
    ok = True

    worst = 0.0
    for m, ss, ms in ((3, 0.25, 1.0), (1, 0.05, 0.4), (4, 2.0, 0.3), (3, 0.5, 0.5)):
        z = np.linspace(0, 40 * max(ss, ms), 4001)
        worst = max(worst, float(np.abs(random_delay_pdf(z, m, ss, ms)
                                        - gamma_sum_pdf(z, [(m, ms), (1, ss)])).max()))
    ok &= _check("2 blocks reproduce random_delay_pdf's closed form", worst < 1e-14,
                 f"max abs diff = {worst:.2e}")

    worst = 0.0
    for n, s in ((3, 1.0), (5, 0.02), (7, 0.3)):
        z = np.linspace(0, 20 * n * s, 2001)
        ref = stats.gamma.pdf(z, a=n, scale=s)
        worst = max(worst, float(np.abs(gamma_sum_pdf(z, [(n, s)]) - ref).max() / ref.max()))
    ok &= _check("1 block reproduces scipy's Gamma density", worst < 1e-12,
                 f"max rel diff = {worst:.2e}")

    def _pdf_at(x, m, ss, ms, js, nl):
        """Scalar view of the density, for scipy.integrate.quad."""
        return float(residual_delay_pdf(x, m, ss, ms, js, nl))

    for m, ss, ms, nl, js in ((3, 0.25, 1.0, 4, 0.05), (3, 0.0025, 0.01, 4, 0.05),
                              (3, 0.25, 1.0, 4, 1.0), (1, 0.5, 0.5, 2, 0.2)):
        mean_t, var_t = residual_moments(m, ss, ms, js, nl)
        hi = mean_t + 30 * np.sqrt(var_t)
        args_ = (m, ss, ms, js, nl)
        i0 = integrate.quad(_pdf_at, 0, hi, args=args_, limit=500)[0]
        i1 = integrate.quad(lambda x, *a: x * _pdf_at(x, *a), 0, hi, args=args_, limit=500)[0]
        i2 = integrate.quad(lambda x, *a: x * x * _pdf_at(x, *a), 0, hi, args=args_, limit=500)[0]
        ok &= _check(f"m={m} mix={ms} jitter={js}: integrates to 1, mean+var match the closed form",
                     abs(i0 - 1) < 1e-8 and abs(i1 - mean_t) < 1e-6 * max(mean_t, 1)
                     and abs(i2 - i1 ** 2 - var_t) < 1e-6 * max(var_t, 1),
                     f"int={i0:.10f}  mean={i1:.6f}/{mean_t:.6f}  var={i2-i1**2:.6e}/{var_t:.6e}")

    def _conv_ref(z0, m, ss, ms, nl, js):
        """Independent reference: numerically convolve f_Z with the jitter's Gamma(nl, js)."""
        return integrate.quad(
            lambda u: random_delay_pdf(np.array([z0 - u]), m, ss, ms)[0]
            * stats.gamma.pdf(u, a=nl, scale=js), 0, z0, limit=400)[0]

    m, ss, ms, nl = 3, 0.25, 1.0, 4
    worst_gap, worst_rel = None, 0.0
    for gap in (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 1e-3, 1e-5, 0.0,
                -1e-3, -0.01, -0.1, -0.5, -0.9, -0.99):
        js = ms * (1.0 + gap)
        mean_t, _ = residual_moments(m, ss, ms, js, nl)
        zs = np.linspace(0.2 * mean_t, 2.5 * mean_t, 5)
        ref = np.array([_conv_ref(z0, m, ss, ms, nl, js) for z0 in zs])
        rel = float(np.abs(residual_delay_pdf(zs, m, ss, ms, js, nl) - ref).max() / ref.max())
        if rel > worst_rel:
            worst_rel, worst_gap = rel, gap
    ok &= _check("accurate across the WHOLE rate plane, both sides of the algorithm switch",
                 worst_rel < 1e-10,
                 f"worst rel err = {worst_rel:.2e} at rate gap {worst_gap:+.1e} (17 gaps tested)")

    rng = np.random.default_rng(1)
    worst_p = 1.0
    for js in (0.05, 0.87, 3.0):
        mean_t, var_t = residual_moments(m, ss, ms, js, nl)
        hi = mean_t + 30 * np.sqrt(var_t)
        samp = rng.gamma(m, ms, 200_000) + rng.gamma(nl, js, 200_000) + rng.exponential(ss, 200_000)
        grid = np.linspace(0, hi, 40001)
        pdf = residual_delay_pdf(grid, m, ss, ms, js, nl)
        cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
        worst_p = min(worst_p, stats.kstest(
            samp, lambda x, g=grid, c=cdf: np.interp(x, g, c)).pvalue)
    ok &= _check("matches Monte-Carlo samples of Z + E (KS)", worst_p > 0.01,
                 f"worst of 3 cells: p = {worst_p:.3f}")

    z = np.linspace(0, 30, 3001)
    base = random_delay_pdf(z, 3, 0.25, 1.0)
    ok &= _check("jitter off delegates to random_delay_pdf, bit-identical",
                 np.array_equal(base, residual_delay_pdf(z, 3, 0.25, 1.0, 0.0, 4))
                 and np.array_equal(base, residual_delay_pdf(z, 3, 0.25, 1.0, 0.05, 0)))
    return ok


def check_moments(args):
    """2. jitter_moments' closed form vs sampling; ac_path_links = k+1, receiver-flag independent."""
    print("\n== [moments] the jitter's own closed form ==")
    ok = True
    rng = np.random.default_rng(0)
    for n, s in ((4, 0.05), (2, 1.0), (6, 0.003)):
        mean_t, var_t = jitter_moments(n, s)
        samp = rng.gamma(n, s, 400_000)
        ok &= _check(f"E[E] = n*scale, Var(E) = n*scale^2 at n={n}, scale={s}",
                     abs(samp.mean() - mean_t) < 5 * samp.std() / np.sqrt(samp.size)
                     and abs(samp.var() - var_t) < 0.02 * var_t,
                     f"mean {samp.mean():.6f}/{mean_t:.6f}, var {samp.var():.3e}/{var_t:.3e}")
    ok &= _check("sigma_eps = sqrt(n)*scale (the eta denominator's jitter term)",
                 abs(np.sqrt(jitter_moments(4, 0.05)[1]) - 2 * 0.05) < 1e-15)
    ok &= _check("ac_path_links(k) = k+1 -- a HOLD at R is not a LINK",
                 all(ac_path_links(k) == k + 1 for k in (1, 2, 3, 5)))
    return ok


def check_regression(args):
    """3. Jitter off (None or scale = 0) reproduces every pre-jitter number bit-identically."""
    print("\n== [regression] sigma_eps -> 0 goes back to our case ==")
    ok = True
    N, T, k = args.N, args.T, args.hops
    for label, width in (("single path", None), ("mix-net W=2", 2)):
        t0, p0, g0 = _build(N, T, k, args.count, args.seed, 1.0, None, width=width)
        t1, p1, g1 = _build(N, T, k, args.count, args.seed, 1.0, 0.0, width=width)
        ok &= _check(f"[{label}] jitter=None vs scale=0: trace + posterior bit-identical",
                     np.array_equal(t0.obs_time, t1.obs_time)
                     and np.array_equal(g0.posterior, g1.posterior)
                     and p0.jitter_scale == p1.jitter_scale == 0.0)
        ok &= _check(f"[{label}] every measure identical",
                     deanon_top1(g0, t0) == deanon_top1(g1, t1)
                     and posterior_entropy(g0, t0) == posterior_entropy(g1, t1),
                     f"top1 = {deanon_top1(g0, t0):.6f}")

    exp = load_experiment(CONFIG_PATH)
    cfg = replace(exp.base_cfg, T=args.T)
    a = run_once(cfg, rng=np.random.default_rng(args.seed))
    b = run_once(replace(cfg, latency=replace(cfg.latency, jitter=JitterParams(scale=0.0))),
                 rng=np.random.default_rng(args.seed))
    diffs = {kk: (a[kk], b[kk]) for kk in a if kk in b and a[kk] != b[kk] and a[kk] == a[kk]}
    ok &= _check("run_once on the committed config: identical with jitter=0", not diffs,
                 f"{len(a)} columns compared" if not diffs else f"differ: {diffs}")
    return ok


def check_generation(args):
    """4. The layer draws what the attack assumes: E[E] on the trace, Var on the residual."""
    print("\n== [generation] the layer's draw matches the assumed law ==")
    ok = True
    N, T, k, eps = args.N, args.T, args.hops, 0.05
    t0, _, _ = _build(N, T, k, args.count, args.seed, 1.0, None)
    t1, p1, _ = _build(N, T, k, args.count, args.seed, 1.0, eps)
    half = len(t0) // 2
    d0 = float(np.mean(t0.obs_time[half:] - t0.obs_time[:half]))
    d1 = float(np.mean(t1.obs_time[half:] - t1.obs_time[:half]))
    want = ac_path_links(k) * eps
    ok &= _check("mean AC delay rises by exactly E[E] = (k+1)*scale",
                 abs((d1 - d0) - want) < 0.02 * want, f"measured {d1-d0:.5f} vs {want:.5f}")
    ok &= _check("the profile carries the scale to the attack", p1.jitter_scale == eps)

    obs = observe_broadcasts(t1)
    true_sender = t1.true_source[obs.broadcast_row]
    resid = obs.y - p1.mu(true_sender, obs.receiver)
    mean_t, var_t = residual_moments(k, 1.0 / 4.0, 1.0, eps, ac_path_links(k))
    sem = np.sqrt(var_t / resid.size)
    ok &= _check("the realised residual has the mean the attack's density assumes",
                 abs(resid.mean() - mean_t) < 4 * sem,
                 f"{resid.mean():.5f} vs {mean_t:.5f} ({(resid.mean()-mean_t)/sem:+.2f} sem)")
    ok &= _check("...and its variance", abs(resid.var() - var_t) < 0.06 * var_t,
                 f"{resid.var():.5f} vs {var_t:.5f}")
    return ok


def check_eta(args):
    """5. eta = sigma_d/sqrt(sigma_Z^2 + sigma_eps^2), with sigma_d unchanged."""
    print("\n== [eta] the SNR redefinition ==")
    ok = True
    exp = load_experiment(CONFIG_PATH)
    cfg = replace(exp.base_cfg, T=args.T)
    k = cfg.layer_params.hops
    base = run_once(cfg, rng=np.random.default_rng(args.seed))
    for eps in (0.05, 0.5):
        r = run_once(replace(cfg, latency=replace(cfg.latency, jitter=JitterParams(scale=eps))),
                     rng=np.random.default_rng(args.seed))
        want_eps = np.sqrt(ac_path_links(k)) * eps
        want_eta = r["sigma_d"] / np.sqrt(r["sigma_Z"] ** 2 + want_eps ** 2)
        ok &= _check(f"scale={eps}: sigma_eps = sqrt(k+1)*scale",
                     abs(r["sigma_eps"] - want_eps) < 1e-12,
                     f"{r['sigma_eps']:.6f} vs {want_eps:.6f}")
        ok &= _check(f"scale={eps}: eta = sigma_d/sqrt(sigma_Z^2 + sigma_eps^2)",
                     abs(r["eta"] - want_eta) < 1e-12, f"{r['eta']:.6f}")
        ok &= _check(f"scale={eps}: sigma_Z is UNMOVED (the jitter is not a design knob)",
                     r["sigma_Z"] == base["sigma_Z"], f"{r['sigma_Z']:.4f}")
        ok &= _check(f"scale={eps}: sigma_d is UNMOVED (it stays the quenched signal)",
                     abs(r["sigma_d"] - base["sigma_d"]) < 1e-12,
                     f"{r['sigma_d']:.6f} vs {base['sigma_d']:.6f}")
    ok &= _check("jitter off: sigma_eps = 0 and eta is the pre-jitter value exactly",
                 base["sigma_eps"] == 0.0
                 and abs(base["eta"] - base["sigma_d"] / base["sigma_Z"]) < 1e-15)
    return ok


def check_misspec(args):
    """
    6. The old f_Z on jittered traffic is a mis-specified adversary: confidently wrong
    (sharper posterior, worse hit rate), so reporting it would overstate the anonymity.
    """
    print("\n== [misspec] the old f_Z against jittered traffic ==")
    ok = True
    N, T, k = args.N, args.T, args.hops
    cells = [(0.02, 0.02), (0.005, 0.005), (0.001, 0.02), (1e-5, 0.005)]
    print("     mix      jitter | top1 faithful  top1 mis-spec   ratio | H faithful  H mis-spec")
    for mix, eps in cells:
        tg, tb, hg, hb = [], [], [], []
        for s in range(args.seed, args.seed + args.seeds):
            tr, prof, good = _build(N, T, k, args.count, s, mix, eps)
            blind = run_bayes_attribution(
                tr, params=BayesAttributionParams(k, mix, mix / 4.0, False,
                                                  replace(prof, jitter_scale=0.0)))
            tg.append(deanon_top1(good, tr)); tb.append(deanon_top1(blind, tr))
            hg.append(posterior_entropy(good, tr)); hb.append(posterior_entropy(blind, tr))
        tg, tb, hg, hb = map(np.mean, (tg, tb, hg, hb))
        print(f"     {mix:<8g} {eps:<6g} | {tg:.5f}        {tb:.5f}       {tg/tb:5.2f}x"
              f" | {hg:.4f}     {hb:.4f}")
        ok &= _check(f"  mix={mix:g}, jitter={eps:g}: the faithful attack BEATS the mis-specified one",
                     tg > tb, f"Top-1 {tg:.5f} vs {tb:.5f} -- mis-spec understates the leak "
                              f"{tg/tb:.1f}x")
        ok &= _check("  ...while being LESS confident (mis-spec is confidently wrong)", hg > hb,
                     f"entropy {hg:.4f} vs {hb:.4f}")
    return ok


def check_extreme(args):
    """7. No mixing at all, yet the link jitter alone keeps the adversary confused."""
    print("\n== [extreme] sigma_Z -> 0, the jitter carrying the anonymity ==")
    ok = True
    exp = load_experiment(CONFIG_PATH)
    cfg = replace(exp.base_cfg, T=args.T)
    lp = replace(cfg.layer_params, mix_scale=1e-6, sender_scale=1e-6 / 4.0, sender_auto=False)
    cfg = replace(cfg, layer_params=lp)
    off = run_once(replace(cfg, latency=replace(cfg.latency, jitter=None)),
                   rng=np.random.default_rng(args.seed))
    on = run_once(replace(cfg, latency=replace(cfg.latency, jitter=JitterParams(scale=0.5))),
                  rng=np.random.default_rng(args.seed))
    ok &= _check("mixing off, jitter off: the sender is DE-ANONYMISED", off["deanon_top1"] > 0.9,
                 f"top1 = {off['deanon_top1']:.4f}, H = {off['posterior_entropy']:.4f}")
    ok &= _check("mixing off, jitter ON: back on the null, on nature's noise alone",
                 on["deanon_top1"] < 0.08 and on["posterior_entropy"] > 4.2,
                 f"top1 = {on['deanon_top1']:.4f} (null 0.0499), "
                 f"H = {on['posterior_entropy']:.4f} (ceiling 4.3243)")
    return ok


def check_cost(args):
    """
    8. The latency face: AC-path latency mu + Z + E rises by E[E] = (k+1)*scale, yet ell falls.

    The `latency` column is the leg mean(exit - entry), broadcast excluded, and
    ell = 1 + latency/E[D_br], so E[D_br] = latency/(ell - 1). Jitter adds (k+1)*eps to the
    numerator and E[ecc]*eps to the much smaller baseline E[D_br], so the ratio falls; as
    eps -> infinity both legs are jitter-dominated and ell -> 1 + (k+1)/E[ecc] = 1.8.
    """
    print("\n== [cost] jitter is not free, and ell moves the counter-intuitive way ==")
    ok = True
    exp = load_experiment(CONFIG_PATH)
    cfg = replace(exp.base_cfg, T=args.T)
    k = cfg.layer_params.hops
    base = run_once(cfg, rng=np.random.default_rng(args.seed))
    eps = 0.5
    r = run_once(replace(cfg, latency=replace(cfg.latency, jitter=JitterParams(scale=eps))),
                 rng=np.random.default_rng(args.seed))
    want = ac_path_links(k) * eps
    ok &= _check("AC-path latency mu+Z+E rises by exactly E[E] = (k+1)*scale (mean = sd: noise costs delay)",
                 abs((r["latency"] - base["latency"]) - want) < 0.02 * want,
                 f"+{r['latency']-base['latency']:.4f} vs +{want:.4f} s")

    d_br_base = base["latency"] / (base["latency_overhead"] - 1.0)
    d_br_jit = r["latency"] / (r["latency_overhead"] - 1.0)
    ecc = mean_eccentricity_theory(cfg.N, cfg.C)
    ok &= _check("E[D_br] rises by E[ecc]*scale -- the jitter loads the gossip graph too",
                 abs((d_br_jit - d_br_base) - ecc * eps) < 0.06 * ecc * eps,
                 f"{d_br_base:.4f} -> {d_br_jit:.4f} s (+{d_br_jit-d_br_base:.4f} vs "
                 f"E[ecc]*eps = {ecc*eps:.4f})")
    ok &= _check("ell FALLS -- because the BASELINE IS ~14x SMALLER, so the same order of "
                 "added delay is a far bigger RELATIVE change to it (NOT 'jitter is free')",
                 r["latency_overhead"] < base["latency_overhead"],
                 f"ell {base['latency_overhead']:.3f} -> {r['latency_overhead']:.3f} while the "
                 f"ABSOLUTE AC delay ROSE {base['latency']:.3f} -> {r['latency']:.3f} s; "
                 f"numerator x{r['latency']/base['latency']:.2f} vs denominator "
                 f"x{d_br_jit/d_br_base:.2f}")

    big = 50.0
    rb = run_once(replace(cfg, latency=replace(cfg.latency, jitter=JitterParams(scale=big))),
                  rng=np.random.default_rng(args.seed))
    limit = 1.0 + ac_path_links(k) / ecc
    ok &= _check("ell -> 1 + (k+1)/E[ecc] as the jitter dominates (the hop-count ratio)",
                 abs(rb["latency_overhead"] - limit) < 0.02,
                 f"ell({big} s) = {rb['latency_overhead']:.4f} vs limit {limit:.4f}; "
                 f"absolute AC delay is {rb['latency']:.1f} s -- cheap in RATIO, ruinous in seconds")
    return ok


SECTIONS = {"pdf": check_pdf, "moments": check_moments, "regression": check_regression,
            "generation": check_generation, "eta": check_eta, "misspec": check_misspec,
            "extreme": check_extreme, "cost": check_cost}


def main():
    """Run the selected sections and exit non-zero on any failure."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--T", type=int, default=40_000,
                    help="deliberately << the full epoch (tests override T down for speed)")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--count", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="seeds averaged in the misspec section")
    ap.add_argument("--section", choices=sorted(SECTIONS))
    args = ap.parse_args()

    names = [args.section] if args.section else list(SECTIONS)
    ok = all([SECTIONS[n](args) for n in names])          # list first: no short-circuit skipping
    print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
