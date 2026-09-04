"""
Shifted log-normal link-latency validation (src/network/lognormal_latency.py).

Checks: the (mean, sd) <-> (mu, sigma) moment map; the floor as a pure shift;
no exact candidate ties; the per-set sigma_d anchor E[var] = sigma^2 (m-1)/m and
its Jensen gap; the shipped config's calibration to the ping data; the end-to-end
attack (null at sd = 0, de-anonymisation at large eta); wiring of the two laws; and
the gossip broadcast E[D_br] under the same law. Also saves the calibration figure.

Run: python tests/network-lognormal-latency-validation.py [--reps N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotstyle  # noqa: E402

from consensus import DEFAULT_F, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import DummyParams, SinglePathMixParams, inject_dummies  # noqa: E402
from anonymity.single_path_mix import apply as spm  # noqa: E402
from network import LatencyProfileParams, sample_latency_profile  # noqa: E402
from network.latency import broadcast_latency_theory, lam_from_rho  # noqa: E402
from network.latency_profile import is_homogeneous, profile_broadcast_latency  # noqa: E402
from network.ping_data import ping_link_moments, ping_link_population  # noqa: E402
from network.lognormal_latency import (  # noqa: E402
    LogNormalParams, link_moments, lognormal_draw, lognormal_from_moments, lognormal_moments,
    per_set_sd_moments, sample_lognormal_links,
)
from adversary import BayesAttributionParams, run_bayes_attribution  # noqa: E402
from metrics import deanon_top1, posterior_entropy  # noqa: E402
from metrics.trilemma_cost import latency_overhead  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "single_path_mix_lognormal_attribution.yaml"
SET_SIZE = 20
UNIFORM_HIGH = 6.22
SENDER_HOLD = 0.25

PING_TOL = 0.02


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line and return the boolean."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def _per_set_sd(x, set_size):
    """Per-set sigma_d: std with the 1/m divisor over each set, averaged over sets."""
    return float(np.mean(np.std(x.reshape(-1, set_size), axis=1)))


def _config_lognormal():
    """Load the shipped config's `latency.lognormal` block as LogNormalParams."""
    import yaml
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return LogNormalParams(**yaml.safe_load(fh)["latency"]["lognormal"])


def check_moment_map(reps, seed):
    """Check the (mean, sd) <-> (mu, sigma) map round-trips and the draw hits its moments."""
    print("\n[1] The (mean, sd) <-> (mu, sigma) map")
    ok = True
    worst = 0.0
    for mean, sd in ((1.0, 0.5), (3.11, 1.895), (0.05, 0.027), (100.0, 5.0), (2.0, 20.0)):
        mu, sigma = lognormal_from_moments(mean, sd)
        m2, v2 = lognormal_moments(mu, sigma)
        worst = max(worst, abs(m2 - mean) / mean, abs(np.sqrt(v2) - sd) / max(sd, 1e-12))
    ok &= _check("round trip (mean, sd) -> (mu, sigma) -> (mean, sd) exact",
                 worst < 1e-12, f"max relative error {worst:.2e} over 5 parameter pairs")

    mu, sigma = lognormal_from_moments(3.11, 0.0)
    ok &= _check("sd = 0 -> sigma = 0, mu = log(mean) (the homogeneous limit)",
                 sigma == 0.0 and abs(np.exp(mu) - 3.11) < 1e-12, f"sigma={sigma}, exp(mu)={np.exp(mu):.6f}")

    p = LogNormalParams(floor=0.3, mean=3.11, sd=1.895)
    d_s, _, _ = sample_lognormal_links(reps, 3, p, rng=np.random.default_rng(seed))
    tm, tv = link_moments(p)
    em, esd = float(d_s.mean()), float(d_s.std())
    sem_m = esd / np.sqrt(reps)
    ok &= _check("drawn sample -> target mean", abs(em - tm) < 4 * sem_m,
                 f"target {tm:.4f}, drawn {em:.4f} +- {sem_m:.4f} (4-sem gate)")
    ok &= _check("drawn sample -> target sd", abs(esd - np.sqrt(tv)) / np.sqrt(tv) < 0.02,
                 f"target {np.sqrt(tv):.4f}, drawn {esd:.4f}  ({abs(esd-np.sqrt(tv))/np.sqrt(tv):.2%})")
    ok &= _check("link_moments is exact by construction (mean = floor + part mean)",
                 abs(tm - p.mean) < 1e-12 and abs(np.sqrt(tv) - p.sd) < 1e-12,
                 f"link_moments -> ({tm:.10f}, {np.sqrt(tv):.10f}) vs params ({p.mean}, {p.sd})")
    return ok


def check_floor(reps, seed):
    """Check the floor is a pure shift: it moves the mean and never the variance."""
    print("\n[2] The floor L -- 'to prevent very small values ... close to zero'")
    ok = True
    base = LogNormalParams(floor=0.0, mean=3.11, sd=1.895)
    lifted = LogNormalParams(floor=0.3, mean=3.41, sd=1.895)
    a, _, _ = sample_lognormal_links(reps, 3, base, rng=np.random.default_rng(seed))
    b, _, _ = sample_lognormal_links(reps, 3, lifted, rng=np.random.default_rng(seed))
    ok &= _check("same params + shift -> the draws differ by EXACTLY the floor",
                 float(np.max(np.abs((b - a) - 0.3))) < 1e-12,
                 f"max |(b - a) - 0.3| = {float(np.max(np.abs((b - a) - 0.3))):.2e}")
    ok &= _check("the floor does not move the variance (so not sigma_d / eta / the posterior)",
                 abs(float(a.std()) - float(b.std())) < 1e-12,
                 f"sd {a.std():.10f} vs {b.std():.10f}")
    p = LogNormalParams(floor=0.3, mean=3.11, sd=1.895)
    d_s, d_r, d_mix = sample_lognormal_links(reps, 3, p, rng=np.random.default_rng(seed + 1))
    ok &= _check("every drawn link respects the floor (strictly positive latencies)",
                 float(d_s.min()) > 0.3 and float(d_r.min()) > 0.3 and d_mix > 2 * 0.3,
                 f"min d_i^S = {float(d_s.min()):.4f}, min d_r^R = {float(d_r.min()):.4f} (floor 0.3)")
    return ok


def check_no_ties(reps, seed):
    """Check a continuous law never produces exact candidate ties within a sender set."""
    print("\n[3] No exact candidate ties (a continuous law grants no free Top-1 ceiling)")
    ok = True
    n_sets = max(reps // SET_SIZE, 1000)
    p = LogNormalParams(floor=0.3, mean=3.11, sd=1.895)

    d_s, _, _ = sample_lognormal_links(n_sets * SET_SIZE, 3, p, rng=np.random.default_rng(seed))
    sets = d_s.reshape(n_sets, SET_SIZE)
    dup_ln = float(np.mean([len(np.unique(row)) < SET_SIZE for row in sets]))

    ok &= _check("log-normal: P(any exact duplicate in a sender set of 20) = 0",
                 dup_ln == 0.0, f"{dup_ln:.6f} over {n_sets:,} sets")
    tied_ln = float(np.mean([(row == row[0]).sum() for row in sets]))
    ok &= _check("candidates sharing a given candidate's EXACT d_i^S: 1 (itself alone)",
                 abs(tied_ln - 1.0) < 1e-12, f"log-normal {tied_ln:.3f}")
    return ok


def check_per_set_sigma(reps, seed):
    """Check E[per-set var] = sigma^2 (m-1)/m and that sigma_d sits under it by Jensen."""
    print("\n[4] Per-set sigma_d: the exact (m-1)/m anchor and the Jensen gap it sits under")
    ok = True
    p = LogNormalParams(floor=0.3, mean=3.11, sd=1.895)
    n_sets = max(reps // SET_SIZE, 20_000)
    rng = np.random.default_rng(seed)

    d_s, _, _ = sample_lognormal_links(n_sets * SET_SIZE, 3, p, rng=rng)
    sets = d_s.reshape(n_sets, SET_SIZE)
    mean_var = float(np.mean(np.var(sets, axis=1)))
    anchor, sigma = per_set_sd_moments(SET_SIZE, p)
    exact_var = sigma ** 2 * (SET_SIZE - 1) / SET_SIZE
    ok &= _check("E[per-set variance] = sigma^2 (m-1)/m  (EXACT, distribution-free)",
                 abs(mean_var - exact_var) / exact_var < 0.02,
                 f"measured {mean_var:.4f} vs exact {exact_var:.4f} "
                 f"({abs(mean_var-exact_var)/exact_var:.2%}, {n_sets:,} sets)")

    sd_ln = _per_set_sd(d_s, SET_SIZE)
    ok &= _check("sigma_d = E[sqrt(.)] sits strictly UNDER the rms anchor (Jensen)",
                 sd_ln < anchor, f"sigma_d {sd_ln:.4f} < anchor {anchor:.4f} "
                                 f"(gap {1 - sd_ln/anchor:.2%})")

    u = np.random.default_rng(seed + 5).uniform(0.0, UNIFORM_HIGH, size=n_sets * SET_SIZE)
    sd_u = _per_set_sd(u, SET_SIZE)
    anchor_u = (UNIFORM_HIGH / np.sqrt(12)) * np.sqrt((SET_SIZE - 1) / SET_SIZE)
    gap_ln, gap_u = 1 - sd_ln / anchor, 1 - sd_u / anchor_u
    ok &= _check("the Jensen gap is LARGER for the skewed law than the uniform",
                 gap_ln > gap_u, f"log-normal {gap_ln:.2%} vs uniform {gap_u:.2%} "
                                 "-- why sd = 1.895 > 1.796 for the same sigma_d")
    return ok


def _ping_moments():
    """Return the ping population's one-way (mean, sd, min) via `ping_link_moments`."""
    mean, var, minimum = ping_link_moments()
    return mean, float(np.sqrt(var)), minimum


def check_config_claims(reps, seed):
    """Check the shipped config is moment-matched to the ping data and derive its eta."""
    print("\n[5] The shipped config: is it CALIBRATED to the pings, and what eta does it give?")
    ok = True
    p = _config_lognormal()
    m_ping, sd_ping, min_ping = _ping_moments()
    print(f"      config: floor={p.floor}  mean={p.mean}  sd={p.sd}")
    print(f"      pings : min={min_ping:.4f}  mean={m_ping:.4f}  sd={sd_ping:.4f}  (one-way, s)")

    ok &= _check("mean is moment-matched to the ping population",
                 abs(p.mean - m_ping) / m_ping < PING_TOL,
                 f"config {p.mean*1000:.1f} ms vs pings {m_ping*1000:.1f} ms")
    ok &= _check("sd is moment-matched to the ping population (the STRUCTURAL spread)",
                 abs(p.sd - sd_ping) / sd_ping < PING_TOL,
                 f"config {p.sd*1000:.1f} ms vs pings {sd_ping*1000:.1f} ms")
    ok &= _check("floor is the smallest MEASURED one-way latency (not a chosen constant)",
                 abs(p.floor - min_ping) / min_ping < PING_TOL,
                 f"config {p.floor*1000:.1f} ms vs min ping {min_ping*1000:.1f} ms")
    ok &= _check("the magnitudes are PHYSICAL (a link is tens of ms, not seconds)",
                 0.005 < p.mean < 0.5, f"mean link {p.mean*1000:.1f} ms")

    n_sets = max(reps // SET_SIZE, 40_000)
    d_s, _, _ = sample_lognormal_links(n_sets * SET_SIZE, 3, p, rng=np.random.default_rng(seed))
    sd_meas = _per_set_sd(d_s, SET_SIZE)
    sigma_Z = float(np.sqrt(SENDER_HOLD ** 2 + 3 * 1.0 ** 2))
    ok &= _check("per-set sigma_d sits just under the distribution sd (Jensen, skew-driven)",
                 0.8 * p.sd < sd_meas < p.sd,
                 f"sigma_d = {sd_meas*1000:.1f} ms vs sd {p.sd*1000:.1f} ms "
                 f"(gap {1 - sd_meas/p.sd:.1%}), {n_sets:,} sets")

    eta_op, eta_max = sd_meas / sigma_Z, sd_meas / SENDER_HOLD
    ok &= _check("eta = 1 is STRUCTURALLY UNREACHABLE (sigma_Z floors at the sender hold)",
                 eta_max < 1.0,
                 f"eta(op) = {eta_op:.4f}, eta_max = sigma_d/{SENDER_HOLD} = {eta_max:.4f} < 1 "
                 f"=> eta = 1 would need sigma_d >= {SENDER_HOLD*1000:.0f} ms")

    skew = float(np.mean(((d_s - d_s.mean()) / d_s.std()) ** 3))
    ok &= _check("the law is genuinely right-skewed", skew > 1.0,
                 f"skew {skew:.3f} (uniform = 0); median {np.median(d_s)*1000:.1f} ms "
                 f"< mean {d_s.mean()*1000:.1f} ms")
    return ok


def _attack(N, T, count, params, seed, mix_scale=1.0, sender_scale=0.25, k=3):
    """Run consensus + cover + single_path_mix + attack under the log-normal profile."""
    rng = np.random.default_rng(seed)
    alpha = sample_relative_stakes(N, 1.33, rng=rng)
    slots, nodes = simulate_events(alpha, f=DEFAULT_F, T=T, rng=rng)
    s, n, isd, g = inject_dummies(slots, nodes, N, params=DummyParams(count=count), T=T, rng=rng)
    prof = sample_latency_profile(N, k, 0.3, LatencyProfileParams(lognormal=params),
                                  rng=np.random.default_rng(seed + 1))
    tr = spm(s, n, isd, g, params=SinglePathMixParams(k, mix_scale, N, sender_scale=sender_scale),
             latency_oracle=prof, rng=rng)
    guess = run_bayes_attribution(
        tr, params=BayesAttributionParams(hops=k, mix_scale=mix_scale, sender_scale=sender_scale,
                                          receiver_delays=False, latency_profile=prof), rng=rng)
    return guess, tr


def check_end_to_end(seed):
    """Run the law through the full engine: exact null at sd = 0, de-anonymisation at large eta."""
    print("\n[6] End to end through the engine (consensus -> cover -> layer -> attack)")
    ok = True
    N, T, count = 400, 20_000, SET_SIZE - 1

    flat =LogNormalParams(floor=0.3, mean=3.11, sd=0.0)
    g, tr = _attack(N, T, count, flat, seed)
    top1, H = deanon_top1(g, tr), posterior_entropy(g, tr)
    sizes = np.diff(g.start).astype(float)
    null_top1, null_H = float(np.mean(1.0 / sizes)), float(np.mean(np.log2(sizes)))
    ok &= _check("sd = 0 (homogeneous) -> the EXACT per-broadcast null mean_t 1/|S_t| / log2|S_t|",
                 abs(top1 - null_top1) < 1e-9 and abs(H - null_H) < 1e-9,
                 f"P_Top1 = {top1:.9f} vs {null_top1:.9f}, H = {H:.9f} vs {null_H:.9f}  "
                 f"(|S_t| in {int(sizes.min())}-{int(sizes.max())}, "
                 f"{float(np.mean(sizes > SET_SIZE)):.2%} multi-leader slots)")

    op = LogNormalParams(floor=0.3, mean=3.11, sd=1.895)
    g, tr = _attack(N, T, count, op, seed)
    top1_op = deanon_top1(g, tr)
    ok &= _check("eta ~ 1 -> a leak above the null but far from de-anonymised",
                 1 / SET_SIZE < top1_op < 0.6, f"P_Top1 = {top1_op:.4f} (null {1/SET_SIZE:.3f})")

    g, tr = _attack(N, T, count, op, seed, mix_scale=0.001, sender_scale=0.001)
    top1_hi = deanon_top1(g, tr)
    ok &= _check("eta -> large (almost no mixing) -> de-anonymisation",
                 top1_hi > 0.8, f"P_Top1 = {top1_hi:.4f}")
    ok &= _check("monotone in the mixing budget (null < operating < no-mix)",
                 1 / SET_SIZE <= top1_op < top1_hi, f"{1/SET_SIZE:.3f} < {top1_op:.3f} < {top1_hi:.3f}")
    return ok


def check_wiring(seed):
    """Check the two laws are mutually exclusive and existing uniform draws are unchanged."""
    print("\n[7] Wiring: two mutually exclusive laws, and no existing draw moves")
    ok = True
    ln = LogNormalParams(floor=0.3, mean=3.11, sd=1.895)

    def raises(params):
        try:
            sample_latency_profile(50, 3, 0.3, params, rng=np.random.default_rng(0))
            return False
        except ValueError:
            return True

    ok &= _check("lognormal + uniform knobs -> rejected",
                 raises(LatencyProfileParams(lognormal=ln, sender_high=6.22)))
    ok &= _check("lognormal + mix_total -> rejected",
                 raises(LatencyProfileParams(lognormal=ln, mix_total=0.6)))

    for bad, why in (((0.5, 0.3, 1.0), "mean <= floor"), ((-1.0, 3.0, 1.0), "negative floor"),
                     ((0.0, 3.0, -1.0), "negative sd")):
        try:
            LogNormalParams(*bad)
            ok &= _check(f"rejects {why}", False, f"{bad} was accepted")
        except ValueError:
            ok &= _check(f"rejects {why}", True)

    uni =LatencyProfileParams(sender_low=0.0, sender_high=6.22, receiver_low=0.0,
                               receiver_high=6.22, mix_low=0.0, mix_high=6.22)
    a = sample_latency_profile(200, 3, 0.3, uni, rng=np.random.default_rng(seed))
    b = sample_latency_profile(200, 3, 0.3, LatencyProfileParams(**{**uni.__dict__}),
                               rng=np.random.default_rng(seed))
    ok &= _check("uniform draws byte-identical with the new field present (default None)",
                 np.array_equal(a.d_sender, b.d_sender) and a.d_mix == b.d_mix,
                 "max |diff| = 0")

    p1 = sample_latency_profile(200, 3, 0.3, LatencyProfileParams(lognormal=ln),
                                rng=np.random.default_rng(seed))
    p2 = sample_latency_profile(200, 7, 0.3, LatencyProfileParams(lognormal=ln),
                                rng=np.random.default_rng(seed))
    ok &= _check("changing k leaves d_sender (hence sigma_d / eta / the posterior) bit-identical",
                 np.array_equal(p1.d_sender, p2.d_sender) and p1.d_mix != p2.d_mix,
                 f"D_M {p1.d_mix:.4f} (k=3) vs {p2.d_mix:.4f} (k=7), d_sender unchanged")
    return ok


def check_broadcast(seed):
    """Check the gossip broadcast E[D_br] follows the same link law and moves only ell."""
    print("\n[8] The gossip broadcast E[D_br] follows the same link law")
    ok = True
    N, C, d = 1000, 8, 0.3

    def dbr(params, reps=6, sd=seed):
        rng = np.random.default_rng(sd)
        out = []
        for _ in range(reps):
            prof = sample_latency_profile(N, 1, d, params, rng=rng)
            out.append(profile_broadcast_latency(N, C=C, d=d, params=params, profile=prof, rng=rng))
        return float(np.mean(out))

    hom =LatencyProfileParams()
    exact = broadcast_latency_theory(N, C, d, lam_from_rho(0.0, d))
    ok &= _check("homogeneous -> the exact closed form d*E[ecc] (unchanged, no Dijkstra)",
                 is_homogeneous(hom) and abs(dbr(hom) - exact) < 1e-12,
                 f"E[D_br] = {dbr(hom):.6f} s = d*E[ecc] = {exact:.6f} s")

    p_ln =_config_lognormal()
    uni = LatencyProfileParams(sender_low=0.0, sender_high=UNIFORM_HIGH, receiver_low=0.0,
                               receiver_high=UNIFORM_HIGH, mix_low=0.0, mix_high=UNIFORM_HIGH)
    ecc = 5.0
    for label, params, edge_mean in (("log-normal (physical)", LatencyProfileParams(lognormal=p_ln),
                                      link_moments(p_ln)[0]),
                                     ("uniform (mechanism)", uni, UNIFORM_HIGH / 2)):
        val = dbr(params)
        ok &= _check(f"{label}: Dijkstra broadcast tracks its OWN edge scale",
                     edge_mean < val < edge_mean * ecc,
                     f"E[D_br] = {val:.3f} s, between one edge ({edge_mean:.3f}) and "
                     f"mean_edge x E[ecc] ({edge_mean*ecc:.3f}) -- routing saves "
                     f"{1 - val/(edge_mean*ecc):.0%}")

    m, s =3.11, 1.75
    half = s * np.sqrt(3.0)
    matched_u = LatencyProfileParams(sender_low=m - half, sender_high=m + half)
    matched_l = LatencyProfileParams(lognormal=LogNormalParams(floor=0.0, mean=m, sd=s))
    du, dl = dbr(matched_u), dbr(matched_l)
    ok &= _check("at IDENTICAL mean+sd the two laws give DIFFERENT E[D_br] (the lower tail governs)",
                 abs(dl - du) / du > 0.05,
                 f"uniform {du:.3f} s vs log-normal {dl:.3f} s = {abs(dl-du)/du:.1%} apart at "
                 f"mean {m}, sd {s} -- the uniform reaches down to {m-half:.2f}, "
                 f"the log-normal's mass piles at its median {np.exp(np.log(m) - np.log1p((s/m)**2)/2):.2f}")
    ok &= _check("...and BOTH sit below mean_edge x E[ecc] (paths route around expensive edges)",
                 du < m * ecc and dl < m * ecc,
                 f"naive {m*ecc:.2f} s vs uniform {du:.2f} ({du/(m*ecc):.2f}x), "
                 f"log-normal {dl:.2f} ({dl/(m*ecc):.2f}x)")

    g1, tr1 =_attack(400, 20_000, SET_SIZE - 1, LogNormalParams(floor=0.3, mean=3.11, sd=1.895), seed)
    ell_flat = latency_overhead(tr1, broadcast_mean=exact)
    ell_ln = latency_overhead(tr1, broadcast_mean=dbr(LatencyProfileParams(lognormal=p_ln)))
    t1_a, h_a = deanon_top1(g1, tr1), posterior_entropy(g1, tr1)
    t1_b, h_b = deanon_top1(g1, tr1), posterior_entropy(g1, tr1)
    ok &= _check("the broadcast law moves ell", abs(ell_flat - ell_ln) > 1.0,
                 f"same trace: ell = {ell_flat:.3f} at the flat D_br = {exact:.2f} s vs "
                 f"{ell_ln:.3f} under the log-normal gossip")
    ok &= _check("...and CANNOT move anonymity (the measures never see broadcast_mean)",
                 t1_a == t1_b and h_a == h_b,
                 f"P_Top1 = {t1_a:.6f}, H = {h_a:.6f} -- functions of (guess, trace) only")
    return ok


def save_calibration_figure(out_path, reps=400_000, seed=0, bins=22, density=True):
    """
    Plot the ping population against a sample from the shipped log-normal.

    Both series share bin edges and are normalised to the same total (a density in ms^-1,
    or probability mass per bin with density=False). The figure documents the mismatch of a
    unimodal law to a continent mixture; the matched moments, the unmatched median and the
    weighted KS distance are printed, not asserted.
    """
    from scipy.special import ndtr

    p = _config_lognormal()
    mu, sigma = lognormal_from_moments(p.mean - p.floor, p.sd)
    vals, wts, intra = ping_link_population()
    x = vals * 1e3
    sample = lognormal_draw(p)(np.random.default_rng(seed), reps) * 1e3

    edges = np.linspace(0.0, max(x.max(), np.percentile(sample, 99.5)) * 1.02, bins + 1)
    scale =1.0 / (edges[1] - edges[0]) if density else 1.0
    fig, ax = plt.subplots()
    ax.hist(x, bins=edges, weights=wts * scale, color=plt.cm.viridis(0.55),
            edgecolor="white", linewidth=0.6, label="WonderNetwork pings")
    ax.hist(sample, bins=edges, weights=np.full(reps, scale / reps), histtype="step",
            color="#d95f02", linewidth=2.4, label="shifted log-normal (sample)")

    m_ping = float(wts @ x)
    ax.axvline(m_ping, color="0.25", linestyle="--", linewidth=1.6,
               label=f"mean, both = {m_ping:.1f} ms (matched)")
    med_ping = float(np.sort(x)[np.searchsorted(np.cumsum(wts[np.argsort(x)]), 0.5)])
    med_model = (p.floor + np.exp(mu)) * 1e3
    ax.axvline(med_ping, color="0.25", linestyle=":", linewidth=1.6,
               label=f"median: pings {med_ping:.0f} vs model {med_model:.0f} ms")
    ax.axvline(med_model, color="#d95f02", linestyle=":", linewidth=1.6)

    ax.set_xlabel("one-way link latency (ms)")
    ax.set_ylabel(r"density (ms$^{-1}$)" if density else "probability mass per bin")
    ax.set_xlim(0.0, edges[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.legend(fontsize=12, framealpha=0.9)
    saved = plotstyle.save(fig, out_path)

    order = np.argsort(x)
    xs, ws = x[order], wts[order]
    cdf_data = np.cumsum(ws)
    cdf_model = ndtr((np.log(np.maximum(xs - p.floor * 1e3, 1e-12)) - (mu + np.log(1e3))) / sigma)
    ks = float(np.max(np.abs(np.column_stack([cdf_data - cdf_model,
                                              cdf_model - (cdf_data - ws)]))))
    sd_ping = float(np.sqrt(wts @ x ** 2 - m_ping ** 2))
    print(f"\nFigure written to {saved.relative_to(REPO_ROOT)}")
    print(f"  pings : mean {m_ping:6.2f}  sd {sd_ping:6.2f}  median {med_ping:6.2f} ms "
          f"({wts[intra].sum():.1%} of the mass intra-continental)")
    print(f"  model : mean {p.mean*1e3:6.2f}  sd {p.sd*1e3:6.2f}  median {med_model:6.2f} ms")
    print(f"  -> the two MATCHED moments agree to {abs(m_ping - p.mean*1e3)/m_ping:.2%} (mean) and "
          f"{abs(sd_ping - p.sd*1e3)/sd_ping:.2%} (sd);")
    print(f"     the median, which was NOT matched, is off by {abs(med_ping - med_model):.1f} ms "
          f"({abs(med_ping - med_model)/med_ping:.1%}); weighted KS distance {ks:.3f}.")
    print(f"     The right tail is unbounded where the data is not: {(sample > x.max()).mean():.1%} "
          f"of the model's mass sits above the\n     slowest measured pair ({x.max():.0f} ms), and "
          f"the model puts its mode where the mixture has its trough.")
    print("     A unimodal law cannot reproduce a continent mixture's modes. Quote it "
          "as a limitation, not a fit.")
    return True


def main():
    """Run all checks, save the calibration figure and return a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=400_000, help="Monte-Carlo draws (default 400k)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("SHIFTED LOG-NORMAL LINK LAW -- validation")
    print(f"reps = {args.reps:,}   seed = {args.seed}   |S| = {SET_SIZE}")
    print("=" * 78)

    results = [
        check_moment_map(args.reps, args.seed),
        check_floor(args.reps, args.seed),
        check_no_ties(args.reps, args.seed),
        check_per_set_sigma(args.reps, args.seed),
        check_config_claims(args.reps, args.seed),
        check_end_to_end(args.seed),
        check_wiring(args.seed),
        check_broadcast(args.seed),
    ]
    save_calibration_figure(REPO_ROOT / "results" / "figures" / "stage_2_figures"
                            / "lognormal_calibration_histogram.png", seed=args.seed)

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
