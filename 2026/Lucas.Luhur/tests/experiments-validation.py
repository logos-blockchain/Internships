"""
Experiments-framework validation: the contract (registries + guess-type wall), the
compose function (run_once) and the YAML loader, in three sections (--section).

Checks: interface conformance and pluggability by name; run_once == manual compose,
reproducibility, layer-swap invariance of the stake measures, the fast_counts path and
its guard, quenched cover_runs averaging; and the loader against an independent parse
of single_path_mix_stake.yaml (key coverage, base config, sweep axes, apply_cell,
YAML == Python, a quick sweep).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from pipeline_contract import (  # noqa: E402
    AttackSpec, MeasureSpec, GUESS_TYPES, POSTERIOR, SCALAR, ScoreContext, validate_pairing,
)
from consensus import DEFAULT_F, sample_relative_stakes, simulate_events  # noqa: E402
from anonymity import LAYERS, DummyParams, SinglePathMixParams, delay_moments, inject_dummies  # noqa: E402
from adversary import ATTACKS, SetStakeInferenceParams  # noqa: E402
from metrics import MEASURES, bandwidth_overhead, latency_overhead, mean_latency  # noqa: E402
from network.jitter import ac_path_links  # noqa: E402
from network.latency import broadcast_latency_theory, lam_from_rho  # noqa: E402
from network.latency_profile import profile_broadcast_latency, sample_latency_profile  # noqa: E402
from experiments import Config, load_experiment, make_apply_cell, run_once, sweep  # noqa: E402
from experiments.pipeline import (  # noqa: E402
    _inform_gpa, run_once_stake_counts, run_realisation_stake_counts,
)

STAKE_YAML = REPO_ROOT / "experiments" / "configs" / "single_path_mix_stake.yaml"
STAKE_MEASURES = ("stake_confidence", "stake_top_jaccard", "stake_top1_hit")


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line and return the boolean."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def check_contract():
    """Interface conformance, the guess-type wall, and pluggability-by-name (plumbing only)."""
    print("== contract ==")
    N, T, shape, f = 1000, 20_000, 1.33, DEFAULT_F
    rng = np.random.default_rng(0)
    alpha = sample_relative_stakes(N, shape, rng=rng)
    slots, nodes = simulate_events(alpha, f=f, T=T, rng=rng)
    s, n, is_dummy, grp = inject_dummies(slots, nodes, N, params=DummyParams(count=1), T=T, rng=rng)
    tor_tr = LAYERS["single_path_mix"](s, n, is_dummy, grp, params=SinglePathMixParams(3, 5.0, N),
                                latency_oracle=None, rng=rng)
    print(f"  registries: attacks={sorted(ATTACKS)}, measures={sorted(MEASURES)}")
    ok = True

    attacks_ok = all(callable(a.run) and a.produces in GUESS_TYPES for a in ATTACKS.values())
    measures_ok = all(callable(m.score) and m.consumes in GUESS_TYPES for m in MEASURES.values())
    tags_ok = ATTACKS["set_stake_inference"].produces == SCALAR
    ok &= _check("interface conformance (run/produces, score/consumes, tags)",
                 attacks_ok and measures_ok and tags_ok)

    try:
        validate_pairing("set_stake_inference", list(STAKE_MEASURES), ATTACKS, MEASURES)
        accepts = True
    except ValueError:
        accepts = False
    fake_attacks = {"post": AttackSpec(run=lambda *a, **k: None, produces=POSTERIOR)}
    fake_measures = {"scal": MeasureSpec(score=lambda *a, **k: 0.0, consumes=SCALAR)}
    try:
        validate_pairing("post", ["scal"], fake_attacks, fake_measures)
        rejects = False
    except ValueError:
        rejects = True
    ok &= _check("guess-type wall (accept scalar x family-B; reject posterior x scalar)",
                 accepts and rejects)

    g_scalar = ATTACKS["set_stake_inference"].run(
        tor_tr, params=SetStakeInferenceParams(f=f, T=T, N=N))
    types_ok = isinstance(g_scalar, np.ndarray) and g_scalar.shape == (N,)
    ctx = ScoreContext(alpha=alpha, gamma=0.1, top_frac=0.01, f=f, T=T)
    scores = {name: spec.score(g_scalar, ctx)
              for name, spec in MEASURES.items() if spec.consumes == SCALAR}
    scores_ok = all(np.isfinite(v) for v in scores.values())
    ok &= _check("pluggability by name (attack -> declared type; scalar measures -> finite score)",
                 types_ok and scores_ok,
                 f"scalar guess {g_scalar.shape}, measures scored: {sorted(scores)}")
    return ok


def _stake_config(layer="single_path_mix"):
    """A small stake cell: single_path_mix (or none) + set_stake_inference + the family-B measures."""
    return Config(
        N=1000, T=20_000, shape=1.33,
        dummy=DummyParams(count=1),
        layer=layer,
        layer_params=SinglePathMixParams(hops=3, mix_scale=5.0, n_nodes=1000) if layer == "single_path_mix" else None,
        attack="set_stake_inference", attack_params=None,
        measures=STAKE_MEASURES,
    )


def check_pipeline():
    """run_once is a faithful, reproducible, type-safe composition -- the wiring, not the physics."""
    print("== pipeline ==")
    ok = True
    cfg = _stake_config("single_path_mix")

    res = run_once(cfg, rng=0)
    keys_ok = set(res) == set(cfg.measures) | {"bandwidth_overhead", "latency", "latency_overhead",
                                                "eta", "inv_eta", "sigma_d", "sigma_Z",
                                                "sigma_eps", "sigma_noise"}
    vals_ok = all(0.0 <= res[m] <= 1.0 for m in cfg.measures)
    ok &= _check("composes -> measure dict (single_path_mix + stake + family-B)",
                 keys_ok and vals_ok,
                 f"top1={res['stake_top1_hit']:.2f}, jaccard={res['stake_top_jaccard']:.2f}, "
                 f"conf={res['stake_confidence']:.3f}")

    r = np.random.default_rng(0)
    alpha = sample_relative_stakes(cfg.N, cfg.shape, rng=r)
    slots, nodes = simulate_events(alpha, f=cfg.f, T=cfg.T, rng=r)
    s, n, d, g = inject_dummies(slots, nodes, cfg.N, params=cfg.dummy, T=cfg.T, rng=r)
    profile = sample_latency_profile(cfg.N, cfg.layer_params.hops, cfg.d, cfg.latency, rng=r.spawn(1)[0])
    trace = LAYERS[cfg.layer](s, n, d, g, params=cfg.layer_params, latency_oracle=profile, rng=r)
    spec = ATTACKS[cfg.attack]
    guess = spec.run(trace, params=_inform_gpa(cfg, spec), rng=r)
    ctx = ScoreContext(alpha=alpha, gamma=cfg.gamma, top_frac=cfg.top_frac, f=cfg.f, T=cfg.T)
    manual = {m: MEASURES[m].score(guess, ctx) for m in cfg.measures}
    manual["bandwidth_overhead"] = bandwidth_overhead(trace)
    manual["latency"] = mean_latency(trace)
    bmean = profile_broadcast_latency(cfg.N, C=cfg.C, d=cfg.d, lam=lam_from_rho(cfg.rho, cfg.d),
                                      params=cfg.latency, profile=profile, rng=r.spawn(1)[0])
    ss = getattr(cfg.layer_params, "sender_scale", 0.0)
    manual["latency_overhead"] = latency_overhead(trace, broadcast_mean=bmean)
    n_stages = cfg.layer_params.hops + (1 if cfg.layer_params.receiver_delays else 0)
    _, var_Z = delay_moments(n_stages, ss, cfg.layer_params.mix_scale)
    sigma_d = float(np.std(profile.d_sender))
    sigma_Z = float(np.sqrt(var_Z))
    manual["sigma_d"] = sigma_d
    manual["sigma_Z"] = sigma_Z
    sigma_eps = 0.0 if cfg.latency is None or cfg.latency.jitter is None else float(
        np.sqrt(ac_path_links(cfg.layer_params.hops)) * cfg.latency.jitter.scale)
    sigma_noise = float(np.sqrt(sigma_Z ** 2 + sigma_eps ** 2))
    manual["sigma_eps"] = sigma_eps
    manual["sigma_noise"] = sigma_noise
    manual["eta"] = sigma_d / sigma_noise
    manual["inv_eta"] = sigma_noise / sigma_d
    ok &= _check("run_once == manual compose (same seed, exact)",
                 res == manual, "identical measure dicts")

    n_emissions = len(trace) // 2
    n_genuine = int((trace.is_entry & ~trace.is_dummy).sum())
    n_stages = cfg.layer_params.hops + (1 if cfg.layer_params.receiver_delays else 0)
    bo_ok = res["bandwidth_overhead"] == n_emissions / n_genuine
    theo_lat = cfg.d * (cfg.layer_params.hops + 1) + ss + n_stages * cfg.layer_params.mix_scale
    lat_ok = abs(res["latency"] - theo_lat) < 0.2
    ell_theo = 1.0 + res["latency"] / bmean
    ell_ok = abs(res["latency_overhead"] - ell_theo) < 1e-9
    ok &= _check("trilemma cost axes (beta=|S|/L; latency=d(k+1)+1/l_S+(k[+1])*scale; ell=1+AC/E[D_br])",
                 bo_ok and lat_ok and ell_ok,
                 f"beta={res['bandwidth_overhead']:.2f} (L={n_genuine}); latency={res['latency']:.3f} "
                 f"(stages={n_stages}); ell={res['latency_overhead']:.3f} (E[D_br]={bmean:.2f})")

    same = run_once(cfg, rng=0) == run_once(cfg, rng=0)
    diff = run_once(cfg, rng=1) != run_once(cfg, rng=0)
    ok &= _check("reproducibility (same seed -> identical; different seed -> different)",
                 same and diff)

    res_tor = run_once(_stake_config("single_path_mix"), rng=0)
    res_none = run_once(_stake_config("none"), rng=0)
    swap_ok = all(res_tor[m] == res_none[m] for m in STAKE_MEASURES)
    ok &= _check("plug in/out = change cfg.layer (stake measures layer-invariant none==single_path_mix)",
                 swap_ok,
                 f"top1 none={res_none['stake_top1_hit']:.2f} single_path_mix={res_tor['stake_top1_hit']:.2f}; "
                 f"jaccard none={res_none['stake_top_jaccard']:.3f} single_path_mix={res_tor['stake_top_jaccard']:.3f}")

    def _means(fast, p_s, reps=80, T=6000, seed=0):
        cfg_fc = Config(N=1000, T=T, dummy=DummyParams(p_s=p_s), layer="single_path_mix",
                        layer_params=SinglePathMixParams(), attack="set_stake_inference",
                        measures=STAKE_MEASURES, fast_counts=fast)
        acc = {m: np.empty(reps) for m in (*STAKE_MEASURES, "bandwidth_overhead")}
        for r, s in enumerate(np.random.SeedSequence(seed).spawn(reps)):
            sc = run_once(cfg_fc, rng=np.random.default_rng(s))
            for m in acc:
                acc[m][r] = sc[m]
        return acc

    fast_ok = True
    detail = []
    for p_s in (0.05, 0.3):
        full = _means(False, p_s, seed=1)
        fast = _means(True, p_s, seed=2)
        for m in (*STAKE_MEASURES, "bandwidth_overhead"):
            reps = full[m].size
            se = np.sqrt(full[m].var(ddof=1) / reps + fast[m].var(ddof=1) / reps)
            z = (fast[m].mean() - full[m].mean()) / se if se > 0 else 0.0
            fast_ok &= abs(z) < 4.0
            if m == "stake_top1_hit":
                detail.append(f"p_s={p_s}: top1 Trace={full[m].mean():.2f}/fast={fast[m].mean():.2f}")
    ok &= _check("fast_counts == Trace path (stake measures + beta agree within 4 SE over reps)",
                 fast_ok, "; ".join(detail))

    guarded = False
    try:
        run_once_stake_counts(replace(_stake_config("single_path_mix"),
                                      dummy=DummyParams(p_s=0.1), attack="bayes_attribution"),
                              rng=0)
    except ValueError:
        guarded = True
    ok &= _check("fast_counts guard (rejects a non-stake / Trace-reading attack)", guarded)

    ann_cfg = Config(N=1000, T=4_000, shape=1.33, dummy=DummyParams(p_s=0.03),
                     layer="single_path_mix", layer_params=SinglePathMixParams(),
                     attack="set_stake_inference", measures=STAKE_MEASURES, fast_counts=True)
    a1 = run_realisation_stake_counts(ann_cfg, rng=np.random.default_rng(7), cover_runs=1)
    r7 = np.random.default_rng(7)
    alpha7 = sample_relative_stakes(ann_cfg.N, ann_cfg.shape, rng=r7)
    b1 = run_once_stake_counts(ann_cfg, rng=r7.spawn(1)[0], alpha=alpha7)
    degen_ok = all(a1[m] == b1[m] for m in STAKE_MEASURES)
    R, M = 60, 8
    flat = {m: [] for m in STAKE_MEASURES}
    for s in np.random.SeedSequence(1).spawn(R):
        sc = run_once_stake_counts(ann_cfg, rng=np.random.default_rng(s))
        for m in STAKE_MEASURES:
            flat[m].append(sc[m])
    ann = {m: [] for m in STAKE_MEASURES}
    for s in np.random.SeedSequence(2).spawn(R):
        sc = run_realisation_stake_counts(ann_cfg, rng=np.random.default_rng(s), cover_runs=M)
        for m in STAKE_MEASURES:
            ann[m].append(sc[m])
    mean_ok = True
    for m in STAKE_MEASURES:
        se = np.sqrt(np.var(flat[m], ddof=1) / R + np.var(ann[m], ddof=1) / R)
        z = abs(np.mean(ann[m]) - np.mean(flat[m])) / se if se > 0 else 0.0
        mean_ok &= z < 4.0
    ok &= _check("quenched cover_runs (M=1 degeneracy + quenched mean == flat mean)",
                 degen_ok and mean_ok,
                 f"top1 flat={np.mean(flat['stake_top1_hit']):.2f} "
                 f"quenched={np.mean(ann['stake_top1_hit']):.2f} (M={M})")
    return ok


def _assemble_from_yaml(raw, **overrides):
    """
    Assemble a Config from the YAML's own values without the loader under test.

    Fields the YAML omits are left at the Config default, so retuning the config cannot
    stale this harness; only a structural change (new block, different params class) can.
    """
    kw = dict(
        gamma=raw["gamma"], top_frac=raw["top_frac"],
        dummy=DummyParams(**raw["dummy"]),
        layer=raw["layer"]["name"],
        layer_params=SinglePathMixParams(**raw["layer"]["params"]),
        attack=raw["attack"]["name"],
        attack_params=SetStakeInferenceParams(),
        measures=tuple(raw["measures"]),
        fast_counts=raw["fast_counts"],
        cover_runs=raw["cover_runs"],
    )
    kw.update(overrides)
    return Config(**kw)


def check_loader():
    """YAML == the Python Config: same base + sweep axes, apply_cell nests, a loaded cell runs same."""
    print("== loader ==")
    exp = load_experiment(STAKE_YAML)
    raw = yaml.safe_load(STAKE_YAML.read_text(encoding="utf-8"))
    ok = True

    consumed = {"gamma", "top_frac", "dummy", "layer", "attack", "measures",
                "fast_counts", "cover_runs"}
    non_config = {"name", "sweep", "seed", "reps", "plots"}
    unknown = set(raw) - consumed - non_config
    ok &= _check("key coverage (no top-level YAML key silently ignored)", not unknown,
                 f"consumed {len(consumed & set(raw))}, non-Config {len(non_config & set(raw))}"
                 + (f", UNKNOWN {sorted(unknown)}" if unknown else ""))

    ok &= _check("base config == Config assembled from the YAML (nested params too)",
                 exp.base_cfg == _assemble_from_yaml(raw),
                 f"cover_runs={exp.base_cfg.cover_runs}, fast_counts={exp.base_cfg.fast_counts}, "
                 f"p_s={exp.base_cfg.dummy.p_s}")

    axes_ok = (exp.axes == {k: v["values"] for k, v in raw["sweep"].items()}
               and exp.path_map == {k: v["path"] for k, v in raw["sweep"].items()}
               and exp.seed == raw["seed"] and exp.reps == raw["reps"])
    ok &= _check("sweep axes + path map + seed/reps", axes_ok,
                 f"axes={ {k: len(v) for k, v in exp.axes.items()} }, "
                 f"seed={exp.seed}, reps={exp.reps}")

    p_s_grid = raw["sweep"]["p_s"]["values"]
    probe = p_s_grid[len(p_s_grid) // 2]
    apply_cell = make_apply_cell(exp.path_map)
    cell_cfg = apply_cell(exp.base_cfg, {"p_s": probe})
    nested_ok = (cell_cfg.dummy.p_s == probe
                 and cell_cfg.dummy.count == exp.base_cfg.dummy.count)
    ok &= _check("apply_cell sets nested dummy.p_s", nested_ok, f"p_s -> {probe}")

    cfg_yaml = replace(apply_cell(exp.base_cfg, {"p_s": probe}), T=4_000)
    cfg_py = _assemble_from_yaml(raw, T=4_000, dummy=DummyParams(**{**raw["dummy"], "p_s": probe}))

    def _same(a, b):
        if a.keys() != b.keys():
            return False
        for k in a:
            va, vb = a[k], b[k]
            both_nan = isinstance(va, float) and isinstance(vb, float) and np.isnan(va) and np.isnan(vb)
            if not (both_nan or va == vb):
                return False
        return True

    same = _same(run_once(cfg_yaml, rng=0), run_once(cfg_py, rng=0))
    ok &= _check("YAML cell == Python Config (run_once identical)", same,
                 f"top1={run_once(cfg_py, rng=0)['stake_top1_hit']:.2f}")

    small = replace(exp.base_cfg, T=2_000)
    df = sweep(small, exp.axes, apply_cell, seed=0, reps=1)
    cols_ok = {"p_s", "shape", "rep", "stake_top1_hit"}.issubset(df.columns)
    n_cells = int(np.prod([len(v) for v in exp.axes.values()]))
    ok &= _check("loaded experiment sweeps (DataFrame shape + columns)",
                 cols_ok and len(df) == n_cells,
                 f"{len(df)} rows = {' x '.join(str(len(v)) for v in exp.axes.values())} "
                 f"({' x '.join(exp.axes)}); cols={list(df.columns)}")
    return ok


SECTIONS = {"contract": check_contract, "pipeline": check_pipeline, "loader": check_loader}


def main():
    """Run the selected sections and return a process exit code."""
    ap = argparse.ArgumentParser(description="Experiments-framework wiring validation (3 sections).")
    ap.add_argument("--section", choices=list(SECTIONS),
                    help="run only one section (default: all three)")
    args = ap.parse_args()

    ok = True
    for i, name in enumerate([args.section] if args.section else list(SECTIONS)):
        if i:
            print()
        ok &= SECTIONS[name]()

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
