"""
Experiment execution engine: run_once composes one cell end to end, each component
selected by name from its registry (consensus -> inject_dummies -> LAYERS[layer] ->
ATTACKS[attack].run -> MEASURES[m].score), and sweep runs it over a grid of config
axes with decorrelated per-cell RNGs, returning a tidy pandas DataFrame.
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd

from consensus import (
    sample_relative_stakes,
    simulate_events,
    wins_per_node_from_events,
)
from anonymity import LAYERS, MixnetParams, SinglePathMixParams, delay_moments, inject_dummies
from adversary import ATTACKS
from adversary.mixnet_attribution import CHUNK_ELEMENTS
from adversary.stake_inference import estimate_stake_from_sets
from metrics import MEASURES, bandwidth_overhead, latency_overhead, mean_latency
from network.jitter import ac_path_links, jitter_moments
from network.latency import broadcast_latency_theory, lam_from_rho
from network.latency_profile import profile_broadcast_latency, sample_latency_profile
from network.mixnet_latency import sample_mixnet_lognormal_profile
from pipeline_contract import validate_pairing, SCALAR, ScoreContext


def gpa_knowledge(cfg, profile=None):
    """
    Return the GPA's public knowledge as {name: value}, resolved from the system config.

    Covers the protocol constants (f, T, N, p_s), the AC delay params (hops, mix_scale,
    sender_scale, receiver_delays) and the quenched latency profile (mu). Under the
    fixed-`count` cover control the equivalent p_s is 1 - (1 - 1/N)^count.
    """
    p_s = cfg.dummy.p_s
    if p_s is None:
        p_s = 1.0 - (1.0 - 1.0 / cfg.N) ** max(int(cfg.dummy.count), 0)
    pub = {"f": cfg.f, "p_s": float(p_s), "T": cfg.T, "N": cfg.N}
    lp = cfg.layer_params
    if lp is None and cfg.layer == "single_path_mix":
        lp = SinglePathMixParams()
    if lp is None and cfg.layer == "mixnet":
        lp = MixnetParams()
    if lp is not None:
        pub.update(hops=getattr(lp, "hops", None), mix_scale=getattr(lp, "mix_scale", None),
                   sender_scale=getattr(lp, "sender_scale", None),
                   receiver_delays=getattr(lp, "receiver_delays", None))
    if profile is not None:
        pub["latency_profile"] = profile
    return pub


def _inform_gpa(cfg, spec, profile=None):
    """Fill the attack's declared public knowledge (spec.knows) from the system config + profile."""
    if not spec.knows:
        return cfg.attack_params
    params = cfg.attack_params if cfg.attack_params is not None else spec.params_cls()
    public = gpa_knowledge(cfg, profile)
    return replace(params, **{k: public[k] for k in spec.knows})


def run_once(cfg, rng=None):
    """
    Compose one experiment cell and return {measure_name: value}.

    Runs consensus -> traffic dummies -> the anonymisation layer -> the attack -> the
    measures, each plugged by name from its registry. The layer's latency_oracle is a
    quenched LatencyProfile answering mu(i, r) by array indexing. One rng is threaded
    through every stage, so run_once(cfg, rng=seed) is bit-reproducible.
    """
    if cfg.fast_counts:
        if getattr(cfg, "cover_runs", 1) > 1:
            return run_realisation_stake_counts(cfg, rng, cfg.cover_runs)
        return run_once_stake_counts(cfg, rng)

    rng = np.random.default_rng(rng)
    validate_pairing(cfg.attack, cfg.measures, ATTACKS, MEASURES)

    alpha = sample_relative_stakes(cfg.N, cfg.shape, rng=rng)
    slots, nodes = simulate_events(alpha, f=cfg.f, T=cfg.T, rng=rng)

    slots, nodes, is_dummy, group = inject_dummies(
        slots, nodes, cfg.N, params=cfg.dummy, T=cfg.T, rng=rng)

    lp = cfg.layer_params
    if lp is None and cfg.layer == "single_path_mix":
        lp = SinglePathMixParams()
    if lp is None and cfg.layer == "mixnet":
        lp = MixnetParams()
    if cfg.layer == "mixnet":
        if cfg.latency is None or cfg.latency.lognormal is None:
            raise ValueError(
                "the mixnet layer requires the log-normal latency law (set latency.lognormal) -- "
                "the grid is per-link, and the log-normal is the project's only structural "
                "law. The uniform knobs have no grid form.")
        profile = sample_mixnet_lognormal_profile(cfg.N, lp.width, lp.hops,
                                                  cfg.latency.lognormal, rng=rng.spawn(1)[0],
                                                  assignment=lp.entry_assignment,
                                                  jitter=cfg.latency.jitter)
    else:
        hops = getattr(lp, "hops", 1)
        profile = sample_latency_profile(cfg.N, hops, cfg.d, cfg.latency, rng=rng.spawn(1)[0])

    trace = LAYERS[cfg.layer](slots, nodes, is_dummy, group,
                              params=cfg.layer_params, latency_oracle=profile, rng=rng)

    spec = ATTACKS[cfg.attack]
    guess = spec.run(trace, params=_inform_gpa(cfg, spec, profile), rng=rng)

    if ATTACKS[cfg.attack].produces == SCALAR:
        ctx = ScoreContext(alpha=alpha, gamma=cfg.gamma, top_frac=cfg.top_frac,
                           f=cfg.f, T=cfg.T)
        scores = {m: MEASURES[m].score(guess, ctx) for m in cfg.measures}
    else:
        scores = {m: MEASURES[m].score(guess, trace) for m in cfg.measures}

    scores["bandwidth_overhead"] = bandwidth_overhead(trace)
    scores["latency"] = mean_latency(trace)
    broadcast_mean = profile_broadcast_latency(
        cfg.N, C=cfg.C, d=cfg.d, lam=lam_from_rho(cfg.rho, cfg.d), params=cfg.latency,
        profile=profile, rng=rng.spawn(1)[0])
    scores["latency_overhead"] = latency_overhead(trace, broadcast_mean=broadcast_mean)

    if lp is not None and getattr(lp, "mix_scale", None) is not None:
        n_stages = int(lp.hops) + (1 if lp.receiver_delays else 0)
        _, var_Z = delay_moments(n_stages, lp.sender_scale, lp.mix_scale)
        sigma_Z = float(np.sqrt(var_Z))
        eps_scale = 0.0 if cfg.latency is None or cfg.latency.jitter is None else float(
            cfg.latency.jitter.scale)
        _, var_eps = jitter_moments(ac_path_links(int(lp.hops)), eps_scale) if eps_scale > 0 else (0.0, 0.0)
        sigma_eps = float(np.sqrt(var_eps))
        sigma_noise = float(np.sqrt(sigma_Z ** 2 + sigma_eps ** 2))
        cand = getattr(guess, "candidate", None)
        if cand is not None and cand.size:
            vals = (profile.sender_leg(cand) if cfg.layer == "mixnet"
                    else profile.d_sender[cand])
            counts = np.diff(guess.start).astype(float)
            seg_mean = np.add.reduceat(vals, guess.start[:-1]) / counts
            seg_var = np.add.reduceat(vals * vals, guess.start[:-1]) / counts - seg_mean ** 2
            sigma_d = float(np.mean(np.sqrt(np.maximum(seg_var, 0.0))))
        else:
            sigma_d = float(np.std(profile.sender_leg(np.arange(cfg.N)) if cfg.layer == "mixnet"
                                   else profile.d_sender))
        scores["sigma_d"] = sigma_d
        scores["sigma_Z"] = sigma_Z
        scores["sigma_eps"] = sigma_eps
        scores["sigma_noise"] = sigma_noise
        scores["eta"] = sigma_d / sigma_noise if sigma_noise > 0 else float("inf")
        scores["inv_eta"] = sigma_noise / sigma_d if sigma_d > 0 else float("inf")
    else:
        scores["eta"] = float("nan")
        scores["inv_eta"] = float("nan")
        scores["sigma_eps"] = float("nan")
        scores["sigma_noise"] = float("nan")
    return scores


def run_once_stake_counts(cfg, rng=None, *, alpha=None):
    """
    Count-based fast path for the timing-independent sender-set stake attack.

    Returns the same {measure: value} dict as run_once without materialising the cover
    Trace: per node n_i = wins_i + Binomial(T - wins_i, p_s), the same law as the Bernoulli
    cover draw. Valid only for set_stake_inference; the latency/eta columns are nan.
    `alpha` optionally fixes the (quenched) relative-stake vector instead of drawing one.
    Matches run_once in distribution, not byte-for-byte (different RNG consumption).
    """
    if cfg.attack != "set_stake_inference":
        raise ValueError(
            f"fast_counts is only valid for the timing-independent 'set_stake_inference' "
            f"attack (got {cfg.attack!r}) -- every other attack reads the Trace.")
    p_s = cfg.dummy.p_s
    if p_s is None:
        raise ValueError("fast_counts needs dummy.p_s (the per-node cover probability); "
                         "the fixed-`count` cover is not supported on this path.")
    rng = np.random.default_rng(rng)
    validate_pairing(cfg.attack, cfg.measures, ATTACKS, MEASURES)

    if alpha is None:
        alpha = sample_relative_stakes(cfg.N, cfg.shape, rng=rng)
    _, nodes = simulate_events(alpha, f=cfg.f, T=cfg.T, rng=rng)
    wins = wins_per_node_from_events(nodes, cfg.N)
    n_genuine = int(nodes.size)

    cover = rng.binomial(np.maximum(cfg.T - wins, 0), p_s)
    entry_counts = wins + cover

    alpha_hat = estimate_stake_from_sets(entry_counts, cfg.T, p_s=p_s, f=cfg.f)

    ctx = ScoreContext(alpha=alpha, gamma=cfg.gamma, top_frac=cfg.top_frac, f=cfg.f, T=cfg.T)
    scores = {m: MEASURES[m].score(alpha_hat, ctx) for m in cfg.measures}

    n_emissions = int(entry_counts.sum())
    scores["bandwidth_overhead"] = n_emissions / n_genuine if n_genuine else float("nan")

    for key in ("latency", "latency_overhead", "eta", "inv_eta", "sigma_d", "sigma_Z",
                "sigma_eps", "sigma_noise"):
        scores[key] = float("nan")
    return scores


def run_realisation_stake_counts(cfg, rng, cover_runs):
    """
    Run one quenched stake realisation, thermally averaged over `cover_runs` (= M) cover runs.

    Draws one relative-stake vector, runs the count fast path M times with it held fixed
    while the lottery and cover are redrawn, and returns the per-measure mean. A sweep rep
    then indexes a stake realisation, so the spread across reps is the quenched-disorder
    error bar. The latency/eta columns stay nan.
    """
    rng = np.random.default_rng(rng)
    alpha = sample_relative_stakes(cfg.N, cfg.shape, rng=rng)
    runs = [run_once_stake_counts(cfg, rng=sub, alpha=alpha)
            for sub in rng.spawn(int(cover_runs))]
    keys = runs[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = float("nan") if all(v != v for v in vals) else float(np.mean(vals))
    return out


class _NullBar:
    """No-op progress bar (used when progress is off or tqdm is unavailable)."""

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _make_bar(enabled, total):
    """A tqdm bar if enabled and installed; otherwise the silent _NullBar."""
    if not enabled:
        return _NullBar()
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc="sweep", unit="run")
    except ImportError:
        return _NullBar()


_BYTES_PER_TRACE_ROW = 34        # 3 int64 + float64 + int8 + bool
_PEAK_FACTOR = 2.3
_PROC_BASELINE_BYTES = 0.30e9
_CHUNK_BYTES_PER_ELEM = 65
_ROUTE_ARRAYS = 4


def _resolve_cpu_cap(n_jobs):
    """n_jobs -> worker ceiling. 1 = serial; <=0 or None = auto (cores - 2, capped 16)."""
    cores = os.cpu_count() or 2
    if n_jobs is None or n_jobs <= 0:
        return max(1, min(cores - 2, 16))
    return max(1, int(n_jobs))


def _total_ram_bytes():
    """
    Return the RAM the pool may plan against: min(physical total, currently available).

    Available rather than total, since the schedule must fit the memory that exists at run
    time. Falls back to 8 GB if psutil is missing. Affects only the worker count, never a result.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(min(vm.total, vm.available))
    except Exception:
        return 8 * 1024 ** 3


def _est_peak_bytes(cfg):
    """
    Estimate the peak working-set bytes of one run_once(cfg).

    Two terms: the Trace (2 rows per emission, ~p_s*N*T cover emissions) and the mix-net
    attack's [K, P] route arrays (K candidate rows x P = W^k or W^(k-1) routes), the latter
    capped at the attack's CHUNK_ELEMENTS streaming bound. The count fast path returns 0.
    """
    if getattr(cfg, "fast_counts", False):
        return 0.0
    p_s = cfg.dummy.p_s
    if p_s is None:
        p_s = 1.0 - (1.0 - 1.0 / cfg.N) ** max(int(cfg.dummy.count), 0)
    cover_rows = max(0.0, float(p_s)) * cfg.N * cfg.T
    trace_rows = 2.0 * cover_rows
    trace_bytes = trace_rows * _BYTES_PER_TRACE_ROW * _PEAK_FACTOR

    route_bytes = 0.0
    lp = getattr(cfg, "layer_params", None)
    if getattr(cfg, "layer", None) == "mixnet" and isinstance(lp, MixnetParams):
        k = int(lp.hops)
        P = int(lp.width) ** (k - 1 if lp.entry_assignment == "split" else k)
        lam = float(-np.log1p(-cfg.f))
        cover = cfg.N * cfg.dummy.p_s if cfg.dummy.p_s is not None else cfg.dummy.count
        K = cfg.T * lam * (float(cover) + 1.0 + lam)
        route_bytes = min(_ROUTE_ARRAYS * K * P * 8.0,
                          _CHUNK_BYTES_PER_ELEM * float(CHUNK_ELEMENTS))
    return trace_bytes + route_bytes


def _cell_workers(cfg, cpu_cap, mem_budget_bytes, reps):
    """Workers for one cell = min(cpu ceiling, memory ceiling, reps). Always >= 1."""
    per_worker = _est_peak_bytes(cfg) + _PROC_BASELINE_BYTES
    mem_cap = int(mem_budget_bytes // per_worker) if per_worker > 0 else cpu_cap
    return max(1, min(cpu_cap, max(1, mem_cap), reps))


def _cell_worker(task):
    """Top-level (picklable) worker: run one cell/rep from its (cfg, SeedSequence)."""
    cfg, seed_seq = task
    return run_once(cfg, rng=np.random.default_rng(seed_seq))


def _run_cell_parallel(cfg, cell_seeds, workers, bar):
    """Run one cell's reps across a fresh pool of `workers`; results in rep order."""
    results = [None] * len(cell_seeds)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_cell_worker, (cfg, s)): r for r, s in enumerate(cell_seeds)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            bar.update(1)
    return results


def sweep(base_cfg, axes, apply_cell, *, seed=0, reps=1, progress=False,
          n_jobs=1, mem_fraction=0.6):
    """
    Run run_once over the Cartesian product of axes and return a tidy DataFrame.

    axes -- {axis_name: iterable_of_values}; apply_cell -- (base_cfg, cell_dict) -> Config.
    Each (cell, rep) run gets a decorrelated child RNG via SeedSequence(seed).spawn.
    n_jobs: 1 = serial, <=0/None = auto (cores - 2, capped at 16); mem_fraction bounds each
    cell's worker count by estimated peak memory. Parallelism changes no result: rows are
    one per (cell, rep), cell-major, with the axis columns, rep and one column per measure.
    """
    names = list(axes)
    cells = list(itertools.product(*(list(axes[n]) for n in names)))
    child_seeds = np.random.SeedSequence(seed).spawn(len(cells) * reps)
    bar = _make_bar(progress, len(cells) * reps)

    cpu_cap = _resolve_cpu_cap(n_jobs)
    mem_budget = mem_fraction * _total_ram_bytes()

    rows = []
    for c, values in enumerate(cells):
        cell = dict(zip(names, values))
        cfg = apply_cell(base_cfg, cell)
        cell_seeds = child_seeds[c * reps:(c + 1) * reps]
        workers = _cell_workers(cfg, cpu_cap, mem_budget, reps) if cpu_cap > 1 else 1
        if workers > 1:
            results = _run_cell_parallel(cfg, cell_seeds, workers, bar)
        else:
            results = []
            for s in cell_seeds:
                results.append(run_once(cfg, rng=np.random.default_rng(s)))
                bar.update(1)
        for r, result in enumerate(results):
            rows.append({**cell, "rep": r, **result})
    bar.close()
    return pd.DataFrame(rows)
