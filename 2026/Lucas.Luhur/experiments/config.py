"""
Experiment configuration: the Config dataclass and its YAML loader.

A Config describes one pipeline run (system params, the layer/attack plug names and
their params, the dummy params and the measures). load_experiment parses a YAML file
into an Experiment: a base Config plus sweep axes, each giving the dotted Config path
it sets (e.g. `layer_params.mix_scale`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from consensus import DEFAULT_F, DEFAULT_SHAPE, DEFAULT_T
from network import (DEFAULT_C, DEFAULT_D, DEFAULT_RHO, JitterParams,
                     LatencyProfileParams, LogNormalParams)
from anonymity import DummyParams, MixnetParams, SinglePathMixParams
from adversary import ATTACKS


@dataclass(frozen=True)
class Config:
    """
    One experiment cell; every axis a sweep varies is a field here.

    N -- number of nodes
    f -- active-slots coefficient (protocol constant)
    T -- observation window in slots (the protocol epoch, 6*floor(k/f) = 388_800)
    shape -- Pareto shape k (stake inequality)
    C -- gossip-graph connectivity
    d -- homogeneous per-link delay (T_e = d)
    rho -- link jitter ratio; 0 = noise-free channel, rho > 0 gives T_e = d + Exp(lam)
    latency -- quenched AC-path latency profile params (homogeneous default -> sigma_d = 0)
    gamma -- accuracy band for the stake measures: alpha_hat in [alpha (1 +/- gamma)]
    top_frac -- top-staker fraction x for the Jaccard measure
    dummy -- traffic-level dummy params
    layer / layer_params -- anonymity.LAYERS key and its params object (None -> layer default)
    attack / attack_params -- adversary.ATTACKS key and its params object
    measures -- metrics.MEASURES keys to score
    fast_counts -- sample sender-set counts directly (set_stake_inference only; no Trace)
    cover_runs -- inner cover runs M per quenched stake realisation (fast_counts path)
    """

    N: int = 1000
    f: float = DEFAULT_F
    T: int = DEFAULT_T
    shape: float = DEFAULT_SHAPE

    C: int = DEFAULT_C
    d: float = DEFAULT_D
    rho: float = DEFAULT_RHO

    latency: LatencyProfileParams = field(default_factory=LatencyProfileParams)

    gamma: float = 0.1
    top_frac: float = 0.01

    dummy: DummyParams = field(default_factory=DummyParams)

    layer: str = "single_path_mix"
    layer_params: object = None
    attack: str = "set_stake_inference"
    attack_params: object = None

    measures: tuple = ("stake_confidence", "stake_top_jaccard", "stake_top1_hit")

    fast_counts: bool = False

    cover_runs: int = 1


_LAYER_PARAMS = {"none": None, "single_path_mix": SinglePathMixParams, "mixnet": MixnetParams}
_ATTACK_PARAMS = {name: spec.params_cls for name, spec in ATTACKS.items()}

_ATTACK_KNOWS = {name: set(spec.knows) for name, spec in ATTACKS.items()}


@dataclass
class Experiment:
    """A parsed experiment: the base Config + its sweep + optional plotting spec(s)."""

    name: str
    base_cfg: Config
    axes: dict
    path_map: dict
    seed: int
    reps: int
    plot: dict | None
    plots: list | None = None


def load_experiment(path):
    """Parse a YAML config file into an Experiment."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    axes, path_map = {}, {}
    for name, spec in (d.get("sweep") or {}).items():
        axes[name] = list(spec["values"])
        path_map[name] = spec.get("path", spec.get("paths"))
    return Experiment(
        name=d.get("name", Path(path).stem),
        base_cfg=_build_config(d),
        axes=axes,
        path_map=path_map,
        seed=int(d.get("seed", 0)),
        reps=int(d.get("reps", 1)),
        plot=d.get("plot"),
        plots=d.get("plots"),
    )


def make_apply_cell(path_map):
    """
    Return apply_cell(base_cfg, cell), which sets each swept axis onto its Config path.

    The dotted path is descended with dataclasses.replace at each level.
    """
    def apply_cell(base_cfg, cell):
        cfg = base_cfg
        for name, value in cell.items():
            targets = path_map[name]
            for pth in ([targets] if isinstance(targets, str) else targets):
                cfg = _set_path(cfg, pth, value)
        return cfg

    return apply_cell


def _build_config(d):
    """Assemble the base Config from the parsed YAML (omitted fields -> Config defaults)."""
    kw = {}
    for fld in ("N", "f", "T", "shape", "C", "d", "rho", "gamma", "top_frac",
                "fast_counts", "cover_runs"):
        if d.get(fld) is not None:
            kw[fld] = d[fld]
    if d.get("dummy"):
        kw["dummy"] = DummyParams(**d["dummy"])
    if d.get("latency"):
        kw["latency"] = _build_latency(d["latency"])
    if "layer" in d:
        kw["layer"] = d["layer"]["name"]
        kw["layer_params"] = _build_params(d["layer"], _LAYER_PARAMS, "layer")
    if "attack" in d:
        name = d["attack"]["name"]
        restated = set(d["attack"].get("params") or {}) & _ATTACK_KNOWS.get(name, set())
        if restated:
            raise ValueError(
                f"attack {name!r} must not restate the GPA's public knowledge "
                f"{sorted(restated)} in attack.params -- run_once fills it from the system "
                "config, so declaring it here could silently disagree with the system. "
                "Set it on the system fields (N / f / T / dummy.p_s) and sweep it there.")
        kw["attack"] = name
        kw["attack_params"] = _build_params(d["attack"], _ATTACK_PARAMS, "attack")
    if "measures" in d:
        kw["measures"] = tuple(d["measures"])
    return Config(**kw)


def _build_latency(block):
    """
    Build the LatencyProfileParams from its YAML block.

    The uniform law uses flat keys (sender_low/high); the log-normal law
    (d = floor + LogNormal, given by moments) is the nested `lognormal:` sub-block and the
    per-message jitter term the nested `jitter:` sub-block. Sweeps reach them via the dotted
    paths latency.lognormal.<field> and latency.jitter.scale.
    """
    block = dict(block)
    lognormal = block.pop("lognormal", None)
    if lognormal is not None:
        lognormal = LogNormalParams(**dict(lognormal))
    jitter = block.pop("jitter", None)
    if jitter is not None:
        jitter = JitterParams(**dict(jitter))
    return LatencyProfileParams(lognormal=lognormal, jitter=jitter, **block)


def _build_params(block, registry, kind):
    """Build a params object for a layer/attack block, or None if it takes none."""
    name = block["name"]
    if name not in registry:
        raise ValueError(f"unknown {kind} {name!r}; known: {sorted(registry)}")
    cls = registry[name]
    params = block.get("params") or {}
    if cls is None:
        if params:
            raise ValueError(f"{kind} {name!r} takes no params, got {params!r}")
        return None
    return cls(**params)


def _set_path(obj, dotted, value):
    """Return a copy of a (nested) frozen dataclass with the dotted path set to value."""
    head, _, rest = dotted.partition(".")
    if not rest:
        return replace(obj, **{head: value})
    return replace(obj, **{head: _set_path(getattr(obj, head), rest, value)})
