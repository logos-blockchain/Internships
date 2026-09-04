"""
The mixnet layer: a stratified W-wide x k-deep mix grid generalising single_path_mix
(W = 1 recovers the single path). Each message takes a uniform-random route, so the
deterministic latency mu depends on the route while the intentional delay
Z = X_S + sum_j X_{M,j} is unchanged. The drawn route is recorded in `true_route`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import BROADCAST, ENTRY, EXIT, make_trace

try:
    from network.jitter import ac_path_links
    from network.mixnet_latency import ENTRY_ASSIGNMENTS
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from network.jitter import ac_path_links
    from network.mixnet_latency import ENTRY_ASSIGNMENTS


@dataclass(frozen=True)
class MixnetParams:
    """
    Parameters of the mixnet layer (the grid generalisation of SinglePathMixParams).

    width            -- W, mix nodes per layer; W = 1 reproduces single_path_mix, W^k routes otherwise
    hops             -- k, number of layers (mix hops on any route); Z has Gamma(k, mix_scale) holds
    mix_scale        -- mean per-hop mixing delay 1/lambda_M (slot units), the swept delay dial
    n_nodes          -- size of the destination universe; None -> from the latency oracle or the data
    sender_scale     -- mean sender hold 1/lambda_S (slot units), or "auto" = mix_scale / rho
    receiver_delays  -- if True the receiver also mixes (k+1 stages); default False (k stages)
    rho              -- target ratio lambda_S/lambda_M used when sender_scale = "auto"
    entry_assignment -- "split" (sender attached to one entry; W^(k-1) routes) or "uniform"
                        (entry re-picked per message; W^k routes); must match the latency profile
    sender_auto      -- derived flag: True once sender_scale was given as "auto"
    """

    width: int = 2
    hops: int = 3
    mix_scale: float = 1.0
    n_nodes: int | None = None
    sender_scale: float | str = 0.25
    receiver_delays: bool = False
    rho: float = 4.0
    entry_assignment: str = "split"
    sender_auto: bool = False

    def __post_init__(self):
        """Validate entry_assignment and resolve sender_scale = "auto" to mix_scale / rho."""
        if self.entry_assignment not in ENTRY_ASSIGNMENTS:
            raise ValueError(
                f"unknown entry_assignment {self.entry_assignment!r}; expected one of "
                f"{ENTRY_ASSIGNMENTS} (see src/network/mixnet_latency.py)")
        auto = self.sender_auto or isinstance(self.sender_scale, str)
        if isinstance(self.sender_scale, str) and self.sender_scale != "auto":
            raise ValueError(
                f"sender_scale must be a number or the string 'auto' (= mix_scale/rho), "
                f"got {self.sender_scale!r}")
        if auto:
            if not (self.rho > 0.0):
                raise ValueError(f"sender_scale='auto' needs rho > 0, got {self.rho!r}")
            object.__setattr__(self, "sender_auto", True)
            object.__setattr__(self, "sender_scale", float(self.mix_scale) / float(self.rho))


def apply(slots, nodes, is_dummy=None, group=None, *, params=None, latency_oracle=None, rng=None):
    """
    Route each candidate batch through the W x k mix grid on a uniform-random route per message.

    Same signature and Trace contract as single_path_mix.apply. latency_oracle is a
    MixnetLatencyProfile supplying mu_on_route(i, r, p) (None -> mu = 0). Returns a
    Trace with 2*M rows; the drawn route is recorded in true_route.
    """
    rng = np.random.default_rng(rng)
    params = params or MixnetParams()
    slots = np.asarray(slots, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)
    m = slots.size
    dummy = np.zeros(m, dtype=bool) if is_dummy is None else np.asarray(is_dummy, dtype=bool)
    grp = np.arange(m, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64)

    k = int(params.hops)
    W = int(params.width)
    if k < 1:
        raise ValueError(f"mixnet needs hops >= 1, got {k}")
    if W < 1:
        raise ValueError(f"mixnet needs width >= 1, got {W}")

    entry_time = slots.astype(np.float64)

    n_stages = k + 1 if params.receiver_delays else k
    Z = rng.gamma(shape=n_stages, scale=params.mix_scale, size=m)
    if params.sender_scale > 0.0:
        Z = Z + rng.exponential(params.sender_scale, size=m)

    if params.n_nodes is not None:
        N = int(params.n_nodes)
    elif latency_oracle is not None:
        N = int(latency_oracle.n_nodes)
    else:
        N = int(nodes.max()) + 1 if m else 1
    exit_node = rng.integers(0, N, size=m) if m else np.empty(0, dtype=np.int64)

    if latency_oracle is not None:
        oracle_rule = getattr(latency_oracle, "assignment", "uniform")
        if oracle_rule != params.entry_assignment:
            raise ValueError(
                f"entry_assignment mismatch: the layer was configured {params.entry_assignment!r} "
                f"but its latency profile was built {oracle_rule!r}. The profile drives both "
                f"the generated mu and the attack's, so these must agree -- build the profile with "
                f"assignment=params.entry_assignment (run_once does).")
        choice = (rng.integers(0, latency_oracle.n_routes_per_sender, size=m) if m
                  else np.empty(0, dtype=np.int64))
        route = latency_oracle.route_index(nodes, choice)
        mu = latency_oracle.mu_on_route(nodes, exit_node, route)
    else:
        route = np.full(m, -1, dtype=np.int64)
        mu = 0.0
    exit_time = entry_time + mu + Z

    jitter_scale = float(getattr(latency_oracle, "jitter_scale", 0.0) or 0.0)
    if jitter_scale > 0.0 and m:
        exit_time = exit_time + rng.gamma(shape=ac_path_links(k), scale=jitter_scale, size=m)

    exit_kind = np.where(dummy, EXIT, BROADCAST)
    return make_trace(
        broadcast_id=np.concatenate([grp, grp]),
        true_source=np.concatenate([nodes, nodes]),
        obs_node=np.concatenate([nodes, exit_node]),
        obs_time=np.concatenate([entry_time, exit_time]),
        kind=np.concatenate([np.full(m, ENTRY), exit_kind]),
        is_dummy=np.concatenate([dummy, dummy]),
        true_route=np.concatenate([route, route]),
    )
