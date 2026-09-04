"""
Quenched mix-net AC-path latency profile: the shifted log-normal law over a W-wide x k-deep
stratified grid of mix nodes, generalising latency_profile.py from one route to W^k routes.

Every grid link is its own i.i.d. draw from d = floor + LogNormal, and a route p has
mu(i, p, r) = d_sender[i, entry_p] + D_int_p + d_receiver[r, exit_p]; the GPA marginalises the
likelihood over the routes a candidate's message can take. The sender -> entry `assignment` is
"split" (a sender is pinned to one entry node, W^(k-1) routes, as in Nym/Loopix) or "uniform"
(route drawn per message, W^k routes). W = 1 reproduces the single-path profile bit-identically.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

try:
    from .lognormal_latency import lognormal_draw
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from network.lognormal_latency import lognormal_draw

MAX_ROUTES = 65_536

ENTRY_ASSIGNMENTS = ("split", "uniform")


@dataclass(frozen=True)
class MixnetLatencyProfile:
    """
    A frozen realisation of the deterministic mix-net AC-path latencies (quenched log-normal links).

    d_sender   -- [N, W] float: sender i -> layer-1 node w; the only leg that reaches the posterior.
    d_receiver -- [N, W] float: layer-k node w -> receiver r; cancels in the residual.
    route_entry / route_exit -- [P] int: the layer-1 / layer-k node of each of the P = W^k routes.
    route_internal -- [P] float: D_int_p, the k-1 inter-layer draws summed along each route (s).
    width / hops -- W, k, the grid shape.
    grid_internal -- [k-1, W, W] float: the raw inter-layer draws (provenance); None at k = 1.
    assignment -- "split" (default) or "uniform"; carried on the profile so the layer and the
                  attack read one rule.
    entry_of   -- [N] int: each sender's entry node ("split" only).
    routes_by_entry -- [W, W^(k-1)] int: global route indices per entry node ("split" only).
    jitter_scale -- 1/lambda_eps of the annealed per-message jitter; 0.0 = off.
    """

    d_sender: np.ndarray
    d_receiver: np.ndarray
    route_entry: np.ndarray
    route_exit: np.ndarray
    route_internal: np.ndarray
    width: int
    hops: int
    grid_internal: np.ndarray | None = None
    assignment: str = "split"
    entry_of: np.ndarray | None = None
    routes_by_entry: np.ndarray | None = None
    jitter_scale: float = 0.0

    @property
    def n_routes(self):
        """P = W^k, the grid's total enumerated routes (both assignments materialise all of them)."""
        return int(self.route_internal.size)

    @property
    def n_routes_per_sender(self):
        """
        Routes one sender's message can take: W^k under "uniform", W^(k-1) under "split".

        The size of the likelihood's mixture; at W = 1 both give a single route.
        """
        if self.assignment == "split":
            return int(self.routes_by_entry.shape[1])
        return self.n_routes

    @property
    def n_nodes(self):
        """N -- the destination universe R is drawn from (the layer reads it here)."""
        return int(self.d_sender.shape[0])

    def route_index(self, sender, choice):
        """
        Map a per-sender route choice j in [0, n_routes_per_sender) to a global route index.

        Identity under "uniform"; under "split" it selects within the sender's own entry's
        row of routes_by_entry.
        """
        j = np.asarray(choice, dtype=np.int64)
        if self.assignment != "split":
            return j
        return self.routes_by_entry[self.entry_of[np.asarray(sender, dtype=np.int64)], j]

    def route_mu(self, sender, receiver):
        """
        mu(i, p, r) for every candidate x the routes its messages can take -> [len(sender), P'].

        mu(i, p, r) = d_sender[i, entry_p] + D_int_p + d_receiver[r, exit_p]. Under "uniform"
        P' = W^k (the same route block for every candidate); under "split" P' = W^(k-1), each
        candidate's own entry's routes. The attack averages f_Z(y - mu) over the columns.
        """
        i = np.asarray(sender, dtype=np.int64)
        r = np.asarray(receiver, dtype=np.int64)
        if self.assignment == "split":
            routes = self.routes_by_entry[self.entry_of[i]]
            entry = self.d_sender[i, self.entry_of[i]][:, None]
            exit_ = self.d_receiver[r[:, None], self.route_exit[routes]]
            return entry + self.route_internal[routes] + exit_
        entry = self.d_sender[i[:, None], self.route_entry[None, :]]
        exit_ = self.d_receiver[r[:, None], self.route_exit[None, :]]
        return entry + self.route_internal[None, :] + exit_

    def sender_leg(self, sender):
        """
        The per-candidate discriminating sender->entry latency -> [len(sender)].

        The mix-net analogue of d_i^S and the numerator of eta = sigma_d/sigma_Z. Under "split"
        it is the single link d_sender[i, entry_of[i]]; under "uniform" it is the route-averaged
        leg (1/P) sum_p d_sender[i, entry_p], whose spread is the single path's over sqrt(W).
        At W = 1 both reduce to d_sender[i, 0]. A descriptive scale for the sweep axis only: the
        attack marginalises f_Z over routes (route_mu) rather than averaging mu.
        """
        i = np.asarray(sender, dtype=np.int64)
        if self.assignment == "split":
            return self.d_sender[i, self.entry_of[i]]
        return self.d_sender[i[:, None], self.route_entry[None, :]].mean(axis=1)

    def mu_on_route(self, sender, receiver, route):
        """
        mu(i, p_i, r_i) for a message that actually took global route p_i -> [len(sender)].

        The layer's generator reads it here; the observed broadcast time is y = t + mu_on_route + Z.
        """
        i = np.asarray(sender, dtype=np.int64)
        r = np.asarray(receiver, dtype=np.int64)
        p = np.asarray(route, dtype=np.int64)
        return (self.d_sender[i, self.route_entry[p]]
                + self.route_internal[p]
                + self.d_receiver[r, self.route_exit[p]])


def _route_choices(k, W):
    """
    Enumerate the W^k routes as node-index tuples (one choice per layer), in lexicographic order.

    The single enumeration the layer's route draw and the attack's marginalisation both index into.
    """
    return list(itertools.product(range(int(W)), repeat=int(k)))


def sample_mixnet_lognormal_profile(N, width, hops, params=None, rng=None, assignment="split",
                                    jitter=None):
    """
    Draw a quenched MixnetLatencyProfile for N nodes on a W x k stratified mix grid.

    Every grid link (sender->layer-1 [N, W], the k-1 inter-layer meshes [W, W], layer-k->receiver
    [N, W]) is one i.i.d. draw from the shared shifted log-normal via lognormal_draw.
    `assignment` is "split" (entry_of = i mod W, deterministic, no RNG) or "uniform"; `jitter`
    is a JitterParams whose scale is carried on the profile, not drawn. Draw order is sender,
    receiver, internal mesh, so at W = 1 the profile is bit-identical to sample_lognormal_links.
    Rejects W^k > MAX_ROUTES, W < 1, k < 1 and an unknown assignment.
    """
    rng = np.random.default_rng(rng)
    W, k = int(width), int(hops)
    if W < 1:
        raise ValueError(f"mixnet needs width >= 1, got {W}")
    if k < 1:
        raise ValueError(f"mixnet needs hops >= 1, got {k}")
    if assignment not in ENTRY_ASSIGNMENTS:
        raise ValueError(f"unknown sender->entry assignment {assignment!r}; expected one of "
                         f"{ENTRY_ASSIGNMENTS} (see network/mixnet_latency.py)")
    P = W ** k
    if P > MAX_ROUTES:
        raise ValueError(
            f"mixnet grid W={W}, k={k} has W^k = {P} routes > MAX_ROUTES={MAX_ROUTES}; the attack "
            "enumerates every route, so keep the grid small.")

    draw = lognormal_draw(params)
    d_sender = draw(rng, (int(N), W))
    d_receiver = draw(rng, (int(N), W))
    internal = draw(rng, (k - 1, W, W)) if k > 1 else None

    choices = _route_choices(k, W)
    route_entry = np.array([c[0] for c in choices], dtype=np.int64)
    route_exit = np.array([c[-1] for c in choices], dtype=np.int64)
    route_internal = np.array(
        [sum(internal[j, c[j], c[j + 1]] for j in range(k - 1)) if k > 1 else 0.0
         for c in choices], dtype=np.float64)

    entry_of = routes_by_entry = None
    if assignment == "split":
        entry_of = (np.arange(int(N), dtype=np.int64) % W)
        routes_by_entry = np.stack(
            [np.nonzero(route_entry == w)[0].astype(np.int64) for w in range(W)])

    return MixnetLatencyProfile(
        d_sender=d_sender, d_receiver=d_receiver, route_entry=route_entry, route_exit=route_exit,
        route_internal=route_internal, width=W, hops=k, grid_internal=internal,
        assignment=assignment, entry_of=entry_of, routes_by_entry=routes_by_entry,
        jitter_scale=0.0 if jitter is None else float(jitter.scale))
