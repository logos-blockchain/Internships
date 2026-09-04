"""
The Trace: the shared observation schema every anonymisation layer emits and the
attacks consume. One row per observable event (an ENTRY and an exit-side row per
message); `true_source`, `is_dummy` and `true_route` are ground truth for scoring
only, and real-vs-cover is exposed to the attack through `kind` (BROADCAST vs EXIT).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENTRY = np.int8(0)
EXIT = np.int8(1)
BROADCAST = np.int8(2)


@dataclass(frozen=True)
class Trace:
    """
    Flat record array of observable events; all columns are parallel arrays of equal length.

    broadcast_id -- candidate-batch label linking rows to a broadcast
    true_source  -- ground-truth sender (scoring only)
    obs_node     -- node observed (entry or exit)
    obs_time     -- observation time (slot + delay)
    kind         -- ENTRY | EXIT | BROADCAST
    is_dummy     -- real winner vs cover (scoring only)
    true_route   -- ground-truth grid route, -1 = no route structure (oracle/scoring only)
    """

    broadcast_id: np.ndarray
    true_source: np.ndarray
    obs_node: np.ndarray
    obs_time: np.ndarray
    kind: np.ndarray
    is_dummy: np.ndarray
    true_route: np.ndarray = None

    def __post_init__(self):
        """Fill a missing true_route with -1 and check all columns share one length."""
        n = self.broadcast_id.size
        if self.true_route is None:
            object.__setattr__(self, "true_route", np.full(n, -1, dtype=np.int64))
        for name in ("true_source", "obs_node", "obs_time", "kind", "is_dummy", "true_route"):
            if getattr(self, name).size != n:
                raise ValueError(
                    f"Trace column {name!r} has length {getattr(self, name).size}, "
                    f"expected {n} to match broadcast_id"
                )

    def __len__(self):
        """Number of rows."""
        return int(self.broadcast_id.size)

    @property
    def is_entry(self):
        """Boolean mask selecting the ENTRY rows."""
        return self.kind == ENTRY

    @property
    def is_exit(self):
        """Boolean mask selecting all exit-side rows (cover EXIT + real BROADCAST)."""
        return (self.kind == EXIT) | (self.kind == BROADCAST)

    @property
    def is_broadcast(self):
        """Boolean mask selecting the real-broadcast rows (the attribution targets)."""
        return self.kind == BROADCAST


def make_trace(broadcast_id, true_source, obs_node, obs_time, kind, is_dummy, true_route=None):
    """
    Build a Trace from raw columns, coercing each to its canonical dtype.

    true_route is optional (None -> all -1, no route structure); only the mixnet
    layer records it.
    """
    return Trace(
        broadcast_id=np.asarray(broadcast_id, dtype=np.int64),
        true_source=np.asarray(true_source, dtype=np.int64),
        obs_node=np.asarray(obs_node, dtype=np.int64),
        obs_time=np.asarray(obs_time, dtype=np.float64),
        kind=np.asarray(kind, dtype=np.int8),
        is_dummy=np.asarray(is_dummy, dtype=bool),
        true_route=None if true_route is None else np.asarray(true_route, dtype=np.int64),
    )


def passthrough(slots, nodes, is_dummy=None, group=None, *, params=None, latency_oracle=None, rng=None) -> Trace:
    """
    The "none" layer: each emission enters and exits at its own source with no delay.

    Every exit is observed at the true source, so the anonymity set is 1 per emission.
    is_dummy (None = all real) rides into the Trace for the scorer only; group,
    params, latency_oracle and rng are accepted for the uniform layer signature but
    unused (each emission is its own broadcast). Returns a Trace with 2*M rows.
    """
    slots = np.asarray(slots, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)
    m = slots.size
    dummy = np.zeros(m, dtype=bool) if is_dummy is None else np.asarray(is_dummy, dtype=bool)

    bid = np.arange(m, dtype=np.int64)
    exit_kind = np.where(dummy, EXIT, BROADCAST)
    return make_trace(
        broadcast_id=np.concatenate([bid, bid]),
        true_source=np.concatenate([nodes, nodes]),
        obs_node=np.concatenate([nodes, nodes]),
        obs_time=np.concatenate([slots, slots]),
        kind=np.concatenate([np.full(m, ENTRY), exit_kind]),
        is_dummy=np.concatenate([dummy, dummy]),
    )
