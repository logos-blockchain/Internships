"""
The global passive adversary (GPA): what it observes on the consensus stream,
rendered into the shape each attack consumes. `observe_sender_sets` gives per-node
sender-set counts (stake inference); `observe_broadcasts` gives per-broadcast
(S_t, r, y) observations (Bayesian sender attribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def observe_sender_sets(trace, N):
    """
    Count each node's sender-set (entry-row) appearances over the trace.

    Reads only obs_node on the entry rows, never true_source / is_dummy.
    Returns counts of length N with counts[i] = entry appearances of node i.
    """
    nodes = trace.obs_node[trace.is_entry]
    return np.bincount(nodes, minlength=N)[:N] if N else np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class Broadcasts:
    """
    The GPA's per-broadcast observation (slot t, sender set S_t, receiver r, time y).

    Candidate sets differ in size, so S_t is stored CSR-style: broadcast b owns
    candidate[start[b]:start[b+1]].

    broadcast_row -- [B] Trace row index of each broadcast row (for the scorer to fetch truth).
    receiver      -- [B] r, the node that initiated the broadcast.
    y             -- [B] the broadcast delay y = obs_time(broadcast) - t (= mu(true,r) + Z).
    start         -- [B+1] CSR offsets into candidate.
    candidate     -- [K] the candidate senders S_t (entry nodes of the broadcast's slot).
    """

    broadcast_row: np.ndarray
    receiver: np.ndarray
    y: np.ndarray
    start: np.ndarray
    candidate: np.ndarray

    def __len__(self):
        return int(self.broadcast_row.size)


def observe_broadcasts(trace):
    """
    Render the Trace into per-broadcast (S_t, r, y) observations for sender attribution.

    Reads only the observable columns (obs_node / obs_time / kind / broadcast_id), never
    true_source / is_dummy. For each broadcast row, r = obs_node, y = obs_time - t and
    S_t = the entry senders of the broadcast's slot. Returns a Broadcasts view.
    """
    ent = trace.is_entry
    ent_batch = trace.broadcast_id[ent]
    ent_node = trace.obs_node[ent]
    ent_time = trace.obs_time[ent]

    bc = trace.is_broadcast
    broadcast_row = np.nonzero(bc)[0].astype(np.int64)
    bc_batch = trace.broadcast_id[bc]
    receiver = trace.obs_node[bc].astype(np.int64)
    bc_time = trace.obs_time[bc]
    B = broadcast_row.size

    if B == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return Broadcasts(broadcast_row=empty_i, receiver=empty_i, y=np.zeros(0),
                          start=np.zeros(1, dtype=np.int64), candidate=empty_i)

    order = np.argsort(ent_batch, kind="stable")
    sb, sn, st = ent_batch[order], ent_node[order], ent_time[order]
    batch_vals, batch_start = np.unique(sb, return_index=True)
    batch_end = np.append(batch_start[1:], sb.size)
    batch_t = st[batch_start]

    bidx = np.searchsorted(batch_vals, bc_batch)
    y = bc_time - batch_t[bidx]

    counts = (batch_end - batch_start)[bidx]
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    seg = np.repeat(np.arange(B), counts)
    within = np.arange(start[-1], dtype=np.int64) - start[seg]
    candidate = sn[batch_start[bidx][seg] + within].astype(np.int64)

    return Broadcasts(broadcast_row=broadcast_row, receiver=receiver, y=y,
                      start=start, candidate=candidate)
