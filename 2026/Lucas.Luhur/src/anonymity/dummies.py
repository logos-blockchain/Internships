"""
Slot-synchronous cover traffic: in every consensus slot a subset of nodes sends a
message, a genuine block proposal if elected leader and a cover message otherwise.
The slot is the candidate batch (the Trace's broadcast_id). Cover senders are drawn
uniformly over non-winners, either per-node Bernoulli(p_s) or a fixed count per slot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DummyParams:
    """
    Cover-traffic parameters; cover senders are drawn uniformly over a slot's non-winners.

    p_s    -- per-node sender probability: |S_t| ~ Binomial(N, p_s), E|S| = p_s*N; overrides count
    count  -- fixed number of distinct cover senders per slot (used when p_s is None; must be < N);
              |S_t| = count + W_t on a slot with W_t winners
    window -- +/- integer slot jitter on a cover emission time (0 = its slot); batch stays the slot
    """

    count: int = 1
    window: int = 0
    p_s: float | None = None


def _bernoulli_cover(N, T, p_s, slots, nodes, rng, chunk=4096):
    """
    Draw per-slot cover senders as an independent Bernoulli(p_s) mask over nodes.

    Each slot's winners are excluded from its cover, so S_t is a set with
    |S_t| ~ Binomial(N, p_s). Chunked over slots to bound the mask memory.
    Returns (group, emitter) index arrays.
    """
    groups, emitters = [], []
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        mask = rng.random((hi - lo, N)) < p_s
        in_chunk = (slots >= lo) & (slots < hi)
        mask[slots[in_chunk] - lo, nodes[in_chunk]] = False
        s, i = np.nonzero(mask)
        groups.append(s + lo)
        emitters.append(i)
    empty = np.zeros(0, dtype=np.int64)
    return (np.concatenate(groups).astype(np.int64) if groups else empty,
            np.concatenate(emitters).astype(np.int64) if emitters else empty)


def _fixed_count_cover(N, T, count, slots, nodes, rng, chunk=4096):
    """
    Draw `count` distinct cover senders per slot, uniform over that slot's non-winners.

    One uniform key per (slot, node); the `count` smallest keys form a uniform subset
    without replacement, and winners get key = +inf so they are never selected.
    Chunked over slots. Returns (group, emitter) index arrays.
    """
    if count >= N:
        raise ValueError(f"cover count={count} needs count < N={N}: cannot draw {count} distinct "
                         f"cover senders from {N} nodes")
    max_w = int(np.bincount(slots, minlength=max(T, 1)).max()) if slots.size else 0
    if count > N - max_w:
        raise ValueError(f"cover count={count} exceeds N - max winners/slot = {N} - {max_w} = "
                         f"{N - max_w}: no room to draw distinct non-winner cover")

    groups, emitters = [], []
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        keys = rng.random((hi - lo, N))
        in_chunk = (slots >= lo) & (slots < hi)
        keys[slots[in_chunk] - lo, nodes[in_chunk]] = np.inf
        idx = np.argpartition(keys, count - 1, axis=1)[:, :count]
        groups.append(np.repeat(np.arange(lo, hi, dtype=np.int64), count))
        emitters.append(idx.ravel().astype(np.int64))
    empty = np.zeros(0, dtype=np.int64)
    return (np.concatenate(groups) if groups else empty,
            np.concatenate(emitters) if emitters else empty)


def inject_dummies(slots, nodes, N, *, params=DummyParams(), T=None, rng=None):
    """
    Emit cover senders in every slot and merge the real winners into their slots.

    Cover is drawn per slot by params.p_s (Bernoulli per node) or params.count (fixed),
    jittered by +/- params.window, then merged with the real events and sorted by slot.
    Returns (out_slots, out_nodes, is_dummy, group), where group is the slot label that
    becomes the Trace's broadcast_id. T (epoch length) is inferred from the max real slot
    if None.
    """
    rng = np.random.default_rng(rng)
    slots = np.asarray(slots, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.int64)
    M = slots.size
    count = int(params.count)
    T = int(T) if T is not None else ((int(slots.max()) + 1) if M else 1)

    if params.p_s is not None:
        cover_group, cover_nodes = _bernoulli_cover(
            N, T, float(params.p_s), slots, nodes, rng)
    elif count > 0:
        cover_group, cover_nodes = _fixed_count_cover(N, T, count, slots, nodes, rng)
    else:
        return slots, nodes, np.zeros(M, dtype=bool), slots.copy()

    n_cover = int(cover_group.size)
    if n_cover == 0:
        return slots, nodes, np.zeros(M, dtype=bool), slots.copy()

    if params.window > 0:
        jitter = rng.integers(-params.window, params.window + 1, size=n_cover)
        cover_slots = np.clip(cover_group + jitter, 0, T - 1)
    else:
        cover_slots = cover_group.copy()

    out_slots = np.concatenate([slots, cover_slots])
    out_nodes = np.concatenate([nodes, cover_nodes])
    is_dummy = np.concatenate([np.zeros(M, dtype=bool), np.ones(n_cover, dtype=bool)])
    group = np.concatenate([slots, cover_group])

    order = np.argsort(out_slots, kind="stable")
    return out_slots[order], out_nodes[order], is_dummy[order], group[order]
