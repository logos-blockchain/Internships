"""
Leader election as the consensus message generator. Each node i independently wins
a slot with probability phi(alpha_i) = 1 - (1 - f)^{alpha_i} (Ouroboros Crypsinous),
and each win over T slots is a (slot, node) broadcast event.
"""

from __future__ import annotations

import numpy as np

DEFAULT_F = 1.0 / 30.0
DEFAULT_T = 388_800          # = 6 * floor(k / f) with k = 2160


def lottery(alpha, f=DEFAULT_F):
    """
    Per-slot win probability phi(alpha) = 1 - (1 - f)^alpha.

    Monotonic increasing in stake with phi(0) = 0 and phi(1) = f.
    """
    return 1.0 - (1.0 - f) ** np.asarray(alpha, dtype=float)


def expected_winners_per_slot(alpha, f=DEFAULT_F):
    """E[n(t)] = sum_i phi(alpha_i)."""
    return float(np.sum(lottery(alpha, f)))


def var_winners_per_slot(alpha, f=DEFAULT_F):
    """Var[n(t)] = sum_i phi(alpha_i) (1 - phi(alpha_i))."""
    phi = lottery(alpha, f)
    return float(np.sum(phi * (1.0 - phi)))


def prob_at_least_one(f=DEFAULT_F):
    """
    P(n(t) >= 1) = f, independent of the stake distribution.

    prod_i (1 - phi(alpha_i)) = (1 - f)^{sum alpha_i} = 1 - f when stakes sum to 1.
    """
    return float(f)


def prob_at_least_two(alpha, f=DEFAULT_F):
    """
    P(n(t) >= 2) = f - (1 - f) * sum_i phi(alpha_i) / (1 - phi(alpha_i)), the multi-leader rate.

    Safe to evaluate since 1 - phi >= 1 - f > 0.
    """
    phi = lottery(alpha, f)
    return float(f - (1.0 - f) * np.sum(phi / (1.0 - phi)))


def simulate_events(alpha, f=DEFAULT_F, T=DEFAULT_T, rng=None, chunk=20_000):
    """
    Run the leader election over T slots and return the winner events.

    Returns (slots, nodes): equal-length int arrays where the j-th broadcast was emitted
    by node nodes[j] in slot slots[j]; a slot may appear zero, one or several times.
    Slots are processed in chunks to bound peak memory at O(N * chunk).
    """
    rng = np.random.default_rng(rng)
    alpha = np.asarray(alpha, dtype=float)
    phi = lottery(alpha, f)[:, None]
    N = alpha.size

    slot_parts, node_parts = [], []
    done = 0
    while done < T:
        c = min(chunk, T - done)
        hits = rng.random((N, c)) < phi
        nodes, slots = np.nonzero(hits)
        slot_parts.append(slots + done)
        node_parts.append(nodes)
        done += c

    slots = np.concatenate(slot_parts) if slot_parts else np.empty(0, dtype=int)
    nodes = np.concatenate(node_parts) if node_parts else np.empty(0, dtype=int)
    return slots.astype(np.int64), nodes.astype(np.int64)


def winner_counts_from_events(slots, T):
    """Reduce winner events to the per-slot count vector n(t) (length T)."""
    return np.bincount(slots, minlength=T).astype(np.int64)


def wins_per_node_from_events(nodes, N):
    """
    Reduce winner events to per-node total wins (length N).

    A node's win count over a long horizon is the stake-inference adversary's signal.
    """
    return np.bincount(nodes, minlength=N).astype(np.int64)


def plot_lottery(out_path=None, f_values=(DEFAULT_F, 0.1, 0.3, 0.6, 0.9)):
    """
    Plot phi(alpha) = 1 - (1 - f)^alpha over alpha in [0, 1] for a family of f.

    Each curve ends at height f; at f = 1/30 it is nearly linear, phi ~ f * alpha.
    """
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle                       

    if out_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        out_path = repo_root / "results" / "figures" / "stage_2_figures" / "lottery_function.png"

    alpha = np.linspace(0.0, 1.0, 400)
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    cmap = plt.cm.viridis

    fig, ax = plt.subplots()
    for i, f in enumerate(f_values):
        ax.plot(alpha, lottery(alpha, f), color=cmap(f),
                linestyle=styles[i % len(styles)], linewidth=2,
                label=fr"$f={f:.4g}$")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)                  
    ax.set_xlabel(r"Relative stake  $\alpha$")
    ax.set_ylabel(r"Win probability")
    ax.legend(loc="upper left", fontsize=18)

    return plotstyle.save(fig, out_path)


if __name__ == "__main__":
    saved = plot_lottery()
    print(f"Figure written to {saved}")
