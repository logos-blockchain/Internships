"""
Communication-network topology: the quenched random C-regular graph on N nodes.
The per-broadcast link latencies (the annealed disorder) live in latency.py.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

DEFAULT_C = 8


def sample_regular_graph(N, C=DEFAULT_C, rng=None):
    """
    Sample a random C-regular graph on N nodes (the quenched topology).

    Requires N*C even and C < N. Returns a networkx.Graph; downstream latency code
    only needs its edge list, via edge_endpoints.
    """
    rng = np.random.default_rng(rng)
    seed =int(rng.integers(0, 2**31 - 1))
    return nx.random_regular_graph(C, N, seed=seed)


def edge_endpoints(G):
    """
    Return (u, v): two int arrays of the graph's edge endpoints, each undirected edge once.

    Plain arrays let the latency oracle resample weights and rebuild a sparse matrix
    cheaply on every broadcast.
    """
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    return edges[:, 0], edges[:, 1]


def graph_distance_survival(ell, N, C=DEFAULT_C):
    """
    Discrete Gompertz tail Pr(L > ell) = exp(-eta * ((C-1)^ell - 1)), eta = C / ((C - 2) N).

    The unweighted shortest-path distance law of a random C-regular graph (Tishby, Biham,
    Kuhn & Katzav 2022). Pr(L > 0) = 1, since distinct nodes are at distance >= 1.
    """
    eta = C / ((C - 2.0) * N)
    return np.exp(-eta * ((C - 1.0) ** np.asarray(ell, dtype=float) - 1.0))


def mean_graph_distance_theory(N, C=DEFAULT_C, ell_max=64):
    """
    Mean unweighted shortest-path length E[L] of a random C-regular graph

    Computed exactly from the Gompertz law as E[L] = sum_{ell>=0} Pr(L > ell)
    (the tail-sum identity for a non-negative integer variable).
    """
    ells = np.arange(0, ell_max)
    return float(np.sum(graph_distance_survival(ells, N, C)))


def eccentricity_survival(ell, N, C=DEFAULT_C):
    """
    Eccentricity tail Pr(ecc > ell) = 1 - (1 - Pr(L > ell))^(N-1).

    The global-broadcast latency is the source's eccentricity, the largest hop distance to
    any of the other N-1 nodes; treating those distances as independent Gompertz draws
    makes its CDF the (N-1)-th power of the distance CDF.
    """
    S = graph_distance_survival(ell, N, C)
    return 1.0 - (1.0 - S) ** (N - 1)


def mean_eccentricity_theory(N, C=DEFAULT_C, ell_max=64):
    """
    Mean eccentricity E[ecc] in hops, via the tail sum E[ecc] = sum_{ell>=0} Pr(ecc > ell).

    With deterministic links T_e = d (rho_net = 0) the global-broadcast latency is exactly
    d * E[ecc], which is what latency.broadcast_latency_theory returns in that limit; the
    Lambert-W Chernoff form diverges as lam -> inf and cannot be continued to it.
    """
    ells = np.arange(0, ell_max)
    return float(np.sum(eccentricity_survival(ells, N, C)))
