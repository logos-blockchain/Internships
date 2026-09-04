"""
The WonderNetwork ping dataset and the link-latency calibration derived from it.

data/wondernetwork_pings.csv holds one row per ordered (source -> destination) measurement:
125 rows over 12 cities (4 per continent), mean-of-30 ICMP round-trip times in milliseconds,
covering all 66 unordered city pairs (59 in both directions, 7 in one). Everything here is
derived from that file; `ping_link_moments` is the log-normal law's calibration source.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

CONTINENTS = ("europe", "north america", "asia")

REGION_WEIGHTS = {"europe": 0.4, "north america": 0.4, "asia": 0.3}

PINGS_CSV = Path(__file__).resolve().parents[2] / "data" / "wondernetwork_pings.csv"

DEFAULT_RTT_TO_ONEWAY = 0.5


def normalise_weights(raw):
    """
    Validate and renormalise {continent: weight} into {continent: probability}.

    The raw REGION_WEIGHTS sum to 1.1, so the renormalisation is load-bearing.
    """
    raw = dict(raw)
    unknown = set(raw) - set(CONTINENTS)
    if unknown:
        raise ValueError(f"unknown continent(s) {sorted(unknown)}; known: {list(CONTINENTS)}")
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError(f"region weights must sum to > 0, got {raw!r}")
    return {c: float(w) / total for c, w in raw.items()}


@lru_cache(maxsize=1)
def load_pings():
    """
    Read PINGS_CSV into a tuple of (source, destination, rtt_ms), one per ordered pair.

    Validates that each city maps to exactly one known continent.
    """
    if not PINGS_CSV.exists():
        raise FileNotFoundError(
            f"the ping data is missing: {PINGS_CSV}. It is the source of truth for the link "
            "latency calibration -- restore it from git rather than hardcoding the moments.")
    rows, seen = [], {}
    with PINGS_CSV.open(newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            for city, cont in ((rec["source"], rec["source_continent"]),
                               (rec["destination"], rec["destination_continent"])):
                if cont not in CONTINENTS:
                    raise ValueError(f"{PINGS_CSV.name}: unknown continent {cont!r} for {city!r}; "
                                     f"known: {list(CONTINENTS)}")
                if seen.setdefault(city, cont) != cont:
                    raise ValueError(f"{PINGS_CSV.name}: {city!r} is assigned to both "
                                     f"{seen[city]!r} and {cont!r}")
            rows.append((rec["source"], rec["destination"], float(rec["rtt_ms"])))
    return tuple(rows)


@lru_cache(maxsize=1)
def city_continent():
    """{city: continent} as declared by the CSV (4 cities per continent, spread not hub-clustered)."""
    out = {}
    with PINGS_CSV.open(newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            out[rec["source"]] = rec["source_continent"]
            out[rec["destination"]] = rec["destination_continent"]
    return out


def symmetrised_city_pairs():
    """
    Return {frozenset({city_a, city_b}): mean RTT ms}, averaging the two directions.

    66 pairs: the 59 measured both ways are averaged, the 7 one-way pairs pass through.
    """
    pairs = defaultdict(list)
    for src, dst, ms in load_pings():
        pairs[frozenset((src, dst))].append(ms)
    return {k: float(np.mean(v)) for k, v in pairs.items()}


def directional_asymmetry():
    """
    Return (mean_fraction, max_fraction, n_pairs) of |A->B - B->A| / mean over two-way pairs.

    A data-quality check (both directions measure the same round trip), not a test of one-way
    symmetry, which an RTT cannot decompose.
    """
    both = defaultdict(list)
    for src, dst, ms in load_pings():
        both[frozenset((src, dst))].append(ms)
    fracs = [abs(v[0] - v[1]) / np.mean(v) for v in both.values() if len(v) == 2]
    return float(np.mean(fracs)), float(np.max(fracs)), len(fracs)


def ping_link_moments(params=None):
    """
    Return the ping population's one-way (mean, variance, minimum) in seconds.

    The calibration source for the shipped log-normal configs (floor = minimum, mean, sd =
    sqrt(variance); measured 62.897 / 33.340 / 7.16 ms). Pairs are weighted by the continent
    mixture and uniformly within a continent, w(a, b) = [pi_{c(a)} / n_{c(a)}] * [pi_{c(b)} /
    n_{c(b)}], the link population the simulator's node placement induces. Parameters are
    moment-matched rather than MLE-fitted (see lognormal_latency.py). `params` is duck-typed
    (any object with .weight_map() and .rtt_to_oneway); None -> REGION_WEIGHTS, DEFAULT_RTT_TO_ONEWAY.
    """
    vals, wts, _ = ping_link_population(params)
    mean = float(wts @ vals)
    return mean, max(float(wts @ vals ** 2 - mean ** 2), 0.0), float(vals.min())


def ping_link_population(params=None):
    """
    Return the weighted link population as (values_s, weights, intra_continental).

    132 ordered city pairs (66 symmetrised x 2 orderings), one-way in seconds, with the
    continent-mixture weights (summing to 1) that `ping_link_moments` reduces.
    `intra_continental` flags pairs whose endpoints share a continent, which is what makes
    the population multimodal.
    """
    if params is None:
        pi, f = normalise_weights(REGION_WEIGHTS), DEFAULT_RTT_TO_ONEWAY
    else:
        pi, f = params.weight_map(), params.rtt_to_oneway

    where = city_continent()
    n_in = defaultdict(int)
    for c in where.values():
        n_in[c] += 1

    f = f / 1000.0                                     # ms (RTT) -> s (one-way)
    vals, wts, intra = [], [], []
    for pair, ms in symmetrised_city_pairs().items():
        a, b = tuple(pair)
        for x, y in ((where[a], where[b]), (where[b], where[a])):
            vals.append(ms * f)
            wts.append(pi.get(x, 0.0) / n_in[x] * pi.get(y, 0.0) / n_in[y])
            intra.append(x == y)
    vals = np.asarray(vals, dtype=np.float64)
    wts = np.asarray(wts, dtype=np.float64)
    total = wts.sum()
    if total <= 0.0:
        raise ValueError("the continent weights give every city pair zero weight")
    return vals, wts / total, np.asarray(intra, dtype=bool)
