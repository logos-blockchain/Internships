"""
Pipeline contract: the guess-type vocabulary and the AttackSpec / MeasureSpec records that
adversary.ATTACKS and metrics.MEASURES are built from. An attack produces a guess of one
type (scalar or posterior) and a measure consumes one type; validate_pairing enforces the
match so the two packages agree without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SCALAR = "scalar"
POSTERIOR = "posterior"
GUESS_TYPES = (SCALAR, POSTERIOR)


@dataclass(frozen=True)
class PosteriorGuess:
    """
    Posterior guess: one posterior over the candidate sender set per observed broadcast.

    Flat CSR layout: broadcast b owns the candidate/posterior rows [start[b] : start[b+1]].

    broadcast_row -- [B] int: Trace row index of each observed broadcast (for the scorer's truth lookup).
    start         -- [B+1] int: CSR offsets into candidate/posterior (start[0]=0, start[B]=K).
    candidate     -- [K] int: candidate sender nodes, concatenated over the B broadcasts.
    posterior     -- [K] float: Pr(L = candidate | y, r, S_t); each broadcast's slice sums to 1.
    """

    broadcast_row: np.ndarray
    start: np.ndarray
    candidate: np.ndarray
    posterior: np.ndarray

    def __len__(self):
        return int(self.broadcast_row.size)

    def slice(self, b):
        """(candidates, posterior) arrays for observed broadcast b."""
        sl = slice(int(self.start[b]), int(self.start[b + 1]))
        return self.candidate[sl], self.posterior[sl]


@dataclass(frozen=True)
class AttackSpec:
    """
    An attack and the guess-type it produces; run(trace, *, params, rng) -> Guess.

    params_cls -- the attack's params dataclass (None = takes none); the config loader builds
                  the `attack.params` block from it.
    knows      -- names of the public protocol constants run_once fills onto the params from
                  the system config (keys of the public-knowledge dict).
    """

    run: Callable
    produces: str
    params_cls: type | None = None
    knows: tuple = ()


@dataclass(frozen=True)
class MeasureSpec:
    """
    A measure and the guess-type it consumes.

    score(guess, x) -> float | dict, where x is the Trace for a family-A (posterior)
    measure and a ScoreContext for a family-B (scalar) measure.
    """

    score: Callable
    consumes: str


@dataclass(frozen=True)
class ScoreContext:
    """
    Run context handed to family-B (scalar) measures in place of the Trace.

    alpha    -- the true relative-stake vector (length N)
    gamma    -- accuracy band half-width: alpha_hat in [alpha (1 +/- gamma)]
    top_frac -- top-staker fraction x for the Jaccard measure
    f, T     -- protocol constant / observation window (for analytic comparison)
    """

    alpha: object
    gamma: float
    top_frac: float
    f: float
    T: int


def validate_pairing(attack, measures, attacks, measures_registry):
    """
    Raise ValueError if any measure cannot grade the attack's guess-type.

    attacks / measures_registry are the ATTACKS / MEASURES dicts (passed in to keep
    this module import-free).
    """
    produced = attacks[attack].produces
    for m in measures:
        consumed = measures_registry[m].consumes
        if consumed != produced:
            raise ValueError(
                f"guess-type mismatch: attack {attack!r} produces {produced!r} but "
                f"measure {m!r} consumes {consumed!r} (guess-type family wall)"
            )
