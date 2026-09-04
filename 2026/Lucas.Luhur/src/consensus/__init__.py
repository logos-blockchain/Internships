"""
Consensus package: the stake-proportional PoS leader election treated as a
random message generator (stake sampling in `stake`, the lottery in `election`).
"""

from .stake import (
    DEFAULT_SHAPE,
    expected_max_relative_stake,
    gini,
    gini_from_shape,
    sample_relative_stakes,
    shape_from_gini,
    simulate_max_relative_stake,
)
from .election import (
    DEFAULT_F,
    DEFAULT_T,
    expected_winners_per_slot,
    lottery,
    prob_at_least_one,
    prob_at_least_two,
    simulate_events,
    var_winners_per_slot,
    winner_counts_from_events,
    wins_per_node_from_events,
)

__all__ = [
    "DEFAULT_SHAPE",
    "DEFAULT_F",
    "DEFAULT_T",
    "sample_relative_stakes",
    "gini",
    "gini_from_shape",
    "shape_from_gini",
    "expected_max_relative_stake",
    "simulate_max_relative_stake",
    "lottery",
    "expected_winners_per_slot",
    "var_winners_per_slot",
    "prob_at_least_one",
    "prob_at_least_two",
    "simulate_events",
    "winner_counts_from_events",
    "wins_per_node_from_events",
]
