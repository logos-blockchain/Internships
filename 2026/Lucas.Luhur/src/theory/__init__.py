"""
Closed forms for the reported privacy curves: the analytic twins of the measures in
src/metrics, derived from the model parameters alone (no consensus loop, cover, Trace or
attack). Modules: stake_top1, stake_confidence, stake_jaccard, stake_epochs (stake
inference), attribution (Bayesian sender attribution) and sigma_hat (the sweep axis).
"""

from .attribution import (
    candidate_set_law,
    deanon_top1,
    lognormal_link_cdf,
)
from .sigma_hat import (
    expected_sigma_d_hat,
    population_sigma_d,
    sigma_d_bias,
)
from .stake_confidence import (
    confidence_probability,
    confidence_probability_normal,
    count_at_estimate,
    expected_confidence,
    per_node_confidence,
)
from .stake_epochs import (
    cumulative_top1,
    epochs_to_level,
    epochs_to_years,
    expected_cumulative_top1,
    naive_cumulative,
    subsampled_cumulative,
)
from .stake_jaccard import (
    expected_jaccard,
    jaccard_probability,
    random_set_jaccard,
    top_set_size,
)
from .stake_top1 import (
    expected_top1,
    participation_prob,
    top1_probability,
    top1_probability_normal,
)

__all__ = [
    "candidate_set_law",
    "confidence_probability",
    "confidence_probability_normal",
    "count_at_estimate",
    "cumulative_top1",
    "deanon_top1",
    "epochs_to_level",
    "epochs_to_years",
    "expected_confidence",
    "expected_cumulative_top1",
    "expected_jaccard",
    "expected_sigma_d_hat",
    "expected_top1",
    "jaccard_probability",
    "lognormal_link_cdf",
    "naive_cumulative",
    "participation_prob",
    "per_node_confidence",
    "population_sigma_d",
    "random_set_jaccard",
    "sigma_d_bias",
    "subsampled_cumulative",
    "top1_probability",
    "top1_probability_normal",
    "top_set_size",
]
