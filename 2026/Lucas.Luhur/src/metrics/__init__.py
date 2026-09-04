"""
Metrics package: the measures that grade an attack's guess (stake-privacy measures for
the scalar stake estimate, unlinkability measures for the attribution posterior) and the
trilemma cost axes. `MEASURES` is the name -> MeasureSpec registry.
"""

from pipeline_contract import MeasureSpec, POSTERIOR, SCALAR

from .stake_privacy import (
    inference_confidence,
    time_to_confidence,
    stake_confidence,
    stake_top_jaccard,
    stake_top1_hit,
)
from .unlinkability import (
    deanon_top1,
    mean_true_posterior,
    posterior_entropy,
)
from .trilemma_cost import bandwidth_overhead, latency_overhead, mean_latency


MEASURES = {
    "stake_confidence": MeasureSpec(score=stake_confidence, consumes=SCALAR),
    "stake_top_jaccard": MeasureSpec(score=stake_top_jaccard, consumes=SCALAR),
    "stake_top1_hit": MeasureSpec(score=stake_top1_hit, consumes=SCALAR),
    "deanon_top1": MeasureSpec(score=deanon_top1, consumes=POSTERIOR),
    "mean_true_posterior": MeasureSpec(score=mean_true_posterior, consumes=POSTERIOR),
    "posterior_entropy": MeasureSpec(score=posterior_entropy, consumes=POSTERIOR),
}

__all__ = [
    "inference_confidence",
    "time_to_confidence",
    "stake_confidence",
    "stake_top_jaccard",
    "stake_top1_hit",
    "deanon_top1",
    "mean_true_posterior",
    "posterior_entropy",
    "bandwidth_overhead",
    "latency_overhead",
    "mean_latency",
    "MEASURES",
]
